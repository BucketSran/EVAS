"""Shadow runtime for 094h event due-program evaluation.

This module intentionally does not plug into the production simulator.  It
connects the 094h trigger expression batches to existing Rust detector/timer
primitives so we can audit fired event indices before replacing engine-owned
event scheduling.
"""

from __future__ import annotations

from array import array
from typing import MutableSequence, Optional, Tuple

from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import RustBackend
from evas.simulator.schedule_ir import (
    EVENT_DUE_ABOVE,
    EVENT_DUE_CROSS,
    EVENT_DUE_INITIAL_STEP,
    EVENT_DUE_TIMER,
    EventDueProgram,
    encode_event_due_program,
)
from evas.simulator.stmt_ir import (
    EventBodyProgram,
    EventStatementIR,
    encode_event_body_program,
    lower_stmt,
)


class RustEventDueRuntime:
    """Shadow-only mixed event due runtime backed by Rust primitives."""

    def __init__(self, program: EventDueProgram, backend: RustBackend):
        self.program = program
        self.backend = backend

        self._expr_trigger_indices: list[int] = []
        expr_segments = []
        self._cross_slots: dict[int, int] = {}
        self._above_slots: dict[int, int] = {}
        self._periodic_timer_slots: dict[int, int] = {}
        self._absolute_timer_slots: dict[int, int] = {}

        timer_segments = []
        self._periodic_timer_segment_indices: list[tuple[int, int]] = []
        self._absolute_timer_segment_indices: list[int] = []

        for trigger_index, trigger in enumerate(program.triggers):
            if trigger.kind in {EVENT_DUE_CROSS, EVENT_DUE_ABOVE}:
                self._expr_trigger_indices.append(trigger_index)
                expr_segments.append(trigger.expr_ops)
                if trigger.kind == EVENT_DUE_CROSS:
                    self._cross_slots[trigger_index] = len(self._cross_slots)
                else:
                    self._above_slots[trigger_index] = len(self._above_slots)
            elif trigger.kind == EVENT_DUE_TIMER:
                start_segment_index = len(timer_segments)
                timer_segments.append(trigger.timer_start_ops)
                if trigger.timer_period_ops:
                    period_segment_index = len(timer_segments)
                    timer_segments.append(trigger.timer_period_ops)
                    self._periodic_timer_slots[trigger_index] = len(
                        self._periodic_timer_slots
                    )
                    self._periodic_timer_segment_indices.append(
                        (start_segment_index, period_segment_index)
                    )
                else:
                    self._absolute_timer_slots[trigger_index] = len(
                        self._absolute_timer_slots
                    )
                    self._absolute_timer_segment_indices.append(start_segment_index)
            elif trigger.kind == EVENT_DUE_INITIAL_STEP:
                continue
            else:
                raise ValueError(f"unsupported event due trigger kind: {trigger.kind}")

        self._expr_batch = (
            backend.make_body_expr_batch(expr_segments) if expr_segments else None
        )
        self._timer_batch = (
            backend.make_body_expr_batch(timer_segments) if timer_segments else None
        )
        self._timer_values_initialized = False

        cross_count = len(self._cross_slots)
        self.cross_prev_values = array("d", [0.0] * cross_count)
        self.cross_prev_times = array("d", [0.0] * cross_count)
        self.cross_pprev_values = array("d", [0.0] * cross_count)
        self.cross_pprev_times = array("d", [0.0] * cross_count)
        self.cross_initialized_flags = [0] * cross_count
        self.cross_last_cross_times = array("d", [-1.0] * cross_count)
        self.cross_triggered_flags = [0] * cross_count
        self.cross_times = array("d", [0.0] * cross_count)
        self.cross_trigger_directions = [0] * cross_count
        self.cross_went_beyond_flags = [0] * cross_count

        above_count = len(self._above_slots)
        self.above_prev_values = array("d", [0.0] * above_count)
        self.above_prev_times = array("d", [0.0] * above_count)
        self.above_pprev_values = array("d", [0.0] * above_count)
        self.above_pprev_times = array("d", [0.0] * above_count)
        self.above_initialized_flags = [0] * above_count
        self.above_triggered_flags = [0] * above_count
        self.above_cross_times = array("d", [0.0] * above_count)

        periodic_count = len(self._periodic_timer_slots)
        self.periodic_next_fire_times = array("d", [0.0] * periodic_count)
        self.periodic_has_state_flags = [0] * periodic_count
        self.periodic_periods = array("d", [0.0] * periodic_count)
        self.periodic_starts = array("d", [0.0] * periodic_count)
        self.periodic_has_start_flags = [1] * periodic_count
        self.periodic_due_flags = [0] * periodic_count
        self.periodic_skipped_flags = [0] * periodic_count

        absolute_count = len(self._absolute_timer_slots)
        self.absolute_next_fire_times = array("d", [0.0] * absolute_count)
        self.absolute_has_state_flags = [0] * absolute_count
        self.absolute_last_fired_times = array("d", [0.0] * absolute_count)
        self.absolute_has_last_fired_flags = [0] * absolute_count
        self.absolute_targets = array("d", [0.0] * absolute_count)
        self.absolute_due_flags = [0] * absolute_count
        self.absolute_expired_flags = [0] * absolute_count

    def step(
        self,
        *,
        time: float,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
        initial_step: bool = False,
    ) -> Tuple[int, ...]:
        state_values = state_values if state_values is not None else array("d")
        param_values = param_values if param_values is not None else array("d")
        fired = set()

        if initial_step:
            for trigger_index, trigger in enumerate(self.program.triggers):
                if trigger.kind == EVENT_DUE_INITIAL_STEP:
                    fired.add(trigger_index)

        expr_values = self._evaluate_expr_values(node_values, state_values, param_values)
        self._step_cross_triggers(time, expr_values, node_values, state_values, param_values, fired)
        self._step_above_triggers(time, expr_values, fired)
        self._step_timer_triggers(time, node_values, state_values, param_values, fired)

        return tuple(
            trigger_index
            for trigger_index in range(len(self.program.triggers))
            if trigger_index in fired
        )

    def _evaluate_expr_values(
        self,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
    ) -> dict[int, float]:
        if self._expr_batch is None:
            return {}
        values = self.backend.evaluate_body_expr_batch(
            self._expr_batch, node_values, state_values, param_values
        )
        return dict(zip(self._expr_trigger_indices, values))

    def _step_cross_triggers(
        self,
        time: float,
        expr_values: dict[int, float],
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
        fired: set[int],
    ) -> None:
        for trigger_index, slot in self._cross_slots.items():
            trigger = self.program.triggers[trigger_index]
            time_tol = self._eval_optional_scalar(
                trigger.time_tol_ops, node_values, state_values, param_values, 0.0
            )
            expr_tol = self._eval_optional_scalar(
                trigger.expr_tol_ops, node_values, state_values, param_values, 1.0e-12
            )
            prev_values = array("d", [self.cross_prev_values[slot]])
            prev_times = array("d", [self.cross_prev_times[slot]])
            pprev_values = array("d", [self.cross_pprev_values[slot]])
            pprev_times = array("d", [self.cross_pprev_times[slot]])
            initialized = [self.cross_initialized_flags[slot]]
            last_cross_times = array("d", [self.cross_last_cross_times[slot]])
            triggered = [0]
            cross_times = array("d", [self.cross_times[slot]])
            trigger_directions = [0]
            went_beyond = [0]
            self.backend.cross_detector_step(
                prev_values,
                prev_times,
                pprev_values,
                pprev_times,
                initialized,
                [trigger.direction],
                last_cross_times,
                array("d", [expr_values[trigger_index]]),
                triggered,
                cross_times,
                trigger_directions,
                went_beyond,
                time,
                time_tol=time_tol,
                expr_tol=expr_tol,
            )
            self.cross_prev_values[slot] = prev_values[0]
            self.cross_prev_times[slot] = prev_times[0]
            self.cross_pprev_values[slot] = pprev_values[0]
            self.cross_pprev_times[slot] = pprev_times[0]
            self.cross_initialized_flags[slot] = initialized[0]
            self.cross_last_cross_times[slot] = last_cross_times[0]
            self.cross_triggered_flags[slot] = triggered[0]
            self.cross_times[slot] = cross_times[0]
            self.cross_trigger_directions[slot] = trigger_directions[0]
            self.cross_went_beyond_flags[slot] = went_beyond[0]
            if triggered[0]:
                fired.add(trigger_index)

    def _step_above_triggers(
        self,
        time: float,
        expr_values: dict[int, float],
        fired: set[int],
    ) -> None:
        if not self._above_slots:
            return
        current_values = array(
            "d",
            [
                expr_values[trigger_index]
                for trigger_index, _slot in sorted(
                    self._above_slots.items(), key=lambda item: item[1]
                )
            ],
        )
        directions = [
            self.program.triggers[trigger_index].direction
            for trigger_index, _slot in sorted(
                self._above_slots.items(), key=lambda item: item[1]
            )
        ]
        self.backend.above_detector_step(
            self.above_prev_values,
            self.above_prev_times,
            self.above_pprev_values,
            self.above_pprev_times,
            self.above_initialized_flags,
            directions,
            current_values,
            self.above_triggered_flags,
            self.above_cross_times,
            time,
        )
        for trigger_index, slot in self._above_slots.items():
            if self.above_triggered_flags[slot]:
                fired.add(trigger_index)

    def _step_timer_triggers(
        self,
        time: float,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
        fired: set[int],
    ) -> None:
        self._ensure_timer_values(node_values, state_values, param_values)
        if self._periodic_timer_slots:
            self.backend.timer_periodic_step(
                self.periodic_next_fire_times,
                self.periodic_has_state_flags,
                self.periodic_periods,
                self.periodic_starts,
                self.periodic_has_start_flags,
                self.periodic_due_flags,
                self.periodic_skipped_flags,
                time,
                reschedule_on_due=True,
            )
            for trigger_index, slot in self._periodic_timer_slots.items():
                if self.periodic_due_flags[slot]:
                    fired.add(trigger_index)
        if self._absolute_timer_slots:
            self.backend.timer_absolute_step(
                self.absolute_next_fire_times,
                self.absolute_has_state_flags,
                self.absolute_last_fired_times,
                self.absolute_has_last_fired_flags,
                self.absolute_targets,
                self.absolute_due_flags,
                self.absolute_expired_flags,
                time,
            )
            for trigger_index, slot in self._absolute_timer_slots.items():
                if self.absolute_due_flags[slot]:
                    fired.add(trigger_index)

    def _ensure_timer_values(
        self,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
    ) -> None:
        if self._timer_values_initialized or self._timer_batch is None:
            return
        values = self.backend.evaluate_body_expr_batch(
            self._timer_batch, node_values, state_values, param_values
        )
        for slot, (start_idx, period_idx) in enumerate(
            self._periodic_timer_segment_indices
        ):
            self.periodic_starts[slot] = values[start_idx]
            self.periodic_periods[slot] = values[period_idx]
        for slot, target_idx in enumerate(self._absolute_timer_segment_indices):
            self.absolute_targets[slot] = values[target_idx]
        self._timer_values_initialized = True

    def _eval_optional_scalar(
        self,
        expr_ops,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
        default: float,
    ) -> float:
        if not expr_ops:
            return default
        return self.backend.evaluate_body_expr(
            expr_ops, node_values, state_values, param_values
        )


class RustEventStatementRuntime:
    """Shadow runtime for one event statement's due check plus Rust body batch."""

    def __init__(
        self,
        *,
        due_program: EventDueProgram,
        body_program: EventBodyProgram,
        backend: RustBackend,
    ):
        self.due_runtime = RustEventDueRuntime(due_program, backend)
        self.backend = backend
        self.body_batch = backend.make_body_ir_batch(
            stmt_ops=body_program.body_program.stmt_ops,
            expr_ops=body_program.body_program.expr_ops,
        )

    def step(
        self,
        *,
        time: float,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
        initial_step: bool = False,
    ) -> Tuple[int, ...]:
        state_values = state_values if state_values is not None else array("d")
        param_values = param_values if param_values is not None else array("d")
        fired_indices = self.due_runtime.step(
            time=time,
            node_values=node_values,
            state_values=state_values,
            param_values=param_values,
            initial_step=initial_step,
        )
        if fired_indices:
            self.backend.evaluate_body_ir(
                self.body_batch,
                node_values=node_values,
                state_values=state_values,
                param_values=param_values,
            )
        return fired_indices


class RustAnalogBlockEventRuntime:
    """Shadow runtime for source-ordered event statements in one analog block."""

    def __init__(self, event_runtimes: Tuple[RustEventStatementRuntime, ...]):
        self.event_runtimes = tuple(event_runtimes)

    def step(
        self,
        *,
        time: float,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
        initial_step: bool = False,
    ) -> Tuple[int, ...]:
        state_values = state_values if state_values is not None else array("d")
        param_values = param_values if param_values is not None else array("d")
        fired_statements = []
        for stmt_index, runtime in enumerate(self.event_runtimes):
            fired_triggers = runtime.step(
                time=time,
                node_values=node_values,
                state_values=state_values,
                param_values=param_values,
                initial_step=initial_step,
            )
            if fired_triggers:
                fired_statements.append(stmt_index)
        return tuple(fired_statements)


def try_build_rust_event_only_analog_block_runtime(
    module,
    backend: RustBackend,
    node_slots: dict[str, int],
) -> Optional[RustAnalogBlockEventRuntime]:
    """Build a shadow Rust runtime for event statements in one analog block.

    This is deliberately event-only: continuous contributions, transition
    outputs, record/CSV, and adaptive stepping remain Python-owned.  Returning
    ``None`` means at least one event statement cannot be represented by the
    current 094 due/body Rust batches, so callers must keep the existing Python
    evaluator as the semantic owner.
    """

    analog_block = getattr(module, "analog_block", None)
    body = getattr(analog_block, "body", None)
    statements = getattr(body, "statements", None)
    if statements is None:
        return None

    bindings = build_state_binding_ir(module)
    event_runtimes: list[RustEventStatementRuntime] = []
    for stmt in statements:
        stmt_ir = lower_stmt(stmt)
        if not isinstance(stmt_ir, EventStatementIR):
            continue
        due_program = encode_event_due_program(stmt_ir.event, bindings, node_slots)
        body_program = encode_event_body_program(stmt_ir, bindings, node_slots)
        if due_program is None or body_program is None:
            return None
        event_runtimes.append(
            RustEventStatementRuntime(
                due_program=due_program,
                body_program=body_program,
                backend=backend,
            )
        )

    if not event_runtimes:
        return None
    return RustAnalogBlockEventRuntime(tuple(event_runtimes))
