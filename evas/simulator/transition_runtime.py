"""Shadow runtime for 094n transition contribution batches.

This module is intentionally not wired into production simulation.  It bridges
094 ExprIR/StmtIR lowering to the existing Rust transition-state primitive so
we can validate continuous ``V(node[, ref]) <+ transition(...)`` contributions
before attempting a guarded engine.py dispatch.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import MutableSequence, Optional, Tuple

from evas.simulator.expr_ir import (
    BinaryExprIR,
    BindingTableIR,
    BodyExprOp,
    BranchAccessIR,
    FunctionCallIR,
    LiteralIR,
    UnaryExprIR,
    encode_body_expr_ops,
    static_node_ref_name,
)
from evas.simulator.rust_backend import RustBackend
from evas.simulator.stmt_ir import (
    AssignmentIR,
    BlockIR,
    ContributionIR,
    EventStatementIR,
    ForStatementIR,
    StmtIR,
    unroll_static_for_statement,
)


@dataclass(frozen=True)
class TransitionContributionProgram:
    """Continuous transition contributions encoded as target/delay/rise/fall/bias/scale ops."""

    output_node_slots: Tuple[int, ...]
    reference_node_slots: Tuple[Optional[int], ...]
    expr_segments: Tuple[Tuple[BodyExprOp, ...], ...]

    @property
    def contribution_count(self) -> int:
        return len(self.output_node_slots)


def encode_transition_contribution_program(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
) -> Optional[TransitionContributionProgram]:
    """Encode direct ``transition()`` voltage contributions from an analog block.

    Event statements are ignored; this collector owns only continuous
    contributions.  Any unsupported continuous statement or non-transition
    contribution returns ``None`` so callers can fall back to the Python
    evaluator.
    """

    output_slots: list[int] = []
    reference_slots: list[Optional[int]] = []
    expr_segments: list[Tuple[BodyExprOp, ...]] = []
    if not _append_transition_contribution_specs(
        stmt_ir,
        bindings,
        node_slots,
        output_slots,
        reference_slots,
        expr_segments,
    ):
        return None
    if not output_slots:
        return None
    return TransitionContributionProgram(
        output_node_slots=tuple(output_slots),
        reference_node_slots=tuple(reference_slots),
        expr_segments=tuple(expr_segments),
    )


class RustTransitionContributionRuntime:
    """Shadow runtime for a batch of continuous transition contributions."""

    def __init__(
        self,
        program: TransitionContributionProgram,
        backend: RustBackend,
        *,
        default_transition: float,
    ):
        self.program = program
        self.backend = backend
        self.default_transition = float(default_transition)
        count = program.contribution_count
        self.expr_batch = backend.make_body_expr_batch(program.expr_segments)
        self.current_values = array("d", [0.0] * count)
        self.target_values = array("d", [0.0] * count)
        self.start_times = array("d", [0.0] * count)
        self.start_values = array("d", [0.0] * count)
        self.delays = array("d", [0.0] * count)
        self.rise_times = array("d", [0.0] * count)
        self.fall_times = array("d", [0.0] * count)
        self.active_flags = array("B", [0] * count)
        self.initialized_flags = array("B", [0] * count)
        self.output_values = array("d", [0.0] * count)

    def step(
        self,
        *,
        time: float,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
        initial_condition_mode: bool = False,
    ) -> Tuple[float, ...]:
        state_values = state_values if state_values is not None else array("d")
        param_values = param_values if param_values is not None else array("d")
        values = self.backend.evaluate_body_expr_batch(
            self.expr_batch,
            node_values,
            state_values,
            param_values,
        )
        input_targets = array("d")
        input_delays = array("d")
        input_rises = array("d")
        input_falls = array("d")
        for idx in range(self.program.contribution_count):
            base = idx * 6
            input_targets.append(float(values[base]))
            input_delays.append(float(values[base + 1]))
            input_rises.append(float(values[base + 2]))
            input_falls.append(float(values[base + 3]))

        self.backend.transition_state_step(
            self.current_values,
            self.target_values,
            self.start_times,
            self.start_values,
            self.delays,
            self.rise_times,
            self.fall_times,
            self.active_flags,
            self.initialized_flags,
            input_targets,
            input_delays,
            input_rises,
            input_falls,
            self.output_values,
            time,
            self.default_transition,
            initial_condition_mode=initial_condition_mode,
        )

        for idx, output_slot in enumerate(self.program.output_node_slots):
            reference_slot = self.program.reference_node_slots[idx]
            reference = 0.0 if reference_slot is None else float(node_values[reference_slot])
            output_bias = float(values[idx * 6 + 4])
            output_scale = float(values[idx * 6 + 5])
            node_values[output_slot] = (
                reference + output_bias + output_scale * float(self.output_values[idx])
            )
        return tuple(float(value) for value in self.output_values)

    def next_breakpoint(
        self,
        time: float,
        min_ramp_time: float = 0.0,
    ) -> Optional[float]:
        """Return the next transition breakpoint visible to a simulator loop.

        The runtime owns the typed-array transition state, so the outer engine
        cannot infer active ramp points unless this contract is explicit.
        """
        return self.backend.next_transition_breakpoint(
            self.start_times,
            self.start_values,
            self.target_values,
            self.delays,
            self.rise_times,
            self.fall_times,
            self.active_flags,
            float(time),
            float(min_ramp_time),
        )


def try_build_rust_transition_contribution_runtime(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    backend: RustBackend,
    *,
    default_transition: float,
) -> Optional[RustTransitionContributionRuntime]:
    program = encode_transition_contribution_program(stmt_ir, bindings, node_slots)
    if program is None:
        return None
    return RustTransitionContributionRuntime(
        program,
        backend,
        default_transition=default_transition,
    )


def _append_transition_contribution_specs(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    output_slots: list[int],
    reference_slots: list[Optional[int]],
    expr_segments: list[Tuple[BodyExprOp, ...]],
) -> bool:
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            if not _append_transition_contribution_specs(
                child,
                bindings,
                node_slots,
                output_slots,
                reference_slots,
                expr_segments,
            ):
                return False
        return True

    if isinstance(stmt_ir, EventStatementIR):
        return True

    if isinstance(stmt_ir, ForStatementIR):
        unrolled = unroll_static_for_statement(stmt_ir)
        if unrolled is None:
            return False
        loop_var = getattr(getattr(stmt_ir.init, "target", None), "name", None)
        if loop_var is not None:
            unrolled = BlockIR(
                tuple(
                    child
                    for child in unrolled.statements
                    if not (
                        isinstance(child, AssignmentIR)
                        and getattr(child.target, "name", None) == loop_var
                    )
                )
            )
        return _append_transition_contribution_specs(
            unrolled,
            bindings,
            node_slots,
            output_slots,
            reference_slots,
            expr_segments,
        )

    if not isinstance(stmt_ir, ContributionIR):
        return False

    target = _encode_transition_contribution_target(stmt_ir.branch, node_slots)
    if target is None:
        return False
    transition_args = _transition_call_args(stmt_ir.expr)
    if transition_args is None:
        return False

    output_slot, reference_slot = target
    encoded_segments = []
    for expr in transition_args:
        encoded = encode_body_expr_ops(expr, bindings, node_slots)
        if encoded is None:
            return False
        encoded_segments.append(encoded)

    output_slots.append(output_slot)
    reference_slots.append(reference_slot)
    expr_segments.extend(encoded_segments)
    return True


def _encode_transition_contribution_target(
    branch: BranchAccessIR,
    node_slots: dict[str, int],
) -> Optional[tuple[int, Optional[int]]]:
    if branch.access_type != "V":
        return None
    output_name = static_node_ref_name(
        branch.node1,
        branch.node1_index,
        branch.node1_index2,
    )
    if output_name is None:
        return None
    output_slot = node_slots.get(output_name)
    if output_slot is None:
        return None
    reference_slot = None
    if branch.node2 is not None:
        reference_name = static_node_ref_name(
            branch.node2,
            branch.node2_index,
            branch.node2_index2,
        )
        if reference_name is None:
            return None
        reference_slot = node_slots.get(reference_name)
        if reference_slot is None:
            return None
    return output_slot, reference_slot


def _transition_call_args(expr) -> Optional[Tuple[object, object, object, object, object, object]]:
    parts = _scaled_transition_parts(expr)
    if parts is None:
        return None
    target, delay, rise, fall, output_bias, output_scale = parts
    return target, delay, rise, fall, output_bias, output_scale


def _direct_transition_call_args(expr) -> Optional[Tuple[object, object, object, object]]:
    if not isinstance(expr, FunctionCallIR) or expr.name != "transition":
        return None
    if len(expr.args) > 4:
        return None
    zero = LiteralIR(0.0)
    target = expr.args[0] if len(expr.args) > 0 else zero
    delay = expr.args[1] if len(expr.args) > 1 else zero
    rise = expr.args[2] if len(expr.args) > 2 else zero
    fall = expr.args[3] if len(expr.args) > 3 else rise
    return target, delay, rise, fall


def _scaled_transition_parts(expr) -> Optional[Tuple[object, object, object, object, object, object]]:
    direct = _direct_transition_call_args(expr)
    if direct is not None:
        return (*direct, LiteralIR(0.0), LiteralIR(1.0))

    if isinstance(expr, UnaryExprIR) and expr.op == "-":
        child = _scaled_transition_parts(expr.operand)
        if child is None:
            return None
        target, delay, rise, fall, bias, scale = child
        return target, delay, rise, fall, UnaryExprIR("-", bias), UnaryExprIR("-", scale)

    if not isinstance(expr, BinaryExprIR):
        return None

    if expr.op in {"+", "-"}:
        left = _scaled_transition_parts(expr.left)
        right = _scaled_transition_parts(expr.right)
        if left is not None and right is not None:
            return None
        if left is not None:
            target, delay, rise, fall, bias, scale = left
            output_bias = BinaryExprIR(expr.op, bias, expr.right)
            return target, delay, rise, fall, output_bias, scale
        if right is not None:
            target, delay, rise, fall, bias, scale = right
            if expr.op == "+":
                output_bias = BinaryExprIR("+", expr.left, bias)
                output_scale = scale
            else:
                output_bias = BinaryExprIR("-", expr.left, bias)
                output_scale = UnaryExprIR("-", scale)
            return target, delay, rise, fall, output_bias, output_scale
        return None

    if expr.op == "*":
        left = _scaled_transition_parts(expr.left)
        right = _scaled_transition_parts(expr.right)
        if left is not None and right is not None:
            return None
        if left is not None:
            target, delay, rise, fall, bias, scale = left
            return (
                target,
                delay,
                rise,
                fall,
                BinaryExprIR("*", bias, expr.right),
                BinaryExprIR("*", scale, expr.right),
            )
        if right is not None:
            target, delay, rise, fall, bias, scale = right
            return (
                target,
                delay,
                rise,
                fall,
                BinaryExprIR("*", expr.left, bias),
                BinaryExprIR("*", expr.left, scale),
            )
        return None

    return None
