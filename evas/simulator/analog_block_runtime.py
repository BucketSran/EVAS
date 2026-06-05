"""Shadow analog-block runtime for 094 event + transition batches.

This is a conservative integration layer.  It only accepts analog blocks whose
event statements appear before continuous direct-transition contributions.  It
is not connected to production simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence, Optional, Tuple

from evas.simulator.event_due_runtime import (
    RustAnalogBlockEventRuntime,
    try_build_rust_event_only_analog_block_runtime,
)
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import RustBackend
from evas.simulator.stmt_ir import BlockIR, ContributionIR, EventStatementIR, lower_stmt
from evas.simulator.transition_runtime import (
    RustTransitionContributionRuntime,
    try_build_rust_transition_contribution_runtime,
)


@dataclass(frozen=True)
class RustAnalogBlockStepResult:
    fired_event_statements: Tuple[int, ...]
    transition_outputs: Tuple[float, ...]


class RustAnalogBlockShadowRuntime:
    """Shadow runtime for event statements followed by transition outputs."""

    def __init__(
        self,
        *,
        event_runtime: RustAnalogBlockEventRuntime,
        transition_runtime: RustTransitionContributionRuntime,
    ):
        self.event_runtime = event_runtime
        self.transition_runtime = transition_runtime

    def step(
        self,
        *,
        time: float,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        param_values: MutableSequence[float],
        initial_step: bool = False,
    ) -> RustAnalogBlockStepResult:
        fired = self.event_runtime.step(
            time=time,
            node_values=node_values,
            state_values=state_values,
            param_values=param_values,
            initial_step=initial_step,
        )
        transition_outputs = self.transition_runtime.step(
            time=time,
            node_values=node_values,
            state_values=state_values,
            param_values=param_values,
            initial_condition_mode=initial_step,
        )
        return RustAnalogBlockStepResult(
            fired_event_statements=fired,
            transition_outputs=transition_outputs,
        )

    def next_breakpoint(
        self,
        time: float,
        min_ramp_time: float = 0.0,
    ) -> Optional[float]:
        """Return the next runtime-owned transition breakpoint, if any."""
        return self.transition_runtime.next_breakpoint(time, min_ramp_time)


def try_build_event_then_transition_shadow_runtime(
    module,
    backend: RustBackend,
    node_slots: dict[str, int],
    *,
    default_transition: float,
) -> Optional[RustAnalogBlockShadowRuntime]:
    """Build the 094o conservative analog-block shadow runtime.

    The accepted source shape is:

    1. one or more event statements;
    2. followed by one or more direct ``transition()`` voltage contributions.

    Unsupported statements or event statements after continuous contributions
    return ``None``.
    """

    body = lower_stmt(getattr(getattr(module, "analog_block", None), "body", None))
    if not isinstance(body, BlockIR):
        return None

    seen_contribution = False
    event_count = 0
    contribution_count = 0
    for stmt in body.statements:
        if isinstance(stmt, EventStatementIR):
            if seen_contribution:
                return None
            event_count += 1
            continue
        if isinstance(stmt, ContributionIR):
            seen_contribution = True
            contribution_count += 1
            continue
        return None

    if event_count == 0 or contribution_count == 0:
        return None

    event_runtime = try_build_rust_event_only_analog_block_runtime(
        module,
        backend,
        node_slots,
    )
    if event_runtime is None:
        return None

    bindings = build_state_binding_ir(module)
    transition_runtime = try_build_rust_transition_contribution_runtime(
        body,
        bindings,
        node_slots,
        backend,
        default_transition=default_transition,
    )
    if transition_runtime is None:
        return None

    return RustAnalogBlockShadowRuntime(
        event_runtime=event_runtime,
        transition_runtime=transition_runtime,
    )
