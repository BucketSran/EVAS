"""Diagnostic planner for ordered event/output/transition Rust segments.

This module does not execute simulation.  It checks whether a lowered analog
block has the source-order shape needed by a future whole-segment Rust runtime:
event statements first, followed by continuous output/transition statements.
The result is coverage metadata only; production simulation remains Python
owned unless engine.py explicitly opts into a runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Mapping, Tuple

from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.schedule_ir import encode_event_due_program
from evas.simulator.stmt_ir import (
    AssignmentIR,
    BlockIR,
    CaseStatementIR,
    ContributionIR,
    EventStatementIR,
    ForStatementIR,
    IfStatementIR,
    SystemTaskIR,
    WhileStatementIR,
    classify_body_stmt_ops_rejection,
    lower_stmt,
)
from evas.simulator.transition_runtime import encode_transition_contribution_program


EVENT_TRANSITION_CORE_TAGS = frozenset(
    {
        "event_statement",
        "event_initial_step",
        "event_cross",
        "event_timer",
        "event_above",
        "event_combined",
        "event_final_step",
        "transition_expr",
        "complex_if_write_set",
        "special_identifier:$abstime",
        "differential_output_target",
    }
)

EVENT_TRANSITION_ORDERED_V1_TAGS = frozenset(
    EVENT_TRANSITION_CORE_TAGS
    | {
        "unsupported_binary_operator:>>",
        "unsupported_binary_operator:<<",
        "unsupported_unary_operator:~",
        "array_assignment_target",
        "array_read_or_dynamic_index",
        "for_loop",
        "indexed_output_target",
        "indexed_branch_read",
        "system_task:$bound_step",
    }
)

EVENT_TRANSITION_SIDE_EFFECT_BOUNDARY_TAGS = frozenset(
    EVENT_TRANSITION_ORDERED_V1_TAGS
    | {
        "non_numeric_literal",
        "system_task:$strobe",
        "system_task:$fwrite",
        "system_task:$fclose",
        "system_function:$fopen",
    }
)

EVENT_TRANSITION_PROFILE_SUPPORT = {
    "event_transition_core": EVENT_TRANSITION_CORE_TAGS,
    "event_transition_ordered_v1": EVENT_TRANSITION_ORDERED_V1_TAGS,
    "event_transition_with_side_effect_boundary": (
        EVENT_TRANSITION_SIDE_EFFECT_BOUNDARY_TAGS
    ),
}


@dataclass(frozen=True)
class EventTransitionSegmentPlan:
    accepted: bool
    profile: str
    reason: str
    blocker_tags: Tuple[str, ...] = ()
    event_count: int = 0
    due_trigger_count: int = 0
    output_write_count: int = 0
    transition_count: int = 0
    state_assignment_count: int = 0
    side_effect_count: int = 0
    control_flow_count: int = 0


def analyze_event_transition_segment_plan(
    module: object,
    node_names: Tuple[str, ...],
    *,
    profile: str,
    supported_tags: AbstractSet[str],
) -> EventTransitionSegmentPlan:
    """Return a conservative source-order plan for a future Rust segment.

    The tag support set tells the planner which semantic blockers a planned
    runtime claims to handle.  The planner adds an independent source-shape
    check: top-level event statements must appear before continuous statements.
    This keeps the diagnostic closer to implementation reality than a pure
    rejection-tag count.
    """

    analog_block = getattr(module, "analog_block", None)
    body_ast = getattr(analog_block, "body", None)
    body = lower_stmt(body_ast)
    if not isinstance(body, BlockIR):
        return _reject(profile, "stmt_lower_failed", ("stmt_lower_failed",))

    bindings = build_state_binding_ir(module)
    node_slots = {name: index for index, name in enumerate(node_names)}
    whole_tags = _normal_tags(
        classify_body_stmt_ops_rejection(body, bindings, node_slots)
    )
    unsupported = sorted(set(whole_tags) - set(supported_tags))
    if unsupported:
        return _reject(profile, "unsupported_tags", tuple(unsupported))

    seen_continuous = False
    event_count = 0
    due_trigger_count = 0
    output_write_count = 0
    transition_count = 0
    state_assignment_count = 0
    side_effect_count = 0
    control_flow_count = 0

    for stmt in body.statements:
        if isinstance(stmt, EventStatementIR):
            if seen_continuous:
                return _reject(
                    profile,
                    "event_after_continuous_statement",
                    ("event_after_continuous_statement",),
                )
            due_program = encode_event_due_program(stmt.event, bindings, node_slots)
            if due_program is None:
                return _reject(
                    profile,
                    "event_due_unencodable",
                    ("event_due_unencodable",),
                )
            event_body_tags = _normal_tags(
                classify_body_stmt_ops_rejection(stmt.body, bindings, node_slots)
            )
            body_unsupported = sorted(set(event_body_tags) - set(supported_tags))
            if body_unsupported:
                return _reject(
                    profile,
                    "event_body_unsupported_tags",
                    tuple(body_unsupported),
                )
            event_count += 1
            due_trigger_count += len(due_program.triggers)
            continue

        if isinstance(stmt, ContributionIR):
            seen_continuous = True
            stmt_tags = _normal_tags(
                classify_body_stmt_ops_rejection(stmt, bindings, node_slots)
            )
            stmt_unsupported = sorted(set(stmt_tags) - set(supported_tags))
            if stmt_unsupported:
                return _reject(
                    profile,
                    "continuous_contribution_unsupported_tags",
                    tuple(stmt_unsupported),
                )
            transition_program = encode_transition_contribution_program(
                BlockIR((stmt,)), bindings, node_slots
            )
            if transition_program is not None:
                transition_count += transition_program.contribution_count
            else:
                output_write_count += 1
            continue

        if isinstance(stmt, AssignmentIR):
            seen_continuous = True
            stmt_tags = _normal_tags(
                classify_body_stmt_ops_rejection(stmt, bindings, node_slots)
            )
            stmt_unsupported = sorted(set(stmt_tags) - set(supported_tags))
            if stmt_unsupported:
                return _reject(
                    profile,
                    "continuous_assignment_unsupported_tags",
                    tuple(stmt_unsupported),
                )
            state_assignment_count += 1
            continue

        if isinstance(
            stmt,
            (IfStatementIR, ForStatementIR, WhileStatementIR, CaseStatementIR),
        ):
            seen_continuous = True
            stmt_tags = _normal_tags(
                classify_body_stmt_ops_rejection(stmt, bindings, node_slots)
            )
            stmt_unsupported = sorted(set(stmt_tags) - set(supported_tags))
            if stmt_unsupported:
                return _reject(
                    profile,
                    "control_flow_unsupported_tags",
                    tuple(stmt_unsupported),
                )
            control_flow_count += 1
            continue

        if isinstance(stmt, SystemTaskIR):
            seen_continuous = True
            stmt_tags = _normal_tags(
                classify_body_stmt_ops_rejection(stmt, bindings, node_slots)
            )
            stmt_unsupported = sorted(set(stmt_tags) - set(supported_tags))
            if stmt_unsupported:
                return _reject(
                    profile,
                    "side_effect_unsupported_tags",
                    tuple(stmt_unsupported),
                )
            side_effect_count += 1
            continue

        return _reject(
            profile,
            "unsupported_top_level_statement",
            (type(stmt).__name__,),
        )

    if (
        event_count == 0
        and output_write_count == 0
        and transition_count == 0
        and state_assignment_count == 0
        and control_flow_count == 0
    ):
        return _reject(profile, "empty_or_side_effect_only_segment", ())

    return EventTransitionSegmentPlan(
        accepted=True,
        profile=profile,
        reason="ok",
        event_count=event_count,
        due_trigger_count=due_trigger_count,
        output_write_count=output_write_count,
        transition_count=transition_count,
        state_assignment_count=state_assignment_count,
        side_effect_count=side_effect_count,
        control_flow_count=control_flow_count,
    )


def summarize_event_transition_plans(
    plans: Mapping[str, EventTransitionSegmentPlan],
) -> dict:
    """Summarize planner results with stable JSON-compatible values."""

    accepted = [name for name, plan in plans.items() if plan.accepted]
    return {
        "accepted_profiles": tuple(accepted),
        "rejection_reasons": {
            name: plan.reason for name, plan in plans.items() if not plan.accepted
        },
        "blocker_tags": {
            name: plan.blocker_tags
            for name, plan in plans.items()
            if not plan.accepted and plan.blocker_tags
        },
        "event_count": max((plan.event_count for plan in plans.values()), default=0),
        "due_trigger_count": max(
            (plan.due_trigger_count for plan in plans.values()), default=0
        ),
        "output_write_count": max(
            (plan.output_write_count for plan in plans.values()), default=0
        ),
        "transition_count": max(
            (plan.transition_count for plan in plans.values()), default=0
        ),
        "state_assignment_count": max(
            (plan.state_assignment_count for plan in plans.values()), default=0
        ),
        "side_effect_count": max(
            (plan.side_effect_count for plan in plans.values()), default=0
        ),
        "control_flow_count": max(
            (plan.control_flow_count for plan in plans.values()), default=0
        ),
    }


def _reject(
    profile: str,
    reason: str,
    blocker_tags: Tuple[str, ...],
) -> EventTransitionSegmentPlan:
    return EventTransitionSegmentPlan(
        accepted=False,
        profile=profile,
        reason=reason,
        blocker_tags=tuple(sorted(blocker_tags)),
    )


def _normal_tags(tags: Tuple[str, ...]) -> Tuple[str, ...]:
    """Drop the legacy generic marker used when no diagnostic tag is emitted."""

    return tuple(tag for tag in tags if tag != "body_stmt_ops_unsupported")
