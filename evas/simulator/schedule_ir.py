"""Event schedule IR for Verilog-A body lowering.

094c keeps event controls separate from statement bodies so future Rust
executors can own event ordering without re-parsing statement IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

from evas.compiler.ast_nodes import CombinedEvent, EventExpr
from evas.simulator.expr_ir import (
    BindingTableIR,
    ExprIR,
    LoweringContext,
    emit_python,
    encode_body_expr_ops,
    lower_expr,
)
from evas.simulator.rust_backend import (
    BODY_EXPR_READ_NODE,
    BODY_EXPR_READ_STATE,
    BodyExprOp,
)

EVENT_DUE_INITIAL_STEP = "initial_step"
EVENT_DUE_CROSS = "cross"
EVENT_DUE_ABOVE = "above"
EVENT_DUE_TIMER = "timer"
EVENT_DUE_FINAL_STEP = "final_step"


@dataclass(frozen=True)
class EventTriggerIR:
    event_type: str
    args: Tuple[ExprIR, ...]
    direction: Optional[int] = None
    time_tol: Optional[ExprIR] = None
    expr_tol: Optional[ExprIR] = None


@dataclass(frozen=True)
class CombinedEventIR:
    events: Tuple["EventIR", ...]


EventIR = Union[EventTriggerIR, CombinedEventIR]


@dataclass(frozen=True)
class EventDueTriggerProgram:
    """A conservative Rust-ready trigger expression segment.

    The trigger plan is intentionally separate from event ordering and body
    execution.  It only describes the typed-array inputs Rust needs to evaluate
    trigger values or static timer parameters.
    """

    kind: str
    expr_ops: Tuple[BodyExprOp, ...] = ()
    direction: int = 0
    time_tol_ops: Tuple[BodyExprOp, ...] = ()
    expr_tol_ops: Tuple[BodyExprOp, ...] = ()
    timer_start_ops: Tuple[BodyExprOp, ...] = ()
    timer_period_ops: Tuple[BodyExprOp, ...] = ()


@dataclass(frozen=True)
class EventDueProgram:
    """Ordered trigger list for a future Rust event due/order scheduler."""

    triggers: Tuple[EventDueTriggerProgram, ...]


def lower_event(
    event: object,
    context: Optional[LoweringContext] = None,
) -> Optional[EventIR]:
    ctx = context or LoweringContext.veriloga_body()

    if isinstance(event, CombinedEvent):
        lowered = []
        for child in event.events:
            child_ir = lower_event(child, ctx)
            if child_ir is None:
                return None
            lowered.append(child_ir)
        return CombinedEventIR(tuple(lowered))

    if isinstance(event, EventExpr):
        args = []
        for arg in event.args:
            arg_ir = lower_expr(arg, ctx)
            if arg_ir is None:
                return None
            args.append(arg_ir)
        time_tol = lower_expr(event.time_tol_expr, ctx) if event.time_tol_expr else None
        if event.time_tol_expr is not None and time_tol is None:
            return None
        expr_tol = lower_expr(event.expr_tol_expr, ctx) if event.expr_tol_expr else None
        if event.expr_tol_expr is not None and expr_tol is None:
            return None
        return EventTriggerIR(
            event_type=event.event_type.name,
            args=tuple(args),
            direction=event.direction,
            time_tol=time_tol,
            expr_tol=expr_tol,
        )

    return None


def encode_event_due_program(
    event_ir: EventIR,
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
) -> Optional[EventDueProgram]:
    """Encode trigger expressions for Rust-side due/order staging.

    Supported subset:
    - ``initial_step`` without arguments.
    - ``cross(expr, direction?, ttol?, vtol?)`` where ``expr`` can be evaluated
      by the 094e body expression ABI.
    - ``above(expr)`` using the same expression ABI.
    - ``timer(start)`` where ``start`` can be a typed body expression, including
      state-owned absolute timer targets.
    - static ``timer(start, period)`` where periodic timer expressions do not
      read node or state values.
    - ``final_step`` without arguments.

    The returned plan does not execute event bodies and does not decide global
    ordering.  It is the reusable data layout that later production schedulers
    can feed into Rust detector/timer queues.
    """

    triggers: list[EventDueTriggerProgram] = []
    if not _append_event_due_program(event_ir, bindings, node_slots, triggers):
        return None
    return EventDueProgram(tuple(triggers))


def event_due_expr_segments(program: EventDueProgram) -> Tuple[Tuple[BodyExprOp, ...], ...]:
    """Return trigger-value expression segments in source event order."""

    return tuple(trigger.expr_ops for trigger in program.triggers if trigger.expr_ops)


def event_due_timer_segments(
    program: EventDueProgram,
) -> Tuple[Tuple[BodyExprOp, ...], ...]:
    """Return timer start/period expression segments in source event order."""

    segments: list[Tuple[BodyExprOp, ...]] = []
    for trigger in program.triggers:
        if trigger.kind != EVENT_DUE_TIMER:
            continue
        segments.append(trigger.timer_start_ops)
        if trigger.timer_period_ops:
            segments.append(trigger.timer_period_ops)
    return tuple(segments)


def emit_event_python(event_ir: EventIR) -> str:
    if isinstance(event_ir, CombinedEventIR):
        children = ", ".join(emit_event_python(child) for child in event_ir.events)
        return f"combined_event(({children},))"

    if isinstance(event_ir, EventTriggerIR):
        args = ", ".join(emit_python(arg) for arg in event_ir.args)
        args_tuple = "()" if not args else f"({args},)"
        time_tol = "None" if event_ir.time_tol is None else emit_python(event_ir.time_tol)
        expr_tol = "None" if event_ir.expr_tol is None else emit_python(event_ir.expr_tol)
        return (
            f"event_trigger({event_ir.event_type!r}, {args_tuple}, "
            f"{event_ir.direction!r}, {time_tol}, {expr_tol})"
        )

    raise TypeError(f"unsupported EventIR node: {event_ir!r}")


def _append_event_due_program(
    event_ir: EventIR,
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
    triggers: list[EventDueTriggerProgram],
) -> bool:
    if isinstance(event_ir, CombinedEventIR):
        for child in event_ir.events:
            if not _append_event_due_program(child, bindings, node_slots, triggers):
                return False
        return True

    if not isinstance(event_ir, EventTriggerIR):
        return False

    event_type = event_ir.event_type.upper()
    if event_type == "INITIAL_STEP":
        if event_ir.args or event_ir.time_tol is not None or event_ir.expr_tol is not None:
            return False
        triggers.append(EventDueTriggerProgram(kind=EVENT_DUE_INITIAL_STEP))
        return True

    if event_type == "CROSS":
        if len(event_ir.args) != 1:
            return False
        expr_ops = encode_body_expr_ops(event_ir.args[0], bindings, node_slots)
        if expr_ops is None:
            return False
        time_tol_ops = _encode_optional_event_expr(
            event_ir.time_tol, bindings, node_slots
        )
        expr_tol_ops = _encode_optional_event_expr(
            event_ir.expr_tol, bindings, node_slots
        )
        if time_tol_ops is None or expr_tol_ops is None:
            return False
        triggers.append(
            EventDueTriggerProgram(
                kind=EVENT_DUE_CROSS,
                expr_ops=expr_ops,
                direction=0 if event_ir.direction is None else int(event_ir.direction),
                time_tol_ops=time_tol_ops,
                expr_tol_ops=expr_tol_ops,
            )
        )
        return True

    if event_type == "ABOVE":
        if len(event_ir.args) != 1:
            return False
        expr_ops = encode_body_expr_ops(event_ir.args[0], bindings, node_slots)
        if expr_ops is None:
            return False
        triggers.append(
            EventDueTriggerProgram(
                kind=EVENT_DUE_ABOVE,
                expr_ops=expr_ops,
                direction=1 if event_ir.direction is None else int(event_ir.direction),
            )
        )
        return True

    if event_type == "TIMER":
        if len(event_ir.args) not in {1, 2}:
            return False
        start_ops = encode_body_expr_ops(event_ir.args[0], bindings, node_slots)
        if start_ops is None:
            return False
        period_ops: Tuple[BodyExprOp, ...] = ()
        if len(event_ir.args) == 2:
            if not _is_static_timer_expr(start_ops):
                return False
            encoded_period = encode_body_expr_ops(event_ir.args[1], bindings, node_slots)
            if encoded_period is None or not _is_static_timer_expr(encoded_period):
                return False
            period_ops = encoded_period
        triggers.append(
            EventDueTriggerProgram(
                kind=EVENT_DUE_TIMER,
                timer_start_ops=start_ops,
                timer_period_ops=period_ops,
            )
        )
        return True

    if event_type == "FINAL_STEP":
        if event_ir.args or event_ir.time_tol is not None or event_ir.expr_tol is not None:
            return False
        triggers.append(EventDueTriggerProgram(kind=EVENT_DUE_FINAL_STEP))
        return True

    return False


def _encode_optional_event_expr(
    expr_ir: Optional[ExprIR],
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
) -> Optional[Tuple[BodyExprOp, ...]]:
    if expr_ir is None:
        return ()
    return encode_body_expr_ops(expr_ir, bindings, node_slots)


def _is_static_timer_expr(expr_ops: Tuple[BodyExprOp, ...]) -> bool:
    for op in expr_ops:
        if op.op_kind in {BODY_EXPR_READ_NODE, BODY_EXPR_READ_STATE}:
            return False
    return True
