"""Statement IR for Verilog-A body lowering.

094b represents Verilog-A analog block statements independently from the
generated Python model source.  It is a validation/data structure layer only;
production simulation still uses the existing compiled Python evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from evas.compiler.ast_nodes import (
    ArrayAccess,
    Assignment,
    Block,
    CaseStatement,
    Contribution,
    EventStatement,
    ForStatement,
    Identifier,
    IfStatement,
    SystemTask,
    WhileStatement,
)
from evas.simulator.expr_ir import (
    SYMBOL_STATE_SCALAR,
    ArrayAccessIR,
    BinaryExprIR,
    BindingTableIR,
    BranchAccessIR,
    ExprIR,
    FunctionCallIR,
    IdentifierIR,
    LiteralIR,
    LoweringContext,
    MethodCallIR,
    TernaryExprIR,
    UnaryExprIR,
    emit_python,
    encode_body_expr_ops,
    iter_identifier_names,
    lower_expr,
    resolve_static_array_element_binding,
    static_array_element_name,
    static_node_ref_name,
)
from evas.simulator.rust_backend import (
    BODY_STMT_BOUND_STEP,
    BODY_STMT_ELSE,
    BODY_STMT_ENDIF,
    BODY_STMT_ENDWHILE,
    BODY_STMT_FILE_CLOSE,
    BODY_STMT_FILE_OPEN,
    BODY_STMT_FILE_WRITE,
    BODY_STMT_IF,
    BODY_STMT_STROBE,
    BODY_STMT_WHILE,
    BODY_TARGET_NODE,
    BODY_TARGET_STATE,
    BodyExprOp,
    BodyStmtOp,
)
from evas.simulator.schedule_ir import (
    CombinedEventIR,
    EventIR,
    EventTriggerIR,
    emit_event_python,
    lower_event,
)


@dataclass(frozen=True)
class StatementLoweringContext:
    expr_context: LoweringContext = LoweringContext.veriloga_body()
    allowed_system_tasks: frozenset[str] = frozenset(
        {
            "$bound_step",
            "$fclose",
            "$fdisplay",
            "$fwrite",
            "$strobe",
            "$display",
        }
    )

    @classmethod
    def veriloga_body(cls) -> "StatementLoweringContext":
        return cls()


@dataclass(frozen=True)
class AssignmentIR:
    target: Union[IdentifierIR, ArrayAccessIR]
    value: ExprIR


@dataclass(frozen=True)
class ContributionIR:
    branch: BranchAccessIR
    expr: ExprIR


@dataclass(frozen=True)
class EventStatementIR:
    event: EventIR
    body: "StmtIR"


@dataclass(frozen=True)
class BlockIR:
    statements: Tuple["StmtIR", ...]


@dataclass(frozen=True)
class IfStatementIR:
    cond: ExprIR
    then_body: "StmtIR"
    else_body: Optional["StmtIR"] = None


@dataclass(frozen=True)
class ForStatementIR:
    init: AssignmentIR
    cond: ExprIR
    update: AssignmentIR
    body: "StmtIR"


@dataclass(frozen=True)
class WhileStatementIR:
    cond: ExprIR
    body: "StmtIR"


@dataclass(frozen=True)
class CaseItemIR:
    values: Tuple[ExprIR, ...]
    body: "StmtIR"


@dataclass(frozen=True)
class CaseStatementIR:
    expr: ExprIR
    items: Tuple[CaseItemIR, ...]


@dataclass(frozen=True)
class SystemTaskIR:
    name: str
    args: Tuple[ExprIR, ...]


@dataclass(frozen=True)
class BodyStmtProgram:
    """094f statement write-set encoded for the 094e Rust body ABI."""

    stmt_ops: Tuple[BodyStmtOp, ...]
    expr_ops: Tuple[BodyExprOp, ...]


@dataclass(frozen=True)
class EventBodyProgram:
    """094g event trigger paired with a Rust body write-set program."""

    event: EventIR
    body_program: BodyStmtProgram


StmtIR = Union[
    AssignmentIR,
    ContributionIR,
    EventStatementIR,
    BlockIR,
    IfStatementIR,
    ForStatementIR,
    WhileStatementIR,
    CaseStatementIR,
    SystemTaskIR,
]


def lower_stmt(
    stmt: object,
    context: Optional[StatementLoweringContext] = None,
) -> Optional[StmtIR]:
    ctx = context or StatementLoweringContext.veriloga_body()

    if isinstance(stmt, Assignment):
        target = _lower_assignment_target(stmt.target, ctx)
        value = lower_expr(stmt.value, ctx.expr_context)
        if target is None or value is None:
            return None
        return AssignmentIR(target, value)

    if isinstance(stmt, Contribution):
        branch = lower_expr(stmt.branch, ctx.expr_context)
        expr = lower_expr(stmt.expr, ctx.expr_context)
        if not isinstance(branch, BranchAccessIR) or expr is None:
            return None
        return ContributionIR(branch, expr)

    if isinstance(stmt, EventStatement):
        event = lower_event(stmt.event, ctx.expr_context)
        body = lower_stmt(stmt.body, ctx)
        if event is None or body is None:
            return None
        return EventStatementIR(event, body)

    if isinstance(stmt, Block):
        lowered = []
        for child in stmt.statements:
            child_ir = lower_stmt(child, ctx)
            if child_ir is None:
                return None
            lowered.append(child_ir)
        return BlockIR(tuple(lowered))

    if isinstance(stmt, IfStatement):
        cond = lower_expr(stmt.cond, ctx.expr_context)
        then_body = lower_stmt(stmt.then_body, ctx)
        else_body = lower_stmt(stmt.else_body, ctx) if stmt.else_body is not None else None
        if cond is None or then_body is None:
            return None
        if stmt.else_body is not None and else_body is None:
            return None
        return IfStatementIR(cond, then_body, else_body)

    if isinstance(stmt, ForStatement):
        init = lower_stmt(stmt.init, ctx)
        cond = lower_expr(stmt.cond, ctx.expr_context)
        update = lower_stmt(stmt.update, ctx)
        body = lower_stmt(stmt.body, ctx)
        if (
            not isinstance(init, AssignmentIR)
            or cond is None
            or not isinstance(update, AssignmentIR)
            or body is None
        ):
            return None
        return ForStatementIR(init, cond, update, body)

    if isinstance(stmt, WhileStatement):
        cond = lower_expr(stmt.cond, ctx.expr_context)
        body = lower_stmt(stmt.body, ctx)
        if cond is None or body is None:
            return None
        return WhileStatementIR(cond, body)

    if isinstance(stmt, CaseStatement):
        expr = lower_expr(stmt.expr, ctx.expr_context)
        if expr is None:
            return None
        items = []
        for item in stmt.items:
            values = []
            for value in item.values:
                value_ir = lower_expr(value, ctx.expr_context)
                if value_ir is None:
                    return None
                values.append(value_ir)
            body = lower_stmt(item.body, ctx)
            if body is None:
                return None
            items.append(CaseItemIR(tuple(values), body))
        return CaseStatementIR(expr, tuple(items))

    if isinstance(stmt, SystemTask):
        if stmt.name not in ctx.allowed_system_tasks:
            return None
        args = []
        for arg in stmt.args:
            arg_ir = lower_expr(arg, ctx.expr_context)
            if arg_ir is None:
                return None
            args.append(arg_ir)
        return SystemTaskIR(str(stmt.name), tuple(args))

    if stmt is None:
        return BlockIR(())

    return None


def emit_python_statement(stmt_ir: StmtIR, indent: str = "    ") -> Tuple[str, ...]:
    """Emit Python statement lines for compile-only round-trip validation."""

    lines: list[str] = []
    _emit_stmt_lines(stmt_ir, lines, indent, 1)
    return tuple(lines)


def encode_body_stmt_ops(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    *,
    side_effects: object | None = None,
) -> Optional[BodyStmtProgram]:
    """Encode a conservative statement subset for the 094e Rust body ABI.

    This is a write-set encoder, not an event scheduler.  It accepts ordered
    blocks of scalar state assignments, single-node voltage contributions, and
    a conservative if/else subset that can be represented as expression-level
    selects.  Loops, events, dynamic array writes, and differential contribution
    targets return ``None`` so callers can keep the Python runtime as the
    semantic owner.
    """

    stmt_ops: list[BodyStmtOp] = []
    expr_ops: list[BodyExprOp] = []
    if not _append_body_stmt_ops(
        stmt_ir,
        bindings,
        node_slots,
        stmt_ops,
        expr_ops,
        side_effects=side_effects,
    ):
        return None
    return BodyStmtProgram(tuple(stmt_ops), tuple(expr_ops))


def unroll_static_for_statement(stmt_ir: ForStatementIR) -> Optional[BlockIR]:
    """Expand a statically bounded ``for`` loop into ordered statement IR.

    This is a compile-time transform for digital-style state loops such as
    ``for (i=0; i<4; i=i+1) bits[i] = ...``.  Dynamic loop bounds, dynamic
    updates, loop-body writes to the loop variable, or non-terminating loops
    still return ``None`` so production can keep the Python evaluator.
    """

    return _unroll_static_for_statement(stmt_ir)


def classify_body_stmt_ops_rejection(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
) -> Tuple[str, ...]:
    """Return diagnostic tags explaining why body-op encoding may reject.

    This helper is diagnostic-only.  It does not widen the conservative
    production encoder; it lets release-wide coverage audits distinguish event,
    transition, array, loop, and expression blockers instead of collapsing every
    rejected model into ``body_stmt_ops_unsupported``.
    """

    tags = sorted(set(_iter_body_rejection_tags(stmt_ir, bindings, node_slots)))
    return tuple(tags) if tags else ("body_stmt_ops_unsupported",)


def _iter_body_rejection_tags(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            yield from _iter_body_rejection_tags(child, bindings, node_slots)
        return

    if isinstance(stmt_ir, AssignmentIR):
        encoded_target = _encode_assignment_target(stmt_ir.target, bindings)
        if encoded_target is None:
            if isinstance(stmt_ir.target, ArrayAccessIR):
                yield "array_assignment_target"
            else:
                binding = bindings.resolve(stmt_ir.target.name)
                if binding is None:
                    yield "unbound_assignment_target"
                elif binding.kind != SYMBOL_STATE_SCALAR:
                    yield f"non_scalar_state_assignment_target:{binding.kind}"
        if encode_body_expr_ops(stmt_ir.value, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(stmt_ir.value, bindings, node_slots)
        return

    if isinstance(stmt_ir, ContributionIR):
        if _encode_contribution_target(stmt_ir.branch, node_slots) is None:
            yield from _iter_contribution_target_tags(stmt_ir.branch, node_slots)
        if encode_body_expr_ops(stmt_ir.expr, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(stmt_ir.expr, bindings, node_slots)
        return

    if isinstance(stmt_ir, EventStatementIR):
        yield "event_statement"
        yield from _iter_event_rejection_tags(stmt_ir.event, bindings, node_slots)
        yield from _iter_body_rejection_tags(stmt_ir.body, bindings, node_slots)
        return

    if isinstance(stmt_ir, IfStatementIR):
        cond_ops = encode_body_expr_ops(stmt_ir.cond, bindings, node_slots)
        if cond_ops is None:
            yield from _iter_expr_rejection_tags(stmt_ir.cond, bindings, node_slots)
        yield from _iter_body_rejection_tags(stmt_ir.then_body, bindings, node_slots)
        if stmt_ir.else_body is not None:
            yield from _iter_body_rejection_tags(stmt_ir.else_body, bindings, node_slots)
        return

    if isinstance(stmt_ir, ForStatementIR):
        unrolled = _unroll_static_for_statement(stmt_ir)
        if unrolled is not None:
            yield from _iter_body_rejection_tags(unrolled, bindings, node_slots)
            return
        guarded_while = _for_statement_to_guarded_while(stmt_ir)
        yield from _iter_body_rejection_tags(guarded_while, bindings, node_slots)
        return

    if isinstance(stmt_ir, WhileStatementIR):
        if encode_body_expr_ops(stmt_ir.cond, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(stmt_ir.cond, bindings, node_slots)
        yield from _iter_body_rejection_tags(stmt_ir.body, bindings, node_slots)
        return

    if isinstance(stmt_ir, CaseStatementIR):
        yield "case_statement"
        if encode_body_expr_ops(stmt_ir.expr, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(stmt_ir.expr, bindings, node_slots)
        for item in stmt_ir.items:
            for value in item.values:
                if encode_body_expr_ops(value, bindings, node_slots) is None:
                    yield from _iter_expr_rejection_tags(value, bindings, node_slots)
            yield from _iter_body_rejection_tags(item.body, bindings, node_slots)
        return

    if isinstance(stmt_ir, SystemTaskIR):
        if _is_noop_body_system_task(stmt_ir):
            return
        if stmt_ir.name == "$bound_step" and stmt_ir.args:
            if encode_body_expr_ops(stmt_ir.args[0], bindings, node_slots) is None:
                yield from _iter_expr_rejection_tags(stmt_ir.args[0], bindings, node_slots)
            return
        yield f"system_task:{stmt_ir.name}"
        for arg in stmt_ir.args:
            if encode_body_expr_ops(arg, bindings, node_slots) is None:
                yield from _iter_expr_rejection_tags(arg, bindings, node_slots)
        return

    yield "unknown_statement"


def _iter_event_rejection_tags(
    event_ir: EventIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    if isinstance(event_ir, CombinedEventIR):
        yield "event_combined"
        for child in event_ir.events:
            yield from _iter_event_rejection_tags(child, bindings, node_slots)
        return

    if not isinstance(event_ir, EventTriggerIR):
        yield "event_unknown"
        return

    event_type = str(event_ir.event_type).lower()
    if event_type:
        yield f"event_{event_type}"
    for expr in event_ir.args:
        if encode_body_expr_ops(expr, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(expr, bindings, node_slots)
    for expr in (event_ir.time_tol, event_ir.expr_tol):
        if expr is not None and encode_body_expr_ops(expr, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(expr, bindings, node_slots)


def _iter_contribution_target_tags(
    branch: BranchAccessIR,
    node_slots: dict[str, int],
):
    if branch.access_type != "V":
        yield "current_contribution_target"
    if branch.node2 is not None:
        yield "differential_output_target"
    node1_name = static_node_ref_name(
        branch.node1,
        branch.node1_index,
        branch.node1_index2,
    )
    if node1_name is None:
        yield "indexed_output_target"
    node2_name = (
        None
        if branch.node2 is None
        else static_node_ref_name(
            branch.node2,
            branch.node2_index,
            branch.node2_index2,
        )
    )
    if branch.node2 is not None and node2_name is None:
        yield "indexed_output_target"
    if node1_name is not None and node1_name not in node_slots:
        yield "unresolved_output_node"
    if node2_name is not None and node2_name not in node_slots:
        yield "unresolved_output_reference_node"


def _iter_expr_rejection_tags(
    expr_ir: ExprIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    if encode_body_expr_ops(expr_ir, bindings, node_slots) is not None:
        return

    if isinstance(expr_ir, LiteralIR):
        if not isinstance(expr_ir.value, (int, float)):
            yield "non_numeric_literal"
        return

    if isinstance(expr_ir, IdentifierIR):
        if expr_ir.name in {"$abstime", "$realtime", "$temperature", "$vt"}:
            yield f"special_identifier:{expr_ir.name}"
            return
        binding = bindings.resolve(expr_ir.name)
        if binding is None:
            yield "unbound_identifier"
        else:
            yield f"unsupported_identifier_kind:{binding.kind}"
        return

    if isinstance(expr_ir, ArrayAccessIR):
        if resolve_static_array_element_binding(expr_ir, bindings) is None:
            yield "array_read_or_dynamic_index"
        if encode_body_expr_ops(expr_ir.index, bindings, node_slots) is None:
            yield from _iter_expr_rejection_tags(expr_ir.index, bindings, node_slots)
        return

    if isinstance(expr_ir, BranchAccessIR):
        if expr_ir.access_type != "V":
            yield "current_read"
        node1_name = static_node_ref_name(
            expr_ir.node1,
            expr_ir.node1_index,
            expr_ir.node1_index2,
        )
        node2_name = (
            None
            if expr_ir.node2 is None
            else static_node_ref_name(
                expr_ir.node2,
                expr_ir.node2_index,
                expr_ir.node2_index2,
            )
        )
        if node1_name is None or (expr_ir.node2 is not None and node2_name is None):
            yield "indexed_branch_read"
        if node1_name is not None and node1_name not in node_slots:
            yield "unresolved_node_read"
        if node2_name is not None and node2_name not in node_slots:
            yield "unresolved_reference_node_read"
        return

    if isinstance(expr_ir, BinaryExprIR):
        if expr_ir.op not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            ">",
            "<",
            ">=",
            "<=",
            "==",
            "!=",
            "&&",
            "||",
            "&",
            "|",
            "^",
            "<<",
            ">>",
        }:
            yield f"unsupported_binary_operator:{expr_ir.op}"
        yield from _iter_expr_rejection_tags(expr_ir.left, bindings, node_slots)
        yield from _iter_expr_rejection_tags(expr_ir.right, bindings, node_slots)
        return

    if isinstance(expr_ir, UnaryExprIR):
        if expr_ir.op not in {"+", "-", "!", "~"}:
            yield f"unsupported_unary_operator:{expr_ir.op}"
        yield from _iter_expr_rejection_tags(expr_ir.operand, bindings, node_slots)
        return

    if isinstance(expr_ir, TernaryExprIR):
        yield from _iter_expr_rejection_tags(expr_ir.cond, bindings, node_slots)
        yield from _iter_expr_rejection_tags(expr_ir.true_expr, bindings, node_slots)
        yield from _iter_expr_rejection_tags(expr_ir.false_expr, bindings, node_slots)
        return

    if isinstance(expr_ir, FunctionCallIR):
        name = str(expr_ir.name)
        if name == "transition":
            yield "transition_expr"
        elif name in {"cross", "above", "timer", "idtmod", "last_crossing", "slew"}:
            yield f"stateful_analog_function:{name}"
        elif name.startswith("$"):
            yield f"system_function:{name}"
        elif name not in {
            "abs",
            "sqrt",
            "exp",
            "ln",
            "log",
            "sin",
            "cos",
            "floor",
            "ceil",
            "min",
            "max",
            "pow",
        }:
            yield f"unsupported_function:{name}"
        for arg in expr_ir.args:
            yield from _iter_expr_rejection_tags(arg, bindings, node_slots)
        return

    if isinstance(expr_ir, MethodCallIR):
        yield f"method_call:{expr_ir.method}"
        for arg in expr_ir.args:
            yield from _iter_expr_rejection_tags(arg, bindings, node_slots)
        return

    yield "expr_unsupported"


def encode_event_body_program(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
) -> Optional[EventBodyProgram]:
    """Encode one event statement body for future Rust event scheduling.

    The scheduler remains Python-owned.  This helper only proves that an event
    body's ordered write-set can be represented by the 094e Rust body ABI once a
    separate due/order layer decides that the event has fired.
    """

    if not isinstance(stmt_ir, EventStatementIR):
        return None
    body_program = encode_body_stmt_ops(stmt_ir.body, bindings, node_slots)
    if body_program is None:
        return None
    return EventBodyProgram(event=stmt_ir.event, body_program=body_program)


def _lower_assignment_target(
    target: object,
    context: StatementLoweringContext,
) -> Optional[Union[IdentifierIR, ArrayAccessIR]]:
    if not isinstance(target, (Identifier, ArrayAccess)):
        return None
    lowered = lower_expr(target, context.expr_context)
    if isinstance(lowered, (IdentifierIR, ArrayAccessIR)):
        return lowered
    return None


def _append_body_stmt_ops(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    stmt_ops: list[BodyStmtOp],
    expr_ops: list[BodyExprOp],
    *,
    side_effects: object | None = None,
) -> bool:
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            if not _append_body_stmt_ops(
                child,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
                side_effects=side_effects,
            ):
                return False
        return True

    if isinstance(stmt_ir, AssignmentIR):
        target = _encode_assignment_target(stmt_ir.target, bindings)
        if target is None:
            return False
        target_kind, target_id, target_integer = target
        if (
            target_kind == BODY_TARGET_STATE
            and isinstance(stmt_ir.value, FunctionCallIR)
            and str(stmt_ir.value.name) == "$fopen"
            and side_effects is not None
        ):
            spec_id = _add_file_open_side_effect(side_effects, stmt_ir.value)
            if spec_id is None:
                return False
            stmt_ops.append(
                BodyStmtOp(
                    target_kind=BODY_STMT_FILE_OPEN,
                    target_id=target_id,
                    expr_start=spec_id,
                    expr_count=0,
                    target_integer=target_integer,
                )
            )
            return True
        return _append_body_write(
            target_kind,
            target_id,
            target_integer,
            stmt_ir.value,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
        )

    if isinstance(stmt_ir, ContributionIR):
        target_id = _encode_contribution_target(stmt_ir.branch, node_slots)
        if target_id is None:
            return False
        expr = _contribution_expr_with_reference(stmt_ir.branch, stmt_ir.expr)
        return _append_body_write(
            BODY_TARGET_NODE,
            target_id,
            False,
            expr,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
        )

    if isinstance(stmt_ir, IfStatementIR):
        cond_ops = encode_body_expr_ops(stmt_ir.cond, bindings, node_slots)
        if cond_ops is None:
            return False
        expr_start = len(expr_ops)
        expr_ops.extend(cond_ops)
        stmt_ops.append(
            BodyStmtOp(
                target_kind=BODY_STMT_IF,
                target_id=0,
                expr_start=expr_start,
                expr_count=len(cond_ops),
                target_integer=False,
            )
        )
        if not _append_body_stmt_ops(
            stmt_ir.then_body,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
            side_effects=side_effects,
        ):
            return False
        if stmt_ir.else_body is not None:
            stmt_ops.append(
                BodyStmtOp(
                    target_kind=BODY_STMT_ELSE,
                    target_id=0,
                    expr_start=0,
                    expr_count=0,
                    target_integer=False,
                )
            )
            if not _append_body_stmt_ops(
                stmt_ir.else_body,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
                side_effects=side_effects,
            ):
                return False
        stmt_ops.append(
            BodyStmtOp(
                target_kind=BODY_STMT_ENDIF,
                target_id=0,
                expr_start=0,
                expr_count=0,
                target_integer=False,
            )
        )
        return True

    if isinstance(stmt_ir, ForStatementIR):
        unrolled = _unroll_static_for_statement(stmt_ir)
        if unrolled is None:
            unrolled = _for_statement_to_guarded_while(stmt_ir)
        return _append_body_stmt_ops(
            unrolled,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
            side_effects=side_effects,
        )

    if isinstance(stmt_ir, WhileStatementIR):
        encoded_cond = encode_body_expr_ops(stmt_ir.cond, bindings, node_slots)
        if encoded_cond is None:
            return False
        expr_start = len(expr_ops)
        expr_ops.extend(encoded_cond)
        stmt_ops.append(
            BodyStmtOp(
                target_kind=BODY_STMT_WHILE,
                target_id=4096,
                expr_start=expr_start,
                expr_count=len(encoded_cond),
                target_integer=False,
            )
        )
        if not _append_body_stmt_ops(
            stmt_ir.body,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
            side_effects=side_effects,
        ):
            return False
        stmt_ops.append(
            BodyStmtOp(
                target_kind=BODY_STMT_ENDWHILE,
                target_id=0,
                expr_start=0,
                expr_count=0,
                target_integer=False,
            )
        )
        return True

    if isinstance(stmt_ir, CaseStatementIR):
        lowered = _case_statement_to_if_chain(stmt_ir)
        if lowered is None:
            return False
        return _append_body_stmt_ops(
            lowered,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
            side_effects=side_effects,
        )

    if isinstance(stmt_ir, SystemTaskIR):
        if side_effects is not None and stmt_ir.name in {"$display", "$strobe"}:
            return _append_strobe_stmt(
                stmt_ir,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
                side_effects,
            )
        if _is_noop_body_system_task(stmt_ir):
            return True
        if side_effects is not None and stmt_ir.name in {
            "$fwrite",
            "$fdisplay",
            "$fstrobe",
        }:
            return _append_file_write_stmt(
                stmt_ir,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
                side_effects,
            )
        if side_effects is not None and stmt_ir.name == "$fclose":
            return _append_file_close_stmt(
                stmt_ir,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
                side_effects,
            )
        if stmt_ir.name != "$bound_step" or not stmt_ir.args:
            return False
        encoded_expr = encode_body_expr_ops(stmt_ir.args[0], bindings, node_slots)
        if encoded_expr is None:
            return False
        expr_start = len(expr_ops)
        expr_ops.extend(encoded_expr)
        stmt_ops.append(
            BodyStmtOp(
                target_kind=BODY_STMT_BOUND_STEP,
                target_id=0,
                expr_start=expr_start,
                expr_count=len(encoded_expr),
                target_integer=False,
            )
        )
        return True

    return False


def _add_file_open_side_effect(side_effects: object, call_ir: FunctionCallIR) -> Optional[int]:
    filename = (
        _resolve_side_effect_string(side_effects, call_ir.args[0])
        if call_ir.args
        else "output.txt"
    )
    mode = (
        _resolve_side_effect_string(side_effects, call_ir.args[1])
        if len(call_ir.args) > 1
        else "w"
    )
    if filename is None or mode is None:
        return None
    add = getattr(side_effects, "add_file_open", None)
    if add is None:
        return None
    return int(add(filename, mode))


def _append_file_write_stmt(
    stmt_ir: SystemTaskIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    stmt_ops: list[BodyStmtOp],
    expr_ops: list[BodyExprOp],
    side_effects: object,
) -> bool:
    if not stmt_ir.args:
        return False
    fmt = ""
    numeric_args = tuple(stmt_ir.args)
    if len(stmt_ir.args) > 1:
        resolved_fmt = _resolve_side_effect_string(side_effects, stmt_ir.args[1])
        if resolved_fmt is None:
            return False
        fmt = resolved_fmt
        numeric_args = (stmt_ir.args[0], *stmt_ir.args[2:])
    add = getattr(side_effects, "add_file_write", None)
    if add is None:
        return False
    spec_id = int(add(fmt))
    expr_start = len(expr_ops)
    for arg in numeric_args:
        encoded = encode_body_expr_ops(arg, bindings, node_slots)
        if encoded is None:
            return False
        expr_ops.extend(encoded)
    stmt_ops.append(
        BodyStmtOp(
            target_kind=BODY_STMT_FILE_WRITE,
            target_id=spec_id,
            expr_start=expr_start,
            expr_count=len(expr_ops) - expr_start,
            target_integer=False,
        )
    )
    return True


def _append_file_close_stmt(
    stmt_ir: SystemTaskIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    stmt_ops: list[BodyStmtOp],
    expr_ops: list[BodyExprOp],
    side_effects: object,
) -> bool:
    if not stmt_ir.args:
        return False
    add = getattr(side_effects, "add_file_close", None)
    if add is None:
        return False
    spec_id = int(add())
    encoded = encode_body_expr_ops(stmt_ir.args[0], bindings, node_slots)
    if encoded is None:
        return False
    expr_start = len(expr_ops)
    expr_ops.extend(encoded)
    stmt_ops.append(
        BodyStmtOp(
            target_kind=BODY_STMT_FILE_CLOSE,
            target_id=spec_id,
            expr_start=expr_start,
            expr_count=len(encoded),
            target_integer=False,
        )
    )
    return True


def _append_strobe_stmt(
    stmt_ir: SystemTaskIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    stmt_ops: list[BodyStmtOp],
    expr_ops: list[BodyExprOp],
    side_effects: object,
) -> bool:
    fmt = ""
    numeric_args = tuple(stmt_ir.args)
    if stmt_ir.args:
        resolved_fmt = _resolve_side_effect_string(side_effects, stmt_ir.args[0])
        if resolved_fmt is None:
            return False
        fmt = resolved_fmt
        numeric_args = tuple(stmt_ir.args[1:])
    add = getattr(side_effects, "add_strobe", None)
    if add is None:
        return False
    spec_id = int(add(fmt))
    expr_start = len(expr_ops)
    for arg in numeric_args:
        encoded = encode_body_expr_ops(arg, bindings, node_slots)
        if encoded is None:
            return False
        expr_ops.extend(encoded)
    stmt_ops.append(
        BodyStmtOp(
            target_kind=BODY_STMT_STROBE,
            target_id=spec_id,
            expr_start=expr_start,
            expr_count=len(expr_ops) - expr_start,
            target_integer=False,
        )
    )
    return True


def _resolve_side_effect_string(side_effects: object, expr_ir: ExprIR) -> Optional[str]:
    resolve = getattr(side_effects, "resolve_string", None)
    if resolve is None:
        return None
    value = resolve(expr_ir)
    return None if value is None else str(value)


def _collect_conditional_write_specs(
    stmt_ir: IfStatementIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    cond_refs = set(iter_identifier_names(stmt_ir.cond))
    then_writes = _collect_body_write_specs(stmt_ir.then_body, bindings, node_slots)
    if then_writes is None or not then_writes:
        return None
    then_writes = _move_cond_ref_writes_to_branch_end(cond_refs, then_writes)
    if then_writes is None:
        return None

    # We lower control flow by reusing the existing expression-level SELECT
    # operator.  That means the condition is evaluated for each target write.
    # It is still safe when a condition-referenced state is only written as the
    # final write in that branch; there are then no later SELECTs that can
    # observe the mutated condition.
    if not _conditional_cond_refs_are_order_safe(cond_refs, then_writes):
        return None

    if stmt_ir.else_body is None:
        return tuple(
            (
                target_key,
                target_kind,
                target_id,
                target_integer,
                current_expr,
                TernaryExprIR(stmt_ir.cond, value_expr, current_expr),
                target_name,
            )
            for (
                target_key,
                target_kind,
                target_id,
                target_integer,
                current_expr,
                value_expr,
                target_name,
            ) in then_writes
        )

    else_writes = _collect_body_write_specs(stmt_ir.else_body, bindings, node_slots)
    if else_writes is None:
        return None
    else_writes = _move_cond_ref_writes_to_branch_end(cond_refs, else_writes)
    if else_writes is None:
        return None

    if not _conditional_cond_refs_are_order_safe(cond_refs, then_writes):
        return None
    if not _conditional_cond_refs_are_order_safe(cond_refs, else_writes):
        return None
    then_by_key = {write[0]: write for write in then_writes}
    else_by_key = {write[0]: write for write in else_writes}
    ordered_keys = [write[0] for write in then_writes]
    ordered_keys.extend(write[0] for write in else_writes if write[0] not in then_by_key)
    merged = []
    for target_key in ordered_keys:
        then_write = then_by_key.get(target_key)
        else_write = else_by_key.get(target_key)
        shape_write = then_write if then_write is not None else else_write
        if shape_write is None:
            return None
        (
            target_key,
            target_kind,
            target_id,
            target_integer,
            current_expr,
            _shape_expr,
            target_name,
        ) = shape_write
        then_expr = then_write[5] if then_write is not None else current_expr
        else_expr = else_write[5] if else_write is not None else current_expr
        merged.append(
            (
                target_key,
                target_kind,
                target_id,
                target_integer,
                current_expr,
                TernaryExprIR(stmt_ir.cond, then_expr, else_expr),
                target_name,
            )
        )
    return tuple(merged)


def _conditional_cond_refs_are_order_safe(
    cond_refs: set[str],
    writes,
) -> bool:
    """Check whether expression-level SELECT preserves one-shot if semantics."""

    if not writes:
        return True
    indices = [
        idx
        for idx, write in enumerate(writes)
        if write[6] is not None and write[6] in cond_refs
    ]
    if not indices:
        return True
    return len(indices) == 1 and indices[0] == len(writes) - 1


def _move_cond_ref_writes_to_branch_end(
    cond_refs: set[str],
    writes,
):
    """Move condition-state writes to the end when that preserves semantics."""

    if not writes:
        return writes
    cond_writes = [write for write in writes if write[6] is not None and write[6] in cond_refs]
    if not cond_writes:
        return writes
    names = [write[6] for write in cond_writes]
    if len(set(names)) != len(names):
        return None
    for idx, write in enumerate(writes):
        target_name = write[6]
        if target_name is None or target_name not in cond_refs:
            continue
        for later in writes[idx + 1 :]:
            if _expr_references_name_outside_select_conditions(later[5], target_name):
                return None
    return tuple(
        [write for write in writes if write[6] is None or write[6] not in cond_refs]
        + cond_writes
    )


def _expr_references_name_outside_select_conditions(expr_ir: ExprIR, name: str) -> bool:
    if isinstance(expr_ir, IdentifierIR):
        return expr_ir.name == name
    if isinstance(expr_ir, ArrayAccessIR):
        return _expr_references_name_outside_select_conditions(expr_ir.index, name)
    if isinstance(expr_ir, BinaryExprIR):
        return _expr_references_name_outside_select_conditions(
            expr_ir.left,
            name,
        ) or _expr_references_name_outside_select_conditions(expr_ir.right, name)
    if isinstance(expr_ir, UnaryExprIR):
        return _expr_references_name_outside_select_conditions(expr_ir.operand, name)
    if isinstance(expr_ir, TernaryExprIR):
        # SELECT conditions model branch guards and should use the pre-branch
        # value.  Payload expressions must not depend on a state write we move.
        return _expr_references_name_outside_select_conditions(
            expr_ir.true_expr,
            name,
        ) or _expr_references_name_outside_select_conditions(expr_ir.false_expr, name)
    if isinstance(expr_ir, FunctionCallIR):
        return any(
            _expr_references_name_outside_select_conditions(arg, name)
            for arg in expr_ir.args
        )
    if isinstance(expr_ir, BranchAccessIR):
        for child in (
            expr_ir.node1_index,
            expr_ir.node1_index2,
            expr_ir.node2_index,
            expr_ir.node2_index2,
        ):
            if child is not None and _expr_references_name_outside_select_conditions(child, name):
                return True
        return False
    if isinstance(expr_ir, MethodCallIR):
        return any(
            _expr_references_name_outside_select_conditions(arg, name)
            for arg in expr_ir.args
        )
    return False


def _collect_body_write_specs(
    stmt_ir: StmtIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    if isinstance(stmt_ir, BlockIR):
        writes = []
        for child in stmt_ir.statements:
            child_writes = _collect_body_write_specs(child, bindings, node_slots)
            if child_writes is None:
                return None
            writes.extend(child_writes)
        return tuple(writes)

    if isinstance(stmt_ir, AssignmentIR):
        encoded = _encode_assignment_target(stmt_ir.target, bindings)
        if encoded is None:
            return None
        target_kind, target_id, target_integer = encoded
        current_expr = _assignment_current_expr(stmt_ir.target, bindings)
        target_name = _assignment_target_name(stmt_ir.target, bindings)
        if current_expr is None or target_name is None:
            return None
        target_key = (target_kind, target_id)
        return (
            (
                target_key,
                target_kind,
                target_id,
                target_integer,
                current_expr,
                stmt_ir.value,
                target_name,
            ),
        )

    if isinstance(stmt_ir, ContributionIR):
        target_id = _encode_contribution_target(stmt_ir.branch, node_slots)
        if target_id is None:
            return None
        target_key = (BODY_TARGET_NODE, target_id)
        expr = _contribution_expr_with_reference(stmt_ir.branch, stmt_ir.expr)
        return (
            (
                target_key,
                BODY_TARGET_NODE,
                target_id,
                False,
                stmt_ir.branch,
                expr,
                None,
            ),
        )

    if isinstance(stmt_ir, IfStatementIR):
        return _collect_conditional_write_specs(stmt_ir, bindings, node_slots)

    if isinstance(stmt_ir, ForStatementIR):
        unrolled = _unroll_static_for_statement(stmt_ir)
        if unrolled is None:
            return None
        return _collect_body_write_specs(unrolled, bindings, node_slots)

    if isinstance(stmt_ir, CaseStatementIR):
        lowered = _case_statement_to_if_chain(stmt_ir)
        if lowered is None:
            return None
        return _collect_body_write_specs(lowered, bindings, node_slots)

    if isinstance(stmt_ir, SystemTaskIR):
        if _is_noop_body_system_task(stmt_ir):
            return ()
        return None

    return None


def _is_noop_body_system_task(stmt_ir: SystemTaskIR) -> bool:
    return stmt_ir.name in {"$display", "$strobe"}


def _contribution_expr_with_reference(
    branch: BranchAccessIR,
    expr: ExprIR,
) -> ExprIR:
    if branch.node2 is None:
        return expr
    reference = BranchAccessIR(
        access_type="V",
        node1=str(branch.node2),
        node2=None,
        node1_index=branch.node2_index,
        node1_index2=branch.node2_index2,
    )
    return BinaryExprIR("+", reference, expr)


def _case_statement_to_if_chain(stmt_ir: CaseStatementIR) -> Optional[StmtIR]:
    """Lower a side-effect-free case tree into an ordered if/else chain."""

    default_body: Optional[StmtIR] = None
    conditional_items: list[tuple[ExprIR, StmtIR]] = []
    for item in stmt_ir.items:
        if not item.values:
            if default_body is None:
                default_body = item.body
            continue
        cond: Optional[ExprIR] = None
        for value in item.values:
            equal = BinaryExprIR("==", stmt_ir.expr, value)
            cond = equal if cond is None else BinaryExprIR("||", cond, equal)
        if cond is None:
            return None
        conditional_items.append((cond, item.body))

    else_body: StmtIR = default_body if default_body is not None else BlockIR(())
    for cond, body in reversed(conditional_items):
        else_body = IfStatementIR(cond=cond, then_body=body, else_body=else_body)
    return else_body


def _unroll_static_for_statement(stmt_ir: ForStatementIR) -> Optional[BlockIR]:
    loop = _static_for_loop_values(stmt_ir)
    if loop is None:
        return None
    loop_var, values = loop
    if _stmt_writes_name(stmt_ir.body, loop_var):
        return None

    statements: list[StmtIR] = [stmt_ir.init]
    for value in values:
        env = {loop_var: int(value)}
        body = _substitute_stmt_ir(stmt_ir.body, env)
        update = _substitute_assignment_ir(stmt_ir.update, env)
        if body is None or update is None:
            return None
        if isinstance(body, BlockIR):
            statements.extend(body.statements)
        else:
            statements.append(body)
        statements.append(update)
    return BlockIR(tuple(statements))


def _for_statement_to_guarded_while(stmt_ir: ForStatementIR) -> BlockIR:
    """Represent a dynamic ``for`` loop with the existing guarded while opcode."""

    return BlockIR(
        (
            stmt_ir.init,
            WhileStatementIR(
                cond=stmt_ir.cond,
                body=BlockIR((stmt_ir.body, stmt_ir.update)),
            ),
        )
    )


def _static_for_loop_values(stmt_ir: ForStatementIR) -> Optional[tuple[str, tuple[int, ...]]]:
    if not isinstance(stmt_ir.init.target, IdentifierIR):
        return None
    if not isinstance(stmt_ir.update.target, IdentifierIR):
        return None
    loop_var = stmt_ir.init.target.name
    if stmt_ir.update.target.name != loop_var:
        return None
    initial = _static_integer_expr_value(stmt_ir.init.value, {})
    if initial is None:
        return None

    values: list[int] = []
    current = int(initial)
    for _guard in range(4096):
        env = {loop_var: current}
        cond = _static_numeric_expr_value(stmt_ir.cond, env)
        if cond is None:
            return None
        if float(cond) == 0.0:
            return loop_var, tuple(values)
        values.append(current)
        next_value = _static_integer_expr_value(stmt_ir.update.value, env)
        if next_value is None or int(next_value) == current:
            return None
        current = int(next_value)
    return None


def _static_integer_expr_value(expr_ir: ExprIR, env: dict[str, int]) -> Optional[int]:
    value = _static_numeric_expr_value(expr_ir, env)
    if value is None:
        return None
    idx = int(value)
    if float(value) != float(idx):
        return None
    return idx


def _static_numeric_expr_value(expr_ir: ExprIR, env: dict[str, int]) -> Optional[float]:
    if isinstance(expr_ir, LiteralIR):
        if not isinstance(expr_ir.value, (int, float)):
            return None
        return float(expr_ir.value)

    if isinstance(expr_ir, IdentifierIR):
        if expr_ir.name not in env:
            return None
        return float(env[expr_ir.name])

    if isinstance(expr_ir, UnaryExprIR):
        value = _static_numeric_expr_value(expr_ir.operand, env)
        if value is None:
            return None
        if expr_ir.op == "+":
            return value
        if expr_ir.op == "-":
            return -value
        if expr_ir.op == "!":
            return 0.0 if value else 1.0
        if expr_ir.op == "~":
            return float(~int(value))
        return None

    if isinstance(expr_ir, BinaryExprIR):
        left = _static_numeric_expr_value(expr_ir.left, env)
        right = _static_numeric_expr_value(expr_ir.right, env)
        if left is None or right is None:
            return None
        try:
            if expr_ir.op == "+":
                return left + right
            if expr_ir.op == "-":
                return left - right
            if expr_ir.op == "*":
                return left * right
            if expr_ir.op == "/":
                return None if right == 0.0 else left / right
            if expr_ir.op == "%":
                return None if right == 0.0 else float(int(left) % int(right))
            if expr_ir.op == "<<":
                return float(int(left) << int(right))
            if expr_ir.op == ">>":
                return float(int(left) >> int(right))
            if expr_ir.op == "&":
                return float(int(left) & int(right))
            if expr_ir.op == "|":
                return float(int(left) | int(right))
            if expr_ir.op == "^":
                return float(int(left) ^ int(right))
            if expr_ir.op == ">":
                return 1.0 if left > right else 0.0
            if expr_ir.op == "<":
                return 1.0 if left < right else 0.0
            if expr_ir.op == ">=":
                return 1.0 if left >= right else 0.0
            if expr_ir.op == "<=":
                return 1.0 if left <= right else 0.0
            if expr_ir.op == "==":
                return 1.0 if left == right else 0.0
            if expr_ir.op == "!=":
                return 1.0 if left != right else 0.0
            if expr_ir.op == "&&":
                return 1.0 if left and right else 0.0
            if expr_ir.op == "||":
                return 1.0 if left or right else 0.0
        except (OverflowError, ValueError):
            return None
        return None

    if isinstance(expr_ir, TernaryExprIR):
        cond = _static_numeric_expr_value(expr_ir.cond, env)
        if cond is None:
            return None
        branch = expr_ir.true_expr if cond else expr_ir.false_expr
        return _static_numeric_expr_value(branch, env)

    return None


def _stmt_writes_name(stmt_ir: StmtIR, name: str) -> bool:
    if isinstance(stmt_ir, AssignmentIR):
        if isinstance(stmt_ir.target, IdentifierIR):
            return stmt_ir.target.name == name
        return False
    if isinstance(stmt_ir, BlockIR):
        return any(_stmt_writes_name(child, name) for child in stmt_ir.statements)
    if isinstance(stmt_ir, IfStatementIR):
        return _stmt_writes_name(stmt_ir.then_body, name) or (
            stmt_ir.else_body is not None and _stmt_writes_name(stmt_ir.else_body, name)
        )
    if isinstance(stmt_ir, ForStatementIR):
        return (
            _stmt_writes_name(stmt_ir.init, name)
            or _stmt_writes_name(stmt_ir.update, name)
            or _stmt_writes_name(stmt_ir.body, name)
        )
    if isinstance(stmt_ir, WhileStatementIR):
        return _stmt_writes_name(stmt_ir.body, name)
    if isinstance(stmt_ir, CaseStatementIR):
        return any(_stmt_writes_name(item.body, name) for item in stmt_ir.items)
    return False


def _substitute_assignment_ir(
    stmt_ir: AssignmentIR,
    env: dict[str, int],
) -> Optional[AssignmentIR]:
    target = _substitute_assignment_target(stmt_ir.target, env)
    value = _substitute_expr_ir(stmt_ir.value, env)
    if target is None or value is None:
        return None
    return AssignmentIR(target=target, value=value)


def _substitute_stmt_ir(stmt_ir: StmtIR, env: dict[str, int]) -> Optional[StmtIR]:
    if isinstance(stmt_ir, AssignmentIR):
        return _substitute_assignment_ir(stmt_ir, env)
    if isinstance(stmt_ir, ContributionIR):
        branch = _substitute_expr_ir(stmt_ir.branch, env)
        expr = _substitute_expr_ir(stmt_ir.expr, env)
        if not isinstance(branch, BranchAccessIR) or expr is None:
            return None
        return ContributionIR(branch=branch, expr=expr)
    if isinstance(stmt_ir, BlockIR):
        statements = []
        for child in stmt_ir.statements:
            substituted = _substitute_stmt_ir(child, env)
            if substituted is None:
                return None
            statements.append(substituted)
        return BlockIR(tuple(statements))
    if isinstance(stmt_ir, IfStatementIR):
        cond = _substitute_expr_ir(stmt_ir.cond, env)
        then_body = _substitute_stmt_ir(stmt_ir.then_body, env)
        else_body = (
            None
            if stmt_ir.else_body is None
            else _substitute_stmt_ir(stmt_ir.else_body, env)
        )
        if cond is None or then_body is None:
            return None
        if stmt_ir.else_body is not None and else_body is None:
            return None
        return IfStatementIR(cond=cond, then_body=then_body, else_body=else_body)
    if isinstance(stmt_ir, ForStatementIR):
        init = _substitute_assignment_ir(stmt_ir.init, env)
        cond = _substitute_expr_ir(stmt_ir.cond, env)
        update = _substitute_assignment_ir(stmt_ir.update, env)
        body = _substitute_stmt_ir(stmt_ir.body, env)
        if init is None or cond is None or update is None or body is None:
            return None
        return ForStatementIR(init=init, cond=cond, update=update, body=body)
    if isinstance(stmt_ir, SystemTaskIR):
        args = []
        for arg in stmt_ir.args:
            substituted = _substitute_expr_ir(arg, env)
            if substituted is None:
                return None
            args.append(substituted)
        return SystemTaskIR(name=stmt_ir.name, args=tuple(args))
    return None


def _substitute_assignment_target(
    target: Union[IdentifierIR, ArrayAccessIR],
    env: dict[str, int],
) -> Optional[Union[IdentifierIR, ArrayAccessIR]]:
    if isinstance(target, IdentifierIR):
        return target
    if isinstance(target, ArrayAccessIR):
        index = _substitute_expr_ir(target.index, env)
        if index is None:
            return None
        return ArrayAccessIR(target.name, index)
    return None


def _substitute_expr_ir(expr_ir: ExprIR, env: dict[str, int]) -> Optional[ExprIR]:
    if isinstance(expr_ir, IdentifierIR) and expr_ir.name in env:
        return LiteralIR(float(env[expr_ir.name]))
    if isinstance(expr_ir, (LiteralIR, IdentifierIR)):
        return expr_ir
    if isinstance(expr_ir, ArrayAccessIR):
        index = _substitute_expr_ir(expr_ir.index, env)
        if index is None:
            return None
        return ArrayAccessIR(expr_ir.name, index)
    if isinstance(expr_ir, BinaryExprIR):
        left = _substitute_expr_ir(expr_ir.left, env)
        right = _substitute_expr_ir(expr_ir.right, env)
        if left is None or right is None:
            return None
        return BinaryExprIR(expr_ir.op, left, right)
    if isinstance(expr_ir, UnaryExprIR):
        operand = _substitute_expr_ir(expr_ir.operand, env)
        if operand is None:
            return None
        return UnaryExprIR(expr_ir.op, operand)
    if isinstance(expr_ir, TernaryExprIR):
        cond = _substitute_expr_ir(expr_ir.cond, env)
        true_expr = _substitute_expr_ir(expr_ir.true_expr, env)
        false_expr = _substitute_expr_ir(expr_ir.false_expr, env)
        if cond is None or true_expr is None or false_expr is None:
            return None
        return TernaryExprIR(cond, true_expr, false_expr)
    if isinstance(expr_ir, FunctionCallIR):
        args = []
        for arg in expr_ir.args:
            substituted = _substitute_expr_ir(arg, env)
            if substituted is None:
                return None
            args.append(substituted)
        return FunctionCallIR(expr_ir.name, tuple(args))
    if isinstance(expr_ir, BranchAccessIR):
        children = []
        for child in (
            expr_ir.node1_index,
            expr_ir.node2_index,
            expr_ir.node1_index2,
            expr_ir.node2_index2,
        ):
            if child is None:
                children.append(None)
                continue
            substituted = _substitute_expr_ir(child, env)
            if substituted is None:
                return None
            children.append(substituted)
        return BranchAccessIR(
            access_type=expr_ir.access_type,
            node1=expr_ir.node1,
            node2=expr_ir.node2,
            node1_index=children[0],
            node2_index=children[1],
            node1_index2=children[2],
            node2_index2=children[3],
        )
    if isinstance(expr_ir, MethodCallIR):
        args = []
        for arg in expr_ir.args:
            substituted = _substitute_expr_ir(arg, env)
            if substituted is None:
                return None
            args.append(substituted)
        return MethodCallIR(expr_ir.obj, expr_ir.method, tuple(args))
    return None


def _append_body_write(
    target_kind: int,
    target_id: int,
    target_integer: bool,
    expr_ir: ExprIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
    stmt_ops: list[BodyStmtOp],
    expr_ops: list[BodyExprOp],
) -> bool:
    encoded_expr = encode_body_expr_ops(expr_ir, bindings, node_slots)
    if encoded_expr is None:
        return False
    expr_start = len(expr_ops)
    expr_ops.extend(encoded_expr)
    stmt_ops.append(
        BodyStmtOp(
            target_kind=target_kind,
            target_id=target_id,
            expr_start=expr_start,
            expr_count=len(encoded_expr),
            target_integer=target_integer,
        )
    )
    return True


def _encode_assignment_target(
    target: Union[IdentifierIR, ArrayAccessIR],
    bindings: BindingTableIR,
) -> Optional[tuple[int, int, bool]]:
    if isinstance(target, IdentifierIR):
        binding = bindings.resolve(target.name)
    elif isinstance(target, ArrayAccessIR):
        binding = resolve_static_array_element_binding(target, bindings)
    else:
        binding = None
    if binding is None or binding.kind != SYMBOL_STATE_SCALAR:
        return None
    return (BODY_TARGET_STATE, binding.slot, binding.integer)


def _assignment_current_expr(
    target: Union[IdentifierIR, ArrayAccessIR],
    bindings: BindingTableIR,
) -> Optional[ExprIR]:
    if isinstance(target, IdentifierIR):
        return target
    if isinstance(target, ArrayAccessIR):
        if resolve_static_array_element_binding(target, bindings) is None:
            return None
        return target
    return None


def _assignment_target_name(
    target: Union[IdentifierIR, ArrayAccessIR],
    bindings: BindingTableIR,
) -> Optional[str]:
    if isinstance(target, IdentifierIR):
        return target.name
    if isinstance(target, ArrayAccessIR):
        return static_array_element_name(target, bindings)
    return None


def _encode_contribution_target(
    branch: BranchAccessIR,
    node_slots: dict[str, int],
) -> Optional[int]:
    if branch.access_type != "V":
        return None
    node_name = static_node_ref_name(
        branch.node1,
        branch.node1_index,
        branch.node1_index2,
    )
    if node_name is None:
        return None
    return node_slots.get(node_name)


def _emit_stmt_lines(stmt_ir: StmtIR, lines: list[str], indent: str, level: int) -> None:
    prefix = indent * level

    if isinstance(stmt_ir, AssignmentIR):
        lines.append(f"{prefix}{_emit_assignment(stmt_ir)}")
        return

    if isinstance(stmt_ir, ContributionIR):
        lines.append(
            f"{prefix}contribute({_emit_branch_target(stmt_ir.branch)}, "
            f"{emit_python(stmt_ir.expr)})"
        )
        return

    if isinstance(stmt_ir, EventStatementIR):
        lines.append(f"{prefix}if event_due({emit_event_python(stmt_ir.event)}):")
        _emit_body(stmt_ir.body, lines, indent, level + 1)
        return

    if isinstance(stmt_ir, BlockIR):
        _emit_body(stmt_ir, lines, indent, level)
        return

    if isinstance(stmt_ir, IfStatementIR):
        lines.append(f"{prefix}if {emit_python(stmt_ir.cond)}:")
        _emit_body(stmt_ir.then_body, lines, indent, level + 1)
        if stmt_ir.else_body is not None:
            lines.append(f"{prefix}else:")
            _emit_body(stmt_ir.else_body, lines, indent, level + 1)
        return

    if isinstance(stmt_ir, ForStatementIR):
        lines.append(f"{prefix}{_emit_assignment(stmt_ir.init)}")
        lines.append(f"{prefix}while {emit_python(stmt_ir.cond)}:")
        _emit_body(stmt_ir.body, lines, indent, level + 1)
        lines.append(f"{indent * (level + 1)}{_emit_assignment(stmt_ir.update)}")
        return

    if isinstance(stmt_ir, WhileStatementIR):
        lines.append(f"{prefix}while {emit_python(stmt_ir.cond)}:")
        _emit_body(stmt_ir.body, lines, indent, level + 1)
        return

    if isinstance(stmt_ir, CaseStatementIR):
        selector = "_case_value"
        lines.append(f"{prefix}{selector} = {emit_python(stmt_ir.expr)}")
        first = True
        for item in stmt_ir.items:
            if item.values:
                values = ", ".join(emit_python(value) for value in item.values)
                keyword = "if" if first else "elif"
                lines.append(f"{prefix}{keyword} {selector} in ({values},):")
            else:
                keyword = "if" if first else "else"
                lines.append(f"{prefix}{keyword} True:")
            _emit_body(item.body, lines, indent, level + 1)
            first = False
        if not stmt_ir.items:
            lines.append(f"{prefix}pass")
        return

    if isinstance(stmt_ir, SystemTaskIR):
        args = ", ".join(emit_python(arg) for arg in stmt_ir.args)
        lines.append(f"{prefix}system_task({stmt_ir.name!r}, {args})")
        return

    raise TypeError(f"unsupported StmtIR node: {stmt_ir!r}")


def _emit_body(stmt_ir: StmtIR, lines: list[str], indent: str, level: int) -> None:
    before = len(lines)
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            _emit_stmt_lines(child, lines, indent, level)
    else:
        _emit_stmt_lines(stmt_ir, lines, indent, level)
    if len(lines) == before:
        lines.append(f"{indent * level}pass")


def _emit_assignment(stmt_ir: AssignmentIR) -> str:
    if isinstance(stmt_ir.target, IdentifierIR):
        return f"set_var({stmt_ir.target.name!r}, {emit_python(stmt_ir.value)})"
    if isinstance(stmt_ir.target, ArrayAccessIR):
        return (
            f"set_array({stmt_ir.target.name!r}, "
            f"int({emit_python(stmt_ir.target.index)}), {emit_python(stmt_ir.value)})"
        )
    raise TypeError(f"unsupported assignment target: {stmt_ir.target!r}")


def _emit_branch_target(branch: BranchAccessIR) -> str:
    return (
        f"branch_target({branch.access_type!r}, {branch.node1!r}, "
        f"{branch.node2!r})"
    )
