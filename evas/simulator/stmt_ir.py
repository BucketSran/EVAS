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
    BranchAccess,
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
    ArrayAccessIR,
    BindingTableIR,
    BranchAccessIR,
    ExprIR,
    IdentifierIR,
    LoweringContext,
    SYMBOL_STATE_SCALAR,
    TernaryExprIR,
    emit_python,
    encode_body_expr_ops,
    iter_identifier_names,
    lower_expr,
)
from evas.simulator.rust_backend import (
    BODY_TARGET_NODE,
    BODY_TARGET_STATE,
    BodyExprOp,
    BodyStmtOp,
)
from evas.simulator.schedule_ir import EventIR, emit_event_python, lower_event


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
    if not _append_body_stmt_ops(stmt_ir, bindings, node_slots, stmt_ops, expr_ops):
        return None
    return BodyStmtProgram(tuple(stmt_ops), tuple(expr_ops))


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
) -> bool:
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            if not _append_body_stmt_ops(child, bindings, node_slots, stmt_ops, expr_ops):
                return False
        return True

    if isinstance(stmt_ir, AssignmentIR):
        target = _encode_assignment_target(stmt_ir.target, bindings)
        if target is None:
            return False
        target_kind, target_id, target_integer = target
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
        return _append_body_write(
            BODY_TARGET_NODE,
            target_id,
            False,
            stmt_ir.expr,
            bindings,
            node_slots,
            stmt_ops,
            expr_ops,
        )

    if isinstance(stmt_ir, IfStatementIR):
        writes = _collect_conditional_write_specs(stmt_ir, bindings, node_slots)
        if writes is None:
            return False
        for write in writes:
            (
                _target_key,
                target_kind,
                target_id,
                target_integer,
                _current_expr,
                value_expr,
                _target_name,
            ) = write
            if not _append_body_write(
                target_kind,
                target_id,
                target_integer,
                value_expr,
                bindings,
                node_slots,
                stmt_ops,
                expr_ops,
            ):
                return False
        return True

    return False


def _collect_conditional_write_specs(
    stmt_ir: IfStatementIR,
    bindings: BindingTableIR,
    node_slots: dict[str, int],
):
    cond_refs = set(iter_identifier_names(stmt_ir.cond))
    then_writes = _collect_body_write_specs(stmt_ir.then_body, bindings, node_slots)
    if then_writes is None or not then_writes:
        return None

    # We lower control flow by reusing the existing expression-level SELECT
    # operator.  That means the condition is evaluated for each target write.
    # Keep this conservative: reject multi-write branches where the condition
    # reads a state that the branch itself writes, because the second SELECT
    # could observe a mutated condition.
    written_names = {
        target_name
        for (
            _target_key,
            _target_kind,
            _target_id,
            _target_integer,
            _current_expr,
            _value_expr,
            target_name,
        ) in then_writes
        if target_name is not None
    }
    if len(then_writes) > 1 and cond_refs & written_names:
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
    if else_writes is None or len(then_writes) != len(else_writes):
        return None

    merged = []
    for then_write, else_write in zip(then_writes, else_writes):
        if then_write[0] != else_write[0]:
            return None
        (
            target_key,
            target_kind,
            target_id,
            target_integer,
            current_expr,
            then_expr,
            target_name,
        ) = then_write
        (
            _else_target_key,
            else_target_kind,
            else_target_id,
            else_target_integer,
            _else_current_expr,
            else_expr,
            _else_target_name,
        ) = else_write
        if (
            target_kind != else_target_kind
            or target_id != else_target_id
            or target_integer != else_target_integer
        ):
            return None
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
        if not isinstance(stmt_ir.target, IdentifierIR):
            return None
        target_key = (target_kind, target_id)
        return (
            (
                target_key,
                target_kind,
                target_id,
                target_integer,
                stmt_ir.target,
                stmt_ir.value,
                stmt_ir.target.name,
            ),
        )

    if isinstance(stmt_ir, ContributionIR):
        target_id = _encode_contribution_target(stmt_ir.branch, node_slots)
        if target_id is None:
            return None
        target_key = (BODY_TARGET_NODE, target_id)
        return (
            (
                target_key,
                BODY_TARGET_NODE,
                target_id,
                False,
                stmt_ir.branch,
                stmt_ir.expr,
                None,
            ),
        )

    if isinstance(stmt_ir, IfStatementIR):
        return _collect_conditional_write_specs(stmt_ir, bindings, node_slots)

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
    if not isinstance(target, IdentifierIR):
        return None
    binding = bindings.resolve(target.name)
    if binding is None or binding.kind != SYMBOL_STATE_SCALAR:
        return None
    return (BODY_TARGET_STATE, binding.slot, binding.integer)


def _encode_contribution_target(
    branch: BranchAccessIR,
    node_slots: dict[str, int],
) -> Optional[int]:
    if branch.access_type != "V":
        return None
    if branch.node2 is not None:
        return None
    if any(
        child is not None
        for child in (
            branch.node1_index,
            branch.node1_index2,
            branch.node2_index,
            branch.node2_index2,
        )
    ):
        return None
    return node_slots.get(branch.node1)


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
