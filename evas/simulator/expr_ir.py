"""General expression IR for Verilog-A body lowering.

This module is the audit-094a bridge between the parser AST and future
statement/event Rust executors.  It intentionally does not change simulator
production behavior.  The first consumer is round-trip validation:

    Verilog-A AST expression -> ExprIR -> Python expression string

The IR keeps source-level structure instead of collapsing everything into the
older static-linear sublanguage in :mod:`evas.simulator.evaluate_ir`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Tuple, Union

from evas.compiler.ast_nodes import (
    ArrayAccess,
    Assignment,
    BinaryExpr,
    Block,
    BranchAccess,
    CaseStatement,
    CombinedEvent,
    Contribution,
    EventExpr,
    EventStatement,
    Expr,
    ForStatement,
    FunctionCall,
    Identifier,
    IfStatement,
    MethodCall,
    NumberLiteral,
    ParamType,
    StringLiteral,
    SystemTask,
    TernaryExpr,
    UnaryExpr,
    WhileStatement,
)
from evas.simulator.rust_backend import (
    BODY_EXPR_ABS,
    BODY_EXPR_ADD,
    BODY_EXPR_BITAND,
    BODY_EXPR_BITOR,
    BODY_EXPR_BITXOR,
    BODY_EXPR_CEIL,
    BODY_EXPR_CONST,
    BODY_EXPR_COS,
    BODY_EXPR_DIV,
    BODY_EXPR_EQ,
    BODY_EXPR_EXP,
    BODY_EXPR_FLOOR,
    BODY_EXPR_GE,
    BODY_EXPR_GT,
    BODY_EXPR_LAND,
    BODY_EXPR_LE,
    BODY_EXPR_LN,
    BODY_EXPR_LOG10,
    BODY_EXPR_LOR,
    BODY_EXPR_LT,
    BODY_EXPR_MAX,
    BODY_EXPR_MIN,
    BODY_EXPR_MOD,
    BODY_EXPR_MUL,
    BODY_EXPR_NE,
    BODY_EXPR_NEG,
    BODY_EXPR_NOT,
    BODY_EXPR_POW,
    BODY_EXPR_READ_NODE,
    BODY_EXPR_READ_PARAM,
    BODY_EXPR_READ_STATE,
    BODY_EXPR_SELECT,
    BODY_EXPR_SIN,
    BODY_EXPR_SQRT,
    BODY_EXPR_SUB,
    BodyExprOp,
)


PURE_MATH_FUNCTIONS = frozenset(
    {
        "abs",
        "ceil",
        "cos",
        "exp",
        "floor",
        "ln",
        "log",
        "max",
        "min",
        "pow",
        "sin",
        "sqrt",
        "tan",
        "tanh",
    }
)

STATEFUL_ANALOG_FUNCTIONS = frozenset(
    {
        "cross",
        "idtmod",
        "last_crossing",
        "slew",
        "transition",
    }
)

SUPPORTED_SYSTEM_FUNCTIONS = frozenset(
    {
        "$dist_uniform",
        "$fopen",
        "$random",
        "$rdist_normal",
    }
)

SUPPORTED_METHODS = frozenset({"substr"})

_DOLLAR_MATH_ALIASES = frozenset(f"${name}" for name in PURE_MATH_FUNCTIONS)
SPECIAL_IDENTIFIER_NAMES = frozenset(
    {
        "$abstime",
        "$realtime",
        "$temperature",
        "$vt",
        "inf",
    }
)

SYMBOL_PARAMETER = "parameter"
SYMBOL_PORT = "port"
SYMBOL_SPECIAL = "special"
SYMBOL_STATE_ARRAY = "state_array"
SYMBOL_STATE_SCALAR = "state_scalar"

_BODY_BINARY_OPS = {
    "+": BODY_EXPR_ADD,
    "-": BODY_EXPR_SUB,
    "*": BODY_EXPR_MUL,
    "/": BODY_EXPR_DIV,
    "%": BODY_EXPR_MOD,
    ">": BODY_EXPR_GT,
    "<": BODY_EXPR_LT,
    ">=": BODY_EXPR_GE,
    "<=": BODY_EXPR_LE,
    "==": BODY_EXPR_EQ,
    "!=": BODY_EXPR_NE,
    "&&": BODY_EXPR_LAND,
    "||": BODY_EXPR_LOR,
    "&": BODY_EXPR_BITAND,
    "|": BODY_EXPR_BITOR,
    "^": BODY_EXPR_BITXOR,
}

_BODY_UNARY_OPS = {
    "-": BODY_EXPR_NEG,
    "!": BODY_EXPR_NOT,
}

_BODY_FUNCTION_OPS = {
    "abs": (BODY_EXPR_ABS, 1),
    "sqrt": (BODY_EXPR_SQRT, 1),
    "exp": (BODY_EXPR_EXP, 1),
    "ln": (BODY_EXPR_LN, 1),
    "log": (BODY_EXPR_LOG10, 1),
    "sin": (BODY_EXPR_SIN, 1),
    "cos": (BODY_EXPR_COS, 1),
    "floor": (BODY_EXPR_FLOOR, 1),
    "ceil": (BODY_EXPR_CEIL, 1),
    "min": (BODY_EXPR_MIN, 2),
    "max": (BODY_EXPR_MAX, 2),
    "pow": (BODY_EXPR_POW, 2),
}


@dataclass(frozen=True)
class LoweringContext:
    """Policy for expression lowering.

    The default context only admits pure math functions.  Broader contexts are
    used for body round-trip validation, where the IR must preserve expressions
    that later statement/event lowering will decide whether Rust can execute.
    """

    allowed_functions: frozenset[str] = PURE_MATH_FUNCTIONS
    allowed_system_functions: frozenset[str] = frozenset()
    allowed_methods: frozenset[str] = frozenset()
    allowed_branch_access_types: frozenset[str] = frozenset({"V", "I"})

    @classmethod
    def pure_math(cls) -> "LoweringContext":
        return cls()

    @classmethod
    def veriloga_body(cls) -> "LoweringContext":
        return cls(
            allowed_functions=PURE_MATH_FUNCTIONS | STATEFUL_ANALOG_FUNCTIONS,
            allowed_system_functions=SUPPORTED_SYSTEM_FUNCTIONS,
            allowed_methods=SUPPORTED_METHODS,
        )


@dataclass(frozen=True)
class LiteralIR:
    value: object
    raw: Optional[str] = None


@dataclass(frozen=True)
class IdentifierIR:
    name: str


@dataclass(frozen=True)
class ArrayAccessIR:
    name: str
    index: "ExprIR"


@dataclass(frozen=True)
class BinaryExprIR:
    op: str
    left: "ExprIR"
    right: "ExprIR"


@dataclass(frozen=True)
class UnaryExprIR:
    op: str
    operand: "ExprIR"


@dataclass(frozen=True)
class TernaryExprIR:
    cond: "ExprIR"
    true_expr: "ExprIR"
    false_expr: "ExprIR"


@dataclass(frozen=True)
class FunctionCallIR:
    name: str
    args: Tuple["ExprIR", ...]


@dataclass(frozen=True)
class BranchAccessIR:
    access_type: str
    node1: str
    node2: Optional[str] = None
    node1_index: Optional["ExprIR"] = None
    node2_index: Optional["ExprIR"] = None
    node1_index2: Optional["ExprIR"] = None
    node2_index2: Optional["ExprIR"] = None


@dataclass(frozen=True)
class MethodCallIR:
    obj: str
    method: str
    args: Tuple["ExprIR", ...]


@dataclass(frozen=True)
class StateBindingIR:
    """One source-level symbol bound to a stable typed-array slot."""

    name: str
    kind: str
    slot: int
    integer: bool = False
    lo: Optional[int] = None
    hi: Optional[int] = None


@dataclass(frozen=True)
class BindingTableIR:
    """Bindings for state, parameter, port, and special identifiers."""

    bindings: Tuple[StateBindingIR, ...]

    def resolve(self, name: str) -> Optional[StateBindingIR]:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        return None


ExprIR = Union[
    LiteralIR,
    IdentifierIR,
    ArrayAccessIR,
    BinaryExprIR,
    UnaryExprIR,
    TernaryExprIR,
    FunctionCallIR,
    BranchAccessIR,
    MethodCallIR,
]


def lower_expr(
    ast_expr: Expr,
    context: Optional[LoweringContext] = None,
) -> Optional[ExprIR]:
    """Lower a parser AST expression into ExprIR.

    Returns ``None`` for constructs outside the supplied policy.  This is a
    deliberate gate: later production lowerings can widen the context only when
    their runtime semantics are ready.
    """

    ctx = context or LoweringContext.pure_math()

    if isinstance(ast_expr, NumberLiteral):
        return LiteralIR(float(ast_expr.value), getattr(ast_expr, "raw", None))

    if isinstance(ast_expr, StringLiteral):
        return LiteralIR(str(ast_expr.value), None)

    if isinstance(ast_expr, Identifier):
        return IdentifierIR(str(ast_expr.name))

    if isinstance(ast_expr, ArrayAccess):
        index = lower_expr(ast_expr.index, ctx)
        if index is None:
            return None
        return ArrayAccessIR(str(ast_expr.name), index)

    if isinstance(ast_expr, BinaryExpr):
        left = lower_expr(ast_expr.left, ctx)
        right = lower_expr(ast_expr.right, ctx)
        if left is None or right is None:
            return None
        return BinaryExprIR(str(ast_expr.op), left, right)

    if isinstance(ast_expr, UnaryExpr):
        operand = lower_expr(ast_expr.operand, ctx)
        if operand is None:
            return None
        return UnaryExprIR(str(ast_expr.op), operand)

    if isinstance(ast_expr, TernaryExpr):
        cond = lower_expr(ast_expr.cond, ctx)
        true_expr = lower_expr(ast_expr.true_expr, ctx)
        false_expr = lower_expr(ast_expr.false_expr, ctx)
        if cond is None or true_expr is None or false_expr is None:
            return None
        return TernaryExprIR(cond, true_expr, false_expr)

    if isinstance(ast_expr, FunctionCall):
        name = _normalize_function_name(str(ast_expr.name))
        if name not in ctx.allowed_functions:
            if str(ast_expr.name) not in ctx.allowed_system_functions:
                return None
            name = str(ast_expr.name)
        args = _lower_expr_tuple(ast_expr.args, ctx)
        if args is None:
            return None
        return FunctionCallIR(name, args)

    if isinstance(ast_expr, BranchAccess):
        access_type = str(ast_expr.access_type)
        if access_type not in ctx.allowed_branch_access_types:
            return None
        node1_index = _lower_optional_expr(ast_expr.node1_index, ctx)
        node1_index2 = _lower_optional_expr(ast_expr.node1_index2, ctx)
        node2_index = _lower_optional_expr(ast_expr.node2_index, ctx)
        node2_index2 = _lower_optional_expr(ast_expr.node2_index2, ctx)
        if _any_missing(node1_index, ast_expr.node1_index):
            return None
        if _any_missing(node1_index2, ast_expr.node1_index2):
            return None
        if _any_missing(node2_index, ast_expr.node2_index):
            return None
        if _any_missing(node2_index2, ast_expr.node2_index2):
            return None
        return BranchAccessIR(
            access_type=access_type,
            node1=str(ast_expr.node1),
            node2=str(ast_expr.node2) if ast_expr.node2 is not None else None,
            node1_index=node1_index,
            node2_index=node2_index,
            node1_index2=node1_index2,
            node2_index2=node2_index2,
        )

    if isinstance(ast_expr, MethodCall):
        method = str(ast_expr.method)
        if method not in ctx.allowed_methods:
            return None
        args = _lower_expr_tuple(ast_expr.args, ctx)
        if args is None:
            return None
        return MethodCallIR(str(ast_expr.obj), method, args)

    return None


def emit_python(expr_ir: ExprIR) -> str:
    """Emit a valid Python expression for round-trip validation."""

    if isinstance(expr_ir, LiteralIR):
        if isinstance(expr_ir.value, float):
            if expr_ir.value == float("inf"):
                return "float('inf')"
            if expr_ir.value == float("-inf"):
                return "float('-inf')"
            if expr_ir.value != expr_ir.value:
                return "float('nan')"
        return repr(expr_ir.value)

    if isinstance(expr_ir, IdentifierIR):
        return _emit_identifier(expr_ir.name)

    if isinstance(expr_ir, ArrayAccessIR):
        return f"array_value({expr_ir.name!r}, int({emit_python(expr_ir.index)}))"

    if isinstance(expr_ir, BinaryExprIR):
        left = emit_python(expr_ir.left)
        right = emit_python(expr_ir.right)
        op = expr_ir.op
        if op == "&&":
            return f"(({left}) and ({right}))"
        if op == "||":
            return f"(({left}) or ({right}))"
        if op == "^":
            return f"(int({left}) ^ int({right}))"
        if op == "&":
            return f"(int({left}) & int({right}))"
        if op == "|":
            return f"(int({left}) | int({right}))"
        if op == "<<":
            return f"(int({left}) << int({right}))"
        if op == ">>":
            return f"(int({left}) >> int({right}))"
        return f"(({left}) {op} ({right}))"

    if isinstance(expr_ir, UnaryExprIR):
        operand = emit_python(expr_ir.operand)
        if expr_ir.op == "!":
            return f"(not ({operand}))"
        if expr_ir.op == "~":
            return f"(~int({operand}))"
        return f"({expr_ir.op}({operand}))"

    if isinstance(expr_ir, TernaryExprIR):
        cond = emit_python(expr_ir.cond)
        true_expr = emit_python(expr_ir.true_expr)
        false_expr = emit_python(expr_ir.false_expr)
        return f"(({true_expr}) if ({cond}) else ({false_expr}))"

    if isinstance(expr_ir, FunctionCallIR):
        return _emit_function_call(expr_ir)

    if isinstance(expr_ir, BranchAccessIR):
        return _emit_branch_access(expr_ir)

    if isinstance(expr_ir, MethodCallIR):
        args = ", ".join(emit_python(arg) for arg in expr_ir.args)
        if expr_ir.method == "substr":
            return f"method_substr({expr_ir.obj!r}, {args})"
        return f"method_call({expr_ir.obj!r}, {expr_ir.method!r}, {args})"

    raise TypeError(f"unsupported ExprIR node: {expr_ir!r}")


def iter_exprs_from_statement(stmt: object) -> Iterator[Expr]:
    """Yield expressions contained in a statement/event tree."""

    if stmt is None:
        return

    if isinstance(stmt, Block):
        for child in stmt.statements:
            yield from iter_exprs_from_statement(child)
        return

    if isinstance(stmt, Assignment):
        yield stmt.target
        yield stmt.value
        return

    if isinstance(stmt, Contribution):
        yield stmt.branch
        yield stmt.expr
        return

    if isinstance(stmt, EventStatement):
        yield from iter_exprs_from_event(stmt.event)
        yield from iter_exprs_from_statement(stmt.body)
        return

    if isinstance(stmt, IfStatement):
        yield stmt.cond
        yield from iter_exprs_from_statement(stmt.then_body)
        yield from iter_exprs_from_statement(stmt.else_body)
        return

    if isinstance(stmt, ForStatement):
        yield from iter_exprs_from_statement(stmt.init)
        yield stmt.cond
        yield from iter_exprs_from_statement(stmt.update)
        yield from iter_exprs_from_statement(stmt.body)
        return

    if isinstance(stmt, WhileStatement):
        yield stmt.cond
        yield from iter_exprs_from_statement(stmt.body)
        return

    if isinstance(stmt, CaseStatement):
        yield stmt.expr
        for item in stmt.items:
            for value in item.values:
                yield value
            yield from iter_exprs_from_statement(item.body)
        return

    if isinstance(stmt, SystemTask):
        for arg in stmt.args:
            yield arg


def iter_exprs_from_event(event: object) -> Iterator[Expr]:
    if isinstance(event, CombinedEvent):
        for child in event.events:
            yield from iter_exprs_from_event(child)
        return

    if isinstance(event, EventExpr):
        for arg in event.args:
            yield arg
        if event.time_tol_expr is not None:
            yield event.time_tol_expr
        if event.expr_tol_expr is not None:
            yield event.expr_tol_expr


def build_state_binding_ir(module: object) -> BindingTableIR:
    """Build stable symbol bindings from a parsed module.

    The binding is representation-only.  Later Rust ABI work can choose which
    binding kinds become arrays and which remain Python fallback metadata.
    """

    bindings: list[StateBindingIR] = []

    for slot, name in enumerate(sorted(SPECIAL_IDENTIFIER_NAMES)):
        bindings.append(StateBindingIR(name=name, kind=SYMBOL_SPECIAL, slot=slot))

    for slot, param in enumerate(getattr(module, "parameters", ()) or ()):
        integer = getattr(param, "param_type", None) == ParamType.INTEGER
        bindings.append(
            StateBindingIR(
                name=str(param.name),
                kind=SYMBOL_PARAMETER,
                slot=slot,
                integer=integer,
            )
        )

    for slot, name in enumerate(getattr(module, "ports", ()) or ()):
        bindings.append(StateBindingIR(name=str(name), kind=SYMBOL_PORT, slot=slot))

    scalar_slot = 0
    array_slot = 0
    for variable in getattr(module, "variables", ()) or ():
        integer = getattr(variable, "var_type", None) == ParamType.INTEGER
        if getattr(variable, "is_array", False):
            bindings.append(
                StateBindingIR(
                    name=str(variable.name),
                    kind=SYMBOL_STATE_ARRAY,
                    slot=array_slot,
                    integer=integer,
                    lo=getattr(variable, "array_lo", None),
                    hi=getattr(variable, "array_hi", None),
                )
            )
            array_slot += 1
        else:
            bindings.append(
                StateBindingIR(
                    name=str(variable.name),
                    kind=SYMBOL_STATE_SCALAR,
                    slot=scalar_slot,
                    integer=integer,
                )
            )
            scalar_slot += 1

    return BindingTableIR(tuple(bindings))


def iter_identifier_names(expr_ir: ExprIR) -> Iterator[str]:
    """Yield source identifier names referenced by an ExprIR tree."""

    if isinstance(expr_ir, IdentifierIR):
        yield expr_ir.name
        return

    if isinstance(expr_ir, ArrayAccessIR):
        yield expr_ir.name
        yield from iter_identifier_names(expr_ir.index)
        return

    if isinstance(expr_ir, BinaryExprIR):
        yield from iter_identifier_names(expr_ir.left)
        yield from iter_identifier_names(expr_ir.right)
        return

    if isinstance(expr_ir, UnaryExprIR):
        yield from iter_identifier_names(expr_ir.operand)
        return

    if isinstance(expr_ir, TernaryExprIR):
        yield from iter_identifier_names(expr_ir.cond)
        yield from iter_identifier_names(expr_ir.true_expr)
        yield from iter_identifier_names(expr_ir.false_expr)
        return

    if isinstance(expr_ir, FunctionCallIR):
        for arg in expr_ir.args:
            yield from iter_identifier_names(arg)
        return

    if isinstance(expr_ir, BranchAccessIR):
        for child in (
            expr_ir.node1_index,
            expr_ir.node1_index2,
            expr_ir.node2_index,
            expr_ir.node2_index2,
        ):
            if child is not None:
                yield from iter_identifier_names(child)
        return

    if isinstance(expr_ir, MethodCallIR):
        yield expr_ir.obj
        for arg in expr_ir.args:
            yield from iter_identifier_names(arg)


def encode_body_expr_ops(
    expr_ir: ExprIR,
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
) -> Optional[Tuple[BodyExprOp, ...]]:
    """Encode ExprIR into the 094e Rust body stack-machine op stream.

    The encoder is intentionally conservative.  It only accepts scalar
    parameters, scalar states, and statically resolved voltage node reads.  More
    complex language features still return ``None`` so production callers can
    fall back to the Python evaluator without changing semantics.
    """

    ops: list[BodyExprOp] = []
    if not _append_body_expr_ops(expr_ir, bindings, node_slots, ops):
        return None
    return tuple(ops)


def _append_body_expr_ops(
    expr_ir: ExprIR,
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
    ops: list[BodyExprOp],
) -> bool:
    if isinstance(expr_ir, LiteralIR):
        if not isinstance(expr_ir.value, (int, float)):
            return False
        ops.append(BodyExprOp(BODY_EXPR_CONST, value=float(expr_ir.value)))
        return True

    if isinstance(expr_ir, IdentifierIR):
        if expr_ir.name == "inf":
            ops.append(BodyExprOp(BODY_EXPR_CONST, value=float("inf")))
            return True
        binding = bindings.resolve(expr_ir.name)
        if binding is None:
            return False
        if binding.kind == SYMBOL_PARAMETER:
            ops.append(BodyExprOp(BODY_EXPR_READ_PARAM, index=binding.slot))
            return True
        if binding.kind == SYMBOL_STATE_SCALAR:
            ops.append(BodyExprOp(BODY_EXPR_READ_STATE, index=binding.slot))
            return True
        if binding.kind == SYMBOL_PORT and expr_ir.name in node_slots:
            ops.append(BodyExprOp(BODY_EXPR_READ_NODE, index=node_slots[expr_ir.name]))
            return True
        return False

    if isinstance(expr_ir, BranchAccessIR):
        return _append_branch_body_expr_ops(expr_ir, bindings, node_slots, ops)

    if isinstance(expr_ir, BinaryExprIR):
        op_kind = _BODY_BINARY_OPS.get(expr_ir.op)
        if op_kind is None:
            return False
        if not _append_body_expr_ops(expr_ir.left, bindings, node_slots, ops):
            return False
        if not _append_body_expr_ops(expr_ir.right, bindings, node_slots, ops):
            return False
        ops.append(BodyExprOp(op_kind))
        return True

    if isinstance(expr_ir, UnaryExprIR):
        if expr_ir.op == "+":
            return _append_body_expr_ops(expr_ir.operand, bindings, node_slots, ops)
        op_kind = _BODY_UNARY_OPS.get(expr_ir.op)
        if op_kind is None:
            return False
        if not _append_body_expr_ops(expr_ir.operand, bindings, node_slots, ops):
            return False
        ops.append(BodyExprOp(op_kind))
        return True

    if isinstance(expr_ir, TernaryExprIR):
        if not _append_body_expr_ops(expr_ir.cond, bindings, node_slots, ops):
            return False
        if not _append_body_expr_ops(expr_ir.true_expr, bindings, node_slots, ops):
            return False
        if not _append_body_expr_ops(expr_ir.false_expr, bindings, node_slots, ops):
            return False
        ops.append(BodyExprOp(BODY_EXPR_SELECT))
        return True

    if isinstance(expr_ir, FunctionCallIR):
        op_info = _BODY_FUNCTION_OPS.get(expr_ir.name)
        if op_info is None:
            return False
        op_kind, arity = op_info
        if len(expr_ir.args) != arity:
            return False
        for arg in expr_ir.args:
            if not _append_body_expr_ops(arg, bindings, node_slots, ops):
                return False
        ops.append(BodyExprOp(op_kind))
        return True

    return False


def _append_branch_body_expr_ops(
    expr_ir: BranchAccessIR,
    bindings: BindingTableIR,
    node_slots: Mapping[str, int],
    ops: list[BodyExprOp],
) -> bool:
    if expr_ir.access_type != "V":
        return False
    if any(
        child is not None
        for child in (
            expr_ir.node1_index,
            expr_ir.node1_index2,
            expr_ir.node2_index,
            expr_ir.node2_index2,
        )
    ):
        return False
    node1_slot = node_slots.get(expr_ir.node1)
    if node1_slot is None:
        return False
    ops.append(BodyExprOp(BODY_EXPR_READ_NODE, index=node1_slot))
    if expr_ir.node2 is None:
        return True
    node2_slot = node_slots.get(expr_ir.node2)
    if node2_slot is None:
        return False
    ops.append(BodyExprOp(BODY_EXPR_READ_NODE, index=node2_slot))
    ops.append(BodyExprOp(BODY_EXPR_SUB))
    return True


def _lower_expr_tuple(
    exprs: Iterable[Expr],
    context: LoweringContext,
) -> Optional[Tuple[ExprIR, ...]]:
    lowered = []
    for expr in exprs:
        item = lower_expr(expr, context)
        if item is None:
            return None
        lowered.append(item)
    return tuple(lowered)


def _lower_optional_expr(
    expr: Optional[Expr],
    context: LoweringContext,
) -> Optional[ExprIR]:
    if expr is None:
        return None
    return lower_expr(expr, context)


def _any_missing(lowered: Optional[ExprIR], original: Optional[Expr]) -> bool:
    return original is not None and lowered is None


def _normalize_function_name(name: str) -> str:
    if name in _DOLLAR_MATH_ALIASES:
        return name[1:]
    return name


def _emit_identifier(name: str) -> str:
    if name == "inf":
        return "float('inf')"
    if name in {"$abstime", "$realtime"}:
        return "time_value"
    if name == "$temperature":
        return "(temperature_c + 273.15)"
    if name == "$vt":
        return "(1.380649e-23 * (temperature_c + 273.15) / 1.602176634e-19)"
    return f"var({name!r})"


def _emit_function_call(expr_ir: FunctionCallIR) -> str:
    args = ", ".join(emit_python(arg) for arg in expr_ir.args)
    name = expr_ir.name
    if name == "ln":
        return f"math.log({args})"
    if name == "log":
        return f"math.log10({args})"
    if name in {"exp", "sqrt", "sin", "cos", "tan", "tanh", "floor", "ceil"}:
        return f"math.{name}({args})"
    if name in {"abs", "pow", "min", "max"}:
        return f"{name}({args})"
    if name.startswith("$"):
        helper = "fn_" + name[1:].replace("$", "").replace("-", "_")
        return f"{helper}({args})"
    return f"fn_{name}({args})"


def _emit_branch_access(expr_ir: BranchAccessIR) -> str:
    n1 = _emit_node_ref(expr_ir.node1, expr_ir.node1_index, expr_ir.node1_index2)
    if expr_ir.access_type == "I":
        if expr_ir.node2 is None:
            return f"current({n1})"
        n2 = _emit_node_ref(expr_ir.node2, expr_ir.node2_index, expr_ir.node2_index2)
        return f"current({n1}, {n2})"
    if expr_ir.node2 is None:
        return f"voltage({n1})"
    n2 = _emit_node_ref(expr_ir.node2, expr_ir.node2_index, expr_ir.node2_index2)
    return f"(voltage({n1}) - voltage({n2}))"


def _emit_node_ref(
    name: str,
    index1: Optional[ExprIR],
    index2: Optional[ExprIR],
) -> str:
    if index1 is None:
        return repr(name)
    idx1 = f"int({emit_python(index1)})"
    if index2 is None:
        return f"node_ref({name!r}, {idx1})"
    idx2 = f"int({emit_python(index2)})"
    return f"node_ref({name!r}, {idx1}, {idx2})"
