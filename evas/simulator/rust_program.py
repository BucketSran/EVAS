"""Typed EVAS2 Rust simulation program schema.

This module is intentionally schema-first: Python lowers supported EVAS
semantics into a typed program, and Rust owns the transient loop for programs
that are fully lowered.  Unsupported features are reported explicitly instead
of silently falling back when strict EVAS2 is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

from evas.simulator.evaluate_ir import (
    SOURCE_NODE,
    SOURCE_STATE,
    TARGET_NODE,
    TARGET_STATE,
    normalize_linear_ops,
)
from evas.simulator.expr_ir import (
    SYMBOL_PORT,
    SYMBOL_STATE_ARRAY,
    SYMBOL_STATE_SCALAR,
    ArrayAccessIR,
    BinaryExprIR,
    BindingTableIR,
    BranchAccessIR,
    ExprIR,
    FunctionCallIR,
    IdentifierIR,
    LiteralIR,
    StateBindingIR,
    TernaryExprIR,
    UnaryExprIR,
    build_state_binding_ir,
    static_node_ref_name,
)
from evas.simulator.rust_backend import (
    BODY_EXPR_READ_NODE,
    BODY_EXPR_READ_PARAM,
    BODY_EXPR_READ_STATE,
    BODY_STMT_FILE_CLOSE,
    BODY_STMT_FILE_OPEN,
    BODY_STMT_FILE_WRITE,
    BODY_STMT_STROBE,
    BODY_TARGET_NODE,
    BODY_TARGET_STATE,
    BodyExprOp,
    BodyStmtOp,
)
from evas.simulator.schedule_ir import (
    EVENT_DUE_ABOVE,
    EVENT_DUE_CROSS,
    CombinedEventIR,
    EventIR,
    EventTriggerIR,
    encode_event_due_program,
)
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
    encode_body_stmt_ops,
    lower_stmt,
    unroll_static_for_statement,
)
from evas.simulator.transition_runtime import encode_transition_contribution_program

SOURCE_DC = "dc"
SOURCE_PULSE = "pulse"
SOURCE_SINE = "sine"
SOURCE_PWL = "pwl"
EVENT_DUE_ALWAYS = "always"
EVENT_PHASE_PRE = 0
EVENT_PHASE_POST = 1


@dataclass(frozen=True)
class RustSimNode:
    """A voltage-domain node slot owned by the Rust loop."""

    name: str
    node_id: int
    initial_value: float = 0.0


@dataclass(frozen=True)
class RustSimState:
    """A model state slot owned by the Rust loop."""

    name: str
    state_id: int
    initial_value: float = 0.0
    is_integer: bool = False


@dataclass(frozen=True)
class RustSimParam:
    """A model parameter slot used by Rust evaluate/event opcodes."""

    name: str
    param_id: int
    value: float


@dataclass(frozen=True)
class RustSimSource:
    """A source opcode with fixed node ownership and typed waveform payload."""

    node: str
    node_id: int
    kind: str
    params: Tuple[float, ...]
    data_start: int = 0
    data_count: int = 0
    flags: int = 0


@dataclass(frozen=True)
class RustSimEvent:
    """An event detector entry such as initial_step, timer, cross, or above."""

    kind: str
    event_id: int
    phase: int = EVENT_PHASE_PRE
    direction: int = 0
    expr_start: int = 0
    expr_count: int = 0
    time_tol_start: int = 0
    time_tol_count: int = 0
    expr_tol_start: int = 0
    expr_tol_count: int = 0
    timer_start_expr_start: int = 0
    timer_start_expr_count: int = 0
    timer_period_expr_start: int = 0
    timer_period_expr_count: int = 0
    body_stmt_start: int = 0
    body_stmt_count: int = 0


@dataclass(frozen=True)
class RustSimBodyOp:
    """A lowered event/evaluate body operation."""

    op_kind: str
    target_kind: str
    target_id: int
    args: Tuple[float, ...] = ()
    source_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RustSimLinearTerm:
    """One linear source term in a continuous evaluate opcode."""

    source_kind: int
    source_id: int
    gain: float


@dataclass(frozen=True)
class RustSimLinearCondition:
    """A conditional select for a continuous linear write."""

    op_kind: int
    left_bias: float
    left_terms: Tuple[RustSimLinearTerm, ...]
    right_bias: float
    right_terms: Tuple[RustSimLinearTerm, ...]


@dataclass(frozen=True)
class RustSimLinearOp:
    """A continuous evaluate write executed by the Rust loop."""

    target_kind: int
    target_id: int
    bias: float
    terms: Tuple[RustSimLinearTerm, ...]
    condition: Optional[RustSimLinearCondition] = None
    false_bias: float = 0.0
    false_terms: Tuple[RustSimLinearTerm, ...] = ()
    target_integer: bool = False


@dataclass(frozen=True)
class RustSimTransition:
    """A transition() state slot and its output target."""

    transition_id: int
    output_node_id: int
    reference_node_id: Optional[int]
    target_expr_start: int
    target_expr_count: int
    delay_expr_start: int
    delay_expr_count: int
    rise_expr_start: int
    rise_expr_count: int
    fall_expr_start: int
    fall_expr_count: int
    output_bias_expr_start: int = 0
    output_bias_expr_count: int = 0
    output_scale_expr_start: int = 0
    output_scale_expr_count: int = 0
    default_transition: float = 1.0e-12


@dataclass(frozen=True)
class RustSimRecord:
    """A recorded waveform column."""

    name: str
    node_id: int


@dataclass(frozen=True)
class RustSimSideEffect:
    """Python-owned metadata for Rust-triggered file side effects."""

    kind: str
    filename: str = ""
    mode: str = "w"
    fmt: str = ""
    owner: Any = None


@dataclass(frozen=True)
class RustSimProgram:
    """Complete typed program accepted by the strict Rust EVAS2 loop."""

    nodes: Tuple[RustSimNode, ...]
    states: Tuple[RustSimState, ...]
    params: Tuple[RustSimParam, ...]
    sources: Tuple[RustSimSource, ...]
    events: Tuple[RustSimEvent, ...]
    body_ops: Tuple[RustSimBodyOp, ...]
    transitions: Tuple[RustSimTransition, ...]
    records: Tuple[RustSimRecord, ...]
    side_effects: Tuple[RustSimSideEffect, ...] = ()
    continuous_linear_ops: Tuple[RustSimLinearOp, ...] = ()
    body_stmt_ops: Tuple[BodyStmtOp, ...] = ()
    body_expr_ops: Tuple[BodyExprOp, ...] = ()
    source_data: Tuple[float, ...] = ()
    bound_step_ops: Tuple[Any, ...] = ()
    final_step_ops: Tuple[RustSimBodyOp, ...] = ()

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def node_names(self) -> Tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def record_names(self) -> Tuple[str, ...]:
        return tuple(record.name for record in self.records)

    @property
    def record_node_ids(self) -> Tuple[int, ...]:
        return tuple(record.node_id for record in self.records)


@dataclass(frozen=True)
class RustSimCompileReport:
    """Result of lowering Python EVAS objects into a RustSimProgram."""

    program: Optional[RustSimProgram]
    supported: bool
    reasons: Tuple[str, ...] = ()


def _waveform_metadata(waveform: Any) -> Optional[Mapping[str, Any]]:
    meta = getattr(waveform, "_evas_waveform", None)
    return meta if isinstance(meta, Mapping) else None


def _add_node(name: str, node_ids: dict[str, int], nodes: list[RustSimNode]) -> int:
    if name not in node_ids:
        node_id = len(node_ids)
        node_ids[name] = node_id
        nodes.append(RustSimNode(name=name, node_id=node_id))
    return node_ids[name]


def _model_state_names(model: Any) -> Tuple[str, ...]:
    model_cls = getattr(model, "__class__", type(model))
    names = [str(name) for name in tuple(getattr(model_cls, "_state_scalar_names", ()) or ())]
    slot_name_fn = getattr(model, "_state_array_slot_name", None)
    for array_name, lo, hi, _integer in (
        tuple(getattr(model_cls, "_state_array_ranges", ()) or ())
    ):
        for idx in range(int(lo), int(hi) + 1):
            if slot_name_fn is not None:
                names.append(str(slot_name_fn(str(array_name), idx)))
            else:
                names.append(f"{array_name}[{idx}]")
    return tuple(names)


def _model_integer_state_names(model: Any) -> set[str]:
    model_cls = getattr(model, "__class__", type(model))
    names = set(getattr(model_cls, "_integer_state_names", ()) or ())
    slot_name_fn = getattr(model, "_state_array_slot_name", None)
    for array_name, lo, hi, integer in (
        tuple(getattr(model_cls, "_state_array_ranges", ()) or ())
    ):
        if not bool(integer):
            continue
        for idx in range(int(lo), int(hi) + 1):
            if slot_name_fn is not None:
                names.add(str(slot_name_fn(str(array_name), idx)))
            else:
                names.add(f"{array_name}[{idx}]")
    return names


def _state_array_slot_ref(model: Any, state_name: str) -> Optional[tuple[str, int]]:
    slot_ref_fn = getattr(model, "_state_array_slot_ref", None)
    if slot_ref_fn is not None:
        try:
            ref = slot_ref_fn(state_name)
        except Exception:
            ref = None
        if ref is not None:
            array_name, idx = ref
            return str(array_name), int(idx)
    if not isinstance(state_name, str) or "[" not in state_name or not state_name.endswith("]"):
        return None
    array_name, raw_idx = state_name[:-1].rsplit("[", 1)
    try:
        return str(array_name), int(raw_idx)
    except ValueError:
        return None


def _model_state_value(model: Any, state_name: str) -> float:
    state_values = getattr(model, "state", {}) or {}
    if state_name in state_values:
        return float(state_values.get(state_name, 0.0))
    array_ref = _state_array_slot_ref(model, state_name)
    if array_ref is not None:
        array_name, idx = array_ref
        return float((getattr(model, "arrays", {}) or {}).get(array_name, {}).get(idx, 0.0))
    return 0.0


def _binding_array_slot_name(name: str, idx: int) -> str:
    return f"{name}[{int(idx)}]"


def _static_int_expr_value(expr_ir: ExprIR) -> Optional[int]:
    if isinstance(expr_ir, LiteralIR):
        if not isinstance(expr_ir.value, (int, float)):
            return None
        idx = int(expr_ir.value)
        if float(expr_ir.value) != float(idx):
            return None
        return idx
    if isinstance(expr_ir, UnaryExprIR):
        value = _static_int_expr_value(expr_ir.operand)
        if value is None:
            return None
        if expr_ir.op == "+":
            return value
        if expr_ir.op == "-":
            return -value
    return None


def _iter_static_state_array_accesses_expr(expr_ir: ExprIR):
    if isinstance(expr_ir, ArrayAccessIR):
        idx = _static_int_expr_value(expr_ir.index)
        if idx is not None:
            yield str(expr_ir.name), idx
        yield from _iter_static_state_array_accesses_expr(expr_ir.index)
        return
    if isinstance(expr_ir, BranchAccessIR):
        for child in (
            expr_ir.node1_index,
            expr_ir.node1_index2,
            expr_ir.node2_index,
            expr_ir.node2_index2,
        ):
            if child is not None:
                yield from _iter_static_state_array_accesses_expr(child)
        return
    if isinstance(expr_ir, BinaryExprIR):
        yield from _iter_static_state_array_accesses_expr(expr_ir.left)
        yield from _iter_static_state_array_accesses_expr(expr_ir.right)
        return
    if isinstance(expr_ir, UnaryExprIR):
        yield from _iter_static_state_array_accesses_expr(expr_ir.operand)
        return
    if isinstance(expr_ir, TernaryExprIR):
        yield from _iter_static_state_array_accesses_expr(expr_ir.cond)
        yield from _iter_static_state_array_accesses_expr(expr_ir.true_expr)
        yield from _iter_static_state_array_accesses_expr(expr_ir.false_expr)
        return
    if isinstance(expr_ir, FunctionCallIR):
        for arg in expr_ir.args:
            yield from _iter_static_state_array_accesses_expr(arg)


def _iter_static_state_array_accesses_event(event_ir: EventIR):
    if isinstance(event_ir, CombinedEventIR):
        for child in event_ir.events:
            yield from _iter_static_state_array_accesses_event(child)
        return
    if not isinstance(event_ir, EventTriggerIR):
        return
    for expr in event_ir.args:
        yield from _iter_static_state_array_accesses_expr(expr)
    for expr in (event_ir.time_tol, event_ir.expr_tol):
        if expr is not None:
            yield from _iter_static_state_array_accesses_expr(expr)


def _iter_static_state_array_accesses_stmt(stmt_ir: object):
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            yield from _iter_static_state_array_accesses_stmt(child)
        return
    if isinstance(stmt_ir, AssignmentIR):
        if isinstance(stmt_ir.target, ArrayAccessIR):
            idx = _static_int_expr_value(stmt_ir.target.index)
            if idx is not None:
                yield str(stmt_ir.target.name), idx
            yield from _iter_static_state_array_accesses_expr(stmt_ir.target.index)
        yield from _iter_static_state_array_accesses_expr(stmt_ir.value)
        return
    if isinstance(stmt_ir, ContributionIR):
        yield from _iter_static_state_array_accesses_expr(stmt_ir.branch)
        yield from _iter_static_state_array_accesses_expr(stmt_ir.expr)
        return
    if isinstance(stmt_ir, EventStatementIR):
        yield from _iter_static_state_array_accesses_event(stmt_ir.event)
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.body)
        return
    if isinstance(stmt_ir, IfStatementIR):
        yield from _iter_static_state_array_accesses_expr(stmt_ir.cond)
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.then_body)
        if stmt_ir.else_body is not None:
            yield from _iter_static_state_array_accesses_stmt(stmt_ir.else_body)
        return
    if isinstance(stmt_ir, ForStatementIR):
        unrolled = unroll_static_for_statement(stmt_ir)
        if unrolled is not None:
            yield from _iter_static_state_array_accesses_stmt(unrolled)
            return
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.init)
        yield from _iter_static_state_array_accesses_expr(stmt_ir.cond)
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.update)
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.body)
        return
    if isinstance(stmt_ir, WhileStatementIR):
        yield from _iter_static_state_array_accesses_expr(stmt_ir.cond)
        yield from _iter_static_state_array_accesses_stmt(stmt_ir.body)
        return
    if isinstance(stmt_ir, CaseStatementIR):
        yield from _iter_static_state_array_accesses_expr(stmt_ir.expr)
        for item in stmt_ir.items:
            for value in item.values:
                yield from _iter_static_state_array_accesses_expr(value)
            yield from _iter_static_state_array_accesses_stmt(item.body)
        return
    if isinstance(stmt_ir, SystemTaskIR):
        for arg in stmt_ir.args:
            yield from _iter_static_state_array_accesses_expr(arg)


def _extend_bindings_from_static_array_accesses(
    bindings: BindingTableIR,
    stmt_ir: object,
) -> BindingTableIR:
    accesses: dict[str, set[int]] = {}
    for name, idx in _iter_static_state_array_accesses_stmt(stmt_ir):
        accesses.setdefault(str(name), set()).add(int(idx))
    if not accesses:
        return bindings

    by_name = {binding.name: binding for binding in bindings.bindings}
    next_scalar_slot = (
        max(
            (
                int(binding.slot)
                for binding in bindings.bindings
                if binding.kind == SYMBOL_STATE_SCALAR
            ),
            default=-1,
        )
        + 1
    )
    rewritten: list[StateBindingIR] = []
    for binding in bindings.bindings:
        if binding.kind == SYMBOL_STATE_ARRAY and binding.name in accesses:
            indices = set(accesses[binding.name])
            if binding.lo is not None:
                indices.add(int(binding.lo))
            if binding.hi is not None:
                indices.add(int(binding.hi))
            binding = StateBindingIR(
                name=binding.name,
                kind=binding.kind,
                slot=binding.slot,
                integer=binding.integer,
                lo=min(indices),
                hi=max(indices),
            )
        rewritten.append(binding)

    present = {binding.name for binding in rewritten}
    for name in sorted(accesses):
        array_binding = by_name.get(name)
        if array_binding is None or array_binding.kind != SYMBOL_STATE_ARRAY:
            continue
        for idx in sorted(accesses[name]):
            slot_name = _binding_array_slot_name(name, idx)
            if slot_name in present:
                continue
            rewritten.append(
                StateBindingIR(
                    name=slot_name,
                    kind=SYMBOL_STATE_SCALAR,
                    slot=next_scalar_slot,
                    integer=array_binding.integer,
                    lo=idx,
                    hi=idx,
                )
            )
            present.add(slot_name)
            next_scalar_slot += 1
    return BindingTableIR(tuple(rewritten))


def _model_params(model: Any) -> Tuple[tuple[int, str, float], ...]:
    module = getattr(getattr(model, "__class__", type(model)), "_module_ast", None)
    params = getattr(module, "parameters", ()) or ()
    values = getattr(model, "params", {}) or {}
    result: list[tuple[int, str, float]] = []
    for local_slot, param in enumerate(params):
        name = str(getattr(param, "name", ""))
        if not name:
            continue
        try:
            value = float(values.get(name, 0.0))
        except (TypeError, ValueError):
            continue
        result.append((local_slot, name, value))
    return tuple(result)


class _RustSimSideEffectBuilder:
    def __init__(self, model: Any):
        self._model = model
        self.effects: list[RustSimSideEffect] = []

    def resolve_string(self, expr_ir: ExprIR) -> Optional[str]:
        if isinstance(expr_ir, LiteralIR) and isinstance(expr_ir.value, str):
            return str(expr_ir.value)
        if isinstance(expr_ir, IdentifierIR):
            value = (getattr(self._model, "params", {}) or {}).get(expr_ir.name)
            if isinstance(value, str):
                return value
        return None

    def add_file_open(self, filename: str, mode: str) -> int:
        return self._append(RustSimSideEffect(kind="fopen", filename=filename, mode=mode))

    def add_file_write(self, fmt: str) -> int:
        return self._append(RustSimSideEffect(kind="fwrite", fmt=fmt))

    def add_file_close(self) -> int:
        return self._append(RustSimSideEffect(kind="fclose"))

    def add_strobe(self, fmt: str) -> int:
        return self._append(RustSimSideEffect(kind="strobe", fmt=fmt, owner=self._model))

    def _append(self, effect: RustSimSideEffect) -> int:
        self.effects.append(effect)
        return len(self.effects) - 1


def _external_node(model: Any, local_name: str) -> str:
    node_map = getattr(model, "node_map", {}) or {}
    if local_name in node_map:
        return str(node_map[local_name])
    local_folded = str(local_name).casefold()
    for key, value in node_map.items():
        if str(key).casefold() == local_folded:
            return str(value)
    return str(local_name)


def _scalar(model: Any, value: Any) -> tuple[Optional[float], Optional[str]]:
    scalar_eval = getattr(model, "_evaluate_rust_static_affine_scalar", None)
    if scalar_eval is not None:
        try:
            return float(scalar_eval(value, getattr(model, "params", {}) or {})), None
        except Exception as exc:
            return None, f"scalar_eval_failed:{exc}"
    try:
        return float(value), None
    except Exception:
        return None, f"non_constant_scalar:{value!r}"


def _reject_model_dynamic_semantics(model: Any, model_index: int) -> Tuple[str, ...]:
    model_cls = getattr(model, "__class__", type(model))
    prefix = f"model:{model_index}:{getattr(model_cls, '__name__', 'unknown')}"
    reasons: list[str] = []
    if getattr(model, "_child_models", None):
        reasons.append(f"{prefix}:child_models_not_lowered")
    has_dynamic = getattr(model, "_has_dynamic_breakpoints_tree", None)
    if has_dynamic is not None and bool(has_dynamic()):
        reasons.append(f"{prefix}:event_breakpoints_not_lowered")
    if bool(getattr(model_cls, "_has_post_update_events", True)):
        reasons.append(f"{prefix}:post_update_not_lowered")
    if tuple(getattr(model_cls, "_event_static_linear_ir_ops", ()) or ()):
        reasons.append(f"{prefix}:event_body_not_lowered")
    if tuple(getattr(model_cls, "_event_timer_static_linear_ir_ops", ()) or ()):
        reasons.append(f"{prefix}:timer_event_not_lowered")
    if tuple(getattr(model_cls, "_transition_target_ir_ops", ()) or ()):
        reasons.append(f"{prefix}:transition_not_lowered")
    return tuple(reasons)


def _ensure_model_state_slots(
    *,
    model: Any,
    model_index: int,
    state_ids: dict[tuple[int, str], int],
    states: list[RustSimState],
    bindings: Optional[BindingTableIR] = None,
) -> dict[int, int]:
    local_to_global: dict[int, int] = {}
    if bindings is not None:
        state_bindings = tuple(
            sorted(
                (
                    binding
                    for binding in bindings.bindings
                    if binding.kind == SYMBOL_STATE_SCALAR
                ),
                key=lambda binding: int(binding.slot),
            )
        )
        state_items = tuple(
            (int(binding.slot), str(binding.name), bool(binding.integer))
            for binding in state_bindings
        )
    else:
        integer_names = _model_integer_state_names(model)
        state_items = tuple(
            (local_id, state_name, state_name in integer_names)
            for local_id, state_name in enumerate(_model_state_names(model))
        )
    for local_id, state_name, is_integer in state_items:
        key = (model_index, state_name)
        if key not in state_ids:
            state_id = len(state_ids)
            state_ids[key] = state_id
            states.append(
                RustSimState(
                    name=f"{model_index}:{state_name}",
                    state_id=state_id,
                    initial_value=_model_state_value(model, state_name),
                    is_integer=is_integer,
                )
            )
        local_to_global[local_id] = state_ids[key]
    return local_to_global


def _ensure_model_param_slots(
    *,
    model: Any,
    model_index: int,
    param_ids: dict[tuple[int, str], int],
    params: list[RustSimParam],
) -> dict[int, int]:
    local_to_global: dict[int, int] = {}
    for local_id, param_name, value in _model_params(model):
        key = (model_index, param_name)
        if key not in param_ids:
            param_id = len(param_ids)
            param_ids[key] = param_id
            params.append(
                RustSimParam(
                    name=f"{model_index}:{param_name}",
                    param_id=param_id,
                    value=float(value),
                )
            )
        local_to_global[local_id] = param_ids[key]
    return local_to_global


def _node_slot_maps(
    *,
    model: Any,
    bindings: BindingTableIR,
    node_ids: dict[str, int],
    nodes: list[RustSimNode],
) -> tuple[dict[str, int], dict[int, int]]:
    local_node_slots: dict[str, int] = {}
    local_to_global: dict[int, int] = {}
    for binding in bindings.bindings:
        if binding.kind != SYMBOL_PORT:
            continue
        external = _external_node(model, binding.name)
        global_id = _add_node(external, node_ids, nodes)
        local_slot = int(binding.slot)
        local_node_slots[binding.name] = local_slot
        local_to_global[local_slot] = global_id
    return local_node_slots, local_to_global


def _iter_static_branch_node_names(expr_ir: ExprIR):
    if isinstance(expr_ir, BranchAccessIR):
        node1_name = static_node_ref_name(
            expr_ir.node1,
            expr_ir.node1_index,
            expr_ir.node1_index2,
        )
        if node1_name is not None:
            yield node1_name
        if expr_ir.node2 is not None:
            node2_name = static_node_ref_name(
                expr_ir.node2,
                expr_ir.node2_index,
                expr_ir.node2_index2,
            )
            if node2_name is not None:
                yield node2_name
        for child in (
            expr_ir.node1_index,
            expr_ir.node1_index2,
            expr_ir.node2_index,
            expr_ir.node2_index2,
        ):
            if child is not None:
                yield from _iter_static_branch_node_names(child)
        return
    if isinstance(expr_ir, ArrayAccessIR):
        yield from _iter_static_branch_node_names(expr_ir.index)
        return
    if isinstance(expr_ir, BinaryExprIR):
        yield from _iter_static_branch_node_names(expr_ir.left)
        yield from _iter_static_branch_node_names(expr_ir.right)
        return
    if isinstance(expr_ir, UnaryExprIR):
        yield from _iter_static_branch_node_names(expr_ir.operand)
        return
    if isinstance(expr_ir, TernaryExprIR):
        yield from _iter_static_branch_node_names(expr_ir.cond)
        yield from _iter_static_branch_node_names(expr_ir.true_expr)
        yield from _iter_static_branch_node_names(expr_ir.false_expr)
        return
    if isinstance(expr_ir, FunctionCallIR):
        for arg in expr_ir.args:
            yield from _iter_static_branch_node_names(arg)


def _iter_static_branch_node_names_from_stmt(stmt_ir: object):
    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            yield from _iter_static_branch_node_names_from_stmt(child)
        return
    if isinstance(stmt_ir, AssignmentIR):
        yield from _iter_static_branch_node_names(stmt_ir.value)
        if isinstance(stmt_ir.target, ArrayAccessIR):
            yield from _iter_static_branch_node_names(stmt_ir.target.index)
        return
    if isinstance(stmt_ir, ContributionIR):
        yield from _iter_static_branch_node_names(stmt_ir.branch)
        yield from _iter_static_branch_node_names(stmt_ir.expr)
        return
    if isinstance(stmt_ir, EventStatementIR):
        yield from _iter_static_branch_node_names_from_event(stmt_ir.event)
        yield from _iter_static_branch_node_names_from_stmt(stmt_ir.body)
        return
    if isinstance(stmt_ir, IfStatementIR):
        yield from _iter_static_branch_node_names(stmt_ir.cond)
        yield from _iter_static_branch_node_names_from_stmt(stmt_ir.then_body)
        if stmt_ir.else_body is not None:
            yield from _iter_static_branch_node_names_from_stmt(stmt_ir.else_body)
        return
    if isinstance(stmt_ir, (ForStatementIR, WhileStatementIR)):
        if isinstance(stmt_ir, ForStatementIR):
            unrolled = unroll_static_for_statement(stmt_ir)
            if unrolled is not None:
                yield from _iter_static_branch_node_names_from_stmt(unrolled)
                return
            yield from _iter_static_branch_node_names_from_stmt(stmt_ir.init)
            yield from _iter_static_branch_node_names(stmt_ir.cond)
            yield from _iter_static_branch_node_names_from_stmt(stmt_ir.update)
        else:
            yield from _iter_static_branch_node_names(stmt_ir.cond)
        yield from _iter_static_branch_node_names_from_stmt(stmt_ir.body)
        return
    if isinstance(stmt_ir, CaseStatementIR):
        yield from _iter_static_branch_node_names(stmt_ir.expr)
        for item in stmt_ir.items:
            for value in item.values:
                yield from _iter_static_branch_node_names(value)
            yield from _iter_static_branch_node_names_from_stmt(item.body)


def _iter_static_branch_node_names_from_event(event_ir: EventIR):
    if isinstance(event_ir, CombinedEventIR):
        for child in event_ir.events:
            yield from _iter_static_branch_node_names_from_event(child)
        return
    if not isinstance(event_ir, EventTriggerIR):
        return
    for expr in event_ir.args:
        yield from _iter_static_branch_node_names(expr)
    for expr in (event_ir.time_tol, event_ir.expr_tol):
        if expr is not None:
            yield from _iter_static_branch_node_names(expr)


def _extend_node_slots_from_static_branches(
    *,
    model: Any,
    stmt_ir: object,
    node_ids: dict[str, int],
    nodes: list[RustSimNode],
    local_node_slots: dict[str, int],
    node_slot_to_global: dict[int, int],
) -> None:
    next_slot = max(local_node_slots.values(), default=-1) + 1
    for local_name in sorted(set(_iter_static_branch_node_names_from_stmt(stmt_ir))):
        if local_name in local_node_slots:
            continue
        local_node_slots[local_name] = next_slot
        node_slot_to_global[next_slot] = _add_node(
            _external_node(model, local_name),
            node_ids,
            nodes,
        )
        next_slot += 1


def _remap_body_expr_ops(
    expr_ops: Tuple[BodyExprOp, ...],
    *,
    node_slot_to_global: Mapping[int, int],
    state_slot_to_global: Mapping[int, int],
    param_slot_to_global: Mapping[int, int],
) -> Tuple[BodyExprOp, ...]:
    remapped: list[BodyExprOp] = []
    for op in expr_ops:
        index = int(op.index)
        if op.op_kind == BODY_EXPR_READ_NODE:
            index = int(node_slot_to_global.get(index, index))
        elif op.op_kind == BODY_EXPR_READ_STATE:
            index = int(state_slot_to_global.get(index, index))
        elif op.op_kind == BODY_EXPR_READ_PARAM:
            index = int(param_slot_to_global.get(index, index))
        remapped.append(
            BodyExprOp(
                op_kind=int(op.op_kind),
                index=index,
                value=float(op.value),
            )
        )
    return tuple(remapped)


def _append_expr_segment(
    expr_ops: list[BodyExprOp],
    segment: Tuple[BodyExprOp, ...],
    *,
    node_slot_to_global: Mapping[int, int],
    state_slot_to_global: Mapping[int, int],
    param_slot_to_global: Mapping[int, int],
) -> tuple[int, int]:
    start = len(expr_ops)
    remapped = _remap_body_expr_ops(
        segment,
        node_slot_to_global=node_slot_to_global,
        state_slot_to_global=state_slot_to_global,
        param_slot_to_global=param_slot_to_global,
    )
    expr_ops.extend(remapped)
    return start, len(remapped)


def _append_body_program(
    body_stmt_ops: list[BodyStmtOp],
    body_expr_ops: list[BodyExprOp],
    program,
    *,
    node_slot_to_global: Mapping[int, int],
    state_slot_to_global: Mapping[int, int],
    param_slot_to_global: Mapping[int, int],
    side_effect_slot_offset: int = 0,
) -> tuple[int, int]:
    stmt_start = len(body_stmt_ops)
    for stmt in tuple(getattr(program, "stmt_ops", ()) or ()):
        target_id = int(stmt.target_id)
        if int(stmt.target_kind) == BODY_STMT_FILE_OPEN:
            expr_start = int(stmt.expr_start) + int(side_effect_slot_offset)
            expr_count = 0
            target_id = int(state_slot_to_global.get(target_id, target_id))
        else:
            expr_start, expr_count = _append_expr_segment(
                body_expr_ops,
                tuple(getattr(program, "expr_ops", ()) or ())[
                    int(stmt.expr_start) : int(stmt.expr_start) + int(stmt.expr_count)
                ],
                node_slot_to_global=node_slot_to_global,
                state_slot_to_global=state_slot_to_global,
                param_slot_to_global=param_slot_to_global,
            )
        if stmt.target_kind == BODY_TARGET_NODE:
            target_id = int(node_slot_to_global.get(target_id, target_id))
        elif stmt.target_kind == BODY_TARGET_STATE:
            target_id = int(state_slot_to_global.get(target_id, target_id))
        elif int(stmt.target_kind) in {
            BODY_STMT_FILE_WRITE,
            BODY_STMT_FILE_CLOSE,
            BODY_STMT_STROBE,
        }:
            target_id += int(side_effect_slot_offset)
        body_stmt_ops.append(
            BodyStmtOp(
                target_kind=int(stmt.target_kind),
                target_id=target_id,
                expr_start=expr_start,
                expr_count=expr_count,
                target_integer=bool(stmt.target_integer),
            )
        )
    return stmt_start, len(body_stmt_ops) - stmt_start


def _is_continuous_body_stmt(stmt_ir: object) -> bool:
    """Return True for ordered non-event state writes that Rust can own per step."""

    if isinstance(stmt_ir, AssignmentIR):
        return True
    if isinstance(stmt_ir, ContributionIR):
        return not _expr_contains_transition_call(stmt_ir.expr)
    if isinstance(stmt_ir, IfStatementIR):
        return _is_continuous_body_stmt(stmt_ir.then_body) and (
            stmt_ir.else_body is None or _is_continuous_body_stmt(stmt_ir.else_body)
        )
    if isinstance(stmt_ir, ForStatementIR):
        unrolled = unroll_static_for_statement(stmt_ir)
        return (
            unrolled is not None
            and not _stmt_has_rustsim_event_transition_candidate(unrolled)
            and _is_continuous_body_stmt(unrolled)
        )
    if isinstance(stmt_ir, CaseStatementIR):
        return all(_is_continuous_body_stmt(item.body) for item in stmt_ir.items)
    if isinstance(stmt_ir, SystemTaskIR):
        return stmt_ir.name in {"$bound_step", "$display", "$strobe"}
    if isinstance(stmt_ir, BlockIR):
        return all(_is_continuous_body_stmt(child) for child in stmt_ir.statements)
    return False


def _collect_contributed_nodes(stmt_ir: object) -> frozenset[str]:
    nodes: set[str] = set()

    if isinstance(stmt_ir, BlockIR):
        for child in stmt_ir.statements:
            nodes.update(_collect_contributed_nodes(child))
        return frozenset(nodes)

    if isinstance(stmt_ir, ContributionIR):
        if stmt_ir.branch.access_type == "V":
            # Only node1 is DRIVEN by a voltage contribution; node2 is the
            # reference (V(n1, n2) <+ x sets n1 = n2 + x). Counting node2 as
            # contributed wrongly classified every cross expression that
            # references a rail (e.g. V(in, VSS) with VSS used as the node2 of
            # output contributions) as post-phase, blocking pre-phase-only
            # features such as the cross-acceptance law mode for the common
            # ground-referenced benchmark style.
            node1_name = static_node_ref_name(
                stmt_ir.branch.node1,
                stmt_ir.branch.node1_index,
                stmt_ir.branch.node1_index2,
            )
            nodes.add(str(node1_name or stmt_ir.branch.node1))
        return frozenset(nodes)

    if isinstance(stmt_ir, EventStatementIR):
        return _collect_contributed_nodes(stmt_ir.body)

    if isinstance(stmt_ir, IfStatementIR):
        nodes.update(_collect_contributed_nodes(stmt_ir.then_body))
        if stmt_ir.else_body is not None:
            nodes.update(_collect_contributed_nodes(stmt_ir.else_body))
        return frozenset(nodes)

    if isinstance(stmt_ir, (ForStatementIR, WhileStatementIR)):
        return _collect_contributed_nodes(stmt_ir.body)

    if isinstance(stmt_ir, CaseStatementIR):
        for item in stmt_ir.items:
            nodes.update(_collect_contributed_nodes(item.body))
        return frozenset(nodes)

    return frozenset()


def _collect_global_contributed_nodes(models: Iterable[Any]) -> frozenset[str]:
    nodes: set[str] = set()
    for model in models:
        model_cls = getattr(model, "__class__", type(model))
        module = getattr(model_cls, "_module_ast", None)
        analog_block = (
            getattr(module, "analog_block", None) if module is not None else None
        )
        body_ast = getattr(analog_block, "body", None)
        if body_ast is None:
            continue
        try:
            body_ir = lower_stmt(body_ast)
        except Exception:
            continue
        for local_name in _collect_contributed_nodes(body_ir):
            nodes.add(_external_node(model, local_name))
    return frozenset(nodes)


def _expr_references_nodes(expr_ir: ExprIR, nodes: frozenset[str]) -> bool:
    if not nodes:
        return False
    if isinstance(expr_ir, BranchAccessIR):
        return (
            expr_ir.node1 in nodes
            or (expr_ir.node2 is not None and expr_ir.node2 in nodes)
            or (
                expr_ir.node1_index is not None
                and _expr_references_nodes(expr_ir.node1_index, nodes)
            )
            or (
                expr_ir.node1_index2 is not None
                and _expr_references_nodes(expr_ir.node1_index2, nodes)
            )
            or (
                expr_ir.node2_index is not None
                and _expr_references_nodes(expr_ir.node2_index, nodes)
            )
            or (
                expr_ir.node2_index2 is not None
                and _expr_references_nodes(expr_ir.node2_index2, nodes)
            )
        )
    if isinstance(expr_ir, ArrayAccessIR):
        return _expr_references_nodes(expr_ir.index, nodes)
    if isinstance(expr_ir, BinaryExprIR):
        return _expr_references_nodes(expr_ir.left, nodes) or _expr_references_nodes(
            expr_ir.right,
            nodes,
        )
    if isinstance(expr_ir, UnaryExprIR):
        return _expr_references_nodes(expr_ir.operand, nodes)
    if isinstance(expr_ir, TernaryExprIR):
        return (
            _expr_references_nodes(expr_ir.cond, nodes)
            or _expr_references_nodes(expr_ir.true_expr, nodes)
            or _expr_references_nodes(expr_ir.false_expr, nodes)
        )
    if isinstance(expr_ir, FunctionCallIR):
        return any(_expr_references_nodes(arg, nodes) for arg in expr_ir.args)
    return False


def _event_phase(event_ir: EventIR, contributed_nodes: frozenset[str]) -> int:
    if isinstance(event_ir, CombinedEventIR):
        if any(_event_phase(child, contributed_nodes) == EVENT_PHASE_POST for child in event_ir.events):
            return EVENT_PHASE_POST
        return EVENT_PHASE_PRE
    if not isinstance(event_ir, EventTriggerIR):
        return EVENT_PHASE_PRE
    event_type = str(event_ir.event_type).lower()
    if event_type in {EVENT_DUE_CROSS, EVENT_DUE_ABOVE} and event_ir.args:
        if _expr_references_nodes(event_ir.args[0], contributed_nodes):
            return EVENT_PHASE_POST
    return EVENT_PHASE_PRE


def _event_trigger_phases(
    event_ir: EventIR,
    contributed_nodes: frozenset[str],
) -> Tuple[int, ...]:
    if isinstance(event_ir, CombinedEventIR):
        phases: list[int] = []
        for child in event_ir.events:
            phases.extend(_event_trigger_phases(child, contributed_nodes))
        return tuple(phases)
    return (_event_phase(event_ir, contributed_nodes),)


def _prefer_existing_timer_static_linear_path(model_cls: Any) -> bool:
    """Keep stable periodic timer-static-linear models on their older Rust path."""

    if not tuple(getattr(model_cls, "_event_timer_static_linear_ir_ops", ()) or ()):
        return False
    if tuple(getattr(model_cls, "_transition_target_ir_ops", ()) or ()):
        return False
    return True


def _stmt_has_display_strobe(stmt_ir: object) -> bool:
    if isinstance(stmt_ir, BlockIR):
        return any(_stmt_has_display_strobe(child) for child in stmt_ir.statements)
    if isinstance(stmt_ir, EventStatementIR):
        return _stmt_has_display_strobe(stmt_ir.body)
    if isinstance(stmt_ir, IfStatementIR):
        return _stmt_has_display_strobe(stmt_ir.then_body) or (
            stmt_ir.else_body is not None and _stmt_has_display_strobe(stmt_ir.else_body)
        )
    if isinstance(stmt_ir, ForStatementIR):
        unrolled = unroll_static_for_statement(stmt_ir)
        if unrolled is not None:
            return _stmt_has_display_strobe(unrolled)
        return (
            _stmt_has_display_strobe(stmt_ir.init)
            or _stmt_has_display_strobe(stmt_ir.update)
            or _stmt_has_display_strobe(stmt_ir.body)
        )
    if isinstance(stmt_ir, WhileStatementIR):
        return _stmt_has_display_strobe(stmt_ir.body)
    if isinstance(stmt_ir, CaseStatementIR):
        return any(_stmt_has_display_strobe(item.body) for item in stmt_ir.items)
    if isinstance(stmt_ir, SystemTaskIR):
        return stmt_ir.name in {"$display", "$strobe"}
    return False


def _stmt_has_rustsim_event_transition_candidate(stmt_ir: object) -> bool:
    if isinstance(stmt_ir, BlockIR):
        return any(
            _stmt_has_rustsim_event_transition_candidate(child)
            for child in stmt_ir.statements
        )
    if isinstance(stmt_ir, EventStatementIR):
        return True
    if isinstance(stmt_ir, ContributionIR):
        return _expr_contains_transition_call(stmt_ir.expr)
    if isinstance(stmt_ir, IfStatementIR):
        return _stmt_has_rustsim_event_transition_candidate(stmt_ir.then_body) or (
            stmt_ir.else_body is not None
            and _stmt_has_rustsim_event_transition_candidate(stmt_ir.else_body)
        )
    if isinstance(stmt_ir, (ForStatementIR, WhileStatementIR)):
        return _stmt_has_rustsim_event_transition_candidate(stmt_ir.body)
    if isinstance(stmt_ir, CaseStatementIR):
        return any(
            _stmt_has_rustsim_event_transition_candidate(item.body)
            for item in stmt_ir.items
        )
    return False


def _expr_contains_transition_call(expr_ir: ExprIR) -> bool:
    if isinstance(expr_ir, FunctionCallIR):
        return str(expr_ir.name) == "transition" or any(
            _expr_contains_transition_call(arg) for arg in expr_ir.args
        )
    if isinstance(expr_ir, BinaryExprIR):
        return _expr_contains_transition_call(
            expr_ir.left
        ) or _expr_contains_transition_call(expr_ir.right)
    if isinstance(expr_ir, UnaryExprIR):
        return _expr_contains_transition_call(expr_ir.operand)
    if isinstance(expr_ir, TernaryExprIR):
        return (
            _expr_contains_transition_call(expr_ir.cond)
            or _expr_contains_transition_call(expr_ir.true_expr)
            or _expr_contains_transition_call(expr_ir.false_expr)
        )
    if isinstance(expr_ir, ArrayAccessIR):
        return _expr_contains_transition_call(expr_ir.index)
    if isinstance(expr_ir, BranchAccessIR):
        return any(
            child is not None and _expr_contains_transition_call(child)
            for child in (
                expr_ir.node1_index,
                expr_ir.node1_index2,
                expr_ir.node2_index,
                expr_ir.node2_index2,
            )
        )
    return False


def _model_has_rustsim_event_transition_candidate(model_cls: Any) -> bool:
    module = getattr(model_cls, "_module_ast", None)
    analog_block = getattr(module, "analog_block", None)
    body_ast = getattr(analog_block, "body", None)
    body_ir = lower_stmt(body_ast)
    if body_ir is None:
        return False
    return _stmt_has_rustsim_event_transition_candidate(body_ir)


def _model_has_rustsim_continuous_body_candidate(model_cls: Any) -> bool:
    module = getattr(model_cls, "_module_ast", None)
    analog_block = getattr(module, "analog_block", None)
    body_ast = getattr(analog_block, "body", None)
    body_ir = lower_stmt(body_ast)
    if not isinstance(body_ir, BlockIR):
        return False
    return any(_is_continuous_body_stmt(stmt) for stmt in body_ir.statements)


def _convert_continuous_linear_ops(
    *,
    model: Any,
    model_index: int,
    node_ids: dict[str, int],
    nodes: list[RustSimNode],
    state_ids: dict[tuple[int, str], int],
    states: list[RustSimState],
) -> tuple[Tuple[RustSimLinearOp, ...], Tuple[str, ...]]:
    model_cls = getattr(model, "__class__", type(model))
    raw_ops = tuple(
        getattr(model_cls, "_evaluate_ir_static_linear_non_event_ops", ()) or ()
    )
    if not raw_ops:
        raw_ops = tuple(getattr(model_cls, "_evaluate_ir_static_linear_ops", ()) or ())
    if not raw_ops:
        return (), (f"model:{model_index}:{model_cls.__name__}:no_continuous_linear_ir",)

    integer_names = _model_integer_state_names(model)
    for state_name in _model_state_names(model):
        key = (model_index, state_name)
        if key not in state_ids:
            state_id = len(state_ids)
            state_ids[key] = state_id
            states.append(
                RustSimState(
                    name=f"{model_index}:{state_name}",
                    state_id=state_id,
                    initial_value=_model_state_value(model, state_name),
                    is_integer=state_name in integer_names,
                )
            )

    try:
        ops = normalize_linear_ops(raw_ops)
    except Exception as exc:
        return (), (f"model:{model_index}:{model_cls.__name__}:linear_ir_decode_failed:{exc}",)

    reasons: list[str] = []

    def convert_terms(ir_terms) -> Tuple[RustSimLinearTerm, ...]:
        terms: list[RustSimLinearTerm] = []
        for term in ir_terms:
            gain, reason = _scalar(model, term.gain)
            if reason is not None or gain is None:
                reasons.append(
                    f"model:{model_index}:{model_cls.__name__}:term_gain:{reason}"
                )
                continue
            if term.source_kind == SOURCE_NODE:
                node_name = _external_node(model, term.source_name)
                source_id = _add_node(node_name, node_ids, nodes)
            elif term.source_kind == SOURCE_STATE:
                key = (model_index, str(term.source_name))
                if key not in state_ids:
                    state_id = len(state_ids)
                    state_ids[key] = state_id
                    states.append(
                        RustSimState(
                            name=f"{model_index}:{term.source_name}",
                            state_id=state_id,
                            initial_value=_model_state_value(
                                model,
                                str(term.source_name),
                            ),
                            is_integer=str(term.source_name) in integer_names,
                        )
                    )
                source_id = state_ids[key]
            else:
                reasons.append(
                    f"model:{model_index}:{model_cls.__name__}:unsupported_source_kind:{term.source_kind}"
                )
                continue
            terms.append(
                RustSimLinearTerm(
                    source_kind=int(term.source_kind),
                    source_id=source_id,
                    gain=gain,
                )
            )
        return tuple(terms)

    def convert_condition(condition) -> Optional[RustSimLinearCondition]:
        if condition is None:
            return None
        left_bias, reason = _scalar(model, condition.left_bias)
        if reason is not None or left_bias is None:
            reasons.append(
                f"model:{model_index}:{model_cls.__name__}:condition_left:{reason}"
            )
            return None
        right_bias, reason = _scalar(model, condition.right_bias)
        if reason is not None or right_bias is None:
            reasons.append(
                f"model:{model_index}:{model_cls.__name__}:condition_right:{reason}"
            )
            return None
        return RustSimLinearCondition(
            op_kind=int(condition.op_kind),
            left_bias=left_bias,
            left_terms=convert_terms(condition.left_terms),
            right_bias=right_bias,
            right_terms=convert_terms(condition.right_terms),
        )

    converted: list[RustSimLinearOp] = []
    for op in ops:
        bias, reason = _scalar(model, op.bias)
        if reason is not None or bias is None:
            reasons.append(f"model:{model_index}:{model_cls.__name__}:bias:{reason}")
            continue
        false_bias, reason = _scalar(model, op.false_bias)
        if reason is not None or false_bias is None:
            reasons.append(
                f"model:{model_index}:{model_cls.__name__}:false_bias:{reason}"
            )
            continue
        if op.target_kind == TARGET_NODE:
            target_id = _add_node(_external_node(model, op.target_name), node_ids, nodes)
        elif op.target_kind == TARGET_STATE:
            key = (model_index, str(op.target_name))
            if key not in state_ids:
                state_id = len(state_ids)
                state_ids[key] = state_id
                states.append(
                    RustSimState(
                        name=f"{model_index}:{op.target_name}",
                        state_id=state_id,
                        initial_value=_model_state_value(
                            model,
                            str(op.target_name),
                        ),
                        is_integer=bool(op.target_integer),
                    )
                )
            target_id = state_ids[key]
        else:
            reasons.append(
                f"model:{model_index}:{model_cls.__name__}:unsupported_target_kind:{op.target_kind}"
            )
            continue
        converted.append(
            RustSimLinearOp(
                target_kind=int(op.target_kind),
                target_id=target_id,
                bias=bias,
                terms=convert_terms(op.terms),
                condition=convert_condition(op.condition),
                false_bias=false_bias,
                false_terms=convert_terms(op.false_terms),
                target_integer=bool(op.target_integer),
            )
        )

    return tuple(converted), tuple(reasons)


def _convert_event_transition_ops(
    *,
    model: Any,
    model_index: int,
    global_contributed_nodes: frozenset[str],
    node_ids: dict[str, int],
    nodes: list[RustSimNode],
    state_ids: dict[tuple[int, str], int],
    states: list[RustSimState],
    param_ids: dict[tuple[int, str], int],
    params: list[RustSimParam],
    events: list[RustSimEvent],
    transitions: list[RustSimTransition],
    body_stmt_ops: list[BodyStmtOp],
    body_expr_ops: list[BodyExprOp],
    side_effects: list[RustSimSideEffect],
) -> Tuple[str, ...]:
    model_cls = getattr(model, "__class__", type(model))
    prefix = f"model:{model_index}:{getattr(model_cls, '__name__', 'unknown')}"
    module = getattr(model_cls, "_module_ast", None)
    if module is None:
        return (f"{prefix}:module_ast_unavailable",)
    analog_block = getattr(module, "analog_block", None)
    body_ast = getattr(analog_block, "body", None)
    body_ir = lower_stmt(body_ast)
    if not isinstance(body_ir, BlockIR):
        return (f"{prefix}:stmt_lower_failed",)
    if _prefer_existing_timer_static_linear_path(
        model_cls
    ) and not _stmt_has_display_strobe(body_ir):
        return (f"{prefix}:timer_static_linear_specialized_path_preferred",)
    if getattr(model, "_child_models", None):
        return (f"{prefix}:child_models_not_lowered",)
    bindings = _extend_bindings_from_static_array_accesses(
        build_state_binding_ir(module),
        body_ir,
    )
    local_node_slots, node_slot_to_global = _node_slot_maps(
        model=model,
        bindings=bindings,
        node_ids=node_ids,
        nodes=nodes,
    )
    _extend_node_slots_from_static_branches(
        model=model,
        stmt_ir=body_ir,
        node_ids=node_ids,
        nodes=nodes,
        local_node_slots=local_node_slots,
        node_slot_to_global=node_slot_to_global,
    )
    state_slot_to_global = _ensure_model_state_slots(
        model=model,
        model_index=model_index,
        state_ids=state_ids,
        states=states,
        bindings=bindings,
    )
    param_slot_to_global = _ensure_model_param_slots(
        model=model,
        model_index=model_index,
        param_ids=param_ids,
        params=params,
    )

    reasons: list[str] = []
    converted_events = 0
    converted_always_bodies = 0
    converted_transitions = 0
    pending_continuous: list[object] = []
    seen_transition = False
    contributed_nodes = _collect_contributed_nodes(body_ir)
    side_effect_slot_offset = len(side_effects)
    side_effect_builder = _RustSimSideEffectBuilder(model)
    phase_contributed_nodes = set(contributed_nodes)
    if global_contributed_nodes:
        for local_name in local_node_slots:
            if _external_node(model, local_name) in global_contributed_nodes:
                phase_contributed_nodes.add(local_name)
    phase_contributed_nodes = frozenset(phase_contributed_nodes)

    def flush_continuous_body(phase: int = EVENT_PHASE_PRE) -> None:
        nonlocal converted_always_bodies, pending_continuous
        if not pending_continuous:
            return
        body_program = encode_body_stmt_ops(
            BlockIR(tuple(pending_continuous)),
            bindings,
            local_node_slots,
            side_effects=side_effect_builder,
        )
        pending_continuous = []
        if body_program is None:
            reasons.append(f"{prefix}:continuous_body_not_lowered")
            return
        body_start, body_count = _append_body_program(
            body_stmt_ops,
            body_expr_ops,
            body_program,
            node_slot_to_global=node_slot_to_global,
            state_slot_to_global=state_slot_to_global,
            param_slot_to_global=param_slot_to_global,
            side_effect_slot_offset=side_effect_slot_offset,
        )
        if body_count <= 0:
            return
        events.append(
            RustSimEvent(
                kind=EVENT_DUE_ALWAYS,
                event_id=len(events),
                phase=phase,
                body_stmt_start=body_start,
                body_stmt_count=body_count,
            )
        )
        converted_always_bodies += 1

    for stmt in body_ir.statements:
        if _is_continuous_body_stmt(stmt):
            if seen_transition:
                pending_continuous.append(stmt)
                flush_continuous_body(EVENT_PHASE_POST)
            else:
                pending_continuous.append(stmt)
            continue

        if isinstance(stmt, EventStatementIR):
            flush_continuous_body()
            due_program = encode_event_due_program(stmt.event, bindings, local_node_slots)
            body_program = encode_body_stmt_ops(
                stmt.body,
                bindings,
                local_node_slots,
                side_effects=side_effect_builder,
            )
            if due_program is None:
                reasons.append(f"{prefix}:event_due_not_lowered")
                continue
            if body_program is None:
                reasons.append(f"{prefix}:event_body_not_lowered")
                for tag in classify_body_stmt_ops_rejection(
                    stmt.body,
                    bindings,
                    local_node_slots,
                ):
                    reasons.append(f"{prefix}:event_body:{tag}")
                continue
            trigger_phases = _event_trigger_phases(stmt.event, phase_contributed_nodes)
            if len(trigger_phases) != len(due_program.triggers):
                reasons.append(f"{prefix}:event_phase_trigger_mismatch")
                continue
            body_start, body_count = _append_body_program(
                body_stmt_ops,
                body_expr_ops,
                body_program,
                node_slot_to_global=node_slot_to_global,
                state_slot_to_global=state_slot_to_global,
                param_slot_to_global=param_slot_to_global,
                side_effect_slot_offset=side_effect_slot_offset,
            )
            for trigger, trigger_phase in zip(due_program.triggers, trigger_phases):
                expr_start, expr_count = _append_expr_segment(
                    body_expr_ops,
                    tuple(trigger.expr_ops),
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                time_tol_start, time_tol_count = _append_expr_segment(
                    body_expr_ops,
                    tuple(trigger.time_tol_ops),
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                expr_tol_start, expr_tol_count = _append_expr_segment(
                    body_expr_ops,
                    tuple(trigger.expr_tol_ops),
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                timer_start_expr_start, timer_start_expr_count = _append_expr_segment(
                    body_expr_ops,
                    tuple(trigger.timer_start_ops),
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                timer_period_expr_start, timer_period_expr_count = _append_expr_segment(
                    body_expr_ops,
                    tuple(trigger.timer_period_ops),
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                events.append(
                    RustSimEvent(
                        kind=str(trigger.kind),
                        event_id=len(events),
                        phase=int(trigger_phase),
                        direction=int(trigger.direction),
                        expr_start=expr_start,
                        expr_count=expr_count,
                        time_tol_start=time_tol_start,
                        time_tol_count=time_tol_count,
                        expr_tol_start=expr_tol_start,
                        expr_tol_count=expr_tol_count,
                        timer_start_expr_start=timer_start_expr_start,
                        timer_start_expr_count=timer_start_expr_count,
                        timer_period_expr_start=timer_period_expr_start,
                        timer_period_expr_count=timer_period_expr_count,
                        body_stmt_start=body_start,
                        body_stmt_count=body_count,
                    )
                )
                converted_events += 1
            continue

        if isinstance(stmt, (ContributionIR, ForStatementIR)):
            flush_continuous_body()
            transition_program = encode_transition_contribution_program(
                BlockIR((stmt,)),
                bindings,
                local_node_slots,
            )
            if transition_program is None:
                direct_program = encode_body_stmt_ops(
                    BlockIR((stmt,)),
                    bindings,
                    local_node_slots,
                    side_effects=side_effect_builder,
                )
                if direct_program is not None:
                    body_start, body_count = _append_body_program(
                        body_stmt_ops,
                        body_expr_ops,
                        direct_program,
                        node_slot_to_global=node_slot_to_global,
                        state_slot_to_global=state_slot_to_global,
                        param_slot_to_global=param_slot_to_global,
                        side_effect_slot_offset=side_effect_slot_offset,
                    )
                    if body_count > 0:
                        events.append(
                            RustSimEvent(
                                kind=EVENT_DUE_ALWAYS,
                                event_id=len(events),
                                phase=EVENT_PHASE_PRE,
                                body_stmt_start=body_start,
                                body_stmt_count=body_count,
                            )
                        )
                        converted_always_bodies += 1
                    continue
                reasons.append(f"{prefix}:continuous_contribution_not_lowered")
                continue
            for idx, output_slot in enumerate(transition_program.output_node_slots):
                expr_base = idx * 6
                if expr_base + 5 >= len(transition_program.expr_segments):
                    reasons.append(f"{prefix}:transition_expr_segment_mismatch")
                    continue
                target_start, target_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                delay_start, delay_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base + 1],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                rise_start, rise_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base + 2],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                fall_start, fall_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base + 3],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                output_bias_start, output_bias_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base + 4],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                output_scale_start, output_scale_count = _append_expr_segment(
                    body_expr_ops,
                    transition_program.expr_segments[expr_base + 5],
                    node_slot_to_global=node_slot_to_global,
                    state_slot_to_global=state_slot_to_global,
                    param_slot_to_global=param_slot_to_global,
                )
                reference_slot = transition_program.reference_node_slots[idx]
                transitions.append(
                    RustSimTransition(
                        transition_id=len(transitions),
                        output_node_id=int(node_slot_to_global.get(int(output_slot), output_slot)),
                        reference_node_id=(
                            None
                            if reference_slot is None
                            else int(node_slot_to_global.get(int(reference_slot), reference_slot))
                        ),
                        target_expr_start=target_start,
                        target_expr_count=target_count,
                        delay_expr_start=delay_start,
                        delay_expr_count=delay_count,
                        rise_expr_start=rise_start,
                        rise_expr_count=rise_count,
                        fall_expr_start=fall_start,
                        fall_expr_count=fall_count,
                        output_bias_expr_start=output_bias_start,
                        output_bias_expr_count=output_bias_count,
                        output_scale_expr_start=output_scale_start,
                        output_scale_expr_count=output_scale_count,
                        default_transition=float(
                            getattr(model, "default_transition", 1.0e-12)
                            or 1.0e-12
                        ),
                    )
                )
                converted_transitions += 1
                seen_transition = True
            continue

    flush_continuous_body()

    if converted_events == 0 and converted_always_bodies == 0 and converted_transitions == 0:
        return (f"{prefix}:no_event_transition_ir",)
    side_effects.extend(side_effect_builder.effects)
    return tuple(reasons)


def _source_from_metadata(
    *,
    node: str,
    node_id: int,
    meta: Mapping[str, Any],
    source_data: list[float],
) -> tuple[Optional[RustSimSource], Optional[str]]:
    kind = str(meta.get("kind", ""))
    if kind == SOURCE_DC:
        return (
            RustSimSource(
                node=node,
                node_id=node_id,
                kind=SOURCE_DC,
                params=(float(meta.get("voltage", 0.0) or 0.0),),
            ),
            None,
        )
    if kind == SOURCE_PULSE:
        flags = 0
        if bool(meta.get("has_width", False)):
            flags |= 1
        if bool(meta.get("one_shot", False)):
            flags |= 2
        return (
            RustSimSource(
                node=node,
                node_id=node_id,
                kind=SOURCE_PULSE,
                flags=flags,
                params=(
                    float(meta.get("v_lo", 0.0) or 0.0),
                    float(meta.get("v_hi", 0.0) or 0.0),
                    float(meta.get("period", 0.0) or 0.0),
                    float(meta.get("duty", 0.5) or 0.5),
                    float(meta.get("rise", 0.0) or 0.0),
                    float(meta.get("fall", 0.0) or 0.0),
                    float(meta.get("delay", 0.0) or 0.0),
                    float(meta.get("width", 0.0) or 0.0),
                ),
            ),
            None,
        )
    if kind == SOURCE_SINE:
        return (
            RustSimSource(
                node=node,
                node_id=node_id,
                kind=SOURCE_SINE,
                params=(
                    float(meta.get("offset", 0.0) or 0.0),
                    float(meta.get("amplitude", 0.0) or 0.0),
                    float(meta.get("freq", 0.0) or 0.0),
                    float(meta.get("phase", 0.0) or 0.0),
                ),
            ),
            None,
        )
    if kind == SOURCE_PWL:
        times = tuple(float(value) for value in meta.get("times", ()) or ())
        values = tuple(float(value) for value in meta.get("values", ()) or ())
        if not times or len(times) != len(values):
            return None, f"source:{node}:invalid_pwl_payload"
        data_start = len(source_data)
        source_data.extend(times)
        source_data.extend(values)
        return (
            RustSimSource(
                node=node,
                node_id=node_id,
                kind=SOURCE_PWL,
                data_start=data_start,
                data_count=len(times),
                params=(),
            ),
            None,
        )
    return None, f"source:{node}:unsupported_waveform:{kind or 'unknown'}"


def build_source_record_rust_program(
    *,
    sources: Iterable[Any],
    recorded_signals: Iterable[str],
    models: Iterable[Any],
) -> RustSimCompileReport:
    """Lower source+record/no-model simulation into RustSimProgram.

    This is the first strict EVAS2 production slice.  Model semantics are not
    accepted here yet; they must be represented by body/event/transition opcodes
    before a model-containing circuit can run on the Rust-owned loop.
    """

    source_list = tuple(sources)
    record_names = tuple(str(name) for name in recorded_signals)
    model_list = tuple(models)
    global_contributed_nodes = _collect_global_contributed_nodes(model_list)
    reasons: list[str] = []
    if not record_names:
        reasons.append("no_recorded_signals")

    nodes: list[RustSimNode] = []
    states: list[RustSimState] = []
    params: list[RustSimParam] = []
    node_ids: dict[str, int] = {}
    state_ids: dict[tuple[int, str], int] = {}
    param_ids: dict[tuple[int, str], int] = {}
    rust_sources: list[RustSimSource] = []
    continuous_linear_ops: list[RustSimLinearOp] = []
    rust_events: list[RustSimEvent] = []
    rust_transitions: list[RustSimTransition] = []
    body_stmt_ops: list[BodyStmtOp] = []
    body_expr_ops: list[BodyExprOp] = []
    side_effects: list[RustSimSideEffect] = []
    source_data: list[float] = []
    for source in source_list:
        node = str(getattr(source, "node", ""))
        if not node:
            reasons.append("source_without_node")
            continue
        node_id = _add_node(node, node_ids, nodes)
        meta = _waveform_metadata(getattr(source, "waveform", None))
        if meta is None:
            reasons.append(f"source:{node}:missing_waveform_metadata")
            continue
        rust_source, reason = _source_from_metadata(
            node=node,
            node_id=node_id,
            meta=meta,
            source_data=source_data,
        )
        if reason is not None:
            reasons.append(reason)
        elif rust_source is not None:
            rust_sources.append(rust_source)

    for model_index, model in enumerate(model_list):
        model_cls = getattr(model, "__class__", type(model))
        has_rustsim_program_ir = bool(
            tuple(getattr(model_cls, "_event_static_linear_ir_ops", ()) or ())
            or tuple(getattr(model_cls, "_event_timer_static_linear_ir_ops", ()) or ())
            or tuple(getattr(model_cls, "_transition_target_ir_ops", ()) or ())
            or tuple(getattr(model_cls, "_whole_segment_candidates", ()) or ())
            or _model_has_rustsim_event_transition_candidate(model_cls)
            or _model_has_rustsim_continuous_body_candidate(model_cls)
        )
        if has_rustsim_program_ir:
            event_reasons = _convert_event_transition_ops(
                model=model,
                model_index=model_index,
                global_contributed_nodes=global_contributed_nodes,
                node_ids=node_ids,
                nodes=nodes,
                state_ids=state_ids,
                states=states,
                param_ids=param_ids,
                params=params,
                events=rust_events,
                transitions=rust_transitions,
                body_stmt_ops=body_stmt_ops,
                body_expr_ops=body_expr_ops,
                side_effects=side_effects,
            )
            if event_reasons:
                reasons.extend(event_reasons)
                continue
        else:
            dynamic_reasons = _reject_model_dynamic_semantics(model, model_index)
            if dynamic_reasons:
                reasons.extend(dynamic_reasons)
                continue
        model_ops, model_reasons = _convert_continuous_linear_ops(
            model=model,
            model_index=model_index,
            node_ids=node_ids,
            nodes=nodes,
            state_ids=state_ids,
            states=states,
        )
        if has_rustsim_program_ir and all(
            str(reason).endswith(":no_continuous_linear_ir")
            for reason in model_reasons
        ):
            model_reasons = ()
        if model_reasons:
            reasons.extend(model_reasons)
        continuous_linear_ops.extend(model_ops)

    records: list[RustSimRecord] = []
    for name in record_names:
        node_id = _add_node(name, node_ids, nodes)
        records.append(RustSimRecord(name=name, node_id=node_id))

    if reasons:
        return RustSimCompileReport(
            program=None,
            supported=False,
            reasons=tuple(reasons),
        )

    return RustSimCompileReport(
        program=RustSimProgram(
            nodes=tuple(nodes),
            states=tuple(states),
            params=tuple(params),
            sources=tuple(rust_sources),
            events=tuple(rust_events),
            body_ops=(),
            transitions=tuple(rust_transitions),
            records=tuple(records),
            side_effects=tuple(side_effects),
            continuous_linear_ops=tuple(continuous_linear_ops),
            body_stmt_ops=tuple(body_stmt_ops),
            body_expr_ops=tuple(body_expr_ops),
            source_data=tuple(source_data),
            bound_step_ops=(),
            final_step_ops=(),
        ),
        supported=True,
        reasons=(),
    )
