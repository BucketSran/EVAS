"""Evaluate IR helpers for native/static simulator hot paths.

The IR in this module is intentionally small.  It covers linear
assignments/contributions over static node and scalar-state sources, plus
conditional selects.  Some piecewise-linear function calls such as
``abs()``, ``min()``, and ``max()`` are lowered as conditional linear selects
when their arguments are static linear expressions.  Event operators, dynamic
bus nodes, arrays with dynamic indexes, and general non-linear function calls
stay on the generated Python evaluator until their semantics can be lowered
separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple

SOURCE_NODE = 0
SOURCE_STATE = 1
TARGET_NODE = 0
TARGET_STATE = 1
COND_ALWAYS = 0
COND_GT = 1
COND_LT = 2
COND_GE = 3
COND_LE = 4
COND_EQ = 5
COND_NE = 6


@dataclass(frozen=True)
class LinearTermIR:
    """One ``gain * source`` term in a static linear evaluate operation."""

    source_kind: int
    source_name: str
    gain: object


@dataclass(frozen=True)
class LinearOpIR:
    """One ordered static linear write in a model evaluate body."""

    target_kind: int
    target_name: str
    bias: object
    terms: Tuple[LinearTermIR, ...]
    condition: "ConditionIR | None" = None
    false_bias: object = 0.0
    false_terms: Tuple[LinearTermIR, ...] = ()
    target_integer: bool = False


@dataclass(frozen=True)
class TransitionTargetIR:
    """One transition() target expression lowered to linear array IR."""

    output_node: str
    reference_node: str | None
    transition_key: Optional[str]
    bias: object
    terms: Tuple[LinearTermIR, ...]
    condition: "ConditionIR | None" = None
    false_bias: object = 0.0
    false_terms: Tuple[LinearTermIR, ...] = ()
    delay: object = 0.0
    rise: object = 0.0
    fall: object = 0.0


@dataclass(frozen=True)
class ConditionIR:
    """One comparison over two static linear expressions."""

    op_kind: int
    left_bias: object
    left_terms: Tuple[LinearTermIR, ...]
    right_bias: object
    right_terms: Tuple[LinearTermIR, ...]


def linear_op_uses_params(op: LinearOpIR) -> bool:
    """Return True when an op contains parameter-dependent coefficients."""

    return (
        _scalar_uses_params(op.bias)
        or any(_scalar_uses_params(term.gain) for term in op.terms)
        or _scalar_uses_params(op.false_bias)
        or any(_scalar_uses_params(term.gain) for term in op.false_terms)
        or (
            op.condition is not None
            and (
                _scalar_uses_params(op.condition.left_bias)
                or any(
                    _scalar_uses_params(term.gain)
                    for term in op.condition.left_terms
                )
                or _scalar_uses_params(op.condition.right_bias)
                or any(
                    _scalar_uses_params(term.gain)
                    for term in op.condition.right_terms
                )
            )
        )
    )


def _scalar_uses_params(value: object) -> bool:
    return isinstance(value, tuple)


def evaluate_linear_python(
    ops: Iterable[LinearOpIR],
    *,
    node_values: MutableSequence[float],
    state_values: MutableSequence[float],
    node_ids: Mapping[str, int],
    state_ids: Mapping[str, int],
    params: Mapping[str, object],
    scalar_eval,
) -> None:
    """Execute static linear IR using Python arrays.

    This is the B1 parity executor: it exercises the same node/state-id layout
    that the Rust backend consumes, but it does not replace the general Python
    evaluator for unsupported models.
    """

    for op in ops:
        selected_bias = op.bias
        selected_terms = op.terms
        if op.condition is not None:
            if not _evaluate_condition_python(
                op.condition,
                node_values=node_values,
                state_values=state_values,
                node_ids=node_ids,
                state_ids=state_ids,
                params=params,
                scalar_eval=scalar_eval,
            ):
                selected_bias = op.false_bias
                selected_terms = op.false_terms

        value = float(scalar_eval(selected_bias, params))
        for term in selected_terms:
            coeff = float(scalar_eval(term.gain, params))
            if term.source_kind == SOURCE_NODE:
                value += coeff * float(node_values[node_ids[term.source_name]])
            elif term.source_kind == SOURCE_STATE:
                value += coeff * float(state_values[state_ids[term.source_name]])
            else:
                raise ValueError(f"unsupported source kind: {term.source_kind!r}")

        if op.target_kind == TARGET_NODE:
            node_values[node_ids[op.target_name]] = value
        elif op.target_kind == TARGET_STATE:
            if op.target_integer:
                value = float(_to_integer(value))
            state_values[state_ids[op.target_name]] = value
        else:
            raise ValueError(f"unsupported target kind: {op.target_kind!r}")


def evaluate_transition_targets_python(
    ops: Iterable[TransitionTargetIR],
    *,
    node_values: MutableSequence[float],
    state_values: MutableSequence[float],
    target_values: MutableSequence[float],
    delay_values: MutableSequence[float],
    rise_values: MutableSequence[float],
    fall_values: MutableSequence[float],
    node_ids: Mapping[str, int],
    state_ids: Mapping[str, int],
    params: Mapping[str, object],
    scalar_eval,
) -> None:
    """Execute transition target IR using Python arrays."""

    for idx, op in enumerate(ops):
        if idx >= len(target_values):
            raise ValueError("transition target output buffer is too small")
        selected_bias = op.bias
        selected_terms = op.terms
        if op.condition is not None:
            if not _evaluate_condition_python(
                op.condition,
                node_values=node_values,
                state_values=state_values,
                node_ids=node_ids,
                state_ids=state_ids,
                params=params,
                scalar_eval=scalar_eval,
            ):
                selected_bias = op.false_bias
                selected_terms = op.false_terms
        target_values[idx] = _evaluate_linear_value_python(
            selected_bias,
            selected_terms,
            node_values=node_values,
            state_values=state_values,
            node_ids=node_ids,
            state_ids=state_ids,
            params=params,
            scalar_eval=scalar_eval,
        )
        delay_values[idx] = float(scalar_eval(op.delay, params))
        rise_values[idx] = float(scalar_eval(op.rise, params))
        fall_values[idx] = float(scalar_eval(op.fall, params))


def _evaluate_condition_python(
    condition: ConditionIR,
    *,
    node_values: MutableSequence[float],
    state_values: MutableSequence[float],
    node_ids: Mapping[str, int],
    state_ids: Mapping[str, int],
    params: Mapping[str, object],
    scalar_eval,
) -> bool:
    left = _evaluate_linear_value_python(
        condition.left_bias,
        condition.left_terms,
        node_values=node_values,
        state_values=state_values,
        node_ids=node_ids,
        state_ids=state_ids,
        params=params,
        scalar_eval=scalar_eval,
    )
    right = _evaluate_linear_value_python(
        condition.right_bias,
        condition.right_terms,
        node_values=node_values,
        state_values=state_values,
        node_ids=node_ids,
        state_ids=state_ids,
        params=params,
        scalar_eval=scalar_eval,
    )
    if condition.op_kind == COND_GT:
        return left > right
    if condition.op_kind == COND_LT:
        return left < right
    if condition.op_kind == COND_GE:
        return left >= right
    if condition.op_kind == COND_LE:
        return left <= right
    if condition.op_kind == COND_EQ:
        return left == right
    if condition.op_kind == COND_NE:
        return left != right
    raise ValueError(f"unsupported condition kind: {condition.op_kind!r}")


def _evaluate_linear_value_python(
    bias: object,
    terms: Tuple[LinearTermIR, ...],
    *,
    node_values: MutableSequence[float],
    state_values: MutableSequence[float],
    node_ids: Mapping[str, int],
    state_ids: Mapping[str, int],
    params: Mapping[str, object],
    scalar_eval,
) -> float:
    value = float(scalar_eval(bias, params))
    for term in terms:
        coeff = float(scalar_eval(term.gain, params))
        if term.source_kind == SOURCE_NODE:
            value += coeff * float(node_values[node_ids[term.source_name]])
        elif term.source_kind == SOURCE_STATE:
            value += coeff * float(state_values[state_ids[term.source_name]])
        else:
            raise ValueError(f"unsupported source kind: {term.source_kind!r}")
    return value


def normalize_linear_ops(raw_ops: Sequence[tuple]) -> Tuple[LinearOpIR, ...]:
    """Convert tuple metadata stored on generated model classes into IR objects."""

    ops = []
    for raw_op in raw_ops:
        if len(raw_op) == 4:
            target_kind, target_name, bias, raw_terms = raw_op
            raw_condition = None
            false_bias = 0.0
            raw_false_terms = ()
            target_integer = False
        elif len(raw_op) == 7:
            (
                target_kind,
                target_name,
                bias,
                raw_terms,
                raw_condition,
                false_bias,
                raw_false_terms,
            ) = raw_op
            target_integer = False
        elif len(raw_op) == 8:
            (
                target_kind,
                target_name,
                bias,
                raw_terms,
                raw_condition,
                false_bias,
                raw_false_terms,
                target_integer,
            ) = raw_op
        else:
            raise ValueError(f"unsupported static linear op metadata: {raw_op!r}")
        terms = tuple(
            LinearTermIR(int(source_kind), str(source_name), gain)
            for source_kind, source_name, gain in raw_terms
        )
        false_terms = tuple(
            LinearTermIR(int(source_kind), str(source_name), gain)
            for source_kind, source_name, gain in raw_false_terms
        )
        condition = None
        if raw_condition is not None:
            (
                op_kind,
                left_bias,
                raw_left_terms,
                right_bias,
                raw_right_terms,
            ) = raw_condition
            condition = ConditionIR(
                int(op_kind),
                left_bias,
                tuple(
                    LinearTermIR(int(source_kind), str(source_name), gain)
                    for source_kind, source_name, gain in raw_left_terms
                ),
                right_bias,
                tuple(
                    LinearTermIR(int(source_kind), str(source_name), gain)
                    for source_kind, source_name, gain in raw_right_terms
                ),
            )
        ops.append(
            LinearOpIR(
                int(target_kind),
                str(target_name),
                bias,
                terms,
                condition,
                false_bias,
                false_terms,
                bool(target_integer),
            )
        )
    return tuple(ops)


def normalize_transition_target_ops(
    raw_ops: Sequence[tuple],
) -> Tuple[TransitionTargetIR, ...]:
    """Convert generated transition target metadata into IR objects."""

    ops = []
    for raw_op in raw_ops:
        if len(raw_op) == 10:
            transition_key = None
            (
                output_node,
                reference_node,
                bias,
                raw_terms,
                raw_condition,
                false_bias,
                raw_false_terms,
                delay,
                rise,
                fall,
            ) = raw_op
        elif len(raw_op) == 11:
            (
                output_node,
                reference_node,
                transition_key,
                bias,
                raw_terms,
                raw_condition,
                false_bias,
                raw_false_terms,
                delay,
                rise,
                fall,
            ) = raw_op
        else:
            raise ValueError(f"unsupported transition target metadata: {raw_op!r}")
        terms = tuple(
            LinearTermIR(int(source_kind), str(source_name), gain)
            for source_kind, source_name, gain in raw_terms
        )
        false_terms = tuple(
            LinearTermIR(int(source_kind), str(source_name), gain)
            for source_kind, source_name, gain in raw_false_terms
        )
        condition = None
        if raw_condition is not None:
            (
                op_kind,
                left_bias,
                raw_left_terms,
                right_bias,
                raw_right_terms,
            ) = raw_condition
            condition = ConditionIR(
                int(op_kind),
                left_bias,
                tuple(
                    LinearTermIR(int(source_kind), str(source_name), gain)
                    for source_kind, source_name, gain in raw_left_terms
                ),
                right_bias,
                tuple(
                    LinearTermIR(int(source_kind), str(source_name), gain)
                    for source_kind, source_name, gain in raw_right_terms
                ),
            )
        ops.append(
            TransitionTargetIR(
                str(output_node),
                None if reference_node is None else str(reference_node),
                None if transition_key is None else str(transition_key),
                bias,
                terms,
                condition,
                false_bias,
                false_terms,
                delay,
                rise,
                fall,
            )
        )
    return tuple(ops)


def _to_integer(value: object) -> int:
    """Verilog-A real-to-integer assignment rounds to nearest."""

    import math

    v = float(value)
    if not math.isfinite(v):
        return 0
    if v >= 0.0:
        return math.floor(v + 0.5)
    return math.ceil(v - 0.5)
