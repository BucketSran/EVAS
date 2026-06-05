"""Optional ctypes bridge for the EVAS Rust kernel prototype."""

from __future__ import annotations

import ctypes
import os
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableSequence, Optional, Tuple, Union


class RustBackendError(RuntimeError):
    """Raised when an opt-in Rust backend call fails."""


def _usize_max() -> int:
    return ctypes.c_size_t(-1).value


class RustStaticAffineOp(ctypes.Structure):
    _fields_ = [
        ("read_node_id", ctypes.c_size_t),
        ("write_node_id", ctypes.c_size_t),
        ("gain", ctypes.c_double),
        ("bias", ctypes.c_double),
    ]


class RustLinearTerm(ctypes.Structure):
    _fields_ = [
        ("source_kind", ctypes.c_uint8),
        ("source_id", ctypes.c_size_t),
        ("gain", ctypes.c_double),
    ]


class RustLinearCondition(ctypes.Structure):
    _fields_ = [
        ("op_kind", ctypes.c_uint8),
        ("left_term_start", ctypes.c_size_t),
        ("left_term_count", ctypes.c_size_t),
        ("left_bias", ctypes.c_double),
        ("right_term_start", ctypes.c_size_t),
        ("right_term_count", ctypes.c_size_t),
        ("right_bias", ctypes.c_double),
    ]


class RustLinearOp(ctypes.Structure):
    _fields_ = [
        ("target_kind", ctypes.c_uint8),
        ("target_integer", ctypes.c_uint8),
        ("target_id", ctypes.c_size_t),
        ("term_start", ctypes.c_size_t),
        ("term_count", ctypes.c_size_t),
        ("bias", ctypes.c_double),
        ("condition_id", ctypes.c_size_t),
        ("false_term_start", ctypes.c_size_t),
        ("false_term_count", ctypes.c_size_t),
        ("false_bias", ctypes.c_double),
    ]


class RustTransitionTargetOp(ctypes.Structure):
    _fields_ = [
        ("target_id", ctypes.c_size_t),
        ("term_start", ctypes.c_size_t),
        ("term_count", ctypes.c_size_t),
        ("bias", ctypes.c_double),
        ("condition_id", ctypes.c_size_t),
        ("false_term_start", ctypes.c_size_t),
        ("false_term_count", ctypes.c_size_t),
        ("false_bias", ctypes.c_double),
        ("delay", ctypes.c_double),
        ("rise", ctypes.c_double),
        ("fall", ctypes.c_double),
    ]


class RustBodyExprOp(ctypes.Structure):
    _fields_ = [
        ("op_kind", ctypes.c_uint8),
        ("index", ctypes.c_size_t),
        ("value", ctypes.c_double),
    ]


class RustBodyStmtOp(ctypes.Structure):
    _fields_ = [
        ("target_kind", ctypes.c_uint8),
        ("target_integer", ctypes.c_uint8),
        ("target_id", ctypes.c_size_t),
        ("expr_start", ctypes.c_size_t),
        ("expr_count", ctypes.c_size_t),
    ]


BODY_EXPR_CONST = 0
BODY_EXPR_READ_NODE = 1
BODY_EXPR_READ_STATE = 2
BODY_EXPR_READ_PARAM = 3
BODY_EXPR_NEG = 10
BODY_EXPR_NOT = 11
BODY_EXPR_ADD = 20
BODY_EXPR_SUB = 21
BODY_EXPR_MUL = 22
BODY_EXPR_DIV = 23
BODY_EXPR_MOD = 24
BODY_EXPR_GT = 30
BODY_EXPR_LT = 31
BODY_EXPR_GE = 32
BODY_EXPR_LE = 33
BODY_EXPR_EQ = 34
BODY_EXPR_NE = 35
BODY_EXPR_LAND = 36
BODY_EXPR_LOR = 37
BODY_EXPR_BITAND = 38
BODY_EXPR_BITOR = 39
BODY_EXPR_BITXOR = 40
BODY_EXPR_SELECT = 50
BODY_EXPR_ABS = 60
BODY_EXPR_SQRT = 61
BODY_EXPR_EXP = 62
BODY_EXPR_LN = 63
BODY_EXPR_LOG10 = 64
BODY_EXPR_SIN = 65
BODY_EXPR_COS = 66
BODY_EXPR_FLOOR = 67
BODY_EXPR_CEIL = 68
BODY_EXPR_MIN = 69
BODY_EXPR_MAX = 70
BODY_EXPR_POW = 71

BODY_TARGET_NODE = 0
BODY_TARGET_STATE = 1


class RustLfsrEventBatch:
    """ctypes-backed LFSR event-body write batch."""

    def __init__(
        self,
        *,
        lfsr_slots: Iterable[int],
        tmp_slots: Iterable[int],
        tap_slots: Iterable[int],
        gate_node_id: Optional[int] = None,
        gate_threshold: float = 0.5,
        high_node_id: Optional[int] = None,
        low_node_id: Optional[int] = None,
        output_state_id: Optional[int] = None,
        output_node_id: Optional[int] = None,
        loop_state_id: Optional[int] = None,
        loop_final_value: float = 0.0,
    ):
        self.lfsr_slots: Tuple[int, ...] = tuple(int(slot) for slot in lfsr_slots)
        self.tmp_slots: Tuple[int, ...] = tuple(int(slot) for slot in tmp_slots)
        self.tap_slots: Tuple[int, ...] = tuple(int(slot) for slot in tap_slots)
        self.gate_node_id = _usize_max() if gate_node_id is None else int(gate_node_id)
        self.gate_threshold = float(gate_threshold)
        self.high_node_id = _usize_max() if high_node_id is None else int(high_node_id)
        self.low_node_id = _usize_max() if low_node_id is None else int(low_node_id)
        self.output_state_id = (
            _usize_max() if output_state_id is None else int(output_state_id)
        )
        self.output_node_id = _usize_max() if output_node_id is None else int(output_node_id)
        self.loop_state_id = _usize_max() if loop_state_id is None else int(loop_state_id)
        self.loop_final_value = float(loop_final_value)
        lfsr_array_type = ctypes.c_size_t * len(self.lfsr_slots)
        tmp_array_type = ctypes.c_size_t * len(self.tmp_slots)
        tap_array_type = ctypes.c_size_t * len(self.tap_slots)
        self._c_lfsr_slots = lfsr_array_type(*self.lfsr_slots)
        self._c_tmp_slots = tmp_array_type(*self.tmp_slots)
        self._c_tap_slots = tap_array_type(*self.tap_slots)

    def __len__(self) -> int:
        return len(self.lfsr_slots)

    @property
    def lfsr_ptr(self):
        return self._c_lfsr_slots

    @property
    def tmp_ptr(self):
        return self._c_tmp_slots

    @property
    def tap_ptr(self):
        return self._c_tap_slots


class RustNodeIdBatch:
    """ctypes-backed node-id batch with stable storage for FFI calls."""

    def __init__(self, node_ids: Iterable[int]):
        self.node_ids: Tuple[int, ...] = tuple(int(node_id) for node_id in node_ids)
        array_type = ctypes.c_size_t * len(self.node_ids)
        self._c_node_ids = array_type(*self.node_ids)

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def ptr(self):
        return self._c_node_ids


@dataclass(frozen=True)
class StaticAffineOp:
    """Python-side static affine operation consumed by the Rust ABI."""

    read_node_id: int
    write_node_id: int
    gain: float
    bias: float


@dataclass(frozen=True)
class LinearTerm:
    """Python-side linear source term consumed by the Rust ABI."""

    source_kind: int
    source_id: int
    gain: float


@dataclass(frozen=True)
class LinearCondition:
    """Python-side conditional select consumed by the Rust ABI."""

    op_kind: int
    left_bias: float
    left_terms: Tuple[LinearTerm, ...]
    right_bias: float
    right_terms: Tuple[LinearTerm, ...]


@dataclass(frozen=True)
class LinearOp:
    """Python-side static linear write consumed by the Rust ABI."""

    target_kind: int
    target_id: int
    bias: float
    terms: Tuple[LinearTerm, ...]
    condition: Optional[LinearCondition] = None
    false_bias: float = 0.0
    false_terms: Tuple[LinearTerm, ...] = ()
    target_integer: bool = False


@dataclass(frozen=True)
class TransitionTargetOp:
    """Python-side transition target expression consumed by the Rust ABI."""

    target_id: int
    bias: float
    terms: Tuple[LinearTerm, ...]
    condition: Optional[LinearCondition] = None
    false_bias: float = 0.0
    false_terms: Tuple[LinearTerm, ...] = ()
    delay: float = 0.0
    rise: float = 0.0
    fall: float = 0.0


@dataclass(frozen=True)
class BodyExprOp:
    """One stack-machine expression op consumed by the 094e Rust body ABI."""

    op_kind: int
    index: int = 0
    value: float = 0.0


@dataclass(frozen=True)
class BodyStmtOp:
    """One body statement write consumed by the 094e Rust body ABI."""

    target_kind: int
    target_id: int
    expr_start: int
    expr_count: int
    target_integer: bool = False


class RustStaticAffineBatch:
    """ctypes-backed operation batch with stable storage for FFI calls."""

    def __init__(self, ops: Iterable[StaticAffineOp]):
        self.ops: Tuple[StaticAffineOp, ...] = tuple(ops)
        array_type = RustStaticAffineOp * len(self.ops)
        self._c_ops = array_type(
            *(
                RustStaticAffineOp(
                    int(op.read_node_id),
                    int(op.write_node_id),
                    float(op.gain),
                    float(op.bias),
                )
                for op in self.ops
            )
        )

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def ptr(self):
        return self._c_ops


class RustLinearBatch:
    """ctypes-backed static linear operation batch with stable storage."""

    def __init__(self, ops: Iterable[LinearOp]):
        self.ops: Tuple[LinearOp, ...] = tuple(ops)
        terms = []
        conditions = []
        c_ops = []
        for op in self.ops:
            term_start = len(terms)
            terms.extend(op.terms)
            false_term_start = len(terms)
            terms.extend(op.false_terms)
            condition_id = _usize_max()
            if op.condition is not None:
                condition_id = len(conditions)
                left_term_start = len(terms)
                terms.extend(op.condition.left_terms)
                right_term_start = len(terms)
                terms.extend(op.condition.right_terms)
                conditions.append(
                    RustLinearCondition(
                        int(op.condition.op_kind),
                        int(left_term_start),
                        int(len(op.condition.left_terms)),
                        float(op.condition.left_bias),
                        int(right_term_start),
                        int(len(op.condition.right_terms)),
                        float(op.condition.right_bias),
                    )
                )
            c_ops.append(
                RustLinearOp(
                    int(op.target_kind),
                    int(bool(op.target_integer)),
                    int(op.target_id),
                    int(term_start),
                    int(len(op.terms)),
                    float(op.bias),
                    int(condition_id),
                    int(false_term_start),
                    int(len(op.false_terms)),
                    float(op.false_bias),
                )
            )

        term_array_type = RustLinearTerm * len(terms)
        condition_array_type = RustLinearCondition * len(conditions)
        op_array_type = RustLinearOp * len(c_ops)
        self.terms: Tuple[LinearTerm, ...] = tuple(terms)
        self.conditions = tuple(conditions)
        self._c_terms = term_array_type(
            *(
                RustLinearTerm(
                    int(term.source_kind),
                    int(term.source_id),
                    float(term.gain),
                )
                for term in self.terms
            )
        )
        self._c_conditions = condition_array_type(*conditions)
        self._c_ops = op_array_type(*c_ops)

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def op_ptr(self):
        return self._c_ops

    @property
    def term_ptr(self):
        return self._c_terms

    @property
    def condition_ptr(self):
        return self._c_conditions


class RustTransitionTargetBatch:
    """ctypes-backed transition target operation batch."""

    def __init__(self, ops: Iterable[TransitionTargetOp]):
        self.ops: Tuple[TransitionTargetOp, ...] = tuple(ops)
        terms = []
        conditions = []
        c_ops = []
        for op in self.ops:
            term_start = len(terms)
            terms.extend(op.terms)
            false_term_start = len(terms)
            terms.extend(op.false_terms)
            condition_id = _usize_max()
            if op.condition is not None:
                condition_id = len(conditions)
                left_term_start = len(terms)
                terms.extend(op.condition.left_terms)
                right_term_start = len(terms)
                terms.extend(op.condition.right_terms)
                conditions.append(
                    RustLinearCondition(
                        int(op.condition.op_kind),
                        int(left_term_start),
                        int(len(op.condition.left_terms)),
                        float(op.condition.left_bias),
                        int(right_term_start),
                        int(len(op.condition.right_terms)),
                        float(op.condition.right_bias),
                    )
                )
            c_ops.append(
                RustTransitionTargetOp(
                    int(op.target_id),
                    int(term_start),
                    int(len(op.terms)),
                    float(op.bias),
                    int(condition_id),
                    int(false_term_start),
                    int(len(op.false_terms)),
                    float(op.false_bias),
                    float(op.delay),
                    float(op.rise),
                    float(op.fall),
                )
            )

        term_array_type = RustLinearTerm * len(terms)
        condition_array_type = RustLinearCondition * len(conditions)
        op_array_type = RustTransitionTargetOp * len(c_ops)
        self.terms: Tuple[LinearTerm, ...] = tuple(terms)
        self.conditions = tuple(conditions)
        self._c_terms = term_array_type(
            *(
                RustLinearTerm(
                    int(term.source_kind),
                    int(term.source_id),
                    float(term.gain),
                )
                for term in self.terms
            )
        )
        self._c_conditions = condition_array_type(*conditions)
        self._c_ops = op_array_type(*c_ops)

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def op_ptr(self):
        return self._c_ops

    @property
    def term_ptr(self):
        return self._c_terms

    @property
    def condition_ptr(self):
        return self._c_conditions


class RustBodyIrBatch:
    """ctypes-backed 094e body IR operation batch."""

    def __init__(
        self,
        *,
        stmt_ops: Iterable[BodyStmtOp],
        expr_ops: Iterable[BodyExprOp],
    ):
        self.stmt_ops: Tuple[BodyStmtOp, ...] = tuple(stmt_ops)
        self.expr_ops: Tuple[BodyExprOp, ...] = tuple(expr_ops)
        stmt_array_type = RustBodyStmtOp * len(self.stmt_ops)
        expr_array_type = RustBodyExprOp * len(self.expr_ops)
        self._c_stmt_ops = stmt_array_type(
            *(
                RustBodyStmtOp(
                    int(op.target_kind),
                    int(bool(op.target_integer)),
                    int(op.target_id),
                    int(op.expr_start),
                    int(op.expr_count),
                )
                for op in self.stmt_ops
            )
        )
        self._c_expr_ops = expr_array_type(
            *(
                RustBodyExprOp(
                    int(op.op_kind),
                    int(op.index),
                    float(op.value),
                )
                for op in self.expr_ops
            )
        )

    def __len__(self) -> int:
        return len(self.stmt_ops)

    @property
    def stmt_ptr(self):
        return self._c_stmt_ops

    @property
    def expr_ptr(self):
        return self._c_expr_ops


class RustBodyExprBatch:
    """ctypes-backed stack expression segment batch for event trigger staging."""

    def __init__(self, expr_segments: Iterable[Iterable[BodyExprOp]]):
        self.expr_segments: Tuple[Tuple[BodyExprOp, ...], ...] = tuple(
            tuple(segment) for segment in expr_segments
        )
        starts = []
        counts = []
        expr_ops = []
        for segment in self.expr_segments:
            starts.append(len(expr_ops))
            counts.append(len(segment))
            expr_ops.extend(segment)
        self.expr_ops: Tuple[BodyExprOp, ...] = tuple(expr_ops)
        self.expr_starts: Tuple[int, ...] = tuple(starts)
        self.expr_counts: Tuple[int, ...] = tuple(counts)

        expr_array_type = RustBodyExprOp * len(self.expr_ops)
        size_array_type = ctypes.c_size_t * len(self.expr_segments)
        self._c_expr_ops = expr_array_type(
            *(
                RustBodyExprOp(
                    int(op.op_kind),
                    int(op.index),
                    float(op.value),
                )
                for op in self.expr_ops
            )
        )
        self._c_expr_starts = size_array_type(*self.expr_starts)
        self._c_expr_counts = size_array_type(*self.expr_counts)

    def __len__(self) -> int:
        return len(self.expr_segments)

    @property
    def expr_ptr(self):
        return self._c_expr_ops

    @property
    def start_ptr(self):
        return self._c_expr_starts

    @property
    def count_ptr(self):
        return self._c_expr_counts


class RustBackend:
    """Loaded Rust dynamic library wrapper."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self._lib = ctypes.CDLL(str(self.library_path))
        fn = self._lib.evas_rust_evaluate_static_affine
        fn.argtypes = [
            ctypes.POINTER(RustStaticAffineOp),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_int
        self._evaluate_static_affine = fn
        linear_fn = self._lib.evas_rust_evaluate_static_linear
        linear_fn.argtypes = [
            ctypes.POINTER(RustLinearOp),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearTerm),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearCondition),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        linear_fn.restype = ctypes.c_int
        self._evaluate_static_linear = linear_fn
        try:
            body_ir_fn = self._lib.evas_rust_evaluate_body_ir
        except AttributeError:
            body_ir_fn = None
        if body_ir_fn is not None:
            body_ir_fn.argtypes = [
                ctypes.POINTER(RustBodyStmtOp),
                ctypes.c_size_t,
                ctypes.POINTER(RustBodyExprOp),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
            ]
            body_ir_fn.restype = ctypes.c_int
        self._evaluate_body_ir = body_ir_fn
        try:
            body_expr_fn = self._lib.evas_rust_evaluate_body_expr
        except AttributeError:
            body_expr_fn = None
        if body_expr_fn is not None:
            body_expr_fn.argtypes = [
                ctypes.POINTER(RustBodyExprOp),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
            ]
            body_expr_fn.restype = ctypes.c_int
        self._evaluate_body_expr = body_expr_fn
        try:
            body_expr_batch_fn = self._lib.evas_rust_evaluate_body_expr_batch
        except AttributeError:
            body_expr_batch_fn = None
        if body_expr_batch_fn is not None:
            body_expr_batch_fn.argtypes = [
                ctypes.POINTER(RustBodyExprOp),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
            ]
            body_expr_batch_fn.restype = ctypes.c_int
        self._evaluate_body_expr_batch = body_expr_batch_fn
        try:
            timer_linear_trace_fn = self._lib.evas_rust_timer_static_linear_trace
        except AttributeError:
            timer_linear_trace_fn = None
        if timer_linear_trace_fn is not None:
            timer_linear_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearOp),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearTerm),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearCondition),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearOp),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearTerm),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearCondition),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            timer_linear_trace_fn.restype = ctypes.c_int
        self._timer_static_linear_trace = timer_linear_trace_fn
        try:
            timer_linear_queue_trace_fn = (
                self._lib.evas_rust_timer_static_linear_queue_trace
            )
        except AttributeError:
            timer_linear_queue_trace_fn = None
        if timer_linear_queue_trace_fn is not None:
            timer_linear_queue_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearOp),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearTerm),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearCondition),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearOp),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearTerm),
                ctypes.c_size_t,
                ctypes.POINTER(RustLinearCondition),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            timer_linear_queue_trace_fn.restype = ctypes.c_int
        self._timer_static_linear_queue_trace = timer_linear_queue_trace_fn
        try:
            lfsr_event_fn = self._lib.evas_rust_event_lfsr_shift_xor_step
        except AttributeError:
            lfsr_event_fn = None
        if lfsr_event_fn is not None:
            lfsr_event_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_uint8),
            ]
            lfsr_event_fn.restype = ctypes.c_int
        self._event_lfsr_shift_xor_step = lfsr_event_fn
        try:
            timer_lfsr_output_fn = self._lib.evas_rust_timer_lfsr_output_step
        except AttributeError:
            timer_lfsr_output_fn = None
        if timer_lfsr_output_fn is not None:
            timer_lfsr_output_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
            ]
            timer_lfsr_output_fn.restype = ctypes.c_int
        self._timer_lfsr_output_step = timer_lfsr_output_fn
        try:
            prbs7_trace_fn = self._lib.evas_rust_prbs7_trace
        except AttributeError:
            prbs7_trace_fn = None
        if prbs7_trace_fn is not None:
            prbs7_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            prbs7_trace_fn.restype = ctypes.c_int
        self._prbs7_trace = prbs7_trace_fn
        try:
            lfsr_transition_trace_fn = self._lib.evas_rust_lfsr_transition_trace
        except AttributeError:
            lfsr_transition_trace_fn = None
        if lfsr_transition_trace_fn is not None:
            lfsr_transition_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int64,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lfsr_transition_trace_fn.restype = ctypes.c_int
        self._lfsr_transition_trace = lfsr_transition_trace_fn
        try:
            gain_timer_reduction_trace_fn = self._lib.evas_rust_gain_timer_reduction_trace
        except AttributeError:
            gain_timer_reduction_trace_fn = None
        if gain_timer_reduction_trace_fn is not None:
            gain_timer_reduction_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            gain_timer_reduction_trace_fn.restype = ctypes.c_int
        self._gain_timer_reduction_trace = gain_timer_reduction_trace_fn
        try:
            gain_measurement_flow_trace_fn = self._lib.evas_rust_gain_measurement_flow_trace
        except AttributeError:
            gain_measurement_flow_trace_fn = None
        if gain_measurement_flow_trace_fn is not None:
            gain_measurement_flow_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            gain_measurement_flow_trace_fn.restype = ctypes.c_int
        self._gain_measurement_flow_trace = gain_measurement_flow_trace_fn
        try:
            cmp_delay_trace_fn = self._lib.evas_rust_cmp_delay_trace
        except AttributeError:
            cmp_delay_trace_fn = None
        if cmp_delay_trace_fn is not None:
            cmp_delay_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            cmp_delay_trace_fn.restype = ctypes.c_int
        self._cmp_delay_trace = cmp_delay_trace_fn
        try:
            sar_loop_trace_fn = self._lib.evas_rust_sar_loop_trace
        except AttributeError:
            sar_loop_trace_fn = None
        if sar_loop_trace_fn is not None:
            sar_loop_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            sar_loop_trace_fn.restype = ctypes.c_int
        self._sar_loop_trace = sar_loop_trace_fn
        try:
            cppll_reacquire_trace_fn = self._lib.evas_rust_cppll_reacquire_trace
        except AttributeError:
            cppll_reacquire_trace_fn = None
        if cppll_reacquire_trace_fn is not None:
            cppll_reacquire_trace_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
            ]
            cppll_reacquire_trace_fn.restype = ctypes.c_int
        self._cppll_reacquire_trace = cppll_reacquire_trace_fn
        transition_target_fn = self._lib.evas_rust_evaluate_transition_targets
        transition_target_fn.argtypes = [
            ctypes.POINTER(RustTransitionTargetOp),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearTerm),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearCondition),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        transition_target_fn.restype = ctypes.c_int
        self._evaluate_transition_targets = transition_target_fn
        ordered_transition_fn = self._lib.evas_rust_evaluate_ordered_transition_segment
        ordered_transition_fn.argtypes = [
            ctypes.POINTER(RustLinearOp),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearTerm),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearCondition),
            ctypes.c_size_t,
            ctypes.POINTER(RustTransitionTargetOp),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearTerm),
            ctypes.c_size_t,
            ctypes.POINTER(RustLinearCondition),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        ordered_transition_fn.restype = ctypes.c_int
        self._evaluate_ordered_transition_segment = ordered_transition_fn
        copy_fn = self._lib.evas_rust_copy_f64
        copy_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        copy_fn.restype = ctypes.c_int
        self._copy_f64 = copy_fn
        err_ratio_fn = self._lib.evas_rust_max_err_ratio
        err_ratio_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        err_ratio_fn.restype = ctypes.c_int
        self._max_err_ratio = err_ratio_fn
        record_values_fn = self._lib.evas_rust_record_values_for_node_ids
        record_values_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        record_values_fn.restype = ctypes.c_int
        self._record_values_for_node_ids = record_values_fn
        try:
            interpolate_event_values_fn = self._lib.evas_rust_interpolate_event_values
        except AttributeError:
            interpolate_event_values_fn = None
        if interpolate_event_values_fn is not None:
            interpolate_event_values_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
            ]
            interpolate_event_values_fn.restype = ctypes.c_int
        self._interpolate_event_values = interpolate_event_values_fn
        transition_bp_fn = self._lib.evas_rust_next_transition_breakpoint
        transition_bp_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
        ]
        transition_bp_fn.restype = ctypes.c_int
        self._next_transition_breakpoint = transition_bp_fn
        try:
            transition_state_fn = self._lib.evas_rust_transition_state_step
        except AttributeError:
            transition_state_fn = None
        if transition_state_fn is not None:
            transition_state_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint8,
            ]
            transition_state_fn.restype = ctypes.c_int
        self._transition_state_step = transition_state_fn
        timer_bp_fn = self._lib.evas_rust_next_timer_breakpoint
        timer_bp_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
        ]
        timer_bp_fn.restype = ctypes.c_int
        self._next_timer_breakpoint = timer_bp_fn
        try:
            timer_periodic_fn = self._lib.evas_rust_timer_periodic_step
        except AttributeError:
            timer_periodic_fn = None
        if timer_periodic_fn is not None:
            timer_periodic_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_double,
                ctypes.c_uint8,
                ctypes.c_double,
            ]
            timer_periodic_fn.restype = ctypes.c_int
        self._timer_periodic_step = timer_periodic_fn
        try:
            timer_absolute_fn = self._lib.evas_rust_timer_absolute_step
        except AttributeError:
            timer_absolute_fn = None
        if timer_absolute_fn is not None:
            timer_absolute_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_double,
                ctypes.c_double,
            ]
            timer_absolute_fn.restype = ctypes.c_int
        self._timer_absolute_step = timer_absolute_fn
        try:
            cross_detector_fn = self._lib.evas_rust_cross_detector_step
        except AttributeError:
            cross_detector_fn = None
        if cross_detector_fn is not None:
            cross_detector_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
            ]
            cross_detector_fn.restype = ctypes.c_int
        self._cross_detector_step = cross_detector_fn
        try:
            above_detector_fn = self._lib.evas_rust_above_detector_step
        except AttributeError:
            above_detector_fn = None
        if above_detector_fn is not None:
            above_detector_fn.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
            ]
            above_detector_fn.restype = ctypes.c_int
        self._above_detector_step = above_detector_fn
        try:
            dynamic_bus_offsets_fn = self._lib.evas_rust_dynamic_bus_offsets
        except AttributeError:
            dynamic_bus_offsets_fn = None
        if dynamic_bus_offsets_fn is not None:
            dynamic_bus_offsets_fn.argtypes = [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            dynamic_bus_offsets_fn.restype = ctypes.c_int
        self._dynamic_bus_offsets = dynamic_bus_offsets_fn

    def make_static_affine_batch(
        self,
        ops: Iterable[StaticAffineOp],
    ) -> RustStaticAffineBatch:
        return RustStaticAffineBatch(ops)

    def make_static_linear_batch(
        self,
        ops: Iterable[LinearOp],
    ) -> RustLinearBatch:
        return RustLinearBatch(ops)

    def make_body_ir_batch(
        self,
        *,
        stmt_ops: Iterable[BodyStmtOp],
        expr_ops: Iterable[BodyExprOp],
    ) -> RustBodyIrBatch:
        return RustBodyIrBatch(stmt_ops=stmt_ops, expr_ops=expr_ops)

    def make_body_expr_batch(
        self,
        expr_segments: Iterable[Iterable[BodyExprOp]],
    ) -> RustBodyExprBatch:
        return RustBodyExprBatch(expr_segments)

    def make_transition_target_batch(
        self,
        ops: Iterable[TransitionTargetOp],
    ) -> RustTransitionTargetBatch:
        return RustTransitionTargetBatch(ops)

    def make_lfsr_event_batch(
        self,
        *,
        lfsr_slots: Iterable[int],
        tmp_slots: Iterable[int],
        tap_slots: Iterable[int],
        gate_node_id: Optional[int] = None,
        gate_threshold: float = 0.5,
        high_node_id: Optional[int] = None,
        low_node_id: Optional[int] = None,
        output_state_id: Optional[int] = None,
        output_node_id: Optional[int] = None,
        loop_state_id: Optional[int] = None,
        loop_final_value: float = 0.0,
    ) -> RustLfsrEventBatch:
        return RustLfsrEventBatch(
            lfsr_slots=lfsr_slots,
            tmp_slots=tmp_slots,
            tap_slots=tap_slots,
            gate_node_id=gate_node_id,
            gate_threshold=gate_threshold,
            high_node_id=high_node_id,
            low_node_id=low_node_id,
            output_state_id=output_state_id,
            output_node_id=output_node_id,
            loop_state_id=loop_state_id,
            loop_final_value=loop_final_value,
        )

    def make_node_id_batch(self, node_ids: Iterable[int]) -> RustNodeIdBatch:
        return RustNodeIdBatch(node_ids)

    def timer_static_linear_trace(
        self,
        times: MutableSequence[float],
        *,
        source_node_ids: MutableSequence[int],
        source_values: MutableSequence[float],
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        event_batch: RustLinearBatch,
        evaluate_batch: RustLinearBatch,
        record_node_ids: MutableSequence[int],
        timer_start: float,
        timer_period: float,
        has_start: bool = True,
        eps: float = 1.0e-18,
    ) -> Tuple[array, int]:
        if self._timer_static_linear_trace is None:
            raise RustBackendError(
                "Rust timer static-linear trace is not available in this library"
            )
        point_count = len(times)
        source_count = len(source_node_ids)
        record_count = len(record_node_ids)
        if len(source_values) != point_count * source_count:
            raise RustBackendError("timer trace source matrix has the wrong size")

        times_buffer, _ = self._double_buffer(times)
        source_id_buffer, _ = self._size_t_buffer(source_node_ids)
        source_value_buffer, _ = self._double_buffer(source_values)
        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        record_id_buffer, _ = self._size_t_buffer(record_node_ids)
        out_values = array("d", [0.0]) * (point_count * record_count)
        out_buffer, _ = self._double_buffer(out_values)
        out_event_count = ctypes.c_size_t(0)

        rc = self._timer_static_linear_trace(
            times_buffer,
            point_count,
            source_id_buffer,
            source_count,
            source_value_buffer,
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            event_batch.op_ptr,
            len(event_batch),
            event_batch.term_ptr,
            len(event_batch.terms),
            event_batch.condition_ptr,
            len(event_batch.conditions),
            evaluate_batch.op_ptr,
            len(evaluate_batch),
            evaluate_batch.term_ptr,
            len(evaluate_batch.terms),
            evaluate_batch.condition_ptr,
            len(evaluate_batch.conditions),
            record_id_buffer,
            record_count,
            out_buffer,
            len(out_values),
            float(timer_start),
            float(timer_period),
            1 if bool(has_start) else 0,
            float(eps),
            ctypes.byref(out_event_count),
        )
        if rc != 0:
            raise RustBackendError(f"Rust timer static-linear trace failed with code {rc}")

        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)
        return out_values, int(out_event_count.value)

    def timer_static_linear_queue_trace(
        self,
        times: MutableSequence[float],
        *,
        source_node_ids: MutableSequence[int],
        source_values: MutableSequence[float],
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        timer_starts: MutableSequence[float],
        timer_periods: MutableSequence[float],
        event_op_starts: MutableSequence[int],
        event_op_counts: MutableSequence[int],
        event_batch: RustLinearBatch,
        evaluate_batch: RustLinearBatch,
        record_node_ids: MutableSequence[int],
        eps: float = 1.0e-18,
    ) -> Tuple[array, int]:
        if self._timer_static_linear_queue_trace is None:
            raise RustBackendError(
                "Rust timer static-linear queue trace is not available in this library"
            )
        point_count = len(times)
        source_count = len(source_node_ids)
        record_count = len(record_node_ids)
        timer_count = len(timer_starts)
        if len(source_values) != point_count * source_count:
            raise RustBackendError("timer queue trace source matrix has the wrong size")
        if (
            len(timer_periods) != timer_count
            or len(event_op_starts) != timer_count
            or len(event_op_counts) != timer_count
        ):
            raise RustBackendError("timer queue metadata has mismatched lengths")

        times_buffer, _ = self._double_buffer(times)
        source_id_buffer, _ = self._size_t_buffer(source_node_ids)
        source_value_buffer, _ = self._double_buffer(source_values)
        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        timer_start_buffer, _ = self._double_buffer(timer_starts)
        timer_period_buffer, _ = self._double_buffer(timer_periods)
        event_op_start_buffer, _ = self._size_t_buffer(event_op_starts)
        event_op_count_buffer, _ = self._size_t_buffer(event_op_counts)
        record_id_buffer, _ = self._size_t_buffer(record_node_ids)
        out_values = array("d", [0.0]) * (point_count * record_count)
        out_buffer, _ = self._double_buffer(out_values)
        out_event_count = ctypes.c_size_t(0)

        rc = self._timer_static_linear_queue_trace(
            times_buffer,
            point_count,
            source_id_buffer,
            source_count,
            source_value_buffer,
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            timer_start_buffer,
            timer_period_buffer,
            event_op_start_buffer,
            event_op_count_buffer,
            timer_count,
            event_batch.op_ptr,
            len(event_batch),
            event_batch.term_ptr,
            len(event_batch.terms),
            event_batch.condition_ptr,
            len(event_batch.conditions),
            evaluate_batch.op_ptr,
            len(evaluate_batch),
            evaluate_batch.term_ptr,
            len(evaluate_batch.terms),
            evaluate_batch.condition_ptr,
            len(evaluate_batch.conditions),
            record_id_buffer,
            record_count,
            out_buffer,
            len(out_values),
            float(eps),
            ctypes.byref(out_event_count),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust timer static-linear queue trace failed with code {rc}"
            )

        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)
        return out_values, int(out_event_count.value)

    def copy_f64(
        self,
        source_values: MutableSequence[float],
        target_values: MutableSequence[float],
    ) -> None:
        source_count = len(source_values)
        target_count = len(target_values)
        if source_count != target_count:
            raise RustBackendError(
                f"cannot copy mismatched buffers: {source_count} != {target_count}"
            )
        source_buffer, _ = self._double_buffer(source_values)
        target_buffer, copied_target = self._double_buffer(target_values)

        rc = self._copy_f64(source_buffer, source_count, target_buffer, target_count)
        if rc != 0:
            raise RustBackendError(f"Rust f64 copy failed with code {rc}")

        if copied_target:
            for idx, value in enumerate(target_buffer):
                target_values[idx] = float(value)

    def max_err_ratio(
        self,
        values: MutableSequence[float],
        previous_values: MutableSequence[float],
        node_ids: RustNodeIdBatch,
        reltol: float,
        vabstol: float,
    ) -> float:
        value_count = len(values)
        previous_count = len(previous_values)
        if value_count != previous_count:
            raise RustBackendError(
                f"cannot scan mismatched buffers: {value_count} != {previous_count}"
            )
        value_buffer, _ = self._double_buffer(values)
        previous_buffer, _ = self._double_buffer(previous_values)
        out_ratio = ctypes.c_double(0.0)

        rc = self._max_err_ratio(
            value_buffer,
            value_count,
            previous_buffer,
            previous_count,
            node_ids.ptr,
            len(node_ids),
            float(reltol),
            float(vabstol),
            ctypes.byref(out_ratio),
        )
        if rc != 0:
            raise RustBackendError(f"Rust err_ratio scan failed with code {rc}")
        return float(out_ratio.value)

    def record_values_for_ids(
        self,
        values: MutableSequence[float],
        node_ids: RustNodeIdBatch,
        *,
        default: float = 0.0,
    ) -> array:
        value_count = len(values)
        out_values = array("d", [0.0]) * len(node_ids)
        value_buffer, _ = self._double_buffer(values)
        out_buffer, copied_out = self._double_buffer(out_values)

        rc = self._record_values_for_node_ids(
            value_buffer,
            value_count,
            node_ids.ptr,
            len(node_ids),
            float(default),
            out_buffer,
            len(out_values),
        )
        if rc != 0:
            raise RustBackendError(f"Rust record value scan failed with code {rc}")

        if copied_out:
            for idx, value in enumerate(out_buffer):
                out_values[idx] = float(value)
        return out_values

    def interpolate_event_values(
        self,
        previous_values: MutableSequence[float],
        current_values: MutableSequence[float],
        previous_time: float,
        current_time: float,
        event_time: float,
    ) -> array:
        if self._interpolate_event_values is None:
            raise RustBackendError(
                "Rust event interpolation is not available in this library"
            )
        count = len(previous_values)
        if len(current_values) != count:
            raise RustBackendError("cannot interpolate mismatched event buffers")

        previous_buffer, _ = self._double_buffer(previous_values)
        current_buffer, _ = self._double_buffer(current_values)
        out_values = array("d", [0.0]) * count
        out_buffer, copied_out = self._double_buffer(out_values)

        rc = self._interpolate_event_values(
            previous_buffer,
            count,
            current_buffer,
            len(current_values),
            out_buffer,
            len(out_values),
            float(previous_time),
            float(current_time),
            float(event_time),
        )
        if rc != 0:
            raise RustBackendError(f"Rust event interpolation failed with code {rc}")

        if copied_out:
            for idx, value in enumerate(out_buffer):
                out_values[idx] = float(value)
        return out_values

    def next_transition_breakpoint(
        self,
        start_times: MutableSequence[float],
        start_values: MutableSequence[float],
        target_values: MutableSequence[float],
        delays: MutableSequence[float],
        rise_times: MutableSequence[float],
        fall_times: MutableSequence[float],
        active_flags: MutableSequence[int],
        time: float,
        min_ramp_time: float,
    ) -> Optional[float]:
        count = len(start_times)
        if (
            len(start_values) != count
            or len(target_values) != count
            or len(delays) != count
            or len(rise_times) != count
            or len(fall_times) != count
            or len(active_flags) != count
        ):
            raise RustBackendError("cannot scan mismatched transition buffers")

        start_time_buffer, _ = self._double_buffer(start_times)
        start_value_buffer, _ = self._double_buffer(start_values)
        target_value_buffer, _ = self._double_buffer(target_values)
        delay_buffer, _ = self._double_buffer(delays)
        rise_time_buffer, _ = self._double_buffer(rise_times)
        fall_time_buffer, _ = self._double_buffer(fall_times)
        active_array_type = ctypes.c_uint8 * count
        active_buffer = active_array_type(
            *(1 if bool(active) else 0 for active in active_flags)
        )
        found = ctypes.c_uint8(0)
        out_time = ctypes.c_double(0.0)

        rc = self._next_transition_breakpoint(
            start_time_buffer,
            count,
            start_value_buffer,
            target_value_buffer,
            delay_buffer,
            rise_time_buffer,
            fall_time_buffer,
            active_buffer,
            float(time),
            float(min_ramp_time),
            ctypes.byref(found),
            ctypes.byref(out_time),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust transition breakpoint scan failed with code {rc}"
            )
        if not found.value:
            return None
        return float(out_time.value)

    def transition_state_step(
        self,
        current_values: MutableSequence[float],
        target_values: MutableSequence[float],
        start_times: MutableSequence[float],
        start_values: MutableSequence[float],
        delays: MutableSequence[float],
        rise_times: MutableSequence[float],
        fall_times: MutableSequence[float],
        active_flags: MutableSequence[int],
        initialized_flags: MutableSequence[int],
        input_targets: MutableSequence[float],
        input_delays: MutableSequence[float],
        input_rises: MutableSequence[float],
        input_falls: MutableSequence[float],
        output_values: MutableSequence[float],
        time: float,
        default_transition: float,
        initial_condition_mode: bool = False,
    ) -> None:
        if self._transition_state_step is None:
            raise RustBackendError(
                "Rust transition state step is not available in this library"
            )
        count = len(current_values)
        sequences = (
            target_values,
            start_times,
            start_values,
            delays,
            rise_times,
            fall_times,
            active_flags,
            initialized_flags,
            input_targets,
            input_delays,
            input_rises,
            input_falls,
            output_values,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot step mismatched transition state buffers")

        current_buffer, copied_current = self._double_buffer(current_values)
        target_buffer, copied_target = self._double_buffer(target_values)
        start_time_buffer, copied_start_time = self._double_buffer(start_times)
        start_value_buffer, copied_start_value = self._double_buffer(start_values)
        delay_buffer, copied_delay = self._double_buffer(delays)
        rise_time_buffer, copied_rise_time = self._double_buffer(rise_times)
        fall_time_buffer, copied_fall_time = self._double_buffer(fall_times)
        active_buffer, copied_active = self._uint8_buffer(active_flags)
        initialized_buffer, copied_initialized = self._uint8_buffer(initialized_flags)
        input_target_buffer, _ = self._double_buffer(input_targets)
        input_delay_buffer, _ = self._double_buffer(input_delays)
        input_rise_buffer, _ = self._double_buffer(input_rises)
        input_fall_buffer, _ = self._double_buffer(input_falls)
        output_buffer, copied_output = self._double_buffer(output_values)

        rc = self._transition_state_step(
            current_buffer,
            count,
            target_buffer,
            start_time_buffer,
            start_value_buffer,
            delay_buffer,
            rise_time_buffer,
            fall_time_buffer,
            active_buffer,
            initialized_buffer,
            input_target_buffer,
            input_delay_buffer,
            input_rise_buffer,
            input_fall_buffer,
            output_buffer,
            float(time),
            float(default_transition),
            1 if bool(initial_condition_mode) else 0,
        )
        if rc != 0:
            raise RustBackendError(f"Rust transition state step failed with code {rc}")

        copy_specs = (
            (copied_current, current_buffer, current_values, float),
            (copied_target, target_buffer, target_values, float),
            (copied_start_time, start_time_buffer, start_times, float),
            (copied_start_value, start_value_buffer, start_values, float),
            (copied_delay, delay_buffer, delays, float),
            (copied_rise_time, rise_time_buffer, rise_times, float),
            (copied_fall_time, fall_time_buffer, fall_times, float),
            (copied_active, active_buffer, active_flags, int),
            (copied_initialized, initialized_buffer, initialized_flags, int),
            (copied_output, output_buffer, output_values, float),
        )
        for copied, buffer, values, caster in copy_specs:
            if copied:
                for idx, value in enumerate(buffer):
                    values[idx] = caster(value)

    def next_timer_breakpoint(
        self,
        next_fire_times: MutableSequence[float],
        last_fired_times: MutableSequence[float],
        has_last_fired_flags: MutableSequence[int],
        time: float,
    ) -> Optional[float]:
        count = len(next_fire_times)
        if len(last_fired_times) != count or len(has_last_fired_flags) != count:
            raise RustBackendError("cannot scan mismatched timer buffers")

        next_fire_buffer, _ = self._double_buffer(next_fire_times)
        last_fired_buffer, _ = self._double_buffer(last_fired_times)
        flag_buffer, _ = self._uint8_buffer(has_last_fired_flags)
        found = ctypes.c_uint8(0)
        out_time = ctypes.c_double(0.0)

        rc = self._next_timer_breakpoint(
            next_fire_buffer,
            count,
            last_fired_buffer,
            flag_buffer,
            float(time),
            ctypes.byref(found),
            ctypes.byref(out_time),
        )
        if rc != 0:
            raise RustBackendError(f"Rust timer breakpoint scan failed with code {rc}")
        if not found.value:
            return None
        return float(out_time.value)

    def timer_periodic_step(
        self,
        next_fire_times: MutableSequence[float],
        has_state_flags: MutableSequence[int],
        periods: MutableSequence[float],
        starts: MutableSequence[float],
        has_start_flags: MutableSequence[int],
        due_flags: MutableSequence[int],
        skipped_flags: MutableSequence[int],
        time: float,
        reschedule_on_due: bool = True,
        eps: float = 1.0e-18,
    ) -> None:
        if self._timer_periodic_step is None:
            raise RustBackendError(
                "Rust timer periodic step is not available in this library"
            )
        count = len(next_fire_times)
        sequences = (
            has_state_flags,
            periods,
            starts,
            has_start_flags,
            due_flags,
            skipped_flags,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot step mismatched periodic timer buffers")

        next_fire_buffer, copied_next_fire = self._double_buffer(next_fire_times)
        has_state_buffer, copied_has_state = self._uint8_buffer(has_state_flags)
        period_buffer, _ = self._double_buffer(periods)
        start_buffer, _ = self._double_buffer(starts)
        has_start_buffer, _ = self._uint8_buffer(has_start_flags)
        due_buffer, copied_due = self._uint8_buffer(due_flags)
        skipped_buffer, copied_skipped = self._uint8_buffer(skipped_flags)

        rc = self._timer_periodic_step(
            next_fire_buffer,
            count,
            has_state_buffer,
            period_buffer,
            start_buffer,
            has_start_buffer,
            due_buffer,
            skipped_buffer,
            float(time),
            1 if bool(reschedule_on_due) else 0,
            float(eps),
        )
        if rc != 0:
            raise RustBackendError(f"Rust periodic timer step failed with code {rc}")

        copy_specs = (
            (copied_next_fire, next_fire_buffer, next_fire_times, float),
            (copied_has_state, has_state_buffer, has_state_flags, int),
            (copied_due, due_buffer, due_flags, int),
            (copied_skipped, skipped_buffer, skipped_flags, int),
        )
        for copied, buffer, values, caster in copy_specs:
            if copied:
                for idx, value in enumerate(buffer):
                    values[idx] = caster(value)

    def timer_absolute_step(
        self,
        next_fire_times: MutableSequence[float],
        has_state_flags: MutableSequence[int],
        last_fired_times: MutableSequence[float],
        has_last_fired_flags: MutableSequence[int],
        targets: MutableSequence[float],
        due_flags: MutableSequence[int],
        expired_flags: MutableSequence[int],
        time: float,
        eps: float = 1.0e-18,
    ) -> None:
        if self._timer_absolute_step is None:
            raise RustBackendError(
                "Rust timer absolute step is not available in this library"
            )
        count = len(next_fire_times)
        sequences = (
            has_state_flags,
            last_fired_times,
            has_last_fired_flags,
            targets,
            due_flags,
            expired_flags,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot step mismatched absolute timer buffers")

        next_fire_buffer, copied_next_fire = self._double_buffer(next_fire_times)
        has_state_buffer, copied_has_state = self._uint8_buffer(has_state_flags)
        last_fired_buffer, copied_last_fired = self._double_buffer(last_fired_times)
        has_last_fired_buffer, copied_has_last = self._uint8_buffer(
            has_last_fired_flags
        )
        target_buffer, _ = self._double_buffer(targets)
        due_buffer, copied_due = self._uint8_buffer(due_flags)
        expired_buffer, copied_expired = self._uint8_buffer(expired_flags)

        rc = self._timer_absolute_step(
            next_fire_buffer,
            count,
            has_state_buffer,
            last_fired_buffer,
            has_last_fired_buffer,
            target_buffer,
            due_buffer,
            expired_buffer,
            float(time),
            float(eps),
        )
        if rc != 0:
            raise RustBackendError(f"Rust absolute timer step failed with code {rc}")

        copy_specs = (
            (copied_next_fire, next_fire_buffer, next_fire_times, float),
            (copied_has_state, has_state_buffer, has_state_flags, int),
            (copied_last_fired, last_fired_buffer, last_fired_times, float),
            (copied_has_last, has_last_fired_buffer, has_last_fired_flags, int),
            (copied_due, due_buffer, due_flags, int),
            (copied_expired, expired_buffer, expired_flags, int),
        )
        for copied, buffer, values, caster in copy_specs:
            if copied:
                for idx, value in enumerate(buffer):
                    values[idx] = caster(value)

    def cross_detector_step(
        self,
        prev_values: MutableSequence[float],
        prev_times: MutableSequence[float],
        pprev_values: MutableSequence[float],
        pprev_times: MutableSequence[float],
        initialized_flags: MutableSequence[int],
        directions: MutableSequence[int],
        last_cross_times: MutableSequence[float],
        current_values: MutableSequence[float],
        triggered_flags: MutableSequence[int],
        cross_times: MutableSequence[float],
        trigger_directions: MutableSequence[int],
        went_beyond_flags: MutableSequence[int],
        time: float,
        time_tol: float = 0.0,
        expr_tol: float = 1.0e-12,
    ) -> None:
        if self._cross_detector_step is None:
            raise RustBackendError(
                "Rust cross detector step is not available in this library"
            )
        count = len(prev_values)
        sequences = (
            prev_times,
            pprev_values,
            pprev_times,
            initialized_flags,
            directions,
            last_cross_times,
            current_values,
            triggered_flags,
            cross_times,
            trigger_directions,
            went_beyond_flags,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot step mismatched cross detector buffers")

        prev_value_buffer, copied_prev_values = self._double_buffer(prev_values)
        prev_time_buffer, copied_prev_times = self._double_buffer(prev_times)
        pprev_value_buffer, copied_pprev_values = self._double_buffer(pprev_values)
        pprev_time_buffer, copied_pprev_times = self._double_buffer(pprev_times)
        initialized_buffer, copied_initialized = self._uint8_buffer(initialized_flags)
        direction_buffer, _ = self._int_buffer(directions)
        last_cross_buffer, copied_last_cross = self._double_buffer(last_cross_times)
        current_value_buffer, _ = self._double_buffer(current_values)
        triggered_buffer, copied_triggered = self._uint8_buffer(triggered_flags)
        cross_time_buffer, copied_cross_times = self._double_buffer(cross_times)
        trigger_direction_buffer, copied_trigger_directions = self._int_buffer(
            trigger_directions
        )
        went_beyond_buffer, copied_went_beyond = self._uint8_buffer(went_beyond_flags)

        rc = self._cross_detector_step(
            prev_value_buffer,
            count,
            prev_time_buffer,
            pprev_value_buffer,
            pprev_time_buffer,
            initialized_buffer,
            direction_buffer,
            last_cross_buffer,
            current_value_buffer,
            triggered_buffer,
            cross_time_buffer,
            trigger_direction_buffer,
            went_beyond_buffer,
            float(time),
            float(time_tol),
            float(expr_tol),
        )
        if rc != 0:
            raise RustBackendError(f"Rust cross detector step failed with code {rc}")

        copy_specs = (
            (copied_prev_values, prev_value_buffer, prev_values, float),
            (copied_prev_times, prev_time_buffer, prev_times, float),
            (copied_pprev_values, pprev_value_buffer, pprev_values, float),
            (copied_pprev_times, pprev_time_buffer, pprev_times, float),
            (copied_initialized, initialized_buffer, initialized_flags, int),
            (copied_last_cross, last_cross_buffer, last_cross_times, float),
            (copied_triggered, triggered_buffer, triggered_flags, int),
            (copied_cross_times, cross_time_buffer, cross_times, float),
            (
                copied_trigger_directions,
                trigger_direction_buffer,
                trigger_directions,
                int,
            ),
            (copied_went_beyond, went_beyond_buffer, went_beyond_flags, int),
        )
        for copied, buffer, values, caster in copy_specs:
            if copied:
                for idx, value in enumerate(buffer):
                    values[idx] = caster(value)

    def above_detector_step(
        self,
        prev_values: MutableSequence[float],
        prev_times: MutableSequence[float],
        pprev_values: MutableSequence[float],
        pprev_times: MutableSequence[float],
        initialized_flags: MutableSequence[int],
        directions: MutableSequence[int],
        current_values: MutableSequence[float],
        triggered_flags: MutableSequence[int],
        cross_times: MutableSequence[float],
        time: float,
    ) -> None:
        if self._above_detector_step is None:
            raise RustBackendError(
                "Rust above detector step is not available in this library"
            )
        count = len(prev_values)
        sequences = (
            prev_times,
            pprev_values,
            pprev_times,
            initialized_flags,
            directions,
            current_values,
            triggered_flags,
            cross_times,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot step mismatched above detector buffers")

        prev_value_buffer, copied_prev_values = self._double_buffer(prev_values)
        prev_time_buffer, copied_prev_times = self._double_buffer(prev_times)
        pprev_value_buffer, copied_pprev_values = self._double_buffer(pprev_values)
        pprev_time_buffer, copied_pprev_times = self._double_buffer(pprev_times)
        initialized_buffer, copied_initialized = self._uint8_buffer(initialized_flags)
        direction_buffer, _ = self._int_buffer(directions)
        current_value_buffer, _ = self._double_buffer(current_values)
        triggered_buffer, copied_triggered = self._uint8_buffer(triggered_flags)
        cross_time_buffer, copied_cross_times = self._double_buffer(cross_times)

        rc = self._above_detector_step(
            prev_value_buffer,
            count,
            prev_time_buffer,
            pprev_value_buffer,
            pprev_time_buffer,
            initialized_buffer,
            direction_buffer,
            current_value_buffer,
            triggered_buffer,
            cross_time_buffer,
            float(time),
        )
        if rc != 0:
            raise RustBackendError(f"Rust above detector step failed with code {rc}")

        copy_specs = (
            (copied_prev_values, prev_value_buffer, prev_values, float),
            (copied_prev_times, prev_time_buffer, prev_times, float),
            (copied_pprev_values, pprev_value_buffer, pprev_values, float),
            (copied_pprev_times, pprev_time_buffer, pprev_times, float),
            (copied_initialized, initialized_buffer, initialized_flags, int),
            (copied_triggered, triggered_buffer, triggered_flags, int),
            (copied_cross_times, cross_time_buffer, cross_times, float),
        )
        for copied, buffer, values, caster in copy_specs:
            if copied:
                for idx, value in enumerate(buffer):
                    values[idx] = caster(value)

    def dynamic_bus_offsets(
        self,
        base_offsets: MutableSequence[int],
        outer_lengths: MutableSequence[int],
        inner_strides: MutableSequence[int],
        inner_lengths: MutableSequence[int],
        first_indices: MutableSequence[int],
        second_indices: MutableSequence[int],
        has_second_index_flags: MutableSequence[int],
        out_node_ids: MutableSequence[int],
    ) -> None:
        if self._dynamic_bus_offsets is None:
            raise RustBackendError(
                "Rust dynamic bus offsets are not available in this library"
            )
        count = len(base_offsets)
        sequences = (
            outer_lengths,
            inner_strides,
            inner_lengths,
            first_indices,
            second_indices,
            has_second_index_flags,
            out_node_ids,
        )
        if any(len(values) != count for values in sequences):
            raise RustBackendError("cannot resolve mismatched dynamic bus buffers")

        base_buffer, _ = self._size_t_buffer(base_offsets)
        outer_buffer, _ = self._size_t_buffer(outer_lengths)
        stride_buffer, _ = self._size_t_buffer(inner_strides)
        inner_buffer, _ = self._size_t_buffer(inner_lengths)
        first_buffer, _ = self._int64_buffer(first_indices)
        second_buffer, _ = self._int64_buffer(second_indices)
        has_second_buffer, _ = self._uint8_buffer(has_second_index_flags)
        out_buffer, copied_out = self._size_t_buffer(out_node_ids)

        rc = self._dynamic_bus_offsets(
            base_buffer,
            count,
            outer_buffer,
            stride_buffer,
            inner_buffer,
            first_buffer,
            second_buffer,
            has_second_buffer,
            out_buffer,
        )
        if rc != 0:
            raise RustBackendError(f"Rust dynamic bus offset failed with code {rc}")

        if copied_out:
            for idx, value in enumerate(out_buffer):
                out_node_ids[idx] = int(value)

    def evaluate_static_affine(
        self,
        batch: RustStaticAffineBatch,
        values: MutableSequence[float],
    ) -> None:
        if len(batch) == 0:
            return
        value_count = len(values)
        if value_count == 0:
            raise RustBackendError("cannot evaluate Rust static affine batch on empty values")

        try:
            value_buffer = (ctypes.c_double * value_count).from_buffer(values)
            copied = False
        except TypeError:
            # Tests and defensive callers may pass plain lists.  Production
            # opt-in code converts the indexed voltage array to array('d') so
            # this copy path is not used in the hot loop.
            value_buffer = (ctypes.c_double * value_count)(
                *(float(value) for value in values)
            )
            copied = True

        rc = self._evaluate_static_affine(batch.ptr, len(batch), value_buffer, value_count)
        if rc != 0:
            raise RustBackendError(f"Rust static affine evaluation failed with code {rc}")

        if copied:
            for idx, value in enumerate(value_buffer):
                values[idx] = float(value)

    def evaluate_static_linear(
        self,
        batch: RustLinearBatch,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
    ) -> None:
        if len(batch) == 0:
            return
        node_count = len(node_values)
        if node_count == 0:
            raise RustBackendError("cannot evaluate Rust static linear batch on empty node values")
        state_values = state_values if state_values is not None else []
        state_count = len(state_values)

        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)

        rc = self._evaluate_static_linear(
            batch.op_ptr,
            len(batch),
            batch.term_ptr,
            len(batch.terms),
            batch.condition_ptr,
            len(batch.conditions),
            node_buffer,
            node_count,
            state_buffer,
            state_count,
        )
        if rc != 0:
            raise RustBackendError(f"Rust static linear evaluation failed with code {rc}")

        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)

    def evaluate_body_ir(
        self,
        batch: RustBodyIrBatch,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
    ) -> None:
        if self._evaluate_body_ir is None:
            raise RustBackendError("Rust body IR evaluation batch is unavailable")
        if len(batch) == 0:
            return
        state_values = state_values if state_values is not None else []
        param_values = param_values if param_values is not None else []

        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        param_buffer, _ = self._double_buffer(param_values)

        rc = self._evaluate_body_ir(
            batch.stmt_ptr,
            len(batch),
            batch.expr_ptr,
            len(batch.expr_ops),
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            param_buffer,
            len(param_values),
        )
        if rc != 0:
            raise RustBackendError(f"Rust body IR evaluation failed with code {rc}")

        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)

    def evaluate_body_expr(
        self,
        expr_ops: Iterable[BodyExprOp],
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
    ) -> float:
        if self._evaluate_body_expr is None:
            raise RustBackendError("Rust body expression evaluation is unavailable")
        expr_ops = tuple(expr_ops)
        state_values = state_values if state_values is not None else []
        param_values = param_values if param_values is not None else []

        expr_array_type = RustBodyExprOp * len(expr_ops)
        expr_buffer = expr_array_type(
            *(
                RustBodyExprOp(
                    int(op.op_kind),
                    int(op.index),
                    float(op.value),
                )
                for op in expr_ops
            )
        )
        node_buffer, _ = self._double_buffer(node_values)
        state_buffer, _ = self._double_buffer(state_values)
        param_buffer, _ = self._double_buffer(param_values)
        out_value = ctypes.c_double(0.0)

        rc = self._evaluate_body_expr(
            expr_buffer,
            len(expr_ops),
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            param_buffer,
            len(param_values),
            ctypes.byref(out_value),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust body expression evaluation failed with code {rc}"
            )
        return float(out_value.value)

    def evaluate_body_expr_batch(
        self,
        batch: RustBodyExprBatch,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]] = None,
        param_values: Optional[MutableSequence[float]] = None,
    ) -> Tuple[float, ...]:
        if self._evaluate_body_expr_batch is None:
            raise RustBackendError("Rust body expression batch evaluation is unavailable")
        if len(batch) == 0:
            return ()
        state_values = state_values if state_values is not None else []
        param_values = param_values if param_values is not None else []

        node_buffer, _ = self._double_buffer(node_values)
        state_buffer, _ = self._double_buffer(state_values)
        param_buffer, _ = self._double_buffer(param_values)
        out_array_type = ctypes.c_double * len(batch)
        out_buffer = out_array_type()

        rc = self._evaluate_body_expr_batch(
            batch.expr_ptr,
            len(batch.expr_ops),
            batch.start_ptr,
            len(batch),
            batch.count_ptr,
            len(batch),
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            param_buffer,
            len(param_values),
            out_buffer,
            len(batch),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust body expression batch evaluation failed with code {rc}"
            )
        return tuple(float(value) for value in out_buffer)

    def event_lfsr_shift_xor_step(
        self,
        batch: RustLfsrEventBatch,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
    ) -> bool:
        if self._event_lfsr_shift_xor_step is None:
            raise RustBackendError("Rust LFSR event write batch is unavailable")
        if len(batch) == 0:
            return False
        node_buffer, _ = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        executed = ctypes.c_uint8(0)
        rc = self._event_lfsr_shift_xor_step(
            state_buffer,
            len(state_values),
            node_buffer,
            len(node_values),
            batch.lfsr_ptr,
            len(batch.lfsr_slots),
            batch.tmp_ptr,
            len(batch.tmp_slots),
            batch.tap_ptr,
            len(batch.tap_slots),
            int(batch.gate_node_id),
            float(batch.gate_threshold),
            int(batch.high_node_id),
            int(batch.low_node_id),
            int(batch.output_state_id),
            int(batch.loop_state_id),
            float(batch.loop_final_value),
            ctypes.byref(executed),
        )
        if rc != 0:
            raise RustBackendError(f"Rust LFSR event write batch failed with code {rc}")
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)
        return bool(executed.value)

    def timer_lfsr_output_step(
        self,
        batch: RustLfsrEventBatch,
        node_values: MutableSequence[float],
        state_values: MutableSequence[float],
        next_fire_times: MutableSequence[float],
        has_state_flags: MutableSequence[int],
        period: float,
        start: float = 0.0,
        has_start: bool = False,
        time: float = 0.0,
        eps: float = 1.0e-18,
    ) -> Tuple[bool, bool, bool, bool]:
        if self._timer_lfsr_output_step is None:
            raise RustBackendError("Rust timer/LFSR/output batch is unavailable")
        if len(batch) == 0:
            return False, False, False, False
        if len(next_fire_times) != 1 or len(has_state_flags) != 1:
            raise RustBackendError("timer/LFSR/output batch expects one timer slot")

        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        next_fire_buffer, copied_next_fire = self._double_buffer(next_fire_times)
        has_state_buffer, copied_has_state = self._uint8_buffer(has_state_flags)
        due = ctypes.c_uint8(0)
        skipped = ctypes.c_uint8(0)
        executed = ctypes.c_uint8(0)
        output_written = ctypes.c_uint8(0)

        rc = self._timer_lfsr_output_step(
            state_buffer,
            len(state_values),
            node_buffer,
            len(node_values),
            next_fire_buffer,
            has_state_buffer,
            float(period),
            float(start),
            1 if bool(has_start) else 0,
            float(time),
            float(eps),
            batch.lfsr_ptr,
            len(batch.lfsr_slots),
            batch.tmp_ptr,
            len(batch.tmp_slots),
            batch.tap_ptr,
            len(batch.tap_slots),
            int(batch.gate_node_id),
            float(batch.gate_threshold),
            int(batch.high_node_id),
            int(batch.low_node_id),
            int(batch.output_state_id),
            int(batch.output_node_id),
            int(batch.loop_state_id),
            float(batch.loop_final_value),
            ctypes.byref(due),
            ctypes.byref(skipped),
            ctypes.byref(executed),
            ctypes.byref(output_written),
        )
        if rc != 0:
            raise RustBackendError(f"Rust timer/LFSR/output batch failed with code {rc}")
        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)
        if copied_next_fire:
            next_fire_times[0] = float(next_fire_buffer[0])
        if copied_has_state:
            has_state_flags[0] = int(has_state_buffer[0])
        return (
            bool(due.value),
            bool(skipped.value),
            bool(executed.value),
            bool(output_written.value),
        )

    def prbs7_trace(
        self,
        times: MutableSequence[float],
        *,
        clk: Mapping[str, object],
        rst_n: Mapping[str, object],
        en_voltage: float,
        vdd: float,
        vth: float,
        trf: float,
        td: float,
        seed: int,
    ) -> Tuple[array, int]:
        if self._prbs7_trace is None:
            raise RustBackendError("Rust PRBS7 full-model trace is unavailable")
        signal_count = 11
        point_count = len(times)
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        event_count = ctypes.c_size_t(0)

        def pulse_args(meta: Mapping[str, object]):
            return (
                float(meta.get("v_lo", 0.0)),
                float(meta.get("v_hi", 0.0)),
                float(meta.get("period", 0.0)),
                float(meta.get("duty", 0.5)),
                float(meta.get("rise", 0.0)),
                float(meta.get("fall", 0.0)),
                float(meta.get("delay", 0.0)),
                float(meta.get("width", 0.0) or 0.0),
                1 if bool(meta.get("has_width", False)) else 0,
            )

        clk_args = pulse_args(clk)
        rst_args = pulse_args(rst_n)
        rc = self._prbs7_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            *clk_args,
            *rst_args,
            float(en_voltage),
            float(vdd),
            float(vth),
            float(trf),
            float(td),
            int(seed),
            ctypes.byref(event_count),
        )
        if rc != 0:
            raise RustBackendError(f"Rust PRBS7 full-model trace failed with code {rc}")
        return values, int(event_count.value)

    def lfsr_transition_trace(
        self,
        times: MutableSequence[float],
        *,
        clk: Mapping[str, object],
        rst_n: Mapping[str, object],
        en_voltage: float,
        vdd: float,
        vth: float,
        trf: float,
        td: float,
        seed: int,
        width: int,
        taps: Iterable[int],
        shift_sources: Iterable[int],
        output_bits: Iterable[int],
        zero_guard_index: int = -1,
    ) -> Tuple[array, int]:
        if self._lfsr_transition_trace is None:
            raise RustBackendError("Rust LFSR transition trace is unavailable")
        tap_values = tuple(int(tap) for tap in taps)
        shift_values = tuple(int(source) for source in shift_sources)
        output_bit_values = tuple(int(bit) for bit in output_bits)
        width_i = int(width)
        if width_i <= 0:
            raise RustBackendError("LFSR trace width must be positive")
        if len(shift_values) != width_i:
            raise RustBackendError("LFSR shift source count must match width")
        if not tap_values:
            raise RustBackendError("LFSR trace requires at least one tap")
        if not output_bit_values:
            raise RustBackendError("LFSR trace requires at least one output bit")

        signal_count = 3 + len(output_bit_values)
        point_count = len(times)
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        tap_array_type = ctypes.c_size_t * len(tap_values)
        shift_array_type = ctypes.c_int32 * len(shift_values)
        output_array_type = ctypes.c_size_t * len(output_bit_values)
        tap_buffer = tap_array_type(*tap_values)
        shift_buffer = shift_array_type(*shift_values)
        output_buffer = output_array_type(*output_bit_values)
        event_count = ctypes.c_size_t(0)

        def pulse_args(meta: Mapping[str, object]):
            return (
                float(meta.get("v_lo", 0.0)),
                float(meta.get("v_hi", 0.0)),
                float(meta.get("period", 0.0)),
                float(meta.get("duty", 0.5)),
                float(meta.get("rise", 0.0)),
                float(meta.get("fall", 0.0)),
                float(meta.get("delay", 0.0)),
                float(meta.get("width", 0.0) or 0.0),
                1 if bool(meta.get("has_width", False)) else 0,
            )

        clk_args = pulse_args(clk)
        rst_args = pulse_args(rst_n)
        rc = self._lfsr_transition_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            *clk_args,
            *rst_args,
            float(en_voltage),
            float(vdd),
            float(vth),
            float(trf),
            float(td),
            int(seed),
            width_i,
            tap_buffer,
            len(tap_values),
            shift_buffer,
            len(shift_values),
            output_buffer,
            len(output_bit_values),
            int(zero_guard_index),
            ctypes.byref(event_count),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust LFSR transition trace failed with code {rc}"
            )
        return values, int(event_count.value)

    def gain_timer_reduction_trace(
        self,
        times: MutableSequence[float],
        sample_times: MutableSequence[float],
        *,
        point_vdd: MutableSequence[float],
        point_vss: MutableSequence[float],
        point_vinp: MutableSequence[float],
        point_vinn: MutableSequence[float],
        point_voutp: MutableSequence[float],
        point_voutn: MutableSequence[float],
        sample_vdd: MutableSequence[float],
        sample_vss: MutableSequence[float],
        sample_vinp: MutableSequence[float],
        sample_vinn: MutableSequence[float],
        sample_voutp: MutableSequence[float],
        sample_voutn: MutableSequence[float],
        start_time: float,
        gain_scale: float,
        min_input_span: float,
        tedge: float,
    ) -> Tuple[array, int]:
        if self._gain_timer_reduction_trace is None:
            raise RustBackendError("Rust gain timer reduction trace is unavailable")
        point_count = len(times)
        sample_count = len(sample_times)
        for values in (
            point_vdd,
            point_vss,
            point_vinp,
            point_vinn,
            point_voutp,
            point_voutn,
        ):
            if len(values) != point_count:
                raise RustBackendError("gain trace point arrays must match times")
        for values in (
            sample_vdd,
            sample_vss,
            sample_vinp,
            sample_vinn,
            sample_voutp,
            sample_voutn,
        ):
            if len(values) != sample_count:
                raise RustBackendError("gain trace sample arrays must match sample_times")

        signal_count = 8
        times_buffer, _ = self._double_buffer(times)
        sample_time_buffer, _ = self._double_buffer(sample_times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        point_buffers = [
            self._double_buffer(items)[0]
            for items in (
                point_vdd,
                point_vss,
                point_vinp,
                point_vinn,
                point_voutp,
                point_voutn,
            )
        ]
        sample_buffers = [
            self._double_buffer(items)[0]
            for items in (
                sample_vdd,
                sample_vss,
                sample_vinp,
                sample_vinn,
                sample_voutp,
                sample_voutn,
            )
        ]
        sample_events = ctypes.c_size_t(0)
        rc = self._gain_timer_reduction_trace(
            times_buffer,
            point_count,
            sample_time_buffer,
            sample_count,
            value_buffer,
            len(values),
            signal_count,
            *point_buffers,
            *sample_buffers,
            float(start_time),
            float(gain_scale),
            float(min_input_span),
            float(tedge),
            ctypes.byref(sample_events),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust gain timer reduction trace failed with code {rc}"
            )
        return values, int(sample_events.value)

    def gain_measurement_flow_trace(
        self,
        times: MutableSequence[float],
        *,
        vin_event_times: MutableSequence[float],
        vin_event_vinp: MutableSequence[float],
        vin_event_vinn: MutableSequence[float],
        lfsr_event_times: MutableSequence[float],
        vcm: float,
        vth: float,
        dither_amp: float,
        actual_gain: float,
        vin_transition: float,
        lfsr_transition: float,
        vdd: float,
        vss: float,
        lfsr_seed: int,
    ) -> Tuple[array, int, int]:
        if self._gain_measurement_flow_trace is None:
            raise RustBackendError("Rust gain measurement flow trace is unavailable")
        point_count = len(times)
        vin_event_count = len(vin_event_times)
        lfsr_event_count = len(lfsr_event_times)
        if len(vin_event_vinp) != vin_event_count or len(vin_event_vinn) != vin_event_count:
            raise RustBackendError("gain measurement vin event arrays must match")

        signal_count = 4
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        vin_time_buffer, _ = self._double_buffer(vin_event_times)
        vin_vinp_buffer, _ = self._double_buffer(vin_event_vinp)
        vin_vinn_buffer, _ = self._double_buffer(vin_event_vinn)
        lfsr_time_buffer, _ = self._double_buffer(lfsr_event_times)
        vin_events = ctypes.c_size_t(0)
        lfsr_events = ctypes.c_size_t(0)
        rc = self._gain_measurement_flow_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            vin_time_buffer,
            vin_event_count,
            vin_vinp_buffer,
            vin_vinn_buffer,
            lfsr_time_buffer,
            lfsr_event_count,
            float(vcm),
            float(vth),
            float(dither_amp),
            float(actual_gain),
            float(vin_transition),
            float(lfsr_transition),
            float(vdd),
            float(vss),
            int(lfsr_seed),
            ctypes.byref(vin_events),
            ctypes.byref(lfsr_events),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust gain measurement flow trace failed with code {rc}"
            )
        return values, int(vin_events.value), int(lfsr_events.value)

    def cmp_delay_trace(
        self,
        times: MutableSequence[float],
        *,
        point_clk: MutableSequence[float],
        point_vinn: MutableSequence[float],
        point_vinp: MutableSequence[float],
        point_vdd: MutableSequence[float],
        voffset: float,
        tau: float,
        td0: float,
        td_min: float,
        td_max: float,
        tedge: float,
        edge_vth: float,
    ) -> Tuple[array, int]:
        if self._cmp_delay_trace is None:
            raise RustBackendError("Rust comparator delay trace is unavailable")
        point_count = len(times)
        for values in (point_clk, point_vinn, point_vinp, point_vdd):
            if len(values) != point_count:
                raise RustBackendError("cmp-delay trace point arrays must match times")

        signal_count = 6
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        point_buffers = [
            self._double_buffer(items)[0]
            for items in (point_clk, point_vinn, point_vinp, point_vdd)
        ]
        clock_events = ctypes.c_size_t(0)
        rc = self._cmp_delay_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            *point_buffers,
            float(voffset),
            float(tau),
            float(td0),
            float(td_min),
            float(td_max),
            float(tedge),
            float(edge_vth),
            ctypes.byref(clock_events),
        )
        if rc != 0:
            raise RustBackendError(f"Rust comparator delay trace failed with code {rc}")
        return values, int(clock_events.value)

    def sar_loop_trace(
        self,
        times: MutableSequence[float],
        *,
        point_vin: MutableSequence[float],
        point_clk: MutableSequence[float],
        point_rst: MutableSequence[float],
        vdd: float,
        vth: float,
        sh_tr: float,
        default_tr: float,
        width: int,
    ) -> Tuple[array, int]:
        if self._sar_loop_trace is None:
            raise RustBackendError("Rust SAR loop trace is unavailable")
        point_count = len(times)
        width = int(width)
        if width <= 0:
            raise RustBackendError("SAR loop width must be positive")
        for values in (point_vin, point_clk, point_rst):
            if len(values) != point_count:
                raise RustBackendError("SAR loop point arrays must match times")

        signal_count = 11 + width
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        point_buffers = [
            self._double_buffer(items)[0]
            for items in (point_vin, point_clk, point_rst)
        ]
        clock_events = ctypes.c_size_t(0)
        rc = self._sar_loop_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            *point_buffers,
            float(vdd),
            float(vth),
            float(sh_tr),
            float(default_tr),
            width,
            ctypes.byref(clock_events),
        )
        if rc != 0:
            raise RustBackendError(f"Rust SAR loop trace failed with code {rc}")
        return values, int(clock_events.value)

    def cppll_reacquire_trace(
        self,
        times: MutableSequence[float],
        *,
        ref_events: tuple[MutableSequence[float], MutableSequence[float]],
        dco_events: tuple[MutableSequence[float], MutableSequence[float]],
        fb_events: tuple[MutableSequence[float], MutableSequence[float]],
        lock_events: tuple[MutableSequence[float], MutableSequence[float]],
        vctrl_events: tuple[MutableSequence[float], MutableSequence[float]],
        vh: float,
        vl: float,
        ref_tedge: float,
        pll_tedge: float,
    ) -> array:
        if self._cppll_reacquire_trace is None:
            raise RustBackendError("Rust CPPLL reacquire trace is unavailable")
        point_count = len(times)
        event_pairs = (ref_events, dco_events, fb_events, lock_events, vctrl_events)
        for event_times, event_values in event_pairs:
            if len(event_times) != len(event_values):
                raise RustBackendError("CPPLL event time/value arrays must match")

        signal_count = 7
        times_buffer, _ = self._double_buffer(times)
        values = array("d", [0.0]) * (point_count * signal_count)
        value_buffer, _ = self._double_buffer(values)
        event_buffers = []
        for event_times, event_values in event_pairs:
            time_buffer, _ = self._double_buffer(event_times)
            value_event_buffer, _ = self._double_buffer(event_values)
            event_buffers.append((time_buffer, value_event_buffer, len(event_times)))

        rc = self._cppll_reacquire_trace(
            times_buffer,
            point_count,
            value_buffer,
            len(values),
            signal_count,
            event_buffers[0][0],
            event_buffers[0][1],
            event_buffers[0][2],
            event_buffers[1][0],
            event_buffers[1][1],
            event_buffers[1][2],
            event_buffers[2][0],
            event_buffers[2][1],
            event_buffers[2][2],
            event_buffers[3][0],
            event_buffers[3][1],
            event_buffers[3][2],
            event_buffers[4][0],
            event_buffers[4][1],
            event_buffers[4][2],
            float(vh),
            float(vl),
            float(ref_tedge),
            float(pll_tedge),
        )
        if rc != 0:
            raise RustBackendError(f"Rust CPPLL reacquire trace failed with code {rc}")
        return values

    def evaluate_transition_targets(
        self,
        batch: RustTransitionTargetBatch,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]],
        target_values: MutableSequence[float],
        delay_values: MutableSequence[float],
        rise_values: MutableSequence[float],
        fall_values: MutableSequence[float],
    ) -> None:
        if len(batch) == 0:
            return
        state_values = state_values if state_values is not None else []
        count = len(target_values)
        if (
            len(delay_values) != count
            or len(rise_values) != count
            or len(fall_values) != count
        ):
            raise RustBackendError("cannot evaluate transition targets on mismatched buffers")

        node_buffer, _ = self._double_buffer(node_values)
        state_buffer, _ = self._double_buffer(state_values)
        target_buffer, copied_targets = self._double_buffer(target_values)
        delay_buffer, copied_delays = self._double_buffer(delay_values)
        rise_buffer, copied_rises = self._double_buffer(rise_values)
        fall_buffer, copied_falls = self._double_buffer(fall_values)

        rc = self._evaluate_transition_targets(
            batch.op_ptr,
            len(batch),
            batch.term_ptr,
            len(batch.terms),
            batch.condition_ptr,
            len(batch.conditions),
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            target_buffer,
            len(target_values),
            delay_buffer,
            len(delay_values),
            rise_buffer,
            len(rise_values),
            fall_buffer,
            len(fall_values),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust transition target evaluation failed with code {rc}"
            )

        if copied_targets:
            for idx, value in enumerate(target_buffer):
                target_values[idx] = float(value)
        if copied_delays:
            for idx, value in enumerate(delay_buffer):
                delay_values[idx] = float(value)
        if copied_rises:
            for idx, value in enumerate(rise_buffer):
                rise_values[idx] = float(value)
        if copied_falls:
            for idx, value in enumerate(fall_buffer):
                fall_values[idx] = float(value)

    def evaluate_ordered_transition_segment(
        self,
        linear_batch: RustLinearBatch,
        transition_batch: RustTransitionTargetBatch,
        node_values: MutableSequence[float],
        state_values: Optional[MutableSequence[float]],
        target_values: MutableSequence[float],
        delay_values: MutableSequence[float],
        rise_values: MutableSequence[float],
        fall_values: MutableSequence[float],
    ) -> None:
        if len(transition_batch) == 0:
            return
        state_values = state_values if state_values is not None else []
        count = len(target_values)
        if (
            len(delay_values) != count
            or len(rise_values) != count
            or len(fall_values) != count
        ):
            raise RustBackendError(
                "cannot evaluate ordered transition segment on mismatched buffers"
            )

        node_buffer, copied_nodes = self._double_buffer(node_values)
        state_buffer, copied_states = self._double_buffer(state_values)
        target_buffer, copied_targets = self._double_buffer(target_values)
        delay_buffer, copied_delays = self._double_buffer(delay_values)
        rise_buffer, copied_rises = self._double_buffer(rise_values)
        fall_buffer, copied_falls = self._double_buffer(fall_values)

        rc = self._evaluate_ordered_transition_segment(
            linear_batch.op_ptr,
            len(linear_batch),
            linear_batch.term_ptr,
            len(linear_batch.terms),
            linear_batch.condition_ptr,
            len(linear_batch.conditions),
            transition_batch.op_ptr,
            len(transition_batch),
            transition_batch.term_ptr,
            len(transition_batch.terms),
            transition_batch.condition_ptr,
            len(transition_batch.conditions),
            node_buffer,
            len(node_values),
            state_buffer,
            len(state_values),
            target_buffer,
            len(target_values),
            delay_buffer,
            len(delay_values),
            rise_buffer,
            len(rise_values),
            fall_buffer,
            len(fall_values),
        )
        if rc != 0:
            raise RustBackendError(
                f"Rust ordered transition segment evaluation failed with code {rc}"
            )

        if copied_nodes:
            for idx, value in enumerate(node_buffer):
                node_values[idx] = float(value)
        if copied_states:
            for idx, value in enumerate(state_buffer):
                state_values[idx] = float(value)
        if copied_targets:
            for idx, value in enumerate(target_buffer):
                target_values[idx] = float(value)
        if copied_delays:
            for idx, value in enumerate(delay_buffer):
                delay_values[idx] = float(value)
        if copied_rises:
            for idx, value in enumerate(rise_buffer):
                rise_values[idx] = float(value)
        if copied_falls:
            for idx, value in enumerate(fall_buffer):
                fall_values[idx] = float(value)

    @staticmethod
    def _double_buffer(values: MutableSequence[float]):
        value_count = len(values)
        try:
            return (ctypes.c_double * value_count).from_buffer(values), False
        except TypeError:
            return (
                (ctypes.c_double * value_count)(*(float(value) for value in values)),
                True,
            )

    @staticmethod
    def _uint8_buffer(values: MutableSequence[int]):
        value_count = len(values)
        try:
            return (ctypes.c_uint8 * value_count).from_buffer(values), False
        except TypeError:
            return (
                (ctypes.c_uint8 * value_count)(
                    *(1 if bool(value) else 0 for value in values)
                ),
                True,
            )

    @staticmethod
    def _int_buffer(values: MutableSequence[int]):
        value_count = len(values)
        try:
            return (ctypes.c_int * value_count).from_buffer(values), False
        except TypeError:
            return (
                (ctypes.c_int * value_count)(*(int(value) for value in values)),
                True,
            )

    @staticmethod
    def _int64_buffer(values: MutableSequence[int]):
        value_count = len(values)
        try:
            return (ctypes.c_int64 * value_count).from_buffer(values), False
        except TypeError:
            return (
                (ctypes.c_int64 * value_count)(*(int(value) for value in values)),
                True,
            )

    @staticmethod
    def _size_t_buffer(values: MutableSequence[int]):
        value_count = len(values)
        try:
            return (ctypes.c_size_t * value_count).from_buffer(values), False
        except TypeError:
            return (
                (ctypes.c_size_t * value_count)(*(int(value) for value in values)),
                True,
            )


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libevas_rust_core.dylib"
    if os.name == "nt":
        return "evas_rust_core.dll"
    return "libevas_rust_core.so"


def default_rust_core_library_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rust_core"
        / "target"
        / "release"
        / _library_filename()
    )


def load_rust_backend(path: Optional[Union[os.PathLike[str], str]] = None) -> RustBackend:
    candidate = Path(path) if path is not None else Path(
        os.environ.get("EVAS_RUST_CORE_LIB", default_rust_core_library_path())
    )
    if not candidate.exists():
        raise RustBackendError(f"Rust backend library not found: {candidate}")
    return RustBackend(candidate)


def load_optional_rust_backend(
    path: Optional[Union[os.PathLike[str], str]] = None,
) -> Optional[RustBackend]:
    try:
        return load_rust_backend(path)
    except (OSError, RustBackendError):
        return None
