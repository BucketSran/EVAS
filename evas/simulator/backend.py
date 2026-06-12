"""
compiler_backend.py — Compile Verilog-A AST into executable Python model classes.

Takes a parsed Module AST and generates a Python class that implements
the behavioral model with proper event handling, transition operators,
and state variable management.
"""
import math
import random
from array import array
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evas.compiler.ast_nodes import *
from evas.simulator.engine import (
    AboveDetector,
    CrossDetector,
    TransitionState,
)
from evas.simulator.evaluate_ir import (
    COND_EQ,
    COND_GE,
    COND_GT,
    COND_LE,
    COND_LT,
    COND_NE,
    SOURCE_NODE,
    SOURCE_STATE,
    TARGET_NODE,
    TARGET_STATE,
    normalize_linear_ops,
)
from evas.simulator.event_transition_plan import (
    EVENT_TRANSITION_PROFILE_SUPPORT,
    analyze_event_transition_segment_plan,
    summarize_event_transition_plans,
)
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import BODY_TARGET_NODE, BODY_TARGET_STATE
from evas.simulator.stmt_ir import (
    classify_body_stmt_ops_rejection,
    encode_body_stmt_ops,
    lower_stmt,
)
from evas.simulator.whole_segment import validate_whole_segment_candidate


class CompilationError(Exception):
    pass


class CompiledModel:
    """Base class for compiled Verilog-A models."""
    _module_ast = None
    _module_registry: Dict[str, Any] = {}
    _module_ports: List[str] = []
    _static_voltage_read_nodes = ()
    _event_trigger_voltage_read_nodes = ()
    _event_voltage_read_nodes = ()
    _event_body_voltage_read_nodes = ()
    _static_output_write_nodes = ()
    _dynamic_branch_accesses = ()
    _dynamic_voltage_read_count = 0
    _dynamic_output_write_count = 0
    _state_scalar_names = ()
    _integer_state_names = ()
    _state_array_ranges = ()
    _indexed_state_fastpath_codegen = False
    _rust_static_affine_ops = ()
    _evaluate_ir_static_linear_ops = ()
    _evaluate_ir_static_linear_non_event_ops = ()
    _evaluate_ir_static_linear_rejections = ()
    _transition_target_ir_ops = ()
    _ordered_transition_segment_ir_ops = ((), ())
    _event_lfsr_shift_ir_ops = ()
    _event_static_linear_ir_ops = ()
    _event_timer_static_linear_ir_ops = ()
    _event_lfsr_output_nodes_by_state = ()
    _event_lfsr_output_hold_states = ()
    _whole_segment_candidates = ()
    _rust_body_ir_node_names = ()
    _rust_body_ir_param_names = ()
    _rust_body_ir_state_names = ()
    _rust_body_ir_integer_state_names = ()
    _rust_body_ir_stmt_ops = ()
    _rust_body_ir_expr_ops = ()
    _rust_body_ir_target_node_slots = ()
    _rust_body_ir_target_state_slots = ()
    _rust_body_ir_rejection_reason = "not_collected"
    _rust_body_ir_rejection_tags = ()
    _event_transition_plan_profiles = ()
    _event_transition_plan_rejection_reasons = {}
    _event_transition_plan_blocker_tags = {}
    _event_transition_plan_event_count = 0
    _event_transition_plan_due_trigger_count = 0
    _event_transition_plan_output_write_count = 0
    _event_transition_plan_transition_count = 0
    _event_transition_plan_state_assignment_count = 0
    _event_transition_plan_side_effect_count = 0
    _event_transition_plan_control_flow_count = 0
    _state_owned_timer_targets = ()
    _static_branch_fastpath_codegen = False
    _dynamic_node_cache_limit = 4096
    _cmp_eps: float = 0.0
    _needs_future_node_voltages: bool = False
    _has_dynamic_breakpoints: bool = True
    _has_post_update_events: bool = True
    _uses_bound_step: bool = True
    _transition_target_probe_count: int = 0

    def __init__(self):
        self.params: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.arrays: Dict[str, Dict[int, Any]] = {}
        self.transitions: Dict[str, TransitionState] = {}
        self._active_transition_keys: set[str] = set()
        self._transition_active_keys_known: bool = False
        # Audit 088: per-step batch queue. Populated by _transition_output_lazy
        # during evaluate(); drained by _flush_transitions() at evaluate end.
        # Typed-array buffers are built fresh at flush time at exact size n
        # (this allocs 14 arrays per flush, vs 14 per call in 086 baseline —
        # still a big win for any model with N>1 transitions per step).
        self._transition_pending_count: int = 0
        self._transition_pending_input: Optional[Dict[str, List]] = None
        self.slew_states: Dict[str, Dict[str, float]] = {}
        self.cross_detectors: Dict[str, CrossDetector] = {}
        self.above_detectors: Dict[str, AboveDetector] = {}
        self.output_nodes: Dict[str, float] = {}
        self._output_nodes_version: int = 0
        self.node_map: Dict[str, str] = {}  # port_name -> external_node
        self.default_transition: float = 1e-12
        self._initial_step_done: bool = False
        self._initial_condition_mode: bool = False
        self._strobe_log: List[str] = []
        self._event_time: float = 0.0  # $abstime inside cross/above event bodies
        self._temperature: float = 27.0  # degrees Celsius (expressions convert to Kelvin)
        self.timer_states: Dict[str, float] = {}  # key → next_fire_time
        self.timer_last_fired: Dict[str, float] = {}  # key → last absolute-time fire target
        self.timer_kinds: Dict[str, str] = {}  # key → absolute | periodic
        self._timer_state_version: int = 0
        self._timer_breakpoint_cache_version: int = -1
        self._timer_breakpoint_cache: Optional[float] = None
        self._timer_key_ids: Dict[str, int] = {}
        self._timer_keys_by_id: List[str] = []
        self._timer_next_fire_values = array("d")
        self._timer_last_fired_values = array("d")
        self._timer_has_last_fired_flags = array("B")
        self._timer_array_sidecar_version: int = -1
        self._bound_step: float = 0.0  # $bound_step limit (0 = no limit)
        self._nominal_step: float = 0.0
        self._discrete_update_buckets: Dict[str, int] = {}
        self._transition_breakpoint_min_ramp: float = 0.0
        self._event_context_active: bool = False
        self._step_prev_node_voltages: Dict[str, float] = {}
        self._step_curr_node_voltages: Dict[str, float] = {}
        self._step_future_node_voltages: Dict[str, float] = {}
        self._step_prev_time: float = 0.0
        self._step_time: float = 0.0
        self._step_latest_cross_event_time: float = -math.inf
        self._step_event_fired: bool = False
        self._needs_future_node_voltages = bool(
            getattr(self.__class__, "_needs_future_node_voltages", False)
        )
        self._needs_future_node_voltages_tree_cache: Optional[bool] = None
        self._has_dynamic_breakpoints_tree_cache: Optional[bool] = None
        self._uses_bound_step_tree_cache: Optional[bool] = None
        self._event_interpolated_nodes: set[str] = set()
        self._event_interpolated_node_values: Dict[str, float] = {}
        self._event_node_cross_directions: Dict[str, int] = {}
        self._event_trace_audit_enabled: bool = False
        self._event_trace_audit_max_records: int = 4096
        self._event_trace_audit_records: List[Dict[str, Any]] = []
        self._event_trace_audit_context_stack: List[Dict[str, Any]] = []
        self._event_trace_audit_write_targets: set[str] = set()
        self._event_trace_audit_seq: int = 0
        self._event_trace_audit_phase: str = "evaluate"
        self._perf_stats: Dict[str, int] = {
            "timer_periodic_checks": 0,
            "timer_periodic_fires": 0,
            "timer_periodic_skips": 0,
            "timer_absolute_checks": 0,
            "timer_absolute_fires": 0,
            "timer_absolute_expirations": 0,
            "timer_reschedules": 0,
            "timer_breakpoint_hits": 0,
            "timer_breakpoint_scans": 0,
            "timer_breakpoint_cache_hits": 0,
            "timer_state_updates": 0,
            "timer_batch_due_calls": 0,
            "timer_batch_due_events": 0,
            "timer_batch_due_fires": 0,
            "timer_batch_due_fallbacks": 0,
            "timer_state_owned_checks": 0,
            "timer_state_owned_fast_skips": 0,
            "timer_state_owned_target_reads": 0,
            "timer_state_owned_fires": 0,
            "timer_state_owned_fallbacks": 0,
            "cross_fires": 0,
            "above_fires": 0,
            "static_branch_fastpath_fallbacks": 0,
            "dynamic_node_cache_hits": 0,
            "dynamic_node_cache_misses": 0,
            "dynamic_node_cache_bypasses": 0,
            "transition_unchanged_target_fastpath": 0,
            "transition_calls": 0,
            "transition_output_fastpath_calls": 0,
            "transition_set_target_calls": 0,
            "transition_evaluate_calls": 0,
            "rust_transition_state_production_calls": 0,
            "rust_transition_state_production_outputs": 0,
            "rust_transition_state_production_fallbacks": 0,
            "rust_transition_state_buffer_reuse_calls": 0,
            "rust_transition_state_buffer_alloc_total": 0,
            # Audit 088: per-step batch
            "rust_transition_batch_flushes": 0,
            "rust_transition_batch_slot_total": 0,
            "rust_transition_batch_fallbacks": 0,
            "rust_transition_batch_max_slots": 0,
            "rust_transition_lazy_enqueues": 0,
            # Audit 089: cross/above production gate
            "rust_cross_production_calls": 0,
            "rust_cross_production_fires": 0,
            "rust_cross_production_fallbacks": 0,
            "rust_above_production_calls": 0,
            "rust_above_production_fires": 0,
            "rust_above_production_fallbacks": 0,
            "rust_event_interpolation_batches": 0,
            "rust_event_interpolation_nodes": 0,
            "rust_event_interpolation_cache_hits": 0,
            "rust_event_interpolation_fallbacks": 0,
            "transition_breakpoint_scans": 0,
            "transition_breakpoint_active_scans": 0,
            "transition_breakpoint_inactive_skips": 0,
            "rust_transition_breakpoint_scans": 0,
            "rust_transition_breakpoint_state_scans": 0,
            "rust_transition_breakpoint_fallbacks": 0,
            "rust_timer_breakpoint_scans": 0,
            "rust_timer_breakpoint_state_scans": 0,
            "rust_timer_breakpoint_fallbacks": 0,
            "rust_timer_event_production_periodic_calls": 0,
            "rust_timer_event_production_absolute_calls": 0,
            "rust_timer_event_production_fires": 0,
            "rust_timer_event_production_skips": 0,
            "rust_timer_event_production_expirations": 0,
            "rust_timer_event_production_fallbacks": 0,
            "rust_event_due_shadow_cross_checks": 0,
            "rust_event_due_shadow_above_checks": 0,
            "rust_event_due_shadow_timer_periodic_checks": 0,
            "rust_event_due_shadow_timer_absolute_checks": 0,
            "rust_event_due_shadow_matches": 0,
            "rust_event_due_shadow_mismatches": 0,
            "rust_event_due_shadow_errors": 0,
            "rust_event_due_shadow_max_time_diff": 0.0,
            "rust_event_write_shadow_checks": 0,
            "rust_event_write_shadow_matches": 0,
            "rust_event_write_shadow_mismatches": 0,
            "rust_event_write_shadow_errors": 0,
            "rust_event_write_production_calls": 0,
            "rust_event_write_production_executed": 0,
            "rust_event_write_production_fallbacks": 0,
            "rust_event_linear_write_batches": 0,
            "rust_event_linear_write_shadow_checks": 0,
            "rust_event_linear_write_shadow_matches": 0,
            "rust_event_linear_write_shadow_mismatches": 0,
            "rust_event_linear_write_shadow_errors": 0,
            "rust_event_linear_write_production_calls": 0,
            "rust_event_linear_write_production_executed": 0,
            "rust_event_linear_write_production_fallbacks": 0,
            "rust_timer_lfsr_output_batches": 0,
            "rust_timer_lfsr_output_calls": 0,
            "rust_timer_lfsr_output_due": 0,
            "rust_timer_lfsr_output_skips": 0,
            "rust_timer_lfsr_output_executed": 0,
            "rust_timer_lfsr_output_writes": 0,
            "rust_timer_lfsr_output_fallbacks": 0,
            "event_trace_audit_events": 0,
            "event_trace_audit_body_entries": 0,
            "event_trace_audit_cross_events": 0,
            "event_trace_audit_above_events": 0,
            "event_trace_audit_timer_events": 0,
            "event_trace_audit_initial_step_events": 0,
            "event_trace_audit_final_step_events": 0,
            "event_trace_audit_combined_events": 0,
            "event_trace_audit_state_writes": 0,
            "event_trace_audit_array_writes": 0,
            "event_trace_audit_output_writes": 0,
            "event_trace_audit_timer_state_writes": 0,
            "event_trace_audit_timer_last_fired_writes": 0,
            "event_trace_audit_transition_writes": 0,
            "event_trace_audit_transition_output_writes": 0,
            "event_trace_audit_in_event_writes": 0,
            "event_trace_audit_records_dropped": 0,
            "timer_array_sidecar_updates": 0,
            "timer_array_sidecar_rebuilds": 0,
            "timer_array_sidecar_scans": 0,
            "indexed_state_scalar_writes": 0,
            "indexed_state_scalar_reads": 0,
            "indexed_state_array_writes": 0,
            "indexed_state_array_reads": 0,
            "indexed_state_array_oob_writes": 0,
            "rust_body_ir_production_batches": 0,
            "rust_body_ir_production_calls": 0,
            "rust_body_ir_production_executed": 0,
            "rust_body_ir_production_fallbacks": 0,
            "rust_body_ir_production_node_writes": 0,
            "rust_body_ir_production_state_writes": 0,
        }
        # Lazy-allocated integrator states (only used when idt/idtmod appears)
        self._idt_states: Optional[Dict[str, Dict[str, float]]] = None
        self._file_handles: Dict[int, Any] = {}  # fd → file object
        self._next_fd: int = 1
        self._child_models: List["CompiledModel"] = []
        self._parent_model: Optional["CompiledModel"] = None
        self._indexed_output_writer: Optional[Callable[[str, float], None]] = None
        self._indexed_voltage_probe: Optional[Callable[[str, str, float, bool], None]] = None
        self._indexed_voltage_reader: Optional[Callable[[str, str], Optional[float]]] = None
        self._static_branch_fastpath_enabled: bool = False
        self._transition_unchanged_fastpath_enabled: bool = False
        self._static_branch_indexed_values: Optional[List[float]] = None
        self._static_branch_read_node_ids: tuple[int, ...] = ()
        self._static_branch_write_node_ids: tuple[int, ...] = ()
        self._static_branch_write_external_nodes: tuple[str, ...] = ()
        self._node_resolution_cache_enabled: bool = False
        self._node_resolution_cache: Dict[str, str] = {}
        self._dynamic_node_cache: Dict[Tuple[str, int, Optional[int]], str] = {}
        self._indexed_state_ids: Dict[str, int] = {}
        self._indexed_integer_state_names: set[str] = set()
        self._indexed_state_values: Optional[List[float]] = None
        self._indexed_state_array_layouts: Dict[str, Tuple[int, int, bool]] = {}
        self._indexed_state_array_values: Dict[str, List[float]] = {}
        self._indexed_state_array_element_ids: Dict[Tuple[str, int], int] = {}
        self._indexed_state_array_slot_refs: Dict[str, Tuple[str, int]] = {}
        self._rust_transition_breakpoint_scanner: Optional[
            Callable[..., Optional[float]]
        ] = None
        self._rust_transition_state_backend: Optional[Any] = None
        self._rust_transition_state_production_enabled: bool = False
        # Audit 086: persistent typed-array buffers shared across all
        # _transition_rust_production() calls. None until first call;
        # then a single set of size-1 buffers is reused for every
        # subsequent call, eliminating the 14x per-call array() alloc
        # that the original wire-up paid. See audit 086 for rationale.
        self._rust_transition_buffers: Optional[Dict[str, Any]] = None
        self._rust_event_interpolation_backend: Optional[Any] = None
        self._rust_timer_breakpoint_scanner: Optional[
            Callable[..., Optional[float]]
        ] = None
        self._rust_timer_event_backend: Optional[Any] = None
        self._rust_timer_event_production_enabled: bool = False
        self._rust_timer_event_min_timers: int = 2
        self._rust_event_due_shadow_backend: Optional[Any] = None
        # Audit 089: production Rust ownership of cross()/above() detector
        # state evolution. When the flag is on, _check_cross/_check_above
        # delegate the detector math to the Rust primitive instead of
        # CrossDetector.check()/AboveDetector.check(). All event side
        # effects (retrograde suppression, event_time context, interp
        # nodes cache, ordering) remain in Python — Rust only owns the
        # detector state machine, not the event queue.
        self._rust_cross_above_production_backend: Optional[Any] = None
        self._rust_cross_above_production_enabled: bool = False
        self._rust_event_write_backend: Optional[Any] = None
        self._rust_event_write_node_values: Optional[List[float]] = None
        self._rust_event_write_shadow_enabled: bool = False
        self._rust_event_write_production_enabled: bool = False
        self._rust_event_write_batches: Dict[str, Dict[str, Any]] = {}
        self._rust_event_linear_write_batches: Dict[str, Dict[str, Any]] = {}
        self._rust_timer_lfsr_output_batches: Dict[str, Dict[str, Any]] = {}
        self._rust_timer_lfsr_output_hold_by_state: Dict[str, Dict[str, Any]] = {}
        self._rust_body_ir_backend: Optional[Any] = None
        self._rust_body_ir_batch: Optional[Any] = None
        self._rust_body_ir_production_enabled: bool = False
        # Per-instance deterministic RNG streams.
        self._rng_default = random.Random(0)
        self._rng_streams: Dict[int, random.Random] = {}

    def initial_step(self, node_voltages: Dict[str, float], time: float):
        pass

    def evaluate(self, node_voltages: Dict[str, float], time: float):
        pass

    def post_update_events(self, node_voltages: Dict[str, float], time: float) -> bool:
        return False

    def refresh_outputs(self, node_voltages: Dict[str, float], time: float):
        pass

    def final_step(self, node_voltages: Dict[str, float], time: float):
        pass

    def _set_rust_body_ir_backend(self, backend, *, production: bool = False):
        """Install the opt-in 094 body-IR production backend for this tree."""
        self._rust_body_ir_backend = backend
        self._rust_body_ir_batch = None
        self._rust_body_ir_production_enabled = bool(production)
        if backend is not None and production:
            stmt_ops = tuple(getattr(self.__class__, "_rust_body_ir_stmt_ops", ()) or ())
            expr_ops = tuple(getattr(self.__class__, "_rust_body_ir_expr_ops", ()) or ())
            if stmt_ops:
                try:
                    self._rust_body_ir_batch = backend.make_body_ir_batch(
                        stmt_ops=stmt_ops,
                        expr_ops=expr_ops,
                    )
                    self._perf_stats["rust_body_ir_production_batches"] = 1
                except Exception:
                    self._rust_body_ir_batch = None
                    self._perf_stats["rust_body_ir_production_fallbacks"] += 1
        for child in self._child_models:
            child._set_rust_body_ir_backend(backend, production=production)

    def _try_evaluate_rust_body_ir(
        self,
        node_voltages: Dict[str, float],
        time: float,
    ) -> bool:
        """Try to execute this model's conservative 094 body IR in Rust.

        This first production hook still packs one model's local node, state,
        and parameter arrays per call.  It proves the 094 body IR can own real
        evaluate semantics under an explicit gate; it is not the final
        whole-step typed-array fast path.
        """

        if (
            not self._rust_body_ir_production_enabled
            or self._rust_body_ir_batch is None
            or self._event_trace_audit_enabled
            or self._event_context_active
        ):
            return False
        if self._child_models or self._transition_pending_count > 0:
            return False

        node_names = tuple(getattr(self.__class__, "_rust_body_ir_node_names", ()) or ())
        state_names = tuple(getattr(self.__class__, "_rust_body_ir_state_names", ()) or ())
        param_names = tuple(getattr(self.__class__, "_rust_body_ir_param_names", ()) or ())
        try:
            node_values = array(
                "d",
                (float(self._get_voltage(name, node_voltages)) for name in node_names),
            )
            state_values = array(
                "d",
                (
                    float(self._state_get_by_slot(slot, name))
                    for slot, name in enumerate(state_names)
                ),
            )
            param_values = array(
                "d",
                (float(self.params.get(name, 0.0)) for name in param_names),
            )
        except Exception:
            self._perf_stats["rust_body_ir_production_fallbacks"] += 1
            return False

        backend = self._rust_body_ir_backend
        try:
            backend.evaluate_body_ir(
                self._rust_body_ir_batch,
                node_values=node_values,
                state_values=state_values,
                param_values=param_values,
            )
        except Exception:
            self._perf_stats["rust_body_ir_production_fallbacks"] += 1
            return False

        self._event_trace_audit_phase = "evaluate"
        self._event_time = time
        self._bound_step = 0.0

        target_state_slots = tuple(
            getattr(self.__class__, "_rust_body_ir_target_state_slots", ()) or ()
        )
        for slot in target_state_slots:
            if 0 <= slot < len(state_names) and slot < len(state_values):
                self._state_set_by_slot(slot, state_names[slot], state_values[slot])

        target_node_slots = tuple(
            getattr(self.__class__, "_rust_body_ir_target_node_slots", ()) or ()
        )
        for slot in target_node_slots:
            if 0 <= slot < len(node_names) and slot < len(node_values):
                self._set_output(node_names[slot], node_values[slot], node_voltages)

        self._perf_stats["rust_body_ir_production_calls"] += 1
        self._perf_stats["rust_body_ir_production_executed"] += 1
        self._perf_stats["rust_body_ir_production_state_writes"] += len(target_state_slots)
        self._perf_stats["rust_body_ir_production_node_writes"] += len(target_node_slots)
        return True

    @staticmethod
    def _rust_static_affine_scalar_uses_params(value: Any) -> bool:
        return isinstance(value, tuple)

    @staticmethod
    def _evaluate_rust_static_affine_scalar(value: Any, params: Dict[str, Any]) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, tuple) or not value:
            raise ValueError(f"unsupported Rust static affine scalar: {value!r}")

        op = value[0]
        if op == "param":
            return float(params[value[1]])
        if op == "neg":
            return -CompiledModel._evaluate_rust_static_affine_scalar(value[1], params)

        left = CompiledModel._evaluate_rust_static_affine_scalar(value[1], params)
        right = CompiledModel._evaluate_rust_static_affine_scalar(value[2], params)
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "div":
            if right == 0.0:
                raise ValueError("division by zero in Rust static affine scalar")
            return left / right
        raise ValueError(f"unsupported Rust static affine scalar op: {op!r}")

    def _set_initial_condition_mode(self, enabled: bool):
        """Treat transition-like operators as operating-point values."""
        self._initial_condition_mode = bool(enabled)
        for child in self._child_models:
            child._set_initial_condition_mode(enabled)

    def _set_nominal_step(self, step: float):
        """Record the user-visible transient step for step-normalized state updates."""
        self._nominal_step = float(step) if step and step > 0.0 else 0.0
        self._discrete_update_buckets.clear()
        for child in self._child_models:
            child._set_nominal_step(step)

    def _set_indexed_output_writer(self, writer: Optional[Callable[[str, float], None]]):
        """Install an opt-in output write-through hook for indexed sidecars."""
        self._indexed_output_writer = writer
        for child in self._child_models:
            child._set_indexed_output_writer(writer)

    def _set_indexed_voltage_probe(
        self,
        probe: Optional[Callable[[str, str, float, bool], None]],
    ):
        """Install an opt-in voltage-read probe for indexed sidecars."""
        self._indexed_voltage_probe = probe
        for child in self._child_models:
            child._set_indexed_voltage_probe(probe)

    def _set_indexed_voltage_reader(
        self,
        reader: Optional[Callable[[str, str], Optional[float]]],
    ):
        """Install an opt-in non-event voltage reader for indexed sidecars."""
        self._indexed_voltage_reader = reader
        for child in self._child_models:
            child._set_indexed_voltage_reader(reader)

    def _read_indexed_voltage(self, local_node: str, external_node: str) -> Optional[float]:
        """Read from an indexed sidecar only outside event interpolation contexts."""
        if self._event_context_active or self._indexed_voltage_reader is None:
            return None
        return self._indexed_voltage_reader(local_node, external_node)

    def _set_indexed_state_storage(
        self,
        scalar_ids: Optional[Dict[str, int]] = None,
        integer_state_names: Tuple[str, ...] = (),
        array_layouts: Tuple[Tuple[str, int, int, bool], ...] = (),
    ):
        """Install an opt-in indexed mirror for model-local state."""
        if scalar_ids is None:
            self._commit_indexed_state_storage()
            self._indexed_state_ids = {}
            self._indexed_integer_state_names = set()
            self._indexed_state_values = None
            self._indexed_state_array_layouts = {}
            self._indexed_state_array_values = {}
            self._indexed_state_array_element_ids = {}
            self._indexed_state_array_slot_refs = {}
            return

        self._indexed_state_ids = dict(scalar_ids)
        self._indexed_integer_state_names = set(integer_state_names)
        slot_count = max(self._indexed_state_ids.values(), default=-1) + 1
        values = [0.0] * slot_count

        layouts: Dict[str, Tuple[int, int, bool]] = {}
        array_values: Dict[str, List[float]] = {}
        array_element_ids: Dict[Tuple[str, int], int] = {}
        array_slot_refs: Dict[str, Tuple[str, int]] = {}
        for name, lo, hi, integer in array_layouts:
            lo_i = int(lo)
            hi_i = int(hi)
            length = max(0, hi_i - lo_i + 1)
            layouts[name] = (lo_i, hi_i, bool(integer))
            slots = [0.0] * length
            array_data = self.arrays.get(name, {}) or {}
            for idx_i in range(lo_i, hi_i + 1):
                raw_value = array_data.get(idx_i, 0)
                if integer:
                    raw_value = self._to_integer(raw_value)
                stored = float(raw_value)
                slots[idx_i - lo_i] = stored
                slot_name = self._state_array_slot_name(name, idx_i)
                flat_slot = self._indexed_state_ids.get(slot_name)
                if flat_slot is not None and 0 <= flat_slot < slot_count:
                    values[flat_slot] = stored
                    array_element_ids[(name, idx_i)] = flat_slot
                    array_slot_refs[slot_name] = (name, idx_i)
            array_values[name] = slots

        for name, slot in self._indexed_state_ids.items():
            if name in array_slot_refs:
                continue
            raw_value = self.state.get(name, 0)
            if name in self._indexed_integer_state_names:
                raw_value = self._to_integer(raw_value)
            values[slot] = float(raw_value)
        self._indexed_state_values = values

        self._indexed_state_array_layouts = layouts
        self._indexed_state_array_values = array_values
        self._indexed_state_array_element_ids = array_element_ids
        self._indexed_state_array_slot_refs = array_slot_refs

    def _commit_indexed_state_storage(self):
        """Copy indexed scalar state back to ``self.state`` before teardown."""
        values = self._indexed_state_values
        if values is None:
            return
        for name, slot in self._indexed_state_ids.items():
            if slot < 0 or slot >= len(values):
                continue
            value = values[slot]
            array_ref = self._indexed_state_array_slot_refs.get(name)
            if array_ref is not None:
                array_name, idx = array_ref
                layout = self._indexed_state_array_layouts.get(array_name)
                if layout is None:
                    continue
                lo, hi, integer = layout
                if not (lo <= idx <= hi):
                    continue
                stored = self._to_integer(value) if integer else value
                if array_name not in self.arrays:
                    self.arrays[array_name] = {}
                self.arrays[array_name][idx] = stored
                self._indexed_state_array_values[array_name][idx - lo] = float(stored)
                continue
            if name in self._indexed_integer_state_names:
                self.state[name] = self._to_integer(value)
            else:
                self.state[name] = value

    def _state_set(self, name: str, val: Any):
        """Set scalar state and mirror it to indexed storage when installed."""
        self.state[name] = val
        self._event_trace_audit_note_write("state", name)
        values = self._indexed_state_values
        if values is None:
            return
        slot = self._indexed_state_ids.get(name)
        if slot is None or slot >= len(values):
            return
        stored = self._to_integer(val) if name in self._indexed_integer_state_names else val
        values[slot] = float(stored)
        self._perf_stats["indexed_state_scalar_writes"] += 1

    def _state_get_by_slot(self, slot: int, name: str) -> Any:
        """Read scalar state from indexed storage when installed.

        The generated code still carries ``name`` as a fallback key so the same
        compiled model remains valid when the indexed state sidecar is disabled.
        """
        values = self._indexed_state_values
        if values is not None and 0 <= slot < len(values):
            value = values[slot]
            self._perf_stats["indexed_state_scalar_reads"] += 1
            if name in self._indexed_integer_state_names:
                return self._to_integer(value)
            return value
        return self.state[name]

    def _state_set_by_slot(self, slot: int, name: str, val: Any):
        """Set scalar state and update indexed storage by precomputed slot."""
        self.state[name] = val
        self._event_trace_audit_note_write("state", name)
        values = self._indexed_state_values
        if values is None or slot < 0 or slot >= len(values):
            return
        stored = self._to_integer(val) if name in self._indexed_integer_state_names else val
        values[slot] = float(stored)
        self._perf_stats["indexed_state_scalar_writes"] += 1

    @staticmethod
    def _state_array_slot_name(name: str, idx: int) -> str:
        return f"{name}[{int(idx)}]"

    @staticmethod
    def _state_array_slot_ref(slot_name: str) -> Optional[Tuple[str, int]]:
        if not isinstance(slot_name, str) or not slot_name.endswith("]"):
            return None
        if "[" not in slot_name:
            return None
        name, raw_idx = slot_name[:-1].rsplit("[", 1)
        if not name:
            return None
        try:
            return name, int(raw_idx)
        except ValueError:
            return None

    def _set_static_branch_fastpath_enabled(self, enabled: bool):
        """Enable opt-in helpers for static, non-dynamic branch read/write code."""
        self._static_branch_fastpath_enabled = bool(enabled)
        if enabled:
            self._perf_stats["static_branch_fastpath_fallbacks"] = 0
        for child in self._child_models:
            child._set_static_branch_fastpath_enabled(enabled)

    def _set_transition_unchanged_fastpath_enabled(self, enabled: bool):
        """Enable the no-op transition target fast path."""
        self._transition_unchanged_fastpath_enabled = bool(enabled)
        for child in self._child_models:
            child._set_transition_unchanged_fastpath_enabled(enabled)

    def _set_rust_transition_breakpoint_scanner(self, scanner):
        """Install an optional Rust/array scanner for transition breakpoints."""
        self._rust_transition_breakpoint_scanner = scanner
        for child in self._child_models:
            child._set_rust_transition_breakpoint_scanner(scanner)

    def _set_rust_transition_state_backend(self, backend, *, production: bool = False):
        """Install an optional Rust backend for transition() state stepping."""
        self._rust_transition_state_backend = backend
        self._rust_transition_state_production_enabled = bool(production)
        for child in self._child_models:
            child._set_rust_transition_state_backend(backend, production=production)

    def _set_rust_event_interpolation_backend(self, backend):
        """Install an optional Rust backend for event-time voltage interpolation."""
        self._rust_event_interpolation_backend = backend
        for child in self._child_models:
            child._set_rust_event_interpolation_backend(backend)

    def _set_rust_timer_breakpoint_scanner(self, scanner):
        """Install an optional Rust/array scanner for timer breakpoints."""
        self._rust_timer_breakpoint_scanner = scanner
        for child in self._child_models:
            child._set_rust_timer_breakpoint_scanner(scanner)

    def _set_rust_timer_event_backend(self, backend, *, production: bool = False):
        """Install an optional Rust backend for timer due/reschedule state."""
        self._rust_timer_event_backend = backend
        self._rust_timer_event_production_enabled = bool(production)
        for child in self._child_models:
            child._set_rust_timer_event_backend(backend, production=production)

    def _set_rust_event_due_shadow_backend(self, backend):
        """Install an optional Rust shadow backend for event due checks."""
        self._rust_event_due_shadow_backend = backend
        for child in self._child_models:
            child._set_rust_event_due_shadow_backend(backend)

    def _set_rust_cross_above_production_backend(self, backend, *, production: bool = False):
        """Audit 089: install optional Rust production owner of cross()/above()
        detector state evolution."""
        self._rust_cross_above_production_backend = backend
        self._rust_cross_above_production_enabled = bool(production)
        for child in self._child_models:
            child._set_rust_cross_above_production_backend(backend, production=production)

    def _set_rust_event_write_backend(
        self,
        backend,
        node_index=None,
        node_values: Optional[List[float]] = None,
        *,
        shadow: bool = False,
        production: bool = False,
    ):
        """Install optional Rust event-body write batches for this model tree."""
        self._rust_event_write_backend = backend
        self._rust_event_write_node_values = node_values
        self._rust_event_write_shadow_enabled = bool(shadow)
        self._rust_event_write_production_enabled = bool(production)
        self._rust_event_write_batches = {}
        self._rust_event_linear_write_batches = {}
        self._rust_timer_lfsr_output_batches = {}
        self._rust_timer_lfsr_output_hold_by_state = {}
        if backend is not None:
            self._rust_event_write_batches = self._build_rust_event_write_batches(backend)
            if node_index is not None and node_values is not None:
                self._rust_event_linear_write_batches = (
                    self._build_rust_event_linear_write_batches(backend, node_index)
                )
                self._perf_stats["rust_event_linear_write_batches"] = len(
                    self._rust_event_linear_write_batches
                )
                self._rust_timer_lfsr_output_batches = (
                    self._build_rust_timer_lfsr_output_batches(backend, node_index)
                )
                self._rust_timer_lfsr_output_hold_by_state = {
                    info["output_state"]: info
                    for info in self._rust_timer_lfsr_output_batches.values()
                    if info.get("output_state") in getattr(
                        self.__class__, "_event_lfsr_output_hold_states", ()
                    )
                }
                self._perf_stats["rust_timer_lfsr_output_batches"] = len(
                    self._rust_timer_lfsr_output_batches
                )
        for child in self._child_models:
            child._set_rust_event_write_backend(
                backend,
                node_index,
                node_values,
                shadow=shadow,
                production=production,
            )

    def _build_rust_event_write_batches(self, backend) -> Dict[str, Dict[str, Any]]:
        batches: Dict[str, Dict[str, Any]] = {}
        state_ids = self._indexed_state_ids
        if not state_ids:
            return batches
        for raw in getattr(self.__class__, "_event_lfsr_shift_ir_ops", ()) or ():
            try:
                (
                    key,
                    lfsr_refs,
                    tmp_refs,
                    tap_refs,
                    gate_node,
                    gate_threshold,
                    high_node,
                    low_node,
                    output_state,
                    loop_state,
                    loop_final_value,
                ) = raw
                lfsr_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in lfsr_refs
                )
                tmp_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in tmp_refs
                )
                tap_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in tap_refs
                )
                output_id = state_ids[output_state] if output_state else None
                loop_id = state_ids[loop_state] if loop_state else None
                batch = backend.make_lfsr_event_batch(
                    lfsr_slots=lfsr_slots,
                    tmp_slots=tmp_slots,
                    tap_slots=tap_slots,
                    gate_node_id=0,
                    gate_threshold=float(gate_threshold),
                    high_node_id=1,
                    low_node_id=2,
                    output_state_id=output_id,
                    loop_state_id=loop_id,
                    loop_final_value=float(loop_final_value),
                )
            except Exception:
                self._perf_stats["rust_event_write_production_fallbacks"] += 1
                continue
            target_slots = set(lfsr_slots) | set(tmp_slots) | set(tap_slots)
            if output_id is not None:
                target_slots.add(output_id)
            if loop_id is not None:
                target_slots.add(loop_id)
            batches[str(key)] = {
                "batch": batch,
                "target_slots": tuple(sorted(target_slots)),
                "node_names": (str(gate_node), str(high_node), str(low_node)),
            }
        return batches

    def _rust_event_linear_required_external_nodes(self) -> Tuple[str, ...]:
        """Return external nodes read by linear event-body write batches."""
        nodes: set[str] = set()
        for raw in getattr(self.__class__, "_event_static_linear_ir_ops", ()) or ():
            try:
                _key, raw_ops = raw
                ir_ops = normalize_linear_ops(tuple(raw_ops or ()))
            except Exception:
                continue
            for op in ir_ops:
                for term in self._event_linear_all_terms(op):
                    if term.source_kind == SOURCE_NODE:
                        nodes.add(self._resolve_external_node(term.source_name))
        for child in self._child_models:
            nodes.update(child._rust_event_linear_required_external_nodes())
        return tuple(sorted(nodes))

    @staticmethod
    def _event_linear_all_terms(op) -> Tuple[Any, ...]:
        terms = list(getattr(op, "terms", ()) or ())
        terms.extend(getattr(op, "false_terms", ()) or ())
        condition = getattr(op, "condition", None)
        if condition is not None:
            terms.extend(getattr(condition, "left_terms", ()) or ())
            terms.extend(getattr(condition, "right_terms", ()) or ())
        return tuple(terms)

    def _build_rust_event_linear_write_batches(
        self,
        backend,
        node_index,
    ) -> Dict[str, Dict[str, Any]]:
        batches: Dict[str, Dict[str, Any]] = {}
        state_ids = self._indexed_state_ids
        state_values = self._indexed_state_values
        if not state_ids or state_values is None or node_index is None:
            return batches

        try:
            from evas.simulator.rust_backend import (
                LinearCondition,
                LinearOp,
                LinearTerm,
            )
        except Exception:
            self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
            return batches

        def eval_scalar(value: Any) -> Optional[float]:
            try:
                return float(
                    self._evaluate_rust_static_affine_scalar(value, self.params)
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return None

        def convert_terms(ir_terms) -> Optional[Tuple[Any, ...]]:
            converted = []
            for term in ir_terms:
                gain = eval_scalar(term.gain)
                if gain is None:
                    return None
                if term.source_kind == SOURCE_NODE:
                    external = self._resolve_external_node(term.source_name)
                    if not node_index.has(external):
                        return None
                    converted.append(
                        LinearTerm(
                            source_kind=SOURCE_NODE,
                            source_id=node_index.id_of(external),
                            gain=gain,
                        )
                    )
                elif term.source_kind == SOURCE_STATE:
                    source_id = state_ids.get(term.source_name)
                    if source_id is None:
                        return None
                    converted.append(
                        LinearTerm(
                            source_kind=SOURCE_STATE,
                            source_id=source_id,
                            gain=gain,
                        )
                    )
                else:
                    return None
            return tuple(converted)

        def convert_condition(ir_condition):
            if ir_condition is None:
                return None
            left_bias = eval_scalar(ir_condition.left_bias)
            right_bias = eval_scalar(ir_condition.right_bias)
            if left_bias is None or right_bias is None:
                return False
            left_terms = convert_terms(ir_condition.left_terms)
            right_terms = convert_terms(ir_condition.right_terms)
            if left_terms is None or right_terms is None:
                return False
            return LinearCondition(
                op_kind=ir_condition.op_kind,
                left_bias=left_bias,
                left_terms=left_terms,
                right_bias=right_bias,
                right_terms=right_terms,
            )

        for raw in getattr(self.__class__, "_event_static_linear_ir_ops", ()) or ():
            try:
                key, raw_ops = raw
                ir_ops = normalize_linear_ops(tuple(raw_ops or ()))
            except Exception:
                self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
                continue
            ops = []
            target_slots: set[int] = set()
            target_states: set[str] = set()
            source_node_pairs: set[Tuple[str, int]] = set()
            uses_node_sources = False
            failed = False
            for ir_op in ir_ops:
                if ir_op.target_kind != TARGET_STATE:
                    failed = True
                    break
                target_id = state_ids.get(ir_op.target_name)
                if target_id is None:
                    failed = True
                    break
                bias = eval_scalar(ir_op.bias)
                false_bias = eval_scalar(ir_op.false_bias)
                if bias is None or false_bias is None:
                    failed = True
                    break
                terms = convert_terms(ir_op.terms)
                false_terms = convert_terms(ir_op.false_terms)
                if terms is None or false_terms is None:
                    failed = True
                    break
                condition = convert_condition(ir_op.condition)
                if condition is False:
                    failed = True
                    break
                uses_node_sources = uses_node_sources or any(
                    term.source_kind == SOURCE_NODE
                    for term in self._event_linear_all_terms(ir_op)
                )
                for term in self._event_linear_all_terms(ir_op):
                    if term.source_kind != SOURCE_NODE:
                        continue
                    external = self._resolve_external_node(term.source_name)
                    if not node_index.has(external):
                        failed = True
                        break
                    source_node_pairs.add((str(external), node_index.id_of(external)))
                if failed:
                    break
                ops.append(
                    LinearOp(
                        target_kind=TARGET_STATE,
                        target_id=target_id,
                        bias=bias,
                        terms=terms,
                        condition=condition,
                        false_bias=false_bias,
                        false_terms=false_terms,
                        target_integer=bool(ir_op.target_integer),
                    )
                )
                target_slots.add(int(target_id))
                target_states.add(str(ir_op.target_name))
            if failed or not ops:
                self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
                continue
            try:
                batch = backend.make_static_linear_batch(ops)
            except Exception:
                self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
                continue
            batches[str(key)] = {
                "batch": batch,
                "target_slots": tuple(sorted(target_slots)),
                "target_states": tuple(sorted(target_states)),
                "source_node_pairs": tuple(sorted(source_node_pairs)),
                "uses_node_sources": bool(uses_node_sources),
            }
        return batches

    def _rust_lfsr_output_node_by_state(self) -> Dict[str, str]:
        return {
            str(state): str(node)
            for state, node in getattr(
                self.__class__, "_event_lfsr_output_nodes_by_state", ()
            ) or ()
        }

    def _rust_timer_lfsr_required_external_nodes(self) -> Tuple[str, ...]:
        nodes: set[str] = set()
        output_by_state = self._rust_lfsr_output_node_by_state()
        for raw in getattr(self.__class__, "_event_lfsr_shift_ir_ops", ()) or ():
            try:
                (
                    _key,
                    _lfsr_refs,
                    _tmp_refs,
                    _tap_refs,
                    gate_node,
                    _gate_threshold,
                    high_node,
                    low_node,
                    output_state,
                    _loop_state,
                    _loop_final_value,
                ) = raw
                output_node = output_by_state.get(str(output_state))
                if output_node is None:
                    continue
                nodes.add(self._resolve_external_node(str(gate_node)))
                nodes.add(self._resolve_external_node(str(high_node)))
                nodes.add(self._resolve_external_node(str(low_node)))
                nodes.add(self._resolve_external_node(str(output_node)))
            except Exception:
                continue
        for child in self._child_models:
            nodes.update(child._rust_timer_lfsr_required_external_nodes())
        return tuple(sorted(nodes))

    def _build_rust_timer_lfsr_output_batches(
        self,
        backend,
        node_index,
    ) -> Dict[str, Dict[str, Any]]:
        batches: Dict[str, Dict[str, Any]] = {}
        state_ids = self._indexed_state_ids
        if not state_ids:
            return batches
        output_by_state = self._rust_lfsr_output_node_by_state()
        if not output_by_state:
            return batches
        for raw in getattr(self.__class__, "_event_lfsr_shift_ir_ops", ()) or ():
            try:
                (
                    key,
                    lfsr_refs,
                    tmp_refs,
                    tap_refs,
                    gate_node,
                    gate_threshold,
                    high_node,
                    low_node,
                    output_state,
                    loop_state,
                    loop_final_value,
                ) = raw
                output_node = output_by_state.get(str(output_state))
                if output_node is None:
                    continue
                gate_external = self._resolve_external_node(str(gate_node))
                high_external = self._resolve_external_node(str(high_node))
                low_external = self._resolve_external_node(str(low_node))
                output_external = self._resolve_external_node(str(output_node))
                if not all(
                    node_index.has(node)
                    for node in (
                        gate_external,
                        high_external,
                        low_external,
                        output_external,
                    )
                ):
                    continue
                lfsr_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in lfsr_refs
                )
                tmp_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in tmp_refs
                )
                tap_slots = tuple(
                    state_ids[CompiledModel._state_array_slot_name(name, idx)]
                    for name, idx in tap_refs
                )
                output_id = state_ids[output_state] if output_state else None
                loop_id = state_ids[loop_state] if loop_state else None
                output_node_id = node_index.id_of(output_external)
                batch = backend.make_lfsr_event_batch(
                    lfsr_slots=lfsr_slots,
                    tmp_slots=tmp_slots,
                    tap_slots=tap_slots,
                    gate_node_id=node_index.id_of(gate_external),
                    gate_threshold=float(gate_threshold),
                    high_node_id=node_index.id_of(high_external),
                    low_node_id=node_index.id_of(low_external),
                    output_state_id=output_id,
                    output_node_id=output_node_id,
                    loop_state_id=loop_id,
                    loop_final_value=float(loop_final_value),
                )
            except Exception:
                self._perf_stats["rust_timer_lfsr_output_fallbacks"] += 1
                continue
            target_slots = set(lfsr_slots) | set(tmp_slots) | set(tap_slots)
            if output_id is not None:
                target_slots.add(output_id)
            if loop_id is not None:
                target_slots.add(loop_id)
            batches[str(key)] = {
                "batch": batch,
                "target_slots": tuple(sorted(target_slots)),
                "output_state": str(output_state),
                "output_local_node": str(output_node),
                "output_external_node": output_external,
                "output_node_id": output_node_id,
            }
        return batches

    def _rust_event_write_make_node_values(self, info: Dict[str, Any], nv) -> List[float]:
        return [self._get_voltage(node, nv) for node in info.get("node_names", ())]

    def _rust_event_write_shadow_begin(self, key: str, nv) -> Optional[Tuple[str, List[float]]]:
        if not self._rust_event_write_shadow_enabled:
            return None
        info = self._rust_event_write_batches.get(str(key))
        if info is None:
            return None
        backend = self._rust_event_write_backend
        state_values = self._indexed_state_values
        if backend is None or state_values is None:
            return None
        expected = list(state_values)
        try:
            node_values = self._rust_event_write_make_node_values(info, nv)
            backend.event_lfsr_shift_xor_step(info["batch"], node_values, expected)
            self._perf_stats["rust_event_write_shadow_checks"] += 1
            return str(key), expected
        except Exception:
            self._perf_stats["rust_event_write_shadow_errors"] += 1
            return None

    def _rust_event_write_shadow_end(self, key: str, shadow) -> None:
        if shadow is None:
            return
        info = self._rust_event_write_batches.get(str(key))
        state_values = self._indexed_state_values
        if info is None or state_values is None:
            return
        _, expected = shadow
        mismatch = False
        for slot in info["target_slots"]:
            if slot >= len(state_values) or slot >= len(expected):
                mismatch = True
                break
            if abs(float(state_values[slot]) - float(expected[slot])) > 1e-12:
                mismatch = True
                break
        if mismatch:
            self._perf_stats["rust_event_write_shadow_mismatches"] += 1
        else:
            self._perf_stats["rust_event_write_shadow_matches"] += 1

    def _rust_event_write_production(self, key: str, nv) -> bool:
        if not self._rust_event_write_production_enabled:
            return False
        info = self._rust_event_write_batches.get(str(key))
        if info is None:
            return False
        backend = self._rust_event_write_backend
        state_values = self._indexed_state_values
        if backend is None or state_values is None:
            self._perf_stats["rust_event_write_production_fallbacks"] += 1
            return False
        try:
            node_values = self._rust_event_write_make_node_values(info, nv)
            executed = backend.event_lfsr_shift_xor_step(
                info["batch"],
                node_values,
                state_values,
            )
        except Exception:
            self._perf_stats["rust_event_write_production_fallbacks"] += 1
            return False
        self._perf_stats["rust_event_write_production_calls"] += 1
        if executed:
            self._perf_stats["rust_event_write_production_executed"] += 1
        self._commit_indexed_state_slots(info["target_slots"])
        return True

    def _rust_event_linear_write_shadow_begin(self, key: str) -> Optional[Tuple[str, List[float]]]:
        if not self._rust_event_write_shadow_enabled:
            return None
        info = self._rust_event_linear_write_batches.get(str(key))
        if info is None:
            return None
        if info.get("uses_node_sources") and self._event_interpolated_nodes:
            return None
        backend = self._rust_event_write_backend
        node_values = self._rust_event_write_node_values
        state_values = self._indexed_state_values
        if backend is None or node_values is None or state_values is None:
            return None
        expected = list(state_values)
        try:
            backend.evaluate_static_linear(info["batch"], node_values, expected)
        except Exception:
            self._perf_stats["rust_event_write_shadow_errors"] += 1
            self._perf_stats["rust_event_linear_write_shadow_errors"] += 1
            return None
        self._perf_stats["rust_event_write_shadow_checks"] += 1
        self._perf_stats["rust_event_linear_write_shadow_checks"] += 1
        return str(key), expected

    def _rust_event_linear_write_shadow_end(self, key: str, shadow) -> None:
        if shadow is None:
            return
        info = self._rust_event_linear_write_batches.get(str(key))
        state_values = self._indexed_state_values
        if info is None or state_values is None:
            return
        _, expected = shadow
        mismatch = False
        for slot in info["target_slots"]:
            if slot >= len(state_values) or slot >= len(expected):
                mismatch = True
                break
            if abs(float(state_values[slot]) - float(expected[slot])) > 1e-12:
                mismatch = True
                break
        if mismatch:
            self._perf_stats["rust_event_write_shadow_mismatches"] += 1
            self._perf_stats["rust_event_linear_write_shadow_mismatches"] += 1
        else:
            self._perf_stats["rust_event_write_shadow_matches"] += 1
            self._perf_stats["rust_event_linear_write_shadow_matches"] += 1

    def _rust_event_linear_write_production(self, key: str) -> bool:
        if not self._rust_event_write_production_enabled or self._event_trace_audit_enabled:
            return False
        info = self._rust_event_linear_write_batches.get(str(key))
        if info is None:
            return False
        if info.get("uses_node_sources") and self._event_interpolated_nodes:
            source_node_pairs = tuple(info.get("source_node_pairs", ()) or ())
            if not source_node_pairs:
                return False
            if any(name not in self._event_interpolated_node_values for name, _node_id in source_node_pairs):
                return False
        backend = self._rust_event_write_backend
        node_values = self._rust_event_write_node_values
        state_values = self._indexed_state_values
        if backend is None or node_values is None or state_values is None:
            self._perf_stats["rust_event_write_production_fallbacks"] += 1
            self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
            return False
        restore_node_values = []
        if info.get("uses_node_sources") and self._event_interpolated_nodes:
            for node_name, node_id in tuple(info.get("source_node_pairs", ()) or ()):
                if node_id < 0 or node_id >= len(node_values):
                    self._perf_stats["rust_event_write_production_fallbacks"] += 1
                    self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
                    return False
                restore_node_values.append((node_id, float(node_values[node_id])))
                node_values[node_id] = float(self._event_interpolated_node_values[node_name])
        try:
            backend.evaluate_static_linear(info["batch"], node_values, state_values)
        except Exception:
            self._perf_stats["rust_event_write_production_fallbacks"] += 1
            self._perf_stats["rust_event_linear_write_production_fallbacks"] += 1
            return False
        finally:
            for node_id, value in restore_node_values:
                node_values[node_id] = value
        self._perf_stats["rust_event_write_production_calls"] += 1
        self._perf_stats["rust_event_write_production_executed"] += 1
        self._perf_stats["rust_event_linear_write_production_calls"] += 1
        self._perf_stats["rust_event_linear_write_production_executed"] += 1
        self._commit_indexed_state_slots(info["target_slots"])
        return True

    def _rust_timer_lfsr_output_production(
        self,
        key: str,
        nv,
        time: float,
        period: float,
        start: Optional[float] = None,
    ) -> bool:
        if (
            not self._rust_timer_event_production_enabled
            or not self._rust_event_write_production_enabled
            or self._event_trace_audit_enabled
        ):
            return False
        info = self._rust_timer_lfsr_output_batches.get(str(key))
        if info is None:
            return False
        backend = self._rust_event_write_backend
        state_values = self._indexed_state_values
        node_values = self._rust_event_write_node_values
        if backend is None or state_values is None or node_values is None:
            self._perf_stats["rust_timer_lfsr_output_fallbacks"] += 1
            return False

        self._perf_stats["timer_periodic_checks"] += 1
        self.timer_kinds[key] = "periodic"
        before_has_state = key in self.timer_states
        before_next_fire = float(self.timer_states.get(key, 0.0))
        next_fire_times = array("d", [before_next_fire])
        has_state_flags = array("B", [1 if before_has_state else 0])
        try:
            due, skipped, executed, output_written = backend.timer_lfsr_output_step(
                info["batch"],
                node_values,
                state_values,
                next_fire_times,
                has_state_flags,
                period,
                float(start) if start is not None else 0.0,
                start is not None,
                time,
                eps=1e-18,
            )
        except Exception:
            self._perf_stats["rust_timer_lfsr_output_fallbacks"] += 1
            return False

        self._perf_stats["rust_timer_lfsr_output_calls"] += 1
        if skipped:
            self._perf_stats["timer_periodic_skips"] += 1
            self._perf_stats["rust_timer_lfsr_output_skips"] += 1
        if bool(has_state_flags[0]):
            next_fire = float(next_fire_times[0])
            if (
                not before_has_state
                or abs(next_fire - before_next_fire) > 1e-18
            ):
                self._set_timer_state(key, next_fire)
        if due:
            self._event_time = time
            self._event_interpolated_nodes = set()
            self._event_interpolated_node_values = {}
            self._event_node_cross_directions = {}
            self._perf_stats["timer_periodic_fires"] += 1
            self._perf_stats["timer_reschedules"] += 1
            self._perf_stats["rust_timer_lfsr_output_due"] += 1
            self._perf_stats["rust_timer_event_production_fires"] += 1
        if executed:
            self._perf_stats["rust_timer_lfsr_output_executed"] += 1
            self._perf_stats["rust_event_write_production_calls"] += 1
            self._perf_stats["rust_event_write_production_executed"] += 1
            self._commit_indexed_state_slots(info["target_slots"])
        if output_written:
            output_node_id = int(info["output_node_id"])
            output_value = float(node_values[output_node_id])
            local_node = str(info["output_local_node"])
            external_node = str(info["output_external_node"])
            if local_node not in self.output_nodes:
                self._output_nodes_version += 1
            self.output_nodes[local_node] = output_value
            nv[external_node] = output_value
            if self._indexed_output_writer is not None:
                self._indexed_output_writer(external_node, output_value)
            self._perf_stats["rust_timer_lfsr_output_writes"] += 1
        return True

    def _rust_state_output_hold_production(
        self,
        state_name: str,
        output_node: str,
        nv,
    ) -> bool:
        if (
            not self._rust_timer_event_production_enabled
            or not self._rust_event_write_production_enabled
            or self._event_trace_audit_enabled
        ):
            return False
        info = self._rust_timer_lfsr_output_hold_by_state.get(str(state_name))
        if info is None or str(output_node) != str(info["output_local_node"]):
            return False
        state_values = self._indexed_state_values
        node_values = self._rust_event_write_node_values
        state_slot = self._indexed_state_ids.get(str(state_name))
        if state_values is None or node_values is None or state_slot is None:
            return False
        if state_slot < 0 or state_slot >= len(state_values):
            return False
        output_node_id = int(info["output_node_id"])
        if output_node_id < 0 or output_node_id >= len(node_values):
            return False
        output_value = float(state_values[state_slot])
        node_values[output_node_id] = output_value
        local_node = str(info["output_local_node"])
        external_node = str(info["output_external_node"])
        current_nv = nv.get(external_node)
        current_output = self.output_nodes.get(local_node)
        if (
            current_nv is not None
            and current_output is not None
            and abs(float(current_nv) - output_value) <= 1e-18
            and abs(float(current_output) - output_value) <= 1e-18
        ):
            return True
        if local_node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[local_node] = output_value
        nv[external_node] = output_value
        if self._indexed_output_writer is not None:
            self._indexed_output_writer(external_node, output_value)
        self._perf_stats["rust_timer_lfsr_output_writes"] += 1
        return True

    def _commit_indexed_state_slots(self, slots) -> None:
        values = self._indexed_state_values
        if values is None:
            return
        slot_set = set(int(slot) for slot in slots)
        for name, slot in self._indexed_state_ids.items():
            if slot not in slot_set or slot < 0 or slot >= len(values):
                continue
            value = values[slot]
            array_ref = self._indexed_state_array_slot_refs.get(name)
            if array_ref is not None:
                array_name, idx = array_ref
                layout = self._indexed_state_array_layouts.get(array_name)
                if layout is None:
                    continue
                lo, hi, integer = layout
                if not (lo <= idx <= hi):
                    continue
                stored = self._to_integer(value) if integer else value
                if array_name not in self.arrays:
                    self.arrays[array_name] = {}
                self.arrays[array_name][idx] = stored
                self._indexed_state_array_values[array_name][idx - lo] = float(stored)
                continue
            if name in self._indexed_integer_state_names:
                self.state[name] = self._to_integer(value)
            else:
                self.state[name] = value

    def _set_event_trace_audit_enabled(self, enabled: bool, max_records: int = 4096):
        """Enable a bounded trace of event firing order and event-body writes."""
        self._event_trace_audit_enabled = bool(enabled)
        self._event_trace_audit_max_records = max(0, int(max_records))
        if enabled:
            self._event_trace_audit_records = []
            self._event_trace_audit_context_stack = []
            self._event_trace_audit_write_targets = set()
            self._event_trace_audit_seq = 0
        else:
            self._event_trace_audit_context_stack = []
        for child in self._child_models:
            child._set_event_trace_audit_enabled(enabled, max_records)

    def _event_trace_audit_append(self, record: Dict[str, Any]):
        if not self._event_trace_audit_enabled:
            return
        if len(self._event_trace_audit_records) >= self._event_trace_audit_max_records:
            self._perf_stats["event_trace_audit_records_dropped"] += 1
            return
        self._event_trace_audit_seq += 1
        record["seq"] = self._event_trace_audit_seq
        self._event_trace_audit_records.append(record)

    def _event_trace_audit_record_event(
        self,
        kind: str,
        key: str,
        time: float,
        event_time: Optional[float] = None,
        phase: Optional[str] = None,
    ):
        if not self._event_trace_audit_enabled:
            return
        event_kind = str(kind)
        self._perf_stats["event_trace_audit_events"] += 1
        if event_kind.startswith("cross"):
            self._perf_stats["event_trace_audit_cross_events"] += 1
        elif event_kind.startswith("above"):
            self._perf_stats["event_trace_audit_above_events"] += 1
        elif event_kind.startswith("timer"):
            self._perf_stats["event_trace_audit_timer_events"] += 1
        elif event_kind == "initial_step":
            self._perf_stats["event_trace_audit_initial_step_events"] += 1
        elif event_kind == "final_step":
            self._perf_stats["event_trace_audit_final_step_events"] += 1
        elif event_kind == "combined":
            self._perf_stats["event_trace_audit_combined_events"] += 1
        evt_t = self._event_time if event_time is None else event_time
        self._event_trace_audit_append(
            {
                "type": "event",
                "phase": phase or self._event_trace_audit_phase,
                "kind": event_kind,
                "key": str(key),
                "time": float(time),
                "event_time": float(evt_t),
            }
        )

    def _event_trace_audit_enter_event(
        self,
        kind: str,
        key: str,
        time: float,
        event_time: Optional[float] = None,
        phase: Optional[str] = None,
    ):
        if not self._event_trace_audit_enabled:
            return
        self._perf_stats["event_trace_audit_body_entries"] += 1
        context = {
            "phase": phase or self._event_trace_audit_phase,
            "kind": str(kind),
            "key": str(key),
            "time": float(time),
            "event_time": float(self._event_time if event_time is None else event_time),
            "writes": {},
        }
        if kind in {"initial_step", "final_step", "combined"}:
            self._event_trace_audit_record_event(
                str(kind),
                str(key),
                float(time),
                context["event_time"],
                context["phase"],
            )
        self._event_trace_audit_context_stack.append(context)
        self._event_trace_audit_append({"type": "body_enter", **context})

    def _event_trace_audit_exit_event(self):
        if not self._event_trace_audit_enabled:
            return
        if not self._event_trace_audit_context_stack:
            return
        context = self._event_trace_audit_context_stack.pop()
        self._event_trace_audit_append({"type": "body_exit", **context})

    def _event_trace_audit_note_write(self, category: str, target: str):
        if not self._event_trace_audit_enabled:
            return
        cat = str(category)
        target_s = str(target)
        stat_key = f"event_trace_audit_{cat}_writes"
        if stat_key in self._perf_stats:
            self._perf_stats[stat_key] += 1
        self._event_trace_audit_write_targets.add(f"{cat}:{target_s}")
        in_event = bool(self._event_context_active or self._event_trace_audit_context_stack)
        if in_event:
            self._perf_stats["event_trace_audit_in_event_writes"] += 1
            target_key = f"event_trace_audit_target::{cat}::{target_s}"
            self._perf_stats[target_key] = int(self._perf_stats.get(target_key, 0) or 0) + 1
        if self._event_trace_audit_context_stack:
            context = self._event_trace_audit_context_stack[-1]
            writes = self._event_trace_audit_context_stack[-1]["writes"]
            writes[cat] = int(writes.get(cat, 0)) + 1
            body_key = (
                "event_trace_audit_body::"
                f"{context['kind']}::{context['key']}::{cat}_writes"
            )
            self._perf_stats[body_key] = int(self._perf_stats.get(body_key, 0) or 0) + 1
        self._event_trace_audit_append(
            {
                "type": "write",
                "phase": self._event_trace_audit_phase,
                "category": cat,
                "target": target_s,
                "time": float(self._step_time if self._step_time else self._event_time),
                "event_time": float(self._event_time),
                "in_event": in_event,
            }
        )

    def _mark_transition_known_inactive(self, key: str, was_active: bool = False):
        """Record that a transition key is known but currently inactive."""
        self._transition_active_keys_known = True
        if was_active:
            self._active_transition_keys.discard(key)

    def _track_transition_active_change(
        self,
        key: str,
        was_active: bool,
        is_active: bool,
    ):
        """Update active transition membership only when the state changed."""
        self._transition_active_keys_known = True
        if was_active == is_active:
            return
        if is_active:
            self._active_transition_keys.add(key)
        else:
            self._active_transition_keys.discard(key)

    def _set_static_branch_indexed_io(
        self,
        read_node_ids: tuple[int, ...] = (),
        write_node_ids: tuple[int, ...] = (),
        write_external_nodes: tuple[str, ...] = (),
        values: Optional[List[float]] = None,
    ):
        """Install run-local node-id bindings for static branch helpers."""
        self._static_branch_read_node_ids = tuple(read_node_ids)
        self._static_branch_write_node_ids = tuple(write_node_ids)
        self._static_branch_write_external_nodes = tuple(write_external_nodes)
        self._static_branch_indexed_values = values

    def _set_node_resolution_cache_enabled(self, enabled: bool):
        """Cache local-to-external node mapping for the duration of a run."""
        self._node_resolution_cache_enabled = bool(enabled)
        self._node_resolution_cache.clear()
        for child in self._child_models:
            child._set_node_resolution_cache_enabled(enabled)

    def _resolve_external_node_uncached(self, node: str) -> str:
        """Resolve a local model node through node_map and one parent indirection."""
        ext = self.node_map.get(node, node)
        if (
            isinstance(ext, str)
            and ext
            and ext[0] == '@'
            and ext.startswith('@parent:')
            and self._parent_model is not None
        ):
            pnode = ext[len('@parent:'):]
            ext = self._parent_model.node_map.get(pnode, pnode)
        return ext

    def _resolve_external_node(self, node: str) -> str:
        """Resolve a node name, optionally reusing the run-local mapping cache."""
        if not self.node_map:
            return node
        if not self._node_resolution_cache_enabled:
            return self._resolve_external_node_uncached(node)
        ext = self._node_resolution_cache.get(node)
        if ext is None:
            ext = self._resolve_external_node_uncached(node)
            self._node_resolution_cache[node] = ext
        return ext

    @staticmethod
    def _format_dynamic_node(base: str, index: Any, index2: Any = None) -> str:
        """Format dynamic bus node names without generated nested f-strings."""
        if index2 is None:
            return f"{base}[{int(index)}]"
        return f"{base}[{int(index)}][{int(index2)}]"

    def _resolve_dynamic_node(self, base: str, index: Any, index2: Any = None) -> str:
        """Resolve a dynamic bus node using a run-local base/index cache."""
        idx = int(index)
        idx2 = None if index2 is None else int(index2)
        key = (base, idx, idx2)
        node = self._dynamic_node_cache.get(key)
        if node is not None:
            self._perf_stats["dynamic_node_cache_hits"] += 1
            return node
        if len(self._dynamic_node_cache) >= self._dynamic_node_cache_limit:
            self._perf_stats["dynamic_node_cache_bypasses"] += 1
            if idx2 is None:
                return f"{base}[{idx}]"
            return f"{base}[{idx}][{idx2}]"
        if idx2 is None:
            node = f"{base}[{idx}]"
        else:
            node = f"{base}[{idx}][{idx2}]"
        self._dynamic_node_cache[key] = node
        self._perf_stats["dynamic_node_cache_misses"] += 1
        return node

    def _should_update_discrete_state(self, key: str, time: float) -> bool:
        """Gate self-referential continuous real updates to the nominal tran grid.

        Spectre accepts simple behavioral state such as ``x = x - decay`` in an
        analog block. EVAS may insert extra internal refine steps near events;
        without this gate, those implementation-detail steps change the user
        model's state trajectory. Event bodies remain ungated.
        """
        if self._event_context_active:
            return True
        step = float(getattr(self, "_nominal_step", 0.0) or 0.0)
        if step <= 0.0:
            return True
        bucket = int(math.floor((float(time) + 1e-18) / step))
        if self._discrete_update_buckets.get(key) == bucket:
            return False
        self._discrete_update_buckets[key] = bucket
        return True

    def _prepare_step(self, prev_node_voltages: Dict[str, float],
                      curr_node_voltages: Dict[str, float],
                      prev_time: float, time: float,
                      future_node_voltages: Optional[Dict[str, float]] = None):
        """Cache step endpoints so event bodies can read values at t_cross."""
        self._step_prev_node_voltages = prev_node_voltages
        self._step_curr_node_voltages = curr_node_voltages
        if self._needs_future_node_voltages and future_node_voltages is not None:
            with_fallback = getattr(future_node_voltages, "_with_fallback", None)
            if with_fallback is not None:
                self._step_future_node_voltages = with_fallback(self._step_curr_node_voltages)
            else:
                self._step_future_node_voltages = dict(future_node_voltages)
        else:
            self._step_future_node_voltages = self._step_curr_node_voltages
        self._step_prev_time = float(prev_time)
        self._step_time = float(time)
        self._step_latest_cross_event_time = -math.inf
        self._step_event_fired = False
        self._event_interpolated_node_values = {}
        for child in self._child_models:
            child_future_node_voltages = (
                future_node_voltages
                if child._needs_future_node_voltages_tree()
                else None
            )
            child._prepare_step(
                prev_node_voltages,
                curr_node_voltages,
                prev_time,
                time,
                child_future_node_voltages,
            )

    def _needs_future_node_voltages_tree(self) -> bool:
        cached = self._needs_future_node_voltages_tree_cache
        if cached is not None:
            return cached
        cached = bool(self._needs_future_node_voltages) or any(
            child._needs_future_node_voltages_tree()
            for child in self._child_models
        )
        self._needs_future_node_voltages_tree_cache = cached
        return cached

    def _has_dynamic_breakpoints_tree(self) -> bool:
        cached = self._has_dynamic_breakpoints_tree_cache
        if cached is not None:
            return cached
        own = bool(getattr(self, "_has_dynamic_breakpoints", True))
        if type(self).next_breakpoint is not CompiledModel.next_breakpoint:
            own = True
        cached = own or any(
            child._has_dynamic_breakpoints_tree()
            for child in self._child_models
        )
        self._has_dynamic_breakpoints_tree_cache = cached
        return cached

    def _uses_bound_step_tree(self) -> bool:
        cached = self._uses_bound_step_tree_cache
        if cached is not None:
            return cached
        cached = bool(getattr(self, "_uses_bound_step", True)) or any(
            child._uses_bound_step_tree()
            for child in self._child_models
        )
        self._uses_bound_step_tree_cache = cached
        return cached

    def next_breakpoint(self, time: float) -> Optional[float]:
        best: Optional[float] = None
        min_ramp = getattr(self, "_transition_breakpoint_min_ramp", 0.0)
        if self._transition_active_keys_known:
            transition_keys = tuple(self._active_transition_keys)
            inactive_count = max(0, len(self.transitions) - len(transition_keys))
            self._perf_stats["transition_breakpoint_inactive_skips"] += inactive_count
        else:
            # Preserve direct-test/backward-compat behavior for manually planted
            # TransitionState objects that did not pass through _transition().
            transition_keys = tuple(self.transitions.keys())
        scanner = self._rust_transition_breakpoint_scanner
        if scanner is not None and self._transition_active_keys_known:
            self._perf_stats["transition_breakpoint_scans"] += len(transition_keys)
            start_times = array("d")
            start_values = array("d")
            target_values = array("d")
            delays = array("d")
            rise_times = array("d")
            fall_times = array("d")
            active_states: List[TransitionState] = []
            for key in transition_keys:
                ts = self.transitions.get(key)
                if ts is None:
                    self._active_transition_keys.discard(key)
                    continue
                if not ts.active:
                    self._active_transition_keys.discard(key)
                    continue
                active_states.append(ts)
                start_times.append(float(ts.start_time))
                start_values.append(float(ts.start_val))
                target_values.append(float(ts.target_val))
                delays.append(float(ts.delay))
                rise_times.append(float(ts.rise_time))
                fall_times.append(float(ts.fall_time))

            self._perf_stats["transition_breakpoint_active_scans"] += len(active_states)
            if active_states:
                self._perf_stats["rust_transition_breakpoint_scans"] += 1
                self._perf_stats["rust_transition_breakpoint_state_scans"] += len(
                    active_states
                )
                try:
                    bp = scanner(
                        start_times,
                        start_values,
                        target_values,
                        delays,
                        rise_times,
                        fall_times,
                        [1] * len(active_states),
                        time,
                        min_ramp,
                    )
                except Exception:
                    self._perf_stats["rust_transition_breakpoint_fallbacks"] += 1
                    bp = None
                    for ts in active_states:
                        candidate = ts.next_breakpoint(time, min_ramp)
                        if candidate is not None and (
                            bp is None or candidate < bp
                        ):
                            bp = candidate
                if bp is not None and (best is None or bp < best):
                    best = bp
        else:
            self._perf_stats["transition_breakpoint_scans"] += len(transition_keys)
            for key in transition_keys:
                ts = self.transitions.get(key)
                if ts is None:
                    self._active_transition_keys.discard(key)
                    continue
                if ts.active:
                    self._perf_stats["transition_breakpoint_active_scans"] += 1
                bp = ts.next_breakpoint(time, min_ramp)
                if bp is None and self._transition_active_keys_known and not ts.active:
                    self._active_transition_keys.discard(key)
                if bp is not None and (best is None or bp < best):
                    best = bp
        for cd in self.cross_detectors.values():
            bp = cd.next_breakpoint()
            if bp is not None and bp > time and (best is None or bp < best):
                best = bp
        for ad in self.above_detectors.values():
            bp = ad.next_breakpoint()
            if bp is not None and bp > time and (best is None or bp < best):
                best = bp
        bp = self._next_timer_breakpoint(time)
        if bp is not None and (best is None or bp < best):
            best = bp
        for child in self._child_models:
            bp = child.next_breakpoint(time)
            if bp is not None and (best is None or bp < best):
                best = bp
        return best

    def _has_transition_target_probes_tree(self) -> bool:
        own = int(getattr(self.__class__, "_transition_target_probe_count", 0)) > 0
        return own or any(
            child._has_transition_target_probes_tree()
            for child in self._child_models
        )

    def _transition_target_probe_values(
        self,
        nv: Dict[str, float],
        time: float,
    ) -> Tuple[float, ...]:
        return ()

    def transition_target_breakpoint(
        self,
        nv: Dict[str, float],
        time: float,
        dt: float,
    ) -> Optional[float]:
        """Predict a step-local breakpoint for discontinuous transition targets.

        The normal transition state scanner only sees breakpoints after
        transition() has been called with a new target.  For targets such as
        ``transition(phase <= tw ? 1 : 0, ...)``, the target may flip inside the
        proposed step.  Spectre-style transient engines shrink the step to that
        discontinuity before updating the transition state; this probe mirrors
        the side-effect-free continuous assignments and detects that flip.
        """
        if dt <= 0.0:
            return None

        best = self._transition_target_breakpoint_local(nv, time, dt)
        for child in self._child_models:
            bp = child.transition_target_breakpoint(nv, time, dt)
            if bp is not None and bp > time and bp < time + dt:
                if best is None or bp < best:
                    best = bp
        return best

    def _transition_target_breakpoint_local(
        self,
        nv: Dict[str, float],
        time: float,
        dt: float,
    ) -> Optional[float]:
        if int(getattr(self.__class__, "_transition_target_probe_count", 0)) <= 0:
            return None
        try:
            start_values = tuple(float(v) for v in self._transition_target_probe_values(nv, time))
            end_values = tuple(float(v) for v in self._transition_target_probe_values(nv, time + dt))
        except Exception:
            return None
        if len(start_values) != len(end_values) or not start_values:
            return None

        eps = 1.0e-15
        best: Optional[float] = None
        for idx, (start, end) in enumerate(zip(start_values, end_values)):
            if not (math.isfinite(start) and math.isfinite(end)):
                continue
            if abs(start - end) <= eps:
                continue
            lo = time
            hi = time + dt
            start_ref = start
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                try:
                    mid_values = self._transition_target_probe_values(nv, mid)
                    mid_val = float(mid_values[idx])
                except Exception:
                    break
                if math.isfinite(mid_val) and abs(mid_val - start_ref) <= eps:
                    lo = mid
                else:
                    hi = mid
            min_sep = max(1.0e-18, abs(dt) * 1.0e-12)
            candidate = max(hi, time + min_sep)
            if candidate > time and candidate < time + dt:
                if best is None or candidate < best:
                    best = candidate
        return best

    def _invalidate_timer_breakpoint_cache(self):
        self._timer_state_version += 1
        self._timer_breakpoint_cache_version = -1
        self._timer_breakpoint_cache = None
        self._perf_stats["timer_state_updates"] += 1

    def _timer_slot_for_key(self, key: str) -> int:
        slot = self._timer_key_ids.get(key)
        if slot is not None:
            return slot
        slot = len(self._timer_keys_by_id)
        self._timer_key_ids[key] = slot
        self._timer_keys_by_id.append(key)
        self._timer_next_fire_values.append(float(self.timer_states.get(key, 0.0)))
        last_fired = self.timer_last_fired.get(key)
        if last_fired is None:
            self._timer_last_fired_values.append(0.0)
            self._timer_has_last_fired_flags.append(0)
        else:
            self._timer_last_fired_values.append(float(last_fired))
            self._timer_has_last_fired_flags.append(1)
        return slot

    def _sync_timer_array_sidecar_from_dict(self) -> None:
        keys = tuple(self.timer_states.keys())
        needs_rebuild = len(keys) != len(self._timer_keys_by_id)
        if not needs_rebuild:
            for idx, key in enumerate(keys):
                if self._timer_keys_by_id[idx] != key:
                    needs_rebuild = True
                    break
        if needs_rebuild:
            self._timer_key_ids = {}
            self._timer_keys_by_id = []
            self._timer_next_fire_values = array("d")
            self._timer_last_fired_values = array("d")
            self._timer_has_last_fired_flags = array("B")
            for key in keys:
                self._timer_slot_for_key(key)
            self._timer_array_sidecar_version = self._timer_state_version
            self._perf_stats["timer_array_sidecar_rebuilds"] += 1
            return

        if self._timer_array_sidecar_version == self._timer_state_version:
            return

        for slot, key in enumerate(self._timer_keys_by_id):
            self._timer_next_fire_values[slot] = float(self.timer_states[key])
            last_fired = self.timer_last_fired.get(key)
            if last_fired is None:
                self._timer_last_fired_values[slot] = 0.0
                self._timer_has_last_fired_flags[slot] = 0
            else:
                self._timer_last_fired_values[slot] = float(last_fired)
                self._timer_has_last_fired_flags[slot] = 1
        self._timer_array_sidecar_version = self._timer_state_version
        self._perf_stats["timer_array_sidecar_updates"] += len(
            self._timer_keys_by_id
        )

    def _set_timer_state(self, key: str, value: float):
        self.timer_states[key] = float(value)
        self._event_trace_audit_note_write("timer_state", key)
        self._invalidate_timer_breakpoint_cache()
        slot = self._timer_slot_for_key(key)
        self._timer_next_fire_values[slot] = float(value)
        self._timer_array_sidecar_version = self._timer_state_version
        self._perf_stats["timer_array_sidecar_updates"] += 1

    def _set_timer_last_fired(self, key: str, value: float):
        self.timer_last_fired[key] = float(value)
        self._event_trace_audit_note_write("timer_last_fired", key)
        self._invalidate_timer_breakpoint_cache()
        slot = self._timer_slot_for_key(key)
        self._timer_last_fired_values[slot] = float(value)
        self._timer_has_last_fired_flags[slot] = 1
        self._timer_array_sidecar_version = self._timer_state_version
        self._perf_stats["timer_array_sidecar_updates"] += 1

    @staticmethod
    def _shadow_float_close(left: float, right: float, tol: float) -> bool:
        left = float(left)
        right = float(right)
        if not math.isfinite(left) or not math.isfinite(right):
            return left == right
        return abs(left - right) <= max(float(tol), 1e-15 * max(abs(left), abs(right), 1.0))

    def _rust_event_due_shadow_record(
        self,
        kind: str,
        matched: bool,
        time_diff: float = 0.0,
    ) -> None:
        if matched:
            self._perf_stats["rust_event_due_shadow_matches"] += 1
        else:
            self._perf_stats["rust_event_due_shadow_mismatches"] += 1
        if time_diff > self._perf_stats["rust_event_due_shadow_max_time_diff"]:
            self._perf_stats["rust_event_due_shadow_max_time_diff"] = float(time_diff)

    def _check_cross_rust_production(
        self,
        detector: CrossDetector,
        before: Tuple[float, float, float, float, int, int, float, float],
        time: float,
        val: float,
        time_tol: float,
        expr_tol: float,
    ) -> Optional[bool]:
        """Audit 089: production Rust ownership of CrossDetector state evolution.

        Returns:
            True/False indicating whether the detector fired, OR None if the
            production backend is unavailable / failed (caller should fall
            back to Python detector.check()).
        """
        backend = self._rust_cross_above_production_backend
        if backend is None or not self._rust_cross_above_production_enabled:
            return None
        stats = self._perf_stats
        try:
            prev_values = array("d", [before[0]])
            prev_times = array("d", [before[1]])
            pprev_values = array("d", [before[2]])
            pprev_times = array("d", [before[3]])
            initialized_flags = array("B", [before[4]])
            directions = array("i", [before[5]])
            last_cross_times = array("d", [before[6]])
            current_values = array("d", [float(val)])
            triggered_flags = array("B", [0])
            cross_times = array("d", [before[7]])
            trigger_directions = array("i", [0])
            went_beyond_flags = array("B", [0])
            backend.cross_detector_step(
                prev_values,
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
                time,
                time_tol=time_tol,
                expr_tol=expr_tol,
            )
        except Exception:
            stats["rust_cross_production_fallbacks"] += 1
            return None
        # Apply Rust's results to the Python detector object so the rest of
        # _check_cross sees identical state to what detector.check() would
        # have produced.
        fired = bool(triggered_flags[0])
        detector.prev_val = float(prev_values[0])
        detector.prev_time = float(prev_times[0])
        detector.pprev_val = float(pprev_values[0])
        detector.pprev_time = float(pprev_times[0])
        detector.initialized = bool(initialized_flags[0])
        detector.last_cross_time = float(last_cross_times[0])
        detector.t_cross = float(cross_times[0])
        detector.last_triggered = fired
        detector.last_trigger_direction = int(trigger_directions[0])
        detector.last_trigger_went_beyond = bool(went_beyond_flags[0])
        stats["rust_cross_production_calls"] += 1
        if fired:
            stats["rust_cross_production_fires"] += 1
        return fired

    def _check_above_rust_production(
        self,
        detector: AboveDetector,
        before: Tuple[float, float, float, float, int, int, float],
        time: float,
        val: float,
    ) -> Optional[bool]:
        """Audit 089: production Rust ownership of AboveDetector state evolution."""
        backend = self._rust_cross_above_production_backend
        if backend is None or not self._rust_cross_above_production_enabled:
            return None
        stats = self._perf_stats
        try:
            prev_values = array("d", [before[0]])
            prev_times = array("d", [before[1]])
            pprev_values = array("d", [before[2]])
            pprev_times = array("d", [before[3]])
            initialized_flags = array("B", [before[4]])
            directions = array("i", [before[5]])
            current_values = array("d", [float(val)])
            triggered_flags = array("B", [0])
            cross_times = array("d", [before[6]])
            backend.above_detector_step(
                prev_values,
                prev_times,
                pprev_values,
                pprev_times,
                initialized_flags,
                directions,
                current_values,
                triggered_flags,
                cross_times,
                time,
            )
        except Exception:
            stats["rust_above_production_fallbacks"] += 1
            return None
        fired = bool(triggered_flags[0])
        detector.prev_val = float(prev_values[0])
        detector.prev_time = float(prev_times[0])
        detector.pprev_val = float(pprev_values[0])
        detector.pprev_time = float(pprev_times[0])
        detector.initialized = bool(initialized_flags[0])
        detector.t_cross = float(cross_times[0])
        stats["rust_above_production_calls"] += 1
        if fired:
            stats["rust_above_production_fires"] += 1
        return fired

    @staticmethod
    def _cross_detector_shadow_state(detector: CrossDetector) -> Tuple[float, float, float, float, int, int, float, float]:
        return (
            float(detector.prev_val),
            float(detector.prev_time),
            float(detector.pprev_val),
            float(detector.pprev_time),
            1 if detector.initialized else 0,
            int(detector.direction),
            float(detector.last_cross_time),
            float(detector.t_cross),
        )

    @staticmethod
    def _above_detector_shadow_state(detector: AboveDetector) -> Tuple[float, float, float, float, int, int, float]:
        return (
            float(detector.prev_val),
            float(detector.prev_time),
            float(detector.pprev_val),
            float(detector.pprev_time),
            1 if detector.initialized else 0,
            int(detector.direction),
            float(detector.t_cross),
        )

    def _rust_shadow_check_cross(
        self,
        before: Tuple[float, float, float, float, int, int, float, float],
        detector: CrossDetector,
        fired: bool,
        time: float,
        val: float,
        time_tol: float,
        expr_tol: float,
    ) -> None:
        backend = self._rust_event_due_shadow_backend
        if backend is None:
            return
        self._perf_stats["rust_event_due_shadow_cross_checks"] += 1
        try:
            prev_values = array("d", [before[0]])
            prev_times = array("d", [before[1]])
            pprev_values = array("d", [before[2]])
            pprev_times = array("d", [before[3]])
            initialized_flags = array("B", [before[4]])
            directions = array("i", [before[5]])
            last_cross_times = array("d", [before[6]])
            current_values = array("d", [float(val)])
            triggered_flags = array("B", [0])
            cross_times = array("d", [before[7]])
            trigger_directions = array("i", [0])
            went_beyond_flags = array("B", [0])
            backend.cross_detector_step(
                prev_values,
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
                time,
                time_tol=time_tol,
                expr_tol=expr_tol,
            )
        except Exception:
            self._perf_stats["rust_event_due_shadow_errors"] += 1
            return

        time_diff = max(
            abs(float(prev_times[0]) - float(detector.prev_time)),
            abs(float(pprev_times[0]) - float(detector.pprev_time)),
            abs(float(last_cross_times[0]) - float(detector.last_cross_time)),
        )
        if fired:
            time_diff = max(time_diff, abs(float(cross_times[0]) - float(detector.t_cross)))
        matched = (
            bool(triggered_flags[0]) == bool(fired)
            and int(trigger_directions[0]) == int(detector.last_trigger_direction)
            and bool(went_beyond_flags[0]) == bool(detector.last_trigger_went_beyond)
            and int(initialized_flags[0]) == (1 if detector.initialized else 0)
            and self._shadow_float_close(prev_values[0], detector.prev_val, max(float(expr_tol or 0.0), 1e-12))
            and self._shadow_float_close(pprev_values[0], detector.pprev_val, max(float(expr_tol or 0.0), 1e-12))
            and self._shadow_float_close(prev_times[0], detector.prev_time, 1e-18)
            and self._shadow_float_close(pprev_times[0], detector.pprev_time, 1e-18)
            and self._shadow_float_close(last_cross_times[0], detector.last_cross_time, max(float(time_tol or 0.0), 1e-18))
            and (
                not fired
                or self._shadow_float_close(cross_times[0], detector.t_cross, max(float(time_tol or 0.0), 1e-18))
            )
        )
        self._rust_event_due_shadow_record("cross", matched, time_diff)

    def _rust_shadow_check_above(
        self,
        before: Tuple[float, float, float, float, int, int, float],
        detector: AboveDetector,
        fired: bool,
        time: float,
        val: float,
    ) -> None:
        backend = self._rust_event_due_shadow_backend
        if backend is None:
            return
        self._perf_stats["rust_event_due_shadow_above_checks"] += 1
        try:
            prev_values = array("d", [before[0]])
            prev_times = array("d", [before[1]])
            pprev_values = array("d", [before[2]])
            pprev_times = array("d", [before[3]])
            initialized_flags = array("B", [before[4]])
            directions = array("i", [before[5]])
            current_values = array("d", [float(val)])
            triggered_flags = array("B", [0])
            cross_times = array("d", [before[6]])
            backend.above_detector_step(
                prev_values,
                prev_times,
                pprev_values,
                pprev_times,
                initialized_flags,
                directions,
                current_values,
                triggered_flags,
                cross_times,
                time,
            )
        except Exception:
            self._perf_stats["rust_event_due_shadow_errors"] += 1
            return

        time_diff = max(
            abs(float(prev_times[0]) - float(detector.prev_time)),
            abs(float(pprev_times[0]) - float(detector.pprev_time)),
        )
        if fired:
            time_diff = max(time_diff, abs(float(cross_times[0]) - float(detector.t_cross)))
        matched = (
            bool(triggered_flags[0]) == bool(fired)
            and int(initialized_flags[0]) == (1 if detector.initialized else 0)
            and self._shadow_float_close(prev_values[0], detector.prev_val, 1e-12)
            and self._shadow_float_close(pprev_values[0], detector.pprev_val, 1e-12)
            and self._shadow_float_close(prev_times[0], detector.prev_time, 1e-18)
            and self._shadow_float_close(pprev_times[0], detector.pprev_time, 1e-18)
            and (
                not fired
                or self._shadow_float_close(cross_times[0], detector.t_cross, 1e-18)
            )
        )
        self._rust_event_due_shadow_record("above", matched, time_diff)

    def _rust_shadow_check_timer_periodic(
        self,
        key: str,
        before_has_state: bool,
        before_next_fire: float,
        due: bool,
        time: float,
        period: float,
        start: Optional[float],
    ) -> None:
        backend = self._rust_event_due_shadow_backend
        if backend is None:
            return
        self._perf_stats["rust_event_due_shadow_timer_periodic_checks"] += 1
        try:
            next_fire_times = array("d", [float(before_next_fire)])
            has_state_flags = array("B", [1 if before_has_state else 0])
            periods = array("d", [float(period)])
            starts = array("d", [float(start) if start is not None else 0.0])
            has_start_flags = array("B", [1 if start is not None else 0])
            due_flags = array("B", [0])
            skipped_flags = array("B", [0])
            backend.timer_periodic_step(
                next_fire_times,
                has_state_flags,
                periods,
                starts,
                has_start_flags,
                due_flags,
                skipped_flags,
                time,
                reschedule_on_due=False,
                eps=1e-18,
            )
        except Exception:
            self._perf_stats["rust_event_due_shadow_errors"] += 1
            return

        after_has_state = key in self.timer_states
        after_next_fire = float(self.timer_states.get(key, 0.0))
        time_diff = abs(float(next_fire_times[0]) - after_next_fire) if after_has_state else 0.0
        matched = (
            bool(due_flags[0]) == bool(due)
            and bool(has_state_flags[0]) == bool(after_has_state)
            and (
                not after_has_state
                or self._shadow_float_close(next_fire_times[0], after_next_fire, 1e-18)
            )
        )
        self._rust_event_due_shadow_record("timer_periodic", matched, time_diff)

    def _rust_shadow_check_timer_absolute(
        self,
        key: str,
        before_has_state: bool,
        before_next_fire: float,
        before_has_last_fired: bool,
        before_last_fired: float,
        due: bool,
        time: float,
        target: float,
    ) -> None:
        backend = self._rust_event_due_shadow_backend
        if backend is None:
            return
        self._perf_stats["rust_event_due_shadow_timer_absolute_checks"] += 1
        try:
            next_fire_times = array("d", [float(before_next_fire)])
            has_state_flags = array("B", [1 if before_has_state else 0])
            last_fired_times = array("d", [float(before_last_fired)])
            has_last_fired_flags = array("B", [1 if before_has_last_fired else 0])
            targets = array("d", [float(target)])
            due_flags = array("B", [0])
            expired_flags = array("B", [0])
            backend.timer_absolute_step(
                next_fire_times,
                has_state_flags,
                last_fired_times,
                has_last_fired_flags,
                targets,
                due_flags,
                expired_flags,
                time,
                eps=1e-18,
            )
        except Exception:
            self._perf_stats["rust_event_due_shadow_errors"] += 1
            return

        after_has_state = key in self.timer_states
        after_next_fire = float(self.timer_states.get(key, 0.0))
        after_has_last = key in self.timer_last_fired
        after_last = float(self.timer_last_fired.get(key, 0.0))
        time_diff = 0.0
        if after_has_state:
            time_diff = max(time_diff, abs(float(next_fire_times[0]) - after_next_fire))
        if after_has_last:
            time_diff = max(time_diff, abs(float(last_fired_times[0]) - after_last))
        matched = (
            bool(due_flags[0]) == bool(due)
            and bool(has_state_flags[0]) == bool(after_has_state)
            and bool(has_last_fired_flags[0]) == bool(after_has_last)
            and (
                not after_has_state
                or self._shadow_float_close(next_fire_times[0], after_next_fire, 1e-18)
            )
            and (
                not after_has_last
                or self._shadow_float_close(last_fired_times[0], after_last, 1e-18)
            )
        )
        self._rust_event_due_shadow_record("timer_absolute", matched, time_diff)

    def _rust_timer_periodic_production(
        self,
        key: str,
        time: float,
        period: float,
        start: Optional[float] = None,
        *,
        reschedule_on_due: bool = False,
    ) -> Optional[bool]:
        if not self._rust_timer_event_production_enabled:
            return None
        backend = self._rust_timer_event_backend
        if backend is None:
            return None
        if len(self.timer_states) < self._rust_timer_event_min_timers:
            return None
        self.timer_kinds[key] = "periodic"
        before_has_state = key in self.timer_states
        before_next_fire = float(self.timer_states.get(key, 0.0))
        try:
            next_fire_times = array("d", [before_next_fire])
            has_state_flags = array("B", [1 if before_has_state else 0])
            periods = array("d", [float(period)])
            starts = array("d", [float(start) if start is not None else 0.0])
            has_start_flags = array("B", [1 if start is not None else 0])
            due_flags = array("B", [0])
            skipped_flags = array("B", [0])
            backend.timer_periodic_step(
                next_fire_times,
                has_state_flags,
                periods,
                starts,
                has_start_flags,
                due_flags,
                skipped_flags,
                time,
                reschedule_on_due=reschedule_on_due,
                eps=1e-18,
            )
        except Exception:
            self._perf_stats["rust_timer_event_production_fallbacks"] += 1
            return None

        self._perf_stats["rust_timer_event_production_periodic_calls"] += 1
        has_state = bool(has_state_flags[0])
        due = bool(due_flags[0])
        skipped = bool(skipped_flags[0])
        if skipped:
            self._perf_stats["timer_periodic_skips"] += 1
            self._perf_stats["rust_timer_event_production_skips"] += 1
        if has_state:
            next_fire = float(next_fire_times[0])
            if (
                not before_has_state
                or abs(next_fire - before_next_fire) > 1e-18
            ):
                self._set_timer_state(key, next_fire)
        if due:
            self._event_time = time
            self._event_interpolated_nodes = set()
            self._event_interpolated_node_values = {}
            self._event_trace_audit_record_event(
                "timer_periodic",
                key,
                time,
                time,
            )
            if reschedule_on_due:
                self._perf_stats["timer_reschedules"] += 1
            self._perf_stats["rust_timer_event_production_fires"] += 1
        return due

    def _rust_timer_absolute_production(
        self,
        key: str,
        time: float,
        target: float,
    ) -> Optional[bool]:
        if not self._rust_timer_event_production_enabled:
            return None
        backend = self._rust_timer_event_backend
        if backend is None:
            return None
        if len(self.timer_states) < self._rust_timer_event_min_timers:
            return None
        self.timer_kinds[key] = "absolute"
        before_has_state = key in self.timer_states
        before_next_fire = float(self.timer_states.get(key, 0.0))
        before_has_last = key in self.timer_last_fired
        before_last = float(self.timer_last_fired.get(key, 0.0))
        try:
            next_fire_times = array("d", [before_next_fire])
            has_state_flags = array("B", [1 if before_has_state else 0])
            last_fired_times = array("d", [before_last])
            has_last_fired_flags = array("B", [1 if before_has_last else 0])
            targets = array("d", [float(target)])
            due_flags = array("B", [0])
            expired_flags = array("B", [0])
            backend.timer_absolute_step(
                next_fire_times,
                has_state_flags,
                last_fired_times,
                has_last_fired_flags,
                targets,
                due_flags,
                expired_flags,
                time,
                eps=1e-18,
            )
        except Exception:
            self._perf_stats["rust_timer_event_production_fallbacks"] += 1
            return None

        self._perf_stats["rust_timer_event_production_absolute_calls"] += 1
        has_state = bool(has_state_flags[0])
        has_last = bool(has_last_fired_flags[0])
        due = bool(due_flags[0])
        expired = bool(expired_flags[0])
        if has_state:
            next_fire = float(next_fire_times[0])
            if (
                not before_has_state
                or abs(next_fire - before_next_fire) > 1e-18
            ):
                self._set_timer_state(key, next_fire)
        if has_last:
            last_fired = float(last_fired_times[0])
            if (
                not before_has_last
                or abs(last_fired - before_last) > 1e-18
            ):
                self._set_timer_last_fired(key, last_fired)
        if expired:
            self._perf_stats["timer_absolute_expirations"] += 1
            self._perf_stats["rust_timer_event_production_expirations"] += 1
        if due:
            self._perf_stats["timer_absolute_fires"] += 1
            self._event_time = time
            self._event_interpolated_nodes = set()
            self._event_interpolated_node_values = {}
            self._event_trace_audit_record_event(
                "timer_absolute",
                key,
                time,
                float(next_fire_times[0]),
            )
            self._perf_stats["rust_timer_event_production_fires"] += 1
        return due

    def _next_timer_breakpoint(self, time: float) -> Optional[float]:
        cached_version = self._timer_breakpoint_cache_version
        cached = self._timer_breakpoint_cache
        if cached_version == self._timer_state_version:
            if cached is None:
                return None
            if cached > time:
                self._perf_stats["timer_breakpoint_hits"] += 1
                self._perf_stats["timer_breakpoint_cache_hits"] += 1
                return cached

        self._perf_stats["timer_breakpoint_scans"] += 1
        scanner = self._rust_timer_breakpoint_scanner
        best: Optional[float] = None
        if (
            scanner is not None
            and len(self.timer_states) >= self._rust_timer_event_min_timers
        ):
            self._sync_timer_array_sidecar_from_dict()
            next_fire_times = self._timer_next_fire_values
            last_fired_times = self._timer_last_fired_values
            has_last_fired_flags = self._timer_has_last_fired_flags
            self._perf_stats["timer_array_sidecar_scans"] += 1
            self._perf_stats["rust_timer_breakpoint_scans"] += 1
            self._perf_stats["rust_timer_breakpoint_state_scans"] += len(
                next_fire_times
            )
            try:
                best = scanner(
                    next_fire_times,
                    last_fired_times,
                    has_last_fired_flags,
                    time,
                )
            except Exception:
                self._perf_stats["rust_timer_breakpoint_fallbacks"] += 1
                best = None
                for key, nf in self.timer_states.items():
                    last_fired = self.timer_last_fired.get(key)
                    if last_fired is not None and abs(last_fired - nf) <= 1e-18:
                        continue
                    if nf > time and (best is None or nf < best):
                        best = nf
        else:
            for key, nf in self.timer_states.items():
                last_fired = self.timer_last_fired.get(key)
                if last_fired is not None and abs(last_fired - nf) <= 1e-18:
                    continue
                if nf > time and (best is None or nf < best):
                    best = nf
        self._timer_breakpoint_cache = best
        self._timer_breakpoint_cache_version = self._timer_state_version
        if best is not None:
            self._perf_stats["timer_breakpoint_hits"] += 1
        return best

    def _check_timer_due(self, key: str, time: float, period: float, start: Optional[float] = None) -> bool:
        self._perf_stats["timer_periodic_checks"] += 1
        if (
            self._rust_timer_event_production_enabled
            and len(self.timer_states) >= self._rust_timer_event_min_timers
        ):
            rust_due = self._rust_timer_periodic_production(
                key,
                time,
                period,
                start,
                reschedule_on_due=False,
            )
            if rust_due is not None:
                return rust_due
        if self.timer_kinds.get(key) != "periodic":
            self.timer_kinds[key] = "periodic"
        shadow_timer = self._rust_event_due_shadow_backend is not None
        states = self.timer_states
        next_fire = states.get(key)
        if shadow_timer:
            before_has_state = next_fire is not None
            before_next_fire = float(next_fire or 0.0)
        else:
            before_has_state = False
            before_next_fire = 0.0
        p = float(period)
        if p <= 0.0 or not math.isfinite(p):
            if shadow_timer:
                self._rust_shadow_check_timer_periodic(
                    key,
                    before_has_state,
                    before_next_fire,
                    False,
                    time,
                    period,
                    start,
                )
            return False
        if next_fire is None:
            next_fire = float(start) if start is not None else p
            if not math.isfinite(next_fire):
                next_fire = p
            if time > next_fire + 1e-18:
                missed = math.floor((time - next_fire) / p) + 1
                self._set_timer_state(key, next_fire + missed * p)
                self._perf_stats["timer_periodic_skips"] += 1
                if shadow_timer:
                    self._rust_shadow_check_timer_periodic(
                        key,
                        before_has_state,
                        before_next_fire,
                        False,
                        time,
                        period,
                        start,
                    )
                return False
            self._set_timer_state(key, next_fire)
        if time > next_fire + 1e-18:
            missed = math.floor((time - next_fire) / p) + 1
            self._set_timer_state(key, next_fire + missed * p)
            self._perf_stats["timer_periodic_skips"] += 1
            if shadow_timer:
                self._rust_shadow_check_timer_periodic(
                    key,
                    before_has_state,
                    before_next_fire,
                    False,
                    time,
                    period,
                    start,
                )
            return False
        due = time >= next_fire - 1e-18
        if due:
            self._event_time = time
            self._event_interpolated_nodes = set()
            self._event_interpolated_node_values = {}
            self._event_trace_audit_record_event(
                "timer_periodic",
                key,
                time,
                time,
            )
        if shadow_timer:
            self._rust_shadow_check_timer_periodic(
                key,
                before_has_state,
                before_next_fire,
                due,
                time,
                period,
                start,
            )
        return due

    def _reschedule_timer(self, key: str, time: float, period: float):
        p = float(period)
        if p <= 0.0 or not math.isfinite(p) or key not in self.timer_states:
            return
        self._set_timer_state(key, self.timer_states[key] + p)
        self._perf_stats["timer_reschedules"] += 1

    def _check_timer_event_batch(self, specs: Tuple[Tuple[Any, ...], ...], time: float) -> Tuple[bool, ...]:
        """Check a compile-time grouped timer segment without per-event calls.

        The compiler only uses this for consecutive timer statements whose
        timer expressions are constants/parameters. Event bodies still execute
        in generated Python source order; this helper only computes the due
        mask and updates timer bookkeeping.
        """
        if not specs:
            return ()
        if self._rust_event_due_shadow_backend is not None:
            self._perf_stats["timer_batch_due_fallbacks"] += len(specs)
            hits = []
            for spec in specs:
                if spec[0] == "periodic":
                    _, key, period, start = spec
                    hits.append(self._check_timer_due(key, time, period, start))
                else:
                    _, key, target = spec
                    hits.append(self._check_timer_at(key, time, target))
            return tuple(hits)

        rust_hits = self._rust_timer_event_batch_production(specs, time)
        if rust_hits is not None:
            return rust_hits

        stats = self._perf_stats
        stats["timer_batch_due_calls"] += 1
        stats["timer_batch_due_events"] += len(specs)
        states = self.timer_states
        last_fired_map = self.timer_last_fired
        kinds = self.timer_kinds
        eps = 1e-18
        due_flags: List[bool] = []

        for spec in specs:
            kind = spec[0]
            if kind == "periodic":
                _, key, period, start = spec
                stats["timer_periodic_checks"] += 1
                if kinds.get(key) != "periodic":
                    kinds[key] = "periodic"
                p = float(period)
                due = False
                if p > 0.0 and math.isfinite(p):
                    next_fire = states.get(key)
                    if next_fire is None:
                        next_fire = float(start) if start is not None else p
                        if not math.isfinite(next_fire):
                            next_fire = p
                        if time > next_fire + eps:
                            missed = math.floor((time - next_fire) / p) + 1
                            self._set_timer_state(key, next_fire + missed * p)
                            stats["timer_periodic_skips"] += 1
                            due_flags.append(False)
                            continue
                        self._set_timer_state(key, next_fire)
                    if time > next_fire + eps:
                        missed = math.floor((time - next_fire) / p) + 1
                        self._set_timer_state(key, next_fire + missed * p)
                        stats["timer_periodic_skips"] += 1
                        due_flags.append(False)
                        continue
                    due = time >= next_fire - eps
                    if due:
                        self._event_time = time
                        self._event_interpolated_nodes = set()
                        self._event_interpolated_node_values = {}
                        self._event_trace_audit_record_event(
                            "timer_periodic",
                            key,
                            time,
                            time,
                        )
                if due:
                    stats["timer_batch_due_fires"] += 1
                due_flags.append(due)
                continue

            _, key, target = spec
            stats["timer_absolute_checks"] += 1
            if kinds.get(key) != "absolute":
                kinds[key] = "absolute"
            due = False
            tgt = float(target)
            if math.isfinite(tgt):
                armed_target = states.get(key)
                last_fired = last_fired_map.get(key)
                first_seen = armed_target is None
                if first_seen or abs(armed_target - tgt) > eps:
                    self._set_timer_state(key, tgt)
                    armed_target = tgt
                if first_seen and time > tgt + eps:
                    self._set_timer_last_fired(key, tgt)
                    stats["timer_absolute_expirations"] += 1
                elif last_fired is not None and abs(last_fired - armed_target) <= eps:
                    due = False
                elif time >= armed_target - eps:
                    self._set_timer_last_fired(key, armed_target)
                    stats["timer_absolute_fires"] += 1
                    self._event_time = time
                    self._event_interpolated_nodes = set()
                    self._event_interpolated_node_values = {}
                    self._event_trace_audit_record_event(
                        "timer_absolute",
                        key,
                        time,
                        armed_target,
                    )
                    due = True
            if due:
                stats["timer_batch_due_fires"] += 1
            due_flags.append(due)

        return tuple(due_flags)

    def _rust_timer_event_batch_production(
        self,
        specs: Tuple[Tuple[Any, ...], ...],
        time: float,
    ) -> Optional[Tuple[bool, ...]]:
        """Compute grouped timer due masks with the Rust array primitive.

        This keeps event body execution in generated Python order.  Only the
        timer state-machine part is moved to Rust, so the semantics match the
        existing grouped Python helper while removing per-event due checks from
        the hot path.
        """
        if not self._rust_timer_event_production_enabled:
            return None
        backend = self._rust_timer_event_backend
        if backend is None or len(specs) < self._rust_timer_event_min_timers:
            return None

        stats = self._perf_stats
        hits = [False] * len(specs)
        periodic_entries: List[Tuple[int, str, float, Optional[float]]] = []
        absolute_entries: List[Tuple[int, str, float]] = []
        for idx, spec in enumerate(specs):
            kind = spec[0]
            if kind == "periodic":
                _, key, period, start = spec
                periodic_entries.append((idx, key, float(period), None if start is None else float(start)))
            else:
                _, key, target = spec
                absolute_entries.append((idx, key, float(target)))

        stats["timer_batch_due_calls"] += 1
        stats["timer_batch_due_events"] += len(specs)

        try:
            if periodic_entries:
                next_fire_times = array(
                    "d",
                    [float(self.timer_states.get(key, 0.0)) for _, key, _, _ in periodic_entries],
                )
                has_state_flags = array(
                    "B",
                    [1 if key in self.timer_states else 0 for _, key, _, _ in periodic_entries],
                )
                periods = array("d", [period for _, _, period, _ in periodic_entries])
                starts = array(
                    "d",
                    [0.0 if start is None else float(start) for _, _, _, start in periodic_entries],
                )
                has_start_flags = array(
                    "B",
                    [1 if start is not None else 0 for _, _, _, start in periodic_entries],
                )
                due_flags = array("B", [0] * len(periodic_entries))
                skipped_flags = array("B", [0] * len(periodic_entries))
                backend.timer_periodic_step(
                    next_fire_times,
                    has_state_flags,
                    periods,
                    starts,
                    has_start_flags,
                    due_flags,
                    skipped_flags,
                    time,
                    reschedule_on_due=False,
                    eps=1e-18,
                )
                stats["rust_timer_event_production_periodic_calls"] += 1
                for slot, (spec_idx, key, _period, _start) in enumerate(periodic_entries):
                    stats["timer_periodic_checks"] += 1
                    if self.timer_kinds.get(key) != "periodic":
                        self.timer_kinds[key] = "periodic"
                    if skipped_flags[slot]:
                        stats["timer_periodic_skips"] += 1
                        stats["rust_timer_event_production_skips"] += 1
                    if has_state_flags[slot]:
                        next_fire = float(next_fire_times[slot])
                        previous = self.timer_states.get(key)
                        if previous is None or abs(float(previous) - next_fire) > 1e-18:
                            self._set_timer_state(key, next_fire)
                    if due_flags[slot]:
                        hits[spec_idx] = True
                        stats["timer_batch_due_fires"] += 1
                        stats["rust_timer_event_production_fires"] += 1
                        self._event_time = time
                        self._event_interpolated_nodes = set()
                        self._event_interpolated_node_values = {}
                        self._event_trace_audit_record_event(
                            "timer_periodic",
                            key,
                            time,
                            time,
                        )

            if absolute_entries:
                next_fire_times = array(
                    "d",
                    [float(self.timer_states.get(key, 0.0)) for _, key, _ in absolute_entries],
                )
                has_state_flags = array(
                    "B",
                    [1 if key in self.timer_states else 0 for _, key, _ in absolute_entries],
                )
                last_fired_times = array(
                    "d",
                    [float(self.timer_last_fired.get(key, 0.0)) for _, key, _ in absolute_entries],
                )
                has_last_fired_flags = array(
                    "B",
                    [1 if key in self.timer_last_fired else 0 for _, key, _ in absolute_entries],
                )
                targets = array("d", [target for _, _, target in absolute_entries])
                due_flags = array("B", [0] * len(absolute_entries))
                expired_flags = array("B", [0] * len(absolute_entries))
                backend.timer_absolute_step(
                    next_fire_times,
                    has_state_flags,
                    last_fired_times,
                    has_last_fired_flags,
                    targets,
                    due_flags,
                    expired_flags,
                    time,
                    eps=1e-18,
                )
                stats["rust_timer_event_production_absolute_calls"] += 1
                for slot, (spec_idx, key, _target) in enumerate(absolute_entries):
                    stats["timer_absolute_checks"] += 1
                    if self.timer_kinds.get(key) != "absolute":
                        self.timer_kinds[key] = "absolute"
                    if has_state_flags[slot]:
                        next_fire = float(next_fire_times[slot])
                        previous = self.timer_states.get(key)
                        if previous is None or abs(float(previous) - next_fire) > 1e-18:
                            self._set_timer_state(key, next_fire)
                    if has_last_fired_flags[slot]:
                        last_fired = float(last_fired_times[slot])
                        previous_last = self.timer_last_fired.get(key)
                        if previous_last is None or abs(float(previous_last) - last_fired) > 1e-18:
                            self._set_timer_last_fired(key, last_fired)
                    if expired_flags[slot]:
                        stats["timer_absolute_expirations"] += 1
                        stats["rust_timer_event_production_expirations"] += 1
                    if due_flags[slot]:
                        hits[spec_idx] = True
                        stats["timer_absolute_fires"] += 1
                        stats["timer_batch_due_fires"] += 1
                        stats["rust_timer_event_production_fires"] += 1
                        self._event_time = time
                        self._event_interpolated_nodes = set()
                        self._event_interpolated_node_values = {}
                        self._event_trace_audit_record_event(
                            "timer_absolute",
                            key,
                            time,
                            float(next_fire_times[slot]),
                        )
        except Exception:
            stats["rust_timer_event_production_fallbacks"] += len(specs)
            return None

        return tuple(hits)

    def _check_timer_at(self, key: str, time: float, target: float) -> bool:
        self._perf_stats["timer_absolute_checks"] += 1
        if (
            self._rust_timer_event_production_enabled
            and len(self.timer_states) >= self._rust_timer_event_min_timers
        ):
            rust_due = self._rust_timer_absolute_production(key, time, target)
            if rust_due is not None:
                return rust_due
        if self.timer_kinds.get(key) != "absolute":
            self.timer_kinds[key] = "absolute"
        shadow_timer = self._rust_event_due_shadow_backend is not None
        states = self.timer_states
        last_fired_map = self.timer_last_fired
        armed_target = states.get(key)
        last_fired = last_fired_map.get(key)
        if shadow_timer:
            before_has_state = armed_target is not None
            before_next_fire = float(armed_target or 0.0)
            before_has_last = last_fired is not None
            before_last = float(last_fired or 0.0)
        else:
            before_has_state = False
            before_next_fire = 0.0
            before_has_last = False
            before_last = 0.0
        tgt = float(target)
        if not math.isfinite(tgt):
            if shadow_timer:
                self._rust_shadow_check_timer_absolute(
                    key,
                    before_has_state,
                    before_next_fire,
                    before_has_last,
                    before_last,
                    False,
                    time,
                    target,
                )
            return False
        first_seen = armed_target is None
        if first_seen or abs(armed_target - tgt) > 1e-18:
            self._set_timer_state(key, tgt)
            armed_target = tgt
        if first_seen and time > tgt + 1e-18:
            self._set_timer_last_fired(key, tgt)
            self._perf_stats["timer_absolute_expirations"] += 1
            if shadow_timer:
                self._rust_shadow_check_timer_absolute(
                    key,
                    before_has_state,
                    before_next_fire,
                    before_has_last,
                    before_last,
                    False,
                    time,
                    target,
                )
            return False
        if last_fired is not None and abs(last_fired - armed_target) <= 1e-18:
            if shadow_timer:
                self._rust_shadow_check_timer_absolute(
                    key,
                    before_has_state,
                    before_next_fire,
                    before_has_last,
                    before_last,
                    False,
                    time,
                    target,
                )
            return False
        if time >= armed_target - 1e-18:
            self._set_timer_last_fired(key, armed_target)
            self._perf_stats["timer_absolute_fires"] += 1
            self._event_time = time
            self._event_interpolated_nodes = set()
            self._event_interpolated_node_values = {}
            self._event_trace_audit_record_event(
                "timer_absolute",
                key,
                time,
                armed_target,
            )
            if shadow_timer:
                self._rust_shadow_check_timer_absolute(
                    key,
                    before_has_state,
                    before_next_fire,
                    before_has_last,
                    before_last,
                    True,
                    time,
                    target,
                )
            return True
        if shadow_timer:
            self._rust_shadow_check_timer_absolute(
                key,
                before_has_state,
                before_next_fire,
                before_has_last,
                before_last,
                False,
                time,
                target,
            )
        return False

    def _check_state_owned_timer_at(
        self,
        key: str,
        time: float,
        state_name: str,
        state_slot: int = -1,
    ) -> bool:
        """Fast absolute timer check for ``@(timer(state_time))`` owner chains.

        The compiler only emits this helper when the target state is assigned
        during initial_step and the timer's own event body. While the armed
        target is still in the future, reading the target state again cannot
        change the schedule, so this path skips the generic state lookup.
        """
        self._perf_stats["timer_state_owned_checks"] += 1
        if (
            self._rust_timer_event_production_enabled
            or self._rust_event_due_shadow_backend is not None
        ):
            self._perf_stats["timer_state_owned_fallbacks"] += 1
            self._perf_stats["timer_state_owned_target_reads"] += 1
            if state_slot >= 0:
                target = self._state_get_by_slot(state_slot, state_name)
            else:
                target = self.state[state_name]
            due = self._check_timer_at(key, time, target)
            if due:
                self._perf_stats["timer_state_owned_fires"] += 1
            return due

        if self.timer_kinds.get(key) != "absolute":
            self.timer_kinds[key] = "absolute"
        armed_target = self.timer_states.get(key)
        last_fired = self.timer_last_fired.get(key)
        if (
            armed_target is not None
            and (last_fired is None or abs(last_fired - armed_target) > 1e-18)
            and time < armed_target - 1e-18
        ):
            self._perf_stats["timer_absolute_checks"] += 1
            self._perf_stats["timer_state_owned_fast_skips"] += 1
            return False

        self._perf_stats["timer_state_owned_target_reads"] += 1
        if state_slot >= 0:
            target = self._state_get_by_slot(state_slot, state_name)
        else:
            target = self.state[state_name]
        due = self._check_timer_at(key, time, target)
        if due:
            self._perf_stats["timer_state_owned_fires"] += 1
        return due

    def _check_timer(self, key: str, time: float, period: float, start: Optional[float] = None) -> bool:
        if self._rust_timer_event_production_enabled:
            self._perf_stats["timer_periodic_checks"] += 1
            rust_due = self._rust_timer_periodic_production(
                key,
                time,
                period,
                start,
                reschedule_on_due=True,
            )
            if rust_due is not None:
                if rust_due:
                    self._perf_stats["timer_periodic_fires"] += 1
                return rust_due
        due = self._check_timer_due(key, time, period, start)
        if due:
            self._perf_stats["timer_periodic_fires"] += 1
            self._reschedule_timer(key, time, period)
        return due

    def _expire_absolute_timers(self, time: float):
        for key, armed_target in self.timer_states.items():
            if self.timer_kinds.get(key) != "absolute":
                continue
            last_fired = self.timer_last_fired.get(key)
            if last_fired is not None and abs(last_fired - armed_target) <= 1e-18:
                continue
            if time >= armed_target - 1e-18:
                self._set_timer_last_fired(key, armed_target)
                self._perf_stats["timer_absolute_expirations"] += 1

    @classmethod
    def _cmp_gt(cls, left: Any, right: Any) -> bool:
        return float(left) > float(right) + cls._cmp_eps

    @classmethod
    def _cmp_lt(cls, left: Any, right: Any) -> bool:
        return float(left) < float(right) - cls._cmp_eps

    @classmethod
    def _cmp_ge(cls, left: Any, right: Any) -> bool:
        return float(left) >= float(right) - cls._cmp_eps

    @classmethod
    def _cmp_le(cls, left: Any, right: Any) -> bool:
        return float(left) <= float(right) + cls._cmp_eps

    @staticmethod
    def _to_integer(value: Any) -> int:
        """Verilog-A real-to-integer assignment rounds to nearest."""
        v = float(value)
        if not math.isfinite(v):
            return 0
        if v >= 0.0:
            return math.floor(v + 0.5)
        return math.ceil(v - 0.5)

    @staticmethod
    def _int_div(left: Any, right: Any) -> int:
        """Verilog-style integer division, truncating toward zero."""
        r = int(right)
        if r == 0:
            return 0
        return int(int(left) / r)

    def _cache_event_interpolated_values(self, nodes: Any):
        """Precompute event-time voltage reads for cross/above event bodies."""
        self._event_interpolated_node_values = {}
        backend = self._rust_event_interpolation_backend
        if backend is None or not nodes:
            return
        t0 = float(self._step_prev_time)
        t1 = float(self._step_time)
        if t1 <= t0 + 1e-30:
            return

        ext_nodes = []
        seen = set()
        for node in nodes:
            if not isinstance(node, str):
                continue
            ext = self._resolve_external_node(node)
            if (
                isinstance(ext, str)
                and ext in self._step_prev_node_voltages
                and ext not in seen
            ):
                seen.add(ext)
                ext_nodes.append(ext)
        if not ext_nodes:
            return

        previous_values = array(
            "d",
            (float(self._step_prev_node_voltages[ext]) for ext in ext_nodes),
        )
        current_values = array(
            "d",
            (
                float(
                    self._step_curr_node_voltages.get(
                        ext,
                        self._step_prev_node_voltages[ext],
                    )
                )
                for ext in ext_nodes
            ),
        )
        try:
            out_values = backend.interpolate_event_values(
                previous_values,
                current_values,
                t0,
                t1,
                float(self._event_time),
            )
        except Exception:
            self._perf_stats["rust_event_interpolation_fallbacks"] += 1
            self._event_interpolated_node_values = {}
            return

        self._event_interpolated_node_values = {
            ext: float(value)
            for ext, value in zip(ext_nodes, out_values)
        }
        self._perf_stats["rust_event_interpolation_batches"] += 1
        self._perf_stats["rust_event_interpolation_nodes"] += len(ext_nodes)

    def _get_voltage(self, node: str, node_voltages: Dict[str, float]) -> float:
        """Get voltage of a node, resolving through node_map."""
        if not self.node_map and not self._event_context_active:
            reader = self._indexed_voltage_reader
            if reader is not None:
                indexed_value = reader(node, node)
                if indexed_value is not None:
                    return indexed_value
            if node in node_voltages:
                value = node_voltages[node]
                if self._indexed_voltage_probe is not None:
                    self._indexed_voltage_probe(node, node, value, False)
                return value
            if node in self.output_nodes:
                value = self.output_nodes[node]
                if self._indexed_voltage_probe is not None:
                    self._indexed_voltage_probe(node, node, value, False)
                return value
            if self._indexed_voltage_probe is not None:
                self._indexed_voltage_probe(node, node, 0.0, False)
            return 0.0

        # Check if it's a mapped external node
        ext = self._resolve_external_node(node)
        if self._event_context_active and isinstance(ext, str):
            if ext in self._step_prev_node_voltages:
                t0 = self._step_prev_time
                t1 = self._step_time
                if t1 > t0 + 1e-30:
                    v0 = self._step_prev_node_voltages[ext]
                    if ext in node_voltages:
                        v1 = node_voltages[ext]
                    elif ext in self._step_curr_node_voltages:
                        v1 = self._step_curr_node_voltages[ext]
                    else:
                        v1 = v0
                    if (
                        ext in self._event_interpolated_node_values
                        and (
                            node in self._event_interpolated_nodes
                            or ext in self._event_interpolated_nodes
                        )
                    ):
                        value = self._event_interpolated_node_values[ext]
                        self._perf_stats["rust_event_interpolation_cache_hits"] += 1
                    else:
                        frac = (float(self._event_time) - t0) / (t1 - t0)
                        frac = max(0.0, min(1.0, frac))
                        value = v0 + frac * (v1 - v0)
                    if ext in self._event_node_cross_directions:
                        cross_dir = self._event_node_cross_directions[ext]
                    elif node in self._event_node_cross_directions:
                        cross_dir = self._event_node_cross_directions[node]
                    else:
                        cross_dir = None
                    if cross_dir is not None:
                        if cross_dir:
                            value += math.copysign(max(1e-12, abs(value) * 1e-12), cross_dir)
                    elif abs(v1 - v0) > 1e-30:
                        value += math.copysign(max(1e-12, abs(value) * 1e-12), v1 - v0)
                    if self._indexed_voltage_probe is not None:
                        self._indexed_voltage_probe(node, ext, value, True)
                    return value
        if self._indexed_voltage_reader is not None:
            indexed_value = self._read_indexed_voltage(node, ext)
            if indexed_value is not None:
                return indexed_value
        if ext in node_voltages:
            value = node_voltages[ext]
            if self._indexed_voltage_probe is not None:
                self._indexed_voltage_probe(node, ext, value, False)
            return value
        # Check output nodes (self-driven)
        if node in self.output_nodes:
            value = self.output_nodes[node]
            if self._indexed_voltage_probe is not None:
                self._indexed_voltage_probe(node, ext, value, False)
            return value
        if self._indexed_voltage_probe is not None:
            self._indexed_voltage_probe(node, ext, 0.0, False)
        return 0.0

    def _get_static_branch_voltage_by_slot(self, slot: int, node_voltages: Dict[str, float]) -> float:
        """Read a static branch node by run-installed node id when available."""
        try:
            node = self.__class__._static_voltage_read_nodes[slot]
        except IndexError:
            return 0.0
        if self._event_context_active:
            self._perf_stats["static_branch_fastpath_fallbacks"] += 1
            return self._get_voltage(node, node_voltages)

        values = self._static_branch_indexed_values
        if values is not None and slot < len(self._static_branch_read_node_ids):
            node_id = self._static_branch_read_node_ids[slot]
            if 0 <= node_id < len(values):
                return values[node_id]
        return self._get_static_branch_voltage(node, node_voltages)

    def _get_static_branch_voltage(self, node: str, node_voltages: Dict[str, float]) -> float:
        """Fast helper for compile-time-static branch voltage reads.

        Event contexts intentionally fall back to ``_get_voltage`` because
        crossing-time reads may need interpolation between previous/current
        step values rather than a current-step dict/array lookup.
        """
        if self._event_context_active:
            self._perf_stats["static_branch_fastpath_fallbacks"] += 1
            return self._get_voltage(node, node_voltages)

        ext = self._resolve_external_node(node)
        if self._indexed_voltage_reader is not None:
            indexed_value = self._read_indexed_voltage(node, ext)
            if indexed_value is not None:
                return indexed_value

        if ext in node_voltages:
            value = node_voltages[ext]
            if self._indexed_voltage_probe is not None:
                self._indexed_voltage_probe(node, ext, value, False)
            return value
        if node in self.output_nodes:
            value = self.output_nodes[node]
            if self._indexed_voltage_probe is not None:
                self._indexed_voltage_probe(node, ext, value, False)
            return value
        if self._indexed_voltage_probe is not None:
            self._indexed_voltage_probe(node, ext, 0.0, False)
        return 0.0

    def _set_output(self, node: str, value: float, node_voltages: Dict[str, float]):
        """Set an output node voltage."""
        if not self.node_map:
            if node not in self.output_nodes:
                self._output_nodes_version += 1
            self.output_nodes[node] = value
            node_voltages[node] = value
            self._event_trace_audit_note_write("output", node)
            if self._indexed_output_writer is not None:
                self._indexed_output_writer(node, value)
            return

        if node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[node] = value
        ext = self._resolve_external_node(node)
        node_voltages[ext] = value
        self._event_trace_audit_note_write("output", ext)
        if self._indexed_output_writer is not None:
            self._indexed_output_writer(ext, value)

    def _set_static_branch_output_by_slot(
        self,
        slot: int,
        value: float,
        node_voltages: Dict[str, float],
    ):
        """Write a static branch output by run-installed node id when available."""
        try:
            node = self.__class__._static_output_write_nodes[slot]
        except IndexError:
            return
        if self._event_context_active:
            self._perf_stats["static_branch_fastpath_fallbacks"] += 1
            self._set_output(node, value, node_voltages)
            return

        if node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[node] = value

        values = self._static_branch_indexed_values
        if (
            values is not None
            and slot < len(self._static_branch_write_node_ids)
            and slot < len(self._static_branch_write_external_nodes)
        ):
            node_id = self._static_branch_write_node_ids[slot]
            if 0 <= node_id < len(values):
                values[node_id] = float(value)
                ext = self._static_branch_write_external_nodes[slot]
                node_voltages[ext] = value
                self._event_trace_audit_note_write("output", ext)
                return

        self._set_static_branch_output(node, value, node_voltages)

    def _set_static_branch_output(self, node: str, value: float, node_voltages: Dict[str, float]):
        """Fast helper for compile-time-static branch voltage contributions."""
        if self._event_context_active:
            self._perf_stats["static_branch_fastpath_fallbacks"] += 1
            self._set_output(node, value, node_voltages)
            return

        if node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[node] = value
        ext = self._resolve_external_node(node)
        node_voltages[ext] = value
        self._event_trace_audit_note_write("output", ext)
        if self._indexed_output_writer is not None:
            self._indexed_output_writer(ext, value)

    def _transition(self, key: str, time: float, target: float,
                    delay: float = 0.0, rise: float = 0.0, fall: float = 0.0) -> float:
        """Evaluate a transition operator."""
        stats = self._perf_stats
        stats["transition_calls"] += 1
        rust_value = self._transition_rust_production(
            key,
            time,
            target,
            delay,
            rise,
            fall,
        )
        if rust_value is not None:
            return rust_value
        if self._initial_condition_mode:
            effective_rise = rise if rise > 0 else self.default_transition
            effective_fall = fall if fall > 0 else self.default_transition
            ts = self.transitions.get(key)
            if ts is None:
                ts = TransitionState(
                    current_val=target,
                    target_val=target,
                    start_val=target,
                    start_time=time,
                    delay=delay,
                    rise_time=effective_rise,
                    fall_time=effective_fall,
                    active=False,
                )
                self.transitions[key] = ts
                self._event_trace_audit_note_write("transition", key)
                self._mark_transition_known_inactive(key)
                return target
            was_active = ts.active
            ts.current_val = target
            ts.target_val = target
            ts.start_val = target
            ts.start_time = time
            ts.delay = delay
            ts.rise_time = effective_rise
            ts.fall_time = effective_fall
            ts.active = False
            self._event_trace_audit_note_write("transition", key)
            self._mark_transition_known_inactive(key, was_active)
            return target
        if key not in self.transitions:
            effective_rise = rise if rise > 0 else self.default_transition
            effective_fall = fall if fall > 0 else self.default_transition
            ts = TransitionState(
                current_val=target,
                target_val=target,
                start_val=target,
                start_time=time,
                delay=delay,
                rise_time=effective_rise,
                fall_time=effective_fall,
                active=False,
            )
            self.transitions[key] = ts
            self._event_trace_audit_note_write("transition", key)
            self._mark_transition_known_inactive(key)
            return target
        ts = self.transitions[key]
        # Advance current_val to the actual value at this time before updating target.
        # Without this, a new target set at t overwrites the in-progress transition
        # before evaluate() can commit its endpoint into current_val.
        stats["transition_evaluate_calls"] += 1
        was_active = ts.active
        current = ts.evaluate(time)
        self._track_transition_active_change(key, was_active, ts.active)
        if (
            self._transition_unchanged_fastpath_enabled
            and target == ts.target_val
        ):
            effective_rise = rise if rise > 0 else self.default_transition
            effective_fall = fall if fall > 0 else self.default_transition
            if (
                delay == ts.delay
                and effective_rise == ts.rise_time
                and effective_fall == ts.fall_time
                and (
                    ts.active or target == ts.current_val
                )
            ):
                stats["transition_unchanged_target_fastpath"] += 1
                return current
        stats["transition_set_target_calls"] += 1
        was_active = ts.active
        ts.set_target(time, target, delay, rise, fall, self.default_transition)
        self._event_trace_audit_note_write("transition", key)
        self._track_transition_active_change(key, was_active, ts.active)
        was_active = ts.active
        value = ts.evaluate(time)
        self._track_transition_active_change(key, was_active, ts.active)
        return value

    def _transition_rust_production(
        self,
        key: str,
        time: float,
        target: float,
        delay: float = 0.0,
        rise: float = 0.0,
        fall: float = 0.0,
    ) -> Optional[float]:
        backend = self._rust_transition_state_backend
        if backend is None or not self._rust_transition_state_production_enabled:
            return None
        stats = self._perf_stats
        ts = self.transitions.get(key)
        created = ts is None
        if ts is None:
            ts = TransitionState()
            self.transitions[key] = ts
        was_active = bool(ts.active)

        # Audit 086 (L2-a): persistent buffer reuse. Allocate once on first
        # call, reuse for every subsequent _transition_rust_production() call.
        # Eliminates 14x array("d", [...]) per call. Semantics unchanged.
        bufs = self._rust_transition_buffers
        if bufs is None:
            bufs = {
                "current": array("d", [0.0]),
                "target": array("d", [0.0]),
                "start_time": array("d", [0.0]),
                "start_val": array("d", [0.0]),
                "delay": array("d", [0.0]),
                "rise": array("d", [0.0]),
                "fall": array("d", [0.0]),
                "active": array("B", [0]),
                "init": array("B", [0]),
                "in_target": array("d", [0.0]),
                "in_delay": array("d", [0.0]),
                "in_rise": array("d", [0.0]),
                "in_fall": array("d", [0.0]),
                "output": array("d", [0.0]),
            }
            self._rust_transition_buffers = bufs
            stats["rust_transition_state_buffer_alloc_total"] += 14

        current_values = bufs["current"]
        target_values = bufs["target"]
        start_times = bufs["start_time"]
        start_values = bufs["start_val"]
        delays = bufs["delay"]
        rise_times = bufs["rise"]
        fall_times = bufs["fall"]
        active_flags = bufs["active"]
        initialized_flags = bufs["init"]
        input_targets = bufs["in_target"]
        input_delays = bufs["in_delay"]
        input_rises = bufs["in_rise"]
        input_falls = bufs["in_fall"]
        output_values = bufs["output"]

        current_values[0] = float(ts.current_val)
        target_values[0] = float(ts.target_val)
        start_times[0] = float(ts.start_time)
        start_values[0] = float(ts.start_val)
        delays[0] = float(ts.delay)
        rise_times[0] = float(ts.rise_time)
        fall_times[0] = float(ts.fall_time)
        active_flags[0] = 1 if ts.active else 0
        initialized_flags[0] = 0 if created else 1
        input_targets[0] = float(target)
        input_delays[0] = float(delay)
        input_rises[0] = float(rise)
        input_falls[0] = float(fall)
        output_values[0] = 0.0

        try:
            backend.transition_state_step(
                current_values,
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
                float(time),
                float(self.default_transition),
                bool(self._initial_condition_mode),
            )
        except Exception:
            stats["rust_transition_state_production_fallbacks"] += 1
            if created:
                self.transitions.pop(key, None)
            return None

        stats["rust_transition_state_production_calls"] += 1
        stats["rust_transition_state_production_outputs"] += 1
        stats["rust_transition_state_buffer_reuse_calls"] += 1
        if not created and not self._initial_condition_mode:
            stats["transition_evaluate_calls"] += 1
            stats["transition_set_target_calls"] += 1

        ts.current_val = float(current_values[0])
        ts.target_val = float(target_values[0])
        ts.start_time = float(start_times[0])
        ts.start_val = float(start_values[0])
        ts.delay = float(delays[0])
        ts.rise_time = float(rise_times[0])
        ts.fall_time = float(fall_times[0])
        ts.active = bool(active_flags[0])

        self._event_trace_audit_note_write("transition", key)
        if created and not ts.active:
            self._mark_transition_known_inactive(key)
        else:
            self._track_transition_active_change(key, was_active, ts.active)
        return float(output_values[0])

    def _transition_output(
        self,
        node: str,
        key: str,
        time: float,
        target: float,
        base: float,
        offset: float,
        scale: float,
        delay: float,
        rise: float,
        fall: float,
        node_voltages: Dict[str, float],
    ):
        """Fused transition contribution for common static-output models."""
        self._perf_stats["transition_output_fastpath_calls"] += 1
        value = float(base) + float(offset) + float(scale) * self._transition(
            key,
            time,
            target,
            delay,
            rise,
            fall,
        )
        if not self.node_map:
            if node not in self.output_nodes:
                self._output_nodes_version += 1
            self.output_nodes[node] = value
            node_voltages[node] = value
            self._event_trace_audit_note_write("transition_output", node)
            self._event_trace_audit_note_write("output", node)
            if self._indexed_output_writer is not None:
                self._indexed_output_writer(node, value)
            return

        if node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[node] = value
        ext = self._resolve_external_node(node)
        node_voltages[ext] = value
        self._event_trace_audit_note_write("transition_output", ext)
        self._event_trace_audit_note_write("output", ext)
        if self._indexed_output_writer is not None:
            self._indexed_output_writer(ext, value)

    # =========================================================================
    # Audit 088: per-step transition batch (lazy + flush).
    # =========================================================================
    #
    # _transition_output_lazy() is the deferred form of _transition_output().
    # The compiler emits it for V(out) <+ scale*transition()+offset+base
    # contributions whose output node is provably not read again in the same
    # analog block (see _collect_transition_defer_unsafe_nodes in
    # _ModuleCompiler). Each lazy call enqueues input parameters in
    # per-model parallel lists and does NOT touch nv[node]; the actual
    # Rust state-evolution call and the output write happen in a single
    # batched _flush_transitions() at end of evaluate().
    #
    # Compared to the immediate _transition_output()/_transition() path:
    # - Rust FFI count drops from N transitions/step → 1 flush/step
    # - All side effects (output_nodes, nv, event_trace_audit) move into flush
    # - On Rust error or empty queue, falls back to immediate per-call path.

    def _transition_output_lazy(
        self,
        node: str,
        key: str,
        time: float,
        target: float,
        base: float,
        offset: float,
        scale: float,
        delay: float,
        rise: float,
        fall: float,
        node_voltages: Dict[str, float],
    ):
        stats = self._perf_stats
        backend = self._rust_transition_state_backend
        if backend is None or not self._rust_transition_state_production_enabled:
            self._transition_output(
                node, key, time, target, base, offset, scale,
                delay, rise, fall, node_voltages,
            )
            return

        # Enqueue input parameters into per-step Python lists. Typed-array
        # buffers are constructed once at flush time at exact size n.
        pending = self._transition_pending_input
        if pending is None:
            pending = {
                "nodes": [], "keys": [],
                "base": [], "offset": [], "scale": [],
                "in_target": [], "in_delay": [], "in_rise": [], "in_fall": [],
            }
            self._transition_pending_input = pending
        pending["nodes"].append(node)
        pending["keys"].append(key)
        pending["base"].append(float(base))
        pending["offset"].append(float(offset))
        pending["scale"].append(float(scale))
        pending["in_target"].append(float(target))
        pending["in_delay"].append(float(delay))
        pending["in_rise"].append(float(rise))
        pending["in_fall"].append(float(fall))

        self._transition_pending_count += 1
        stats["transition_calls"] += 1
        stats["transition_output_fastpath_calls"] += 1
        stats["rust_transition_lazy_enqueues"] += 1
        # Note: nv[node] intentionally NOT written here. The static analyzer
        # in the compiler guarantees no later read of nv[node] in this evaluate.

    def _flush_transitions(self, node_voltages: Dict[str, float], time: float):
        n = self._transition_pending_count
        if n == 0:
            return
        stats = self._perf_stats
        backend = self._rust_transition_state_backend
        pending = self._transition_pending_input

        if backend is None or pending is None:
            self._flush_transitions_fallback(node_voltages, time)
            return

        # Build typed-array buffers at exact size n for this flush.
        # Pre-create any missing TransitionState rows so we can read prior state.
        created_flags = []
        current_list = []
        target_list = []
        start_time_list = []
        start_val_list = []
        delay_list = []
        rise_list = []
        fall_list = []
        active_list = []
        for key in pending["keys"]:
            ts = self.transitions.get(key)
            created = ts is None
            if created:
                ts = TransitionState()
                self.transitions[key] = ts
            created_flags.append(created)
            current_list.append(float(ts.current_val))
            target_list.append(float(ts.target_val))
            start_time_list.append(float(ts.start_time))
            start_val_list.append(float(ts.start_val))
            delay_list.append(float(ts.delay))
            rise_list.append(float(ts.rise_time))
            fall_list.append(float(ts.fall_time))
            active_list.append(1 if ts.active else 0)

        try:
            current_buf   = array("d", current_list)
            target_buf    = array("d", target_list)
            start_time_buf = array("d", start_time_list)
            start_val_buf  = array("d", start_val_list)
            delay_buf     = array("d", delay_list)
            rise_buf      = array("d", rise_list)
            fall_buf      = array("d", fall_list)
            active_buf    = array("B", active_list)
            init_buf      = array("B", [0 if c else 1 for c in created_flags])
            in_target_buf = array("d", pending["in_target"])
            in_delay_buf  = array("d", pending["in_delay"])
            in_rise_buf   = array("d", pending["in_rise"])
            in_fall_buf   = array("d", pending["in_fall"])
            output_buf    = array("d", [0.0] * n)
            stats["rust_transition_state_buffer_alloc_total"] += 14

            backend.transition_state_step(
                current_buf, target_buf, start_time_buf,
                start_val_buf, delay_buf, rise_buf, fall_buf,
                active_buf, init_buf,
                in_target_buf, in_delay_buf, in_rise_buf, in_fall_buf,
                output_buf,
                float(time), float(self.default_transition),
                bool(self._initial_condition_mode),
            )
        except Exception:
            stats["rust_transition_batch_fallbacks"] += 1
            self._flush_transitions_fallback(node_voltages, time)
            return

        nodes = pending["nodes"]
        keys = pending["keys"]
        bases = pending["base"]
        offsets = pending["offset"]
        scales = pending["scale"]
        for i in range(n):
            node = nodes[i]
            key = keys[i]
            ts = self.transitions[key]
            was_active = ts.active
            ts.current_val = float(current_buf[i])
            ts.target_val = float(target_buf[i])
            ts.start_time = float(start_time_buf[i])
            ts.start_val = float(start_val_buf[i])
            ts.delay = float(delay_buf[i])
            ts.rise_time = float(rise_buf[i])
            ts.fall_time = float(fall_buf[i])
            ts.active = bool(active_buf[i])

            value = bases[i] + offsets[i] + scales[i] * float(output_buf[i])
            if not self.node_map:
                if node not in self.output_nodes:
                    self._output_nodes_version += 1
                self.output_nodes[node] = value
                node_voltages[node] = value
                self._event_trace_audit_note_write("transition_output", node)
                self._event_trace_audit_note_write("output", node)
                if self._indexed_output_writer is not None:
                    self._indexed_output_writer(node, value)
            else:
                if node not in self.output_nodes:
                    self._output_nodes_version += 1
                self.output_nodes[node] = value
                ext = self._resolve_external_node(node)
                node_voltages[ext] = value
                self._event_trace_audit_note_write("transition_output", ext)
                self._event_trace_audit_note_write("output", ext)
                if self._indexed_output_writer is not None:
                    self._indexed_output_writer(ext, value)

            self._event_trace_audit_note_write("transition", key)
            if created_flags[i] and not ts.active:
                self._mark_transition_known_inactive(key)
            else:
                self._track_transition_active_change(key, was_active, ts.active)
            stats["rust_transition_state_production_calls"] += 1
            stats["rust_transition_state_production_outputs"] += 1
            stats["transition_evaluate_calls"] += 1
            stats["transition_set_target_calls"] += 1

        stats["rust_transition_batch_flushes"] += 1
        stats["rust_transition_batch_slot_total"] += n
        if n > stats["rust_transition_batch_max_slots"]:
            stats["rust_transition_batch_max_slots"] = n
        self._reset_transition_pending()

    def _flush_transitions_fallback(self, node_voltages: Dict[str, float], time: float):
        pending = self._transition_pending_input
        n = self._transition_pending_count
        if pending is None or n == 0:
            self._reset_transition_pending()
            return
        nodes = pending["nodes"]
        keys = pending["keys"]
        bases = pending["base"]
        offsets = pending["offset"]
        scales = pending["scale"]
        targets = pending["in_target"]
        delays = pending["in_delay"]
        rises = pending["in_rise"]
        falls = pending["in_fall"]
        for i in range(n):
            self._transition_output(
                nodes[i], keys[i], time, targets[i],
                bases[i], offsets[i], scales[i],
                delays[i], rises[i], falls[i], node_voltages,
            )
        self._reset_transition_pending()

    def _reset_transition_pending(self):
        self._transition_pending_count = 0
        pending = self._transition_pending_input
        if pending is not None:
            for v in pending.values():
                v.clear()

    def _slew(self, key: str, time: float, target: float,
              maxrise: float, maxfall: float) -> float:
        """Evaluate a slew-rate limiter.

        This is a transient behavioral approximation:
        - maxrise/maxfall are interpreted as V/s limits.
        - maxrise<=0 or maxfall<=0 means "no limit" in that direction.
        """
        t = float(time)
        tgt = float(target)
        mr = float(maxrise)
        mf = float(maxfall)

        if key not in self.slew_states:
            self.slew_states[key] = {"value": tgt, "last_t": t}
            return tgt

        st = self.slew_states[key]
        cur = float(st["value"])
        dt = t - float(st["last_t"])
        if dt <= 0.0:
            # Same-time re-evaluation: only allow immediate move for unlimited slope.
            if (tgt >= cur and mr <= 0.0) or (tgt < cur and mf <= 0.0):
                st["value"] = tgt
                return tgt
            return cur

        delta = tgt - cur
        if delta >= 0.0:
            if mr <= 0.0:
                nxt = tgt
            else:
                nxt = cur + min(delta, mr * dt)
        else:
            if mf <= 0.0:
                nxt = tgt
            else:
                nxt = cur - min(-delta, mf * dt)

        st["value"] = nxt
        st["last_t"] = t
        return nxt

    def _check_cross(self, key: str, time: float, val: float, direction: int = 0,
                     time_tol: float = 0.0, expr_tol: float = 1e-12,
                     interp_nodes: Optional[List[str]] = None,
                     nudge_nodes: Optional[Any] = None) -> bool:
        if key not in self.cross_detectors:
            self.cross_detectors[key] = CrossDetector(direction=direction)
        detector = self.cross_detectors[key]
        before = self._cross_detector_shadow_state(detector)
        # Audit 089: production gate — let Rust own the detector state
        # evolution math. If unavailable or fails, fall back to Python.
        fired = self._check_cross_rust_production(
            detector, before, time, val, time_tol, expr_tol,
        )
        if fired is None:
            fired = detector.check(time, val, time_tol=time_tol, expr_tol=expr_tol)
        self._rust_shadow_check_cross(
            before,
            detector,
            fired,
            time,
            val,
            time_tol,
            expr_tol,
        )
        if fired:
            cross_time = float(detector.t_cross)
            if cross_time + max(1e-18, float(time_tol or 0.0)) < self._step_latest_cross_event_time:
                # Multiple cross() statements can trigger inside one simulator
                # step. Spectre applies their event bodies in chronological
                # crossing order, not source order. EVAS does not yet replay a
                # full event queue, so suppress a retrograde event body instead
                # of letting an earlier crossing discovered later overwrite the
                # state from a later crossing.
                detector.last_triggered = False
                return False
            self._step_latest_cross_event_time = max(
                self._step_latest_cross_event_time,
                cross_time,
            )
            self._step_event_fired = True
            self._perf_stats["cross_fires"] += 1
            trigger_dir = detector.last_trigger_direction or direction
            trigger_went_beyond = detector.last_trigger_went_beyond
            prev_event_time = self._event_time
            prev_cross_directions = dict(self._event_node_cross_directions)
            self._event_time = cross_time

            def resolve_node(node: str) -> str:
                return self._resolve_external_node(node)

            interp_node_set = set(interp_nodes or [])
            if isinstance(nudge_nodes, dict):
                interp_node_set.update(str(node) for node in nudge_nodes)
            elif nudge_nodes:
                interp_node_set.update(str(node) for node in nudge_nodes)
            ext_interp_nodes = {
                resolve_node(node)
                for node in interp_node_set
                if isinstance(node, str)
            }
            self._event_interpolated_nodes = {
                node
                for node in (interp_node_set | ext_interp_nodes)
                if isinstance(node, str)
            }
            self._cache_event_interpolated_values(interp_node_set)

            def event_node_value(node: str) -> float:
                ext = resolve_node(node)
                cached = self._event_interpolated_node_values.get(ext)
                if cached is not None:
                    return cached
                t0 = self._step_prev_time
                t1 = self._step_time
                if t1 > t0 + 1e-30 and ext in self._step_prev_node_voltages:
                    frac = (float(self._event_time) - t0) / (t1 - t0)
                    frac = max(0.0, min(1.0, frac))
                    v0 = self._step_prev_node_voltages[ext]
                    v1 = self._step_curr_node_voltages.get(ext, v0)
                    return v0 + frac * (v1 - v0)
                return self._step_curr_node_voltages.get(ext, self._step_prev_node_voltages.get(ext, 0.0))

            exact_touch_moves_beyond = False
            if not trigger_went_beyond and isinstance(nudge_nodes, dict) and trigger_dir:
                expr_delta = 0.0
                for node, sign in nudge_nodes.items():
                    if not sign:
                        continue
                    ext = resolve_node(node)
                    future_value = self._step_future_node_voltages.get(
                        ext,
                        self._step_curr_node_voltages.get(ext, event_node_value(node)),
                    )
                    expr_delta += float(sign) * (future_value - event_node_value(node))
                exact_touch_moves_beyond = expr_delta * float(trigger_dir) > max(float(expr_tol or 0.0), 1e-18)
            use_post_side = trigger_went_beyond or exact_touch_moves_beyond
            if isinstance(nudge_nodes, dict):
                cross_directions = {
                    node: int(trigger_dir) * (1 if sign > 0 else -1) if use_post_side else 0
                    for node, sign in nudge_nodes.items()
                    if sign
                }
            else:
                cross_directions = {
                    node: int(trigger_dir) if use_post_side else 0
                    for node in set(nudge_nodes or interp_nodes or [])
                }
            if (
                prev_cross_directions
                and abs(float(prev_event_time) - float(self._event_time)) <= max(1e-18, float(time_tol or 0.0))
            ):
                for node, cross_dir in cross_directions.items():
                    if cross_dir or node not in prev_cross_directions:
                        prev_cross_directions[node] = cross_dir
                self._event_node_cross_directions = prev_cross_directions
            else:
                self._event_node_cross_directions = cross_directions
            self._event_trace_audit_record_event(
                "cross",
                key,
                time,
                cross_time,
            )
        return fired

    def _last_crossing(self, key: str, time: float, val: float, direction: int = 0,
                       time_tol: float = 0.0, expr_tol: float = 1e-12) -> float:
        """Return most recent crossing time (approximation for last_crossing())."""
        if key not in self.cross_detectors:
            self.cross_detectors[key] = CrossDetector(direction=direction)
        cd = self.cross_detectors[key]
        cd.check(time, val, time_tol=time_tol, expr_tol=expr_tol)
        return cd.t_cross

    def _check_above(
        self,
        key: str,
        time: float,
        val: float,
        direction: int = 1,
        interp_nodes: Optional[List[str]] = None,
    ) -> bool:
        if key not in self.above_detectors:
            self.above_detectors[key] = AboveDetector(direction=direction)
        detector = self.above_detectors[key]
        before = self._above_detector_shadow_state(detector)
        # Audit 089: production gate for above() detector math.
        fired = self._check_above_rust_production(detector, before, time, val)
        if fired is None:
            fired = detector.check(time, val)
        self._rust_shadow_check_above(before, detector, fired, time, val)
        if fired:
            self._perf_stats["above_fires"] += 1
            self._step_event_fired = True
            self._event_time = detector.t_cross
            interp_node_set = set(interp_nodes or [])
            ext_interp_nodes = {
                self._resolve_external_node(node)
                for node in interp_node_set
                if isinstance(node, str)
            }
            self._event_interpolated_nodes = {
                node
                for node in (interp_node_set | ext_interp_nodes)
                if isinstance(node, str)
            }
            self._cache_event_interpolated_values(interp_node_set)
            self._event_node_cross_directions = {}
            self._event_trace_audit_record_event(
                "above",
                key,
                time,
                detector.t_cross,
            )
        return fired

    def _idtmod(self, key: str, time: float, x: float,
                ic: float = 0.0, mod: float = 1.0) -> float:
        """
        Minimal idtmod integrator with trapezoidal update.

        idtmod(x, ic, mod) ≈ ic + ∫x dt wrapped into [0, mod).
        Notes:
        - Accuracy depends on external timestep control ($bound_step / tran step).
        - Multiple evaluations at the same time do not re-integrate.
        """
        if self._idt_states is None:
            self._idt_states = {}

        if key not in self._idt_states:
            self._idt_states[key] = {
                "y": float(ic),
                "last_t": float(time),
                "last_x": float(x),
                "last_eval_t": float(time),
            }
            y0 = float(ic)
            if mod is not None and float(mod) != 0.0:
                m = abs(float(mod))
                y0 = y0 % m
                self._idt_states[key]["y"] = y0
            return y0

        st = self._idt_states[key]
        if time == st["last_eval_t"]:
            return st["y"]

        dt = float(time) - float(st["last_t"])
        if dt > 0.0:
            st["y"] += 0.5 * (float(x) + st["last_x"]) * dt
            if mod is not None and float(mod) != 0.0:
                m = abs(float(mod))
                st["y"] = st["y"] % m
            st["last_t"] = float(time)
            st["last_x"] = float(x)
        elif dt < 0.0:
            # Time rollback (e.g., restart): re-seed from current value.
            st["last_t"] = float(time)
            st["last_x"] = float(x)

        st["last_eval_t"] = float(time)
        return st["y"]

    def _array_get(self, name: str, idx: int) -> Any:
        layout = self._indexed_state_array_layouts.get(name)
        if layout is not None:
            lo, hi, integer = layout
            idx_i = int(idx)
            if lo <= idx_i <= hi:
                flat_slot = self._indexed_state_array_element_ids.get((name, idx_i))
                values = self._indexed_state_values
                if (
                    values is not None
                    and flat_slot is not None
                    and 0 <= flat_slot < len(values)
                ):
                    value = values[flat_slot]
                else:
                    value = self._indexed_state_array_values[name][idx_i - lo]
                self._perf_stats["indexed_state_array_reads"] += 1
                return self._to_integer(value) if integer else value
        if name in self.arrays and idx in self.arrays[name]:
            return self.arrays[name][idx]
        return 0

    def _array_set(self, name: str, idx: int, val: Any):
        if name not in self.arrays:
            self.arrays[name] = {}
        layout = self._indexed_state_array_layouts.get(name)
        if layout is None:
            self.arrays[name][idx] = val
            self._event_trace_audit_note_write("array", f"{name}[{int(idx)}]")
            return
        lo, hi, integer = layout
        idx_i = int(idx)
        if not (lo <= idx_i <= hi):
            self.arrays[name][idx] = val
            self._perf_stats["indexed_state_array_oob_writes"] += 1
            self._event_trace_audit_note_write("array", f"{name}[{idx_i}]")
            return
        stored = self._to_integer(val) if integer else val
        self.arrays[name][idx_i] = stored
        self._event_trace_audit_note_write("array", f"{name}[{idx_i}]")
        self._indexed_state_array_values[name][idx_i - lo] = float(stored)
        flat_slot = self._indexed_state_array_element_ids.get((name, idx_i))
        values = self._indexed_state_values
        if values is not None and flat_slot is not None and 0 <= flat_slot < len(values):
            values[flat_slot] = float(stored)
        self._perf_stats["indexed_state_array_writes"] += 1

    def _strobe(self, time: float, fmt: str, *args):
        try:
            msg = (fmt % args) if args else fmt
        except Exception as e:
            msg = f"{fmt}  [format error: {e}]"
        self._strobe_log.append((time, msg))

    def _fopen(self, filename: str, mode: str = 'w') -> int:
        fd = self._next_fd
        self._next_fd += 1
        self._file_handles[fd] = open(filename, mode)
        return fd

    def _fclose(self, fd: int):
        if fd in self._file_handles:
            self._file_handles[fd].close()
            del self._file_handles[fd]

    def _fstrobe(self, target: Any, fmt: str, *args):
        try:
            msg = (fmt % args) if args else fmt
        except Exception as e:
            msg = f"{fmt}  [format error: {e}]"

        try:
            fd = int(target)
        except (TypeError, ValueError):
            fd = None

        if fd is not None and fd in self._file_handles:
            self._file_handles[fd].write(msg + '\n')
            return

        if isinstance(target, str):
            with open(target, 'a') as f:
                f.write(msg + '\n')

    def _cleanup_files(self):
        for f in self._file_handles.values():
            f.close()
        self._file_handles.clear()
        for child in self._child_models:
            child._cleanup_files()

    def _seed_to_stream(self, seed: Optional[float]) -> random.Random:
        if seed is None:
            return self._rng_default
        try:
            sid = int(float(seed))
        except Exception:
            return self._rng_default
        if sid not in self._rng_streams:
            self._rng_streams[sid] = random.Random(sid)
        return self._rng_streams[sid]

    @staticmethod
    def _float_to_u64_bits(value: float) -> int:
        import struct

        return struct.unpack(">Q", struct.pack(">d", float(value)))[0]

    @staticmethod
    def _rotl64(value: int, shift: int) -> int:
        mask = (1 << 64) - 1
        value &= mask
        return ((value << shift) & mask) | (value >> (64 - shift))

    @staticmethod
    def _splitmix64(value: int) -> int:
        mask = (1 << 64) - 1
        value = (int(value) + 0x9E3779B97F4A7C15) & mask
        mixed = value
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        return (mixed ^ (mixed >> 31)) & mask

    @staticmethod
    def _uniform01_from_u64(value: int) -> float:
        mantissa = int(value) >> 11
        scale = 1.0 / float(1 << 53)
        return min(max(float(mantissa) * scale, 1.0e-12), 1.0 - 1.0e-12)

    def _rand_normal(
        self,
        seed: Optional[float],
        mean: float,
        std: float,
        time: float = 0.0,
    ) -> float:
        # Stateless hash of (seed, draw index). Hashing the per-seed draw
        # index instead of wall-clock time keeps the sequence identical across
        # engines even when their event times differ at sub-ps level, and
        # matches the LRM's sequential-stream semantics. `time` is accepted
        # for call-site compatibility but no longer enters the hash.
        del time
        seed_value = 0.0 if seed is None else float(seed)
        seed_bits = self._float_to_u64_bits(seed_value)
        counters = getattr(self, "_rdist_draw_indices", None)
        if counters is None:
            counters = {}
            self._rdist_draw_indices = counters
        index = counters.get(seed_bits, 0)
        counters[seed_bits] = index + 1
        stream = seed_bits ^ self._rotl64(index, 17) ^ 0xD1B54A32D192ED03
        u1 = self._uniform01_from_u64(self._splitmix64(stream))
        u2 = self._uniform01_from_u64(self._splitmix64(stream ^ 0xA0761D6478BD642F))
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        return float(mean) + float(std) * radius * math.cos(angle)

    def _rand_uniform(self, seed: Optional[float], lo: float, hi: float) -> float:
        rng = self._seed_to_stream(seed)
        return rng.uniform(float(lo), float(hi))

    def _rand_int32(self, seed: Optional[float] = None) -> int:
        rng = self._seed_to_stream(seed)
        return rng.randint(-2147483648, 2147483647)


def compile_module(
    module: Module,
    default_transition: float = None,
    static_branch_fastpath_codegen: bool = False,
    indexed_state_fastpath_codegen: bool = False,
) -> type:
    """
    Compile a Module AST into a Python class.

    Returns a class (subclass of CompiledModel) that can be instantiated
    and connected to a Simulator.
    """
    compiler = _ModuleCompiler(
        module,
        default_transition,
        static_branch_fastpath_codegen=static_branch_fastpath_codegen,
        indexed_state_fastpath_codegen=indexed_state_fastpath_codegen,
    )
    return compiler.compile()


class _ModuleCompiler:
    def __init__(
        self,
        module: Module,
        default_transition: float = None,
        static_branch_fastpath_codegen: bool = False,
        indexed_state_fastpath_codegen: bool = False,
    ):
        self.module = module
        self.default_transition = default_transition or 1e-12
        self.static_branch_fastpath_codegen = bool(static_branch_fastpath_codegen)
        self.indexed_state_fastpath_codegen = bool(indexed_state_fastpath_codegen)
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
        self._combined_counter = 0
        self._initial_step_counter = 0
        self._final_step_counter = 0
        self._idt_counter = 0
        self._slew_counter = 0
        self._last_cross_counter = 0
        self._uses_idtmod = False
        self._needs_future_node_voltages = False
        self._indent = 2
        self._in_loop_var = None  # track if we're inside a for loop
        self._event_key_cache: Dict[tuple, str] = {}
        self._stateful_func_key_cache: Dict[tuple, str] = {}
        self._discrete_assignment_key_cache: Dict[int, str] = {}
        self._discrete_assignment_counter = 0
        self._static_branch_read_slot_by_node: Dict[str, int] = {}
        self._static_branch_write_slot_by_node: Dict[str, int] = {}
        self._state_scalar_slot_by_name: Dict[str, int] = {}
        self._state_array_range_by_name: Dict[str, Tuple[int, int, bool]] = {}
        self._state_local_name_by_state: Dict[str, str] = {}
        self._state_local_fastpath_active = False
        self._state_local_fastpath_names: set[str] = set()
        self._param_types = {p.name: p.param_type for p in module.parameters}
        self._var_types = {v.name: v.var_type for v in module.variables}
        self._evaluate_ir_static_param_values: Dict[str, Any] = {}
        self._evaluate_ir_static_loop_values: Dict[str, int] = {}
        self._event_lfsr_shift_ir_ops: Dict[str, tuple] = {}
        self._event_static_linear_ir_ops: Dict[str, tuple] = {}
        self._event_timer_static_linear_ir_ops: Dict[str, tuple] = {}
        self._state_owned_timer_targets: Dict[str, str] = {}
        self._rust_output_hold_state_names: set[str] = set()
        self._random_state_names_cache: Optional[set[str]] = None

    def compile(self) -> type:
        """Generate and return a compiled model class."""
        # Build the class dynamically
        mod = self.module

        self._validate_spectre_operator_rules()

        static_param_values: Dict[str, Any] = {}
        static_param_env: Dict[str, Any] = {}
        for p in mod.parameters:
            val = self._eval_expr_static(p.default_value, static_param_env)
            if p.param_type == ParamType.INTEGER:
                val = CompiledModel._to_integer(val)
            static_param_values[p.name] = val
            static_param_env[p.name] = val
        self._evaluate_ir_static_param_values = static_param_values

        # Collect info (port lists reserved for future use)
        branch_io = (
            self._collect_static_branch_io(mod.analog_block.body)
            if mod.analog_block
            else self._empty_static_branch_io()
        )
        rust_static_affine_ops = (
            self._collect_rust_static_affine_ops(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        self._static_branch_read_slot_by_node = {
            node: idx
            for idx, node in enumerate(sorted(branch_io["static_voltage_read_nodes"]))
        }
        self._static_branch_write_slot_by_node = {
            node: idx
            for idx, node in enumerate(sorted(branch_io["static_output_write_nodes"]))
        }

        state_scalar_names = []
        integer_state_names = []
        state_array_ranges = []
        for v in mod.variables:
            is_integer = self._is_integer_decl(v)
            if v.is_array:
                hi = v.array_hi if v.array_hi is not None else 0
                lo = v.array_lo if v.array_lo is not None else 0
                state_array_ranges.append((v.name, min(hi, lo), max(hi, lo), is_integer))
            else:
                state_scalar_names.append(v.name)
                if is_integer:
                    integer_state_names.append(v.name)
        self._state_array_range_by_name = {
            name: (lo, hi, is_integer)
            for name, lo, hi, is_integer in state_array_ranges
        }
        self._state_scalar_slot_by_name = {
            name: idx for idx, name in enumerate(state_scalar_names)
        }
        self._state_local_name_by_state = {
            name: f"_st_{name}" for name in state_scalar_names
        }
        evaluate_ir_static_linear_ops = (
            self._collect_evaluate_ir_static_linear_ops(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        evaluate_ir_static_linear_non_event_ops = (
            self._collect_evaluate_ir_static_linear_non_event_ops(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        evaluate_ir_static_linear_rejections = (
            self._collect_evaluate_ir_static_linear_rejections(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        transition_target_ir_ops = (
            self._collect_transition_target_ir_ops(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        ordered_transition_segment_ir_ops = (
            self._collect_ordered_transition_segment_ir_ops(mod.analog_block.body)
            if mod.analog_block
            else ((), ())
        )
        event_lfsr_output_nodes_by_state = (
            self._collect_simple_state_output_nodes(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        event_lfsr_output_hold_states = (
            self._collect_lfsr_output_hold_states(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        whole_segment_candidates = (
            self._collect_whole_segment_candidates(mod.analog_block.body)
            if mod.analog_block
            else ()
        )
        rust_body_ir_metadata = (
            self._collect_rust_body_ir_metadata(mod, branch_io)
            if mod.analog_block
            else {
                "rejection_reason": "no_analog_block",
                "node_names": (),
                "param_names": (),
                "state_names": (),
                "integer_state_names": (),
                "stmt_ops": (),
                "expr_ops": (),
                "target_node_slots": (),
                "target_state_slots": (),
                "rejection_tags": (),
            }
        )
        event_transition_plan_metadata = self._collect_event_transition_plan_metadata(
            mod,
            tuple(rust_body_ir_metadata["node_names"]),
        )
        self._rust_output_hold_state_names = set(event_lfsr_output_hold_states)
        # Audit 088: static analyzer for transition defer safety.
        self._transition_defer_unsafe_nodes = (
            self._collect_transition_defer_unsafe_nodes(mod.analog_block.body)
            if mod.analog_block
            else set()
        )

        # Build arrays info
        array_vars = {}
        for v in mod.variables:
            if v.is_array:
                hi = v.array_hi if v.array_hi is not None else 0
                lo = v.array_lo if v.array_lo is not None else 0
                array_vars[v.name] = (hi, lo, v.init_values)

        # Build array port info
        array_ports = {}
        for p in mod.port_decls:
            if p.is_array:
                array_ports[p.name] = (p.array_hi, p.array_lo)

        # Generate code for the class
        lines = []
        lines.append(f"class {mod.name}_Model(CompiledModel):")
        lines.append(f"    _module_ports = {mod.ports!r}")
        lines.append("    def __init__(self):")
        lines.append("        super().__init__()")
        lines.append(f"        self.default_transition = {self.default_transition}")

        static_env: Dict[str, Any] = {}

        # Initialize parameters
        for p in mod.parameters:
            val = self._eval_expr_static(p.default_value, static_env)
            if p.param_type == ParamType.INTEGER:
                val = CompiledModel._to_integer(val)
            lines.append(f"        self.params[{p.name!r}] = {val!r}")
            static_env[p.name] = val

        # Initialize scalar state variables
        for v in mod.variables:
            if not v.is_array:
                init_val = 0
                if v.init_values and len(v.init_values) == 1:
                    init_val = self._eval_expr_static(v.init_values[0], static_env)
                if (
                    v.var_type == ParamType.INTEGER
                    or getattr(v.var_type, "name", "") == "INTEGER"
                    or v.var_type in {"integer", "genvar"}
                ):
                    init_val = CompiledModel._to_integer(init_val)
                lines.append(f"        self.state[{v.name!r}] = {init_val}")
                static_env[v.name] = init_val

        # Initialize array variables
        for name, (hi, lo, init_vals) in array_vars.items():
            lines.append(f"        self.arrays[{name!r}] = {{}}")
            lo_idx = min(hi, lo)
            hi_idx = max(hi, lo)
            if init_vals:
                for i, iv in enumerate(init_vals):
                    idx = hi_idx - i
                    val = self._eval_expr_static(iv, static_env)
                    if self._is_integer_variable(name):
                        val = CompiledModel._to_integer(val)
                    lines.append(f"        self.arrays[{name!r}][{idx}] = {val!r}")
            else:
                for idx in range(lo_idx, hi_idx + 1):
                    lines.append(f"        self.arrays[{name!r}][{idx}] = 0")

        # Initialize hierarchical child instances.
        for inst in mod.instances:
            child_var = f"_child_{inst.instance_name}"
            lines.append(f"        _entry = self._module_registry.get({inst.module_name!r})")
            lines.append("        if _entry is None:")
            lines.append(f"            raise CompilationError('Unknown child module: {inst.module_name} in {mod.name}.{inst.instance_name}')")
            lines.append("        _child_cls, _child_mod = _entry")
            lines.append(f"        {child_var} = _child_cls()")
            lines.append(f"        {child_var}._parent_model = self")
            lines.append(f"        {child_var}.node_map = {{}}")
            # Positional and named port connections.
            for ci, c in enumerate(inst.connections):
                if c.port_name is not None:
                    port_expr = repr(c.port_name)
                else:
                    port_expr = f"_child_mod.ports[{ci}] if {ci} < len(_child_mod.ports) else None"
                target = self._compile_instance_target(c.expr)
                lines.append(f"        _pname = {port_expr!s}")
                lines.append("        if _pname is not None:")
                lines.append(f"            _target = {target}")
                lines.append("            if _target in self._module_ports:")
                lines.append("                _mapped = f'@parent:{_target}'")
                lines.append("            else:")
                lines.append(f"                _mapped = f'__{inst.instance_name}.{{_target}}'")
                lines.append(f"            {child_var}.node_map[_pname] = _mapped")
            lines.append(f"        self._child_models.append({child_var})")

        # Generate initial_step method
        lines.append("")
        lines.append("    def initial_step(self, nv, time):")
        lines.append("        self._event_trace_audit_phase = 'initial_step'")
        lines.append("        if self._initial_step_done:")
        lines.append("            return")
        lines.append("        self._initial_step_done = True")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.initial_step(nv, time)")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_initial_step_statement(stmt, 2)
                lines.extend(stmt_lines)
        # Audit 088 fix: initial_step may emit transition_output_lazy via
        # _compile_contribution. Flush so pending state is not orphaned.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._flush_transitions(nv, time)")

        lines.append("        pass")  # ensure method has body

        # Generate evaluate method
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
        self._combined_counter = 0
        self._initial_step_counter = 0
        self._final_step_counter = 0
        self._idt_counter = 0
        self._slew_counter = 0
        self._last_cross_counter = 0
        self._uses_idtmod = False
        self._needs_future_node_voltages = False
        self._event_key_cache = {}
        self._stateful_func_key_cache = {}
        self._contributed_nodes = set()
        if mod.analog_block:
            self._contributed_nodes = self._collect_contributed_nodes(mod.analog_block.body)

        lines.append("")
        lines.append("    def evaluate(self, nv, time):")
        lines.append("        self._event_trace_audit_phase = 'evaluate'")
        lines.append("        self._event_time = time")
        lines.append("        self._bound_step = 0.0")
        # Audit 088: defensive reset of any pending transition slots left over
        # from a previous evaluate that exited abnormally (exception, etc.).
        # The flush at the end of a normal evaluate also clears these, so the
        # reset here is a no-op for the happy path.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._reset_transition_pending()")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.evaluate(nv, time)")
        loop_state_targets: set[str] = set()
        evaluate_state_accesses: set[str] = set()
        if mod.analog_block:
            self._collect_for_loop_state_targets(mod.analog_block.body, loop_state_targets)
            self._collect_evaluate_state_scalar_accesses(mod.analog_block.body, evaluate_state_accesses)
        state_local_names = [
            name for name in state_scalar_names
            if name in evaluate_state_accesses and name not in loop_state_targets
        ]
        state_local_fastpath = (
            self.indexed_state_fastpath_codegen
            and bool(state_local_names)
        )
        previous_state_local_fastpath = self._state_local_fastpath_active
        previous_state_local_names = self._state_local_fastpath_names
        self._state_local_fastpath_active = state_local_fastpath
        self._state_local_fastpath_names = set(state_local_names)
        if state_local_fastpath:
            lines.append("        _state_values = self._indexed_state_values")
            for name in state_local_names:
                slot = self._state_scalar_slot_by_name[name]
                local = self._state_local_name_by_state[name]
                if name in integer_state_names:
                    lines.append(
                        f"        {local} = self._to_integer(_state_values[{slot}]) "
                        f"if _state_values is not None else self.state[{name!r}]"
                    )
                else:
                    lines.append(
                        f"        {local} = _state_values[{slot}] "
                        f"if _state_values is not None else self.state[{name!r}]"
                    )

        if mod.analog_block:
            lines.extend(self._compile_statement(mod.analog_block.body, 2))

        # Audit 088: flush any deferred transition contributions for this
        # model before declaring evaluate() done. Safe even if the analyzer
        # found no deferred sites — the runtime fast-paths an empty queue.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._flush_transitions(nv, time)")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _bs = _ch._bound_step")
        lines.append("            if _bs > 0.0 and (self._bound_step <= 0.0 or _bs < self._bound_step):")
        lines.append("                self._bound_step = _bs")
        if state_local_fastpath:
            for name in state_local_names:
                slot = self._state_scalar_slot_by_name[name]
                local = self._state_local_name_by_state[name]
                if name in integer_state_names:
                    lines.append(f"        {local} = self._to_integer({local})")
                lines.append("        if _state_values is not None:")
                lines.append(f"            _state_values[{slot}] = float({local})")
                lines.append("            if self._event_trace_audit_enabled:")
                lines.append(f"                self._event_trace_audit_note_write('state', {name!r})")
                lines.append("        else:")
                lines.append(f"            self.state[{name!r}] = {local}")
                lines.append("            if self._event_trace_audit_enabled:")
                lines.append(f"                self._event_trace_audit_note_write('state', {name!r})")
            lines.append("        if _state_values is not None:")
            lines.append(
                f"            self._perf_stats['indexed_state_scalar_reads'] += {len(state_local_names)}"
            )
            lines.append(
                f"            self._perf_stats['indexed_state_scalar_writes'] += {len(state_local_names)}"
            )

        lines.append("        pass")
        self._state_local_fastpath_active = previous_state_local_fastpath
        self._state_local_fastpath_names = previous_state_local_names

        lines.append("")
        lines.append("    def post_update_events(self, nv, time):")
        lines.append("        self._event_trace_audit_phase = 'post_update'")
        lines.append("        self._event_time = time")
        lines.append("        _post_event_fired = False")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_post_update_statement(stmt, 2)
                lines.extend(stmt_lines)
        # Audit 088 fix: post_update may emit transition_output_lazy calls.
        # Flush them here, before returning, otherwise pending state leaks
        # into the next evaluate's defensive reset.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._flush_transitions(nv, time)")
        lines.append("        return _post_event_fired")

        lines.append("")
        lines.append("    def refresh_outputs(self, nv, time):")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_refresh_statement(stmt, 2)
                lines.extend(stmt_lines)
        # Audit 088 fix: same as post_update_events — flush at end.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._flush_transitions(nv, time)")
        lines.append("        pass")

        transition_target_probe_lines, transition_target_probe_count = (
            self._compile_transition_target_probe_method(mod.analog_block.body)
            if mod.analog_block
            else self._compile_transition_target_probe_method(None)
        )
        lines.extend(transition_target_probe_lines)

        # Generate final_step method
        lines.append("")
        lines.append("    def final_step(self, nv, time):")
        lines.append("        self._event_trace_audit_phase = 'final_step'")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                if isinstance(stmt, EventStatement):
                    if self._is_final_step_event(stmt.event):
                        key = self._alloc_event_key("final_step", stmt.event)
                        lines.append(f"        self._event_trace_audit_enter_event('final_step', {key!r}, time)")
                        body_lines = self._compile_statement(stmt.body, 2)
                        lines.extend(body_lines)
                        lines.append("        self._event_trace_audit_exit_event()")
        # Audit 088 fix: final_step may emit transition_output_lazy via
        # _compile_statement on final_step events. Flush before children.
        lines.append("        if self._transition_pending_count > 0:")
        lines.append("            self._flush_transitions(nv, time)")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.final_step(nv, time)")
        lines.append("        pass")

        # Compile the class
        code = '\n'.join(lines)

        # Create namespace with required imports
        namespace = {
            'CompiledModel': CompiledModel,
            'CompilationError': CompilationError,
            'math': math,
            'random': random,
            'pow': pow,
            'abs': abs,
            'int': int,
            'float': float,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            raise CompilationError(
                f"Failed to compile module {mod.name}: {e}\n\nGenerated code:\n{code}"
            )

        cls = namespace[f'{mod.name}_Model']
        cls._module_ast = mod
        cls._uses_idtmod = self._uses_idtmod
        cls._needs_future_node_voltages = self._needs_future_node_voltages
        cls._static_branch_fastpath_codegen = self.static_branch_fastpath_codegen
        cls._indexed_state_fastpath_codegen = self.indexed_state_fastpath_codegen
        cls._static_voltage_read_nodes = tuple(
            sorted(branch_io["static_voltage_read_nodes"])
        )
        cls._event_trigger_voltage_read_nodes = tuple(
            sorted(branch_io["event_trigger_voltage_read_nodes"])
        )
        cls._event_voltage_read_nodes = tuple(
            sorted(branch_io["event_voltage_read_nodes"])
        )
        cls._event_body_voltage_read_nodes = cls._event_voltage_read_nodes
        cls._static_output_write_nodes = tuple(
            sorted(branch_io["static_output_write_nodes"])
        )
        cls._dynamic_branch_accesses = tuple(
            sorted(branch_io["dynamic_branch_accesses"])
        )
        cls._dynamic_voltage_read_count = int(branch_io["dynamic_voltage_read_count"])
        cls._dynamic_output_write_count = int(branch_io["dynamic_output_write_count"])
        cls._state_scalar_names = tuple(state_scalar_names)
        cls._integer_state_names = tuple(integer_state_names)
        cls._state_array_ranges = tuple(state_array_ranges)
        cls._rust_static_affine_ops = tuple(rust_static_affine_ops)
        cls._evaluate_ir_static_linear_ops = tuple(evaluate_ir_static_linear_ops)
        cls._evaluate_ir_static_linear_non_event_ops = tuple(
            evaluate_ir_static_linear_non_event_ops
        )
        cls._evaluate_ir_static_linear_rejections = tuple(
            evaluate_ir_static_linear_rejections
        )
        cls._transition_target_ir_ops = tuple(transition_target_ir_ops)
        cls._transition_target_probe_count = int(transition_target_probe_count)
        cls._ordered_transition_segment_ir_ops = ordered_transition_segment_ir_ops
        cls._event_lfsr_shift_ir_ops = tuple(self._event_lfsr_shift_ir_ops.values())
        cls._event_static_linear_ir_ops = tuple(
            self._event_static_linear_ir_ops.values()
        )
        cls._event_timer_static_linear_ir_ops = tuple(
            self._event_timer_static_linear_ir_ops.values()
        )
        cls._event_lfsr_output_nodes_by_state = tuple(
            event_lfsr_output_nodes_by_state
        )
        cls._event_lfsr_output_hold_states = tuple(event_lfsr_output_hold_states)
        cls._whole_segment_candidates = tuple(whole_segment_candidates)
        cls._rust_body_ir_node_names = tuple(rust_body_ir_metadata["node_names"])
        cls._rust_body_ir_param_names = tuple(rust_body_ir_metadata["param_names"])
        cls._rust_body_ir_state_names = tuple(rust_body_ir_metadata["state_names"])
        cls._rust_body_ir_integer_state_names = tuple(
            rust_body_ir_metadata["integer_state_names"]
        )
        cls._rust_body_ir_stmt_ops = tuple(rust_body_ir_metadata["stmt_ops"])
        cls._rust_body_ir_expr_ops = tuple(rust_body_ir_metadata["expr_ops"])
        cls._rust_body_ir_target_node_slots = tuple(
            rust_body_ir_metadata["target_node_slots"]
        )
        cls._rust_body_ir_target_state_slots = tuple(
            rust_body_ir_metadata["target_state_slots"]
        )
        cls._rust_body_ir_rejection_reason = str(
            rust_body_ir_metadata["rejection_reason"]
        )
        cls._rust_body_ir_rejection_tags = tuple(
            rust_body_ir_metadata["rejection_tags"]
        )
        cls._event_transition_plan_profiles = tuple(
            event_transition_plan_metadata["accepted_profiles"]
        )
        cls._event_transition_plan_rejection_reasons = dict(
            event_transition_plan_metadata["rejection_reasons"]
        )
        cls._event_transition_plan_blocker_tags = dict(
            event_transition_plan_metadata["blocker_tags"]
        )
        cls._event_transition_plan_event_count = int(
            event_transition_plan_metadata["event_count"]
        )
        cls._event_transition_plan_due_trigger_count = int(
            event_transition_plan_metadata["due_trigger_count"]
        )
        cls._event_transition_plan_output_write_count = int(
            event_transition_plan_metadata["output_write_count"]
        )
        cls._event_transition_plan_transition_count = int(
            event_transition_plan_metadata["transition_count"]
        )
        cls._event_transition_plan_state_assignment_count = int(
            event_transition_plan_metadata["state_assignment_count"]
        )
        cls._event_transition_plan_side_effect_count = int(
            event_transition_plan_metadata["side_effect_count"]
        )
        cls._event_transition_plan_control_flow_count = int(
            event_transition_plan_metadata["control_flow_count"]
        )
        cls._state_owned_timer_targets = tuple(
            sorted(self._state_owned_timer_targets.items())
        )
        if mod.analog_block:
            cls._has_dynamic_breakpoints = self._statement_has_dynamic_breakpoints(mod.analog_block.body)
            cls._has_post_update_events = self._has_post_update_event(mod.analog_block.body)
            cls._uses_bound_step = self._statement_uses_bound_step(mod.analog_block.body)
        else:
            cls._has_dynamic_breakpoints = False
            cls._has_post_update_events = False
            cls._uses_bound_step = False
        cls._generated_code = code  # Store for debugging
        return cls

    def _validate_spectre_operator_rules(self) -> None:
        """Reject patterns that Spectre VACOMP does not allow."""
        if not self.module.analog_block:
            return
        self._event_assigned_vars = set()
        self._non_event_assigned_vars = set()
        self._collect_assignment_contexts(self.module.analog_block.body, in_event=False)
        self._continuous_vars = self._infer_continuous_vars(self.module.analog_block.body)
        self._check_stmt_for_restricted_operators(
            self.module.analog_block.body,
            conditional_depth=0,
            in_event=False,
        )
        self._check_transition_targets(self.module.analog_block.body)

    def _check_stmt_for_restricted_operators(
        self,
        stmt,
        conditional_depth: int,
        in_event: bool,
    ) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._check_stmt_for_restricted_operators(child, conditional_depth, in_event)
            return

        if isinstance(stmt, IfStatement):
            self._check_stmt_for_restricted_operators(
                stmt.then_body,
                conditional_depth + 1,
                in_event,
            )
            if stmt.else_body is not None:
                self._check_stmt_for_restricted_operators(
                    stmt.else_body,
                    conditional_depth + 1,
                    in_event,
                )
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._check_stmt_for_restricted_operators(
                    item.body,
                    conditional_depth + 1,
                    in_event,
                )
            return

        if isinstance(stmt, EventStatement):
            if conditional_depth > 0:
                restricted = self._collect_restricted_events_from_event(stmt.event)
                if restricted:
                    ops = ', '.join(sorted(restricted))
                    raise CompilationError(
                        f"Module {self.module.name} uses Spectre-restricted operator(s) "
                        f"{ops} inside a conditionally executed statement. "
                        f"Move these operators out of if/case branches."
                    )
            event_is_initial_step = (
                isinstance(stmt.event, EventExpr)
                and stmt.event.event_type == EventType.INITIAL_STEP
            )
            self._check_stmt_for_restricted_operators(
                stmt.body,
                conditional_depth + 1,
                False if event_is_initial_step else True,
            )
            return

        if isinstance(stmt, ForStatement):
            self._check_stmt_for_restricted_operators(stmt.body, conditional_depth, in_event)
            return

        if isinstance(stmt, WhileStatement):
            self._check_stmt_for_restricted_operators(stmt.body, conditional_depth + 1, in_event)
            return

        if isinstance(stmt, Contribution) and in_event:
            raise CompilationError(
                f"Module {self.module.name} embeds a contribution statement inside "
                "an analog event body. Spectre rejects this with VACOMP-2157; "
                "move the contribution to unconditional analog code and update "
                "state variables in the event body."
            )

        if conditional_depth <= 0:
            return

        restricted = self._collect_restricted_calls_from_stmt(stmt)
        if restricted:
            ops = ', '.join(sorted(restricted))
            raise CompilationError(
                f"Module {self.module.name} uses Spectre-restricted operator(s) "
                f"{ops} inside a conditionally executed statement. "
                f"Move these operators out of if/case branches."
            )

    def _collect_restricted_calls_from_stmt(self, stmt) -> set:
        restricted = set()
        if isinstance(stmt, Assignment):
            restricted |= self._collect_restricted_calls_from_expr(stmt.value)
        elif isinstance(stmt, Contribution):
            restricted |= self._collect_restricted_calls_from_expr(stmt.expr)
        elif isinstance(stmt, SystemTask):
            for arg in stmt.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
        return restricted

    def _collect_restricted_events_from_event(self, event) -> set:
        restricted = set()

        if isinstance(event, EventExpr):
            if event.event_type == EventType.CROSS:
                restricted.add('cross')
            elif event.event_type == EventType.ABOVE:
                restricted.add('above')
            return restricted

        if isinstance(event, CombinedEvent):
            for child in event.events:
                restricted |= self._collect_restricted_events_from_event(child)
            return restricted

        return restricted

    def _collect_restricted_calls_from_expr(self, expr: Expr) -> set:
        restricted = set()

        if isinstance(expr, FunctionCall):
            if expr.name in ('transition', 'idtmod'):
                restricted.add(expr.name)
            for arg in expr.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
            return restricted

        if isinstance(expr, BinaryExpr):
            restricted |= self._collect_restricted_calls_from_expr(expr.left)
            restricted |= self._collect_restricted_calls_from_expr(expr.right)
            return restricted

        if isinstance(expr, UnaryExpr):
            return self._collect_restricted_calls_from_expr(expr.operand)

        if isinstance(expr, TernaryExpr):
            restricted |= self._collect_restricted_calls_from_expr(expr.cond)
            restricted |= self._collect_restricted_calls_from_expr(expr.true_expr)
            restricted |= self._collect_restricted_calls_from_expr(expr.false_expr)
            return restricted

        if isinstance(expr, ArrayAccess):
            return self._collect_restricted_calls_from_expr(expr.index)

        if isinstance(expr, BranchAccess):
            if expr.node1_index is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node1_index)
            if expr.node1_index2 is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node1_index2)
            if expr.node2_index is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node2_index)
            if expr.node2_index2 is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node2_index2)
            return restricted

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
            return restricted

        return restricted

    def _infer_continuous_vars(self, stmt) -> set[str]:
        continuous_vars = set()
        changed = True
        while changed:
            changed = False
            for target_name, value_expr in self._iter_assignments(stmt):
                if target_name not in self._non_event_assigned_vars:
                    continue
                if self._expr_is_continuous(value_expr, continuous_vars) and target_name not in continuous_vars:
                    continuous_vars.add(target_name)
                    changed = True
        return continuous_vars

    def _collect_assignment_contexts(self, stmt, in_event: bool) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_assignment_contexts(child, in_event)
            return

        if isinstance(stmt, EventStatement):
            self._collect_assignment_contexts(stmt.body, True)
            return

        if isinstance(stmt, IfStatement):
            self._collect_assignment_contexts(stmt.then_body, in_event)
            if stmt.else_body is not None:
                self._collect_assignment_contexts(stmt.else_body, in_event)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._collect_assignment_contexts(item.body, in_event)
            return

        if isinstance(stmt, ForStatement):
            self._collect_assignment_contexts(stmt.body, in_event)
            return

        if isinstance(stmt, WhileStatement):
            self._collect_assignment_contexts(stmt.body, in_event)
            return

        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier):
                name = target.name
            elif isinstance(target, ArrayAccess):
                name = target.name
            else:
                return
            if in_event:
                self._event_assigned_vars.add(name)
            else:
                self._non_event_assigned_vars.add(name)

    def _iter_assignments(self, stmt):
        if isinstance(stmt, Block):
            for child in stmt.statements:
                yield from self._iter_assignments(child)
            return

        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier):
                yield target.name, stmt.value
            elif isinstance(target, ArrayAccess):
                yield target.name, stmt.value
            return

        if isinstance(stmt, IfStatement):
            yield from self._iter_assignments(stmt.then_body)
            if stmt.else_body is not None:
                yield from self._iter_assignments(stmt.else_body)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                yield from self._iter_assignments(item.body)
            return

        if isinstance(stmt, EventStatement):
            yield from self._iter_assignments(stmt.body)
            return

        if isinstance(stmt, ForStatement):
            yield from self._iter_assignments(stmt.body)
            return

        if isinstance(stmt, WhileStatement):
            yield from self._iter_assignments(stmt.body)
            return

    def _expr_is_continuous(self, expr: Expr, continuous_vars: set[str]) -> bool:
        if isinstance(expr, NumberLiteral):
            return False

        if isinstance(expr, StringLiteral):
            return False

        if isinstance(expr, Identifier):
            return expr.name in continuous_vars

        if isinstance(expr, ArrayAccess):
            return expr.name in continuous_vars or self._expr_is_continuous(expr.index, continuous_vars)

        if isinstance(expr, BranchAccess):
            return True

        if isinstance(expr, BinaryExpr):
            return (
                self._expr_is_continuous(expr.left, continuous_vars)
                or self._expr_is_continuous(expr.right, continuous_vars)
            )

        if isinstance(expr, UnaryExpr):
            return self._expr_is_continuous(expr.operand, continuous_vars)

        if isinstance(expr, TernaryExpr):
            return (
                self._expr_is_continuous(expr.cond, continuous_vars)
                or self._expr_is_continuous(expr.true_expr, continuous_vars)
                or self._expr_is_continuous(expr.false_expr, continuous_vars)
            )

        if isinstance(expr, MethodCall):
            return any(self._expr_is_continuous(arg, continuous_vars) for arg in expr.args)

        if isinstance(expr, FunctionCall):
            if expr.name == 'transition':
                return True
            if expr.name == 'idtmod':
                return True
            return any(self._expr_is_continuous(arg, continuous_vars) for arg in expr.args)

        return False

    def _check_transition_targets(self, stmt) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._check_transition_targets(child)
            return

        if isinstance(stmt, IfStatement):
            self._check_transition_targets(stmt.then_body)
            if stmt.else_body is not None:
                self._check_transition_targets(stmt.else_body)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._check_transition_targets(item.body)
            return

        if isinstance(stmt, EventStatement):
            self._check_transition_targets(stmt.body)
            return

        if isinstance(stmt, ForStatement):
            self._check_transition_targets(stmt.body)
            return

        if isinstance(stmt, WhileStatement):
            self._check_transition_targets(stmt.body)
            return

        for call in self._iter_function_calls_in_stmt(stmt):
            if call.name == 'transition' and call.args:
                if self._transition_target_is_continuous(call.args[0]):
                    raise CompilationError(
                        f"Module {self.module.name} applies transition() to a continuous-valued "
                        f"expression. Spectre expects the transition target to be piecewise constant. "
                        f"Move continuous scaling outside transition() or contribute the signal directly."
                    )

    def _transition_target_is_continuous(self, expr: Expr) -> bool:
        # Spectre/Virtuoso accepts a wider transition() target surface than the
        # earlier EVAS guard assumed, including continuous-affine expressions
        # such as `transition(V(vres_p) + dither_diff * 0.5, 0)`. EVAS evaluates
        # transition() dynamically at runtime, so blocking these forms at compile
        # time creates benchmark/example mismatches rather than protecting a hard
        # simulator limitation.
        return False

    def _iter_function_calls_in_stmt(self, stmt):
        if isinstance(stmt, Assignment):
            yield from self._iter_function_calls_in_expr(stmt.value)
            return

        if isinstance(stmt, Contribution):
            yield from self._iter_function_calls_in_expr(stmt.expr)
            return

        if isinstance(stmt, SystemTask):
            for arg in stmt.args:
                yield from self._iter_function_calls_in_expr(arg)

    def _iter_function_calls_in_expr(self, expr: Expr):
        if isinstance(expr, FunctionCall):
            yield expr
            for arg in expr.args:
                yield from self._iter_function_calls_in_expr(arg)
            return

        if isinstance(expr, BinaryExpr):
            yield from self._iter_function_calls_in_expr(expr.left)
            yield from self._iter_function_calls_in_expr(expr.right)
            return

        if isinstance(expr, UnaryExpr):
            yield from self._iter_function_calls_in_expr(expr.operand)
            return

        if isinstance(expr, TernaryExpr):
            yield from self._iter_function_calls_in_expr(expr.cond)
            yield from self._iter_function_calls_in_expr(expr.true_expr)
            yield from self._iter_function_calls_in_expr(expr.false_expr)
            return

        if isinstance(expr, ArrayAccess):
            yield from self._iter_function_calls_in_expr(expr.index)
            return

        if isinstance(expr, BranchAccess):
            if expr.node1_index is not None:
                yield from self._iter_function_calls_in_expr(expr.node1_index)
            if expr.node1_index2 is not None:
                yield from self._iter_function_calls_in_expr(expr.node1_index2)
            if expr.node2_index is not None:
                yield from self._iter_function_calls_in_expr(expr.node2_index)
            if expr.node2_index2 is not None:
                yield from self._iter_function_calls_in_expr(expr.node2_index2)
            return

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                yield from self._iter_function_calls_in_expr(arg)

    def _collect_contributed_nodes(self, stmt) -> set[str]:
        nodes: set[str] = set()

        if isinstance(stmt, Block):
            for s in stmt.statements:
                nodes.update(self._collect_contributed_nodes(s))
            return nodes

        if isinstance(stmt, Contribution):
            nodes.add(stmt.branch.node1)
            if stmt.branch.node2 is not None:
                nodes.add(stmt.branch.node2)
            return nodes

        if isinstance(stmt, EventStatement):
            nodes.update(self._collect_contributed_nodes(stmt.body))
            return nodes

        if isinstance(stmt, IfStatement):
            nodes.update(self._collect_contributed_nodes(stmt.then_body))
            if stmt.else_body is not None:
                nodes.update(self._collect_contributed_nodes(stmt.else_body))
            return nodes

        if isinstance(stmt, WhileStatement):
            nodes.update(self._collect_contributed_nodes(stmt.body))
            return nodes

        if isinstance(stmt, ForStatement):
            nodes.update(self._collect_contributed_nodes(stmt.body))
            return nodes

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                nodes.update(self._collect_contributed_nodes(item.body))
            return nodes

        return nodes

    def _empty_static_branch_io(self) -> Dict[str, Any]:
        return {
            "static_voltage_read_nodes": set(),
            "event_trigger_voltage_read_nodes": set(),
            "event_voltage_read_nodes": set(),
            "static_output_write_nodes": set(),
            "dynamic_branch_accesses": set(),
            "dynamic_voltage_read_count": 0,
            "dynamic_output_write_count": 0,
        }

    def _collect_rust_body_ir_metadata(
        self,
        mod: Module,
        branch_io: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect conservative 094 body-IR metadata for generated classes."""

        empty = {
            "node_names": (),
            "param_names": (),
            "state_names": (),
            "integer_state_names": (),
            "stmt_ops": (),
            "expr_ops": (),
            "target_node_slots": (),
            "target_state_slots": (),
            "rejection_tags": (),
        }

        analog_block = getattr(mod, "analog_block", None)
        body = getattr(analog_block, "body", None)
        if body is None:
            return {**empty, "rejection_reason": "no_analog_block"}

        stmt_ir = lower_stmt(body)
        if stmt_ir is None:
            return {
                **empty,
                "rejection_reason": "stmt_lower_failed",
                "rejection_tags": ("stmt_lower_failed",),
            }

        node_names = tuple(
            sorted(
                set(str(name) for name in getattr(mod, "ports", ()) or ())
                | set(str(name) for name in branch_io["static_voltage_read_nodes"])
                | set(str(name) for name in branch_io["static_output_write_nodes"])
            )
        )
        node_slots = {name: idx for idx, name in enumerate(node_names)}
        bindings = build_state_binding_ir(mod)
        program = encode_body_stmt_ops(stmt_ir, bindings, node_slots)
        if program is None:
            rejection_tags = classify_body_stmt_ops_rejection(
                stmt_ir,
                bindings,
                node_slots,
            )
            return {
                **empty,
                "node_names": node_names,
                "rejection_reason": self._primary_rust_body_ir_rejection_reason(
                    rejection_tags
                ),
                "rejection_tags": rejection_tags,
            }
        if not program.stmt_ops:
            return {
                **empty,
                "node_names": node_names,
                "rejection_reason": "empty_body",
                "rejection_tags": ("empty_body",),
            }

        param_names = tuple(str(param.name) for param in getattr(mod, "parameters", ()) or ())
        state_names = tuple(
            str(variable.name)
            for variable in getattr(mod, "variables", ()) or ()
            if not getattr(variable, "is_array", False)
        )
        integer_state_names = tuple(
            str(variable.name)
            for variable in getattr(mod, "variables", ()) or ()
            if (
                not getattr(variable, "is_array", False)
                and self._is_integer_decl(variable)
            )
        )
        target_node_slots = tuple(
            sorted(
                {
                    int(op.target_id)
                    for op in program.stmt_ops
                    if int(op.target_kind) == BODY_TARGET_NODE
                }
            )
        )
        target_state_slots = tuple(
            sorted(
                {
                    int(op.target_id)
                    for op in program.stmt_ops
                    if int(op.target_kind) == BODY_TARGET_STATE
                }
            )
        )
        return {
            "rejection_reason": "ok",
            "node_names": node_names,
            "param_names": param_names,
            "state_names": state_names,
            "integer_state_names": integer_state_names,
            "stmt_ops": tuple(program.stmt_ops),
            "expr_ops": tuple(program.expr_ops),
            "target_node_slots": target_node_slots,
            "target_state_slots": target_state_slots,
            "rejection_tags": (),
        }

    def _collect_event_transition_plan_metadata(
        self,
        mod: Module,
        node_names: Tuple[str, ...],
    ) -> Dict[str, Any]:
        """Collect compiler-visible 101 event/transition planner metadata."""

        plans = {
            profile_name: analyze_event_transition_segment_plan(
                mod,
                node_names,
                profile=profile_name,
                supported_tags=supported_tags,
            )
            for profile_name, supported_tags in EVENT_TRANSITION_PROFILE_SUPPORT.items()
        }
        return summarize_event_transition_plans(plans)

    @staticmethod
    def _primary_rust_body_ir_rejection_reason(tags: Tuple[str, ...]) -> str:
        """Pick a stable primary reason from diagnostic rejection tags."""

        priority = (
            "event_statement",
            "transition_expr",
            "complex_if_write_set",
            "array_assignment_target",
            "array_read_or_dynamic_index",
            "differential_output_target",
            "indexed_output_target",
            "special_identifier:$abstime",
            "for_loop",
            "case_statement",
            "while_loop",
        )
        tag_set = set(tags)
        for tag in priority:
            if tag in tag_set:
                return tag
        for prefix in ("system_task:", "system_function:", "event_", "unsupported_"):
            for tag in tags:
                if tag.startswith(prefix):
                    return tag
        return tags[0] if tags else "body_stmt_ops_unsupported"

    def _collect_static_branch_io(self, stmt) -> Dict[str, Any]:
        acc = self._empty_static_branch_io()
        self._collect_static_branch_io_from_stmt(stmt, acc, in_event_body=False)
        return acc

    def _collect_rust_static_affine_ops(
        self,
        stmt,
    ) -> Tuple[Tuple[str, str, Any, Any], ...]:
        """Collect a conservative Rust-lowerable static affine model body.

        The current Rust prototype only handles unconditional voltage-domain
        contributions shaped as ``V(out) <+ gain * V(in) + bias``.  Coefficients
        may be literal numeric values or parameter-only scalar expressions that
        are evaluated after instance parameter overrides have been applied.
        Anything involving events, state variables, dynamic bus indexes,
        function calls, string parameters, or differential branches falls back
        to the normal Python evaluator.
        """
        ops: List[Tuple[str, str, Any, Any]] = []
        if not self._collect_rust_static_affine_ops_from_stmt(stmt, ops):
            return ()
        return tuple(ops)

    def _collect_rust_static_affine_ops_from_stmt(
        self,
        stmt,
        ops: List[Tuple[str, str, Any, Any]],
    ) -> bool:
        if stmt is None:
            return True

        if isinstance(stmt, Block):
            for child in stmt.statements:
                if not self._collect_rust_static_affine_ops_from_stmt(child, ops):
                    return False
            return True

        if not isinstance(stmt, Contribution):
            return False

        branch = stmt.branch
        if (
            branch.access_type != "V"
            or branch.node2 is not None
            or branch.node1_index is not None
            or branch.node1_index2 is not None
        ):
            return False

        affine = self._rust_affine_expr(stmt.expr)
        if affine is None:
            return False
        read_node, gain, bias = affine
        ops.append((read_node or branch.node1, branch.node1, gain, bias))
        return True

    def _collect_evaluate_ir_static_linear_ops(
        self,
        stmt,
    ) -> Tuple[Tuple[int, str, Any, Tuple[Tuple[int, str, Any], ...]], ...]:
        """Collect an ordered static-linear evaluate IR for native execution.

        This is deliberately stricter than the general Python code generator.
        It accepts only unconditional scalar assignments and voltage
        contributions whose expressions are linear combinations of static node
        voltages, scalar state values, and parameter-only scalar coefficients.
        Unsupported statements return an empty plan, causing the simulator to
        use the normal Python evaluator.
        """

        ops: List[Tuple[int, str, Any, Tuple[Tuple[int, str, Any], ...]]] = []
        if not self._collect_evaluate_ir_static_linear_ops_from_stmt(stmt, ops):
            return ()
        return tuple(ops)

    def _collect_evaluate_ir_static_linear_non_event_ops(
        self,
        stmt,
    ) -> Tuple[Tuple[int, str, Any, Tuple[Tuple[int, str, Any], ...]], ...]:
        """Collect continuous evaluate IR while skipping event statements.

        Whole-segment native executors own event scheduling separately, but
        still need the ordinary continuous output contribution segment.
        """

        ops: List[Tuple[int, str, Any, Tuple[Tuple[int, str, Any], ...]]] = []

        def visit(node) -> bool:
            if node is None:
                return True
            if isinstance(node, Block):
                return all(visit(child) for child in node.statements)
            if isinstance(node, EventStatement):
                return True
            return self._collect_evaluate_ir_static_linear_ops_from_stmt(node, ops)

        if not visit(stmt):
            return ()
        return tuple(ops)

    def _static_float_or_none(self, value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # =====================================================================
    # Audit 088: Transition per-step batch — static defer-safety analyzer.
    # =====================================================================
    #
    # A `V(out) <+ ... transition(...) ...` contribution can be deferred to
    # an end-of-evaluate batch FFI only if no subsequent statement in the
    # same analog block reads V(out). Otherwise the subsequent read would
    # see a stale previous-step value while it expected the current-step
    # write to have already happened. This analyzer returns the set of
    # node names whose transition contributions must therefore stay on the
    # immediate (non-batch) path. Implementation is intentionally
    # conservative — any uncertainty flags the node as unsafe.

    def _expr_contains_transition_call(self, expr) -> bool:
        if expr is None:
            return False
        if isinstance(expr, FunctionCall):
            if expr.name == "transition":
                return True
            return any(
                self._expr_contains_transition_call(a) for a in expr.args
            )
        if isinstance(expr, BinaryExpr):
            return (
                self._expr_contains_transition_call(expr.left)
                or self._expr_contains_transition_call(expr.right)
            )
        if isinstance(expr, UnaryExpr):
            return self._expr_contains_transition_call(expr.operand)
        if isinstance(expr, TernaryExpr):
            return (
                self._expr_contains_transition_call(expr.cond)
                or self._expr_contains_transition_call(expr.true_expr)
                or self._expr_contains_transition_call(expr.false_expr)
            )
        if isinstance(expr, MethodCall):
            return any(
                self._expr_contains_transition_call(a) for a in expr.args
            )
        if isinstance(expr, ArrayAccess):
            return self._expr_contains_transition_call(expr.index)
        return False

    def _expr_voltage_reads(self, expr, out: set) -> None:
        if expr is None:
            return
        if isinstance(expr, BranchAccess):
            if expr.access_type == "V" and expr.node1 is not None:
                out.add(expr.node1)
            for sub in (
                expr.node1_index, expr.node1_index2,
                expr.node2_index, expr.node2_index2,
            ):
                self._expr_voltage_reads(sub, out)
            return
        if isinstance(expr, FunctionCall):
            for a in expr.args:
                self._expr_voltage_reads(a, out)
            return
        if isinstance(expr, BinaryExpr):
            self._expr_voltage_reads(expr.left, out)
            self._expr_voltage_reads(expr.right, out)
            return
        if isinstance(expr, UnaryExpr):
            self._expr_voltage_reads(expr.operand, out)
            return
        if isinstance(expr, TernaryExpr):
            self._expr_voltage_reads(expr.cond, out)
            self._expr_voltage_reads(expr.true_expr, out)
            self._expr_voltage_reads(expr.false_expr, out)
            return
        if isinstance(expr, MethodCall):
            for a in expr.args:
                self._expr_voltage_reads(a, out)
            return
        if isinstance(expr, ArrayAccess):
            self._expr_voltage_reads(expr.index, out)
            return

    def _collect_transition_defer_timeline(self, stmt, timeline: list) -> None:
        """DFS walk in source order, recording (kind, node) events.

        kind in {'transition_write', 'read'}; node is the LHS or read node.
        Conservative: any uncertainty produces an extra 'read' entry so the
        analyzer falls back to immediate (no defer).
        """
        if stmt is None:
            return
        if isinstance(stmt, Block):
            for s in stmt.statements:
                self._collect_transition_defer_timeline(s, timeline)
            return
        if isinstance(stmt, IfStatement):
            reads: set = set()
            self._expr_voltage_reads(stmt.cond, reads)
            for r in reads:
                timeline.append(("read", r))
            self._collect_transition_defer_timeline(stmt.then_body, timeline)
            self._collect_transition_defer_timeline(stmt.else_body, timeline)
            return
        if isinstance(stmt, ForStatement):
            self._collect_transition_defer_timeline(stmt.init, timeline)
            reads = set()
            self._expr_voltage_reads(stmt.cond, reads)
            for r in reads:
                timeline.append(("read", r))
            self._collect_transition_defer_timeline(stmt.body, timeline)
            self._collect_transition_defer_timeline(stmt.update, timeline)
            return
        if isinstance(stmt, WhileStatement):
            reads = set()
            self._expr_voltage_reads(stmt.cond, reads)
            for r in reads:
                timeline.append(("read", r))
            self._collect_transition_defer_timeline(stmt.body, timeline)
            return
        if isinstance(stmt, CaseStatement):
            reads = set()
            self._expr_voltage_reads(stmt.expr, reads)
            for r in reads:
                timeline.append(("read", r))
            for case_item in getattr(stmt, "items", ()):
                self._collect_transition_defer_timeline(
                    getattr(case_item, "body", None), timeline
                )
            return
        if isinstance(stmt, EventStatement):
            self._collect_transition_defer_timeline(stmt.body, timeline)
            return
        if isinstance(stmt, Contribution):
            reads = set()
            self._expr_voltage_reads(stmt.expr, reads)
            for r in reads:
                timeline.append(("read", r))
            has_trans = (
                stmt.branch.access_type == "V"
                and stmt.branch.node1 is not None
                and stmt.branch.node1_index is None
                and stmt.branch.node1_index2 is None
                and self._expr_contains_transition_call(stmt.expr)
            )
            if has_trans:
                timeline.append(("transition_write", stmt.branch.node1))
            return
        if isinstance(stmt, Assignment):
            reads = set()
            self._expr_voltage_reads(stmt.value, reads)
            for r in reads:
                timeline.append(("read", r))
            return
        # Other statement types (initial_step / final_step / refresh / etc.):
        # conservatively walk any expr/body attributes we recognize.
        for attr in ("body", "expr", "value", "cond"):
            child = getattr(stmt, attr, None)
            if isinstance(child, (Block, IfStatement, ForStatement,
                                  WhileStatement, CaseStatement,
                                  EventStatement, Contribution, Assignment)):
                self._collect_transition_defer_timeline(child, timeline)

    def _collect_transition_defer_unsafe_nodes(self, stmt) -> set:
        """Return the set of LHS node names whose transition() contribution
        cannot be safely batch-deferred to end of evaluate(). See audit 088."""
        timeline: list = []
        self._collect_transition_defer_timeline(stmt, timeline)
        unsafe: set = set()
        for i, (kind, node) in enumerate(timeline):
            if kind != "transition_write":
                continue
            for j in range(i + 1, len(timeline)):
                kind2, node2 = timeline[j]
                if kind2 == "read" and node2 == node:
                    unsafe.add(node)
                    break
        return unsafe

    def _collect_simple_state_output_nodes(self, stmt) -> Tuple[Tuple[str, str], ...]:
        """Collect simple ``V(out) <+ state`` mappings outside event bodies."""
        mapping: Dict[str, set[str]] = {}

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                return
            if not isinstance(node, Contribution):
                return
            branch = node.branch
            if (
                branch.access_type != "V"
                or branch.node1_index is not None
                or branch.node1_index2 is not None
                or branch.node2 is not None
                or branch.node2_index is not None
                or branch.node2_index2 is not None
            ):
                return
            linear = self._evaluate_ir_linear_expr(node.expr)
            if linear is None:
                return
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return
            bias, terms, condition, false_bias, false_terms = unpacked
            bias_value = self._static_float_or_none(bias)
            false_bias_value = self._static_float_or_none(false_bias)
            if bias_value is None or false_bias_value is None:
                return
            if (
                condition is not None
                or false_terms
                or abs(bias_value) > 1e-18
                or abs(false_bias_value) > 1e-18
                or len(terms) != 1
            ):
                return
            kind, state_name, gain = terms[0]
            if kind != SOURCE_STATE or abs(float(gain) - 1.0) > 1e-18:
                return
            mapping.setdefault(str(state_name), set()).add(str(branch.node1))

        visit(stmt)
        return tuple(
            (state, next(iter(nodes)))
            for state, nodes in sorted(mapping.items())
            if len(nodes) == 1
        )

    def _simple_state_output_contribution(self, stmt: Contribution) -> Optional[str]:
        branch = stmt.branch
        if (
            branch.access_type != "V"
            or branch.node1_index is not None
            or branch.node1_index2 is not None
            or branch.node2 is not None
            or branch.node2_index is not None
            or branch.node2_index2 is not None
        ):
            return None
        linear = self._evaluate_ir_linear_expr(stmt.expr)
        if linear is None:
            return None
        unpacked = self._evaluate_ir_unpack_linear(linear)
        if unpacked is None:
            return None
        bias, terms, condition, false_bias, false_terms = unpacked
        bias_value = self._static_float_or_none(bias)
        false_bias_value = self._static_float_or_none(false_bias)
        if bias_value is None or false_bias_value is None:
            return None
        if (
            condition is not None
            or false_terms
            or abs(bias_value) > 1e-18
            or abs(false_bias_value) > 1e-18
            or len(terms) != 1
        ):
            return None
        kind, state_name, gain = terms[0]
        if kind != SOURCE_STATE or abs(float(gain) - 1.0) > 1e-18:
            return None
        return str(state_name)

    def _collect_lfsr_event_output_states(self, stmt) -> set[str]:
        states: set[str] = set()
        if stmt is None:
            return states
        if isinstance(stmt, Block):
            for child in stmt.statements:
                states.update(self._collect_lfsr_event_output_states(child))
            return states
        if isinstance(stmt, EventStatement):
            event_ir = self._event_body_lfsr_shift_ir("__output_state_probe__", stmt.body)
            if event_ir is not None:
                states.add(str(event_ir[8]))
        return states

    def _assigned_candidate_states(self, stmt, candidates: set[str]) -> set[str]:
        assigned: set[str] = set()
        if stmt is None or not candidates:
            return assigned
        if isinstance(stmt, Block):
            for child in stmt.statements:
                assigned.update(self._assigned_candidate_states(child, candidates))
            return assigned
        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier) and target.name in candidates:
                assigned.add(str(target.name))
            return assigned
        if isinstance(stmt, EventStatement):
            assigned.update(self._assigned_candidate_states(stmt.body, candidates))
            return assigned
        if isinstance(stmt, IfStatement):
            assigned.update(self._assigned_candidate_states(stmt.then_body, candidates))
            assigned.update(self._assigned_candidate_states(stmt.else_body, candidates))
            return assigned
        if isinstance(stmt, ForStatement):
            assigned.update(self._assigned_candidate_states(stmt.body, candidates))
            assigned.update(self._assigned_candidate_states(stmt.init, candidates))
            assigned.update(self._assigned_candidate_states(stmt.update, candidates))
            return assigned
        if isinstance(stmt, WhileStatement):
            assigned.update(self._assigned_candidate_states(stmt.body, candidates))
            return assigned
        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                assigned.update(self._assigned_candidate_states(item.body, candidates))
        return assigned

    def _collect_lfsr_output_hold_states(self, stmt) -> Tuple[str, ...]:
        candidates = self._collect_lfsr_event_output_states(stmt)
        if not candidates:
            return ()
        unsafe: set[str] = set()

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                if self._is_initial_step_event(node.event) or self._is_final_step_event(node.event):
                    return
                event_ir = self._event_body_lfsr_shift_ir("__hold_state_probe__", node.body)
                if event_ir is not None and str(event_ir[8]) in candidates:
                    return
                unsafe.update(self._assigned_candidate_states(node.body, candidates))
                return
            unsafe.update(self._assigned_candidate_states(node, candidates))

        visit(stmt)
        return tuple(sorted(candidates - unsafe))

    def _collect_whole_segment_candidates(self, stmt) -> Tuple[tuple, ...]:
        """Collect compiler-visible whole-segment lowering candidates.

        Candidates are conservative AST-level summaries.  They do not change
        normal generated-Python execution; the simulator may opt into a native
        whole-segment executor only when the surrounding netlist also matches.
        """

        candidates: List[tuple] = []
        for collector in (
            self._collect_cross_scalar_lfsr_transition_candidate,
            self._collect_gain_timer_reduction_candidate,
            self._collect_cmp_delay_candidate,
            self._collect_edge_interval_timer_candidate,
            self._collect_weighted_sar_adc_candidate,
            self._collect_weighted_dac_candidate,
            self._collect_sample_hold_candidate,
            self._collect_ref_step_clock_candidate,
            self._collect_cppll_timer_candidate,
        ):
            candidate = collector(stmt)
            if candidate is not None:
                contract = validate_whole_segment_candidate(candidate)
                if contract.valid:
                    candidates.append(candidate)
        # Audit 091b: if no specific shape matched, try the generic
        # event-driven state-transition matcher. This is intentionally a
        # last-resort fallback so it does not override a more-specific
        # candidate kind on the same module.
        if not candidates:
            generic = self._collect_generic_event_state_transition_candidate(stmt)
            if generic is not None:
                contract = validate_whole_segment_candidate(generic)
                if contract.valid:
                    candidates.append(generic)
        return tuple(candidates)

    def _collect_generic_event_state_transition_candidate(self, stmt) -> Optional[tuple]:
        """Audit 091b: match the common "event-driven FSM + transition outputs"
        shape that accounts for ~91% of the 249 stuck models in the rust
        coverage manifest. Conservative gates:
        - >=1 cross() or timer() event (initial_step alone is not enough)
        - >=1 contribution outside any event whose RHS contains transition()
        - event bodies contain only Assignment + If/Else (no Contribution,
          no For/While/Case, no system tasks, no array writes)
        - all assignment targets are scalar state variables
        - no dynamic bus access anywhere in the analog block
        - no $strobe / $display / other system tasks
        """
        sem = self._whole_segment_semantic_index(stmt)
        # Event presence: need at least one non-initial-step event
        non_init_events = [
            e for e in sem["events"]
            if any(
                ev.event_type in (EventType.CROSS, EventType.TIMER)
                for ev in e["event_exprs"]
            )
        ]
        if not non_init_events:
            return None
        # Transition outputs: at least one contribution outside events that
        # uses transition()
        outside_contribs = [c for c in sem["contributions"] if c["event"] is None]
        transition_contribs = []
        for c in outside_contribs:
            has_trans = False

            def _scan(e):
                nonlocal has_trans
                if isinstance(e, FunctionCall) and e.name == "transition":
                    has_trans = True
                elif isinstance(e, BinaryExpr):
                    _scan(e.left)
                    _scan(e.right)
                elif isinstance(e, UnaryExpr):
                    _scan(e.operand)
                elif isinstance(e, TernaryExpr):
                    _scan(e.cond)
                    _scan(e.true_expr)
                    _scan(e.false_expr)
                elif isinstance(e, FunctionCall):
                    for a in e.args:
                        _scan(a)

            _scan(c["stmt"].expr)
            if has_trans:
                transition_contribs.append(c)
        if not transition_contribs:
            return None
        # Event-body purity: walk each event body, ensure no Contribution / no
        # For / no While / no Case / no system task / no array-target Assignment
        # and no dynamic bus access.
        rejected = [False]

        def _body_visit(node):
            if rejected[0] or node is None:
                return
            if isinstance(node, Contribution):
                rejected[0] = True
                return
            if isinstance(node, (ForStatement, WhileStatement, CaseStatement)):
                rejected[0] = True
                return
            if isinstance(node, Assignment):
                # Only scalar identifier assignments
                if not isinstance(node.target, Identifier):
                    rejected[0] = True
                    return
                # Scan RHS for system tasks / dynamic bus
                def _expr_scan(e):
                    if rejected[0] or e is None:
                        return
                    if isinstance(e, FunctionCall) and e.name.startswith("$"):
                        rejected[0] = True
                        return
                    if isinstance(e, BranchAccess):
                        if (e.node1_index is not None or e.node1_index2 is not None
                            or e.node2_index is not None or e.node2_index2 is not None):
                            rejected[0] = True
                            return
                    if isinstance(e, BinaryExpr):
                        _expr_scan(e.left)
                        _expr_scan(e.right)
                    elif isinstance(e, UnaryExpr):
                        _expr_scan(e.operand)
                    elif isinstance(e, TernaryExpr):
                        _expr_scan(e.cond)
                        _expr_scan(e.true_expr)
                        _expr_scan(e.false_expr)
                    elif isinstance(e, FunctionCall):
                        for a in e.args:
                            _expr_scan(a)

                _expr_scan(node.value)
                return
            if isinstance(node, IfStatement):
                _body_visit(node.then_body)
                _body_visit(node.else_body)
                return
            if isinstance(node, Block):
                for s in node.statements:
                    _body_visit(s)
                return
            # Unknown node types — reject conservatively.
            rejected[0] = True
        for entry in non_init_events:
            _body_visit(entry["body"])
            if rejected[0]:
                return None
        # State variables targeted by event-body assignments
        target_states: set = set()
        def _collect_targets(node):
            if node is None:
                return
            if isinstance(node, Assignment) and isinstance(node.target, Identifier):
                target_states.add(str(node.target.name))
            elif isinstance(node, IfStatement):
                _collect_targets(node.then_body)
                _collect_targets(node.else_body)
            elif isinstance(node, Block):
                for s in node.statements:
                    _collect_targets(s)
        for entry in non_init_events:
            _collect_targets(entry["body"])
        # Reject if event bodies write to no scalar state (probably empty
        # event body that we shouldn't claim a fastpath for).
        if not target_states:
            return None
        return (
            "generic_event_state_transition_v1",
            str(self.module.name),
            int(len(non_init_events)),
            int(len(transition_contribs)),
            tuple(sorted(target_states)),
            tuple(c["stmt"].branch.node1 for c in transition_contribs),
        )

    def _whole_segment_has_ports(self, *names: str) -> bool:
        ports = set(getattr(self.module, "ports", ()) or ())
        return all(name in ports for name in names)

    def _whole_segment_has_states(self, *names: str) -> bool:
        states = set(self._state_scalar_slot_by_name)
        return all(name in states for name in names)

    def _whole_segment_has_params(self, *names: str) -> bool:
        params = {param.name for param in getattr(self.module, "parameters", ()) or ()}
        return all(name in params for name in names)

    def _whole_segment_array_width(self, name: str) -> Optional[int]:
        entry = self._state_array_range_by_name.get(name)
        if entry is None:
            return None
        lo, hi, _integer = entry
        return abs(int(hi) - int(lo)) + 1

    def _whole_segment_port_array_indices(self, name: str) -> Tuple[int, ...]:
        for decl in getattr(self.module, "port_decls", ()) or ():
            if decl.name != name or not decl.is_array:
                continue
            hi = decl.array_hi if decl.array_hi is not None else 0
            lo = decl.array_lo if decl.array_lo is not None else 0
            if hi >= lo:
                return tuple(range(int(hi), int(lo) - 1, -1))
            return tuple(range(int(hi), int(lo) + 1))
        return ()

    def _whole_segment_param_names(self) -> set[str]:
        return {param.name for param in getattr(self.module, "parameters", ()) or ()}

    def _whole_segment_port_names(self) -> set[str]:
        return set(getattr(self.module, "ports", ()) or ())

    def _whole_segment_port_direction(self, name: str) -> Optional[Direction]:
        for decl in getattr(self.module, "port_decls", ()) or ():
            if decl.name == name:
                return decl.direction
        return None

    def _whole_segment_output_ports(self) -> set[str]:
        return {
            decl.name
            for decl in getattr(self.module, "port_decls", ()) or ()
            if decl.direction == Direction.OUTPUT
        }

    def _whole_segment_input_ports(self) -> set[str]:
        return {
            decl.name
            for decl in getattr(self.module, "port_decls", ()) or ()
            if decl.direction == Direction.INPUT
        }

    def _whole_segment_inout_ports(self) -> set[str]:
        return {
            decl.name
            for decl in getattr(self.module, "port_decls", ()) or ()
            if decl.direction == Direction.INOUT
        }

    def _whole_segment_semantic_index(self, stmt) -> Dict[str, Any]:
        """Build a conservative semantic/dataflow summary for fastpath gating.

        This index is intentionally independent of benchmark names.  Names are
        still returned because the existing executors need them, but a candidate
        is allowed through only after the corresponding event, assignment, and
        output-contribution structure has been observed in the AST.
        """

        index: Dict[str, Any] = {
            "events": [],
            "contributions": [],
            "assignments": [],
            "conditions": [],
        }

        def visit(node, active_event=None) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child, active_event)
                return
            if isinstance(node, EventStatement):
                index["events"].append(
                    {
                        "event": node.event,
                        "body": node.body,
                        "event_exprs": self._whole_segment_event_exprs(node.event),
                    }
                )
                visit(node.body, node.event)
                return
            if isinstance(node, Contribution):
                index["contributions"].append({"stmt": node, "event": active_event})
                return
            if isinstance(node, Assignment):
                index["assignments"].append(
                    {
                        "stmt": node,
                        "target": self._whole_segment_assignment_target_name(node.target),
                        "target_base": self._whole_segment_assignment_target_base(node.target),
                        "value": node.value,
                        "event": active_event,
                    }
                )
                return
            if isinstance(node, IfStatement):
                index["conditions"].append({"expr": node.cond, "event": active_event})
                visit(node.then_body, active_event)
                visit(node.else_body, active_event)
                return
            if isinstance(node, ForStatement):
                visit(node.init, active_event)
                index["conditions"].append({"expr": node.cond, "event": active_event})
                visit(node.body, active_event)
                visit(node.update, active_event)
                return
            if isinstance(node, WhileStatement):
                index["conditions"].append({"expr": node.cond, "event": active_event})
                visit(node.body, active_event)
                return
            if isinstance(node, CaseStatement):
                index["conditions"].append({"expr": node.expr, "event": active_event})
                for item in node.items:
                    for value in item.values:
                        index["conditions"].append({"expr": value, "event": active_event})
                    visit(item.body, active_event)
                return

        visit(stmt)
        return index

    def _whole_segment_assignment_target_name(self, target) -> Optional[str]:
        if isinstance(target, Identifier):
            return str(target.name)
        if isinstance(target, ArrayAccess):
            index = self._whole_segment_int_literal(target.index)
            if index is not None:
                return f"{target.name}[{index}]"
            return str(target.name)
        return None

    def _whole_segment_assignment_target_base(self, target) -> Optional[str]:
        if isinstance(target, Identifier):
            return str(target.name)
        if isinstance(target, ArrayAccess):
            return str(target.name)
        return None

    def _whole_segment_event_exprs(self, event) -> Tuple[EventExpr, ...]:
        if isinstance(event, EventExpr):
            return (event,)
        if isinstance(event, CombinedEvent):
            return tuple(e for e in event.events if isinstance(e, EventExpr))
        return ()

    def _whole_segment_event_entries(
        self,
        sem: Dict[str, Any],
        event_type: EventType,
    ) -> List[Dict[str, Any]]:
        return [
            entry
            for entry in sem["events"]
            if any(event.event_type == event_type for event in entry["event_exprs"])
        ]

    def _whole_segment_expr_identifiers(self, expr) -> set[str]:
        names: set[str] = set()
        if expr is None:
            return names
        if isinstance(expr, Identifier):
            names.add(str(expr.name))
            return names
        if isinstance(expr, ArrayAccess):
            names.add(str(expr.name))
            names.update(self._whole_segment_expr_identifiers(expr.index))
            return names
        if isinstance(expr, BinaryExpr):
            names.update(self._whole_segment_expr_identifiers(expr.left))
            names.update(self._whole_segment_expr_identifiers(expr.right))
            return names
        if isinstance(expr, UnaryExpr):
            names.update(self._whole_segment_expr_identifiers(expr.operand))
            return names
        if isinstance(expr, TernaryExpr):
            names.update(self._whole_segment_expr_identifiers(expr.cond))
            names.update(self._whole_segment_expr_identifiers(expr.true_expr))
            names.update(self._whole_segment_expr_identifiers(expr.false_expr))
            return names
        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                names.update(self._whole_segment_expr_identifiers(arg))
            return names
        if isinstance(expr, BranchAccess):
            if expr.node1_index is not None:
                names.update(self._whole_segment_expr_identifiers(expr.node1_index))
            if expr.node1_index2 is not None:
                names.update(self._whole_segment_expr_identifiers(expr.node1_index2))
            if expr.node2_index is not None:
                names.update(self._whole_segment_expr_identifiers(expr.node2_index))
            if expr.node2_index2 is not None:
                names.update(self._whole_segment_expr_identifiers(expr.node2_index2))
            return names
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                names.update(self._whole_segment_expr_identifiers(arg))
        return names

    def _whole_segment_expr_param_identifiers(self, expr) -> set[str]:
        return self._whole_segment_expr_identifiers(expr) & self._whole_segment_param_names()

    def _whole_segment_branch_accesses(self, expr) -> List[BranchAccess]:
        branches: List[BranchAccess] = []
        if expr is None:
            return branches
        if isinstance(expr, BranchAccess):
            branches.append(expr)
            for index_expr in (
                expr.node1_index,
                expr.node1_index2,
                expr.node2_index,
                expr.node2_index2,
            ):
                branches.extend(self._whole_segment_branch_accesses(index_expr))
            return branches
        if isinstance(expr, BinaryExpr):
            branches.extend(self._whole_segment_branch_accesses(expr.left))
            branches.extend(self._whole_segment_branch_accesses(expr.right))
            return branches
        if isinstance(expr, UnaryExpr):
            branches.extend(self._whole_segment_branch_accesses(expr.operand))
            return branches
        if isinstance(expr, TernaryExpr):
            branches.extend(self._whole_segment_branch_accesses(expr.cond))
            branches.extend(self._whole_segment_branch_accesses(expr.true_expr))
            branches.extend(self._whole_segment_branch_accesses(expr.false_expr))
            return branches
        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                branches.extend(self._whole_segment_branch_accesses(arg))
            return branches
        if isinstance(expr, ArrayAccess):
            branches.extend(self._whole_segment_branch_accesses(expr.index))
            return branches
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                branches.extend(self._whole_segment_branch_accesses(arg))
        return branches

    def _whole_segment_branch_node_name(self, branch: BranchAccess) -> Optional[str]:
        if (
            branch.access_type != "V"
            or branch.node1_index2 is not None
            or branch.node2_index is not None
            or branch.node2_index2 is not None
        ):
            return None
        if branch.node1_index is None:
            return str(branch.node1)
        index = self._whole_segment_int_literal(branch.node1_index)
        if index is None:
            return None
        return f"{branch.node1}[{index}]"

    def _whole_segment_transition_call(self, expr) -> Optional[FunctionCall]:
        affine = self._transition_affine_expr(expr)
        if affine is None:
            return None
        call, _offset_expr, _scale_expr = affine
        return call

    def _whole_segment_transition_contributions(
        self,
        sem: Dict[str, Any],
        *,
        outside_events_only: bool = True,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for item in sem["contributions"]:
            if outside_events_only and item["event"] is not None:
                continue
            stmt = item["stmt"]
            call = self._whole_segment_transition_call(stmt.expr)
            if call is None or not call.args:
                continue
            entries.append(
                {
                    "stmt": stmt,
                    "call": call,
                    "target": call.args[0],
                    "target_ids": self._whole_segment_expr_identifiers(call.args[0]),
                    "branch_node": self._whole_segment_branch_node_name(stmt.branch),
                    "branch_ref": stmt.branch.node2,
                }
            )
        return entries

    def _whole_segment_source_state_for_transition_target(self, expr) -> Optional[str]:
        if isinstance(expr, Identifier):
            return str(expr.name)
        if isinstance(expr, ArrayAccess):
            return self._whole_segment_assignment_target_name(expr)
        if isinstance(expr, TernaryExpr):
            name = self._whole_segment_ternary_bit_name(expr.cond)
            if name is not None:
                return name
        return None

    def _whole_segment_transition_source_node_by_state(
        self,
        sem: Dict[str, Any],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in self._whole_segment_transition_contributions(sem):
            node = entry["branch_node"]
            state = self._whole_segment_source_state_for_transition_target(entry["target"])
            if node is not None and state is not None:
                mapping.setdefault(state, node)
        return mapping

    def _whole_segment_conditions(self, stmt) -> List[Expr]:
        conds: List[Expr] = []

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                visit(node.body)
                return
            if isinstance(node, IfStatement):
                conds.append(node.cond)
                visit(node.then_body)
                visit(node.else_body)
                return
            if isinstance(node, ForStatement):
                visit(node.init)
                conds.append(node.cond)
                visit(node.body)
                visit(node.update)
                return
            if isinstance(node, WhileStatement):
                conds.append(node.cond)
                visit(node.body)
                return
            if isinstance(node, CaseStatement):
                conds.append(node.expr)
                for item in node.items:
                    conds.extend(item.values)
                    visit(item.body)

        visit(stmt)
        return conds

    def _whole_segment_assignments(self, stmt) -> List[Assignment]:
        assignments: List[Assignment] = []

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Assignment):
                assignments.append(node)
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                visit(node.body)
                return
            if isinstance(node, IfStatement):
                visit(node.then_body)
                visit(node.else_body)
                return
            if isinstance(node, ForStatement):
                visit(node.init)
                visit(node.body)
                visit(node.update)
                return
            if isinstance(node, WhileStatement):
                visit(node.body)
                return
            if isinstance(node, CaseStatement):
                for item in node.items:
                    visit(item.body)

        visit(stmt)
        return assignments

    def _whole_segment_assignment_number(
        self,
        stmt,
        expected: float,
    ) -> set[str]:
        targets: set[str] = set()
        for assignment in self._whole_segment_assignments(stmt):
            name = self._whole_segment_assignment_target_name(assignment.target)
            if name is not None and self._whole_segment_expr_is_number(assignment.value, expected):
                targets.add(name)
        return targets

    def _whole_segment_assignment_value_ids(self, stmt, target_name: str) -> set[str]:
        ids: set[str] = set()
        for assignment in self._whole_segment_assignments(stmt):
            if self._whole_segment_assignment_target_name(assignment.target) == target_name:
                ids.update(self._whole_segment_expr_identifiers(assignment.value))
        return ids

    def _whole_segment_first_identifier_param(self, expr) -> Optional[str]:
        params = sorted(self._whole_segment_expr_param_identifiers(expr))
        return params[0] if params else None

    def _collect_gain_timer_reduction_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        timer_entries = [
            entry
            for entry in self._whole_segment_event_entries(sem, EventType.TIMER)
            for event in entry["event_exprs"]
            if event.event_type == EventType.TIMER and len(event.args) == 2
        ]
        if len(timer_entries) != 1:
            return None
        timer_entry = timer_entries[0]
        timer_event = next(
            event for event in timer_entry["event_exprs"]
            if event.event_type == EventType.TIMER and len(event.args) == 2
        )
        sample_period = self._whole_segment_identifier_name(timer_event.args[1])
        if sample_period is None or sample_period not in self._whole_segment_param_names():
            return None

        diff_sources: Dict[str, Tuple[str, str]] = {}
        update_source_by_state: Dict[str, str] = {}
        span_sources: Dict[str, str] = {}
        for assignment in self._whole_segment_assignments(timer_entry["body"]):
            target = self._whole_segment_assignment_target_name(assignment.target)
            if target is None:
                continue
            if (
                isinstance(assignment.value, BranchAccess)
                and assignment.value.access_type == "V"
                and assignment.value.node2 is not None
                and assignment.value.node1_index is None
                and assignment.value.node2_index is None
            ):
                diff_sources[target] = (str(assignment.value.node1), str(assignment.value.node2))
                continue
            if isinstance(assignment.value, Identifier):
                update_source_by_state[target] = str(assignment.value.name)
                continue
            if (
                isinstance(assignment.value, BinaryExpr)
                and assignment.value.op == "-"
                and isinstance(assignment.value.left, Identifier)
                and isinstance(assignment.value.right, Identifier)
            ):
                left_src = update_source_by_state.get(str(assignment.value.left.name))
                right_src = update_source_by_state.get(str(assignment.value.right.name))
                if left_src is not None and left_src == right_src:
                    span_sources[target] = left_src

        gain_state = None
        numerator_span = None
        denominator_span = None
        for assignment in self._whole_segment_assignments(timer_entry["body"]):
            target = self._whole_segment_assignment_target_name(assignment.target)
            value = assignment.value
            if (
                target is not None
                and isinstance(value, BinaryExpr)
                and value.op == "/"
                and isinstance(value.left, Identifier)
                and isinstance(value.right, Identifier)
            ):
                left = str(value.left.name)
                right = str(value.right.name)
                if left in span_sources and right in span_sources:
                    gain_state = target
                    numerator_span = left
                    denominator_span = right
                    break
        if gain_state is None or numerator_span is None or denominator_span is None:
            return None
        input_pair = diff_sources.get(span_sources.get(denominator_span, ""))
        output_pair = diff_sources.get(span_sources.get(numerator_span, ""))
        if input_pair is None or output_pair is None:
            return None

        start_time = None
        min_input_span = None
        for cond in self._whole_segment_conditions(timer_entry["body"]):
            if (
                isinstance(cond, BinaryExpr)
                and cond.op in {">=", ">"}
                and isinstance(cond.left, Identifier)
                and str(cond.left.name) == "$abstime"
                and isinstance(cond.right, Identifier)
            ):
                start_time = str(cond.right.name)
            if (
                isinstance(cond, BinaryExpr)
                and cond.op in {">", ">="}
                and isinstance(cond.left, Identifier)
                and str(cond.left.name) == denominator_span
                and isinstance(cond.right, Identifier)
            ):
                min_input_span = str(cond.right.name)
        if (
            start_time not in self._whole_segment_param_names()
            or min_input_span not in self._whole_segment_param_names()
        ):
            return None

        valid_states = self._whole_segment_assignment_number(timer_entry["body"], 1.0)
        transition_entries = self._whole_segment_transition_contributions(sem)
        gain_entry = next(
            (entry for entry in transition_entries if gain_state in entry["target_ids"]),
            None,
        )
        valid_entry = next(
            (
                entry for entry in transition_entries
                if entry is not gain_entry and entry["target_ids"] & valid_states
            ),
            None,
        )
        if gain_entry is None or valid_entry is None:
            return None
        gain_out = gain_entry["branch_node"]
        valid_out = valid_entry["branch_node"]
        if gain_out is None or valid_out is None:
            return None

        gain_target = gain_entry["target"]
        gain_scale_candidates = self._whole_segment_expr_param_identifiers(gain_target) - {
            sample_period,
            start_time,
            min_input_span,
        }
        gain_scale = next(iter(sorted(gain_scale_candidates)), None)
        tedge = None
        if len(gain_entry["call"].args) > 2:
            tedge = self._whole_segment_identifier_name(gain_entry["call"].args[2])
        supply_branches = [
            branch
            for branch in self._whole_segment_branch_accesses(gain_target)
            if branch.access_type == "V" and branch.node2 is not None
        ]
        if not supply_branches:
            supply_branches = [
                branch
                for branch in self._whole_segment_branch_accesses(valid_entry["target"])
                if branch.access_type == "V" and branch.node2 is not None
            ]
        if not supply_branches:
            return None
        vdd_port = str(supply_branches[0].node1)
        vss_port = str(supply_branches[0].node2)
        if gain_scale not in self._whole_segment_param_names() or tedge not in self._whole_segment_param_names():
            return None
        return (
            "gain_timer_reduction_v1",
            vdd_port, vss_port, input_pair[0], input_pair[1], output_pair[0], output_pair[1],
            gain_out, valid_out,
            sample_period, start_time, gain_scale, min_input_span, tedge,
        )

    def _collect_cmp_delay_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        cross_roles: List[Tuple[int, str, Expr, Any]] = []
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type != EventType.CROSS or not event.args:
                    continue
                role = self._whole_segment_branch_minus_threshold(event.args[0])
                if role is None:
                    continue
                direction = event.direction if event.direction is not None else 0
                cross_roles.append((int(direction), role[0], role[1], entry["body"]))
        rising = next((role for role in cross_roles if role[0] == 1), None)
        falling = next((role for role in cross_roles if role[0] == -1), None)
        if rising is None or falling is None or rising[1] != falling[1]:
            return None
        clk_port = rising[1]
        rising_body = rising[3]

        diff_role = self._whole_segment_first_two_single_ended_branches(rising_body)
        if diff_role is None:
            return None
        vinp_port, vinn_port = diff_role

        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        decision_states = self._whole_segment_positive_negative_decision_states(
            rising_body,
            vinp_port,
            vinn_port,
        )
        if decision_states is None:
            return None
        pos_state, neg_state = decision_states
        outp_port = state_to_node.get(pos_state)
        outn_port = state_to_node.get(neg_state)
        if outp_port is None or outn_port is None:
            return None

        vdd_port = self._whole_segment_assigned_voltage_source_port(stmt, "vdd")
        vss_candidates = sorted(self._whole_segment_inout_ports() - ({vdd_port} if vdd_port else set()))
        vss_port = vss_candidates[0] if vss_candidates else None
        if vdd_port is None or vss_port is None:
            return None

        voffset = self._whole_segment_diff_offset_param(rising_body, vinp_port, vinn_port)
        delay_params = self._whole_segment_delay_logn_params(rising_body)
        clamp_params = self._whole_segment_clamp_params(rising_body)
        tedge = self._whole_segment_common_transition_arg_param(sem, 2)
        if (
            voffset is None
            or delay_params is None
            or clamp_params is None
            or tedge is None
        ):
            return None
        tau, td_0 = delay_params
        td_min, td_max = clamp_params
        return (
            "cmp_delay_log_transition_v1",
            clk_port, vinn_port, vinp_port, outn_port, outp_port, vss_port, vdd_port,
            voffset, tau, td_0, td_min, td_max, tedge,
        )

    def _collect_edge_interval_timer_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        rising: List[Tuple[str, Expr, Any]] = []
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type != EventType.CROSS or int(event.direction or 0) != 1 or not event.args:
                    continue
                role = self._whole_segment_branch_minus_threshold(event.args[0])
                if role is not None:
                    rising.append((role[0], role[1], entry["body"]))
        if len(rising) < 2:
            return None
        threshold = self._whole_segment_identifier_name(rising[0][1])
        if threshold is None or threshold not in self._whole_segment_param_names():
            return None
        first_clk = None
        second_clk = None
        diff_state = None
        start_state = None
        for node, expr, body in rising:
            if not self._whole_segment_same_expr(expr, rising[0][1]):
                continue
            assigns = self._whole_segment_assignments(body)
            if any(
                self._whole_segment_assignment_target_name(a.target) is not None
                and isinstance(a.value, Identifier)
                and str(a.value.name) == "$abstime"
                for a in assigns
            ):
                first_clk = node
                start_state = next(
                    self._whole_segment_assignment_target_name(a.target)
                    for a in assigns
                    if isinstance(a.value, Identifier) and str(a.value.name) == "$abstime"
                )
            for assignment in assigns:
                target = self._whole_segment_assignment_target_name(assignment.target)
                ids = self._whole_segment_expr_identifiers(assignment.value)
                if target is not None and "$abstime" in ids and start_state is not None and start_state in ids:
                    second_clk = node
                    diff_state = target
        if first_clk is None or second_clk is None or diff_state is None:
            return None
        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        out_port = state_to_node.get(diff_state)
        if out_port is None:
            return None
        return ("edge_interval_timer_v1", first_clk, second_clk, out_port, threshold)

    def _collect_weighted_sar_adc_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        crosses: List[Tuple[int, str, Expr, Any]] = []
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type != EventType.CROSS or not event.args:
                    continue
                role = self._whole_segment_branch_minus_threshold(event.args[0])
                if role is not None:
                    crosses.append((int(event.direction or 0), role[0], role[1], entry["body"]))
        rising = next((item for item in crosses if item[0] == 1), None)
        falling = next((item for item in crosses if item[0] == -1), None)
        if rising is None or falling is None or rising[1] != falling[1]:
            return None
        clks_port = rising[1]
        vth = self._whole_segment_identifier_name(rising[2])
        if vth is None or vth not in self._whole_segment_param_names():
            return None

        vin_port = None
        sample_state = None
        for assignment in self._whole_segment_assignments(falling[3]):
            target = self._whole_segment_assignment_target_name(assignment.target)
            value = assignment.value
            if (
                target is not None
                and isinstance(value, BranchAccess)
                and value.access_type == "V"
                and value.node1_index is None
            ):
                vin_port = str(value.node1)
                sample_state = target
                break
        if vin_port is None or sample_state is None:
            return None

        rst_port = None
        for cond in (*self._whole_segment_conditions(rising[3]), *self._whole_segment_conditions(falling[3])):
            role = self._whole_segment_voltage_threshold_condition(cond, {"<", "<=", ">", ">="})
            if role is not None and role[0] not in {clks_port, vin_port}:
                rst_port = role[0]
                break
        if rst_port is None:
            return None

        transition_entries = self._whole_segment_transition_contributions(sem)
        dout_entries = [
            entry
            for entry in transition_entries
            if entry["branch_node"] is not None
            and "[" in entry["branch_node"]
            and self._whole_segment_source_state_for_transition_target(entry["target"]) is not None
        ]
        if not dout_entries:
            return None
        dout_base_counts: Dict[str, int] = {}
        for entry in dout_entries:
            base = entry["branch_node"].split("[", 1)[0]
            dout_base_counts[base] = dout_base_counts.get(base, 0) + 1
        dout_base = max(dout_base_counts, key=dout_base_counts.get)
        dout_ports = tuple(
            sorted(
                (entry["branch_node"] for entry in dout_entries if entry["branch_node"].startswith(f"{dout_base}[")),
                key=self._whole_segment_bus_node_sort_key,
                reverse=True,
            )
        )
        width = len(dout_ports)
        if width <= 0:
            return None

        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        bit_index_port = state_to_node.get("bit_index_v")
        trial_code_port = state_to_node.get("trial_code_v")
        trial_vdac_port = state_to_node.get("trial_vdac_state")
        cmp_decision_port = state_to_node.get("cmp_decision_v")
        conv_done_port = state_to_node.get("conv_done_v")
        vin_sample_port = state_to_node.get(sample_state)
        if any(
            port is None
            for port in (
                bit_index_port,
                trial_code_port,
                trial_vdac_port,
                cmp_decision_port,
                conv_done_port,
                vin_sample_port,
            )
        ):
            return None
        vdd = self._whole_segment_role_param_referenced(stmt, "vdd")
        if vdd is None:
            return None
        return (
            "weighted_sar_adc_v1",
            vin_port, clks_port, rst_port,
            dout_ports,
            bit_index_port, trial_code_port, trial_vdac_port,
            cmp_decision_port, conv_done_port, vin_sample_port,
            vdd, vth, width,
        )

    def _collect_weighted_dac_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        cross_nodes: set[str] = set()
        threshold = None
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type != EventType.CROSS or not event.args:
                    continue
                role = self._whole_segment_branch_minus_threshold(event.args[0])
                if role is None:
                    continue
                cross_nodes.add(role[0])
                local_threshold = self._whole_segment_identifier_name(role[1])
                threshold = threshold or local_threshold
                if local_threshold != threshold:
                    return None
        if len(cross_nodes) < 2 or threshold not in self._whole_segment_param_names():
            return None
        weighted_nodes = self._whole_segment_weighted_accumulator_nodes(
            stmt,
            cross_nodes,
        )
        if len(weighted_nodes) < 3:
            return None
        transition_entries = self._whole_segment_transition_contributions(sem)
        if len(transition_entries) != 1:
            return None
        output_port = transition_entries[0]["branch_node"]
        output_state = self._whole_segment_source_state_for_transition_target(transition_entries[0]["target"])
        if output_port is None or output_state is None:
            return None
        assigned_outputs = {
            self._whole_segment_assignment_target_name(assignment.target)
            for assignment in self._whole_segment_assignments(stmt)
        }
        if output_state not in assigned_outputs:
            return None
        vdd = self._whole_segment_role_param_referenced(stmt, "vdd")
        if vdd is None:
            return None
        din_ports = tuple(sorted(weighted_nodes, key=self._whole_segment_trailing_int_sort_key, reverse=True))
        return ("weighted_dac_v1", din_ports, output_port, vdd, threshold)

    def _whole_segment_weighted_accumulator_nodes(
        self,
        stmt,
        cross_nodes: set[str],
    ) -> set[str]:
        nodes: set[str] = set()

        def self_accum_targets(node) -> set[str]:
            targets: set[str] = set()
            for assignment in self._whole_segment_assignments(node):
                target = self._whole_segment_assignment_target_name(assignment.target)
                if target is None:
                    continue
                value = assignment.value
                if (
                    isinstance(value, BinaryExpr)
                    and value.op in {"+", "-"}
                    and target in self._whole_segment_expr_identifiers(value)
                ):
                    targets.add(target)
            return targets

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                visit(node.body)
                return
            if isinstance(node, IfStatement):
                role = self._whole_segment_voltage_threshold_condition(
                    node.cond,
                    {">", ">="},
                )
                if role is not None and role[0] in cross_nodes:
                    if self_accum_targets(node.then_body):
                        nodes.add(role[0])
                visit(node.then_body)
                visit(node.else_body)
                return
            if isinstance(node, ForStatement):
                visit(node.body)
                return
            if isinstance(node, WhileStatement):
                visit(node.body)
                return
            if isinstance(node, CaseStatement):
                for item in node.items:
                    visit(item.body)

        visit(stmt)
        return nodes

    def _collect_sample_hold_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        rising = None
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type == EventType.CROSS and int(event.direction or 0) == 1 and event.args:
                    role = self._whole_segment_branch_minus_threshold(event.args[0])
                    if role is not None:
                        rising = (role[0], role[1], entry["body"])
                        break
            if rising is not None:
                break
        if rising is None:
            return None
        clk_port, threshold_expr, body = rising
        hold_state = None
        vin_port = None
        for assignment in self._whole_segment_assignments(body):
            target = self._whole_segment_assignment_target_name(assignment.target)
            if (
                target is not None
                and isinstance(assignment.value, BranchAccess)
                and assignment.value.access_type == "V"
            ):
                hold_state = target
                vin_port = str(assignment.value.node1)
                break
        if hold_state is None or vin_port is None:
            return None
        rst_port = None
        for cond in self._whole_segment_conditions(body):
            role = self._whole_segment_voltage_threshold_condition(cond, {">", ">="})
            if role is not None and role[0] not in {clk_port, vin_port}:
                rst_port = role[0]
        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        vout_port = state_to_node.get(hold_state)
        if rst_port is None or vout_port is None:
            return None
        vdd_port, vss_port = self._whole_segment_supply_ports_from_threshold_assignment(stmt, threshold_expr)
        if vdd_port is None or vss_port is None:
            return None
        tr = self._whole_segment_common_transition_arg_param(sem, 2)
        if tr is None:
            return None
        return ("sample_hold_rising_v1", vin_port, clk_port, vdd_port, vss_port, rst_port, vout_port, tr)

    def _collect_ref_step_clock_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        timer_entries = [
            entry
            for entry in self._whole_segment_event_entries(sem, EventType.TIMER)
            for event in entry["event_exprs"]
            if event.event_type == EventType.TIMER and len(event.args) == 1
        ]
        if len(timer_entries) != 1:
            return None
        timer_entry = timer_entries[0]
        toggle_state = self._whole_segment_toggle_state(timer_entry["body"])
        if toggle_state is None:
            return None
        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        clk_port = state_to_node.get(toggle_state)
        if clk_port is None:
            return None
        vdd_port, vss_port = self._whole_segment_supply_ports_from_transition(stmt, toggle_state)
        if vdd_port is None or vss_port is None:
            return None
        required = self._whole_segment_required_params_referenced(
            stmt,
            ("period_pre", "period_post", "t_switch", "tedge"),
        )
        if required is None:
            return None
        period_pre, period_post, t_switch, tedge = required
        return (
            "ref_step_clock_v1",
            vdd_port, vss_port, clk_port,
            period_pre, period_post, t_switch, tedge,
        )

    def _collect_cppll_timer_candidate(self, stmt) -> Optional[tuple]:
        sem = self._whole_segment_semantic_index(stmt)
        timer_entries = [
            entry
            for entry in self._whole_segment_event_entries(sem, EventType.TIMER)
            for event in entry["event_exprs"]
            if event.event_type == EventType.TIMER and len(event.args) == 1
        ]
        if len(timer_entries) != 1:
            return None
        dco_state = self._whole_segment_toggle_state(timer_entries[0]["body"])
        if dco_state is None:
            return None
        state_to_node = self._whole_segment_transition_source_node_by_state(sem)
        dco_port = state_to_node.get(dco_state)
        lock_port = state_to_node.get("lock_state")
        fb_port = state_to_node.get("fb_state")
        vctrl_port = self._whole_segment_direct_state_output_node(sem, "vctrl")
        if any(port is None for port in (dco_port, fb_port, vctrl_port, lock_port)):
            return None
        ref_port = None
        fb_cross_port = None
        for entry in self._whole_segment_event_entries(sem, EventType.CROSS):
            for event in entry["event_exprs"]:
                if event.event_type != EventType.CROSS or int(event.direction or 0) != 1 or not event.args:
                    continue
                role = self._whole_segment_branch_minus_threshold(event.args[0])
                if role is None:
                    continue
                assigned = {
                    self._whole_segment_assignment_target_name(a.target)
                    for a in self._whole_segment_assignments(entry["body"])
                }
                if "vctrl" in assigned or "lock_state" in assigned:
                    ref_port = role[0]
                if "t_fb_last" in assigned:
                    fb_cross_port = role[0]
        if ref_port is None or fb_cross_port != fb_port:
            return None
        vdd_port, vss_port = self._whole_segment_supply_ports_from_transition(stmt, dco_state)
        if vdd_port is None or vss_port is None:
            return None
        required = self._whole_segment_required_params_referenced(
            stmt,
            (
                "div_ratio", "f_center", "kvco_hz_per_v", "f_min", "f_max",
                "kp", "ki", "integ_min", "integ_max", "vctrl_init",
                "tedge", "lock_tol", "lock_count_target",
            ),
        )
        if required is None:
            return None
        return (
            "cppll_timer_v1",
            vdd_port, vss_port, ref_port, fb_port, dco_port, vctrl_port, lock_port,
            *required,
        )

    def _whole_segment_first_two_single_ended_branches(self, stmt) -> Optional[Tuple[str, str]]:
        exprs: List[Expr] = []
        for assignment in self._whole_segment_assignments(stmt):
            exprs.append(assignment.value)
        exprs.extend(self._whole_segment_conditions(stmt))
        for expr in exprs:
            branches = [
                branch
                for branch in self._whole_segment_branch_accesses(expr)
                if (
                    branch.access_type == "V"
                    and branch.node2 is None
                    and branch.node1_index is None
                    and branch.node1_index2 is None
                )
            ]
            for left, right in zip(branches, branches[1:]):
                if left.node1 != right.node1:
                    return str(left.node1), str(right.node1)
        return None

    def _whole_segment_positive_negative_decision_states(
        self,
        stmt,
        pos_node: str,
        neg_node: str,
    ) -> Optional[Tuple[str, str]]:
        def visit(node) -> Optional[Tuple[str, str]]:
            if node is None:
                return None
            if isinstance(node, Block):
                for child in node.statements:
                    result = visit(child)
                    if result is not None:
                        return result
                return None
            if isinstance(node, IfStatement):
                branch_nodes = {
                    str(branch.node1)
                    for branch in self._whole_segment_branch_accesses(node.cond)
                    if branch.access_type == "V"
                }
                if {pos_node, neg_node}.issubset(branch_nodes):
                    then_ones = self._whole_segment_assignment_number(node.then_body, 1.0)
                    else_ones = self._whole_segment_assignment_number(node.else_body, 1.0)
                    then_zeros = self._whole_segment_assignment_number(node.then_body, 0.0)
                    else_zeros = self._whole_segment_assignment_number(node.else_body, 0.0)
                    pos_states = sorted(then_ones & else_zeros)
                    neg_states = sorted(else_ones & then_zeros)
                    if pos_states and neg_states:
                        return pos_states[0], neg_states[0]
                return visit(node.then_body) or visit(node.else_body)
            if isinstance(node, EventStatement):
                return visit(node.body)
            if isinstance(node, ForStatement):
                return visit(node.body)
            if isinstance(node, WhileStatement):
                return visit(node.body)
            if isinstance(node, CaseStatement):
                for item in node.items:
                    result = visit(item.body)
                    if result is not None:
                        return result
            return None

        return visit(stmt)

    def _whole_segment_assigned_voltage_source_port(
        self,
        stmt,
        state_name: str,
    ) -> Optional[str]:
        for assignment in self._whole_segment_assignments(stmt):
            if self._whole_segment_assignment_target_name(assignment.target) != state_name:
                continue
            value = assignment.value
            if (
                isinstance(value, BranchAccess)
                and value.access_type == "V"
                and value.node2 is None
                and value.node1_index is None
            ):
                return str(value.node1)
        return None

    def _whole_segment_diff_offset_param(
        self,
        stmt,
        pos_node: str,
        neg_node: str,
    ) -> Optional[str]:
        params = self._whole_segment_param_names()
        exprs: List[Expr] = []
        for assignment in self._whole_segment_assignments(stmt):
            exprs.append(assignment.value)
        exprs.extend(self._whole_segment_conditions(stmt))
        for expr in exprs:
            branch_nodes = [
                str(branch.node1)
                for branch in self._whole_segment_branch_accesses(expr)
                if branch.access_type == "V"
            ]
            if pos_node in branch_nodes and neg_node in branch_nodes:
                candidates = sorted(self._whole_segment_expr_identifiers(expr) & params)
                if candidates:
                    return candidates[0]
        return None

    def _whole_segment_delay_logn_params(self, stmt) -> Optional[Tuple[str, str]]:
        params = self._whole_segment_param_names()

        def has_ln(expr) -> bool:
            if isinstance(expr, FunctionCall) and expr.name == "ln":
                return True
            return any(call.name == "ln" for call in self._iter_function_calls_in_expr(expr))

        def mul_ln_param(expr) -> Optional[str]:
            if not isinstance(expr, BinaryExpr):
                return None
            if expr.op == "*":
                if isinstance(expr.left, Identifier) and expr.left.name in params and has_ln(expr.right):
                    return str(expr.left.name)
                if isinstance(expr.right, Identifier) and expr.right.name in params and has_ln(expr.left):
                    return str(expr.right.name)
            return mul_ln_param(expr.left) or mul_ln_param(expr.right)

        for assignment in self._whole_segment_assignments(stmt):
            value = assignment.value
            if not has_ln(value):
                continue
            tau = mul_ln_param(value)
            local_params = sorted(self._whole_segment_expr_identifiers(value) & params)
            if tau is None:
                continue
            base_candidates = [name for name in local_params if name != tau]
            if base_candidates:
                return tau, base_candidates[0]
        return None

    def _whole_segment_clamp_params(self, stmt) -> Optional[Tuple[str, str]]:
        lower = None
        upper = None
        params = self._whole_segment_param_names()
        for cond in self._whole_segment_conditions(stmt):
            if not (
                isinstance(cond, BinaryExpr)
                and isinstance(cond.left, Identifier)
                and isinstance(cond.right, Identifier)
                and cond.right.name in params
            ):
                continue
            if cond.op in {"<", "<="}:
                lower = str(cond.right.name)
            elif cond.op in {">", ">="}:
                upper = str(cond.right.name)
        if lower is None or upper is None:
            return None
        return lower, upper

    def _whole_segment_common_transition_arg_param(
        self,
        sem: Dict[str, Any],
        arg_index: int,
    ) -> Optional[str]:
        names: set[str] = set()
        params = self._whole_segment_param_names()
        for entry in self._whole_segment_transition_contributions(sem):
            call = entry["call"]
            if len(call.args) <= arg_index:
                continue
            name = self._whole_segment_identifier_name(call.args[arg_index])
            if name in params or name in self._state_scalar_slot_by_name:
                names.add(str(name))
        if len(names) == 1:
            return next(iter(names))
        return None

    def _whole_segment_bus_node_sort_key(self, name: str) -> int:
        if "[" not in name or not name.endswith("]"):
            return -1
        try:
            return int(name.rsplit("[", 1)[1][:-1])
        except ValueError:
            return -1

    def _whole_segment_trailing_int_sort_key(self, name: str) -> int:
        digits = []
        for ch in reversed(name):
            if not ch.isdigit():
                break
            digits.append(ch)
        if not digits:
            return -1
        return int("".join(reversed(digits)))

    def _whole_segment_semantic_expr_identifiers(self, stmt) -> set[str]:
        sem = self._whole_segment_semantic_index(stmt)
        names: set[str] = set()
        for entry in sem["assignments"]:
            names.update(self._whole_segment_expr_identifiers(entry["value"]))
        for entry in sem["conditions"]:
            names.update(self._whole_segment_expr_identifiers(entry["expr"]))
        for entry in sem["contributions"]:
            names.update(self._whole_segment_expr_identifiers(entry["stmt"].expr))
        for entry in sem["events"]:
            for event in entry["event_exprs"]:
                for arg in event.args:
                    names.update(self._whole_segment_expr_identifiers(arg))
        return names

    def _whole_segment_role_param_referenced(
        self,
        stmt,
        preferred_name: str,
    ) -> Optional[str]:
        if (
            preferred_name in self._whole_segment_param_names()
            and preferred_name in self._whole_segment_semantic_expr_identifiers(stmt)
        ):
            return preferred_name
        return None

    def _whole_segment_required_params_referenced(
        self,
        stmt,
        names: Tuple[str, ...],
    ) -> Optional[Tuple[str, ...]]:
        params = self._whole_segment_param_names()
        referenced = self._whole_segment_semantic_expr_identifiers(stmt)
        if all(name in params and name in referenced for name in names):
            return tuple(names)
        return None

    def _whole_segment_supply_ports_from_threshold_assignment(
        self,
        stmt,
        threshold_expr,
    ) -> Tuple[Optional[str], Optional[str]]:
        threshold_name = self._whole_segment_identifier_name(threshold_expr)
        if threshold_name is None:
            return None, None
        for assignment in self._whole_segment_assignments(stmt):
            if self._whole_segment_assignment_target_name(assignment.target) != threshold_name:
                continue
            branches = [
                branch
                for branch in self._whole_segment_branch_accesses(assignment.value)
                if branch.access_type == "V"
                and branch.node2 is None
                and branch.node1_index is None
            ]
            if len(branches) >= 2:
                return str(branches[0].node1), str(branches[1].node1)
        inouts = sorted(self._whole_segment_inout_ports())
        if len(inouts) >= 2:
            return inouts[0], inouts[1]
        return None, None

    def _whole_segment_supply_ports_from_transition(
        self,
        stmt,
        state_name: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        sem = self._whole_segment_semantic_index(stmt)
        assigned_voltage_ports: Dict[str, str] = {}
        for assignment in sem["assignments"]:
            target = assignment["target"]
            value = assignment["value"]
            if (
                target is not None
                and isinstance(value, BranchAccess)
                and value.access_type == "V"
                and value.node2 is None
                and value.node1_index is None
            ):
                assigned_voltage_ports[target] = str(value.node1)
        for entry in self._whole_segment_transition_contributions(sem):
            if self._whole_segment_source_state_for_transition_target(entry["target"]) != state_name:
                continue
            directed_ports = self._whole_segment_directed_supply_ports_from_transition_expr(
                entry["stmt"].expr,
                assigned_voltage_ports,
            )
            if directed_ports is not None:
                return directed_ports
            ids = [
                name
                for name in self._whole_segment_expr_identifiers(entry["stmt"].expr)
                if name in assigned_voltage_ports
            ]
            unique_ports = []
            for name in ids:
                port = assigned_voltage_ports[name]
                if port not in unique_ports:
                    unique_ports.append(port)
            if len(unique_ports) >= 2:
                return unique_ports[0], unique_ports[1]
            branches = [
                branch
                for branch in self._whole_segment_branch_accesses(entry["stmt"].expr)
                if branch.access_type == "V"
            ]
            if branches:
                first = branches[0]
                if first.node2 is not None:
                    return str(first.node1), str(first.node2)
                if len(branches) >= 2:
                    return str(branches[0].node1), str(branches[1].node1)
        inouts = sorted(self._whole_segment_inout_ports())
        if len(inouts) >= 2:
            return inouts[0], inouts[1]
        return None, None

    def _whole_segment_directed_supply_ports_from_transition_expr(
        self,
        expr,
        assigned_voltage_ports: Dict[str, str],
    ) -> Optional[Tuple[str, str]]:
        affine = self._transition_affine_expr(expr)
        if affine is None:
            return None
        _transition_call, _offset_expr, scale_expr = affine
        if not isinstance(scale_expr, BinaryExpr) or scale_expr.op != "-":
            return None
        high_port = self._whole_segment_voltage_port_from_expr(
            scale_expr.left,
            assigned_voltage_ports,
        )
        low_port = self._whole_segment_voltage_port_from_expr(
            scale_expr.right,
            assigned_voltage_ports,
        )
        if high_port is not None and low_port is not None and high_port != low_port:
            return high_port, low_port
        return None

    def _whole_segment_voltage_port_from_expr(
        self,
        expr,
        assigned_voltage_ports: Dict[str, str],
    ) -> Optional[str]:
        if isinstance(expr, Identifier):
            return assigned_voltage_ports.get(str(expr.name))
        if (
            isinstance(expr, BranchAccess)
            and expr.access_type == "V"
            and expr.node2 is None
            and expr.node1_index is None
        ):
            return str(expr.node1)
        return None

    def _whole_segment_toggle_state(self, stmt) -> Optional[str]:
        for assignment in self._whole_segment_assignments(stmt):
            target = self._whole_segment_assignment_target_name(assignment.target)
            value = assignment.value
            if (
                target is not None
                and isinstance(value, BinaryExpr)
                and value.op == "-"
                and self._whole_segment_expr_is_number(value.left, 1.0)
                and isinstance(value.right, Identifier)
                and str(value.right.name) == target
            ):
                return target
        return None

    def _whole_segment_direct_state_output_node(
        self,
        sem: Dict[str, Any],
        state_name: str,
    ) -> Optional[str]:
        for item in sem["contributions"]:
            if item["event"] is not None:
                continue
            stmt = item["stmt"]
            if isinstance(stmt.expr, Identifier) and str(stmt.expr.name) == state_name:
                return self._whole_segment_branch_node_name(stmt.branch)
        return None

    def _collect_cross_scalar_lfsr_transition_candidate(self, stmt) -> Optional[tuple]:
        if not isinstance(stmt, Block):
            return None

        aliases: Dict[str, str] = {}
        cross_candidates: List[tuple] = []
        for child in stmt.statements:
            if isinstance(child, Assignment):
                if isinstance(child.target, Identifier) and isinstance(child.value, Identifier):
                    aliases[str(child.target.name)] = str(child.value.name)
                continue
            if not isinstance(child, EventStatement):
                continue
            cross = self._whole_segment_rising_cross(child.event)
            if cross is None:
                continue
            body = self._whole_segment_scalar_lfsr_body(child.body)
            if body is None:
                continue
            cross_candidates.append((cross, body))

        if len(cross_candidates) != 1:
            return None
        (clock_node, clock_threshold), body = cross_candidates[0]
        (
            reset_node,
            reset_threshold,
            enable_node,
            enable_threshold,
            seed_name,
            state_names,
            taps,
            shift_sources,
            zero_guard_index,
        ) = body
        if not (
            self._whole_segment_same_expr(clock_threshold, reset_threshold)
            and self._whole_segment_same_expr(clock_threshold, enable_threshold)
        ):
            return None
        threshold_name = self._whole_segment_identifier_name(clock_threshold)
        if threshold_name is None:
            return None

        outputs = self._whole_segment_transition_outputs(
            stmt,
            aliases,
            {name: idx for idx, name in enumerate(state_names)},
        )
        if not outputs:
            return None
        output_nodes = tuple(node for node, _ in outputs)
        output_bits = tuple(bit for _, bit in outputs)
        if any(bit < 0 or bit >= len(state_names) for bit in output_bits):
            return None
        return (
            "cross_scalar_lfsr_transition_bus_v1",
            str(clock_node),
            str(reset_node),
            str(enable_node),
            str(threshold_name),
            str(seed_name),
            "vdd",
            "td",
            "trf",
            tuple(str(name) for name in state_names),
            tuple(int(tap) for tap in taps),
            tuple(int(source) for source in shift_sources),
            output_nodes,
            output_bits,
            int(zero_guard_index),
        )

    def _whole_segment_rising_cross(self, event) -> Optional[Tuple[str, Expr]]:
        if not isinstance(event, EventExpr) or event.event_type != EventType.CROSS:
            return None
        direction = event.direction if event.direction is not None else 0
        if int(direction) != 1 or not event.args:
            return None
        return self._whole_segment_branch_minus_threshold(event.args[0])

    def _whole_segment_branch_minus_threshold(self, expr) -> Optional[Tuple[str, Expr]]:
        if isinstance(expr, BinaryExpr) and expr.op == "-":
            branch = expr.left
            if (
                isinstance(branch, BranchAccess)
                and branch.access_type == "V"
                and branch.node1_index is None
                and branch.node1_index2 is None
                and branch.node2_index is None
                and branch.node2_index2 is None
            ):
                return str(branch.node1), expr.right
        return None

    def _whole_segment_voltage_threshold_condition(
        self,
        expr,
        allowed_ops: set[str],
    ) -> Optional[Tuple[str, Expr]]:
        if not isinstance(expr, BinaryExpr) or expr.op not in allowed_ops:
            return None
        branch = expr.left
        if (
            isinstance(branch, BranchAccess)
            and branch.access_type == "V"
            and branch.node1_index is None
            and branch.node1_index2 is None
            and branch.node2_index is None
            and branch.node2_index2 is None
        ):
            return str(branch.node1), expr.right
        return None

    def _whole_segment_same_expr(self, left, right) -> bool:
        return self._whole_segment_expr_key(left) == self._whole_segment_expr_key(right)

    def _whole_segment_expr_key(self, expr) -> Optional[tuple]:
        if isinstance(expr, Identifier):
            return ("id", str(expr.name))
        if isinstance(expr, NumberLiteral):
            return ("num", float(expr.value))
        if isinstance(expr, BranchAccess):
            return (
                "branch",
                expr.access_type,
                expr.node1,
                expr.node2,
                self._whole_segment_expr_key(expr.node1_index) if expr.node1_index is not None else None,
                self._whole_segment_expr_key(expr.node2_index) if expr.node2_index is not None else None,
            )
        if isinstance(expr, UnaryExpr):
            return ("unary", expr.op, self._whole_segment_expr_key(expr.operand))
        if isinstance(expr, BinaryExpr):
            return (
                "binary",
                expr.op,
                self._whole_segment_expr_key(expr.left),
                self._whole_segment_expr_key(expr.right),
            )
        if isinstance(expr, FunctionCall):
            return (
                "call",
                expr.name,
                tuple(self._whole_segment_expr_key(arg) for arg in expr.args),
            )
        return None

    def _whole_segment_identifier_name(self, expr) -> Optional[str]:
        if isinstance(expr, Identifier):
            return str(expr.name)
        return None

    def _whole_segment_scalar_lfsr_body(self, body) -> Optional[tuple]:
        if not isinstance(body, Block) or len(body.statements) != 1:
            return None
        reset_if = body.statements[0]
        if (
            not isinstance(reset_if, IfStatement)
            or reset_if.else_body is None
            or not isinstance(reset_if.else_body, IfStatement)
        ):
            return None
        reset = self._whole_segment_voltage_threshold_condition(
            reset_if.cond,
            {"<", "<="},
        )
        enable_if = reset_if.else_body
        enable = self._whole_segment_voltage_threshold_condition(
            enable_if.cond,
            {">", ">="},
        )
        if reset is None or enable is None:
            return None
        seed_reset = self._whole_segment_seed_reset(reset_if.then_body)
        if seed_reset is None:
            return None
        seed_name, state_by_bit, zero_guard_index = seed_reset
        state_names = tuple(name for _, name in sorted(state_by_bit.items()))
        shift = self._whole_segment_scalar_lfsr_shift(enable_if.then_body, state_names)
        if shift is None:
            return None
        taps, shift_sources = shift
        return (
            reset[0],
            reset[1],
            enable[0],
            enable[1],
            seed_name,
            state_names,
            taps,
            shift_sources,
            zero_guard_index,
        )

    def _whole_segment_seed_reset(self, stmt) -> Optional[Tuple[str, Dict[int, str], int]]:
        seed_name: Optional[str] = None
        state_by_bit: Dict[int, str] = {}
        guard_state: Optional[str] = None

        def visit(node) -> None:
            nonlocal seed_name, guard_state
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, IfStatement):
                visit(node.then_body)
                return
            if not isinstance(node, Assignment) or not isinstance(node.target, Identifier):
                return
            target = str(node.target.name)
            bit = self._whole_segment_seed_bit_expr(node.value)
            if bit is not None:
                local_seed, bit_index = bit
                if seed_name is None:
                    seed_name = local_seed
                elif seed_name != local_seed:
                    return
                if bit_index >= 0:
                    state_by_bit[int(bit_index)] = target
                return
            if self._whole_segment_expr_is_number(node.value, 1.0):
                guard_state = target

        visit(stmt)
        if seed_name is None or not state_by_bit:
            return None
        width = len(state_by_bit)
        if tuple(sorted(state_by_bit)) != tuple(range(width)):
            return None
        if any(name not in self._state_scalar_slot_by_name for name in state_by_bit.values()):
            return None
        if any(not self._is_integer_variable(name) for name in state_by_bit.values()):
            return None
        zero_guard_index = -1
        if guard_state is not None:
            for bit, state_name in state_by_bit.items():
                if state_name == guard_state:
                    zero_guard_index = int(bit)
                    break
        return seed_name, state_by_bit, zero_guard_index

    def _whole_segment_seed_bit_expr(self, expr) -> Optional[Tuple[str, int]]:
        if not isinstance(expr, BinaryExpr) or expr.op != "&":
            return None
        if not self._whole_segment_expr_is_number(expr.right, 1.0):
            return None
        shift = expr.left
        if not isinstance(shift, BinaryExpr) or shift.op != ">>":
            return None
        if not isinstance(shift.left, Identifier):
            return None
        bit_index = self._whole_segment_int_literal(shift.right)
        if bit_index is None:
            return None
        return str(shift.left.name), int(bit_index)

    def _whole_segment_scalar_lfsr_shift(
        self,
        stmt,
        state_names: Tuple[str, ...],
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if not isinstance(stmt, Block):
            return None
        state_to_bit = {name: idx for idx, name in enumerate(state_names)}
        feedback_name: Optional[str] = None
        taps: Optional[Tuple[int, ...]] = None
        shift_sources: Dict[int, int] = {}
        for child in stmt.statements:
            if not isinstance(child, Assignment) or not isinstance(child.target, Identifier):
                continue
            target = str(child.target.name)
            child_taps = self._whole_segment_xor_state_taps(child.value, state_to_bit)
            if child_taps is not None and target not in state_to_bit:
                feedback_name = target
                taps = child_taps
                continue
            target_bit = state_to_bit.get(target)
            if target_bit is None:
                continue
            if isinstance(child.value, Identifier):
                value_name = str(child.value.name)
                if feedback_name is not None and value_name == feedback_name:
                    shift_sources[target_bit] = -1
                elif value_name in state_to_bit:
                    shift_sources[target_bit] = state_to_bit[value_name]
        if feedback_name is None or taps is None:
            return None
        width = len(state_names)
        if tuple(sorted(shift_sources)) != tuple(range(width)):
            return None
        return taps, tuple(int(shift_sources[idx]) for idx in range(width))

    def _whole_segment_xor_state_taps(
        self,
        expr,
        state_to_bit: Dict[str, int],
    ) -> Optional[Tuple[int, ...]]:
        if isinstance(expr, Identifier) and expr.name in state_to_bit:
            return (state_to_bit[str(expr.name)],)
        if isinstance(expr, BinaryExpr) and expr.op == "^":
            left = self._whole_segment_xor_state_taps(expr.left, state_to_bit)
            right = self._whole_segment_xor_state_taps(expr.right, state_to_bit)
            if left is None or right is None:
                return None
            return (*left, *right)
        return None

    def _whole_segment_transition_outputs(
        self,
        stmt,
        aliases: Dict[str, str],
        state_to_bit: Dict[str, int],
    ) -> Tuple[Tuple[str, int], ...]:
        outputs: List[Tuple[str, int]] = []

        def resolve_name(name: str) -> str:
            seen = set()
            current = str(name)
            while current in aliases and current not in seen:
                seen.add(current)
                current = aliases[current]
            return current

        def visit(node) -> None:
            if node is None:
                return
            if isinstance(node, Block):
                for child in node.statements:
                    visit(child)
                return
            if isinstance(node, EventStatement):
                return
            if not isinstance(node, Contribution):
                return
            out = self._whole_segment_transition_output(node, resolve_name, state_to_bit)
            if out is not None:
                outputs.append(out)

        visit(stmt)
        return tuple(outputs)

    def _whole_segment_transition_output(
        self,
        stmt: Contribution,
        resolve_name,
        state_to_bit: Dict[str, int],
    ) -> Optional[Tuple[str, int]]:
        branch = stmt.branch
        if (
            branch.access_type != "V"
            or branch.node1_index is not None
            or branch.node1_index2 is not None
            or branch.node2 is not None
            or branch.node2_index is not None
            or branch.node2_index2 is not None
        ):
            return None
        transition_affine = self._transition_affine_expr(stmt.expr)
        if transition_affine is None:
            return None
        transition_call, offset_expr, scale_expr = transition_affine
        if not (
            self._whole_segment_expr_is_number(offset_expr, 0.0)
            and self._whole_segment_expr_is_number(scale_expr, 1.0)
        ):
            return None
        if len(transition_call.args) < 1:
            return None
        if len(transition_call.args) > 1 and self._whole_segment_identifier_name(transition_call.args[1]) != "td":
            return None
        if len(transition_call.args) > 2 and self._whole_segment_identifier_name(transition_call.args[2]) != "trf":
            return None
        if len(transition_call.args) > 3 and self._whole_segment_identifier_name(transition_call.args[3]) != "trf":
            return None
        target = transition_call.args[0]
        if not isinstance(target, TernaryExpr):
            return None
        bit_name = self._whole_segment_ternary_bit_name(target.cond)
        if bit_name is None:
            return None
        bit_name = resolve_name(bit_name)
        bit = state_to_bit.get(bit_name)
        if bit is None:
            return None
        if not (
            self._whole_segment_identifier_name(target.true_expr) == "vdd"
            and self._whole_segment_expr_is_number(target.false_expr, 0.0)
        ):
            return None
        return str(branch.node1), int(bit)

    def _whole_segment_ternary_bit_name(self, expr) -> Optional[str]:
        if isinstance(expr, Identifier):
            return str(expr.name)
        if isinstance(expr, ArrayAccess):
            return self._whole_segment_assignment_target_name(expr)
        if isinstance(expr, BinaryExpr) and expr.op in {"!=", ">", ">="}:
            if isinstance(expr.left, Identifier) and self._whole_segment_expr_is_number(expr.right, 0.0):
                return str(expr.left.name)
            if isinstance(expr.left, ArrayAccess) and self._whole_segment_expr_is_number(expr.right, 0.0):
                return self._whole_segment_assignment_target_name(expr.left)
        return None

    def _whole_segment_expr_is_number(self, expr, expected: float) -> bool:
        return isinstance(expr, NumberLiteral) and abs(float(expr.value) - expected) <= 1.0e-18

    def _whole_segment_int_literal(self, expr) -> Optional[int]:
        if isinstance(expr, NumberLiteral):
            value = float(expr.value)
            rounded = int(value)
            if abs(value - rounded) <= 1.0e-18:
                return rounded
        return None

    def _collect_evaluate_ir_static_linear_ops_from_stmt(
        self,
        stmt,
        ops: List[Tuple[int, str, Any, Tuple[Tuple[int, str, Any], ...]]],
    ) -> bool:
        if stmt is None:
            return True

        if isinstance(stmt, Block):
            for child in stmt.statements:
                if not self._collect_evaluate_ir_static_linear_ops_from_stmt(child, ops):
                    return False
            return True

        if isinstance(stmt, EventStatement):
            return self._evaluate_ir_event_is_evaluate_noop(stmt.event)

        if isinstance(stmt, IfStatement):
            condition = self._evaluate_ir_condition_expr(stmt.cond)
            if condition is None or stmt.else_body is None:
                return False
            then_ops: List[tuple] = []
            else_ops: List[tuple] = []
            if not self._collect_evaluate_ir_static_linear_ops_from_stmt(
                stmt.then_body, then_ops
            ):
                return False
            if not self._collect_evaluate_ir_static_linear_ops_from_stmt(
                stmt.else_body, else_ops
            ):
                return False
            if len(then_ops) != len(else_ops):
                return False
            for then_op, else_op in zip(then_ops, else_ops):
                combined = self._combine_evaluate_ir_conditional_ops(
                    condition,
                    then_op,
                    else_op,
                )
                if combined is None:
                    return False
                ops.append(combined)
            return True

        if isinstance(stmt, ForStatement):
            loop = self._evaluate_ir_static_for_values(stmt)
            if loop is None:
                return False
            loop_var, values, final_value = loop
            local_ops: List[tuple] = []
            sentinel = object()
            old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
            try:
                for value in values:
                    self._evaluate_ir_static_loop_values[loop_var] = value
                    if not self._collect_evaluate_ir_static_linear_ops_from_stmt(
                        stmt.body,
                        local_ops,
                    ):
                        return False
            finally:
                if old_value is sentinel:
                    self._evaluate_ir_static_loop_values.pop(loop_var, None)
                else:
                    self._evaluate_ir_static_loop_values[loop_var] = old_value
            self._append_evaluate_ir_static_for_final_state_op(
                loop_var,
                final_value,
                local_ops,
            )
            ops.extend(local_ops)
            return True

        if isinstance(stmt, Contribution):
            branch = stmt.branch
            if (
                branch.access_type != "V"
                or branch.node1_index is not None
                or branch.node1_index2 is not None
                or branch.node2_index is not None
                or branch.node2_index2 is not None
            ):
                return False
            linear = self._evaluate_ir_linear_expr(stmt.expr)
            if linear is None:
                return False
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return False
            bias, terms, condition, false_bias, false_terms = unpacked
            if branch.node2 is not None:
                terms = list(terms)
                terms.append((SOURCE_NODE, branch.node2, 1.0))
                terms = self._evaluate_ir_normalize_terms(terms)
                if condition is not None:
                    false_terms = list(false_terms)
                    false_terms.append((SOURCE_NODE, branch.node2, 1.0))
                    false_terms = self._evaluate_ir_normalize_terms(false_terms)
            ops.append(
                (
                    TARGET_NODE,
                    branch.node1,
                    bias,
                    tuple(terms),
                    condition,
                    false_bias,
                    tuple(false_terms),
                )
            )
            return True

        if isinstance(stmt, Assignment):
            target = None
            target_integer = False
            if isinstance(stmt.target, Identifier):
                target = stmt.target.name
                if target not in self._state_scalar_slot_by_name:
                    return False
                target_integer = self._is_integer_variable(target)
            elif isinstance(stmt.target, ArrayAccess):
                array_slot = self._evaluate_ir_static_array_slot(stmt.target)
                if array_slot is None:
                    return False
                target, target_integer = array_slot
            else:
                return False
            linear = self._evaluate_ir_linear_expr(stmt.value)
            if linear is None:
                return False
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return False
            bias, terms, condition, false_bias, false_terms = unpacked
            all_terms = list(terms) + list(false_terms)
            if condition is not None:
                all_terms.extend(condition[2])
                all_terms.extend(condition[4])
            if any(kind == SOURCE_STATE and name == target for kind, name, _ in all_terms):
                return False
            ops.append(
                (
                    TARGET_STATE,
                    target,
                    bias,
                    tuple(terms),
                    condition,
                    false_bias,
                    tuple(false_terms),
                    target_integer,
                )
            )
            return True

        return False

    def _collect_evaluate_ir_static_linear_rejections(
        self,
        stmt,
    ) -> Tuple[Tuple[str, int], ...]:
        """Explain why a statement tree cannot become static-linear evaluate IR."""
        if self._collect_evaluate_ir_static_linear_ops(stmt):
            return ()

        counts: Dict[str, int] = {}

        def add(reason: str, count: int = 1) -> None:
            token = self._evaluate_ir_reason_token(reason)
            counts[token] = counts.get(token, 0) + int(count)

        self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt, add)
        if not counts:
            add("unknown")
        return tuple(sorted(counts.items()))

    def _collect_evaluate_ir_static_linear_rejections_from_stmt(
        self,
        stmt,
        add: Callable[[str, int], None],
    ) -> None:
        if stmt is None:
            return

        probe_ops: List[tuple] = []
        if self._collect_evaluate_ir_static_linear_ops_from_stmt(stmt, probe_ops):
            return

        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_evaluate_ir_static_linear_rejections_from_stmt(child, add)
            return

        if isinstance(stmt, EventStatement):
            if not self._evaluate_ir_event_is_evaluate_noop(stmt.event):
                add("event_statement")
                self._collect_evaluate_ir_event_rejections(stmt.event, add)
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt.body, add)
            return

        if isinstance(stmt, Contribution):
            branch = stmt.branch
            if branch.access_type != "V":
                add("contribution_non_voltage_branch")
            if self._evaluate_ir_branch_has_dynamic_index(branch):
                add("contribution_dynamic_branch")
            if self._evaluate_ir_linear_expr(stmt.expr) is None:
                add("contribution_non_static_linear_expr")
            self._collect_evaluate_ir_expr_rejections(branch, add)
            self._collect_evaluate_ir_expr_rejections(stmt.expr, add)
            return

        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, ArrayAccess):
                if self._evaluate_ir_static_array_slot(target) is None:
                    add("assignment_array_target")
                    if self._evaluate_ir_static_array_index(target.index) is None:
                        add("assignment_dynamic_array_index")
                self._collect_evaluate_ir_expr_rejections(target.index, add)
            elif not isinstance(target, Identifier):
                add("assignment_unsupported_target")
                self._collect_evaluate_ir_expr_rejections(target, add)
            elif target.name not in self._state_scalar_slot_by_name:
                add("assignment_non_scalar_state_target")

            linear = self._evaluate_ir_linear_expr(stmt.value)
            if linear is None:
                add("assignment_non_static_linear_expr")
            else:
                target_name = None
                if isinstance(target, Identifier):
                    target_name = target.name
                elif isinstance(target, ArrayAccess):
                    array_slot = self._evaluate_ir_static_array_slot(target)
                    if array_slot is not None:
                        target_name = array_slot[0]
                if target_name is None:
                    self._collect_evaluate_ir_expr_rejections(stmt.value, add)
                    return
                unpacked = self._evaluate_ir_unpack_linear(linear)
                if unpacked is not None:
                    bias, terms, condition, false_bias, false_terms = unpacked
                    all_terms = list(terms) + list(false_terms)
                    if condition is not None:
                        all_terms.extend(condition[2])
                        all_terms.extend(condition[4])
                    if any(
                        kind == SOURCE_STATE and name == target_name
                        for kind, name, _ in all_terms
                    ):
                        add("assignment_self_dependent_state")
            self._collect_evaluate_ir_expr_rejections(stmt.value, add)
            return

        if isinstance(stmt, IfStatement):
            add("if_statement")
            if self._evaluate_ir_condition_expr(stmt.cond) is None:
                add("condition_non_static_linear_expr")
            self._collect_evaluate_ir_expr_rejections(stmt.cond, add)
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(
                stmt.then_body, add
            )
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(
                stmt.else_body, add
            )
            return

        if isinstance(stmt, ForStatement):
            loop = self._evaluate_ir_static_for_values(stmt)
            if loop is not None:
                loop_var, values, _final_value = loop
                sentinel = object()
                old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
                try:
                    for value in values:
                        self._evaluate_ir_static_loop_values[loop_var] = value
                        self._collect_evaluate_ir_static_linear_rejections_from_stmt(
                            stmt.body,
                            add,
                        )
                finally:
                    if old_value is sentinel:
                        self._evaluate_ir_static_loop_values.pop(loop_var, None)
                    else:
                        self._evaluate_ir_static_loop_values[loop_var] = old_value
                return

            add("for_statement")
            add("for_dynamic_bounds")
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt.init, add)
            if self._evaluate_ir_static_condition_truth(stmt.cond) is None:
                add("condition_non_static_expr")
            self._collect_evaluate_ir_expr_rejections(stmt.cond, add)
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt.update, add)
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt.body, add)
            return

        if isinstance(stmt, WhileStatement):
            add("while_statement")
            if self._evaluate_ir_condition_expr(stmt.cond) is None:
                add("condition_non_static_linear_expr")
            self._collect_evaluate_ir_expr_rejections(stmt.cond, add)
            self._collect_evaluate_ir_static_linear_rejections_from_stmt(stmt.body, add)
            return

        if isinstance(stmt, CaseStatement):
            add("case_statement")
            self._collect_evaluate_ir_expr_rejections(stmt.expr, add)
            for item in stmt.items:
                for value in item.values:
                    self._collect_evaluate_ir_expr_rejections(value, add)
                self._collect_evaluate_ir_static_linear_rejections_from_stmt(
                    item.body, add
                )
            return

        if isinstance(stmt, SystemTask):
            add("system_task")
            for arg in stmt.args:
                self._collect_evaluate_ir_expr_rejections(arg, add)
            return

        add("unsupported_statement")

    def _collect_evaluate_ir_event_rejections(
        self,
        event,
        add: Callable[[str, int], None],
    ) -> None:
        if isinstance(event, EventExpr):
            if event.event_type == EventType.CROSS:
                add("event_cross")
            elif event.event_type == EventType.ABOVE:
                add("event_above")
            elif event.event_type == EventType.TIMER:
                add("event_timer")
            else:
                add("event_other")
            for arg in event.args:
                self._collect_evaluate_ir_expr_rejections(arg, add)
            if event.time_tol_expr is not None:
                self._collect_evaluate_ir_expr_rejections(event.time_tol_expr, add)
            if event.expr_tol_expr is not None:
                self._collect_evaluate_ir_expr_rejections(event.expr_tol_expr, add)
            return
        if isinstance(event, CombinedEvent):
            add("combined_event")
            for child in event.events:
                self._collect_evaluate_ir_event_rejections(child, add)
            return
        add("event_unknown")

    def _collect_evaluate_ir_expr_rejections(
        self,
        expr,
        add: Callable[[str, int], None],
    ) -> None:
        if expr is None:
            return

        if isinstance(expr, (NumberLiteral, StringLiteral, Identifier)):
            return

        if isinstance(expr, ArrayAccess):
            if self._evaluate_ir_static_array_slot(expr) is None:
                add("expr_array_access")
                if self._evaluate_ir_static_array_index(expr.index) is None:
                    add("expr_dynamic_array_index")
            self._collect_evaluate_ir_expr_rejections(expr.index, add)
            return

        if isinstance(expr, BranchAccess):
            if expr.access_type != "V":
                add("expr_non_voltage_branch")
            if self._evaluate_ir_branch_has_dynamic_index(expr):
                add("expr_dynamic_branch")
            for index_expr in (
                expr.node1_index,
                expr.node1_index2,
                expr.node2_index,
                expr.node2_index2,
            ):
                self._collect_evaluate_ir_expr_rejections(index_expr, add)
            return

        if isinstance(expr, FunctionCall):
            add(f"expr_function_{expr.name}")
            for arg in expr.args:
                self._collect_evaluate_ir_expr_rejections(arg, add)
            return

        if isinstance(expr, MethodCall):
            add("expr_method_call")
            for arg in expr.args:
                self._collect_evaluate_ir_expr_rejections(arg, add)
            return

        if isinstance(expr, TernaryExpr):
            if self._evaluate_ir_condition_expr(expr.cond) is None:
                add("expr_unsupported_ternary_condition")
            if self._evaluate_ir_linear_expr(expr.true_expr) is None:
                add("expr_unsupported_ternary_true")
            if self._evaluate_ir_linear_expr(expr.false_expr) is None:
                add("expr_unsupported_ternary_false")
            self._collect_evaluate_ir_expr_rejections(expr.cond, add)
            self._collect_evaluate_ir_expr_rejections(expr.true_expr, add)
            self._collect_evaluate_ir_expr_rejections(expr.false_expr, add)
            return

        if isinstance(expr, UnaryExpr):
            if expr.op not in {"+", "-"}:
                add("expr_unsupported_unary_op")
            self._collect_evaluate_ir_expr_rejections(expr.operand, add)
            return

        if isinstance(expr, BinaryExpr):
            if self._evaluate_ir_linear_expr(expr) is None and self._rust_scalar_expr(expr) is None:
                add("expr_non_static_linear_binary")
                if expr.op not in {"+", "-", "*", "/", ">", "<", ">=", "<=", "==", "!="}:
                    add("expr_unsupported_binary_op")
            self._collect_evaluate_ir_expr_rejections(expr.left, add)
            self._collect_evaluate_ir_expr_rejections(expr.right, add)
            return

        add("expr_unsupported")

    def _evaluate_ir_branch_has_dynamic_index(self, branch: BranchAccess) -> bool:
        return (
            branch.node1_index is not None
            or branch.node1_index2 is not None
            or branch.node2_index is not None
            or branch.node2_index2 is not None
        )

    def _combine_evaluate_ir_conditional_ops(
        self,
        condition,
        then_op: tuple,
        else_op: tuple,
    ) -> Optional[tuple]:
        if len(then_op) not in {7, 8} or len(else_op) != len(then_op):
            return None
        if then_op[0] != else_op[0] or then_op[1] != else_op[1]:
            return None
        if then_op[4] is not None or else_op[4] is not None:
            return None
        if len(then_op) == 8 and then_op[7] != else_op[7]:
            return None
        common = (
            then_op[0],
            then_op[1],
            then_op[2],
            then_op[3],
            condition,
            else_op[2],
            else_op[3],
        )
        if len(then_op) == 8:
            return (*common, then_op[7])
        return common

    def _evaluate_ir_reason_token(self, reason: str) -> str:
        token = []
        for ch in str(reason).lower():
            token.append(ch if ch.isalnum() else "_")
        return "_".join(part for part in "".join(token).split("_") if part) or "unknown"

    def _evaluate_ir_static_array_slot(
        self,
        expr: ArrayAccess,
    ) -> Optional[Tuple[str, bool]]:
        layout = self._state_array_range_by_name.get(expr.name)
        if layout is None:
            return None
        idx = self._evaluate_ir_static_array_index(expr.index)
        if idx is None:
            return None
        lo, hi, integer = layout
        if not (lo <= idx <= hi):
            return None
        return CompiledModel._state_array_slot_name(expr.name, idx), bool(integer)

    def _evaluate_ir_static_array_index(self, expr: Expr) -> Optional[int]:
        value = self._evaluate_ir_static_array_index_value(expr)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        idx = int(numeric)
        if numeric != float(idx):
            return None
        return idx

    def _evaluate_ir_static_array_index_value(self, expr: Expr) -> Optional[Any]:
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, Identifier):
            if expr.name in self._evaluate_ir_static_loop_values:
                return self._evaluate_ir_static_loop_values[expr.name]
            return self._evaluate_ir_static_param_values.get(expr.name)
        if isinstance(expr, UnaryExpr):
            value = self._evaluate_ir_static_array_index_value(expr.operand)
            if value is None:
                return None
            if expr.op == "+":
                return value
            if expr.op == "-":
                return -value
            return None
        if isinstance(expr, BinaryExpr):
            left = self._evaluate_ir_static_array_index_value(expr.left)
            right = self._evaluate_ir_static_array_index_value(expr.right)
            if left is None or right is None:
                return None
            try:
                if expr.op == "+":
                    return left + right
                if expr.op == "-":
                    return left - right
                if expr.op == "*":
                    return left * right
                if expr.op == "/":
                    if right == 0:
                        return None
                    return left / right
                if expr.op == "%":
                    if right == 0:
                        return None
                    return left % right
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        return None

    def _evaluate_ir_static_condition_truth(self, expr: Expr) -> Optional[bool]:
        if isinstance(expr, BinaryExpr):
            left = self._evaluate_ir_static_array_index_value(expr.left)
            right = self._evaluate_ir_static_array_index_value(expr.right)
            if left is None or right is None:
                return None
            try:
                if expr.op == "<":
                    return left < right
                if expr.op == "<=":
                    return left <= right
                if expr.op == ">":
                    return left > right
                if expr.op == ">=":
                    return left >= right
                if expr.op == "==":
                    return left == right
                if expr.op == "!=":
                    return left != right
            except (TypeError, ValueError):
                return None
            return None

        value = self._evaluate_ir_static_array_index_value(expr)
        if value is None:
            return None
        return bool(value)

    def _evaluate_ir_static_for_values(
        self,
        stmt: ForStatement,
    ) -> Optional[Tuple[str, Tuple[int, ...], int]]:
        if not isinstance(stmt.init, Assignment) or not isinstance(
            stmt.update,
            Assignment,
        ):
            return None
        if not isinstance(stmt.init.target, Identifier) or not isinstance(
            stmt.update.target,
            Identifier,
        ):
            return None
        loop_var = stmt.init.target.name
        if stmt.update.target.name != loop_var:
            return None

        initial = self._evaluate_ir_static_array_index(stmt.init.value)
        if initial is None:
            return None

        values: List[int] = []
        current = initial
        sentinel = object()
        old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
        try:
            for _ in range(4096):
                self._evaluate_ir_static_loop_values[loop_var] = current
                active = self._evaluate_ir_static_condition_truth(stmt.cond)
                if active is None:
                    return None
                if not active:
                    return loop_var, tuple(values), current
                values.append(current)
                next_value = self._evaluate_ir_static_array_index(stmt.update.value)
                if next_value is None or next_value == current:
                    return None
                current = next_value
        finally:
            if old_value is sentinel:
                self._evaluate_ir_static_loop_values.pop(loop_var, None)
            else:
                self._evaluate_ir_static_loop_values[loop_var] = old_value
        return None

    def _append_evaluate_ir_static_for_final_state_op(
        self,
        loop_var: str,
        final_value: int,
        ops: List[tuple],
    ) -> None:
        if loop_var not in self._state_scalar_slot_by_name:
            return
        ops.append(
            (
                TARGET_STATE,
                loop_var,
                float(final_value),
                (),
                None,
                0.0,
                (),
                self._is_integer_variable(loop_var),
            )
        )

    def _collect_transition_target_ir_ops(self, stmt) -> Tuple[tuple, ...]:
        """Collect transition() target expressions that can become array IR.

        This metadata is intentionally non-executing.  It identifies transition
        targets that can already be represented by the static-linear IR value
        form, so the Rust path can later evaluate targets without re-parsing
        generated Python code.
        """

        ops: List[tuple] = []
        self._collect_transition_target_ir_ops_from_stmt(stmt, ops)
        return tuple(ops)

    def _collect_transition_target_ir_ops_from_stmt(
        self,
        stmt,
        ops: List[tuple],
    ) -> None:
        if stmt is None:
            return

        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_transition_target_ir_ops_from_stmt(child, ops)
            return

        if isinstance(stmt, IfStatement):
            self._collect_transition_target_ir_ops_from_stmt(stmt.then_body, ops)
            if stmt.else_body is not None:
                self._collect_transition_target_ir_ops_from_stmt(stmt.else_body, ops)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._collect_transition_target_ir_ops_from_stmt(item.body, ops)
            return

        if isinstance(stmt, EventStatement):
            self._collect_transition_target_ir_ops_from_stmt(stmt.body, ops)
            return

        if isinstance(stmt, ForStatement):
            self._collect_transition_target_ir_ops_from_stmt(stmt.body, ops)
            return

        if isinstance(stmt, WhileStatement):
            self._collect_transition_target_ir_ops_from_stmt(stmt.body, ops)
            return

        transition_op = self._transition_target_ir_op_from_contribution(stmt)
        if transition_op is not None:
            ops.append(transition_op)

    def _transition_target_ir_op_from_contribution(self, stmt) -> Optional[tuple]:
        if not isinstance(stmt, Contribution):
            return None

        branch = stmt.branch
        if (
            branch.access_type != "V"
            or branch.node1_index is not None
            or branch.node1_index2 is not None
            or branch.node2_index is not None
            or branch.node2_index2 is not None
        ):
            return None
        transition_affine = self._transition_affine_expr(stmt.expr)
        if transition_affine is None:
            return None

        transition_call, _, _ = transition_affine
        target_expr = transition_call.args[0] if transition_call.args else NumberLiteral(0.0)
        target_linear = self._evaluate_ir_linear_expr(target_expr)
        if target_linear is None:
            return None
        unpacked = self._evaluate_ir_unpack_linear(target_linear)
        if unpacked is None:
            return None
        bias, terms, condition, false_bias, false_terms = unpacked
        delay = (
            self._rust_scalar_expr(transition_call.args[1])
            if len(transition_call.args) > 1
            else 0.0
        )
        rise = (
            self._rust_scalar_expr(transition_call.args[2])
            if len(transition_call.args) > 2
            else 0.0
        )
        fall = (
            self._rust_scalar_expr(transition_call.args[3])
            if len(transition_call.args) > 3
            else rise
        )
        if delay is None or rise is None or fall is None:
            return None
        return (
            branch.node1,
            branch.node2,
            self._alloc_stateful_func_key("transition", transition_call),
            bias,
            tuple(terms),
            condition,
            false_bias,
            tuple(false_terms),
            delay,
            rise,
            fall,
        )

    def _collect_ordered_transition_segment_ir_ops(self, stmt) -> Tuple[tuple, tuple]:
        """Collect static-linear ops followed by transition target ops.

        This shadow-only segment is intentionally narrower than full evaluate:
        event bodies still execute in generated Python, then this segment replays
        deterministic static-linear assignments and transition target
        expressions against indexed arrays for parity.
        """

        linear_ops: List[tuple] = []
        transition_ops: List[tuple] = []
        if not self._collect_ordered_transition_segment_ir_ops_from_stmt(
            stmt,
            linear_ops,
            transition_ops,
            False,
        ):
            return (), ()
        if not transition_ops:
            return (), ()
        return tuple(linear_ops), tuple(transition_ops)

    def _collect_ordered_transition_segment_ir_ops_from_stmt(
        self,
        stmt,
        linear_ops: List[tuple],
        transition_ops: List[tuple],
        seen_transition: bool,
    ) -> bool:
        if stmt is None:
            return True

        if isinstance(stmt, Block):
            local_seen_transition = seen_transition
            for child in stmt.statements:
                before = len(transition_ops)
                if not self._collect_ordered_transition_segment_ir_ops_from_stmt(
                    child,
                    linear_ops,
                    transition_ops,
                    local_seen_transition,
                ):
                    return False
                if len(transition_ops) > before:
                    local_seen_transition = True
            return True

        if isinstance(stmt, EventStatement):
            return True

        if isinstance(stmt, ForStatement):
            loop = self._evaluate_ir_static_for_values(stmt)
            if loop is None:
                return False
            loop_var, values, final_value = loop
            local_linear_ops: List[tuple] = []
            local_transition_ops: List[tuple] = []
            local_seen_transition = seen_transition
            sentinel = object()
            old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
            try:
                for value in values:
                    self._evaluate_ir_static_loop_values[loop_var] = value
                    before = len(local_transition_ops)
                    if not self._collect_ordered_transition_segment_ir_ops_from_stmt(
                        stmt.body,
                        local_linear_ops,
                        local_transition_ops,
                        local_seen_transition,
                    ):
                        return False
                    if len(local_transition_ops) > before:
                        local_seen_transition = True
            finally:
                if old_value is sentinel:
                    self._evaluate_ir_static_loop_values.pop(loop_var, None)
                else:
                    self._evaluate_ir_static_loop_values[loop_var] = old_value
            if local_transition_ops:
                return False
            self._append_evaluate_ir_static_for_final_state_op(
                loop_var,
                final_value,
                local_linear_ops,
            )
            linear_ops.extend(local_linear_ops)
            transition_ops.extend(local_transition_ops)
            return True

        if isinstance(stmt, Contribution):
            transition_op = self._transition_target_ir_op_from_contribution(stmt)
            if transition_op is not None:
                transition_ops.append(transition_op)
                return True
            if seen_transition:
                return False

            branch = stmt.branch
            if (
                branch.access_type != "V"
                or branch.node1_index is not None
                or branch.node1_index2 is not None
                or branch.node2_index is not None
                or branch.node2_index2 is not None
            ):
                return False
            linear = self._evaluate_ir_linear_expr(stmt.expr)
            if linear is None:
                return False
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return False
            bias, terms, condition, false_bias, false_terms = unpacked
            if branch.node2 is not None:
                terms = list(terms)
                terms.append((SOURCE_NODE, branch.node2, 1.0))
                terms = self._evaluate_ir_normalize_terms(terms)
                if condition is not None:
                    false_terms = list(false_terms)
                    false_terms.append((SOURCE_NODE, branch.node2, 1.0))
                    false_terms = self._evaluate_ir_normalize_terms(false_terms)
            linear_ops.append(
                (
                    TARGET_NODE,
                    branch.node1,
                    bias,
                    tuple(terms),
                    condition,
                    false_bias,
                    tuple(false_terms),
                )
            )
            return True

        if isinstance(stmt, Assignment):
            if seen_transition:
                return False
            target = None
            target_integer = False
            if isinstance(stmt.target, Identifier):
                target = stmt.target.name
                if target not in self._state_scalar_slot_by_name:
                    return False
                target_integer = self._is_integer_variable(target)
            elif isinstance(stmt.target, ArrayAccess):
                array_slot = self._evaluate_ir_static_array_slot(stmt.target)
                if array_slot is None:
                    return False
                target, target_integer = array_slot
            else:
                return False
            linear = self._evaluate_ir_linear_expr(stmt.value)
            if linear is None:
                return False
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return False
            bias, terms, condition, false_bias, false_terms = unpacked
            all_terms = list(terms) + list(false_terms)
            if condition is not None:
                all_terms.extend(condition[2])
                all_terms.extend(condition[4])
            if any(kind == SOURCE_STATE and name == target for kind, name, _ in all_terms):
                return False
            linear_ops.append(
                (
                    TARGET_STATE,
                    target,
                    bias,
                    tuple(terms),
                    condition,
                    false_bias,
                    tuple(false_terms),
                    target_integer,
                )
            )
            return True

        return False

    def _evaluate_ir_event_is_evaluate_noop(self, event) -> bool:
        if isinstance(event, EventExpr):
            return event.event_type in {EventType.INITIAL_STEP, EventType.FINAL_STEP}
        if isinstance(event, CombinedEvent):
            return all(
                child.event_type in {EventType.INITIAL_STEP, EventType.FINAL_STEP}
                for child in event.events
            )
        return False

    def _evaluate_ir_unpack_linear(
        self,
        linear,
    ) -> Optional[
        Tuple[
            Any,
            Tuple[Tuple[int, str, Any], ...],
            Optional[
                Tuple[
                    int,
                    Any,
                    Tuple[Tuple[int, str, Any], ...],
                    Any,
                    Tuple[Tuple[int, str, Any], ...],
                ]
            ],
            Any,
            Tuple[Tuple[int, str, Any], ...],
        ]
    ]:
        if len(linear) == 2:
            bias, terms = linear
            return bias, tuple(terms), None, 0.0, ()
        if len(linear) == 5:
            bias, terms, condition, false_bias, false_terms = linear
            return bias, tuple(terms), condition, false_bias, tuple(false_terms)
        return None

    def _evaluate_ir_is_plain_linear(self, linear) -> bool:
        return linear is not None and len(linear) == 2

    def _evaluate_ir_is_conditional_linear(self, linear) -> bool:
        return linear is not None and len(linear) == 5

    def _evaluate_ir_negate_linear(self, linear):
        if self._evaluate_ir_is_plain_linear(linear):
            bias, terms = linear
            return (
                self._rust_scalar_neg(bias),
                tuple(
                    (kind, name, self._rust_scalar_neg(gain))
                    for kind, name, gain in terms
                ),
            )
        if self._evaluate_ir_is_conditional_linear(linear):
            bias, terms, condition, false_bias, false_terms = linear
            return (
                self._rust_scalar_neg(bias),
                tuple(
                    (kind, name, self._rust_scalar_neg(gain))
                    for kind, name, gain in terms
                ),
                condition,
                self._rust_scalar_neg(false_bias),
                tuple(
                    (kind, name, self._rust_scalar_neg(gain))
                    for kind, name, gain in false_terms
                ),
            )
        return None

    def _evaluate_ir_add_linear_any(self, left, right):
        if self._evaluate_ir_is_plain_linear(left) and self._evaluate_ir_is_plain_linear(right):
            return self._evaluate_ir_add_linear(left, right)
        if self._evaluate_ir_is_conditional_linear(left) and self._evaluate_ir_is_plain_linear(right):
            bias, terms, condition, false_bias, false_terms = left
            true_linear = self._evaluate_ir_add_linear((bias, terms), right)
            false_linear = self._evaluate_ir_add_linear((false_bias, false_terms), right)
            return (
                true_linear[0],
                true_linear[1],
                condition,
                false_linear[0],
                false_linear[1],
            )
        if self._evaluate_ir_is_plain_linear(left) and self._evaluate_ir_is_conditional_linear(right):
            return self._evaluate_ir_add_linear_any(right, left)
        if (
            self._evaluate_ir_is_conditional_linear(left)
            and self._evaluate_ir_is_conditional_linear(right)
            and left[2] == right[2]
        ):
            true_linear = self._evaluate_ir_add_linear((left[0], left[1]), (right[0], right[1]))
            false_linear = self._evaluate_ir_add_linear((left[3], left[4]), (right[3], right[4]))
            return (
                true_linear[0],
                true_linear[1],
                left[2],
                false_linear[0],
                false_linear[1],
            )
        return None

    def _evaluate_ir_scale_linear_any(self, linear, scalar):
        if self._evaluate_ir_is_plain_linear(linear):
            return self._evaluate_ir_scale_linear(linear, scalar)
        if self._evaluate_ir_is_conditional_linear(linear):
            true_linear = self._evaluate_ir_scale_linear((linear[0], linear[1]), scalar)
            false_linear = self._evaluate_ir_scale_linear((linear[3], linear[4]), scalar)
            return (
                true_linear[0],
                true_linear[1],
                linear[2],
                false_linear[0],
                false_linear[1],
            )
        return None

    def _evaluate_ir_piecewise_function_expr(self, expr: FunctionCall):
        name = str(expr.name).lower()
        if name == "abs" and len(expr.args) == 1:
            linear = self._evaluate_ir_linear_expr(expr.args[0])
            if not self._evaluate_ir_is_plain_linear(linear):
                return None
            negated = self._evaluate_ir_negate_linear(linear)
            if negated is None:
                return None
            bias, terms = linear
            return (
                bias,
                terms,
                (COND_GE, bias, tuple(terms), 0.0, ()),
                negated[0],
                negated[1],
            )

        if name in {"min", "max"} and len(expr.args) == 2:
            left = self._evaluate_ir_linear_expr(expr.args[0])
            right = self._evaluate_ir_linear_expr(expr.args[1])
            if (
                not self._evaluate_ir_is_plain_linear(left)
                or not self._evaluate_ir_is_plain_linear(right)
            ):
                return None
            op_kind = COND_LE if name == "min" else COND_GE
            return (
                left[0],
                left[1],
                (op_kind, left[0], tuple(left[1]), right[0], tuple(right[1])),
                right[0],
                right[1],
            )

        return None

    def _evaluate_ir_linear_expr(
        self,
        expr: Expr,
    ) -> Optional[Tuple[Any, Tuple[Tuple[int, str, Any], ...]]]:
        scalar = self._rust_scalar_expr(expr)
        if scalar is not None:
            return scalar, ()

        if isinstance(expr, Identifier):
            if expr.name in self._state_scalar_slot_by_name:
                return 0.0, ((SOURCE_STATE, expr.name, 1.0),)
            return None

        if isinstance(expr, ArrayAccess):
            array_slot = self._evaluate_ir_static_array_slot(expr)
            if array_slot is None:
                return None
            slot_name, _ = array_slot
            return 0.0, ((SOURCE_STATE, slot_name, 1.0),)

        if isinstance(expr, BranchAccess):
            if (
                expr.access_type != "V"
                or expr.node1_index is not None
                or expr.node1_index2 is not None
                or expr.node2_index is not None
                or expr.node2_index2 is not None
            ):
                return None
            terms = [(SOURCE_NODE, expr.node1, 1.0)]
            if expr.node2 is not None:
                terms.append((SOURCE_NODE, expr.node2, -1.0))
            return 0.0, tuple(terms)

        if isinstance(expr, TernaryExpr):
            condition = self._evaluate_ir_condition_expr(expr.cond)
            true_linear = self._evaluate_ir_linear_expr(expr.true_expr)
            false_linear = self._evaluate_ir_linear_expr(expr.false_expr)
            if (
                condition is None
                or not self._evaluate_ir_is_plain_linear(true_linear)
                or not self._evaluate_ir_is_plain_linear(false_linear)
            ):
                return None
            true_bias, true_terms = true_linear
            false_bias, false_terms = false_linear
            return true_bias, true_terms, condition, false_bias, false_terms

        if isinstance(expr, UnaryExpr):
            linear = self._evaluate_ir_linear_expr(expr.operand)
            if not self._evaluate_ir_is_plain_linear(linear):
                return None
            bias, terms = linear
            if expr.op == "+":
                return bias, terms
            if expr.op == "-":
                return self._evaluate_ir_negate_linear(linear)
            return None

        if isinstance(expr, FunctionCall):
            return self._evaluate_ir_piecewise_function_expr(expr)

        if isinstance(expr, BinaryExpr):
            if expr.op in {"+", "-"}:
                left = self._evaluate_ir_linear_expr(expr.left)
                right = self._evaluate_ir_linear_expr(expr.right)
                if left is None or right is None:
                    return None
                if expr.op == "-":
                    right = self._evaluate_ir_negate_linear(right)
                    if right is None:
                        return None
                return self._evaluate_ir_add_linear_any(left, right)

            if expr.op == "*":
                left_scalar = self._rust_scalar_expr(expr.left)
                right_scalar = self._rust_scalar_expr(expr.right)
                if left_scalar is not None:
                    right = self._evaluate_ir_linear_expr(expr.right)
                    if right is None:
                        return None
                    return self._evaluate_ir_scale_linear_any(right, left_scalar)
                if right_scalar is not None:
                    left = self._evaluate_ir_linear_expr(expr.left)
                    if left is None:
                        return None
                    return self._evaluate_ir_scale_linear_any(left, right_scalar)
                return None

            if expr.op == "/":
                right_scalar = self._rust_scalar_expr(expr.right)
                left = self._evaluate_ir_linear_expr(expr.left)
                if right_scalar is None or left is None:
                    return None
                inv = self._rust_scalar_div(1.0, right_scalar)
                if inv is None:
                    return None
                return self._evaluate_ir_scale_linear_any(left, inv)

        return None

    def _evaluate_ir_condition_expr(
        self,
        expr: Expr,
    ) -> Optional[
        Tuple[
            int,
            Any,
            Tuple[Tuple[int, str, Any], ...],
            Any,
            Tuple[Tuple[int, str, Any], ...],
        ]
    ]:
        if not isinstance(expr, BinaryExpr):
            linear = self._evaluate_ir_linear_expr(expr)
            if not self._evaluate_ir_is_plain_linear(linear):
                return None
            return COND_NE, linear[0], tuple(linear[1]), 0.0, ()
        op_kind_by_symbol = {
            ">": COND_GT,
            "<": COND_LT,
            ">=": COND_GE,
            "<=": COND_LE,
            "==": COND_EQ,
            "!=": COND_NE,
        }
        op_kind = op_kind_by_symbol.get(expr.op)
        if op_kind is None:
            return None
        left = self._evaluate_ir_linear_expr(expr.left)
        right = self._evaluate_ir_linear_expr(expr.right)
        if (
            not self._evaluate_ir_is_plain_linear(left)
            or not self._evaluate_ir_is_plain_linear(right)
        ):
            return None
        return op_kind, left[0], tuple(left[1]), right[0], tuple(right[1])

    def _evaluate_ir_add_linear(
        self,
        left: Tuple[Any, Tuple[Tuple[int, str, Any], ...]],
        right: Tuple[Any, Tuple[Tuple[int, str, Any], ...]],
    ) -> Tuple[Any, Tuple[Tuple[int, str, Any], ...]]:
        return (
            self._rust_scalar_add(left[0], right[0]),
            self._evaluate_ir_normalize_terms([*left[1], *right[1]]),
        )

    def _evaluate_ir_scale_linear(
        self,
        linear: Tuple[Any, Tuple[Tuple[int, str, Any], ...]],
        scalar: Any,
    ) -> Tuple[Any, Tuple[Tuple[int, str, Any], ...]]:
        return (
            self._rust_scalar_mul(linear[0], scalar),
            tuple(
                (kind, name, self._rust_scalar_mul(gain, scalar))
                for kind, name, gain in linear[1]
            ),
        )

    def _evaluate_ir_normalize_terms(
        self,
        terms: List[Tuple[int, str, Any]],
    ) -> Tuple[Tuple[int, str, Any], ...]:
        by_source: Dict[Tuple[int, str], Any] = {}
        order: List[Tuple[int, str]] = []
        for kind, name, gain in terms:
            key = (int(kind), str(name))
            if key not in by_source:
                by_source[key] = gain
                order.append(key)
            else:
                by_source[key] = self._rust_scalar_add(by_source[key], gain)
        normalized = []
        for kind, name in order:
            gain = by_source[(kind, name)]
            if self._rust_scalar_is_number(gain) and float(gain) == 0.0:
                continue
            normalized.append((kind, name, gain))
        return tuple(normalized)

    def _rust_affine_expr(self, expr: Expr) -> Optional[Tuple[Optional[str], Any, Any]]:
        scalar = self._rust_scalar_expr(expr)
        if scalar is not None:
            return None, 0.0, scalar

        if isinstance(expr, BranchAccess):
            if (
                expr.access_type == "V"
                and expr.node2 is None
                and expr.node1_index is None
                and expr.node1_index2 is None
            ):
                return expr.node1, 1.0, 0.0
            return None

        if isinstance(expr, UnaryExpr):
            affine = self._rust_affine_expr(expr.operand)
            if affine is None:
                return None
            node, gain, bias = affine
            if expr.op == "-":
                return node, self._rust_scalar_neg(gain), self._rust_scalar_neg(bias)
            if expr.op == "+":
                return node, gain, bias
            return None

        if isinstance(expr, BinaryExpr):
            left = self._rust_affine_expr(expr.left)
            right = self._rust_affine_expr(expr.right)
            if expr.op in {"+", "-"}:
                if left is None or right is None:
                    return None
                if expr.op == "-":
                    right = (right[0], -right[1], -right[2])
                return self._combine_rust_affine(left, right)

            if expr.op == "*":
                left_scalar = self._rust_scalar_expr(expr.left)
                right_scalar = self._rust_scalar_expr(expr.right)
                if left_scalar is not None and right is not None:
                    return (
                        right[0],
                        self._rust_scalar_mul(right[1], left_scalar),
                        self._rust_scalar_mul(right[2], left_scalar),
                    )
                if right_scalar is not None and left is not None:
                    return (
                        left[0],
                        self._rust_scalar_mul(left[1], right_scalar),
                        self._rust_scalar_mul(left[2], right_scalar),
                    )
                return None

            if expr.op == "/":
                right_scalar = self._rust_scalar_expr(expr.right)
                if right_scalar is None or left is None:
                    return None
                gain = self._rust_scalar_div(left[1], right_scalar)
                bias = self._rust_scalar_div(left[2], right_scalar)
                if gain is None or bias is None:
                    return None
                return left[0], gain, bias

        return None

    def _combine_rust_affine(
        self,
        left: Tuple[Optional[str], Any, Any],
        right: Tuple[Optional[str], Any, Any],
    ) -> Optional[Tuple[Optional[str], Any, Any]]:
        left_node, left_gain, left_bias = left
        right_node, right_gain, right_bias = right
        if left_node is not None and right_node is not None and left_node != right_node:
            return None
        return (
            left_node if left_node is not None else right_node,
            self._rust_scalar_add(left_gain, right_gain),
            self._rust_scalar_add(left_bias, right_bias),
        )

    def _rust_scalar_expr(self, expr: Expr) -> Optional[Any]:
        if isinstance(expr, NumberLiteral):
            return float(expr.value)

        if isinstance(expr, Identifier):
            if expr.name in self._evaluate_ir_static_loop_values:
                return float(self._evaluate_ir_static_loop_values[expr.name])
            param_type = self._param_types.get(expr.name)
            if param_type in {ParamType.REAL, ParamType.INTEGER}:
                return ("param", expr.name)
            return None

        if isinstance(expr, UnaryExpr):
            scalar = self._rust_scalar_expr(expr.operand)
            if scalar is None:
                return None
            if expr.op == "-":
                return self._rust_scalar_neg(scalar)
            if expr.op == "+":
                return scalar
            return None

        if isinstance(expr, BinaryExpr):
            left = self._rust_scalar_expr(expr.left)
            right = self._rust_scalar_expr(expr.right)
            if left is None or right is None:
                return None
            if expr.op == "+":
                return self._rust_scalar_add(left, right)
            if expr.op == "-":
                return self._rust_scalar_sub(left, right)
            if expr.op == "*":
                return self._rust_scalar_mul(left, right)
            if expr.op == "/":
                return self._rust_scalar_div(left, right)

        return None

    def _rust_scalar_is_number(self, value: Any) -> bool:
        return isinstance(value, (int, float))

    def _rust_scalar_neg(self, value: Any) -> Any:
        if self._rust_scalar_is_number(value):
            return -float(value)
        return ("neg", value)

    def _rust_scalar_add(self, left: Any, right: Any) -> Any:
        if self._rust_scalar_is_number(left) and self._rust_scalar_is_number(right):
            return float(left) + float(right)
        if self._rust_scalar_is_number(left) and float(left) == 0.0:
            return right
        if self._rust_scalar_is_number(right) and float(right) == 0.0:
            return left
        return ("add", left, right)

    def _rust_scalar_sub(self, left: Any, right: Any) -> Any:
        if self._rust_scalar_is_number(left) and self._rust_scalar_is_number(right):
            return float(left) - float(right)
        if self._rust_scalar_is_number(right) and float(right) == 0.0:
            return left
        return ("sub", left, right)

    def _rust_scalar_mul(self, left: Any, right: Any) -> Any:
        if self._rust_scalar_is_number(left) and self._rust_scalar_is_number(right):
            return float(left) * float(right)
        if self._rust_scalar_is_number(left):
            left_f = float(left)
            if left_f == 0.0:
                return 0.0
            if left_f == 1.0:
                return right
        if self._rust_scalar_is_number(right):
            right_f = float(right)
            if right_f == 0.0:
                return 0.0
            if right_f == 1.0:
                return left
        return ("mul", left, right)

    def _rust_scalar_div(self, left: Any, right: Any) -> Optional[Any]:
        if self._rust_scalar_is_number(right) and float(right) == 0.0:
            return None
        if self._rust_scalar_is_number(left) and self._rust_scalar_is_number(right):
            return float(left) / float(right)
        if self._rust_scalar_is_number(left) and float(left) == 0.0:
            return 0.0
        if self._rust_scalar_is_number(right) and float(right) == 1.0:
            return left
        return ("div", left, right)

    def _dynamic_branch_context(self, in_event_body: bool) -> str:
        return "event_body" if in_event_body else "ordinary"

    def _record_dynamic_branch_access(
        self,
        acc: Dict[str, Any],
        role: str,
        node: str,
        index_expr,
        index_expr2,
        context: str,
    ) -> None:
        dimensions = 2 if index_expr2 is not None else 1
        acc["dynamic_branch_accesses"].add((role, node, dimensions, context))
        self._collect_static_branch_io_from_expr(
            index_expr,
            acc,
            context == "event_body",
        )
        if index_expr2 is not None:
            self._collect_static_branch_io_from_expr(
                index_expr2,
                acc,
                context == "event_body",
            )

    def _record_static_branch_read(
        self,
        acc: Dict[str, Any],
        node: str,
        index_expr,
        index_expr2,
        in_event_body: bool,
    ) -> None:
        if index_expr is not None:
            acc["dynamic_voltage_read_count"] += 1
            self._record_dynamic_branch_access(
                acc,
                "voltage_read",
                node,
                index_expr,
                index_expr2,
                self._dynamic_branch_context(in_event_body),
            )
            return
        target = (
            acc["event_voltage_read_nodes"]
            if in_event_body
            else acc["static_voltage_read_nodes"]
        )
        target.add(node)

    def _collect_static_branch_io_from_branch_read(
        self,
        branch: BranchAccess,
        acc: Dict[str, Any],
        in_event_body: bool,
    ) -> None:
        if branch.access_type != "V":
            return
        self._record_static_branch_read(
            acc,
            branch.node1,
            branch.node1_index,
            branch.node1_index2,
            in_event_body,
        )
        if branch.node2 is not None:
            self._record_static_branch_read(
                acc,
                branch.node2,
                branch.node2_index,
                branch.node2_index2,
                in_event_body,
            )

    def _collect_static_branch_io_from_contribution(
        self,
        stmt: Contribution,
        acc: Dict[str, Any],
        in_event_body: bool,
    ) -> None:
        branch = stmt.branch
        if branch.access_type == "V":
            if branch.node1_index is not None:
                acc["dynamic_output_write_count"] += 1
                self._record_dynamic_branch_access(
                    acc,
                    "output_write",
                    branch.node1,
                    branch.node1_index,
                    branch.node1_index2,
                    self._dynamic_branch_context(in_event_body),
                )
            else:
                acc["static_output_write_nodes"].add(branch.node1)
            if branch.node2 is not None:
                self._record_static_branch_read(
                    acc,
                    branch.node2,
                    branch.node2_index,
                    branch.node2_index2,
                    in_event_body,
                )
        self._collect_static_branch_io_from_expr(stmt.expr, acc, in_event_body)

    def _collect_static_branch_io_from_expr(
        self,
        expr: Expr,
        acc: Dict[str, Any],
        in_event_body: bool,
    ) -> None:
        if isinstance(expr, (NumberLiteral, StringLiteral, Identifier)):
            return

        if isinstance(expr, ArrayAccess):
            self._collect_static_branch_io_from_expr(expr.index, acc, in_event_body)
            return

        if isinstance(expr, BranchAccess):
            self._collect_static_branch_io_from_branch_read(expr, acc, in_event_body)
            return

        if isinstance(expr, BinaryExpr):
            self._collect_static_branch_io_from_expr(expr.left, acc, in_event_body)
            self._collect_static_branch_io_from_expr(expr.right, acc, in_event_body)
            return

        if isinstance(expr, UnaryExpr):
            self._collect_static_branch_io_from_expr(expr.operand, acc, in_event_body)
            return

        if isinstance(expr, TernaryExpr):
            self._collect_static_branch_io_from_expr(expr.cond, acc, in_event_body)
            self._collect_static_branch_io_from_expr(expr.true_expr, acc, in_event_body)
            self._collect_static_branch_io_from_expr(expr.false_expr, acc, in_event_body)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._collect_static_branch_io_from_expr(arg, acc, in_event_body)
            return

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                self._collect_static_branch_io_from_expr(arg, acc, in_event_body)

    def _collect_event_trigger_voltage_nodes_from_expr(
        self,
        expr: Expr,
        acc: Dict[str, Any],
    ) -> None:
        """Record static branch reads used by cross/above trigger expressions.

        These nodes still participate in ordinary evaluate-time expression
        reads, so they remain in ``static_voltage_read_nodes`` as well.  This
        extra set is an IR boundary for future event interpolation/native
        lowering work.
        """
        if isinstance(expr, (NumberLiteral, StringLiteral, Identifier)):
            return

        if isinstance(expr, ArrayAccess):
            self._collect_event_trigger_voltage_nodes_from_expr(expr.index, acc)
            return

        if isinstance(expr, BranchAccess):
            if expr.access_type == "V":
                if expr.node1_index is None:
                    acc["event_trigger_voltage_read_nodes"].add(expr.node1)
                else:
                    self._record_dynamic_branch_access(
                        acc,
                        "voltage_read",
                        expr.node1,
                        expr.node1_index,
                        expr.node1_index2,
                        "event_trigger",
                    )
                if expr.node2 is not None:
                    if expr.node2_index is None:
                        acc["event_trigger_voltage_read_nodes"].add(expr.node2)
                    else:
                        self._record_dynamic_branch_access(
                            acc,
                            "voltage_read",
                            expr.node2,
                            expr.node2_index,
                            expr.node2_index2,
                            "event_trigger",
                        )
            return

        if isinstance(expr, BinaryExpr):
            self._collect_event_trigger_voltage_nodes_from_expr(expr.left, acc)
            self._collect_event_trigger_voltage_nodes_from_expr(expr.right, acc)
            return

        if isinstance(expr, UnaryExpr):
            self._collect_event_trigger_voltage_nodes_from_expr(expr.operand, acc)
            return

        if isinstance(expr, TernaryExpr):
            self._collect_event_trigger_voltage_nodes_from_expr(expr.cond, acc)
            self._collect_event_trigger_voltage_nodes_from_expr(expr.true_expr, acc)
            self._collect_event_trigger_voltage_nodes_from_expr(expr.false_expr, acc)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._collect_event_trigger_voltage_nodes_from_expr(arg, acc)
            return

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                self._collect_event_trigger_voltage_nodes_from_expr(arg, acc)

    def _collect_static_branch_io_from_event(
        self,
        event,
        acc: Dict[str, Any],
        in_event_body: bool,
    ) -> None:
        if isinstance(event, EventExpr):
            if (
                event.event_type in (EventType.CROSS, EventType.ABOVE)
                and event.args
            ):
                self._collect_event_trigger_voltage_nodes_from_expr(event.args[0], acc)
            for arg in event.args:
                self._collect_static_branch_io_from_expr(arg, acc, in_event_body)
            if event.time_tol_expr is not None:
                self._collect_static_branch_io_from_expr(
                    event.time_tol_expr,
                    acc,
                    in_event_body,
                )
            if event.expr_tol_expr is not None:
                self._collect_static_branch_io_from_expr(
                    event.expr_tol_expr,
                    acc,
                    in_event_body,
                )
            return

        if isinstance(event, CombinedEvent):
            for child in event.events:
                self._collect_static_branch_io_from_event(child, acc, in_event_body)

    def _collect_static_branch_io_from_stmt(
        self,
        stmt,
        acc: Dict[str, Any],
        in_event_body: bool,
    ) -> None:
        if stmt is None:
            return

        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_static_branch_io_from_stmt(child, acc, in_event_body)
            return

        if isinstance(stmt, Assignment):
            if isinstance(stmt.target, ArrayAccess):
                self._collect_static_branch_io_from_expr(
                    stmt.target.index,
                    acc,
                    in_event_body,
                )
            self._collect_static_branch_io_from_expr(stmt.value, acc, in_event_body)
            return

        if isinstance(stmt, Contribution):
            self._collect_static_branch_io_from_contribution(stmt, acc, in_event_body)
            return

        if isinstance(stmt, EventStatement):
            self._collect_static_branch_io_from_event(stmt.event, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.body, acc, in_event_body=True)
            return

        if isinstance(stmt, IfStatement):
            self._collect_static_branch_io_from_expr(stmt.cond, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.then_body, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.else_body, acc, in_event_body)
            return

        if isinstance(stmt, ForStatement):
            self._collect_static_branch_io_from_stmt(stmt.init, acc, in_event_body)
            self._collect_static_branch_io_from_expr(stmt.cond, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.update, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.body, acc, in_event_body)
            return

        if isinstance(stmt, WhileStatement):
            self._collect_static_branch_io_from_expr(stmt.cond, acc, in_event_body)
            self._collect_static_branch_io_from_stmt(stmt.body, acc, in_event_body)
            return

        if isinstance(stmt, CaseStatement):
            self._collect_static_branch_io_from_expr(stmt.expr, acc, in_event_body)
            for item in stmt.items:
                for value in item.values:
                    self._collect_static_branch_io_from_expr(value, acc, in_event_body)
                self._collect_static_branch_io_from_stmt(item.body, acc, in_event_body)
            return

        if isinstance(stmt, SystemTask):
            for arg in stmt.args:
                self._collect_static_branch_io_from_expr(arg, acc, in_event_body)

    def _statement_has_dynamic_breakpoints(self, stmt) -> bool:
        if stmt is None:
            return False

        if isinstance(stmt, Block):
            return any(self._statement_has_dynamic_breakpoints(s) for s in stmt.statements)

        if isinstance(stmt, EventStatement):
            if self._event_has_dynamic_breakpoints(stmt.event):
                return True
            return self._statement_has_dynamic_breakpoints(stmt.body)

        if isinstance(stmt, IfStatement):
            return (
                self._expr_has_function_call(stmt.cond, {"transition", "last_crossing"})
                or self._statement_has_dynamic_breakpoints(stmt.then_body)
                or self._statement_has_dynamic_breakpoints(stmt.else_body)
            )

        if isinstance(stmt, WhileStatement):
            return (
                self._expr_has_function_call(stmt.cond, {"transition", "last_crossing"})
                or self._statement_has_dynamic_breakpoints(stmt.body)
            )

        if isinstance(stmt, ForStatement):
            return (
                self._assignment_has_function_call(stmt.init, {"transition", "last_crossing"})
                or self._expr_has_function_call(stmt.cond, {"transition", "last_crossing"})
                or self._assignment_has_function_call(stmt.update, {"transition", "last_crossing"})
                or self._statement_has_dynamic_breakpoints(stmt.body)
            )

        if isinstance(stmt, CaseStatement):
            if self._expr_has_function_call(stmt.expr, {"transition", "last_crossing"}):
                return True
            for item in stmt.items:
                if any(self._expr_has_function_call(v, {"transition", "last_crossing"}) for v in item.values):
                    return True
                if self._statement_has_dynamic_breakpoints(item.body):
                    return True
            return False

        if self._statement_has_function_call(stmt, {"transition", "last_crossing"}):
            return True

        return False

    def _event_has_dynamic_breakpoints(self, event) -> bool:
        if isinstance(event, EventExpr):
            return event.event_type in {
                EventType.CROSS,
                EventType.ABOVE,
                EventType.TIMER,
            }
        if isinstance(event, CombinedEvent):
            return any(self._event_has_dynamic_breakpoints(e) for e in event.events)
        return False

    def _statement_uses_bound_step(self, stmt) -> bool:
        return self._statement_has_system_task(stmt, {"$bound_step"})

    def _statement_has_system_task(self, stmt, names: set[str]) -> bool:
        if stmt is None:
            return False

        if isinstance(stmt, Block):
            return any(self._statement_has_system_task(s, names) for s in stmt.statements)

        if isinstance(stmt, EventStatement):
            return self._statement_has_system_task(stmt.body, names)

        if isinstance(stmt, SystemTask):
            return stmt.name in names

        if isinstance(stmt, IfStatement):
            return (
                self._statement_has_system_task(stmt.then_body, names)
                or self._statement_has_system_task(stmt.else_body, names)
            )

        if isinstance(stmt, WhileStatement):
            return self._statement_has_system_task(stmt.body, names)

        if isinstance(stmt, ForStatement):
            return self._statement_has_system_task(stmt.body, names)

        if isinstance(stmt, CaseStatement):
            return any(
                self._statement_has_system_task(item.body, names)
                for item in stmt.items
            )

        return False

    def _statement_has_function_call(self, stmt, names: set[str]) -> bool:
        if stmt is None:
            return False

        if isinstance(stmt, Block):
            return any(self._statement_has_function_call(s, names) for s in stmt.statements)

        if isinstance(stmt, Contribution):
            return self._expr_has_function_call(stmt.expr, names)

        if isinstance(stmt, Assignment):
            return (
                self._expr_has_function_call(stmt.value, names)
                or self._target_has_function_call(stmt.target, names)
            )

        if isinstance(stmt, EventStatement):
            return (
                self._event_expr_has_function_call(stmt.event, names)
                or self._statement_has_function_call(stmt.body, names)
            )

        if isinstance(stmt, IfStatement):
            return (
                self._expr_has_function_call(stmt.cond, names)
                or self._statement_has_function_call(stmt.then_body, names)
                or self._statement_has_function_call(stmt.else_body, names)
            )

        if isinstance(stmt, WhileStatement):
            return (
                self._expr_has_function_call(stmt.cond, names)
                or self._statement_has_function_call(stmt.body, names)
            )

        if isinstance(stmt, ForStatement):
            return (
                self._assignment_has_function_call(stmt.init, names)
                or self._expr_has_function_call(stmt.cond, names)
                or self._assignment_has_function_call(stmt.update, names)
                or self._statement_has_function_call(stmt.body, names)
            )

        if isinstance(stmt, CaseStatement):
            if self._expr_has_function_call(stmt.expr, names):
                return True
            for item in stmt.items:
                if any(self._expr_has_function_call(v, names) for v in item.values):
                    return True
                if self._statement_has_function_call(item.body, names):
                    return True
            return False

        if isinstance(stmt, SystemTask):
            return any(self._expr_has_function_call(arg, names) for arg in stmt.args)

        return False

    def _assignment_has_function_call(self, stmt, names: set[str]) -> bool:
        if not isinstance(stmt, Assignment):
            return False
        return (
            self._target_has_function_call(stmt.target, names)
            or self._expr_has_function_call(stmt.value, names)
        )

    def _target_has_function_call(self, target, names: set[str]) -> bool:
        if isinstance(target, ArrayAccess):
            return self._expr_has_function_call(target.index, names)
        return False

    def _event_expr_has_function_call(self, event, names: set[str]) -> bool:
        if isinstance(event, EventExpr):
            return any(self._expr_has_function_call(arg, names) for arg in event.args)
        if isinstance(event, CombinedEvent):
            return any(self._event_expr_has_function_call(e, names) for e in event.events)
        return False

    def _expr_has_function_call(self, expr: Expr, names: set[str]) -> bool:
        return any(call.name in names for call in self._iter_function_calls_in_expr(expr))

    def _transition_target_uses_random_state(self, expr: Expr) -> bool:
        if self._expr_has_function_call(expr, {"$rdist_normal"}):
            return True
        return self._expr_references_nodes(expr, self._random_state_names())

    def _random_state_names(self) -> set[str]:
        if self._random_state_names_cache is not None:
            return self._random_state_names_cache
        body = getattr(getattr(self.module, "analog_block", None), "body", None)
        if body is None:
            self._random_state_names_cache = set()
            return self._random_state_names_cache

        assignments = list(self._iter_assignments_in_stmt(body))
        random_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for stmt in assignments:
                if not isinstance(stmt.target, Identifier):
                    continue
                name = stmt.target.name
                value = stmt.value
                if (
                    self._expr_has_function_call(value, {"$rdist_normal"})
                    or self._expr_references_nodes(value, random_names)
                ) and name not in random_names:
                    random_names.add(name)
                    changed = True
        self._random_state_names_cache = random_names
        return random_names

    def _iter_assignments_in_stmt(self, stmt):
        if isinstance(stmt, Assignment):
            yield stmt
            return
        if isinstance(stmt, Block):
            for child in stmt.statements:
                yield from self._iter_assignments_in_stmt(child)
            return
        if isinstance(stmt, IfStatement):
            yield from self._iter_assignments_in_stmt(stmt.then_body)
            if stmt.else_body is not None:
                yield from self._iter_assignments_in_stmt(stmt.else_body)
            return
        if isinstance(stmt, EventStatement):
            yield from self._iter_assignments_in_stmt(stmt.body)
            return
        if isinstance(stmt, ForStatement):
            yield from self._iter_assignments_in_stmt(stmt.init)
            yield from self._iter_assignments_in_stmt(stmt.body)
            yield from self._iter_assignments_in_stmt(stmt.update)
            return
        if isinstance(stmt, WhileStatement):
            yield from self._iter_assignments_in_stmt(stmt.body)
            return
        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                yield from self._iter_assignments_in_stmt(item.body)

    def _expr_references_nodes(self, expr: Expr, nodes: set[str]) -> bool:
        if not nodes:
            return False

        if isinstance(expr, Identifier):
            return expr.name in nodes

        if isinstance(expr, ArrayAccess):
            return expr.name in nodes or self._expr_references_nodes(expr.index, nodes)

        if isinstance(expr, BinaryExpr):
            return (
                self._expr_references_nodes(expr.left, nodes)
                or self._expr_references_nodes(expr.right, nodes)
            )

        if isinstance(expr, UnaryExpr):
            return self._expr_references_nodes(expr.operand, nodes)

        if isinstance(expr, TernaryExpr):
            return (
                self._expr_references_nodes(expr.cond, nodes)
                or self._expr_references_nodes(expr.true_expr, nodes)
                or self._expr_references_nodes(expr.false_expr, nodes)
            )

        if isinstance(expr, FunctionCall):
            return any(self._expr_references_nodes(arg, nodes) for arg in expr.args)

        if isinstance(expr, BranchAccess):
            return (
                expr.node1 in nodes
                or (expr.node2 is not None and expr.node2 in nodes)
                or (expr.node1_index is not None and self._expr_references_nodes(expr.node1_index, nodes))
                or (expr.node1_index2 is not None and self._expr_references_nodes(expr.node1_index2, nodes))
                or (expr.node2_index is not None and self._expr_references_nodes(expr.node2_index, nodes))
                or (expr.node2_index2 is not None and self._expr_references_nodes(expr.node2_index2, nodes))
            )

        if isinstance(expr, MethodCall):
            return any(self._expr_references_nodes(arg, nodes) for arg in expr.args)

        return False

    def _event_requires_post_update(self, event) -> bool:
        if not isinstance(event, EventExpr):
            return False
        if event.event_type not in (EventType.CROSS, EventType.ABOVE):
            return False
        if not hasattr(self, "_contributed_nodes"):
            return False
        return self._expr_references_nodes(event.args[0], self._contributed_nodes)

    def _is_initial_step_event(self, event) -> bool:
        """Check if event includes initial_step."""
        if isinstance(event, EventExpr) and event.event_type == EventType.INITIAL_STEP:
            return True
        if isinstance(event, CombinedEvent):
            return any(e.event_type == EventType.INITIAL_STEP for e in event.events)
        return False

    def _is_final_step_event(self, event) -> bool:
        """Check if event includes final_step."""
        if isinstance(event, EventExpr) and event.event_type == EventType.FINAL_STEP:
            return True
        if isinstance(event, CombinedEvent):
            return any(e.event_type == EventType.FINAL_STEP for e in event.events)
        return False

    def _compile_statement(self, stmt, indent) -> List[str]:
        """Compile a statement to Python code lines."""
        prefix = '    ' * indent
        lines = []

        if isinstance(stmt, Block):
            idx = 0
            stmts = stmt.statements
            while idx < len(stmts):
                if self._is_batchable_timer_event_statement(stmts[idx]):
                    end = idx + 1
                    while (
                        end < len(stmts)
                        and self._is_batchable_timer_event_statement(stmts[end])
                    ):
                        end += 1
                    if end - idx >= 2:
                        lines.extend(
                            self._compile_timer_event_batch_segment(
                                stmts[idx:end],
                                indent,
                            )
                        )
                        idx = end
                        continue
                lines.extend(self._compile_statement(stmts[idx], indent))
                idx += 1

        elif isinstance(stmt, EventStatement):
            lines.extend(self._compile_event_statement(stmt, indent))

        elif isinstance(stmt, Contribution):
            lines.extend(self._compile_contribution(stmt, indent))

        elif isinstance(stmt, Assignment):
            lines.extend(self._compile_assignment(stmt, indent))

        elif isinstance(stmt, IfStatement):
            cond = self._compile_expr(stmt.cond)
            lines.append(f"{prefix}if {cond}:")
            body_lines = self._compile_statement(stmt.then_body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")
            if stmt.else_body:
                lines.append(f"{prefix}else:")
                else_lines = self._compile_statement(stmt.else_body, indent + 1)
                lines.extend(else_lines)
                if not else_lines:
                    lines.append(f"{prefix}    pass")

        elif isinstance(stmt, ForStatement):
            lines.extend(self._compile_for(stmt, indent))

        elif isinstance(stmt, WhileStatement):
            lines.extend(self._compile_while(stmt, indent))

        elif isinstance(stmt, CaseStatement):
            lines.extend(self._compile_case(stmt, indent))

        elif isinstance(stmt, SystemTask):
            # $strobe, $display → collect output
            if stmt.name in ('$strobe', '$display'):
                if stmt.args:
                    fmt_expr = self._compile_expr(stmt.args[0])
                    rest = ', '.join(self._compile_expr(a) for a in stmt.args[1:])
                    if rest:
                        lines.append(f"{prefix}self._strobe(time, {fmt_expr}, {rest})")
                    else:
                        lines.append(f"{prefix}self._strobe(time, {fmt_expr})")
                else:
                    lines.append(f"{prefix}self._strobe(time, '')")
            elif stmt.name == '$bound_step' and stmt.args:
                val = self._compile_expr(stmt.args[0])
                lines.append(f"{prefix}self._bound_step = {val}")
            elif stmt.name == '$fclose' and stmt.args:
                fd = self._compile_expr(stmt.args[0])
                lines.append(f"{prefix}self._fclose(int({fd}))")
            elif stmt.name in ('$fstrobe', '$fwrite', '$fdisplay') and stmt.args:
                fd = self._compile_expr(stmt.args[0])
                if len(stmt.args) > 1:
                    fmt_expr = self._compile_expr(stmt.args[1])
                    rest = ', '.join(self._compile_expr(a) for a in stmt.args[2:])
                    if rest:
                        lines.append(f"{prefix}self._fstrobe({fd}, {fmt_expr}, {rest})")
                    else:
                        lines.append(f"{prefix}self._fstrobe({fd}, {fmt_expr})")
                else:
                    lines.append(f"{prefix}self._fstrobe({fd}, '')")

        return lines

    def _compile_initial_step_statement(self, stmt, indent) -> List[str]:
        """Compile analog statements as seen during the initial_step pass."""
        prefix = '    ' * indent
        lines = []

        if isinstance(stmt, Block):
            for s in stmt.statements:
                lines.extend(self._compile_initial_step_statement(s, indent))

        elif isinstance(stmt, EventStatement):
            if self._is_initial_step_event(stmt.event):
                key = self._alloc_event_key("initial_step", stmt.event)
                lines.append(f"{prefix}self._event_trace_audit_enter_event('initial_step', {key!r}, time)")
                lines.extend(self._compile_initial_step_statement(stmt.body, indent))
                lines.append(f"{prefix}self._event_trace_audit_exit_event()")

        elif isinstance(stmt, Contribution):
            lines.extend(self._compile_contribution(stmt, indent))

        elif isinstance(stmt, Assignment):
            lines.extend(self._compile_assignment(stmt, indent))

        elif isinstance(stmt, IfStatement):
            cond = self._compile_expr(stmt.cond)
            lines.append(f"{prefix}if {cond}:")
            body_lines = self._compile_initial_step_statement(stmt.then_body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")
            if stmt.else_body:
                lines.append(f"{prefix}else:")
                else_lines = self._compile_initial_step_statement(stmt.else_body, indent + 1)
                lines.extend(else_lines)
                if not else_lines:
                    lines.append(f"{prefix}    pass")

        elif isinstance(stmt, WhileStatement):
            cond = self._compile_expr(stmt.cond)
            lines.append(f"{prefix}while {cond}:")
            body_lines = self._compile_initial_step_statement(stmt.body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")

        elif isinstance(stmt, ForStatement):
            loop_var = None
            if isinstance(stmt.init.target, Identifier):
                loop_var = stmt.init.target.name
            elif isinstance(stmt.init.target, ArrayAccess):
                loop_var = stmt.init.target.name
            if loop_var is None:
                return lines

            init_val = self._compile_expr(stmt.init.value)
            prev_loop_var = self._in_loop_var
            self._in_loop_var = loop_var

            lines.append(f"{prefix}self._state_set({loop_var!r}, {init_val})")
            lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
            cond_code = self._compile_expr_with_loop_var(stmt.cond, loop_var)
            lines.append(f"{prefix}while {cond_code}:")
            lines.append(f"{prefix}    self._state_set({loop_var!r}, _loop_{loop_var})")
            body_lines = self._compile_initial_step_statement_with_loop_var(stmt.body, indent + 1, loop_var)
            lines.extend(body_lines)
            update_code = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
            lines.append(f"{prefix}    _loop_{loop_var} = {update_code}")
            lines.append(f"{prefix}self._state_set({loop_var!r}, _loop_{loop_var})")

            self._in_loop_var = prev_loop_var

        elif isinstance(stmt, CaseStatement):
            sel = self._compile_expr(stmt.expr)
            first = True
            default_lines = None
            for item in stmt.items:
                body_lines = self._compile_initial_step_statement(item.body, indent + 1)
                if not body_lines:
                    continue
                if not item.values:
                    default_lines = body_lines
                    continue
                cond_parts = [f"({sel} == {self._compile_expr(v)})" for v in item.values]
                cond = ' or '.join(cond_parts)
                keyword = 'if' if first else 'elif'
                lines.append(f"{prefix}{keyword} {cond}:")
                lines.extend(body_lines)
                first = False
            if default_lines is not None:
                if first:
                    lines.extend(default_lines)
                else:
                    lines.append(f"{prefix}else:")
                    lines.extend(default_lines)

        return lines

    def _event_body_lfsr_shift_ir(self, key: str, body) -> Optional[tuple]:
        if not isinstance(body, Block) or len(body.statements) != 1:
            return None
        guard = body.statements[0]
        if not isinstance(guard, IfStatement) or guard.else_body is not None:
            return None
        gate = self._event_lfsr_gate_condition(guard.cond)
        if gate is None:
            return None
        gate_node, gate_threshold = gate
        guarded = guard.then_body
        if not isinstance(guarded, Block) or len(guarded.statements) != 4:
            return None
        first_loop, feedback_stmt, second_loop, output_stmt = guarded.statements
        first = self._event_lfsr_shift_first_loop(first_loop)
        if first is None:
            return None
        loop_var, final_value, tmp_name, tmp_indices, lfsr_name, lfsr_indices = first
        feedback = self._event_lfsr_feedback_assignment(
            feedback_stmt,
            tmp_name,
            lfsr_name,
            loop_var,
        )
        if feedback is None:
            return None
        feedback_tmp_index, tap_indices = feedback
        second = self._event_lfsr_shift_second_loop(
            second_loop,
            loop_var,
            tmp_name,
            lfsr_name,
        )
        if second is None:
            return None
        second_final, second_lfsr_indices, second_tmp_indices = second
        output = self._event_lfsr_output_assignment(output_stmt, lfsr_name)
        if output is None:
            return None
        output_state, high_node, low_node, output_index = output
        width = len(second_lfsr_indices)
        if (
            width == 0
            or tuple(second_lfsr_indices) != tuple(range(width))
            or tuple(second_tmp_indices) != tuple(range(width))
            or tuple(lfsr_indices) != tuple(range(width))
            or tuple(tmp_indices) != tuple(range(1, width + 1))
            or int(feedback_tmp_index) != 0
            or int(output_index) != width - 1
        ):
            return None
        lfsr_refs = tuple((lfsr_name, idx) for idx in second_lfsr_indices)
        tmp_refs = tuple((tmp_name, idx) for idx in range(0, width + 1))
        tap_refs = tuple((lfsr_name, idx) for idx in tap_indices)
        return (
            key,
            lfsr_refs,
            tmp_refs,
            tap_refs,
            gate_node,
            float(gate_threshold),
            high_node,
            low_node,
            output_state,
            loop_var,
            float(second_final if second_final is not None else final_value),
        )

    def _event_body_static_linear_ir(self, key: str, body) -> Optional[tuple]:
        ops: List[tuple] = []
        if not self._collect_event_body_static_linear_ops_from_stmt(body, ops):
            return None
        if not ops:
            return None
        return key, tuple(ops)

    def _collect_event_body_static_linear_ops_from_stmt(
        self,
        stmt,
        ops: List[tuple],
    ) -> bool:
        """Collect ordered event-body state writes that can run as Rust LinearOp.

        Event bodies are discrete, so self-dependent updates such as
        ``count = count + 1`` are safe to lower here even though continuous
        evaluate-path static IR rejects them.
        """

        if stmt is None:
            return True

        if isinstance(stmt, Block):
            for child in stmt.statements:
                if not self._collect_event_body_static_linear_ops_from_stmt(child, ops):
                    return False
            return True

        if isinstance(stmt, EventStatement):
            return False

        if isinstance(stmt, IfStatement):
            condition = self._evaluate_ir_condition_expr(stmt.cond)
            if condition is None or stmt.else_body is None:
                return False
            then_ops: List[tuple] = []
            else_ops: List[tuple] = []
            if not self._collect_event_body_static_linear_ops_from_stmt(
                stmt.then_body,
                then_ops,
            ):
                return False
            if not self._collect_event_body_static_linear_ops_from_stmt(
                stmt.else_body,
                else_ops,
            ):
                return False
            if len(then_ops) != len(else_ops):
                return False
            for then_op, else_op in zip(then_ops, else_ops):
                combined = self._combine_evaluate_ir_conditional_ops(
                    condition,
                    then_op,
                    else_op,
                )
                if combined is None:
                    return False
                ops.append(combined)
            return True

        if isinstance(stmt, ForStatement):
            loop = self._evaluate_ir_static_for_values(stmt)
            if loop is None:
                return False
            loop_var, values, final_value = loop
            local_ops: List[tuple] = []
            sentinel = object()
            old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
            try:
                for value in values:
                    self._evaluate_ir_static_loop_values[loop_var] = value
                    if not self._collect_event_body_static_linear_ops_from_stmt(
                        stmt.body,
                        local_ops,
                    ):
                        return False
            finally:
                if old_value is sentinel:
                    self._evaluate_ir_static_loop_values.pop(loop_var, None)
                else:
                    self._evaluate_ir_static_loop_values[loop_var] = old_value
            self._append_evaluate_ir_static_for_final_state_op(
                loop_var,
                final_value,
                local_ops,
            )
            ops.extend(local_ops)
            return True

        if isinstance(stmt, Assignment):
            target = None
            target_integer = False
            if isinstance(stmt.target, Identifier):
                target = stmt.target.name
                if target not in self._state_scalar_slot_by_name:
                    return False
                target_integer = self._is_integer_variable(target)
            elif isinstance(stmt.target, ArrayAccess):
                array_slot = self._evaluate_ir_static_array_slot(stmt.target)
                if array_slot is None:
                    return False
                target, target_integer = array_slot
            else:
                return False
            linear = self._evaluate_ir_linear_expr(stmt.value)
            if linear is None:
                return False
            unpacked = self._evaluate_ir_unpack_linear(linear)
            if unpacked is None:
                return False
            bias, terms, condition, false_bias, false_terms = unpacked
            ops.append(
                (
                    TARGET_STATE,
                    target,
                    bias,
                    tuple(terms),
                    condition,
                    false_bias,
                    tuple(false_terms),
                    target_integer,
                )
            )
            return True

        return False

    @staticmethod
    def _event_body_static_linear_target_names(raw_ops: Tuple[tuple, ...]) -> Tuple[str, ...]:
        targets = []
        seen = set()
        for raw in raw_ops:
            if len(raw) >= 2 and raw[0] == TARGET_STATE:
                name = str(raw[1])
                if name not in seen:
                    seen.add(name)
                    targets.append(name)
        return tuple(targets)

    @staticmethod
    def _event_body_static_linear_state_names(raw_ops: Tuple[tuple, ...]) -> Tuple[str, ...]:
        names = []
        seen = set()

        def add(name: str) -> None:
            key = str(name)
            if key not in seen:
                seen.add(key)
                names.append(key)

        def add_terms(raw_terms) -> None:
            for term in raw_terms or ():
                try:
                    source_kind, source_name, _gain = term
                except (TypeError, ValueError):
                    continue
                if source_kind == SOURCE_STATE:
                    add(str(source_name))

        for raw in raw_ops:
            if len(raw) < 4:
                continue
            if raw[0] == TARGET_STATE:
                add(str(raw[1]))
            add_terms(raw[3])
            if len(raw) >= 7:
                condition = raw[4]
                if condition is not None:
                    try:
                        _op, _left_bias, left_terms, _right_bias, right_terms = condition
                    except (TypeError, ValueError):
                        left_terms = ()
                        right_terms = ()
                    add_terms(left_terms)
                    add_terms(right_terms)
                add_terms(raw[6])
        return tuple(names)

    def _event_lfsr_gate_condition(self, expr) -> Optional[Tuple[str, float]]:
        if not isinstance(expr, BinaryExpr) or expr.op not in {">", ">="}:
            return None
        if not (
            isinstance(expr.left, BranchAccess)
            and expr.left.access_type == "V"
            and expr.left.node2 is None
            and expr.left.node1_index is None
            and expr.left.node1_index2 is None
        ):
            return None
        threshold = self._rust_scalar_expr(expr.right)
        if threshold is None:
            return None
        try:
            return expr.left.node1, float(threshold)
        except (TypeError, ValueError):
            return None

    def _event_lfsr_shift_first_loop(self, stmt) -> Optional[tuple]:
        if not isinstance(stmt, ForStatement):
            return None
        loop = self._evaluate_ir_static_for_values(stmt)
        if loop is None:
            return None
        loop_var, values, final_value = loop
        if not isinstance(stmt.body, Block) or len(stmt.body.statements) != 1:
            return None
        assign = stmt.body.statements[0]
        if not isinstance(assign, Assignment):
            return None
        if not isinstance(assign.target, ArrayAccess) or not isinstance(assign.value, ArrayAccess):
            return None
        tmp_name = assign.target.name
        lfsr_name = assign.value.name
        tmp_indices = self._event_lfsr_indices_for_loop(assign.target.index, loop_var, values)
        lfsr_indices = self._event_lfsr_indices_for_loop(assign.value.index, loop_var, values)
        if tmp_indices is None or lfsr_indices is None:
            return None
        return loop_var, final_value, tmp_name, tmp_indices, lfsr_name, lfsr_indices

    def _event_lfsr_shift_second_loop(
        self,
        stmt,
        loop_var: str,
        tmp_name: str,
        lfsr_name: str,
    ) -> Optional[tuple]:
        if not isinstance(stmt, ForStatement):
            return None
        loop = self._evaluate_ir_static_for_values(stmt)
        if loop is None:
            return None
        loop_var2, values, final_value = loop
        if loop_var2 != loop_var:
            return None
        if not isinstance(stmt.body, Block) or len(stmt.body.statements) != 1:
            return None
        assign = stmt.body.statements[0]
        if not isinstance(assign, Assignment):
            return None
        if not isinstance(assign.target, ArrayAccess) or not isinstance(assign.value, ArrayAccess):
            return None
        if assign.target.name != lfsr_name or assign.value.name != tmp_name:
            return None
        lfsr_indices = self._event_lfsr_indices_for_loop(assign.target.index, loop_var, values)
        tmp_indices = self._event_lfsr_indices_for_loop(assign.value.index, loop_var, values)
        if lfsr_indices is None or tmp_indices is None:
            return None
        return final_value, lfsr_indices, tmp_indices

    def _event_lfsr_indices_for_loop(
        self,
        expr,
        loop_var: str,
        values: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        indices: List[int] = []
        sentinel = object()
        old_value = self._evaluate_ir_static_loop_values.get(loop_var, sentinel)
        try:
            for value in values:
                self._evaluate_ir_static_loop_values[loop_var] = value
                idx = self._evaluate_ir_static_array_index(expr)
                if idx is None:
                    return None
                indices.append(idx)
        finally:
            if old_value is sentinel:
                self._evaluate_ir_static_loop_values.pop(loop_var, None)
            else:
                self._evaluate_ir_static_loop_values[loop_var] = old_value
        return tuple(indices)

    def _event_lfsr_feedback_assignment(
        self,
        stmt,
        tmp_name: str,
        lfsr_name: str,
        loop_var: str,
    ) -> Optional[Tuple[int, Tuple[int, ...]]]:
        if not isinstance(stmt, Assignment) or not isinstance(stmt.target, ArrayAccess):
            return None
        if stmt.target.name != tmp_name:
            return None
        idx = self._evaluate_ir_static_array_index(stmt.target.index)
        if idx is None:
            return None
        taps = self._event_lfsr_xor_taps(stmt.value, lfsr_name, loop_var)
        if taps is None or not taps:
            return None
        return idx, taps

    def _event_lfsr_xor_taps(
        self,
        expr,
        lfsr_name: str,
        loop_var: str,
    ) -> Optional[Tuple[int, ...]]:
        if isinstance(expr, ArrayAccess) and expr.name == lfsr_name:
            idx = self._evaluate_ir_static_array_index(expr.index)
            if idx is None:
                return None
            return (idx,)
        if isinstance(expr, BinaryExpr) and expr.op == "^":
            left = self._event_lfsr_xor_taps(expr.left, lfsr_name, loop_var)
            right = self._event_lfsr_xor_taps(expr.right, lfsr_name, loop_var)
            if left is None or right is None:
                return None
            return (*left, *right)
        return None

    def _event_lfsr_output_assignment(self, stmt, lfsr_name: str) -> Optional[Tuple[str, str, str, int]]:
        if not isinstance(stmt, Assignment) or not isinstance(stmt.target, Identifier):
            return None
        value = stmt.value
        if not isinstance(value, TernaryExpr):
            return None
        if not isinstance(value.cond, BinaryExpr) or value.cond.op not in {">", ">="}:
            return None
        if not isinstance(value.cond.left, ArrayAccess) or value.cond.left.name != lfsr_name:
            return None
        output_index = self._evaluate_ir_static_array_index(value.cond.left.index)
        if output_index is None:
            return None
        if not (
            isinstance(value.true_expr, BranchAccess)
            and isinstance(value.false_expr, BranchAccess)
            and value.true_expr.access_type == "V"
            and value.false_expr.access_type == "V"
            and value.true_expr.node2 is None
            and value.false_expr.node2 is None
        ):
            return None
        return stmt.target.name, value.true_expr.node1, value.false_expr.node1, output_index

    def _compile_event_body_with_rust_write(
        self,
        key: str,
        body,
        indent: int,
    ) -> List[str]:
        prefix = '    ' * indent
        lines = []
        shadow_name = f"_rust_event_write_shadow_{key}"
        lines.append(f"{prefix}{shadow_name} = self._rust_event_write_shadow_begin({key!r}, nv)")
        lines.append(f"{prefix}if self._rust_event_write_production({key!r}, nv):")
        lines.append(f"{prefix}    pass")
        lines.append(f"{prefix}else:")
        body_lines = self._compile_statement(body, indent + 1)
        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{prefix}    pass")
        lines.append(f"{prefix}    self._rust_event_write_shadow_end({key!r}, {shadow_name})")
        return lines

    def _compile_event_body_with_rust_linear_write(
        self,
        key: str,
        body,
        indent: int,
        target_names: Tuple[str, ...],
        state_names: Tuple[str, ...],
    ) -> List[str]:
        prefix = '    ' * indent
        lines = []
        shadow_name = f"_rust_event_linear_write_shadow_{key}"
        lines.extend(self._compile_event_linear_local_flush(state_names, indent))
        lines.append(f"{prefix}{shadow_name} = self._rust_event_linear_write_shadow_begin({key!r})")
        lines.append(f"{prefix}if self._rust_event_linear_write_production({key!r}):")
        refresh_lines = self._compile_event_linear_local_refresh(target_names, indent + 1)
        if refresh_lines:
            lines.extend(refresh_lines)
        else:
            lines.append(f"{prefix}    pass")
        lines.append(f"{prefix}else:")
        body_lines = self._compile_statement(body, indent + 1)
        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f"{prefix}    pass")
        lines.extend(self._compile_event_linear_local_flush(target_names, indent + 1))
        lines.append(f"{prefix}    self._rust_event_linear_write_shadow_end({key!r}, {shadow_name})")
        return lines

    def _compile_event_linear_local_flush(
        self,
        state_names: Tuple[str, ...],
        indent: int,
    ) -> List[str]:
        if not self._state_local_fastpath_active:
            return []
        prefix = '    ' * indent
        lines = []
        flushed = False
        for name in state_names:
            if name not in self._state_local_fastpath_names:
                continue
            slot = self._state_scalar_slot_by_name.get(name)
            if slot is None:
                continue
            if not flushed:
                lines.append(f"{prefix}_event_linear_state_values = self._indexed_state_values")
                lines.append(f"{prefix}if _event_linear_state_values is not None:")
                flushed = True
            local = self._state_local_name_by_state[name]
            value_expr = local
            if self._is_integer_variable(name):
                value_expr = f"self._to_integer({local})"
            lines.append(f"{prefix}    _event_linear_state_values[{slot}] = float({value_expr})")
        return lines

    def _compile_event_linear_local_refresh(
        self,
        target_names: Tuple[str, ...],
        indent: int,
    ) -> List[str]:
        if not self._state_local_fastpath_active:
            return []
        prefix = '    ' * indent
        lines = []
        for name in target_names:
            if name not in self._state_local_fastpath_names:
                continue
            slot = self._state_scalar_slot_by_name.get(name)
            if slot is None:
                continue
            local = self._state_local_name_by_state[name]
            lines.append(f"{prefix}{local} = self._state_get_by_slot({slot}, {name!r})")
        return lines

    def _compile_event_body_with_best_rust_write(
        self,
        key: str,
        body,
        indent: int,
    ) -> List[str]:
        event_ir = self._event_body_lfsr_shift_ir(key, body)
        if event_ir is not None:
            self._event_lfsr_shift_ir_ops[key] = event_ir
            return self._compile_event_body_with_rust_write(key, body, indent)

        linear_ir = self._event_body_static_linear_ir(key, body)
        if linear_ir is not None:
            _linear_key, raw_ops = linear_ir
            self._event_static_linear_ir_ops[key] = linear_ir
            return self._compile_event_body_with_rust_linear_write(
                key,
                body,
                indent,
                self._event_body_static_linear_target_names(tuple(raw_ops)),
                self._event_body_static_linear_state_names(tuple(raw_ops)),
            )

        return self._compile_statement(body, indent)

    def _timer_expr_is_constant_or_param(self, expr: Expr) -> bool:
        if isinstance(expr, NumberLiteral):
            return True
        if isinstance(expr, Identifier):
            return expr.name == "inf" or expr.name in self._param_types
        if isinstance(expr, UnaryExpr):
            return expr.op in {"+", "-"} and self._timer_expr_is_constant_or_param(expr.operand)
        if isinstance(expr, BinaryExpr):
            return (
                expr.op in {"+", "-", "*", "/", "%"}
                and self._timer_expr_is_constant_or_param(expr.left)
                and self._timer_expr_is_constant_or_param(expr.right)
            )
        if isinstance(expr, FunctionCall):
            name = expr.name[1:] if expr.name.startswith("$") else expr.name
            if name not in {
                "abs",
                "sqrt",
                "exp",
                "ln",
                "log",
                "sin",
                "cos",
                "tan",
                "tanh",
                "floor",
                "ceil",
                "min",
                "max",
                "pow",
            }:
                return False
            return all(self._timer_expr_is_constant_or_param(arg) for arg in expr.args)
        return False

    def _is_batchable_timer_event_statement(self, stmt) -> bool:
        if not isinstance(stmt, EventStatement):
            return False
        event = stmt.event
        if not isinstance(event, EventExpr) or event.event_type != EventType.TIMER:
            return False
        if len(event.args) == 2 and self._event_body_lfsr_shift_ir("__batch_probe__", stmt.body) is not None:
            return False
        return all(self._timer_expr_is_constant_or_param(arg) for arg in event.args)

    def _state_owned_timer_target_name(self, stmt: EventStatement) -> Optional[str]:
        """Return the scalar state name for a safe ``timer(state)`` fast path."""
        event = stmt.event
        if not isinstance(event, EventExpr) or event.event_type != EventType.TIMER:
            return None
        if len(event.args) != 1 or not isinstance(event.args[0], Identifier):
            return None
        name = event.args[0].name
        if name not in self._state_scalar_slot_by_name:
            return None
        if self._is_integer_variable(name):
            return None
        if not self._statement_assigns_scalar_name(stmt.body, name):
            return None
        analog_body = getattr(getattr(self.module, "analog_block", None), "body", None)
        if analog_body is None:
            return None
        if self._state_timer_target_assigned_outside_owner(analog_body, name, stmt):
            return None
        return name

    def _statement_assigns_scalar_name(self, stmt, name: str) -> bool:
        if stmt is None:
            return False
        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier) and target.name == name:
                return True
            if isinstance(target, ArrayAccess) and target.name == name:
                return True
            return False
        if isinstance(stmt, Block):
            return any(
                self._statement_assigns_scalar_name(child, name)
                for child in stmt.statements
            )
        if isinstance(stmt, EventStatement):
            return self._statement_assigns_scalar_name(stmt.body, name)
        if isinstance(stmt, IfStatement):
            return (
                self._statement_assigns_scalar_name(stmt.then_body, name)
                or self._statement_assigns_scalar_name(stmt.else_body, name)
            )
        if isinstance(stmt, ForStatement):
            return (
                self._statement_assigns_scalar_name(stmt.init, name)
                or self._statement_assigns_scalar_name(stmt.update, name)
                or self._statement_assigns_scalar_name(stmt.body, name)
            )
        if isinstance(stmt, WhileStatement):
            return self._statement_assigns_scalar_name(stmt.body, name)
        if isinstance(stmt, CaseStatement):
            return any(
                self._statement_assigns_scalar_name(item.body, name)
                for item in stmt.items
            )
        return False

    def _is_plain_initial_step_statement(self, stmt) -> bool:
        event = getattr(stmt, "event", None)
        return (
            isinstance(stmt, EventStatement)
            and isinstance(event, EventExpr)
            and event.event_type == EventType.INITIAL_STEP
        )

    def _state_timer_target_assigned_outside_owner(
        self,
        stmt,
        name: str,
        owner: EventStatement,
    ) -> bool:
        if stmt is None or stmt is owner:
            return False
        if isinstance(stmt, EventStatement):
            if self._is_plain_initial_step_statement(stmt):
                return False
            return self._statement_assigns_scalar_name(stmt.body, name)
        if isinstance(stmt, Assignment):
            return self._statement_assigns_scalar_name(stmt, name)
        if isinstance(stmt, Block):
            return any(
                self._state_timer_target_assigned_outside_owner(child, name, owner)
                for child in stmt.statements
            )
        if isinstance(stmt, IfStatement):
            return (
                self._state_timer_target_assigned_outside_owner(stmt.then_body, name, owner)
                or self._state_timer_target_assigned_outside_owner(stmt.else_body, name, owner)
            )
        if isinstance(stmt, ForStatement):
            return (
                self._state_timer_target_assigned_outside_owner(stmt.init, name, owner)
                or self._state_timer_target_assigned_outside_owner(stmt.update, name, owner)
                or self._state_timer_target_assigned_outside_owner(stmt.body, name, owner)
            )
        if isinstance(stmt, WhileStatement):
            return self._state_timer_target_assigned_outside_owner(stmt.body, name, owner)
        if isinstance(stmt, CaseStatement):
            return any(
                self._state_timer_target_assigned_outside_owner(item.body, name, owner)
                for item in stmt.items
            )
        return False

    def _compile_timer_event_due_body(
        self,
        key: str,
        body,
        trace_kind: str,
        indent: int,
        post_body_lines: Tuple[str, ...] = (),
    ) -> List[str]:
        prefix = '    ' * indent
        lines = [
            f"{prefix}self._event_interpolated_nodes = set()",
            f"{prefix}self._event_interpolated_node_values = {{}}",
            f"{prefix}self._event_node_cross_directions = {{}}",
            f"{prefix}self._event_trace_audit_enter_event({trace_kind!r}, {key!r}, time)",
            f"{prefix}self._event_context_active = True",
        ]
        body_lines = self._compile_event_body_with_best_rust_write(key, body, indent)
        lines.extend(body_lines)
        if not body_lines:
            lines.append(f"{prefix}pass")
        lines.append(f"{prefix}self._event_trace_audit_exit_event()")
        lines.append(f"{prefix}self._event_context_active = False")
        lines.append(f"{prefix}self._event_interpolated_nodes = set()")
        lines.append(f"{prefix}self._event_interpolated_node_values = {{}}")
        lines.append(f"{prefix}self._event_node_cross_directions = {{}}")
        lines.extend(post_body_lines)
        lines.append(f"{prefix}self._event_time = time")
        return lines

    def _compile_timer_event_batch_segment(
        self,
        statements: List[EventStatement],
        indent: int,
    ) -> List[str]:
        prefix = '    ' * indent
        specs = []
        entries = []
        for stmt in statements:
            event = stmt.event
            key = self._alloc_event_key("timer", event)
            if len(event.args) == 2:
                self._record_timer_static_linear_segment_ir(key, event, stmt.body)
                start_expr = self._compile_expr(event.args[0])
                period_expr = self._compile_expr(event.args[1])
                specs.append(f"('periodic', {key!r}, {period_expr}, {start_expr})")
                entries.append((stmt, key, "timer_periodic", period_expr, None))
            else:
                target_expr = self._compile_expr(event.args[0])
                specs.append(f"('absolute', {key!r}, {target_expr})")
                entries.append((stmt, key, "timer_absolute", None, target_expr))

        lines = [
            f"{prefix}_timer_batch_hits = self._check_timer_event_batch(({', '.join(specs)}), time)"
        ]
        for idx, (stmt, key, trace_kind, period_expr, target_expr) in enumerate(entries):
            lines.append(f"{prefix}if _timer_batch_hits[{idx}]:")
            post_lines: Tuple[str, ...]
            if period_expr is not None:
                post_lines = (
                    f"{prefix}    self._reschedule_timer({key!r}, time, {period_expr})",
                )
            else:
                post_lines = (
                    f"{prefix}    self._set_timer_state({key!r}, {target_expr})",
                )
            lines.extend(
                self._compile_timer_event_due_body(
                    key,
                    stmt.body,
                    trace_kind,
                    indent + 1,
                    post_lines,
                )
            )
        return lines

    def _compile_event_statement(self, stmt: EventStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []
        event = stmt.event

        if isinstance(event, EventExpr):
            if event.event_type == EventType.INITIAL_STEP:
                # Skip in evaluate — handled in initial_step method
                return []

            elif event.event_type == EventType.FINAL_STEP:
                # Skip in evaluate — handled in final_step method
                return []

            elif event.event_type == EventType.CROSS:
                if self._event_requires_post_update(event):
                    return []
                key = self._alloc_event_key("cross", event)
                lines.append(f"{prefix}if {self._compile_cross_call(event, key)}:")
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('cross', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.ABOVE:
                if self._event_requires_post_update(event):
                    return []
                key = self._alloc_event_key("above", event)
                lines.append(f"{prefix}if {self._compile_above_call(event, key)}:")
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('above', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.TIMER:
                key = self._alloc_event_key("timer", event)
                if len(event.args) == 2:
                    self._record_timer_static_linear_segment_ir(key, event, stmt.body)
                    start_expr = self._compile_expr(event.args[0])
                    period_expr = self._compile_expr(event.args[1])
                    event_ir = self._event_body_lfsr_shift_ir(key, stmt.body)
                    if event_ir is not None:
                        self._event_lfsr_shift_ir_ops[key] = event_ir
                        lines.append(
                            f"{prefix}if self._rust_timer_lfsr_output_production("
                            f"{key!r}, nv, time, {period_expr}, {start_expr}):"
                        )
                        lines.append(f"{prefix}    pass")
                        lines.append(f"{prefix}elif self._check_timer_due({key!r}, time, {period_expr}, {start_expr}):")
                    else:
                        lines.append(f"{prefix}if self._check_timer_due({key!r}, time, {period_expr}, {start_expr}):")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._event_trace_audit_enter_event('timer_periodic', {key!r}, time)")
                    lines.append(f"{prefix}    self._event_context_active = True")
                    body_lines = self._compile_event_body_with_best_rust_write(
                        key,
                        stmt.body,
                        indent + 1,
                    )
                    lines.extend(body_lines)
                    if not body_lines:
                        lines.append(f"{prefix}    pass")
                    lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                    lines.append(f"{prefix}    self._event_context_active = False")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._reschedule_timer({key!r}, time, {period_expr})")
                else:
                    target_expr = self._compile_expr(event.args[0])
                    state_owned_target = self._state_owned_timer_target_name(stmt)
                    if state_owned_target is not None:
                        slot = self._state_scalar_slot_by_name.get(state_owned_target, -1)
                        self._state_owned_timer_targets[key] = state_owned_target
                        target_var = f"_state_timer_target_{key}"
                        last_var = f"_state_timer_last_{key}"
                        hit_var = f"_state_timer_hit_{key}"
                        lines.append(f"{prefix}{target_var} = self.timer_states.get({key!r})")
                        lines.append(f"{prefix}{last_var} = self.timer_last_fired.get({key!r})")
                        lines.append(
                            f"{prefix}if ({target_var} is not None "
                            f"and ({last_var} is None or abs({last_var} - {target_var}) > 1e-18) "
                            f"and time < {target_var} - 1e-18 "
                            f"and not self._rust_timer_event_production_enabled "
                            f"and self._rust_event_due_shadow_backend is None):"
                        )
                        lines.append(f"{prefix}    self._perf_stats['timer_state_owned_checks'] += 1")
                        lines.append(f"{prefix}    self._perf_stats['timer_absolute_checks'] += 1")
                        lines.append(f"{prefix}    self._perf_stats['timer_state_owned_fast_skips'] += 1")
                        lines.append(f"{prefix}    {hit_var} = False")
                        lines.append(f"{prefix}else:")
                        lines.append(
                            f"{prefix}    {hit_var} = self._check_state_owned_timer_at({key!r}, "
                            f"time, {state_owned_target!r}, {slot})"
                        )
                        lines.append(f"{prefix}if {hit_var}:")
                    else:
                        lines.append(f"{prefix}if self._check_timer_at({key!r}, time, {target_expr}):")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._event_trace_audit_enter_event('timer_absolute', {key!r}, time)")
                    lines.append(f"{prefix}    self._event_context_active = True")
                    body_lines = self._compile_event_body_with_best_rust_write(
                        key,
                        stmt.body,
                        indent + 1,
                    )
                    lines.extend(body_lines)
                    if not body_lines:
                        lines.append(f"{prefix}    pass")
                    lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                    lines.append(f"{prefix}    self._event_context_active = False")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._set_timer_state({key!r}, {target_expr})")
                lines.append(f"{prefix}    self._event_time = time")

        elif isinstance(event, CombinedEvent):
            # Combined events: @(initial_step or cross(...))
            conditions = []
            for e in event.events:
                if e.event_type == EventType.INITIAL_STEP:
                    # In evaluate, initial_step never fires again
                    continue
                elif e.event_type == EventType.FINAL_STEP:
                    # In evaluate, final_step never fires
                    continue
                elif e.event_type == EventType.CROSS:
                    if self._event_requires_post_update(e):
                        continue
                    key = self._alloc_event_key("cross", e)
                    conditions.append(self._compile_cross_call(e, key))
                elif e.event_type == EventType.ABOVE:
                    if self._event_requires_post_update(e):
                        continue
                    key = self._alloc_event_key("above", e)
                    conditions.append(self._compile_above_call(e, key))
                elif e.event_type == EventType.TIMER:
                    key = self._alloc_event_key("timer", e)
                    if len(e.args) == 2:
                        start_expr = self._compile_expr(e.args[0])
                        period_expr = self._compile_expr(e.args[1])
                        conditions.append(f"self._check_timer({key!r}, time, {period_expr}, {start_expr})")
                    else:
                        target_expr = self._compile_expr(e.args[0])
                        conditions.append(f"self._check_timer_at({key!r}, time, {target_expr})")

            if conditions:
                hit_vars = []
                for idx, cond in enumerate(conditions):
                    var = f"_event_hit_{idx}"
                    hit_vars.append(var)
                    lines.append(f"{prefix}{var} = {cond}")
                cond = ' or '.join(hit_vars)
                lines.append(f"{prefix}if {cond}:")
                key = self._alloc_event_key("combined", event)
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('combined', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

        return lines

    def _record_timer_static_linear_segment_ir(
        self,
        key: str,
        event: EventExpr,
        body,
    ) -> None:
        """Record timer+linear-event metadata for whole-segment Rust lowering."""
        if len(event.args) != 2:
            return
        start = self._rust_scalar_expr(event.args[0])
        period = self._rust_scalar_expr(event.args[1])
        if start is None or period is None:
            return
        linear_ir = self._event_body_static_linear_ir(key, body)
        if linear_ir is None:
            return
        _linear_key, raw_ops = linear_ir
        self._event_timer_static_linear_ir_ops[key] = (
            key,
            start,
            period,
            tuple(raw_ops),
        )

    def _alloc_event_key(self, kind: str, event) -> str:
        cache_key = (kind, id(event))
        if cache_key in self._event_key_cache:
            return self._event_key_cache[cache_key]
        if kind == "cross":
            key = f"cross_{self._cross_counter}"
            self._cross_counter += 1
        elif kind == "above":
            key = f"above_{self._above_counter}"
            self._above_counter += 1
        elif kind == "timer":
            key = f"timer_{self._timer_counter}"
            self._timer_counter += 1
        elif kind == "combined":
            key = f"combined_{self._combined_counter}"
            self._combined_counter += 1
        elif kind == "initial_step":
            key = f"initial_step_{self._initial_step_counter}"
            self._initial_step_counter += 1
        elif kind == "final_step":
            key = f"final_step_{self._final_step_counter}"
            self._final_step_counter += 1
        else:
            raise ValueError(f"unknown event key kind: {kind}")
        self._event_key_cache[cache_key] = key
        return key

    def _compile_cross_call(self, event: EventExpr, key: str) -> str:
        expr = self._compile_expr(event.args[0])
        direction = event.direction if event.direction is not None else 0
        time_tol = "0.0"
        expr_tol = "1e-12"
        if event.time_tol_expr is not None:
            time_tol = self._compile_expr(event.time_tol_expr)
        if event.expr_tol_expr is not None:
            expr_tol = self._compile_expr(event.expr_tol_expr)
        interp_nodes = sorted(self._collect_branch_nodes_from_expr(event.args[0]))
        raw_nudges = self._collect_branch_nudge_nodes_from_expr(event.args[0])
        nudge_nodes = {node: raw_nudges[node] for node in sorted(raw_nudges)}
        if nudge_nodes:
            self._needs_future_node_voltages = True
        return (
            f"self._check_cross({key!r}, time, {expr}, {direction}, "
            f"{time_tol}, {expr_tol}, {interp_nodes!r}, {nudge_nodes!r})"
        )

    def _compile_above_call(self, event: EventExpr, key: str) -> str:
        expr = self._compile_expr(event.args[0])
        direction = event.direction if event.direction is not None else 1
        interp_nodes = sorted(self._collect_branch_nodes_from_expr(event.args[0]))
        return (
            f"self._check_above({key!r}, time, {expr}, {direction}, "
            f"{interp_nodes!r})"
        )

    def _collect_branch_nodes_from_expr(self, expr: Expr) -> set[str]:
        nodes: set[str] = set()
        if isinstance(expr, BranchAccess):
            nodes.add(expr.node1)
            if expr.node2:
                nodes.add(expr.node2)
            return nodes
        if isinstance(expr, BinaryExpr):
            return self._collect_branch_nodes_from_expr(expr.left) | self._collect_branch_nodes_from_expr(expr.right)
        if isinstance(expr, UnaryExpr):
            return self._collect_branch_nodes_from_expr(expr.operand)
        if isinstance(expr, TernaryExpr):
            return (
                self._collect_branch_nodes_from_expr(expr.cond)
                | self._collect_branch_nodes_from_expr(expr.true_expr)
                | self._collect_branch_nodes_from_expr(expr.false_expr)
            )
        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                nodes |= self._collect_branch_nodes_from_expr(arg)
            return nodes
        if isinstance(expr, ArrayAccess):
            return self._collect_branch_nodes_from_expr(expr.index)
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                nodes |= self._collect_branch_nodes_from_expr(arg)
            return nodes
        return nodes

    def _collect_branch_nudge_nodes_from_expr(self, expr: Expr, polarity: int = 1) -> Dict[str, int]:
        nodes: Dict[str, int] = {}

        def add(node: Optional[str], sign: int) -> None:
            if not node:
                return
            nodes[node] = nodes.get(node, 0) + sign

        def merge(other: Dict[str, int]) -> None:
            for node, sign in other.items():
                add(node, sign)

        if isinstance(expr, BranchAccess):
            add(expr.node1, polarity)
            add(expr.node2, -polarity)
            return {node: 1 if sign > 0 else -1 for node, sign in nodes.items() if sign}
        if isinstance(expr, BinaryExpr):
            merge(self._collect_branch_nudge_nodes_from_expr(expr.left, polarity))
            right_polarity = -polarity if expr.op == "-" else polarity
            merge(self._collect_branch_nudge_nodes_from_expr(expr.right, right_polarity))
            return {node: 1 if sign > 0 else -1 for node, sign in nodes.items() if sign}
        if isinstance(expr, UnaryExpr):
            operand_polarity = -polarity if expr.op == "-" else polarity
            return self._collect_branch_nudge_nodes_from_expr(expr.operand, operand_polarity)
        if isinstance(expr, TernaryExpr):
            merge(self._collect_branch_nudge_nodes_from_expr(expr.cond, polarity))
            merge(self._collect_branch_nudge_nodes_from_expr(expr.true_expr, polarity))
            merge(self._collect_branch_nudge_nodes_from_expr(expr.false_expr, polarity))
            return {node: 1 if sign > 0 else -1 for node, sign in nodes.items() if sign}
        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                merge(self._collect_branch_nudge_nodes_from_expr(arg, polarity))
            return {node: 1 if sign > 0 else -1 for node, sign in nodes.items() if sign}
        if isinstance(expr, ArrayAccess):
            return self._collect_branch_nudge_nodes_from_expr(expr.index, polarity)
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                merge(self._collect_branch_nudge_nodes_from_expr(arg, polarity))
            return {node: 1 if sign > 0 else -1 for node, sign in nodes.items() if sign}
        return nodes

    def _alloc_stateful_func_key(self, kind: str, expr) -> str:
        cache_key = (kind, id(expr))
        if cache_key in self._stateful_func_key_cache:
            return self._stateful_func_key_cache[cache_key]
        if kind == "transition":
            key = f"trans_{self._trans_counter}"
            self._trans_counter += 1
        elif kind == "slew":
            key = f"slew_{self._slew_counter}"
            self._slew_counter += 1
        elif kind == "last_crossing":
            key = f"last_cross_{self._last_cross_counter}"
            self._last_cross_counter += 1
        elif kind == "idtmod":
            key = f"idtmod_{self._idt_counter}"
            self._idt_counter += 1
        else:
            raise ValueError(f"unknown stateful func key kind: {kind}")
        self._stateful_func_key_cache[cache_key] = key
        return key

    def _has_post_update_event(self, stmt) -> bool:
        if isinstance(stmt, Block):
            return any(self._has_post_update_event(s) for s in stmt.statements)
        if isinstance(stmt, EventStatement):
            event = stmt.event
            if isinstance(event, EventExpr):
                return self._event_requires_post_update(event)
            if isinstance(event, CombinedEvent):
                return any(
                    self._event_requires_post_update(e)
                    for e in event.events
                )
            return False
        if isinstance(stmt, IfStatement):
            return self._has_post_update_event(stmt.then_body) or (
                stmt.else_body is not None and self._has_post_update_event(stmt.else_body)
            )
        if isinstance(stmt, WhileStatement):
            return self._has_post_update_event(stmt.body)
        if isinstance(stmt, ForStatement):
            return self._has_post_update_event(stmt.body)
        if isinstance(stmt, CaseStatement):
            return any(self._has_post_update_event(item.body) for item in stmt.items)
        return False

    def _has_refresh_logic(self, stmt) -> bool:
        if isinstance(stmt, Block):
            return any(self._has_refresh_logic(s) for s in stmt.statements)
        if isinstance(stmt, (Contribution, Assignment)):
            return True
        if isinstance(stmt, IfStatement):
            return self._has_refresh_logic(stmt.then_body) or (
                stmt.else_body is not None and self._has_refresh_logic(stmt.else_body)
            )
        if isinstance(stmt, WhileStatement):
            return self._has_refresh_logic(stmt.body)
        if isinstance(stmt, ForStatement):
            return self._has_refresh_logic(stmt.body)
        if isinstance(stmt, CaseStatement):
            return any(self._has_refresh_logic(item.body) for item in stmt.items)
        return False

    def _compile_post_update_statement(self, stmt, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        if not self._has_post_update_event(stmt):
            return lines

        if isinstance(stmt, Block):
            for s in stmt.statements:
                lines.extend(self._compile_post_update_statement(s, indent))
            return lines

        if isinstance(stmt, EventStatement):
            return self._compile_post_update_event_statement(stmt, indent)

        if isinstance(stmt, IfStatement):
            cond = self._compile_expr(stmt.cond)
            then_lines = self._compile_post_update_statement(stmt.then_body, indent + 1)
            else_lines = self._compile_post_update_statement(stmt.else_body, indent + 1) if stmt.else_body else []
            if then_lines:
                lines.append(f"{prefix}if {cond}:")
                lines.extend(then_lines)
            if else_lines:
                if not then_lines:
                    lines.append(f"{prefix}if not ({cond}):")
                else:
                    lines.append(f"{prefix}else:")
                lines.extend(else_lines)
            return lines

        if isinstance(stmt, WhileStatement):
            cond = self._compile_expr(stmt.cond)
            body_lines = self._compile_post_update_statement(stmt.body, indent + 1)
            if body_lines:
                lines.append(f"{prefix}while {cond}:")
                lines.extend(body_lines)
            return lines

        if isinstance(stmt, CaseStatement):
            sel = self._compile_expr(stmt.expr)
            first = True
            default_lines = None
            for item in stmt.items:
                body_lines = self._compile_post_update_statement(item.body, indent + 1)
                if not body_lines:
                    continue
                if not item.values:
                    default_lines = body_lines
                    continue
                cond_parts = [f"({sel} == {self._compile_expr(v)})" for v in item.values]
                cond = ' or '.join(cond_parts)
                keyword = 'if' if first else 'elif'
                lines.append(f"{prefix}{keyword} {cond}:")
                lines.extend(body_lines)
                first = False
            if default_lines is not None:
                if first:
                    lines.extend(default_lines)
                else:
                    lines.append(f"{prefix}else:")
                    lines.extend(default_lines)
            return lines

        if isinstance(stmt, ForStatement):
            return self._compile_post_update_for(stmt, indent)

        return lines

    def _compile_post_update_event_statement(self, stmt: EventStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []
        event = stmt.event

        if isinstance(event, EventExpr):
            if event.event_type == EventType.CROSS:
                if not self._event_requires_post_update(event):
                    return lines
                key = self._alloc_event_key("cross", event)
                lines.append(f"{prefix}if {self._compile_cross_call(event, key)}:")
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('cross', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")
                lines.append(f"{prefix}    _post_event_fired = True")
            elif event.event_type == EventType.ABOVE:
                if not self._event_requires_post_update(event):
                    return lines
                key = self._alloc_event_key("above", event)
                lines.append(f"{prefix}if {self._compile_above_call(event, key)}:")
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('above', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")
                lines.append(f"{prefix}    _post_event_fired = True")
            return lines

        if isinstance(event, CombinedEvent):
            conditions = []
            for e in event.events:
                if not isinstance(e, EventExpr):
                    continue
                if e.event_type == EventType.CROSS:
                    if not self._event_requires_post_update(e):
                        continue
                    key = self._alloc_event_key("cross", e)
                    conditions.append(self._compile_cross_call(e, key))
                elif e.event_type == EventType.ABOVE:
                    if not self._event_requires_post_update(e):
                        continue
                    key = self._alloc_event_key("above", e)
                    conditions.append(self._compile_above_call(e, key))
            if conditions:
                hit_vars = []
                for idx, cond in enumerate(conditions):
                    var = f"_post_event_hit_{idx}"
                    hit_vars.append(var)
                    lines.append(f"{prefix}{var} = {cond}")
                cond = ' or '.join(hit_vars)
                lines.append(f"{prefix}if {cond}:")
                key = self._alloc_event_key("combined", event)
                lines.append(f"{prefix}    self._event_trace_audit_enter_event('combined', {key!r}, time)")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_event_body_with_best_rust_write(
                    key,
                    stmt.body,
                    indent + 1,
                )
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_trace_audit_exit_event()")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_interpolated_node_values = {{}}")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")
                lines.append(f"{prefix}    _post_event_fired = True")
        return lines

    def _compile_refresh_statement(self, stmt, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        if not self._has_refresh_logic(stmt):
            return lines

        if isinstance(stmt, Block):
            for s in stmt.statements:
                lines.extend(self._compile_refresh_statement(s, indent))
            return lines

        if isinstance(stmt, Contribution):
            return self._compile_contribution(stmt, indent)

        if isinstance(stmt, Assignment):
            return self._compile_assignment(stmt, indent)

        if isinstance(stmt, IfStatement):
            cond = self._compile_expr(stmt.cond)
            then_lines = self._compile_refresh_statement(stmt.then_body, indent + 1)
            else_lines = self._compile_refresh_statement(stmt.else_body, indent + 1) if stmt.else_body else []
            if then_lines:
                lines.append(f"{prefix}if {cond}:")
                lines.extend(then_lines)
            if else_lines:
                if not then_lines:
                    lines.append(f"{prefix}if not ({cond}):")
                else:
                    lines.append(f"{prefix}else:")
                lines.extend(else_lines)
            return lines

        if isinstance(stmt, WhileStatement):
            cond = self._compile_expr(stmt.cond)
            body_lines = self._compile_refresh_statement(stmt.body, indent + 1)
            if body_lines:
                lines.append(f"{prefix}while {cond}:")
                lines.extend(body_lines)
            return lines

        if isinstance(stmt, CaseStatement):
            sel = self._compile_expr(stmt.expr)
            first = True
            default_lines = None
            for item in stmt.items:
                body_lines = self._compile_refresh_statement(item.body, indent + 1)
                if not body_lines:
                    continue
                if not item.values:
                    default_lines = body_lines
                    continue
                cond_parts = [f"({sel} == {self._compile_expr(v)})" for v in item.values]
                cond = ' or '.join(cond_parts)
                keyword = 'if' if first else 'elif'
                lines.append(f"{prefix}{keyword} {cond}:")
                lines.extend(body_lines)
                first = False
            if default_lines is not None:
                if first:
                    lines.extend(default_lines)
                else:
                    lines.append(f"{prefix}else:")
                    lines.extend(default_lines)
            return lines

        if isinstance(stmt, ForStatement):
            return self._compile_refresh_for(stmt, indent)

        return lines

    def _compile_post_update_for(self, stmt: ForStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        loop_var = None
        if isinstance(stmt.init.target, Identifier):
            loop_var = stmt.init.target.name
        elif isinstance(stmt.init.target, ArrayAccess):
            loop_var = stmt.init.target.name

        if loop_var is None:
            return [f"{prefix}pass  # could not compile for loop"]

        init_val = self._compile_expr(stmt.init.value)

        prev_loop_var = self._in_loop_var
        self._in_loop_var = loop_var

        lines.append(f"{prefix}self._state_set({loop_var!r}, {init_val})")
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self._state_set({loop_var!r}, _loop_{loop_var})")
        body_lines = self._compile_post_update_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self._state_set({loop_var!r}, _loop_{loop_var})")

        self._in_loop_var = prev_loop_var
        return lines

    def _compile_refresh_for(self, stmt: ForStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        loop_var = None
        if isinstance(stmt.init.target, Identifier):
            loop_var = stmt.init.target.name
        elif isinstance(stmt.init.target, ArrayAccess):
            loop_var = stmt.init.target.name

        if loop_var is None:
            return [f"{prefix}pass  # could not compile for loop"]

        init_val = self._compile_expr(stmt.init.value)

        prev_loop_var = self._in_loop_var
        self._in_loop_var = loop_var

        lines.append(f"{prefix}self._state_set({loop_var!r}, {init_val})")
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self._state_set({loop_var!r}, _loop_{loop_var})")
        body_lines = self._compile_refresh_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self._state_set({loop_var!r}, _loop_{loop_var})")

        self._in_loop_var = prev_loop_var
        return lines

    def _collect_for_loop_state_targets(self, stmt, acc: set[str]) -> None:
        """Collect scalar state names used as generated for-loop variables."""
        if stmt is None:
            return
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_for_loop_state_targets(child, acc)
            return
        if isinstance(stmt, ForStatement):
            target = getattr(stmt.init, "target", None)
            if isinstance(target, Identifier):
                acc.add(target.name)
            elif isinstance(target, ArrayAccess):
                acc.add(target.name)
            self._collect_for_loop_state_targets(stmt.body, acc)
            return
        if isinstance(stmt, EventStatement):
            self._collect_for_loop_state_targets(stmt.body, acc)
            return
        if isinstance(stmt, IfStatement):
            self._collect_for_loop_state_targets(stmt.then_body, acc)
            self._collect_for_loop_state_targets(stmt.else_body, acc)
            return
        if isinstance(stmt, WhileStatement):
            self._collect_for_loop_state_targets(stmt.body, acc)
            return
        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._collect_for_loop_state_targets(item.body, acc)

    def _collect_evaluate_state_scalar_accesses(self, stmt, acc: set[str]) -> None:
        """Collect scalar state touched by generated evaluate-path code."""
        if stmt is None:
            return
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_evaluate_state_scalar_accesses(child, acc)
            return
        if isinstance(stmt, EventStatement):
            if self._event_compiles_in_evaluate(stmt.event):
                self._collect_state_scalar_accesses_from_event(stmt.event, acc)
                self._collect_evaluate_state_scalar_accesses(stmt.body, acc)
            return
        if isinstance(stmt, Contribution):
            self._collect_state_scalar_accesses_from_expr(stmt.expr, acc)
            branch = stmt.branch
            for expr in (
                branch.node1_index,
                branch.node2_index,
                branch.node1_index2,
                branch.node2_index2,
            ):
                self._collect_state_scalar_accesses_from_expr(expr, acc)
            return
        if isinstance(stmt, Assignment):
            if isinstance(stmt.target, Identifier) and stmt.target.name in self._state_scalar_slot_by_name:
                acc.add(stmt.target.name)
            elif isinstance(stmt.target, ArrayAccess):
                self._collect_state_scalar_accesses_from_expr(stmt.target.index, acc)
            self._collect_state_scalar_accesses_from_expr(stmt.value, acc)
            return
        if isinstance(stmt, IfStatement):
            self._collect_state_scalar_accesses_from_expr(stmt.cond, acc)
            self._collect_evaluate_state_scalar_accesses(stmt.then_body, acc)
            self._collect_evaluate_state_scalar_accesses(stmt.else_body, acc)
            return
        if isinstance(stmt, WhileStatement):
            self._collect_state_scalar_accesses_from_expr(stmt.cond, acc)
            self._collect_evaluate_state_scalar_accesses(stmt.body, acc)
            return
        if isinstance(stmt, ForStatement):
            self._collect_evaluate_assignment_accesses(stmt.init, acc)
            self._collect_state_scalar_accesses_from_expr(stmt.cond, acc)
            self._collect_evaluate_assignment_accesses(stmt.update, acc)
            self._collect_evaluate_state_scalar_accesses(stmt.body, acc)
            return
        if isinstance(stmt, CaseStatement):
            self._collect_state_scalar_accesses_from_expr(stmt.expr, acc)
            for item in stmt.items:
                for value in item.values:
                    self._collect_state_scalar_accesses_from_expr(value, acc)
                self._collect_evaluate_state_scalar_accesses(item.body, acc)
            return
        if isinstance(stmt, SystemTask):
            for arg in stmt.args:
                self._collect_state_scalar_accesses_from_expr(arg, acc)

    def _collect_evaluate_assignment_accesses(self, stmt: Assignment, acc: set[str]) -> None:
        if isinstance(stmt.target, Identifier) and stmt.target.name in self._state_scalar_slot_by_name:
            acc.add(stmt.target.name)
        elif isinstance(stmt.target, ArrayAccess):
            self._collect_state_scalar_accesses_from_expr(stmt.target.index, acc)
        self._collect_state_scalar_accesses_from_expr(stmt.value, acc)

    def _event_compiles_in_evaluate(self, event) -> bool:
        if isinstance(event, EventExpr):
            return event.event_type not in (EventType.INITIAL_STEP, EventType.FINAL_STEP)
        if isinstance(event, CombinedEvent):
            return any(
                e.event_type not in (EventType.INITIAL_STEP, EventType.FINAL_STEP)
                for e in event.events
            )
        return False

    def _collect_state_scalar_accesses_from_event(self, event, acc: set[str]) -> None:
        if isinstance(event, EventExpr):
            if event.event_type in (EventType.INITIAL_STEP, EventType.FINAL_STEP):
                return
            for expr in event.args:
                self._collect_state_scalar_accesses_from_expr(expr, acc)
            self._collect_state_scalar_accesses_from_expr(event.time_tol_expr, acc)
            self._collect_state_scalar_accesses_from_expr(event.expr_tol_expr, acc)
            return
        if isinstance(event, CombinedEvent):
            for child in event.events:
                self._collect_state_scalar_accesses_from_event(child, acc)

    def _collect_state_scalar_accesses_from_expr(self, expr: Optional[Expr], acc: set[str]) -> None:
        if expr is None:
            return
        if isinstance(expr, Identifier):
            if expr.name in self._state_scalar_slot_by_name:
                acc.add(expr.name)
            return
        if isinstance(expr, ArrayAccess):
            self._collect_state_scalar_accesses_from_expr(expr.index, acc)
            return
        if isinstance(expr, BinaryExpr):
            self._collect_state_scalar_accesses_from_expr(expr.left, acc)
            self._collect_state_scalar_accesses_from_expr(expr.right, acc)
            return
        if isinstance(expr, UnaryExpr):
            self._collect_state_scalar_accesses_from_expr(expr.operand, acc)
            return
        if isinstance(expr, TernaryExpr):
            self._collect_state_scalar_accesses_from_expr(expr.cond, acc)
            self._collect_state_scalar_accesses_from_expr(expr.true_expr, acc)
            self._collect_state_scalar_accesses_from_expr(expr.false_expr, acc)
            return
        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._collect_state_scalar_accesses_from_expr(arg, acc)
            return
        if isinstance(expr, BranchAccess):
            for sub_expr in (
                expr.node1_index,
                expr.node2_index,
                expr.node1_index2,
                expr.node2_index2,
            ):
                self._collect_state_scalar_accesses_from_expr(sub_expr, acc)
            return
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                self._collect_state_scalar_accesses_from_expr(arg, acc)

    def _compile_contribution(self, stmt: Contribution, indent) -> List[str]:
        prefix = '    ' * indent
        branch = stmt.branch
        node = branch.node1
        transition_affine = None
        if (
            branch.access_type == "V"
            and branch.node1_index is None
            and branch.node1_index2 is None
            and branch.node2_index is None
            and branch.node2_index2 is None
            and not self.static_branch_fastpath_codegen
        ):
            transition_affine = self._transition_affine_expr(stmt.expr)
        if transition_affine is not None:
            transition_call, offset_expr, scale_expr = transition_affine
            key_expr, target, delay, rise, fall = self._compile_transition_call_parts(
                transition_call
            )
            offset = self._compile_expr(offset_expr)
            scale = self._compile_expr(scale_expr)
            if branch.node2 is not None:
                base = self._compile_node_voltage(
                    branch.node2,
                    branch.node2_index,
                    branch.node2_index2,
                )
            else:
                base = "0.0"
            # Audit 088: emit deferred lazy form when the static analyzer
            # has confirmed no subsequent statement in this analog block
            # reads V(node). Otherwise keep the immediate form.
            can_defer = node not in self._transition_defer_unsafe_nodes
            method = "_transition_output_lazy" if can_defer else "_transition_output"
            return [
                f"{prefix}self.{method}({node!r}, {key_expr}, time, "
                f"{target}, {base}, {offset}, {scale}, {delay}, {rise}, {fall}, nv)"
            ]

        state_output_name = self._simple_state_output_contribution(stmt)
        if state_output_name in self._rust_output_hold_state_names:
            expr = self._compile_expr(stmt.expr)
            return [
                f"{prefix}if not self._rust_state_output_hold_production({state_output_name!r}, {node!r}, nv):",
                f"{prefix}    self._set_output({node!r}, {expr}, nv)",
            ]

        expr = self._compile_expr(stmt.expr)
        if branch.node2 is not None:
            node2_expr = self._compile_node_voltage(branch.node2, branch.node2_index, branch.node2_index2)
            expr = f"(({node2_expr}) + ({expr}))"

        if branch.node1_index is not None:
            # Dynamic array-indexed port: V(DOUT[i]) <+ or V(DOUT[i][j]) <+
            idx_expr = self._compile_expr(branch.node1_index)
            if branch.node1_index2 is not None:
                idx_expr2 = self._compile_expr(branch.node1_index2)
                node_expr = f"self._resolve_dynamic_node({node!r}, {idx_expr}, {idx_expr2})"
                return [f"{prefix}self._set_output({node_expr}, {expr}, nv)"]
            node_expr = f"self._resolve_dynamic_node({node!r}, {idx_expr})"
            return [f"{prefix}self._set_output({node_expr}, {expr}, nv)"]
        if self.static_branch_fastpath_codegen:
            slot = self._static_branch_write_slot_by_node.get(node)
            if slot is not None:
                return [f"{prefix}self._set_static_branch_output_by_slot({slot}, {expr}, nv)"]
            return [f"{prefix}self._set_static_branch_output({node!r}, {expr}, nv)"]
        else:
            return [f"{prefix}self._set_output({node!r}, {expr}, nv)"]

    def _compile_assignment(self, stmt: Assignment, indent) -> List[str]:
        prefix = '    ' * indent
        val = self._compile_expr(stmt.value)

        if isinstance(stmt.target, Identifier):
            name = stmt.target.name
            if self._is_integer_variable(name):
                val = f"self._to_integer({val})"
            if name == self._in_loop_var:
                line = f"self._state_set({name!r}, {val})"
            elif self._state_local_fastpath_active and name in self._state_local_fastpath_names:
                line = f"{self._state_local_name_by_state[name]} = {val}"
            elif self.indexed_state_fastpath_codegen and name in self._state_scalar_slot_by_name:
                slot = self._state_scalar_slot_by_name[name]
                line = f"self._state_set_by_slot({slot}, {name!r}, {val})"
            else:
                line = f"self._state_set({name!r}, {val})"
            if self._should_gate_self_referential_assignment(stmt, name):
                key = self._discrete_assignment_key(stmt)
                return [
                    f"{prefix}if self._should_update_discrete_state({key!r}, time):",
                    f"{prefix}    {line}",
                ]
            return [f"{prefix}{line}"]

        elif isinstance(stmt.target, ArrayAccess):
            name = stmt.target.name
            idx = self._compile_expr(stmt.target.index)
            if self._is_integer_variable(name):
                val = f"self._to_integer({val})"
            return [f"{prefix}self._array_set({name!r}, int({idx}), {val})"]

        raise CompilationError(
            f"Module {self.module.name} has invalid assignment target "
            f"{type(stmt.target).__name__}; Spectre requires a variable or array element"
        )

    def _transition_affine_expr(
        self,
        expr: Expr,
    ) -> Optional[Tuple[FunctionCall, Expr, Expr]]:
        """Return ``transition_call, offset, scale`` for affine transition exprs.

        This recognizes common behavioral-output forms such as
        ``transition(q, 0)``, ``vdd * transition(q, 0)``, and
        ``vl + (vh - vl) * transition(q, 0)``.  It intentionally rejects
        expressions with multiple transition calls or transition-dependent
        denominators.
        """
        zero = NumberLiteral(0.0)
        one = NumberLiteral(1.0)

        def contains_transition(candidate: Expr) -> bool:
            return self._expr_has_function_call(candidate, {"transition"})

        def num(value: float) -> NumberLiteral:
            return NumberLiteral(float(value))

        def neg(candidate: Expr) -> Expr:
            if isinstance(candidate, NumberLiteral):
                return num(-float(candidate.value))
            return UnaryExpr("-", candidate)

        def add(left: Expr, right: Expr) -> Expr:
            if isinstance(left, NumberLiteral) and float(left.value) == 0.0:
                return right
            if isinstance(right, NumberLiteral) and float(right.value) == 0.0:
                return left
            return BinaryExpr("+", left, right)

        def sub(left: Expr, right: Expr) -> Expr:
            if isinstance(right, NumberLiteral) and float(right.value) == 0.0:
                return left
            return BinaryExpr("-", left, right)

        def mul(left: Expr, right: Expr) -> Expr:
            if isinstance(left, NumberLiteral):
                left_f = float(left.value)
                if left_f == 0.0:
                    return num(0.0)
                if left_f == 1.0:
                    return right
            if isinstance(right, NumberLiteral):
                right_f = float(right.value)
                if right_f == 0.0:
                    return num(0.0)
                if right_f == 1.0:
                    return left
            return BinaryExpr("*", left, right)

        def div(left: Expr, right: Expr) -> Expr:
            if isinstance(left, NumberLiteral) and float(left.value) == 0.0:
                return num(0.0)
            if isinstance(right, NumberLiteral) and float(right.value) == 1.0:
                return left
            return BinaryExpr("/", left, right)

        def visit(candidate: Expr) -> Optional[Tuple[FunctionCall, Expr, Expr]]:
            if isinstance(candidate, FunctionCall) and candidate.name == "transition":
                return candidate, zero, one

            if isinstance(candidate, UnaryExpr):
                inner = visit(candidate.operand)
                if inner is None:
                    return None
                call, offset, scale = inner
                if candidate.op == "+":
                    return call, offset, scale
                if candidate.op == "-":
                    return call, neg(offset), neg(scale)
                return None

            if not isinstance(candidate, BinaryExpr):
                return None

            left_has = contains_transition(candidate.left)
            right_has = contains_transition(candidate.right)
            if left_has and right_has:
                return None

            if candidate.op in {"+", "-"}:
                if left_has:
                    inner = visit(candidate.left)
                    if inner is None:
                        return None
                    call, offset, scale = inner
                    if candidate.op == "+":
                        return call, add(offset, candidate.right), scale
                    return call, sub(offset, candidate.right), scale
                if right_has:
                    inner = visit(candidate.right)
                    if inner is None:
                        return None
                    call, offset, scale = inner
                    if candidate.op == "+":
                        return call, add(candidate.left, offset), scale
                    return call, sub(candidate.left, offset), neg(scale)
                return None

            if candidate.op == "*":
                if left_has:
                    inner = visit(candidate.left)
                    if inner is None:
                        return None
                    call, offset, scale = inner
                    return call, mul(offset, candidate.right), mul(scale, candidate.right)
                if right_has:
                    inner = visit(candidate.right)
                    if inner is None:
                        return None
                    call, offset, scale = inner
                    return call, mul(candidate.left, offset), mul(candidate.left, scale)
                return None

            if candidate.op == "/":
                if left_has and not right_has:
                    inner = visit(candidate.left)
                    if inner is None:
                        return None
                    call, offset, scale = inner
                    return call, div(offset, candidate.right), div(scale, candidate.right)
                return None

            return None

        return visit(expr)

    def _discrete_assignment_key(self, stmt: Assignment) -> str:
        stmt_id = id(stmt)
        key = self._discrete_assignment_key_cache.get(stmt_id)
        if key is None:
            key = f"assign_{self._discrete_assignment_counter}"
            self._discrete_assignment_counter += 1
            self._discrete_assignment_key_cache[stmt_id] = key
        return key

    def _should_gate_self_referential_assignment(self, stmt: Assignment, name: str) -> bool:
        if self._is_integer_variable(name):
            return False
        return self._expr_references_variable(stmt.value, name)

    def _expr_references_variable(self, expr: Expr, name: str) -> bool:
        if isinstance(expr, Identifier):
            return expr.name == name
        if isinstance(expr, ArrayAccess):
            return expr.name == name or self._expr_references_variable(expr.index, name)
        if isinstance(expr, BinaryExpr):
            return (
                self._expr_references_variable(expr.left, name)
                or self._expr_references_variable(expr.right, name)
            )
        if isinstance(expr, UnaryExpr):
            return self._expr_references_variable(expr.operand, name)
        if isinstance(expr, TernaryExpr):
            return (
                self._expr_references_variable(expr.cond, name)
                or self._expr_references_variable(expr.true_expr, name)
                or self._expr_references_variable(expr.false_expr, name)
            )
        if isinstance(expr, FunctionCall):
            return any(self._expr_references_variable(arg, name) for arg in expr.args)
        if isinstance(expr, BranchAccess):
            return any(
                sub is not None and self._expr_references_variable(sub, name)
                for sub in (
                    expr.node1_index,
                    expr.node2_index,
                    expr.node1_index2,
                    expr.node2_index2,
                )
            )
        if isinstance(expr, MethodCall):
            return any(self._expr_references_variable(arg, name) for arg in expr.args)
        return False

    def _is_integer_variable(self, name: str) -> bool:
        for variable in self.module.variables:
            if variable.name == name:
                return self._is_integer_decl(variable)
        return False

    def _is_integer_decl(self, variable) -> bool:
        return (
            variable.var_type == ParamType.INTEGER
            or getattr(variable.var_type, "name", "") == "INTEGER"
            or variable.var_type in {"integer", "genvar"}
        )

    def _compile_for(self, stmt: ForStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        # Extract loop variable, start, end, step
        loop_var = None
        if isinstance(stmt.init.target, Identifier):
            loop_var = stmt.init.target.name
        elif isinstance(stmt.init.target, ArrayAccess):
            loop_var = stmt.init.target.name

        if loop_var is None:
            return [f"{prefix}pass  # could not compile for loop"]

        init_val = self._compile_expr(stmt.init.value)

        # Track that we're in a loop (for dynamic transition/contribution keys)
        prev_loop_var = self._in_loop_var
        self._in_loop_var = loop_var

        # Use a while loop in Python
        lines.append(f"{prefix}self._state_set({loop_var!r}, {init_val})")
        # Replace loop var references in condition
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self._state_set({loop_var!r}, _loop_{loop_var})")
        body_lines = self._compile_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self._state_set({loop_var!r}, _loop_{loop_var})")

        self._in_loop_var = prev_loop_var
        return lines

    def _compile_while(self, stmt: WhileStatement, indent) -> List[str]:
        prefix = '    ' * indent
        cond = self._compile_expr(stmt.cond)
        lines = [f"{prefix}while {cond}:"]
        body_lines = self._compile_statement(stmt.body, indent + 1)
        lines.extend(body_lines)
        if not body_lines:
            lines.append(f"{prefix}    pass")
        return lines

    def _compile_case(self, stmt: CaseStatement, indent) -> List[str]:
        """Compile a case statement to chained if/elif/else."""
        prefix = '    ' * indent
        lines = []
        sel = self._compile_expr(stmt.expr)
        first = True
        default_lines = None
        for item in stmt.items:
            if not item.values:
                # default branch — emit last
                default_lines = self._compile_statement(item.body, indent + 1)
                continue
            cond_parts = [f"({sel} == {self._compile_expr(v)})" for v in item.values]
            cond = ' or '.join(cond_parts)
            keyword = 'if' if first else 'elif'
            lines.append(f"{prefix}{keyword} {cond}:")
            body_lines = self._compile_statement(item.body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")
            first = False
        if default_lines is not None:
            if first:
                # only default, no value branches
                lines.extend(default_lines)
            else:
                lines.append(f"{prefix}else:")
                lines.extend(default_lines)
                if not default_lines:
                    lines.append(f"{prefix}    pass")
        return lines

    def _compile_transition_target_probe_method(self, stmt) -> Tuple[List[str], int]:
        """Generate a read-only probe for discontinuous transition() targets.

        The generated method mirrors side-effect-free continuous assignments
        into a private ``_probe_state`` and records target values for transition
        calls whose targets are driven by discrete conditions.  It intentionally
        skips event bodies and system/file side effects.
        """
        assignment_exprs: Dict[str, Expr] = {}
        conditionally_assigned: set[str] = set()
        self._collect_transition_probe_assignments(
            stmt,
            assignment_exprs,
            conditionally_assigned,
            under_condition=False,
        )

        lines = [
            "",
            "    def _transition_target_probe_values(self, nv, time):",
            "        _probe_state = dict(self.state)",
            "        _probe_time = time",
            "        _probe_values = []",
            "        try:",
        ]
        body_lines: List[str] = []
        probe_count = self._compile_transition_probe_stmt(
            stmt,
            3,
            body_lines,
            assignment_exprs,
            conditionally_assigned,
        )
        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append("            pass")
        lines.extend(
            [
                "        except Exception:",
                "            return ()",
                "        return tuple(_probe_values)",
            ]
        )
        return lines, probe_count

    def _collect_transition_probe_assignments(
        self,
        stmt,
        assignment_exprs: Dict[str, Expr],
        conditionally_assigned: set[str],
        *,
        under_condition: bool,
    ) -> None:
        if stmt is None:
            return
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_transition_probe_assignments(
                    child,
                    assignment_exprs,
                    conditionally_assigned,
                    under_condition=under_condition,
                )
            return
        if isinstance(stmt, EventStatement):
            return
        if isinstance(stmt, Assignment):
            if isinstance(stmt.target, Identifier):
                name = stmt.target.name
                assignment_exprs[name] = stmt.value
                if under_condition:
                    conditionally_assigned.add(name)
            return
        if isinstance(stmt, IfStatement):
            self._collect_transition_probe_assignments(
                stmt.then_body,
                assignment_exprs,
                conditionally_assigned,
                under_condition=True,
            )
            if stmt.else_body is not None:
                self._collect_transition_probe_assignments(
                    stmt.else_body,
                    assignment_exprs,
                    conditionally_assigned,
                    under_condition=True,
                )
            return
        if isinstance(stmt, ForStatement):
            self._collect_transition_probe_assignments(
                stmt.body,
                assignment_exprs,
                conditionally_assigned,
                under_condition=True,
            )
            return
        if isinstance(stmt, WhileStatement):
            self._collect_transition_probe_assignments(
                stmt.body,
                assignment_exprs,
                conditionally_assigned,
                under_condition=True,
            )
            return
        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._collect_transition_probe_assignments(
                    item.body,
                    assignment_exprs,
                    conditionally_assigned,
                    under_condition=True,
                )

    def _compile_transition_probe_stmt(
        self,
        stmt,
        indent: int,
        lines: List[str],
        assignment_exprs: Dict[str, Expr],
        conditionally_assigned: set[str],
    ) -> int:
        if stmt is None:
            return 0
        prefix = "    " * indent
        probe_count = 0
        if isinstance(stmt, Block):
            for child in stmt.statements:
                probe_count += self._compile_transition_probe_stmt(
                    child,
                    indent,
                    lines,
                    assignment_exprs,
                    conditionally_assigned,
                )
            return probe_count
        if isinstance(stmt, EventStatement):
            return 0
        if isinstance(stmt, Assignment):
            if isinstance(stmt.target, Identifier):
                name = stmt.target.name
                value = self._compile_transition_probe_expr(stmt.value)
                if self._is_integer_variable(name):
                    value = f"self._to_integer({value})"
                lines.append(f"{prefix}_probe_state[{name!r}] = {value}")
            return 0
        if isinstance(stmt, IfStatement):
            cond = self._compile_transition_probe_expr(stmt.cond)
            then_lines: List[str] = []
            then_count = self._compile_transition_probe_stmt(
                stmt.then_body,
                indent + 1,
                then_lines,
                assignment_exprs,
                conditionally_assigned,
            )
            else_lines: List[str] = []
            else_count = 0
            if stmt.else_body is not None:
                else_count = self._compile_transition_probe_stmt(
                    stmt.else_body,
                    indent + 1,
                    else_lines,
                    assignment_exprs,
                    conditionally_assigned,
                )
            lines.append(f"{prefix}if {cond}:")
            lines.extend(then_lines or [f"{prefix}    pass"])
            if stmt.else_body is not None:
                lines.append(f"{prefix}else:")
                lines.extend(else_lines or [f"{prefix}    pass"])
            return then_count + else_count
        if isinstance(stmt, Contribution):
            for target in self._iter_transition_targets_in_expr(stmt.expr):
                if not self._transition_probe_target_is_discrete(
                    target,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=set(),
                ):
                    continue
                target_code = self._compile_transition_probe_expr(target)
                lines.append(f"{prefix}_probe_values.append(float({target_code}))")
                probe_count += 1
            return probe_count
        if isinstance(stmt, SystemTask):
            return 0
        # Loops/case statements can carry complex stateful semantics.  The
        # conservative probe skips them instead of guessing and perturbing the
        # transient schedule.
        return 0

    def _iter_transition_targets_in_expr(self, expr: Expr):
        if isinstance(expr, FunctionCall):
            if expr.name == "transition" and expr.args:
                yield expr.args[0]
            for arg in expr.args:
                yield from self._iter_transition_targets_in_expr(arg)
            return
        if isinstance(expr, BinaryExpr):
            yield from self._iter_transition_targets_in_expr(expr.left)
            yield from self._iter_transition_targets_in_expr(expr.right)
            return
        if isinstance(expr, UnaryExpr):
            yield from self._iter_transition_targets_in_expr(expr.operand)
            return
        if isinstance(expr, TernaryExpr):
            yield from self._iter_transition_targets_in_expr(expr.cond)
            yield from self._iter_transition_targets_in_expr(expr.true_expr)
            yield from self._iter_transition_targets_in_expr(expr.false_expr)
            return
        if isinstance(expr, ArrayAccess):
            yield from self._iter_transition_targets_in_expr(expr.index)
            return
        if isinstance(expr, BranchAccess):
            for sub in (
                expr.node1_index,
                expr.node2_index,
                expr.node1_index2,
                expr.node2_index2,
            ):
                if sub is not None:
                    yield from self._iter_transition_targets_in_expr(sub)
            return
        if isinstance(expr, MethodCall):
            for arg in expr.args:
                yield from self._iter_transition_targets_in_expr(arg)

    def _transition_probe_target_is_discrete(
        self,
        expr: Expr,
        assignment_exprs: Dict[str, Expr],
        conditionally_assigned: set[str],
        *,
        seen: set[str],
    ) -> bool:
        if isinstance(expr, TernaryExpr):
            return True
        if isinstance(expr, BinaryExpr):
            if expr.op in {">", "<", ">=", "<=", "==", "!=", "&&", "||"}:
                return True
            return (
                self._transition_probe_target_is_discrete(
                    expr.left,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=seen,
                )
                or self._transition_probe_target_is_discrete(
                    expr.right,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=seen,
                )
            )
        if isinstance(expr, UnaryExpr):
            if expr.op in {"!", "~"}:
                return True
            return self._transition_probe_target_is_discrete(
                expr.operand,
                assignment_exprs,
                conditionally_assigned,
                seen=seen,
            )
        if isinstance(expr, Identifier):
            name = expr.name
            if name in conditionally_assigned:
                return True
            if name in seen:
                return False
            assigned = assignment_exprs.get(name)
            if assigned is None:
                return False
            return self._transition_probe_target_is_discrete(
                assigned,
                assignment_exprs,
                conditionally_assigned,
                seen=seen | {name},
            )
        if isinstance(expr, ArrayAccess):
            return self._transition_probe_target_is_discrete(
                expr.index,
                assignment_exprs,
                conditionally_assigned,
                seen=seen,
            )
        if isinstance(expr, FunctionCall):
            return any(
                self._transition_probe_target_is_discrete(
                    arg,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=seen,
                )
                for arg in expr.args
            )
        if isinstance(expr, BranchAccess):
            return any(
                sub is not None
                and self._transition_probe_target_is_discrete(
                    sub,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=seen,
                )
                for sub in (
                    expr.node1_index,
                    expr.node2_index,
                    expr.node1_index2,
                    expr.node2_index2,
                )
            )
        if isinstance(expr, MethodCall):
            return any(
                self._transition_probe_target_is_discrete(
                    arg,
                    assignment_exprs,
                    conditionally_assigned,
                    seen=seen,
                )
                for arg in expr.args
            )
        return False

    def _compile_transition_probe_expr(self, expr: Expr) -> str:
        if isinstance(expr, NumberLiteral):
            return repr(expr.value)
        if isinstance(expr, StringLiteral):
            return repr(expr.value)
        if isinstance(expr, Identifier):
            name = expr.name
            for p in self.module.parameters:
                if p.name == name:
                    return f"self.params[{name!r}]"
            if name in ("$abstime", "$realtime"):
                return "_probe_time"
            if name == "inf":
                return "float('inf')"
            if name == "$temperature":
                return "(self._temperature + 273.15)"
            if name == "$vt":
                return "(1.380649e-23 * (self._temperature + 273.15) / 1.602176634e-19)"
            return f"_probe_state.get({name!r}, self.state.get({name!r}, 0.0))"
        if isinstance(expr, ArrayAccess):
            idx = self._compile_transition_probe_expr(expr.index)
            return f"self._array_get({expr.name!r}, int({idx}))"
        if isinstance(expr, BinaryExpr):
            left = self._compile_transition_probe_expr(expr.left)
            right = self._compile_transition_probe_expr(expr.right)
            op = expr.op
            if op == "/" and self._expr_is_integer(expr):
                return f"self._int_div(({left}), ({right}))"
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
            if op == "&&":
                return f"(({left}) and ({right}))"
            if op == "||":
                return f"(({left}) or ({right}))"
            if op == ">":
                return f"self._cmp_gt(({left}), ({right}))"
            if op == "<":
                return f"self._cmp_lt(({left}), ({right}))"
            if op == ">=":
                return f"self._cmp_ge(({left}), ({right}))"
            if op == "<=":
                return f"self._cmp_le(({left}), ({right}))"
            return f"({left} {op} {right})"
        if isinstance(expr, UnaryExpr):
            operand = self._compile_transition_probe_expr(expr.operand)
            if expr.op == "!":
                return f"(not ({operand}))"
            if expr.op == "~":
                return f"(~int({operand}))"
            return f"({expr.op}{operand})"
        if isinstance(expr, TernaryExpr):
            cond = self._compile_transition_probe_expr(expr.cond)
            true_e = self._compile_transition_probe_expr(expr.true_expr)
            false_e = self._compile_transition_probe_expr(expr.false_expr)
            return f"(({true_e}) if ({cond}) else ({false_e}))"
        if isinstance(expr, BranchAccess):
            if expr.node2:
                n1 = self._compile_transition_probe_node_voltage(
                    expr.node1,
                    expr.node1_index,
                    expr.node1_index2,
                )
                n2 = self._compile_transition_probe_node_voltage(
                    expr.node2,
                    expr.node2_index,
                    expr.node2_index2,
                )
                if expr.access_type == "V":
                    return f"({n1} - {n2})"
                return "0.0"
            if expr.access_type == "V":
                return self._compile_transition_probe_node_voltage(
                    expr.node1,
                    expr.node1_index,
                    expr.node1_index2,
                )
            return "0.0"
        if isinstance(expr, FunctionCall):
            return self._compile_transition_probe_function_call(expr)
        if isinstance(expr, MethodCall):
            return self._compile_transition_probe_method_call(expr)
        return "0.0"

    def _compile_transition_probe_node_voltage(
        self,
        node: str,
        index_expr=None,
        index_expr2=None,
    ) -> str:
        if index_expr is not None:
            idx = self._compile_transition_probe_expr(index_expr)
            if index_expr2 is not None:
                idx2 = self._compile_transition_probe_expr(index_expr2)
                node_expr = f"self._resolve_dynamic_node({node!r}, {idx}, {idx2})"
                return f"self._get_voltage({node_expr}, nv)"
            node_expr = f"self._resolve_dynamic_node({node!r}, {idx})"
            return f"self._get_voltage({node_expr}, nv)"
        return f"self._get_voltage({node!r}, nv)"

    def _compile_transition_probe_function_call(self, expr: FunctionCall) -> str:
        name = expr.name
        math_aliases = {
            "ln", "log", "exp", "sqrt", "abs", "pow", "min", "max",
            "sin", "cos", "tan", "tanh", "floor", "ceil",
        }
        if name.startswith("$") and name[1:] in math_aliases:
            name = name[1:]
        args = [self._compile_transition_probe_expr(a) for a in expr.args]
        if name == "ln":
            return f"math.log({args[0]})"
        if name == "log":
            return f"math.log10({args[0]})"
        if name == "exp":
            return f"math.exp({args[0]})"
        if name == "sqrt":
            return f"math.sqrt({args[0]})"
        if name == "abs":
            return f"abs({args[0]})"
        if name == "pow":
            return f"pow({args[0]}, {args[1]})"
        if name == "min":
            return f"min({args[0]}, {args[1]})"
        if name == "max":
            return f"max({args[0]}, {args[1]})"
        if name == "sin":
            return f"math.sin({args[0]})"
        if name == "cos":
            return f"math.cos({args[0]})"
        if name == "tan":
            return f"math.tan({args[0]})"
        if name == "tanh":
            return f"math.tanh({args[0]})"
        if name == "floor":
            return f"math.floor({args[0]})"
        if name == "ceil":
            return f"math.ceil({args[0]})"
        if name == "transition" and expr.args:
            return self._compile_transition_probe_expr(expr.args[0])
        return "0.0"

    def _compile_transition_probe_method_call(self, expr: MethodCall) -> str:
        args = [self._compile_transition_probe_expr(a) for a in expr.args]
        if expr.method == "substr":
            return f"self.params[{expr.obj!r}][int({args[0]}):int({args[1]})+1]"
        return "''"

    def _compile_expr(self, expr: Expr) -> str:
        """Compile an expression to Python code string."""
        if isinstance(expr, NumberLiteral):
            return repr(expr.value)

        if isinstance(expr, StringLiteral):
            return repr(expr.value)

        if isinstance(expr, Identifier):
            name = expr.name
            # Check if it's a parameter
            for p in self.module.parameters:
                if p.name == name:
                    return f"self.params[{name!r}]"
            # Check if it's a variable
            for v in self.module.variables:
                if v.name == name:
                    if (
                        self._state_local_fastpath_active
                        and name != self._in_loop_var
                        and name in self._state_local_fastpath_names
                    ):
                        return self._state_local_name_by_state[name]
                    if (
                        self.indexed_state_fastpath_codegen
                        and name != self._in_loop_var
                        and name in self._state_scalar_slot_by_name
                    ):
                        slot = self._state_scalar_slot_by_name[name]
                        return f"self._state_get_by_slot({slot}, {name!r})"
                    return f"self.state[{name!r}]"
            # Check if it's a special constant
            if name == 'inf':
                return "float('inf')"
            if name in ('$abstime', '$realtime'):
                return "self._event_time"
            if name == '$temperature':
                return "(self._temperature + 273.15)"
            if name == '$vt':
                return "(1.380649e-23 * (self._temperature + 273.15) / 1.602176634e-19)"
            return f"self.state[{name!r}]"

        if isinstance(expr, ArrayAccess):
            idx = self._compile_expr(expr.index)
            return f"self._array_get({expr.name!r}, int({idx}))"

        if isinstance(expr, BinaryExpr):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            op = expr.op
            if op == '/' and self._expr_is_integer(expr):
                return f"self._int_div(({left}), ({right}))"
            if op == '^':
                # In Verilog-A, ^ is XOR for integers
                return f"(int({left}) ^ int({right}))"
            if op == '&':
                return f"(int({left}) & int({right}))"
            if op == '|':
                return f"(int({left}) | int({right}))"
            if op == '<<':
                return f"(int({left}) << int({right}))"
            if op == '>>':
                return f"(int({left}) >> int({right}))"
            if op == '&&':
                return f"(({left}) and ({right}))"
            if op == '||':
                return f"(({left}) or ({right}))"
            if op == '>':
                return f"self._cmp_gt(({left}), ({right}))"
            if op == '<':
                return f"self._cmp_lt(({left}), ({right}))"
            if op == '>=':
                return f"self._cmp_ge(({left}), ({right}))"
            if op == '<=':
                return f"self._cmp_le(({left}), ({right}))"
            return f"({left} {op} {right})"

        if isinstance(expr, UnaryExpr):
            operand = self._compile_expr(expr.operand)
            if expr.op == '!':
                return f"(not ({operand}))"
            if expr.op == '~':
                return f"(~int({operand}))"
            return f"({expr.op}{operand})"

        if isinstance(expr, TernaryExpr):
            cond = self._compile_expr(expr.cond)
            true_e = self._compile_expr(expr.true_expr)
            false_e = self._compile_expr(expr.false_expr)
            return f"(({true_e}) if ({cond}) else ({false_e}))"

        if isinstance(expr, BranchAccess):
            node = expr.node1
            if expr.node2:
                n1 = self._compile_node_voltage(expr.node1, expr.node1_index, expr.node1_index2)
                n2 = self._compile_node_voltage(expr.node2, expr.node2_index, expr.node2_index2)
                if expr.access_type == 'V':
                    return f"({n1} - {n2})"
                return "0.0"  # I() not fully supported yet
            if expr.access_type == 'V':
                return self._compile_node_voltage(node, expr.node1_index, expr.node1_index2)
            return "0.0"

        if isinstance(expr, FunctionCall):
            return self._compile_function_call(expr)

        if isinstance(expr, MethodCall):
            return self._compile_method_call(expr)

        return "0.0"

    def _expr_is_integer(self, expr: Expr) -> bool:
        """Return True for expressions that Spectre evaluates in integer context."""
        integer_like, has_typed_integer = self._expr_integer_kind(expr)
        return integer_like and has_typed_integer

    def _expr_integer_kind(self, expr: Expr) -> tuple[bool, bool]:
        """Return (integer_like, contains_declared_integer)."""
        if isinstance(expr, NumberLiteral):
            raw = getattr(expr, "raw", None)
            if raw:
                token = raw.lstrip("+-")
                is_plain_integer = (
                    token.isdigit()
                    and "." not in token
                    and "e" not in token.lower()
                )
                return is_plain_integer, False
            return float(expr.value).is_integer(), False

        if isinstance(expr, Identifier):
            if self._param_types.get(expr.name) == ParamType.INTEGER:
                return True, True
            if self._var_types.get(expr.name) == ParamType.INTEGER:
                return True, True
            return False, False

        if isinstance(expr, ArrayAccess):
            if self._var_types.get(expr.name) == ParamType.INTEGER:
                return True, True
            return False, False

        if isinstance(expr, UnaryExpr):
            return self._expr_integer_kind(expr.operand)

        if isinstance(expr, BinaryExpr):
            if expr.op in {'%', '<<', '>>', '&', '|', '^'}:
                return True, True
            if expr.op in {'+', '-', '*', '/'}:
                left_like, left_typed = self._expr_integer_kind(expr.left)
                right_like, right_typed = self._expr_integer_kind(expr.right)
                return left_like and right_like, left_typed or right_typed
            return False, False

        if isinstance(expr, TernaryExpr):
            true_like, true_typed = self._expr_integer_kind(expr.true_expr)
            false_like, false_typed = self._expr_integer_kind(expr.false_expr)
            return true_like and false_like, true_typed or false_typed

        return False, False

    def _compile_node_voltage(self, node: str, index_expr=None, index_expr2=None) -> str:
        """Compile a node voltage reference."""
        if index_expr is not None:
            idx = self._compile_expr(index_expr)
            if index_expr2 is not None:
                idx2 = self._compile_expr(index_expr2)
                node_expr = f"self._resolve_dynamic_node({node!r}, {idx}, {idx2})"
                return f"self._get_voltage({node_expr}, nv)"
            node_expr = f"self._resolve_dynamic_node({node!r}, {idx})"
            return f"self._get_voltage({node_expr}, nv)"
        if self.static_branch_fastpath_codegen:
            slot = self._static_branch_read_slot_by_node.get(node)
            if slot is not None:
                return f"self._get_static_branch_voltage_by_slot({slot}, nv)"
            return f"self._get_static_branch_voltage({node!r}, nv)"
        return f"self._get_voltage({node!r}, nv)"

    def _compile_instance_target(self, expr: Expr) -> str:
        """Compile instance connection target into a node-name string expression."""
        if isinstance(expr, Identifier):
            return repr(expr.name)
        if isinstance(expr, ArrayAccess):
            idx = self._compile_expr(expr.index)
            return f"f'{expr.name}[{{int({idx})}}]'"
        # Fallback: allow unusual connection expressions as stringified value.
        return f"str({self._compile_expr(expr)})"

    def _compile_function_call(self, expr: FunctionCall) -> str:
        name = expr.name
        math_aliases = {
            'ln', 'log', 'exp', 'sqrt', 'abs', 'pow', 'min', 'max',
            'sin', 'cos', 'tan', 'tanh', 'floor', 'ceil',
        }
        if name.startswith('$') and name[1:] in math_aliases:
            name = name[1:]
        args = [self._compile_expr(a) for a in expr.args]

        if name == 'transition':
            key_expr, target, delay, rise, fall = self._compile_transition_call_parts(expr)
            return f"self._transition({key_expr}, time, {target}, {delay}, {rise}, {fall})"

        if name == 'slew':
            base_key = self._alloc_stateful_func_key("slew", expr)
            target = args[0] if len(args) > 0 else "0.0"
            maxrise = args[1] if len(args) > 1 else "0.0"
            maxfall = args[2] if len(args) > 2 else maxrise
            if self._in_loop_var:
                return f"self._slew(f'{base_key}_{{int(_loop_{self._in_loop_var})}}', time, {target}, {maxrise}, {maxfall})"
            return f"self._slew({base_key!r}, time, {target}, {maxrise}, {maxfall})"

        if name == 'idtmod':
            key = self._alloc_stateful_func_key("idtmod", expr)
            self._uses_idtmod = True
            x = args[0] if len(args) > 0 else "0.0"
            ic = args[1] if len(args) > 1 else "0.0"
            mod = args[2] if len(args) > 2 else "1.0"
            return f"self._idtmod({key!r}, time, {x}, {ic}, {mod})"

        if name == 'cross':
            # cross() as a function (in some contexts)
            base_key = self._alloc_stateful_func_key("last_crossing", expr) + "_fn"
            val = args[0]
            direction = args[1] if len(args) > 1 else "0"
            time_tol = args[2] if len(args) > 2 else "0.0"
            expr_tol = args[3] if len(args) > 3 else "1e-12"
            return f"self._check_cross({base_key!r}, time, {val}, {direction}, {time_tol}, {expr_tol})"

        if name == 'last_crossing':
            key = self._alloc_stateful_func_key("last_crossing", expr)
            val = args[0] if len(args) > 0 else "0.0"
            direction = args[1] if len(args) > 1 else "0"
            time_tol = args[2] if len(args) > 2 else "0.0"
            expr_tol = args[3] if len(args) > 3 else "1e-12"
            return f"self._last_crossing({key!r}, time, {val}, {direction}, {time_tol}, {expr_tol})"

        if name == 'ln':
            return f"math.log({args[0]})"
        if name == 'log':
            return f"math.log10({args[0]})"
        if name == 'exp':
            return f"math.exp({args[0]})"
        if name == 'sqrt':
            return f"math.sqrt({args[0]})"
        if name == 'abs':
            return f"abs({args[0]})"
        if name == 'pow':
            return f"pow({args[0]}, {args[1]})"
        if name == 'min':
            return f"min({args[0]}, {args[1]})"
        if name == 'max':
            return f"max({args[0]}, {args[1]})"
        if name == 'sin':
            return f"math.sin({args[0]})"
        if name == 'cos':
            return f"math.cos({args[0]})"
        if name == 'tan':
            return f"math.tan({args[0]})"
        if name == 'tanh':
            return f"math.tanh({args[0]})"
        if name == 'floor':
            return f"math.floor({args[0]})"
        if name == 'ceil':
            return f"math.ceil({args[0]})"
        if name == '$rdist_normal':
            # $rdist_normal(seed, mean, std_dev)
            # Also accept $rdist_normal(mean, std_dev) as a seedless shorthand.
            if len(args) >= 3:
                seed, mean, std = args[0], args[1], args[2]
            elif len(args) == 2:
                seed, mean, std = "None", args[0], args[1]
            elif len(args) == 1:
                seed, mean, std = "None", args[0], "1.0"
            else:
                seed, mean, std = "None", "0.0", "1.0"
            return f"self._rand_normal({seed}, {mean}, {std}, time)"
        if name == '$random':
            seed = args[0] if len(args) > 0 else "None"
            return f"self._rand_int32({seed})"
        if name == '$dist_uniform':
            # $dist_uniform(seed, lo, hi) or shorthand $dist_uniform(lo, hi)
            if len(args) >= 3:
                seed, lo, hi = args[0], args[1], args[2]
            elif len(args) == 2:
                seed, lo, hi = "None", args[0], args[1]
            elif len(args) == 1:
                seed, lo, hi = "None", "0.0", args[0]
            else:
                seed, lo, hi = "None", "0.0", "1.0"
            return f"self._rand_uniform({seed}, {lo}, {hi})"
        if name == '$fopen':
            filename = args[0] if len(args) > 0 else "'output.txt'"
            mode = args[1] if len(args) > 1 else "'w'"
            return f"self._fopen({filename}, {mode})"

        raise CompilationError(f"Unsupported Verilog-A function call: {name}()")

    def _compile_transition_call_parts(
        self,
        expr: FunctionCall,
    ) -> Tuple[str, str, str, str, str]:
        args = [self._compile_expr(a) for a in expr.args]
        base_key = self._alloc_stateful_func_key("transition", expr)
        target = args[0] if len(args) > 0 else "0.0"
        delay = args[1] if len(args) > 1 else "0.0"
        rise = args[2] if len(args) > 2 else "0.0"
        fall = args[3] if len(args) > 3 else rise
        if self._in_loop_var:
            key_expr = f"f'{base_key}_{{int(_loop_{self._in_loop_var})}}'"
        else:
            key_expr = repr(base_key)
        return key_expr, target, delay, rise, fall

    def _compile_method_call(self, expr: MethodCall) -> str:
        """Compile method calls like conf.substr(i, i)."""
        obj = expr.obj
        method = expr.method
        args = [self._compile_expr(a) for a in expr.args]

        if method == 'substr':
            # Verilog-A substr(start, end) → Python string slice
            return f"self.params[{obj!r}][int({args[0]}):int({args[1]})+1]"

        return "''"  # unknown method

    def _compile_expr_with_loop_var(self, expr: Expr, loop_var: str) -> str:
        """Compile expression using loop variable from local scope."""
        code = self._compile_expr(expr)
        code = code.replace(f"self.state[{loop_var!r}]", f"_loop_{loop_var}")
        return code

    def _compile_statement_with_loop_var(self, stmt, indent, loop_var) -> List[str]:
        """Compile statement but use loop var from local scope."""
        lines = self._compile_statement(stmt, indent)
        new_lines = []
        for line in lines:
            # Replace state access for loop var with local var
            # But only in index positions, not as assignment targets at top level
            new_line = line.replace(
                f"self.state[{loop_var!r}]",
                f"_loop_{loop_var}"
            )
            new_lines.append(new_line)
        return new_lines

    def _compile_post_update_statement_with_loop_var(self, stmt, indent, loop_var) -> List[str]:
        lines = self._compile_post_update_statement(stmt, indent)
        new_lines = []
        for line in lines:
            new_lines.append(
                line.replace(f"self.state[{loop_var!r}]", f"_loop_{loop_var}")
            )
        return new_lines

    def _compile_refresh_statement_with_loop_var(self, stmt, indent, loop_var) -> List[str]:
        lines = self._compile_refresh_statement(stmt, indent)
        new_lines = []
        for line in lines:
            new_lines.append(
                line.replace(f"self.state[{loop_var!r}]", f"_loop_{loop_var}")
            )
        return new_lines

    def _compile_initial_step_statement_with_loop_var(self, stmt, indent, loop_var) -> List[str]:
        lines = self._compile_initial_step_statement(stmt, indent)
        new_lines = []
        for line in lines:
            new_lines.append(
                line.replace(f"self.state[{loop_var!r}]", f"_loop_{loop_var}")
            )
        return new_lines

    def _eval_expr_static(self, expr: Expr, env: Optional[Dict[str, Any]] = None) -> Any:
        """Evaluate an elaboration-time constant expression statically."""
        env = env or {}
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, UnaryExpr) and expr.op == '-':
            return -self._eval_expr_static(expr.operand, env)
        if isinstance(expr, UnaryExpr) and expr.op == '+':
            return self._eval_expr_static(expr.operand, env)
        if isinstance(expr, Identifier):
            if expr.name == 'inf':
                return float('inf')
            if expr.name in env:
                return env[expr.name]
            return 0
        if isinstance(expr, BinaryExpr):
            lv = self._eval_expr_static(expr.left, env)
            rv = self._eval_expr_static(expr.right, env)
            if expr.op == '+':
                return lv + rv
            if expr.op == '-':
                return lv - rv
            if expr.op == '*':
                return lv * rv
            if expr.op == '/':
                return lv / rv if rv != 0 else 0
            if expr.op == '%':
                return lv % rv if rv != 0 else 0
        if isinstance(expr, FunctionCall):
            args = [self._eval_expr_static(arg, env) for arg in expr.args]
            name = expr.name[1:] if expr.name.startswith('$') else expr.name
            funcs = {
                'abs': abs,
                'sqrt': math.sqrt,
                'exp': math.exp,
                'ln': math.log,
                'log': math.log10,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'tanh': math.tanh,
                'floor': math.floor,
                'ceil': math.ceil,
                'min': min,
                'max': max,
                'pow': pow,
            }
            fn = funcs.get(name)
            if fn is not None:
                try:
                    return fn(*args)
                except Exception:
                    return 0
        return 0


def compile_va_file(
    va_path: str,
    source_dir: str = None,
    static_branch_fastpath_codegen: bool = False,
    indexed_state_fastpath_codegen: bool = False,
) -> type:
    """
    Compile a .va file into a Python model class.

    Usage:
        ModelClass = compile_va_file('veriloga/L2_comparator.va')
        model = ModelClass()
        model.node_map = {'DCMPP': 'out_p', 'CLK': 'clk', ...}
    """
    from evas.compiler.parser import parse
    from evas.compiler.preprocessor import preprocess

    if source_dir is None:
        source_dir = str(Path(va_path).parent)

    source = Path(va_path).read_text(encoding='utf-8', errors='replace')
    pp_src, defines, default_trans = preprocess(source, source_dir=source_dir)
    module = parse(pp_src)
    module.defines = defines

    if default_trans is None:
        default_trans = 1e-12

    return compile_module(
        module,
        default_trans,
        static_branch_fastpath_codegen=static_branch_fastpath_codegen,
        indexed_state_fastpath_codegen=indexed_state_fastpath_codegen,
    )
