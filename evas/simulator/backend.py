"""
compiler_backend.py — Compile Verilog-A AST into executable Python model classes.

Takes a parsed Module AST and generates a Python class that implements
the behavioral model with proper event handling, transition operators,
and state variable management.
"""
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evas.compiler.ast_nodes import *
from evas.simulator.engine import (
    AboveDetector,
    CrossDetector,
    TransitionState,
)


class CompilationError(Exception):
    pass


class CompiledModel:
    """Base class for compiled Verilog-A models."""
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
    _rust_static_affine_ops = ()
    _static_branch_fastpath_codegen = False
    _cmp_eps: float = 0.0
    _needs_future_node_voltages: bool = False
    _has_dynamic_breakpoints: bool = True
    _has_post_update_events: bool = True
    _uses_bound_step: bool = True

    def __init__(self):
        self.params: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.arrays: Dict[str, Dict[int, Any]] = {}
        self.transitions: Dict[str, TransitionState] = {}
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
        self._event_node_cross_directions: Dict[str, int] = {}
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
            "cross_fires": 0,
            "above_fires": 0,
            "static_branch_fastpath_fallbacks": 0,
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
        self._static_branch_indexed_values: Optional[List[float]] = None
        self._static_branch_read_node_ids: tuple[int, ...] = ()
        self._static_branch_write_node_ids: tuple[int, ...] = ()
        self._static_branch_write_external_nodes: tuple[str, ...] = ()
        self._node_resolution_cache_enabled: bool = False
        self._node_resolution_cache: Dict[str, str] = {}
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

    def _set_static_branch_fastpath_enabled(self, enabled: bool):
        """Enable opt-in helpers for static, non-dynamic branch read/write code."""
        self._static_branch_fastpath_enabled = bool(enabled)
        if enabled:
            self._perf_stats["static_branch_fastpath_fallbacks"] = 0
        for child in self._child_models:
            child._set_static_branch_fastpath_enabled(enabled)

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
        for ts in self.transitions.values():
            bp = ts.next_breakpoint(time, min_ramp)
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

    def _invalidate_timer_breakpoint_cache(self):
        self._timer_state_version += 1
        self._timer_breakpoint_cache_version = -1
        self._timer_breakpoint_cache = None
        self._perf_stats["timer_state_updates"] += 1

    def _set_timer_state(self, key: str, value: float):
        self.timer_states[key] = float(value)
        self._invalidate_timer_breakpoint_cache()

    def _set_timer_last_fired(self, key: str, value: float):
        self.timer_last_fired[key] = float(value)
        self._invalidate_timer_breakpoint_cache()

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
        best: Optional[float] = None
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
        self.timer_kinds[key] = "periodic"
        p = float(period)
        if p <= 0.0 or not math.isfinite(p):
            return False
        if key not in self.timer_states:
            next_fire = float(start) if start is not None else p
            if not math.isfinite(next_fire):
                next_fire = p
            if time > next_fire + 1e-18:
                missed = math.floor((time - next_fire) / p) + 1
                self._set_timer_state(key, next_fire + missed * p)
                self._perf_stats["timer_periodic_skips"] += 1
                return False
            self._set_timer_state(key, next_fire)
        next_fire = self.timer_states[key]
        if time > next_fire + 1e-18:
            missed = math.floor((time - next_fire) / p) + 1
            self._set_timer_state(key, next_fire + missed * p)
            self._perf_stats["timer_periodic_skips"] += 1
            return False
        due = time >= next_fire - 1e-18
        if due:
            self._event_time = time
            self._event_interpolated_nodes = set()
        return due

    def _reschedule_timer(self, key: str, time: float, period: float):
        p = float(period)
        if p <= 0.0 or not math.isfinite(p) or key not in self.timer_states:
            return
        self._set_timer_state(key, self.timer_states[key] + p)
        self._perf_stats["timer_reschedules"] += 1

    def _check_timer_at(self, key: str, time: float, target: float) -> bool:
        self._perf_stats["timer_absolute_checks"] += 1
        self.timer_kinds[key] = "absolute"
        tgt = float(target)
        if not math.isfinite(tgt):
            return False
        first_seen = key not in self.timer_states
        if first_seen or abs(self.timer_states[key] - tgt) > 1e-18:
            self._set_timer_state(key, tgt)
        if first_seen and time > tgt + 1e-18:
            self._set_timer_last_fired(key, tgt)
            self._perf_stats["timer_absolute_expirations"] += 1
            return False
        armed_target = self.timer_states[key]
        last_fired = self.timer_last_fired.get(key)
        if last_fired is not None and abs(last_fired - armed_target) <= 1e-18:
            return False
        if time >= armed_target - 1e-18:
            self._set_timer_last_fired(key, armed_target)
            self._perf_stats["timer_absolute_fires"] += 1
            self._event_time = time
            self._event_interpolated_nodes = set()
            return True
        return False

    def _check_timer(self, key: str, time: float, period: float, start: Optional[float] = None) -> bool:
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

    def _get_voltage(self, node: str, node_voltages: Dict[str, float]) -> float:
        """Get voltage of a node, resolving through node_map."""
        if not self.node_map and not self._event_context_active:
            indexed_value = self._read_indexed_voltage(node, node)
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
                    frac = (float(self._event_time) - t0) / (t1 - t0)
                    frac = max(0.0, min(1.0, frac))
                    v0 = self._step_prev_node_voltages[ext]
                    if ext in node_voltages:
                        v1 = node_voltages[ext]
                    elif ext in self._step_curr_node_voltages:
                        v1 = self._step_curr_node_voltages[ext]
                    else:
                        v1 = v0
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
            if self._indexed_output_writer is not None:
                self._indexed_output_writer(node, value)
            return

        if node not in self.output_nodes:
            self._output_nodes_version += 1
        self.output_nodes[node] = value
        ext = self._resolve_external_node(node)
        node_voltages[ext] = value
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
        if self._indexed_output_writer is not None:
            self._indexed_output_writer(ext, value)

    def _transition(self, key: str, time: float, target: float,
                    delay: float = 0.0, rise: float = 0.0, fall: float = 0.0) -> float:
        """Evaluate a transition operator."""
        if self._initial_condition_mode:
            ts = self.transitions.get(key)
            if ts is None:
                ts = TransitionState(current_val=target)
                self.transitions[key] = ts
            ts.current_val = target
            ts.target_val = target
            ts.start_val = target
            ts.start_time = time
            ts.delay = 0.0
            ts.active = False
            return target
        if key not in self.transitions:
            self.transitions[key] = TransitionState(current_val=target)
            return target
        ts = self.transitions[key]
        # Advance current_val to the actual value at this time before updating target.
        # Without this, a new target set at t overwrites the in-progress transition
        # before evaluate() can commit its endpoint into current_val.
        ts.evaluate(time)
        ts.set_target(time, target, delay, rise, fall, self.default_transition)
        return ts.evaluate(time)

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
        fired = self.cross_detectors[key].check(time, val, time_tol=time_tol, expr_tol=expr_tol)
        if fired:
            cross_time = float(self.cross_detectors[key].t_cross)
            if cross_time + max(1e-18, float(time_tol or 0.0)) < self._step_latest_cross_event_time:
                # Multiple cross() statements can trigger inside one simulator
                # step. Spectre applies their event bodies in chronological
                # crossing order, not source order. EVAS does not yet replay a
                # full event queue, so suppress a retrograde event body instead
                # of letting an earlier crossing discovered later overwrite the
                # state from a later crossing.
                self.cross_detectors[key].last_triggered = False
                return False
            self._step_latest_cross_event_time = max(
                self._step_latest_cross_event_time,
                cross_time,
            )
            self._step_event_fired = True
            self._perf_stats["cross_fires"] += 1
            trigger_dir = self.cross_detectors[key].last_trigger_direction or direction
            trigger_went_beyond = self.cross_detectors[key].last_trigger_went_beyond
            prev_event_time = self._event_time
            prev_cross_directions = dict(self._event_node_cross_directions)
            self._event_time = cross_time

            def resolve_node(node: str) -> str:
                return self._resolve_external_node(node)

            def event_node_value(node: str) -> float:
                ext = resolve_node(node)
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
            self._event_interpolated_nodes = set(interp_nodes or [])
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
        return fired

    def _last_crossing(self, key: str, time: float, val: float, direction: int = 0,
                       time_tol: float = 0.0, expr_tol: float = 1e-12) -> float:
        """Return most recent crossing time (approximation for last_crossing())."""
        if key not in self.cross_detectors:
            self.cross_detectors[key] = CrossDetector(direction=direction)
        cd = self.cross_detectors[key]
        cd.check(time, val, time_tol=time_tol, expr_tol=expr_tol)
        return cd.t_cross

    def _check_above(self, key: str, time: float, val: float, direction: int = 1) -> bool:
        if key not in self.above_detectors:
            self.above_detectors[key] = AboveDetector(direction=direction)
        fired = self.above_detectors[key].check(time, val)
        if fired:
            self._perf_stats["above_fires"] += 1
            self._step_event_fired = True
            self._event_time = self.above_detectors[key].t_cross
            self._event_interpolated_nodes = set()
            self._event_node_cross_directions = {}
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
        if name in self.arrays and idx in self.arrays[name]:
            return self.arrays[name][idx]
        return 0

    def _array_set(self, name: str, idx: int, val: Any):
        if name not in self.arrays:
            self.arrays[name] = {}
        self.arrays[name][idx] = val

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

    def _rand_normal(self, seed: Optional[float], mean: float, std: float) -> float:
        rng = self._seed_to_stream(seed)
        return rng.gauss(float(mean), float(std))

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
    )
    return compiler.compile()


class _ModuleCompiler:
    def __init__(
        self,
        module: Module,
        default_transition: float = None,
        static_branch_fastpath_codegen: bool = False,
    ):
        self.module = module
        self.default_transition = default_transition or 1e-12
        self.static_branch_fastpath_codegen = bool(static_branch_fastpath_codegen)
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
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
        self._param_types = {p.name: p.param_type for p in module.parameters}
        self._var_types = {v.name: v.var_type for v in module.variables}

    def compile(self) -> type:
        """Generate and return a compiled model class."""
        # Build the class dynamically
        mod = self.module

        self._validate_spectre_operator_rules()

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
        lines.append("        if self._initial_step_done:")
        lines.append("            return")
        lines.append("        self._initial_step_done = True")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.initial_step(nv, time)")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_initial_step_statement(stmt, 2)
                lines.extend(stmt_lines)

        lines.append("        pass")  # ensure method has body

        # Generate evaluate method
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
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
        lines.append("        self._event_time = time")
        lines.append("        self._bound_step = 0.0")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.evaluate(nv, time)")

        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_statement(stmt, 2)
                lines.extend(stmt_lines)

        lines.append("        for _ch in self._child_models:")
        lines.append("            _bs = _ch._bound_step")
        lines.append("            if _bs > 0.0 and (self._bound_step <= 0.0 or _bs < self._bound_step):")
        lines.append("                self._bound_step = _bs")

        lines.append("        pass")

        lines.append("")
        lines.append("    def post_update_events(self, nv, time):")
        lines.append("        self._event_time = time")
        lines.append("        _post_event_fired = False")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_post_update_statement(stmt, 2)
                lines.extend(stmt_lines)
        lines.append("        return _post_event_fired")

        lines.append("")
        lines.append("    def refresh_outputs(self, nv, time):")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_refresh_statement(stmt, 2)
                lines.extend(stmt_lines)
        lines.append("        pass")

        # Generate final_step method
        lines.append("")
        lines.append("    def final_step(self, nv, time):")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                if isinstance(stmt, EventStatement):
                    if self._is_final_step_event(stmt.event):
                        body_lines = self._compile_statement(stmt.body, 2)
                        lines.extend(body_lines)
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
        cls._uses_idtmod = self._uses_idtmod
        cls._needs_future_node_voltages = self._needs_future_node_voltages
        cls._static_branch_fastpath_codegen = self.static_branch_fastpath_codegen
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

    def _collect_static_branch_io(self, stmt) -> Dict[str, Any]:
        acc = self._empty_static_branch_io()
        self._collect_static_branch_io_from_stmt(stmt, acc, in_event_body=False)
        return acc

    def _collect_rust_static_affine_ops(
        self,
        stmt,
    ) -> Tuple[Tuple[str, str, float, float], ...]:
        """Collect a conservative Rust-lowerable static affine model body.

        The current Rust prototype only handles unconditional voltage-domain
        contributions shaped as ``V(out) <+ gain * V(in) + bias`` with literal
        numeric coefficients.  Anything involving events, state variables,
        parameters, dynamic bus indexes, function calls, or differential
        branches falls back to the normal Python evaluator.
        """
        ops: List[Tuple[str, str, float, float]] = []
        if not self._collect_rust_static_affine_ops_from_stmt(stmt, ops):
            return ()
        return tuple(ops)

    def _collect_rust_static_affine_ops_from_stmt(
        self,
        stmt,
        ops: List[Tuple[str, str, float, float]],
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
        ops.append((read_node or branch.node1, branch.node1, float(gain), float(bias)))
        return True

    def _rust_affine_expr(self, expr: Expr) -> Optional[Tuple[Optional[str], float, float]]:
        if isinstance(expr, NumberLiteral):
            return None, 0.0, float(expr.value)

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
                return node, -gain, -bias
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
                left_const = self._rust_constant_expr(expr.left)
                right_const = self._rust_constant_expr(expr.right)
                if left_const is not None and right is not None:
                    return right[0], right[1] * left_const, right[2] * left_const
                if right_const is not None and left is not None:
                    return left[0], left[1] * right_const, left[2] * right_const
                return None

            if expr.op == "/":
                right_const = self._rust_constant_expr(expr.right)
                if right_const is None or right_const == 0.0 or left is None:
                    return None
                return left[0], left[1] / right_const, left[2] / right_const

        return None

    def _combine_rust_affine(
        self,
        left: Tuple[Optional[str], float, float],
        right: Tuple[Optional[str], float, float],
    ) -> Optional[Tuple[Optional[str], float, float]]:
        left_node, left_gain, left_bias = left
        right_node, right_gain, right_bias = right
        if left_node is not None and right_node is not None and left_node != right_node:
            return None
        return (
            left_node if left_node is not None else right_node,
            left_gain + right_gain,
            left_bias + right_bias,
        )

    def _rust_constant_expr(self, expr: Expr) -> Optional[float]:
        affine = self._rust_affine_expr(expr)
        if affine is None:
            return None
        node, gain, bias = affine
        if node is not None or gain != 0.0:
            return None
        return float(bias)

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
            for s in stmt.statements:
                lines.extend(self._compile_statement(s, indent))

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
                lines.extend(self._compile_initial_step_statement(stmt.body, indent))

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

            lines.append(f"{prefix}self.state[{loop_var!r}] = {init_val}")
            lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
            cond_code = self._compile_expr_with_loop_var(stmt.cond, loop_var)
            lines.append(f"{prefix}while {cond_code}:")
            lines.append(f"{prefix}    self.state[{loop_var!r}] = _loop_{loop_var}")
            body_lines = self._compile_initial_step_statement_with_loop_var(stmt.body, indent + 1, loop_var)
            lines.extend(body_lines)
            update_code = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
            lines.append(f"{prefix}    _loop_{loop_var} = {update_code}")
            lines.append(f"{prefix}self.state[{loop_var!r}] = _loop_{loop_var}")

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
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.ABOVE:
                if self._event_requires_post_update(event):
                    return []
                key = self._alloc_event_key("above", event)
                expr = self._compile_expr(event.args[0])
                direction = event.direction if event.direction is not None else 1
                lines.append(f"{prefix}if self._check_above({key!r}, time, {expr}, {direction}):")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.TIMER:
                key = self._alloc_event_key("timer", event)
                if len(event.args) == 2:
                    start_expr = self._compile_expr(event.args[0])
                    period_expr = self._compile_expr(event.args[1])
                    lines.append(f"{prefix}if self._check_timer_due({key!r}, time, {period_expr}, {start_expr}):")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._event_context_active = True")
                    body_lines = self._compile_statement(stmt.body, indent + 1)
                    lines.extend(body_lines)
                    if not body_lines:
                        lines.append(f"{prefix}    pass")
                    lines.append(f"{prefix}    self._event_context_active = False")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._reschedule_timer({key!r}, time, {period_expr})")
                else:
                    target_expr = self._compile_expr(event.args[0])
                    lines.append(f"{prefix}if self._check_timer_at({key!r}, time, {target_expr}):")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                    lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                    lines.append(f"{prefix}    self._event_context_active = True")
                    body_lines = self._compile_statement(stmt.body, indent + 1)
                    lines.extend(body_lines)
                    if not body_lines:
                        lines.append(f"{prefix}    pass")
                    lines.append(f"{prefix}    self._event_context_active = False")
                    lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
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
                    expr = self._compile_expr(e.args[0])
                    direction = e.direction if e.direction is not None else 1
                    conditions.append(f"self._check_above({key!r}, time, {expr}, {direction})")
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
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")

        return lines

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
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
                lines.append(f"{prefix}    self._event_node_cross_directions = {{}}")
                lines.append(f"{prefix}    self._event_time = time")
                lines.append(f"{prefix}    _post_event_fired = True")
            elif event.event_type == EventType.ABOVE:
                if not self._event_requires_post_update(event):
                    return lines
                key = self._alloc_event_key("above", event)
                expr = self._compile_expr(event.args[0])
                direction = event.direction if event.direction is not None else 1
                lines.append(f"{prefix}if self._check_above({key!r}, time, {expr}, {direction}):")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
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
                    expr = self._compile_expr(e.args[0])
                    direction = e.direction if e.direction is not None else 1
                    conditions.append(f"self._check_above({key!r}, time, {expr}, {direction})")
            if conditions:
                hit_vars = []
                for idx, cond in enumerate(conditions):
                    var = f"_post_event_hit_{idx}"
                    hit_vars.append(var)
                    lines.append(f"{prefix}{var} = {cond}")
                cond = ' or '.join(hit_vars)
                lines.append(f"{prefix}if {cond}:")
                lines.append(f"{prefix}    self._event_context_active = True")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_context_active = False")
                lines.append(f"{prefix}    self._event_interpolated_nodes = set()")
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

        lines.append(f"{prefix}self.state[{loop_var!r}] = {init_val}")
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self.state[{loop_var!r}] = _loop_{loop_var}")
        body_lines = self._compile_post_update_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self.state[{loop_var!r}] = _loop_{loop_var}")

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

        lines.append(f"{prefix}self.state[{loop_var!r}] = {init_val}")
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self.state[{loop_var!r}] = _loop_{loop_var}")
        body_lines = self._compile_refresh_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self.state[{loop_var!r}] = _loop_{loop_var}")

        self._in_loop_var = prev_loop_var
        return lines

    def _compile_contribution(self, stmt: Contribution, indent) -> List[str]:
        prefix = '    ' * indent
        branch = stmt.branch
        node = branch.node1
        expr = self._compile_expr(stmt.expr)
        if branch.node2 is not None:
            node2_expr = self._compile_node_voltage(branch.node2, branch.node2_index, branch.node2_index2)
            expr = f"(({node2_expr}) + ({expr}))"

        if branch.node1_index is not None:
            # Dynamic array-indexed port: V(DOUT[i]) <+ or V(DOUT[i][j]) <+
            idx_expr = self._compile_expr(branch.node1_index)
            if branch.node1_index2 is not None:
                idx_expr2 = self._compile_expr(branch.node1_index2)
                node_expr = f"self._format_dynamic_node({node!r}, {idx_expr}, {idx_expr2})"
                return [f"{prefix}self._set_output({node_expr}, {expr}, nv)"]
            node_expr = f"self._format_dynamic_node({node!r}, {idx_expr})"
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
            line = f"self.state[{name!r}] = {val}"
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
        lines.append(f"{prefix}self.state[{loop_var!r}] = {init_val}")
        # Replace loop var references in condition
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self.state[{loop_var!r}] = _loop_{loop_var}")
        body_lines = self._compile_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self.state[{loop_var!r}] = _loop_{loop_var}")

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
                node_expr = f"self._format_dynamic_node({node!r}, {idx}, {idx2})"
                return f"self._get_voltage({node_expr}, nv)"
            node_expr = f"self._format_dynamic_node({node!r}, {idx})"
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
            base_key = self._alloc_stateful_func_key("transition", expr)
            target = args[0] if len(args) > 0 else "0.0"
            delay = args[1] if len(args) > 1 else "0.0"
            rise = args[2] if len(args) > 2 else "0.0"
            fall = args[3] if len(args) > 3 else rise
            if self._in_loop_var:
                # Dynamic key per loop iteration
                return f"self._transition(f'{base_key}_{{int(_loop_{self._in_loop_var})}}', time, {target}, {delay}, {rise}, {fall})"
            return f"self._transition({base_key!r}, time, {target}, {delay}, {rise}, {fall})"

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
            return f"self._rand_normal({seed}, {mean}, {std})"
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
    )
