"""
simulator.py — Event-driven transient simulator for compiled Verilog-A models.

This is a behavioral simulator that:
- Drives input stimuli (voltage sources)
- Evaluates Verilog-A model behavior at each timestep
- Detects zero-crossing events (@cross, @above)
- Handles transition() operator for smooth output transitions
- Supports @initial_step events
- Records output waveforms
"""
import math
import time as _wall_time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from evas.simulator.indexed import (
    IndexedVoltageArray,
    IndexedVoltageSnapshotter,
    build_indexed_model_io_plan,
)


class _LazyFutureNodeVoltages:
    """Resolve future source values only when cross exact-touch handling needs them."""

    __slots__ = ("_source_waveforms", "_time")

    def __init__(self, source_waveforms: Dict[str, Callable[[float], float]], time: float):
        self._source_waveforms = source_waveforms
        self._time = float(time)

    def _with_fallback(self, fallback: Dict[str, float]):
        return _BoundLazyFutureNodeVoltages(self._source_waveforms, self._time, fallback)


class _BoundLazyFutureNodeVoltages:
    __slots__ = ("_source_waveforms", "_time", "_fallback")

    def __init__(
        self,
        source_waveforms: Dict[str, Callable[[float], float]],
        time: float,
        fallback: Dict[str, float],
    ):
        self._source_waveforms = source_waveforms
        self._time = time
        self._fallback = fallback

    def get(self, node: str, default: float = None):
        waveform = self._source_waveforms.get(node)
        if waveform is not None:
            return waveform(self._time)
        return self._fallback.get(node, default)


@dataclass
class TransitionState:
    """State for a single transition() operator instance."""
    current_val: float = 0.0
    target_val: float = 0.0
    start_time: float = 0.0
    start_val: float = 0.0
    delay: float = 0.0
    rise_time: float = 1e-12  # default 1ps
    fall_time: float = 1e-12
    active: bool = False

    def evaluate(self, time: float) -> float:
        """Compute the transition output at the given time."""
        if not self.active:
            return self.current_val

        t_begin = self.start_time + self.delay
        going_up = self.target_val > self.start_val
        ramp_time = self.rise_time if going_up else self.fall_time

        if time < t_begin:
            return self.start_val
        elif time >= t_begin + ramp_time:
            self.current_val = self.target_val
            self.active = False
            return self.target_val
        else:
            # Linear interpolation
            frac = (time - t_begin) / ramp_time if ramp_time > 0 else 1.0
            frac = max(0.0, min(1.0, frac))
            val = self.start_val + frac * (self.target_val - self.start_val)
            self.current_val = val
            return val

    def set_target(self, time: float, target: float, delay: float = 0.0,
                   rise: float = 0.0, fall: float = 0.0,
                   default_transition: float = 1e-12):
        """Set a new target value, triggering a transition if changed."""
        if rise <= 0:
            rise = default_transition
        if fall <= 0:
            fall = default_transition

        changed = abs(target - self.target_val) > 1e-15
        if changed and self.active:
            t_begin = self.start_time + self.delay
            going_up = self.target_val > self.start_val
            ramp_time = self.rise_time if going_up else self.fall_time
            in_active_region = time >= t_begin and time < t_begin + ramp_time
            if in_active_region:
                vi = self.current_val
                if going_up:
                    basis = self.target_val if target < vi else self.start_val
                    readjust_time = fall if target < vi else rise
                else:
                    basis = self.start_val if target < vi else self.target_val
                    readjust_time = fall if target < vi else rise
                slope = (target - basis) / readjust_time if readjust_time > 0 else 0.0
                if abs(slope) > 1e-30:
                    # Verilog-AMS interrupted-transition semantics: a changed
                    # input during an active transition readjusts the original
                    # transition line; it does not restart a full rise/fall
                    # interval from the instantaneous output value.
                    self.start_val = basis
                    self.start_time = time - (vi - basis) / slope
                    self.delay = 0.0
                    self.target_val = target
                    self.rise_time = rise
                    self.fall_time = fall
                    self.current_val = vi
                    self.active = True
                    return

        if changed or not self.active:
            if abs(target - self.current_val) > 1e-15:
                self.start_val = self.current_val
                self.target_val = target
                self.start_time = time
                self.delay = delay
                self.rise_time = rise
                self.fall_time = fall
                self.active = True
            else:
                self.target_val = target
                self.current_val = target
                self.active = False

    def next_breakpoint(self, time: float, min_ramp_time: float = 0.0) -> Optional[float]:
        """Return next important time for this transition, or None."""
        if not self.active:
            return None
        t_begin = self.start_time + self.delay
        going_up = self.target_val > self.start_val
        ramp_time = self.rise_time if going_up else self.fall_time
        t_end = t_begin + ramp_time

        best: Optional[float] = None
        if t_begin > time:
            best = t_begin
        # Add interior breakpoints only for transitions long enough to be
        # observable at the current tran maxstep. Very short transition()
        # filters are solver-internal smoothing; forcing their midpoints into
        # tran.csv can create non-Spectre-like digital bus glitches.
        if ramp_time > max(0.0, min_ramp_time):
            for frac in (0.25, 0.5, 0.75):
                t_inner = t_begin + frac * ramp_time
                if t_inner > time and t_inner < t_end:
                    if best is None or t_inner < best:
                        best = t_inner
        if t_end > time and (best is None or t_end < best):
            best = t_end
        return best


@dataclass
class CrossDetector:
    """Detect zero-crossings in a signal expression."""
    prev_val: float = 0.0
    prev_time: float = 0.0
    pprev_val: float = 0.0   # second-to-last value for slope extrapolation
    pprev_time: float = 0.0
    initialized: bool = False
    direction: int = 0  # +1=rising, -1=falling, 0=both
    last_triggered: bool = False  # set by check(), read by simulator
    t_cross: float = 0.0  # interpolated exact crossing time
    last_cross_time: float = -1.0  # debounced last-trigger timestamp
    last_trigger_direction: int = 0
    last_trigger_went_beyond: bool = False
    pending_touch_direction: int = 0  # +1/-1 after touching zero from below/above
    pending_touch_time: float = 0.0

    def check(self, time: float, val: float,
              time_tol: float = 0.0, expr_tol: float = 1e-12) -> bool:
        """Check if a zero crossing occurred. Returns True if triggered."""
        if not self.initialized:
            self.pprev_val = self.prev_val = val
            self.pprev_time = self.prev_time = time
            self.initialized = True
            self.last_triggered = False
            self.last_trigger_direction = 0
            self.last_trigger_went_beyond = False
            self.pending_touch_direction = 0
            return False

        e_tol = abs(float(expr_tol)) if expr_tol is not None else 1e-12
        triggered = False
        trigger_direction = 0
        trigger_went_beyond = False
        cross_time = 0.0

        def interpolate_cross_time() -> float:
            dv = val - self.prev_val
            frac = max(0.0, min(1.0, -self.prev_val / dv)) if abs(dv) > 1e-30 else 0.0
            return self.prev_time + frac * (time - self.prev_time)

        if self.direction >= 0 and self.prev_val < -e_tol:
            if val > e_tol:
                triggered = True
                trigger_direction = 1
                trigger_went_beyond = True
                cross_time = interpolate_cross_time()
            elif abs(val) <= e_tol:
                triggered = True
                trigger_direction = 1
                trigger_went_beyond = False
                cross_time = interpolate_cross_time()
        if not triggered and self.direction <= 0 and self.prev_val > e_tol:
            if val < -e_tol:
                triggered = True
                trigger_direction = -1
                trigger_went_beyond = True
                cross_time = interpolate_cross_time()
            elif abs(val) <= e_tol:
                triggered = True
                trigger_direction = -1
                trigger_went_beyond = False
                cross_time = interpolate_cross_time()

        if triggered:
            self.t_cross = cross_time
            t_tol = max(0.0, float(time_tol or 0.0))
            if self.last_cross_time >= 0.0 and abs(self.t_cross - self.last_cross_time) <= t_tol:
                triggered = False
            else:
                self.last_cross_time = self.t_cross
            self.pending_touch_direction = 0
            # Clamp val to the post-crossing side to prevent immediate re-trigger.
            sign_eps = max(e_tol, 1e-18)
            if trigger_went_beyond:
                if trigger_direction < 0:
                    val = min(val, -sign_eps)
                elif trigger_direction > 0:
                    val = max(val, sign_eps)

        self.pprev_val = self.prev_val
        self.pprev_time = self.prev_time
        self.prev_time = time
        self.prev_val = val
        self.last_triggered = triggered
        self.last_trigger_direction = trigger_direction if triggered else 0
        self.last_trigger_went_beyond = trigger_went_beyond if triggered else False
        return triggered

    def next_breakpoint(self) -> Optional[float]:
        """Predict the next zero-crossing time by linear extrapolation."""
        if self.pending_touch_direction != 0:
            probe_time = self.pending_touch_time + 1e-18
            if self.prev_time < probe_time:
                return probe_time
        dt = self.prev_time - self.pprev_time
        if dt <= 1e-30:
            return None
        rate = (self.prev_val - self.pprev_val) / dt
        if self.direction >= 0 and self.prev_val < 0 and rate > 0:
            return self.prev_time + (-self.prev_val / rate)
        if self.direction <= 0 and self.prev_val > 0 and rate < 0:
            return self.prev_time + (-self.prev_val / rate)
        return None

    def would_cross(self, val: float, expr_tol: float = 1e-12) -> bool:
        """Check if a crossing would occur without updating state."""
        if not self.initialized:
            return False
        e_tol = abs(float(expr_tol)) if expr_tol is not None else 1e-12
        if self.pending_touch_direction > 0 and self.direction >= 0 and val > e_tol:
            return True
        if self.pending_touch_direction < 0 and self.direction <= 0 and val < -e_tol:
            return True
        if self.direction >= 0 and self.prev_val < -e_tol and val >= -e_tol:
            return True
        if self.direction <= 0 and self.prev_val > e_tol and val <= e_tol:
            return True
        return False


@dataclass
class AboveDetector:
    """Detect above() condition: triggers when signal crosses above threshold."""
    prev_val: float = 0.0
    prev_time: float = 0.0
    pprev_val: float = 0.0
    pprev_time: float = 0.0
    initialized: bool = False
    direction: int = 1  # +1=above only
    last_triggered: bool = False
    t_cross: float = 0.0

    def check(self, time: float, val: float) -> bool:
        if not self.initialized:
            self.pprev_val = self.prev_val = val
            self.pprev_time = self.prev_time = time
            self.initialized = True
            triggered = self.direction >= 0 and val >= -1e-12
            self.last_triggered = triggered
            if triggered:
                self.t_cross = time
                self.prev_val = max(val, 0.0)
            return triggered

        triggered = False
        if self.direction >= 0 and self.prev_val < 0 and val >= -1e-12:
            triggered = True

        if triggered:
            dv = val - self.prev_val
            frac = max(0.0, min(1.0, -self.prev_val / dv)) if abs(dv) > 1e-30 else 0.0
            self.t_cross = self.prev_time + frac * (time - self.prev_time)
            # Clamp val to prevent immediate re-trigger
            val = max(val, 0.0)

        self.pprev_val = self.prev_val
        self.pprev_time = self.prev_time
        self.prev_time = time
        self.prev_val = val
        self.last_triggered = triggered
        return triggered

    def next_breakpoint(self) -> Optional[float]:
        """Predict next crossing time by linear extrapolation."""
        dt = self.prev_time - self.pprev_time
        if dt <= 1e-30:
            return None
        rate = (self.prev_val - self.pprev_val) / dt
        if self.prev_val < 0 and rate > 0:
            return self.prev_time + (-self.prev_val / rate)
        return None


@dataclass
class Source:
    """A voltage source stimulus."""
    node: str
    waveform: Callable[[float], float]  # time → voltage
    breakpoint_fn: Optional[Callable[[float], Optional[float]]] = field(
        init=False,
        repr=False,
        default=None,
    )

    def __post_init__(self):
        self.breakpoint_fn = getattr(self.waveform, '_next_breakpoint', None)

    def next_breakpoint(self, time: float) -> Optional[float]:
        if self.breakpoint_fn is None:
            return None
        return self.breakpoint_fn(time)


@dataclass
class SimResult:
    """Simulation results."""
    time: np.ndarray
    signals: Dict[str, np.ndarray]  # node_name → voltage array
    step_sizes: np.ndarray = None  # dt for each time point


class Simulator:
    """Event-driven transient simulator."""

    def __init__(self):
        self.sources: List[Source] = []
        self.node_voltages: Dict[str, float] = {}
        self.models: List = []  # compiled model instances
        self.recorded_signals: Dict[str, List[float]] = {}
        self.time_points: List[float] = []
        self._perf_stats: Dict[str, int] = {}

    def add_source(self, node: str, waveform: Callable[[float], float]):
        """Add a voltage source to the simulation."""
        self.sources.append(Source(node=node, waveform=waveform))

    def add_model(self, model):
        """Add a compiled Verilog-A model instance."""
        self.models.append(model)

    def record(self, *nodes: str):
        """Mark nodes for recording."""
        for n in nodes:
            self.recorded_signals[n] = []

    def run(self, tstop: float, tstep: float = None,
            max_step: float = None,
            refine_factor: int = 16,
            refine_steps: int = 8,
            reltol: float = 1e-3,
            vabstol: float = 1e-6,
            min_step: float = None,
            record_step: float = None,
            skip_source_error_control: bool = False,
            profile_sections: bool = False,
            indexed_snapshot_profile: bool = False,
            indexed_arrays: bool = False) -> SimResult:
        """Run transient simulation with adaptive step control near cross events."""
        if tstep is None:
            tstep = tstop / 10000
        if max_step is None:
            max_step = tstep
        if min_step is None:
            min_step = tstep / 4096.0

        time = 0.0
        self.time_points = []
        self._step_sizes = []
        refine_steps_left = 0  # countdown of refined steps after cross
        refine_dt = tstep  # current refined step size
        dynamic_step = tstep  # tolerance-driven adaptive step ceiling
        next_record_time = float(record_step) if record_step and record_step > 0 else None
        self._perf_stats = {
            "source_breakpoint_clamps": 0,
            "model_breakpoint_clamps": 0,
            "bound_step_clamps": 0,
            "min_step_clamps": 0,
            "cross_refine_triggers": 0,
            "cross_event_steps": 0,
            "dynamic_step_shrinks": 0,
            "dynamic_step_grows": 0,
            "output_step_clamps": 0,
            "err_ratio_skipped_outputs": 0,
            "err_ratio_skipped_sources": 0,
            "future_node_snapshots": 0,
            "future_node_lazy_descriptors": 0,
            "indexed_prev_snapshots": 0,
            "indexed_snapshot_dynamic_nodes": 0,
            "indexed_snapshot_mismatches": 0,
            "indexed_snapshot_values_checked": 0,
            "indexed_array_dynamic_nodes": 0,
            "indexed_array_err_ratio_reads": 0,
            "indexed_array_mismatches": 0,
            "indexed_array_record_reads": 0,
            "indexed_array_snapshots": 0,
            "indexed_array_source_updates": 0,
            "indexed_array_syncs": 0,
            "indexed_array_values_checked": 0,
            "indexed_model_io_mapped_ports": 0,
            "indexed_model_io_models": 0,
            "indexed_model_io_outputs": 0,
            "indexed_model_io_refreshes": 0,
            "indexed_output_write_through_nodes": 0,
            "indexed_output_write_throughs": 0,
            "indexed_post_model_sync_repairs": 0,
            "indexed_voltage_probe_event_skips": 0,
            "indexed_voltage_probe_max_abs_diff": 0.0,
            "indexed_voltage_probe_mismatches": 0,
            "indexed_voltage_probe_missing_nodes": 0,
            "indexed_voltage_probes": 0,
            "steps_total": 0,
        }
        self._profile_times: Dict[str, float] = {}
        self._indexed_snapshot_stats: Dict[str, object] = {}
        self._indexed_array_stats: Dict[str, object] = {}
        self._indexed_model_io_stats: Dict[str, object] = {}
        self._indexed_voltage_probe_stats: Dict[str, object] = {}
        profile_clock = (
            _wall_time.perf_counter
            if (profile_sections or indexed_snapshot_profile or indexed_arrays)
            else None
        )

        def _add_profile_time(name: str, start: float):
            if profile_clock is not None:
                elapsed = profile_clock() - start
                self._profile_times[name] = self._profile_times.get(name, 0.0) + elapsed

        def _set_model_indexed_output_writer(writer):
            for model in self.models:
                setter = getattr(model, "_set_indexed_output_writer", None)
                if setter is not None:
                    setter(writer)

        def _set_model_indexed_voltage_probe(probe):
            for model in self.models:
                setter = getattr(model, "_set_indexed_voltage_probe", None)
                if setter is not None:
                    setter(probe)

        _set_model_indexed_output_writer(None)
        _set_model_indexed_voltage_probe(None)

        source_nodes = {src.node for src in self.sources}
        source_future_waveforms = {src.node: src.waveform for src in self.sources}
        source_breakpoint_sources = [src for src in self.sources if src.breakpoint_fn is not None]
        transition_breakpoint_min_ramp = 0.15 * max_step

        def _set_transition_breakpoint_threshold(model):
            if hasattr(model, "_transition_breakpoint_min_ramp"):
                model._transition_breakpoint_min_ramp = transition_breakpoint_min_ramp
            for child in getattr(model, "_child_models", []) or []:
                _set_transition_breakpoint_threshold(child)

        for model in self.models:
            _set_transition_breakpoint_threshold(model)

        for model in self.models:
            if hasattr(model, "_set_nominal_step"):
                model._set_nominal_step(tstep)
        breakpoint_models = [
            model for model in self.models
            if bool(getattr(model, "_has_dynamic_breakpoints_tree", lambda: True)())
        ]
        bound_step_models = [
            model for model in self.models
            if bool(getattr(model, "_uses_bound_step_tree", lambda: True)())
        ]
        model_needs_future_node_voltages = tuple(
            bool(getattr(model, "_needs_future_node_voltages_tree", lambda: False)())
            for model in self.models
        )
        models_need_future_node_voltages = any(model_needs_future_node_voltages)

        def _set_initial_condition_mode(model, enabled: bool):
            if hasattr(model, "_set_initial_condition_mode"):
                model._set_initial_condition_mode(enabled)

        # Initialize node voltages
        for src in self.sources:
            self.node_voltages[src.node] = src.waveform(0.0)
        for model in self.models:
            model._prepare_step(self.node_voltages, self.node_voltages, 0.0, 0.0)

        # Spectre solves initial_step and coincident t=0 events before the
        # first saved transient point.  During that operating-point-like pass,
        # transition() contributes its settled target instead of a ramp from an
        # implicit zero.
        for model in self.models:
            _set_initial_condition_mode(model, True)

        # Fire initial_step events
        for model in self.models:
            model.initial_step(self.node_voltages, 0.0)

        # Evaluate models at t=0 so output nodes are assigned before recording.
        # Without this, output nodes default to 0, producing spurious values in
        # post-processing (e.g. noise = vout_o - vin_i = 0 - 1 = -1 V).
        for model in self.models:
            model.evaluate(self.node_voltages, 0.0)
            model._expire_absolute_timers(0.0)
            if model.post_update_events(self.node_voltages, 0.0):
                model.refresh_outputs(self.node_voltages, 0.0)

        for model in self.models:
            _set_initial_condition_mode(model, False)

        model_output_nodes: set[str] = set()
        model_output_versions: Optional[tuple[int, ...]] = None
        err_ratio_nodes: tuple[str, ...] = ()
        err_ratio_cache_key = None
        err_ratio_skipped_outputs_per_step = 0
        err_ratio_skipped_sources_per_step = 0

        def _refresh_model_output_nodes() -> set[str]:
            nonlocal model_output_nodes, model_output_versions
            versions = tuple(
                int(getattr(model, "_output_nodes_version", len(model.output_nodes)))
                for model in self.models
            )
            if versions != model_output_versions:
                model_output_nodes = set()
                for model in self.models:
                    model_output_nodes.update(model.output_nodes.keys())
                model_output_versions = versions
            return model_output_nodes

        _refresh_model_output_nodes()
        indexed_snapshotter = None
        if indexed_snapshot_profile:
            indexed_snapshotter = IndexedVoltageSnapshotter.from_names(
                sorted(set(self.node_voltages) | set(self.recorded_signals) | source_nodes | model_output_nodes)
            )
            self._indexed_snapshot_stats = {
                "node_count": indexed_snapshotter.node_count,
                "snapshots": 0,
                "checked_values": 0,
                "max_abs_diff": 0.0,
                "max_abs_diff_node": "",
                "dynamic_nodes": 0,
            }

        indexed_array = None
        indexed_model_io_versions = None
        indexed_model_io_plan = None
        indexed_output_nodes_seen: set[str] = set()
        indexed_voltage_probe_max_node = ""
        if indexed_arrays:
            indexed_array = IndexedVoltageArray.from_names(
                sorted(set(self.node_voltages) | set(self.recorded_signals) | source_nodes | model_output_nodes)
            )
            indexed_array.update_from_mapping(self.node_voltages)
            self._indexed_array_stats = {
                "node_count": indexed_array.node_count,
                "snapshots": 0,
                "syncs": 0,
                "source_updates": 0,
                "record_reads": 0,
                "err_ratio_reads": 0,
                "checked_values": 0,
                "max_abs_diff": 0.0,
                "max_abs_diff_node": "",
                "dynamic_nodes": 0,
                "output_write_through_nodes": 0,
                "output_write_throughs": 0,
                "post_model_sync_repairs": 0,
            }

        def _model_tree_output_versions() -> tuple[int, ...]:
            versions: List[int] = []

            def _visit(model):
                versions.append(int(getattr(model, "_output_nodes_version", len(model.output_nodes))))
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)
            return tuple(versions)

        def _refresh_indexed_model_io_plan(force: bool = False):
            nonlocal indexed_model_io_versions, indexed_model_io_plan
            if indexed_array is None:
                return
            versions = _model_tree_output_versions()
            if not force and versions == indexed_model_io_versions:
                return
            indexed_model_io_plan = build_indexed_model_io_plan(
                self,
                extra_nodes=indexed_array.node_index.names,
            )
            indexed_array.ensure_nodes(indexed_model_io_plan.node_index.names)
            indexed_model_io_versions = versions
            self._perf_stats["indexed_model_io_refreshes"] += 1
            self._perf_stats["indexed_model_io_models"] = indexed_model_io_plan.model_count
            self._perf_stats["indexed_model_io_mapped_ports"] = (
                indexed_model_io_plan.mapped_port_count
            )
            self._perf_stats["indexed_model_io_outputs"] = indexed_model_io_plan.output_count
            self._indexed_model_io_stats = {
                "node_count": indexed_model_io_plan.node_count,
                "model_count": indexed_model_io_plan.model_count,
                "mapped_port_count": indexed_model_io_plan.mapped_port_count,
                "output_count": indexed_model_io_plan.output_count,
                "refreshes": self._perf_stats["indexed_model_io_refreshes"],
            }

        def _refresh_indexed_array_stats():
            if indexed_array is None:
                return
            self._indexed_array_stats["node_count"] = indexed_array.node_count
            self._indexed_array_stats["snapshots"] = self._perf_stats["indexed_array_snapshots"]
            self._indexed_array_stats["syncs"] = self._perf_stats["indexed_array_syncs"]
            self._indexed_array_stats["source_updates"] = self._perf_stats[
                "indexed_array_source_updates"
            ]
            self._indexed_array_stats["record_reads"] = self._perf_stats["indexed_array_record_reads"]
            self._indexed_array_stats["err_ratio_reads"] = self._perf_stats[
                "indexed_array_err_ratio_reads"
            ]
            self._indexed_array_stats["checked_values"] = self._perf_stats[
                "indexed_array_values_checked"
            ]
            self._indexed_array_stats["dynamic_nodes"] = indexed_array.dynamic_interns
            self._indexed_array_stats["output_write_throughs"] = self._perf_stats[
                "indexed_output_write_throughs"
            ]
            self._indexed_array_stats["output_write_through_nodes"] = self._perf_stats[
                "indexed_output_write_through_nodes"
            ]
            self._indexed_array_stats["post_model_sync_repairs"] = self._perf_stats[
                "indexed_post_model_sync_repairs"
            ]
            self._perf_stats["indexed_array_dynamic_nodes"] = indexed_array.dynamic_interns
            self._indexed_voltage_probe_stats = {
                "probes": self._perf_stats["indexed_voltage_probes"],
                "event_skips": self._perf_stats["indexed_voltage_probe_event_skips"],
                "missing_nodes": self._perf_stats["indexed_voltage_probe_missing_nodes"],
                "mismatches": self._perf_stats["indexed_voltage_probe_mismatches"],
                "max_abs_diff": self._perf_stats["indexed_voltage_probe_max_abs_diff"],
                "max_abs_diff_node": indexed_voltage_probe_max_node,
            }

        def _record_indexed_array_diff(max_diff: float, max_node: str, checked: int):
            if indexed_array is None:
                return
            self._perf_stats["indexed_array_values_checked"] += checked
            if max_diff > float(self._indexed_array_stats["max_abs_diff"]):
                self._indexed_array_stats["max_abs_diff"] = max_diff
                self._indexed_array_stats["max_abs_diff_node"] = max_node
            if max_diff != 0.0:
                self._perf_stats["indexed_array_mismatches"] += 1
            _refresh_indexed_array_stats()

        if indexed_array is not None:
            _refresh_indexed_model_io_plan(force=True)

            def _indexed_output_write_through(node: str, value: float):
                indexed_array.set(node, value)
                indexed_output_nodes_seen.add(node)
                self._perf_stats["indexed_output_write_throughs"] += 1
                self._perf_stats["indexed_output_write_through_nodes"] = len(
                    indexed_output_nodes_seen
                )

            def _indexed_voltage_probe(
                local_node: str,
                external_node: str,
                dict_value: float,
                event_context: bool,
            ):
                nonlocal indexed_voltage_probe_max_node
                if event_context:
                    self._perf_stats["indexed_voltage_probe_event_skips"] += 1
                    _refresh_indexed_array_stats()
                    return
                self._perf_stats["indexed_voltage_probes"] += 1
                if not indexed_array.node_index.has(external_node):
                    self._perf_stats["indexed_voltage_probe_missing_nodes"] += 1
                    _refresh_indexed_array_stats()
                    return
                array_value = indexed_array.get(external_node, 0.0)
                diff = abs(float(array_value) - float(dict_value))
                if diff > self._perf_stats["indexed_voltage_probe_max_abs_diff"]:
                    self._perf_stats["indexed_voltage_probe_max_abs_diff"] = diff
                    indexed_voltage_probe_max_node = external_node or local_node
                if diff != 0.0:
                    self._perf_stats["indexed_voltage_probe_mismatches"] += 1
                _refresh_indexed_array_stats()

            _set_model_indexed_output_writer(_indexed_output_write_through)
            _set_model_indexed_voltage_probe(_indexed_voltage_probe)
            max_diff, max_node, checked = indexed_array.max_abs_diff_mapping(self.node_voltages)
            _record_indexed_array_diff(max_diff, max_node, checked)

        # Record initial state
        initial_record_reads = self._record_point(0.0, indexed_array)
        if indexed_array is not None:
            self._perf_stats["indexed_array_record_reads"] += initial_record_reads
            _refresh_indexed_array_stats()
        self._step_sizes.append(0.0)

        # Main simulation loop
        while time < tstop:
            force_record_point = False
            if refine_steps_left > 0:
                dt = min(refine_dt, dynamic_step, max_step, tstop - time)
                refine_steps_left -= 1
            else:
                dt = min(dynamic_step, tstep, max_step, tstop - time)

            # Check for breakpoints from sources (PWL knees, pulse edges)
            _section_start = profile_clock() if profile_clock is not None else 0.0
            for src in source_breakpoint_sources:
                bp = src.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    force_record_point = True
                    self._perf_stats["source_breakpoint_clamps"] += 1
                    if dt < 1e-18:
                        dt = 1e-18
            if profile_clock is not None:
                _add_profile_time("source_breakpoint_scan_s", _section_start)

            # Check for breakpoints from transition operators
            _section_start = profile_clock() if profile_clock is not None else 0.0
            for model in breakpoint_models:
                bp = model.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    force_record_point = True
                    self._perf_stats["model_breakpoint_clamps"] += 1
                    if dt < 1e-18:
                        dt = 1e-18
            if profile_clock is not None:
                _add_profile_time("model_breakpoint_scan_s", _section_start)

            # Respect $bound_step from models
            _section_start = profile_clock() if profile_clock is not None else 0.0
            for model in bound_step_models:
                bs = model._bound_step
                if bs > 0 and dt > bs:
                    dt = bs
                    force_record_point = True
                    self._perf_stats["bound_step_clamps"] += 1
            if profile_clock is not None:
                _add_profile_time("bound_step_scan_s", _section_start)

            if next_record_time is not None:
                if next_record_time > time and next_record_time < time + dt:
                    dt = next_record_time - time
                    force_record_point = True
                    self._perf_stats["output_step_clamps"] += 1

            if dt < min_step:
                dt = min_step
                self._perf_stats["min_step_clamps"] += 1

            prev_time = time
            _section_start = profile_clock() if profile_clock is not None else 0.0
            prev_nv = dict(self.node_voltages)
            if profile_clock is not None:
                _add_profile_time("dict_prev_snapshot_s", _section_start)
            prev_indexed_values = None
            if indexed_array is not None:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                prev_indexed_values = indexed_array.snapshot()
                max_diff, max_node, checked = indexed_array.max_abs_diff_mapping(prev_nv)
                self._perf_stats["indexed_array_snapshots"] += 1
                _record_indexed_array_diff(max_diff, max_node, checked)
                if profile_clock is not None:
                    _add_profile_time("indexed_array_prev_snapshot_s", _section_start)
            if indexed_snapshotter is not None:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                prev_indexed_snapshot = indexed_snapshotter.snapshot_from_mapping(prev_nv)
                max_diff, max_node, checked = indexed_snapshotter.max_abs_diff(
                    prev_indexed_snapshot,
                    prev_nv,
                )
                self._perf_stats["indexed_prev_snapshots"] += 1
                self._perf_stats["indexed_snapshot_values_checked"] += checked
                if max_diff > float(self._indexed_snapshot_stats["max_abs_diff"]):
                    self._indexed_snapshot_stats["max_abs_diff"] = max_diff
                    self._indexed_snapshot_stats["max_abs_diff_node"] = max_node
                if max_diff != 0.0:
                    self._perf_stats["indexed_snapshot_mismatches"] += 1
                self._indexed_snapshot_stats["snapshots"] = self._perf_stats["indexed_prev_snapshots"]
                self._indexed_snapshot_stats["checked_values"] = self._perf_stats[
                    "indexed_snapshot_values_checked"
                ]
                self._indexed_snapshot_stats["node_count"] = indexed_snapshotter.node_count
                self._indexed_snapshot_stats["dynamic_nodes"] = indexed_snapshotter.dynamic_interns
                self._perf_stats["indexed_snapshot_dynamic_nodes"] = indexed_snapshotter.dynamic_interns
                if profile_clock is not None:
                    _add_profile_time("indexed_prev_snapshot_s", _section_start)
            time += dt

            # Update source voltages
            _section_start = profile_clock() if profile_clock is not None else 0.0
            for src in self.sources:
                value = src.waveform(time)
                self.node_voltages[src.node] = value
                if indexed_array is not None:
                    indexed_array.set(src.node, value)
                    self._perf_stats["indexed_array_source_updates"] += 1
            if indexed_array is not None:
                _refresh_indexed_array_stats()
            if profile_clock is not None:
                _add_profile_time("source_update_s", _section_start)

            _section_start = profile_clock() if profile_clock is not None else 0.0
            future_nv = None
            if models_need_future_node_voltages:
                future_time = time + max(1e-18, min(dt * 1e-6, 1e-15))
                future_nv = _LazyFutureNodeVoltages(source_future_waveforms, future_time)
                self._perf_stats["future_node_lazy_descriptors"] += 1
            if profile_clock is not None:
                _add_profile_time("dict_future_snapshot_s", _section_start)

            # Evaluate all models
            cross_fired = False
            for model, model_needs_future in zip(self.models, model_needs_future_node_voltages):
                _section_start = profile_clock() if profile_clock is not None else 0.0
                model_future_nv = (
                    future_nv
                    if future_nv is not None
                    and model_needs_future
                    else None
                )
                model._prepare_step(prev_nv, self.node_voltages, prev_time, time, model_future_nv)
                if profile_clock is not None:
                    _add_profile_time("model_prepare_step_s", _section_start)
                _section_start = profile_clock() if profile_clock is not None else 0.0
                model.evaluate(self.node_voltages, time)
                if profile_clock is not None:
                    _add_profile_time("model_evaluate_s", _section_start)
                _section_start = profile_clock() if profile_clock is not None else 0.0
                model._expire_absolute_timers(time)
                if model.post_update_events(self.node_voltages, time):
                    model.refresh_outputs(self.node_voltages, time)
                if profile_clock is not None:
                    _add_profile_time("model_post_update_s", _section_start)
                if getattr(model, "_step_event_fired", False):
                    cross_fired = True

            if indexed_array is not None:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                _refresh_indexed_model_io_plan()
                self._perf_stats["indexed_array_syncs"] += 1
                max_diff, max_node, checked = indexed_array.max_abs_diff_mapping(self.node_voltages)
                _record_indexed_array_diff(max_diff, max_node, checked)
                if max_diff != 0.0:
                    indexed_array.update_from_mapping(self.node_voltages)
                    self._perf_stats["indexed_post_model_sync_repairs"] += 1
                    _refresh_indexed_array_stats()
                if profile_clock is not None:
                    _add_profile_time("indexed_array_sync_s", _section_start)

            # Check if any cross/above event fired this step
            _section_start = profile_clock() if profile_clock is not None else 0.0
            if profile_clock is not None:
                _add_profile_time("cross_above_scan_s", _section_start)

            if cross_fired and refine_steps_left == 0 and dt > tstep / refine_factor:
                refine_dt = dt / refine_factor
                refine_steps_left = refine_steps
                self._perf_stats["cross_refine_triggers"] += 1
            if cross_fired:
                force_record_point = True
                self._perf_stats["cross_event_steps"] += 1

            # Tolerance-guided dynamic step adaptation (voltage-domain heuristic).
            # This is not full LTE/Newton control, but gives user-visible precision control.
            err_ratio = 0.0
            _section_start = profile_clock() if profile_clock is not None else 0.0
            model_output_nodes = _refresh_model_output_nodes()
            if profile_clock is not None:
                _add_profile_time("model_output_set_s", _section_start)
            _section_start = profile_clock() if profile_clock is not None else 0.0
            cache_key = (len(self.node_voltages), model_output_versions, bool(skip_source_error_control))
            if cache_key != err_ratio_cache_key:
                err_ratio_nodes_list = []
                skipped_outputs = 0
                skipped_sources = 0
                for node in self.node_voltages:
                    if node in model_output_nodes:
                        skipped_outputs += 1
                        continue
                    if skip_source_error_control and node in source_nodes:
                        skipped_sources += 1
                        continue
                    err_ratio_nodes_list.append(node)
                err_ratio_nodes = tuple(err_ratio_nodes_list)
                err_ratio_skipped_outputs_per_step = skipped_outputs
                err_ratio_skipped_sources_per_step = skipped_sources
                err_ratio_cache_key = cache_key
            if err_ratio_skipped_outputs_per_step:
                self._perf_stats["err_ratio_skipped_outputs"] += err_ratio_skipped_outputs_per_step
            if err_ratio_skipped_sources_per_step:
                self._perf_stats["err_ratio_skipped_sources"] += err_ratio_skipped_sources_per_step
            for node in err_ratio_nodes:
                if indexed_array is not None:
                    vnew = indexed_array.get(node, self.node_voltages.get(node, 0.0))
                    vold = indexed_array.get_from_snapshot(prev_indexed_values, node, vnew)
                    self._perf_stats["indexed_array_err_ratio_reads"] += 1
                else:
                    vnew = self.node_voltages[node]
                    vold = prev_nv.get(node, vnew)
                dv = abs(vnew - vold)
                vref = max(abs(vnew), abs(vold))
                tol = reltol * vref + vabstol
                if tol > 0.0:
                    er = dv / tol
                    if er > err_ratio:
                        err_ratio = er
            if indexed_array is not None:
                _refresh_indexed_array_stats()
            if profile_clock is not None:
                _add_profile_time("err_ratio_node_scan_s", _section_start)
            if err_ratio > 1.0:
                scale = min(4.0, max(1.2, math.sqrt(err_ratio)))
                dynamic_step = max(min_step, dynamic_step / scale)
                self._perf_stats["dynamic_step_shrinks"] += 1
            elif err_ratio < 0.2:
                dynamic_step = min(tstep, dynamic_step * 1.15)
                self._perf_stats["dynamic_step_grows"] += 1

            if next_record_time is None:
                should_record = True
            else:
                should_record = (
                    force_record_point
                    or time >= next_record_time - 1e-18
                    or time >= tstop - 1e-18
                )

            if should_record:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                record_reads = self._record_point(time, indexed_array)
                if indexed_array is not None:
                    self._perf_stats["indexed_array_record_reads"] += record_reads
                    _refresh_indexed_array_stats()
                self._step_sizes.append(dt)
                if next_record_time is not None:
                    while next_record_time <= time + 1e-18:
                        next_record_time += record_step
                if profile_clock is not None:
                    _add_profile_time("record_point_s", _section_start)
            self._perf_stats["steps_total"] += 1

        # Fire final_step events
        _section_start = profile_clock() if profile_clock is not None else 0.0
        for model in self.models:
            model.final_step(self.node_voltages, time)
        if profile_clock is not None:
            _add_profile_time("final_step_s", _section_start)

        # Close any open file handles
        _section_start = profile_clock() if profile_clock is not None else 0.0
        for model in self.models:
            model._cleanup_files()
        if profile_clock is not None:
            _add_profile_time("cleanup_files_s", _section_start)

        # Convert to arrays
        _section_start = profile_clock() if profile_clock is not None else 0.0
        time_arr = np.array(self.time_points)
        signals = {}
        for name, data in self.recorded_signals.items():
            signals[name] = np.array(data)
        if profile_clock is not None:
            _add_profile_time("result_array_conversion_s", _section_start)

        _set_model_indexed_output_writer(None)
        _set_model_indexed_voltage_probe(None)

        return SimResult(time=time_arr, signals=signals,
                         step_sizes=np.array(self._step_sizes))

    def _record_point(self, time: float, indexed_voltages: Optional[IndexedVoltageArray] = None) -> int:
        self.time_points.append(time)
        reads = 0
        for name in self.recorded_signals:
            if indexed_voltages is not None:
                val = indexed_voltages.get(name, 0.0)
                reads += 1
            else:
                val = self.node_voltages.get(name, 0.0)
            self.recorded_signals[name].append(val)
        return reads


# ─── Waveform helpers ───

def pulse(v_lo, v_hi, period, duty=0.5, rise=1e-12, fall=1e-12, delay=0.0,
          width=None):
    """Create a pulse waveform function.

    When ``width`` is provided, use Spectre vsource pulse semantics: ``width``
    is the high plateau after the rise ramp, so the falling edge starts at
    ``delay + rise + width``.  The older ``duty`` path is kept for direct helper
    callers that model the falling edge as ``delay + period * duty``.  When
    ``period`` is non-positive, match Spectre's useful one-shot behavior: the
    pulse occurs once, and if no width is provided it stays at ``v_hi`` after
    the rise.
    """
    period = float(period)
    rise = float(rise)
    fall = float(fall)
    delay = float(delay)
    one_shot = period <= 0.0
    if width is not None:
        fall_start = rise + float(width)
    elif one_shot:
        fall_start = float("inf")
    else:
        fall_start = period * duty
    fall_end = fall_start + fall
    # Include edge interior breakpoints so source-driven @cross events are
    # scheduled in chronological order when multiple sources switch together.
    knees = {0.0, rise}
    if np.isfinite(fall_start):
        knees.add(fall_start)
    if np.isfinite(fall_end):
        knees.add(fall_end)
    if rise > 0:
        knees.add(0.5 * rise)
    if fall > 0 and np.isfinite(fall_start):
        knees.add(fall_start + 0.5 * fall)
    knees = sorted(knees)

    def wfn(t):
        t_eff = t - delay
        if t_eff < 0:
            return v_lo
        t_mod = t_eff if one_shot else t_eff % period
        if t_mod < rise:
            frac = t_mod / rise if rise > 0 else 1.0
            return v_lo + frac * (v_hi - v_lo)
        elif t_mod < fall_start:
            return v_hi
        elif t_mod < fall_end:
            frac = (t_mod - fall_start) / fall if fall > 0 else 1.0
            return v_hi - frac * (v_hi - v_lo)
        else:
            return v_lo

    def _bpfn(t):
        if t < delay:
            return delay
        if one_shot:
            for k in knees:
                candidate = delay + k
                if candidate > t + 1e-18:
                    return candidate
            return None
        t_eff = t - delay
        n = int(t_eff / period)
        for _ in range(2):
            for k in knees:
                candidate = delay + n * period + k
                if candidate > t + 1e-18:
                    return candidate
            n += 1
        return None

    wfn._next_breakpoint = _bpfn
    return wfn


def dc(voltage):
    """Create a DC voltage waveform."""
    return lambda t: voltage


def sine(offset, amplitude, freq, phase=0.0):
    """Create a sine waveform."""
    return lambda t: offset + amplitude * math.sin(2 * math.pi * freq * t + phase)


def pwl(times, values):
    """Create a piecewise-linear waveform."""
    if not times or not values:
        raise ValueError("PWL waveform requires at least one time/value pair")
    if len(times) != len(values):
        raise ValueError("PWL waveform times and values must have the same length")
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            raise ValueError(
                "PWL waveform times must be strictly increasing "
                f"(t[{i - 1}]={times[i - 1]!r}, t[{i}]={times[i]!r})"
            )

    sorted_t = sorted(set(times))

    def wfn(t):
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        for i in range(len(times) - 1):
            if times[i] <= t < times[i + 1]:
                frac = (t - times[i]) / (times[i + 1] - times[i])
                return values[i] + frac * (values[i + 1] - values[i])
        return values[-1]

    def _bpfn(t):
        for kt in sorted_t:
            if kt > t + 1e-18:
                return kt
        return None

    wfn._next_breakpoint = _bpfn
    return wfn


def ramp(v_start, v_end, t_start, t_end):
    """Create a linear ramp waveform."""
    def wfn(t):
        if t <= t_start:
            return v_start
        if t >= t_end:
            return v_end
        frac = (t - t_start) / (t_end - t_start)
        return v_start + frac * (v_end - v_start)
    return wfn
