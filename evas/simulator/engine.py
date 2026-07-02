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
import os
import random
import time as _wall_time
from array import array
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from evas.simulator.analog_block_runtime import (
    try_build_event_then_transition_shadow_runtime,
)
from evas.simulator.evaluate_ir import (
    SOURCE_NODE,
    SOURCE_STATE,
    TARGET_NODE,
    TARGET_STATE,
    linear_op_uses_params,
    normalize_linear_ops,
    normalize_transition_target_ops,
)
from evas.simulator.indexed import (
    IndexedVoltageArray,
    IndexedVoltageSnapshotter,
    build_indexed_model_io_plan,
)
from evas.simulator.rust_backend import (
    BODY_EXPR_CONST,
    BODY_EXPR_READ_NODE,
    BODY_EXPR_READ_STATE,
    BODY_EXPR_SUB,
    LinearCondition,
    LinearOp,
    LinearTerm,
    RustBackendError,
    TransitionTargetOp,
    load_optional_rust_backend,
)
from evas.simulator.rust_program import build_source_record_rust_program


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
        self.current_sources: List[Tuple[str, str, Source]] = []
        self.node_voltages: Dict[str, float] = {}
        self.models: List = []  # compiled model instances
        self.recorded_signals: Dict[str, List[float]] = {}
        self.time_points: List[float] = []
        self._perf_stats: Dict[str, int] = {}

    def add_source(self, node: str, waveform: Callable[[float], float]):
        """Add a voltage source to the simulation."""
        self.sources.append(Source(node=node, waveform=waveform))

    def add_current_source(self, node_pos: str, node_neg: str, waveform: Callable[[float], float]):
        """Add an independent current source for branch-current probes."""
        self.current_sources.append((node_pos, node_neg, Source(node=node_pos, waveform=waveform)))

    def add_model(self, model):
        """Add a compiled Verilog-A model instance."""
        self.models.append(model)

    def record(self, *nodes: str):
        """Mark nodes for recording."""
        for n in nodes:
            self.recorded_signals[n] = []

    @staticmethod
    def _waveform_metadata(waveform):
        meta = getattr(waveform, "_evas_waveform", None)
        return meta if isinstance(meta, dict) else None

    @staticmethod
    def _dedupe_times(times: List[float], tstop: float) -> array:
        eps = 1.0e-18
        cleaned = []
        for value in sorted(times):
            if value < -eps or value > tstop + eps:
                continue
            value = 0.0 if abs(value) <= eps else value
            value = tstop if abs(value - tstop) <= eps else value
            if cleaned and abs(value - cleaned[-1]) <= eps:
                continue
            cleaned.append(value)
        if not cleaned or cleaned[0] != 0.0:
            cleaned.insert(0, 0.0)
        if cleaned[-1] < tstop - eps:
            cleaned.append(tstop)
        return array("d", cleaned)

    @staticmethod
    def _add_pulse_schedule_times(
        times: List[float],
        meta: Dict[str, object],
        tstop: float,
        vth: Optional[float] = None,
        transition_offsets: Tuple[float, ...] = (),
    ) -> int:
        period = float(meta.get("period", 0.0) or 0.0)
        rise = float(meta.get("rise", 0.0) or 0.0)
        fall = float(meta.get("fall", 0.0) or 0.0)
        delay = float(meta.get("delay", 0.0) or 0.0)
        duty = float(meta.get("duty", 0.5) or 0.5)
        has_width = bool(meta.get("has_width", False))
        width = float(meta.get("width", 0.0) or 0.0)
        fall_start = rise + width if has_width else (
            float("inf") if period <= 0.0 else period * duty
        )
        knees = [0.0, rise]
        if math.isfinite(fall_start):
            knees.append(fall_start)
            knees.append(fall_start + fall)
            if fall > 0.0:
                knees.append(fall_start + 0.5 * fall)
        if rise > 0.0:
            knees.append(0.5 * rise)

        cycles = 1 if period <= 0.0 else int(math.floor(max(0.0, (tstop - delay)) / period)) + 2
        cross_count = 0
        for n in range(max(1, cycles)):
            base = delay + (period * n if period > 0.0 else 0.0)
            for offset in knees:
                if math.isfinite(offset):
                    times.append(base + offset)
            if vth is not None:
                v_lo = float(meta.get("v_lo", 0.0) or 0.0)
                v_hi = float(meta.get("v_hi", 0.0) or 0.0)
                if rise <= 0.0:
                    cross_offset = 0.0
                elif v_hi != v_lo and min(v_lo, v_hi) <= vth <= max(v_lo, v_hi):
                    cross_offset = rise * ((vth - v_lo) / (v_hi - v_lo))
                else:
                    cross_offset = None
                if cross_offset is not None and 0.0 <= cross_offset <= rise:
                    cross_time = base + cross_offset
                    if -1.0e-18 <= cross_time <= tstop + 1.0e-18:
                        times.append(cross_time)
                        for extra in transition_offsets:
                            times.append(cross_time + extra)
                        cross_count += 1
            if period <= 0.0:
                break
        return cross_count

    @staticmethod
    def _pulse_threshold_cross_times(
        meta: Dict[str, object],
        tstop: float,
        vth: float,
        direction: int,
    ) -> List[float]:
        period = float(meta.get("period", 0.0) or 0.0)
        rise = float(meta.get("rise", 0.0) or 0.0)
        fall = float(meta.get("fall", 0.0) or 0.0)
        delay = float(meta.get("delay", 0.0) or 0.0)
        duty = float(meta.get("duty", 0.5) or 0.5)
        has_width = bool(meta.get("has_width", False))
        width = float(meta.get("width", 0.0) or 0.0)
        v_lo = float(meta.get("v_lo", 0.0) or 0.0)
        v_hi = float(meta.get("v_hi", 0.0) or 0.0)
        if v_hi == v_lo:
            return []
        fall_start = rise + width if has_width else (
            float("inf") if period <= 0.0 else period * duty
        )
        times: List[float] = []
        cycles = 1 if period <= 0.0 else int(math.floor(max(0.0, (tstop - delay)) / period)) + 2
        for n in range(max(1, cycles)):
            base = delay + (period * n if period > 0.0 else 0.0)
            if direction >= 0 and rise >= 0.0 and min(v_lo, v_hi) <= vth <= max(v_lo, v_hi):
                frac = 1.0 if rise <= 0.0 else (vth - v_lo) / (v_hi - v_lo)
                t_cross = base + max(0.0, min(1.0, frac)) * rise
                if -1.0e-18 <= t_cross <= tstop + 1.0e-18:
                    times.append(t_cross)
            if (
                direction <= 0
                and math.isfinite(fall_start)
                and fall >= 0.0
                and min(v_lo, v_hi) <= vth <= max(v_lo, v_hi)
            ):
                frac = 1.0 if fall <= 0.0 else (v_hi - vth) / (v_hi - v_lo)
                t_cross = base + fall_start + max(0.0, min(1.0, frac)) * fall
                if -1.0e-18 <= t_cross <= tstop + 1.0e-18:
                    times.append(t_cross)
            if period <= 0.0:
                break
        return times

    @staticmethod
    def _add_source_breakpoint_times(times: List[float], src: Source, tstop: float) -> None:
        current = -1.0e-18
        for _ in range(100000):
            bp = src.next_breakpoint(current)
            if bp is None or bp > tstop + 1.0e-18:
                return
            if bp >= -1.0e-18:
                times.append(bp)
            current = bp

    @staticmethod
    def _model_candidate(model, kind: str) -> Optional[tuple]:
        model_cls = getattr(model, "__class__", type(model))
        for candidate in getattr(model_cls, "_whole_segment_candidates", ()) or ():
            if candidate and candidate[0] == kind:
                return candidate
        return None

    @staticmethod
    def _external_node(model, port: str) -> str:
        node_map = getattr(model, "node_map", {}) or {}
        return node_map.get(port, port)

    def _record_trace_result(
        self,
        times: array,
        columns: Dict[str, object],
        *,
        enabled_kind: str,
        step_count: Optional[int] = None,
    ) -> SimResult:
        time_arr = np.frombuffer(times, dtype=np.float64).copy()
        signals = {}
        for name in self.recorded_signals:
            values = columns[name]
            if isinstance(values, np.ndarray) and values.dtype == np.float64:
                signals[name] = values if values.flags.c_contiguous else np.ascontiguousarray(values)
            else:
                signals[name] = np.array(values, dtype=np.float64)
        for name, values in signals.items():
            self.recorded_signals[name] = values.tolist()
            self.node_voltages[name] = float(values[-1]) if len(values) else 0.0
        self.time_points = time_arr.tolist()
        step_sizes = np.empty(len(time_arr), dtype=np.float64)
        if len(step_sizes):
            step_sizes[0] = 0.0
        if len(step_sizes) > 1:
            step_sizes[1:] = np.diff(time_arr)
        self._step_sizes = step_sizes.tolist()
        self._perf_stats["rust_full_model_fastpath_enabled"] = 1
        self._perf_stats["rust_full_model_whole_segment_points"] = len(time_arr)
        self._perf_stats[f"rust_full_model_{enabled_kind}_enabled"] = 1
        self._perf_stats["steps_total"] = (
            int(step_count) if step_count is not None else max(0, len(time_arr) - 1)
        )
        return SimResult(time=time_arr, signals=signals, step_sizes=step_sizes)

    def _try_rust_sim_program_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
        max_step: float,
        cross_acceptance_slack_factor: float = 0.0,
    ) -> Optional[SimResult]:
        fastpath_t0 = _wall_time.perf_counter()
        self._perf_stats["rust_sim_program_requested"] = 1
        if rust_backend is None:
            self._perf_stats["rust_sim_program_rejections"] += 1
            self._rust_sim_program_last_rejection = "rust_backend_unavailable"
            return None

        lower_t0 = _wall_time.perf_counter()
        report = build_source_record_rust_program(
            sources=self.sources,
            recorded_signals=self.recorded_signals.keys(),
            models=self.models,
        )
        self._perf_stats["rust_sim_program_lower_elapsed_s"] = (
            _wall_time.perf_counter() - lower_t0
        )
        if not report.supported or report.program is None:
            self._perf_stats["rust_sim_program_rejections"] += 1
            self._rust_sim_program_last_rejection = ",".join(report.reasons)
            return None

        program = report.program
        self._perf_stats["rust_sim_program_available"] = 1
        self._perf_stats["rust_sim_program_node_count"] = program.node_count
        self._perf_stats["rust_sim_program_state_count"] = len(program.states)
        self._perf_stats["rust_sim_program_source_count"] = len(program.sources)
        self._perf_stats["rust_sim_program_record_count"] = len(program.records)
        self._perf_stats["rust_sim_program_continuous_linear_ops"] = len(
            program.continuous_linear_ops
        )
        self._perf_stats["rust_sim_program_event_count"] = len(program.events)
        self._perf_stats["rust_sim_program_always_body_count"] = sum(
            1 for event in program.events if str(getattr(event, "kind", "")) == "always"
        )
        self._perf_stats["rust_sim_program_body_stmt_ops"] = len(program.body_stmt_ops)
        self._perf_stats["rust_sim_program_body_expr_ops"] = len(program.body_expr_ops)
        self._perf_stats["rust_sim_program_transition_count"] = len(
            program.transitions
        )
        abi_t0 = _wall_time.perf_counter()
        try:
            rust_program = rust_backend.make_source_record_program(program)
        except RustBackendError as exc:
            self._perf_stats["rust_sim_program_rejections"] += 1
            self._rust_sim_program_last_rejection = f"abi_program_build_failed:{exc}"
            return None
        self._perf_stats["rust_sim_program_abi_build_elapsed_s"] = (
            _wall_time.perf_counter() - abi_t0
        )

        grid_t0 = _wall_time.perf_counter()
        raw_times = list(
            self._whole_segment_uniform_times(
                tstop=tstop,
                record_step=record_step,
                tstep=min(tstep, max_step),
            )
        )
        for src in self.sources:
            if src.breakpoint_fn is not None:
                self._add_source_breakpoint_times(raw_times, src, tstop)
        capacity = max(16, len(self._dedupe_times(raw_times, tstop)) + 8)
        if program.events or program.transitions:
            base_step = max(min(tstep, max_step), 1.0e-30)
            base_steps = int(math.ceil(float(tstop) / base_step)) + 1
            capacity = max(
                capacity,
                128 + 8 * max(base_steps, len(raw_times), len(program.events) + 1),
            )
        self._perf_stats["rust_sim_program_time_grid_elapsed_s"] = (
            _wall_time.perf_counter() - grid_t0
        )

        last_runtime_error: RustBackendError | None = None
        runtime_succeeded = False
        runtime_t0 = _wall_time.perf_counter()
        attempts = 0
        for _attempt in range(6):
            attempts += 1
            try:
                (
                    times,
                    columns,
                    _steps,
                    node_values,
                    state_values,
                    stats,
                ) = rust_backend.run_source_record_program(
                    rust_program,
                    capacity=capacity,
                    tstop=tstop,
                    tstep=tstep,
                    max_step=max_step,
                    record_step=record_step,
                    cross_acceptance_slack_factor=cross_acceptance_slack_factor,
                )
                runtime_succeeded = True
                break
            except RustBackendError as exc:
                last_runtime_error = exc
                message = str(exc)
                if "code -841" not in message and "capacity" not in message:
                    break
                capacity *= 2
        self._perf_stats["rust_sim_program_runtime_elapsed_s"] = (
            _wall_time.perf_counter() - runtime_t0
        )
        self._perf_stats["rust_sim_program_runtime_attempts"] = attempts
        self._perf_stats["rust_sim_program_final_capacity"] = capacity
        if not runtime_succeeded:
            self._perf_stats["rust_sim_program_rejections"] += 1
            self._rust_sim_program_last_rejection = f"runtime_failed:{last_runtime_error}"
            return None

        self._perf_stats["rust_sim_program_enabled"] = 1
        self._perf_stats["rust_sim_program_source_record_enabled"] = 1
        self._perf_stats["rust_sim_program_event_transition_enabled"] = int(
            bool(program.events or program.transitions)
        )
        self._perf_stats["rust_sim_program_points"] = len(times)
        self._perf_stats["rust_sim_program_source_breakpoints"] = int(
            stats.get("source_breakpoints", 0)
        )
        self._perf_stats["rust_sim_program_event_fires"] = int(
            stats.get("event_fires", 0)
        )
        self._perf_stats["rust_sim_program_transition_breakpoints"] = int(
            stats.get("transition_breakpoints", 0)
        )
        self._perf_stats["rust_sim_program_side_effects"] = int(
            stats.get("side_effects", 0)
        )
        record_t0 = _wall_time.perf_counter()
        result = self._record_trace_result(
            times,
            columns,
            enabled_kind=(
                "sim_program_event_transition_record"
                if program.events or program.transitions
                else "sim_program_source_record"
            ),
            step_count=max(0, len(times) - 1),
        )
        self._perf_stats["rust_sim_program_record_replay_elapsed_s"] = (
            _wall_time.perf_counter() - record_t0
        )
        sync_t0 = _wall_time.perf_counter()
        for name, value in zip(program.node_names, node_values):
            self.node_voltages[name] = float(value)
        for state, value in zip(program.states, state_values):
            try:
                model_index_text, state_name = state.name.split(":", 1)
                model = self.models[int(model_index_text)]
            except (ValueError, IndexError):
                continue
            state_value = float(value)
            if state.is_integer and hasattr(model, "_to_integer"):
                state_value = model._to_integer(state_value)
            array_ref_fn = getattr(model, "_state_array_slot_ref", None)
            array_ref = array_ref_fn(state_name) if array_ref_fn is not None else None
            if array_ref is not None:
                array_name, idx = array_ref
                if array_name not in model.arrays:
                    model.arrays[array_name] = {}
                model.arrays[array_name][int(idx)] = state_value
            else:
                model.state[state_name] = state_value
        self._perf_stats["rust_sim_program_state_sync_elapsed_s"] = (
            _wall_time.perf_counter() - sync_t0
        )
        self._perf_stats["rust_sim_program_fastpath_total_elapsed_s"] = (
            _wall_time.perf_counter() - fastpath_t0
        )
        return result

    def _whole_segment_uniform_times(
        self,
        *,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> List[float]:
        sample_step = float(record_step or tstep)
        if sample_step <= 0.0:
            return [0.0, float(tstop)]
        count = int(math.floor(tstop / sample_step + 1.0e-9)) + 1
        raw = [min(tstop, idx * sample_step) for idx in range(max(1, count))]
        if not raw or raw[-1] < tstop - 1.0e-18:
            raw.append(tstop)
        return raw

    def _whole_segment_timer_adaptive_times(
        self,
        *,
        tstop: float,
        tstep: float,
        max_step: float,
        min_step: float,
        timer_starts: Tuple[float, ...],
        timer_periods: Tuple[float, ...],
        source_breakpoint_sources: Tuple[Source, ...] = (),
    ) -> List[float]:
        """Mirror the default trace scheduler for restricted timer-linear models."""
        eps = 1.0e-18
        time = 0.0
        dynamic_step = float(tstep)
        times = [0.0]
        next_timer_times: List[float] = []
        for start, period in zip(timer_starts, timer_periods):
            if period <= 0.0 or not math.isfinite(period):
                continue
            fire_time = max(0.0, float(start))
            while fire_time <= time + eps:
                fire_time += float(period)
            next_timer_times.append(fire_time)

        while time < tstop - eps:
            dt = min(dynamic_step, tstep, max_step, tstop - time)
            next_kind = ""
            next_time: Optional[float] = None

            for fire_time in next_timer_times:
                if fire_time > time + eps and fire_time < time + dt:
                    if next_time is None or fire_time < next_time:
                        next_time = fire_time
                        next_kind = "timer"

            for src in source_breakpoint_sources:
                bp = src.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    if next_time is None or bp < next_time:
                        next_time = float(bp)
                        next_kind = "source"

            if next_time is not None:
                dt = next_time - time
            if dt < min_step:
                dt = min_step

            time = min(float(tstop), time + dt)
            times.append(time)

            timer_due = next_kind == "timer"
            for idx, fire_time in enumerate(tuple(next_timer_times)):
                if time >= fire_time - eps:
                    timer_due = True
                    period = float(timer_periods[idx])
                    while next_timer_times[idx] <= time + eps:
                        next_timer_times[idx] += period

            if timer_due:
                dynamic_step = max(min_step, dynamic_step / 4.0)
            else:
                dynamic_step = min(tstep, dynamic_step * 1.15)

        return times

    @staticmethod
    def _wave_value(source_by_node: Dict[str, Source], node: str, time: float, default: float = 0.0) -> float:
        src = source_by_node.get(node)
        if src is None:
            return default
        return float(src.waveform(time))

    @staticmethod
    def _crossed(prev: float, cur: float, direction: int, eps: float = 1.0e-12) -> bool:
        if direction > 0:
            return prev < -eps and cur >= -eps
        if direction < 0:
            return prev > eps and cur <= eps
        return (prev < -eps and cur >= -eps) or (prev > eps and cur <= eps)

    def _try_compiler_whole_segment_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
        max_step: float,
        min_step: float,
        rust_full_model_required: bool = False,
    ) -> Optional[SimResult]:
        result = self._try_rust_prbs7_full_model_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        result = self._try_timer_static_linear_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
            max_step=max_step,
            min_step=min_step,
        )
        if result is not None:
            return result
        result = self._try_gain_timer_reduction_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        result = self._try_gain_measurement_flow_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        result = self._try_cmp_delay_measurement_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        result = self._try_weighted_sar_loop_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        result = self._try_cppll_reacquire_fastpath(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if result is not None:
            return result
        if rust_full_model_required:
            result = self._try_event_transition_ordered_segment_fastpath(
                rust_backend=rust_backend,
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
            )
            if result is not None:
                return result
        # Audit 091c gate inspector + audit 091d executor body. The
        # inspector still records perf counters; if all gates pass and the
        # caller did not disable the 091d executor, run the generic
        # fixed-grid trace.
        self._inspect_generic_event_state_transition_dispatch(
            rust_backend=rust_backend,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if getattr(self, "_generic_executor_enabled", False):
            result = self._try_generic_event_state_transition_fastpath(
                rust_backend=rust_backend,
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
            )
            if result is not None:
                return result
        return None

    def _try_generic_event_state_transition_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        """Audit 091d executor body.

        For models matching `generic_event_state_transition_v1`, run a
        fixed-grid time loop that calls model.evaluate() at each sample
        point. This bypasses engine's adaptive stepping, source-breakpoint
        scan, err-ratio scan, dynamic-breakpoint scan, and most prepare-step
        orchestration. Body interpretation (event detection, transition
        evolution) stays in the model's compiled Python evaluate(); the
        savings come from skipping engine-side scaffolding.

        Returns SimResult on success, None to fall through to default
        Python path on any gate miss or unsupported feature.
        """
        if len(self.models) != 1 or not self.recorded_signals:
            return None
        model = self.models[0]
        if getattr(model, "_child_models", []) or []:
            return None
        candidate = self._model_candidate(model, "generic_event_state_transition_v1")
        if candidate is None:
            return None
        if tstep is None or tstep <= 0.0 or tstop is None or tstop <= 0.0:
            return None

        model_cls = type(model)
        # Construct the fixed time grid.
        times = self._whole_segment_uniform_times(
            tstop=tstop, record_step=record_step, tstep=tstep,
        )
        # Add source breakpoint times so the executor still hits clk edges.
        for src in self.sources:
            if src.breakpoint_fn is not None:
                self._add_source_breakpoint_times(times, src, tstop)
        times = self._dedupe_times(times, tstop)

        # Build fresh model for the fastpath so we don't perturb the
        # original model's state.
        try:
            fast_model = model_cls()
        except Exception:
            return None
        fast_model.params.update(getattr(model, "params", {}) or {})
        fast_model.node_map = dict(getattr(model, "node_map", {}) or {})

        # Initialize node-voltage dict from source waveforms at t=0.
        nv: Dict[str, float] = {}
        for src in self.sources:
            try:
                nv[src.node] = float(src.waveform(0.0))
            except Exception:
                return None
        try:
            fast_model.initial_step(nv, 0.0)
        except Exception:
            return None

        # Audit 092 Phase B: adaptive refinement. Between consecutive
        # entries in the planned (record-step + source-breakpoint) grid,
        # query model.next_breakpoint() and insert any model-driven
        # breakpoints (transition completion times, timer fires, etc.).
        # This recovers ramp-edge samples that pure fixed-grid sampling
        # would miss. Recording still happens only at the planned grid
        # points, so the CSV column structure matches Python's output.
        # Audit 095: record ALL evaluated time points (planned + adaptive
        # substeps from next_breakpoint), matching Python adaptive
        # stepper's behavior of emitting record samples at cross/transition
        # events. This closes the 091d parity gap by including the
        # high-precision timestamps Python normally emits.
        recorded_names = tuple(self.recorded_signals)
        columns: Dict[str, List[float]] = {name: [] for name in recorded_names}
        emitted_times: List[float] = []
        sources_tuple = tuple(self.sources)
        planned = list(times)
        try:
            prev_t = 0.0
            for record_t in planned:
                rt = float(record_t)
                # Step adaptive substeps until we reach the next planned grid.
                while True:
                    try:
                        bp = fast_model.next_breakpoint(prev_t)
                    except Exception:
                        bp = None
                    if bp is not None and prev_t < bp < rt:
                        sub_t = float(bp)
                        for src in sources_tuple:
                            nv[src.node] = float(src.waveform(sub_t))
                        fast_model.evaluate(nv, sub_t)
                        # Audit 095: record at adaptive breakpoint too.
                        emitted_times.append(sub_t)
                        for name in recorded_names:
                            columns[name].append(float(nv.get(name, 0.0)))
                        prev_t = sub_t
                    else:
                        break
                # Step to the planned grid point and record.
                for src in sources_tuple:
                    nv[src.node] = float(src.waveform(rt))
                fast_model.evaluate(nv, rt)
                emitted_times.append(rt)
                for name in recorded_names:
                    columns[name].append(float(nv.get(name, 0.0)))
                prev_t = rt
        except Exception:
            self._perf_stats["generic_executor_runtime_fallbacks"] = (
                self._perf_stats.get("generic_executor_runtime_fallbacks", 0) + 1
            )
            return None

        # Build SimResult via the existing helper.
        times_arr = array("d", emitted_times)
        self._perf_stats["generic_executor_runs"] = (
            self._perf_stats.get("generic_executor_runs", 0) + 1
        )
        return self._record_trace_result(
            times_arr,
            columns,
            enabled_kind="generic_event_state_transition",
        )

    def _try_event_transition_ordered_segment_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        """EVAS2 strict event+transition ordered-segment runtime.

        This is the first strict-production bridge from the 101 planner into
        the full-model dispatcher.  It is deliberately narrower than the 101
        coverage estimate: no child models, no event-body voltage reads, no
        timer due, and only direct transition contributions.  The useful
        property is that Python ``model.evaluate()`` is not called when this
        path succeeds; the segment trace is produced from typed arrays plus the
        Rust event/body/transition runtime.
        """
        if (
            rust_backend is None
            or len(self.models) != 1
            or not self.recorded_signals
            or tstep is None
            or tstep <= 0.0
            or tstop is None
            or tstop <= 0.0
        ):
            return None
        model = self.models[0]
        if getattr(model, "_child_models", []) or []:
            return None
        model_cls = getattr(model, "__class__", type(model))
        profiles = tuple(getattr(model_cls, "_event_transition_plan_profiles", ()) or ())
        if "event_transition_core" not in profiles:
            return None
        if tuple(getattr(model_cls, "_event_body_voltage_read_nodes", ()) or ()):
            return None

        module = getattr(model_cls, "_module_ast", None)
        node_names = tuple(getattr(model_cls, "_rust_body_ir_node_names", ()) or ())
        if module is None or not node_names:
            return None
        node_slots = {name: idx for idx, name in enumerate(node_names)}
        try:
            runtime = try_build_event_then_transition_shadow_runtime(
                module,
                rust_backend,
                node_slots,
                default_transition=float(
                    getattr(model, "default_transition", 1e-12) or 1e-12
                ),
            )
        except Exception:
            self._perf_stats["rust_full_model_event_transition_core_build_errors"] += 1
            return None
        if runtime is None:
            return None
        if self._event_transition_runtime_uses_timer(runtime):
            return None

        native_result = self._try_event_transition_core_native_trace_fastpath(
            rust_backend=rust_backend,
            runtime=runtime,
            model=model,
            model_cls=model_cls,
            node_names=node_names,
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        if native_result is not None:
            return native_result

        source_by_node = {src.node: src for src in self.sources}
        local_to_external = {
            local: getattr(model, "node_map", {}).get(local, local)
            for local in node_names
        }
        external_to_local = {
            external: local for local, external in local_to_external.items()
        }
        source_local_slots = []
        for local, external in local_to_external.items():
            src = source_by_node.get(external)
            if src is not None:
                source_local_slots.append((node_slots[local], src))

        record_getters = []
        for signal_name in self.recorded_signals:
            source = source_by_node.get(signal_name)
            if source is not None:
                record_getters.append(("source", signal_name, source))
                continue
            local = external_to_local.get(signal_name, signal_name)
            slot = node_slots.get(local)
            if slot is None:
                return None
            record_getters.append(("node", signal_name, slot))

        state_names = tuple(getattr(model_cls, "_state_scalar_names", ()) or ())
        param_names = tuple(
            str(param.name) for param in getattr(module, "parameters", ()) or ()
        )
        node_values = array("d", [0.0] * len(node_names))
        state_values = array(
            "d",
            [float(model.state.get(name, 0.0)) for name in state_names],
        )
        param_values = array("d")
        for name in param_names:
            value = model.params.get(name, 0.0)
            param_values.append(float(value) if isinstance(value, (int, float)) else 0.0)

        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        source_breakpoints = 0
        for src in self.sources:
            if src.breakpoint_fn is not None:
                before = len(raw_times)
                self._add_source_breakpoint_times(raw_times, src, tstop)
                source_breakpoints += len(raw_times) - before
        planned_times = self._dedupe_times(raw_times, tstop)
        if not planned_times or abs(planned_times[0]) > 1e-18:
            planned_times.insert(0, 0.0)

        columns: Dict[str, List[float]] = {
            name: [] for name in self.recorded_signals
        }
        emitted_times: List[float] = []
        fired_total = 0
        transition_breakpoints = 0
        calls_total = 0

        def _refresh_sources(time_value: float) -> None:
            for slot, src in source_local_slots:
                node_values[slot] = float(src.waveform(float(time_value)))

        def _record(time_value: float) -> None:
            emitted_times.append(float(time_value))
            for kind, signal_name, payload in record_getters:
                if kind == "source":
                    columns[signal_name].append(float(payload.waveform(time_value)))
                else:
                    columns[signal_name].append(float(node_values[int(payload)]))

        def _step(time_value: float, *, initial_step: bool = False) -> bool:
            nonlocal calls_total, fired_total
            _refresh_sources(time_value)
            try:
                result = runtime.step(
                    time=float(time_value),
                    node_values=node_values,
                    state_values=state_values,
                    param_values=param_values,
                    initial_step=initial_step,
                )
            except Exception:
                self._perf_stats[
                    "rust_full_model_event_transition_core_runtime_fallbacks"
                ] += 1
                return False
            calls_total += 1
            fired_total += len(getattr(result, "fired_event_statements", ()) or ())
            return True

        if not _step(0.0, initial_step=True):
            return None
        _record(0.0)
        prev_t = 0.0
        min_ramp = 0.15 * float(tstep)
        for planned_t in planned_times:
            target_t = float(planned_t)
            if target_t <= prev_t + 1e-18:
                continue
            while True:
                try:
                    bp = runtime.next_breakpoint(prev_t, min_ramp)
                except Exception:
                    bp = None
                if bp is None or not (prev_t + 1e-18 < bp < target_t - 1e-18):
                    break
                bp = float(bp)
                if not _step(bp):
                    return None
                _record(bp)
                transition_breakpoints += 1
                prev_t = bp
            if not _step(target_t):
                return None
            _record(target_t)
            prev_t = target_t

        integer_names = set(getattr(model_cls, "_integer_state_names", ()) or ())
        for slot, name in enumerate(state_names):
            if slot >= len(state_values):
                break
            value = float(state_values[slot])
            if name in integer_names:
                value = model._to_integer(value)
            model._state_set_by_slot(slot, name, value)

        transition_program = getattr(
            getattr(runtime, "transition_runtime", None),
            "program",
            None,
        )
        for slot in getattr(transition_program, "output_node_slots", ()) or ():
            slot = int(slot)
            if 0 <= slot < len(node_names) and slot < len(node_values):
                model._set_output(
                    node_names[slot],
                    float(node_values[slot]),
                    self.node_voltages,
                )

        self._perf_stats["rust_full_model_event_transition_core_rust_requested"] = 1
        self._perf_stats["rust_full_model_event_transition_core_rust_enabled"] = 1
        self._perf_stats["rust_full_model_event_transition_core_rust_points"] = len(
            emitted_times
        )
        self._perf_stats["rust_full_model_event_transition_core_calls_total"] = (
            calls_total
        )
        self._perf_stats["rust_full_model_event_transition_core_fired_events"] = (
            fired_total
        )
        self._perf_stats[
            "rust_full_model_event_transition_core_transition_breakpoints"
        ] = transition_breakpoints
        self._perf_stats[
            "rust_full_model_event_transition_core_source_breakpoints"
        ] = source_breakpoints
        return self._record_trace_result(
            array("d", emitted_times),
            columns,
            enabled_kind="event_transition_core",
            step_count=max(0, len(emitted_times) - 1),
        )

    def _try_event_transition_core_native_trace_fastpath(
        self,
        *,
        rust_backend,
        runtime,
        model,
        model_cls,
        node_names: Tuple[str, ...],
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        """W1b native scheduler+record for a narrow pulse/cross/transition core."""
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_requested"
        ] = 1
        if rust_backend is None or len(self.recorded_signals) != 1:
            return None
        state_names = tuple(getattr(model_cls, "_state_scalar_names", ()) or ())
        if len(state_names) != 1:
            return None

        transition_program = getattr(
            getattr(runtime, "transition_runtime", None),
            "program",
            None,
        )
        output_slots = tuple(
            int(slot)
            for slot in getattr(transition_program, "output_node_slots", ()) or ()
        )
        reference_slots = tuple(
            getattr(transition_program, "reference_node_slots", ()) or ()
        )
        expr_segments = tuple(
            getattr(transition_program, "expr_segments", ()) or ()
        )
        if len(output_slots) != 1 or reference_slots != (None,) or len(expr_segments) != 4:
            return None
        output_slot = output_slots[0]
        if output_slot < 0 or output_slot >= len(node_names):
            return None
        output_local = node_names[output_slot]
        output_external = getattr(model, "node_map", {}).get(output_local, output_local)
        record_name = next(iter(self.recorded_signals))
        if record_name not in {output_local, output_external}:
            return None

        def _const_value(ops) -> Optional[float]:
            ops = tuple(ops or ())
            if len(ops) != 1 or int(getattr(ops[0], "op_kind", -1)) != BODY_EXPR_CONST:
                return None
            return float(getattr(ops[0], "value", 0.0))

        target_ops = tuple(expr_segments[0] or ())
        if (
            len(target_ops) != 1
            or int(getattr(target_ops[0], "op_kind", -1)) != BODY_EXPR_READ_STATE
            or int(getattr(target_ops[0], "index", -1)) != 0
        ):
            return None
        transition_delay = _const_value(expr_segments[1])
        transition_rise = _const_value(expr_segments[2])
        transition_fall = _const_value(expr_segments[3])
        if transition_delay is None or transition_rise is None or transition_fall is None:
            return None

        event_runtimes = tuple(
            getattr(getattr(runtime, "event_runtime", None), "event_runtimes", ())
            or ()
        )
        if len(event_runtimes) != 2:
            return None

        initial_state_value: Optional[float] = None
        event_state_value: Optional[float] = None
        edge_node_slot: Optional[int] = None
        edge_threshold: Optional[float] = None
        edge_direction: Optional[int] = None

        for event_runtime in event_runtimes:
            due_runtime = getattr(event_runtime, "due_runtime", None)
            triggers = tuple(getattr(getattr(due_runtime, "program", None), "triggers", ()) or ())
            body_batch = getattr(event_runtime, "body_batch", None)
            stmt_ops = tuple(getattr(body_batch, "stmt_ops", ()) or ())
            expr_ops = tuple(getattr(body_batch, "expr_ops", ()) or ())
            if (
                len(stmt_ops) != 1
                or int(getattr(stmt_ops[0], "target_kind", -1)) != TARGET_STATE
                or int(getattr(stmt_ops[0], "target_id", -1)) != 0
                or int(getattr(stmt_ops[0], "expr_start", -1)) != 0
            ):
                return None
            expr_count = int(getattr(stmt_ops[0], "expr_count", -1))
            body_value = _const_value(expr_ops[:expr_count])
            if body_value is None:
                return None
            if len(triggers) != 1:
                return None
            trigger = triggers[0]
            kind = str(getattr(trigger, "kind", ""))
            if kind == "initial_step":
                initial_state_value = body_value
                continue
            if kind != "cross":
                return None
            trigger_expr_ops = tuple(getattr(trigger, "expr_ops", ()) or ())
            if (
                len(trigger_expr_ops) != 3
                or int(getattr(trigger_expr_ops[0], "op_kind", -1)) != BODY_EXPR_READ_NODE
                or int(getattr(trigger_expr_ops[1], "op_kind", -1)) != BODY_EXPR_CONST
                or int(getattr(trigger_expr_ops[2], "op_kind", -1)) != BODY_EXPR_SUB
            ):
                return None
            edge_node_slot = int(getattr(trigger_expr_ops[0], "index", -1))
            edge_threshold = float(getattr(trigger_expr_ops[1], "value", 0.0))
            edge_direction = int(getattr(trigger, "direction", 0))
            event_state_value = body_value

        if (
            initial_state_value is None
            or event_state_value is None
            or edge_node_slot is None
            or edge_threshold is None
            or edge_direction is None
            or edge_node_slot < 0
            or edge_node_slot >= len(node_names)
        ):
            return None

        edge_local = node_names[edge_node_slot]
        edge_external = getattr(model, "node_map", {}).get(edge_local, edge_local)
        source_by_node = {src.node: src for src in self.sources}
        edge_source = source_by_node.get(edge_external)
        if edge_source is None:
            return None
        pulse_meta = self._waveform_metadata(edge_source.waveform)
        if not pulse_meta or pulse_meta.get("kind") != "pulse":
            return None

        sample_step = float(record_step or tstep)
        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        for src in self.sources:
            if src.breakpoint_fn is not None:
                self._add_source_breakpoint_times(raw_times, src, tstop)
        planned_count = len(self._dedupe_times(raw_times, tstop))
        capacity = max(16, planned_count * 6 + 32)
        try:
            time_values, out_values, final_state, stats = (
                rust_backend.event_transition_core_trace_pulse(
                    capacity=capacity,
                    tstop=tstop,
                    sample_step=sample_step,
                    tstep=tstep,
                    pulse_meta=pulse_meta,
                    edge_threshold=edge_threshold,
                    edge_direction=edge_direction,
                    initial_state_value=initial_state_value,
                    event_state_value=event_state_value,
                    transition_delay=transition_delay,
                    transition_rise=transition_rise,
                    transition_fall=transition_fall,
                    default_transition=float(
                        getattr(model, "default_transition", 1e-12) or 1e-12
                    ),
                )
            )
        except RustBackendError:
            self._perf_stats[
                "rust_full_model_event_transition_core_native_trace_fallbacks"
            ] += 1
            return None

        model._state_set_by_slot(0, state_names[0], final_state)
        if len(out_values):
            model._set_output(output_local, float(out_values[-1]), self.node_voltages)
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_enabled"
        ] = 1
        self._perf_stats["rust_full_model_event_transition_core_enabled"] = 1
        self._perf_stats["rust_full_model_event_transition_core_rust_requested"] = 1
        self._perf_stats["rust_full_model_event_transition_core_rust_enabled"] = 1
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_points"
        ] = len(time_values)
        self._perf_stats["rust_full_model_event_transition_core_rust_points"] = len(
            time_values
        )
        self._perf_stats["rust_full_model_event_transition_core_calls_total"] = len(
            time_values
        )
        self._perf_stats["rust_full_model_event_transition_core_fired_events"] = int(
            stats.get("fired_events", 0)
        )
        self._perf_stats[
            "rust_full_model_event_transition_core_transition_breakpoints"
        ] = int(stats.get("transition_breakpoints", 0))
        self._perf_stats[
            "rust_full_model_event_transition_core_source_breakpoints"
        ] = int(stats.get("source_breakpoints", 0))
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_fired_events"
        ] = int(stats.get("fired_events", 0))
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_transition_breakpoints"
        ] = int(stats.get("transition_breakpoints", 0))
        self._perf_stats[
            "rust_full_model_event_transition_core_native_trace_source_breakpoints"
        ] = int(stats.get("source_breakpoints", 0))
        return self._record_trace_result(
            time_values,
            {record_name: out_values},
            enabled_kind="event_transition_core_native_trace",
            step_count=max(0, len(time_values) - 1),
        )

    @staticmethod
    def _event_transition_runtime_uses_timer(runtime) -> bool:
        event_runtimes = tuple(
            getattr(getattr(runtime, "event_runtime", None), "event_runtimes", ())
            or ()
        )
        return any(
            str(getattr(trigger, "kind", "")) == "timer"
            for event_runtime in event_runtimes
            for trigger in getattr(
                getattr(getattr(event_runtime, "due_runtime", None), "program", None),
                "triggers",
                (),
            )
        )

    def _inspect_generic_event_state_transition_dispatch(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> None:
        """Audit 091c gate inspector.

        Walks through the simulator state and records, in perf_stats, whether
        a `generic_event_state_transition_v1` candidate is reachable and (if
        not) which gate blocked it. Returns nothing; the caller falls through
        to the default Python evaluate path. 091d will replace this with an
        actual segment-trace executor.
        """
        stats = self._perf_stats
        # Always increment "saw any model with candidate metadata" so users can
        # quickly see whether the matcher fired during their run.
        eligible_models = 0
        block_reason: Optional[str] = None
        for model in self.models:
            candidate = self._model_candidate(model, "generic_event_state_transition_v1")
            if candidate is None:
                continue
            eligible_models += 1
        stats["generic_executor_models_with_candidate"] = (
            stats.get("generic_executor_models_with_candidate", 0) + eligible_models
        )
        if eligible_models == 0:
            return
        # Walk the same gates a future executor would enforce.
        if len(self.models) != 1:
            block_reason = "multi_model_simulation"
        elif rust_backend is None:
            block_reason = "rust_backend_unavailable"
        else:
            model = self.models[0]
            if getattr(model, "_child_models", []) or []:
                block_reason = "model_has_children"
            elif self._model_candidate(
                model, "generic_event_state_transition_v1"
            ) is None:
                block_reason = "top_model_no_candidate"
            elif not self.recorded_signals:
                block_reason = "no_recorded_signals"
            elif tstep is None or tstep <= 0.0:
                block_reason = "invalid_tstep"
            elif tstop is None or tstop <= 0.0:
                block_reason = "invalid_tstop"
        if block_reason is None:
            stats["generic_executor_dispatchable_runs"] = (
                stats.get("generic_executor_dispatchable_runs", 0) + 1
            )
        else:
            stats["generic_executor_blocked_runs"] = (
                stats.get("generic_executor_blocked_runs", 0) + 1
            )
            key = f"generic_executor_block_reason_{block_reason}"
            stats[key] = stats.get(key, 0) + 1

    def _try_timer_static_linear_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
        max_step: float,
        min_step: float,
    ) -> Optional[SimResult]:
        if (
            rust_backend is None
            or len(self.models) != 1
            or not self.recorded_signals
        ):
            return None
        model = self.models[0]
        if getattr(model, "_child_models", []) or []:
            return None
        model_cls = getattr(model, "__class__", type(model))
        timer_specs = tuple(
            getattr(model_cls, "_event_timer_static_linear_ir_ops", ()) or ()
        )
        evaluate_raw_ops = tuple(
            getattr(model_cls, "_evaluate_ir_static_linear_non_event_ops", ()) or ()
        )
        if not timer_specs or not evaluate_raw_ops:
            return None

        try:
            evaluate_ops_ir = normalize_linear_ops(evaluate_raw_ops)
            timer_starts = []
            timer_periods = []
            event_ops_by_timer = []
            for _key, start_expr, period_expr, event_raw_ops in timer_specs:
                event_ops_ir = normalize_linear_ops(tuple(event_raw_ops))
                timer_start = model._evaluate_rust_static_affine_scalar(
                    start_expr,
                    model.params,
                )
                timer_period = model._evaluate_rust_static_affine_scalar(
                    period_expr,
                    model.params,
                )
                timer_starts.append(float(timer_start))
                timer_periods.append(float(timer_period))
                event_ops_by_timer.append(event_ops_ir)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            return None
        for timer_start, timer_period in zip(timer_starts, timer_periods):
            if timer_period <= 0.0 or not math.isfinite(timer_period):
                return None
            if not math.isfinite(timer_start) or timer_start < -1.0e-18:
                return None

        for event_ops_ir in event_ops_by_timer:
            if any(op.target_kind != TARGET_STATE for op in event_ops_ir):
                return None
            if self._linear_ops_use_node_sources(event_ops_ir):
                return None
        if any(op.target_kind != TARGET_NODE for op in evaluate_ops_ir):
            return None

        state_ids = self._model_state_ids(model)
        if state_ids is None:
            return None
        node_names = self._timer_static_linear_node_names(
            model,
            evaluate_ops_ir,
            extra_nodes=tuple(self.recorded_signals),
        )
        node_names.update(src.node for src in self.sources)
        node_index = {name: idx for idx, name in enumerate(sorted(node_names))}

        try:
            event_batches = [
                self._rust_linear_batch_for_model(
                    rust_backend,
                    model,
                    event_ops_ir,
                    node_index,
                    state_ids,
                )
                for event_ops_ir in event_ops_by_timer
            ]
            evaluate_batch = self._rust_linear_batch_for_model(
                rust_backend,
                model,
                evaluate_ops_ir,
                node_index,
                state_ids,
            )
        except (TypeError, ValueError, KeyError, ZeroDivisionError, RustBackendError):
            return None
        if any(batch is None for batch in event_batches) or evaluate_batch is None:
            return None
        event_op_starts = []
        event_op_counts = []
        combined_event_ops = []
        for batch in event_batches:
            event_op_starts.append(len(combined_event_ops))
            event_op_counts.append(len(batch))
            combined_event_ops.extend(batch.ops)
        combined_event_batch = rust_backend.make_static_linear_batch(combined_event_ops)

        if record_step is None:
            raw_times = self._whole_segment_timer_adaptive_times(
                tstop=tstop,
                tstep=tstep,
                max_step=max_step,
                min_step=min_step,
                timer_starts=tuple(timer_starts),
                timer_periods=tuple(timer_periods),
                source_breakpoint_sources=tuple(
                    src for src in self.sources if src.breakpoint_fn is not None
                ),
            )
        else:
            raw_times = self._whole_segment_uniform_times(
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
            )
            for src in self.sources:
                if src.breakpoint_fn is not None:
                    self._add_source_breakpoint_times(raw_times, src, tstop)
            for timer_start, timer_period in zip(timer_starts, timer_periods):
                self._add_periodic_timer_breakpoint_times(
                    raw_times,
                    tstop=tstop,
                    start=timer_start,
                    period=timer_period,
                )
        times = self._dedupe_times(raw_times, tstop)

        source_by_node = {src.node: src for src in self.sources}
        source_nodes = tuple(sorted(source_by_node))
        source_node_ids = [node_index[node] for node in source_nodes]
        source_values = array("d")
        for time_value in times:
            for node in source_nodes:
                source_values.append(
                    float(source_by_node[node].waveform(float(time_value)))
                )

        node_values = array("d", [0.0]) * len(node_index)
        for node, src in source_by_node.items():
            node_values[node_index[node]] = float(src.waveform(0.0))

        fast_model = model_cls()
        fast_model.params.update(model.params)
        fast_model.node_map = dict(getattr(model, "node_map", {}) or {})
        integer_names = tuple(str(name) for name in getattr(model_cls, "_integer_state_names", ()) or ())
        array_layouts = tuple(getattr(model_cls, "_state_array_ranges", ()) or ())
        fast_model._set_indexed_state_storage(dict(state_ids), integer_names, array_layouts)
        initial_nv = {name: float(node_values[idx]) for name, idx in node_index.items()}
        fast_model.initial_step(initial_nv, 0.0)
        state_values = array("d", fast_model._indexed_state_values or [])

        record_node_ids = []
        for name in self.recorded_signals:
            if name not in node_index:
                return None
            record_node_ids.append(node_index[name])

        self._perf_stats["rust_full_model_timer_static_linear_rust_requested"] = 1
        try:
            if len(timer_starts) == 1:
                flat_values, event_count = rust_backend.timer_static_linear_trace(
                    times,
                    source_node_ids=source_node_ids,
                    source_values=source_values,
                    node_values=node_values,
                    state_values=state_values,
                    event_batch=event_batches[0],
                    evaluate_batch=evaluate_batch,
                    record_node_ids=record_node_ids,
                    timer_start=timer_starts[0],
                    timer_period=timer_periods[0],
                    has_start=True,
                )
            else:
                flat_values, event_count = rust_backend.timer_static_linear_queue_trace(
                    times,
                    source_node_ids=source_node_ids,
                    source_values=source_values,
                    node_values=node_values,
                    state_values=state_values,
                    timer_starts=timer_starts,
                    timer_periods=timer_periods,
                    event_op_starts=event_op_starts,
                    event_op_counts=event_op_counts,
                    event_batch=combined_event_batch,
                    evaluate_batch=evaluate_batch,
                    record_node_ids=record_node_ids,
                )
        except RustBackendError:
            self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
            self._perf_stats["rust_full_model_timer_static_linear_rust_fallbacks"] += 1
            return None

        matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(
            len(times),
            len(record_node_ids),
        )
        columns = {
            name: np.array(matrix[:, idx], copy=True)
            for idx, name in enumerate(self.recorded_signals)
        }
        self._perf_stats["rust_full_model_timer_static_linear_events"] = int(event_count)
        self._perf_stats["rust_full_model_timer_static_linear_rust_enabled"] = 1
        self._perf_stats["rust_full_model_timer_static_linear_rust_points"] = len(times)
        self._perf_stats[
            "rust_full_model_timer_static_linear_default_trace_enabled"
        ] = int(record_step is None)
        self._perf_stats["rust_full_model_timer_static_linear_timer_count"] = len(
            timer_starts
        )
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="timer_static_linear",
            step_count=max(0, len(times) - 1),
        )

    @staticmethod
    def _add_periodic_timer_breakpoint_times(
        times: List[float],
        *,
        tstop: float,
        start: float,
        period: float,
    ) -> None:
        if period <= 0.0 or not math.isfinite(period) or not math.isfinite(start):
            return
        if start < -1.0e-18:
            return
        fire_time = max(0.0, start)
        while fire_time <= tstop + 1.0e-18:
            times.append(min(float(tstop), max(0.0, float(fire_time))))
            fire_time += period

    @staticmethod
    def _linear_ops_use_node_sources(ops) -> bool:
        def terms_use_nodes(terms) -> bool:
            return any(term.source_kind == SOURCE_NODE for term in terms)

        for op in ops:
            if terms_use_nodes(op.terms) or terms_use_nodes(op.false_terms):
                return True
            condition = op.condition
            if condition is not None and (
                terms_use_nodes(condition.left_terms)
                or terms_use_nodes(condition.right_terms)
            ):
                return True
        return False

    @staticmethod
    def _model_state_ids(model) -> Optional[Dict[str, int]]:
        model_cls = getattr(model, "__class__", type(model))
        state_ids: Dict[str, int] = {}
        for name in tuple(getattr(model_cls, "_state_scalar_names", ()) or ()):
            state_ids[str(name)] = len(state_ids)
        slot_name_fn = getattr(model, "_state_array_slot_name", None)
        for array_name, lo, hi, _integer in (
            tuple(getattr(model_cls, "_state_array_ranges", ()) or ())
        ):
            for idx in range(int(lo), int(hi) + 1):
                slot_name = (
                    slot_name_fn(str(array_name), idx)
                    if slot_name_fn is not None
                    else f"{array_name}[{idx}]"
                )
                state_ids[str(slot_name)] = len(state_ids)
        return state_ids

    def _timer_static_linear_node_names(
        self,
        model,
        ops,
        *,
        extra_nodes: Tuple[str, ...],
    ) -> set[str]:
        nodes = {str(name) for name in extra_nodes}

        def add_node(local_name: str) -> None:
            nodes.add(self._external_node(model, str(local_name)))

        def add_terms(terms) -> None:
            for term in terms:
                if term.source_kind == SOURCE_NODE:
                    add_node(term.source_name)

        for op in ops:
            if op.target_kind == TARGET_NODE:
                add_node(op.target_name)
            add_terms(op.terms)
            add_terms(op.false_terms)
            if op.condition is not None:
                add_terms(op.condition.left_terms)
                add_terms(op.condition.right_terms)
        return nodes

    def _rust_linear_batch_for_model(
        self,
        rust_backend,
        model,
        ops,
        node_index: Dict[str, int],
        state_ids: Dict[str, int],
    ):
        converted = []

        def scalar(value):
            return model._evaluate_rust_static_affine_scalar(value, model.params)

        def convert_terms(ir_terms):
            terms = []
            for term in ir_terms:
                gain = scalar(term.gain)
                if term.source_kind == SOURCE_NODE:
                    external = self._external_node(model, term.source_name)
                    terms.append(
                        LinearTerm(
                            source_kind=SOURCE_NODE,
                            source_id=node_index[external],
                            gain=gain,
                        )
                    )
                elif term.source_kind == SOURCE_STATE:
                    terms.append(
                        LinearTerm(
                            source_kind=SOURCE_STATE,
                            source_id=state_ids[term.source_name],
                            gain=gain,
                        )
                    )
                else:
                    raise ValueError(f"unsupported source kind: {term.source_kind!r}")
            return tuple(terms)

        def convert_condition(condition):
            if condition is None:
                return None
            return LinearCondition(
                op_kind=condition.op_kind,
                left_bias=scalar(condition.left_bias),
                left_terms=convert_terms(condition.left_terms),
                right_bias=scalar(condition.right_bias),
                right_terms=convert_terms(condition.right_terms),
            )

        for op in ops:
            if op.target_kind == TARGET_NODE:
                target_id = node_index[self._external_node(model, op.target_name)]
            elif op.target_kind == TARGET_STATE:
                target_id = state_ids[op.target_name]
            else:
                raise ValueError(f"unsupported target kind: {op.target_kind!r}")
            converted.append(
                LinearOp(
                    target_kind=op.target_kind,
                    target_id=target_id,
                    bias=scalar(op.bias),
                    terms=convert_terms(op.terms),
                    condition=convert_condition(op.condition),
                    false_bias=scalar(op.false_bias),
                    false_terms=convert_terms(op.false_terms),
                    target_integer=op.target_integer,
                )
            )
        return rust_backend.make_static_linear_batch(converted)

    def _try_gain_timer_reduction_fastpath(
        self,
        *,
        rust_backend=None,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        if not self.models or len(self.models) != 1:
            return None
        model = self.models[0]
        candidate = self._model_candidate(model, "gain_timer_reduction_v1")
        if candidate is None:
            return None
        try:
            (
                _kind,
                vdd_port,
                vss_port,
                vinp_port,
                vinn_port,
                voutp_port,
                voutn_port,
                gain_port,
                valid_port,
                sample_period_param,
                start_time_param,
                gain_scale_param,
                min_input_span_param,
                tedge_param,
            ) = candidate
        except (TypeError, ValueError):
            return None
        port_nodes = {
            port: self._external_node(model, port)
            for port in (
                vdd_port, vss_port, vinp_port, vinn_port,
                voutp_port, voutn_port, gain_port, valid_port,
            )
        }
        generated_nodes = set(port_nodes.values())
        if any(name not in generated_nodes for name in self.recorded_signals):
            return None
        source_by_node = {src.node: src for src in self.sources}
        required_sources = (
            port_nodes[vdd_port],
            port_nodes[vss_port],
            port_nodes[vinp_port],
            port_nodes[vinn_port],
            port_nodes[voutp_port],
            port_nodes[voutn_port],
        )
        if any(node not in source_by_node for node in required_sources):
            return None
        try:
            sample_period = float(model.params[sample_period_param])
            start_time = float(model.params[start_time_param])
            gain_scale = float(model.params[gain_scale_param])
            min_input_span = float(model.params[min_input_span_param])
            tedge = float(model.params[tedge_param])
        except Exception:
            return None
        if sample_period <= 0.0 or gain_scale == 0.0:
            return None

        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        sample_times: List[float] = []
        sample_count_est = int(math.floor(tstop / sample_period + 1.0e-9)) + 1
        for idx in range(max(1, sample_count_est + 1)):
            event_t = idx * sample_period
            if event_t > tstop + 1.0e-18:
                break
            sample_times.append(event_t)
            raw_times.append(event_t)
            for frac in (0.25, 0.5, 0.75, 1.0):
                raw_times.append(event_t + frac * tedge)
        times = self._dedupe_times(raw_times, tstop)
        rust_gain_trace_enabled = (
            os.environ.get("EVAS_RUST_GAIN_TIMER_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off", "disabled"}
        )
        self._perf_stats["rust_full_model_gain_timer_reduction_rust_requested"] = int(
            bool(rust_backend is not None and rust_gain_trace_enabled)
        )
        if rust_backend is not None and rust_gain_trace_enabled:
            try:
                def values_for(node: str, rows) -> array:
                    return array(
                        "d",
                        (
                            self._wave_value(source_by_node, node, float(time))
                            for time in rows
                        ),
                    )

                flat_values, sample_events = rust_backend.gain_timer_reduction_trace(
                    times,
                    array("d", sample_times),
                    point_vdd=values_for(port_nodes[vdd_port], times),
                    point_vss=values_for(port_nodes[vss_port], times),
                    point_vinp=values_for(port_nodes[vinp_port], times),
                    point_vinn=values_for(port_nodes[vinn_port], times),
                    point_voutp=values_for(port_nodes[voutp_port], times),
                    point_voutn=values_for(port_nodes[voutn_port], times),
                    sample_vdd=values_for(port_nodes[vdd_port], sample_times),
                    sample_vss=values_for(port_nodes[vss_port], sample_times),
                    sample_vinp=values_for(port_nodes[vinp_port], sample_times),
                    sample_vinn=values_for(port_nodes[vinn_port], sample_times),
                    sample_voutp=values_for(port_nodes[voutp_port], sample_times),
                    sample_voutn=values_for(port_nodes[voutn_port], sample_times),
                    start_time=start_time,
                    gain_scale=gain_scale,
                    min_input_span=min_input_span,
                    tedge=tedge,
                )
                matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(
                    len(times),
                    8,
                )
                column_by_node = {
                    port_nodes[vdd_port]: 0,
                    port_nodes[vss_port]: 1,
                    port_nodes[vinp_port]: 2,
                    port_nodes[vinn_port]: 3,
                    port_nodes[voutp_port]: 4,
                    port_nodes[voutn_port]: 5,
                    port_nodes[gain_port]: 6,
                    port_nodes[valid_port]: 7,
                }
                columns = {
                    name: np.array(matrix[:, column_by_node[name]], copy=True)
                    for name in self.recorded_signals
                }
                self._perf_stats["rust_full_model_gain_timer_reduction_samples"] = sample_events
                self._perf_stats["rust_full_model_gain_timer_reduction_rust_enabled"] = 1
                self._perf_stats["rust_full_model_gain_timer_reduction_rust_points"] = len(times)
                return self._record_trace_result(
                    times,
                    columns,
                    enabled_kind="gain_timer_reduction",
                )
            except RustBackendError:
                self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
                self._perf_stats["rust_full_model_gain_timer_reduction_rust_fallbacks"] += 1

        gain_transition = TransitionState()
        valid_transition = TransitionState()
        in_min = 1.0e9
        in_max = -1.0e9
        out_min = 1.0e9
        out_max = -1.0e9
        gain_q = 0.0
        valid_q = 0
        next_sample = 0.0
        sample_events = 0
        columns = {name: [] for name in self.recorded_signals}

        for t in times:
            while next_sample <= t + 1.0e-18:
                vdd_val = self._wave_value(source_by_node, port_nodes[vdd_port], next_sample)
                vss_val = self._wave_value(source_by_node, port_nodes[vss_port], next_sample)
                if next_sample >= start_time - 1.0e-18:
                    vin_diff = (
                        self._wave_value(source_by_node, port_nodes[vinp_port], next_sample)
                        - self._wave_value(source_by_node, port_nodes[vinn_port], next_sample)
                    )
                    vout_diff = (
                        self._wave_value(source_by_node, port_nodes[voutp_port], next_sample)
                        - self._wave_value(source_by_node, port_nodes[voutn_port], next_sample)
                    )
                    in_min = min(in_min, vin_diff)
                    in_max = max(in_max, vin_diff)
                    out_min = min(out_min, vout_diff)
                    out_max = max(out_max, vout_diff)
                    in_span = in_max - in_min
                    out_span = out_max - out_min
                    if in_span > min_input_span:
                        gain_q = out_span / in_span
                        valid_q = 1
                gain_transition.evaluate(next_sample)
                valid_transition.evaluate(next_sample)
                gain_transition.set_target(
                    next_sample,
                    (vdd_val - vss_val) * gain_q / gain_scale,
                    0.0,
                    tedge,
                    tedge,
                )
                valid_transition.set_target(
                    next_sample,
                    (vdd_val - vss_val) if valid_q else 0.0,
                    0.0,
                    tedge,
                    tedge,
                )
                sample_events += 1
                next_sample += sample_period

            values_by_node = {
                port_nodes[vdd_port]: self._wave_value(source_by_node, port_nodes[vdd_port], float(t)),
                port_nodes[vss_port]: self._wave_value(source_by_node, port_nodes[vss_port], float(t)),
                port_nodes[vinp_port]: self._wave_value(source_by_node, port_nodes[vinp_port], float(t)),
                port_nodes[vinn_port]: self._wave_value(source_by_node, port_nodes[vinn_port], float(t)),
                port_nodes[voutp_port]: self._wave_value(source_by_node, port_nodes[voutp_port], float(t)),
                port_nodes[voutn_port]: self._wave_value(source_by_node, port_nodes[voutn_port], float(t)),
            }
            vss_now = values_by_node[port_nodes[vss_port]]
            values_by_node[port_nodes[gain_port]] = vss_now + gain_transition.evaluate(float(t))
            values_by_node[port_nodes[valid_port]] = vss_now + valid_transition.evaluate(float(t))
            for name in columns:
                columns[name].append(float(values_by_node.get(name, 0.0)))

        self._perf_stats["rust_full_model_gain_timer_reduction_samples"] = sample_events
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="gain_timer_reduction",
        )

    def _try_gain_measurement_flow_fastpath(
        self,
        *,
        rust_backend=None,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        if rust_backend is None or len(self.models) != 4:
            return None

        rust_gain_flow_enabled = (
            os.environ.get("EVAS_RUST_GAIN_MEASUREMENT_FLOW_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off", "disabled"}
        )
        if not rust_gain_flow_enabled:
            return None

        def has_shape(model, ports: Tuple[str, ...], params: Tuple[str, ...]) -> bool:
            model_ports = tuple(getattr(model.__class__, "_module_ports", ()) or ())
            if set(model_ports) != set(ports):
                return False
            return all(name in getattr(model, "params", {}) for name in params)

        def one_model(ports: Tuple[str, ...], params: Tuple[str, ...]):
            matches = [model for model in self.models if has_shape(model, ports, params)]
            return matches[0] if len(matches) == 1 else None

        vin_model = one_model(
            ("CLK", "RST_N", "VOUT_P", "VOUT_N"),
            ("vdd", "vth", "ampl", "freq", "sigma", "SEED"),
        )
        lfsr_model = one_model(
            ("DPN", "VDD", "VSS", "CLK", "EN", "RSTB"),
            ("seed",),
        )
        dither_model = one_model(
            ("VRES_P", "VRES_N", "DPN", "VOUT_P", "VOUT_N"),
            ("vth", "DITHER_AMP"),
        )
        amp_model = one_model(
            ("VIN_P", "VIN_N", "VOUT_P", "VOUT_N"),
            ("vdd", "ACTUAL_GAIN"),
        )
        if None in (vin_model, lfsr_model, dither_model, amp_model):
            return None

        lfsr_arrays = tuple(getattr(lfsr_model.__class__, "_state_array_ranges", ()) or ())
        if ("lfsr_r", 0, 31, True) not in lfsr_arrays or ("tmp_lfsr_r", 0, 32, True) not in lfsr_arrays:
            return None

        vin_nodes = {port: self._external_node(vin_model, port) for port in ("CLK", "RST_N", "VOUT_P", "VOUT_N")}
        lfsr_nodes = {port: self._external_node(lfsr_model, port) for port in ("DPN", "VDD", "VSS", "CLK", "EN", "RSTB")}
        dither_nodes = {port: self._external_node(dither_model, port) for port in ("VRES_P", "VRES_N", "DPN", "VOUT_P", "VOUT_N")}
        amp_nodes = {port: self._external_node(amp_model, port) for port in ("VIN_P", "VIN_N", "VOUT_P", "VOUT_N")}

        if (
            vin_nodes["CLK"] != lfsr_nodes["CLK"]
            or vin_nodes["RST_N"] != lfsr_nodes["RSTB"]
            or vin_nodes["VOUT_P"] != dither_nodes["VRES_P"]
            or vin_nodes["VOUT_N"] != dither_nodes["VRES_N"]
            or lfsr_nodes["DPN"] != dither_nodes["DPN"]
            or dither_nodes["VOUT_P"] != amp_nodes["VIN_P"]
            or dither_nodes["VOUT_N"] != amp_nodes["VIN_N"]
        ):
            return None

        generated_nodes = {
            vin_nodes["VOUT_P"],
            vin_nodes["VOUT_N"],
            amp_nodes["VOUT_P"],
            amp_nodes["VOUT_N"],
        }
        if any(name not in generated_nodes for name in self.recorded_signals):
            return None

        source_by_node = {src.node: src for src in self.sources}
        required_source_nodes = (
            vin_nodes["CLK"],
            vin_nodes["RST_N"],
            lfsr_nodes["VDD"],
            lfsr_nodes["VSS"],
        )
        if any(node not in source_by_node for node in required_source_nodes):
            return None

        clk_meta = self._waveform_metadata(source_by_node[vin_nodes["CLK"]].waveform)
        vdd_meta = self._waveform_metadata(source_by_node[lfsr_nodes["VDD"]].waveform)
        vss_meta = self._waveform_metadata(source_by_node[lfsr_nodes["VSS"]].waveform)
        if (
            not clk_meta
            or clk_meta.get("kind") != "pulse"
            or not vdd_meta
            or vdd_meta.get("kind") != "dc"
            or not vss_meta
            or vss_meta.get("kind") != "dc"
        ):
            return None

        try:
            vdd = float(vdd_meta["voltage"])
            vss = float(vss_meta["voltage"])
            vin_vdd = float(vin_model.params["vdd"])
            vin_vth = float(vin_model.params["vth"])
            ampl = float(vin_model.params["ampl"])
            freq = float(vin_model.params["freq"])
            sigma = float(vin_model.params["sigma"])
            seed = int(float(vin_model.params["SEED"]))
            lfsr_seed = int(float(lfsr_model.params["seed"]))
            dither_vth = float(dither_model.params["vth"])
            dither_amp = float(dither_model.params["DITHER_AMP"])
            amp_vdd = float(amp_model.params["vdd"])
            actual_gain = float(amp_model.params["ACTUAL_GAIN"])
            vin_transition = float(getattr(vin_model, "default_transition", 30.0e-12) or 30.0e-12)
            lfsr_transition = float(getattr(lfsr_model, "default_transition", 10.0e-12) or 10.0e-12)
        except Exception:
            return None
        if (
            freq <= 0.0
            or vin_transition <= 0.0
            or lfsr_transition <= 0.0
            or abs(vin_vdd - amp_vdd) > 1.0e-12
            or abs(vin_vdd - vdd) > 1.0e-12
        ):
            return None

        self._perf_stats["rust_full_model_gain_measurement_flow_rust_requested"] = 1
        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        for src in source_by_node.values():
            self._add_source_breakpoint_times(raw_times, src, tstop)

        rst_node = vin_nodes["RST_N"]
        vcm = 0.5 * vin_vdd
        rng = random.Random(seed)
        vin_event_times: List[float] = []
        vin_event_vinp: List[float] = []
        vin_event_vinn: List[float] = []
        for event_t in self._pulse_threshold_cross_times(clk_meta, tstop, vin_vth, +1):
            if self._wave_value(source_by_node, rst_node, event_t) <= vin_vth:
                continue
            target_vinp = (
                vcm
                + ampl * math.sin(2.0 * math.pi * freq * event_t)
                + sigma * rng.gauss(0.0, 1.0)
            )
            vin_event_times.append(event_t)
            vin_event_vinp.append(target_vinp)
            vin_event_vinn.append(vcm)
            raw_times.append(event_t)
            for frac in (0.25, 0.5, 0.75, 1.0):
                raw_times.append(event_t + frac * vin_transition)

        lfsr_event_times: List[float] = []
        for event_t in self._pulse_threshold_cross_times(clk_meta, tstop, 0.5, +1):
            if self._wave_value(source_by_node, rst_node, event_t) <= 0.5:
                continue
            lfsr_event_times.append(event_t)
            raw_times.append(event_t)
            for frac in (0.25, 0.5, 0.75, 1.0):
                raw_times.append(event_t + frac * lfsr_transition)

        times = self._dedupe_times(raw_times, tstop)
        try:
            flat_values, vin_events, lfsr_events = rust_backend.gain_measurement_flow_trace(
                times,
                vin_event_times=array("d", vin_event_times),
                vin_event_vinp=array("d", vin_event_vinp),
                vin_event_vinn=array("d", vin_event_vinn),
                lfsr_event_times=array("d", lfsr_event_times),
                vcm=vcm,
                vth=dither_vth,
                dither_amp=dither_amp,
                actual_gain=actual_gain,
                vin_transition=vin_transition,
                lfsr_transition=lfsr_transition,
                vdd=vdd,
                vss=vss,
                lfsr_seed=lfsr_seed,
            )
        except RustBackendError:
            self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
            self._perf_stats["rust_full_model_gain_measurement_flow_rust_fallbacks"] += 1
            return None

        matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(len(times), 4)
        column_by_node = {
            vin_nodes["VOUT_P"]: 0,
            vin_nodes["VOUT_N"]: 1,
            amp_nodes["VOUT_P"]: 2,
            amp_nodes["VOUT_N"]: 3,
        }
        columns = {
            name: np.array(matrix[:, column_by_node[name]], copy=True)
            for name in self.recorded_signals
        }
        self._perf_stats["rust_full_model_gain_measurement_flow_vin_events"] = vin_events
        self._perf_stats["rust_full_model_gain_measurement_flow_lfsr_events"] = lfsr_events
        self._perf_stats["rust_full_model_gain_measurement_flow_rust_enabled"] = 1
        self._perf_stats["rust_full_model_gain_measurement_flow_rust_points"] = len(times)
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="gain_measurement_flow",
        )

    def _try_cmp_delay_measurement_fastpath(
        self,
        *,
        rust_backend=None,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        cmp_model = None
        edge_model = None
        cmp_candidate = None
        edge_candidate = None
        for model in self.models:
            candidate = self._model_candidate(model, "cmp_delay_log_transition_v1")
            if candidate is not None:
                cmp_model = model
                cmp_candidate = candidate
            candidate = self._model_candidate(model, "edge_interval_timer_v1")
            if candidate is not None:
                edge_model = model
                edge_candidate = candidate
        if cmp_model is None or edge_model is None or cmp_candidate is None or edge_candidate is None:
            return None
        try:
            (
                _kind,
                clk_port,
                vinn_port,
                vinp_port,
                outn_port,
                outp_port,
                vss_port,
                vdd_port,
                voffset_param,
                tau_param,
                td0_param,
                tdmin_param,
                tdmax_param,
                tedge_state,
            ) = cmp_candidate
            (
                _edge_kind,
                edge_clk1_port,
                edge_clk2_port,
                delay_port,
                edge_vth_param,
            ) = edge_candidate
        except (TypeError, ValueError):
            return None
        cmp_nodes = {
            port: self._external_node(cmp_model, port)
            for port in (clk_port, vinn_port, vinp_port, outn_port, outp_port, vss_port, vdd_port)
        }
        edge_nodes = {
            port: self._external_node(edge_model, port)
            for port in (edge_clk1_port, edge_clk2_port, delay_port)
        }
        if (
            edge_nodes[edge_clk1_port] != cmp_nodes[clk_port]
            or edge_nodes[edge_clk2_port] != cmp_nodes[outp_port]
        ):
            return None
        generated_nodes = {
            cmp_nodes[clk_port],
            cmp_nodes[vinn_port],
            cmp_nodes[vinp_port],
            cmp_nodes[outn_port],
            cmp_nodes[outp_port],
            edge_nodes[delay_port],
        }
        if any(name not in generated_nodes for name in self.recorded_signals):
            return None
        source_by_node = {src.node: src for src in self.sources}
        for node in (cmp_nodes[clk_port], cmp_nodes[vinn_port], cmp_nodes[vinp_port], cmp_nodes[vss_port], cmp_nodes[vdd_port]):
            if node not in source_by_node:
                return None
        try:
            voffset = float(cmp_model.params[voffset_param])
            tau = float(cmp_model.params[tau_param])
            td0 = float(cmp_model.params[td0_param])
            td_min = float(cmp_model.params[tdmin_param])
            td_max = float(cmp_model.params[tdmax_param])
            tedge = float(cmp_model.state.get(tedge_state, 30.0e-12))
            edge_vth = float(edge_model.params[edge_vth_param])
        except Exception:
            return None
        clk_src = source_by_node[cmp_nodes[clk_port]]
        clk_meta = self._waveform_metadata(clk_src.waveform)
        if not clk_meta or clk_meta.get("kind") != "pulse":
            return None

        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        for src in source_by_node.values():
            self._add_source_breakpoint_times(raw_times, src, tstop)
        clk_rises = self._pulse_threshold_cross_times(clk_meta, tstop, edge_vth, +1)
        clk_falls = self._pulse_threshold_cross_times(clk_meta, tstop, edge_vth, -1)
        for event_t in (*clk_rises, *clk_falls):
            raw_times.append(event_t)
            for frac in (0.25, 0.5, 0.75, 1.0):
                raw_times.append(event_t + frac * tedge)
        for event_t in clk_rises:
            vdd_val = self._wave_value(source_by_node, cmp_nodes[vdd_port], event_t, 0.9)
            vdiff = (
                self._wave_value(source_by_node, cmp_nodes[vinp_port], event_t)
                - self._wave_value(source_by_node, cmp_nodes[vinn_port], event_t)
                - voffset
            )
            vdiff_eff = max(abs(vdiff), 1.0e-9)
            td = td0 + tau * math.log(vdd_val / vdiff_eff)
            td = min(td_max, max(td_min, td))
            if vdiff > 0.0:
                out_cross = event_t + td + 0.5 * tedge
                raw_times.append(out_cross)
                for frac in (0.25, 0.5, 0.75, 1.0):
                    raw_times.append(event_t + td + frac * tedge)
        times = self._dedupe_times(raw_times, tstop)

        rust_cmp_trace_enabled = (
            os.environ.get("EVAS_RUST_CMP_DELAY_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off", "disabled"}
        )
        self._perf_stats["rust_full_model_cmp_delay_rust_requested"] = int(
            bool(rust_backend is not None and rust_cmp_trace_enabled)
        )
        if rust_backend is not None and rust_cmp_trace_enabled:
            try:
                def values_for(node: str) -> array:
                    return array(
                        "d",
                        (
                            self._wave_value(source_by_node, node, float(time))
                            for time in times
                        ),
                    )

                flat_values, clock_events = rust_backend.cmp_delay_trace(
                    times,
                    point_clk=values_for(cmp_nodes[clk_port]),
                    point_vinn=values_for(cmp_nodes[vinn_port]),
                    point_vinp=values_for(cmp_nodes[vinp_port]),
                    point_vdd=values_for(cmp_nodes[vdd_port]),
                    voffset=voffset,
                    tau=tau,
                    td0=td0,
                    td_min=td_min,
                    td_max=td_max,
                    tedge=tedge,
                    edge_vth=edge_vth,
                )
                matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(
                    len(times),
                    6,
                )
                column_by_node = {
                    cmp_nodes[clk_port]: 0,
                    cmp_nodes[vinn_port]: 1,
                    cmp_nodes[vinp_port]: 2,
                    cmp_nodes[outn_port]: 3,
                    cmp_nodes[outp_port]: 4,
                    edge_nodes[delay_port]: 5,
                }
                columns = {
                    name: np.array(matrix[:, column_by_node[name]], copy=True)
                    for name in self.recorded_signals
                }
                self._perf_stats["rust_full_model_cmp_delay_clock_events"] = clock_events
                self._perf_stats["rust_full_model_cmp_delay_rust_enabled"] = 1
                self._perf_stats["rust_full_model_cmp_delay_rust_points"] = len(times)
                return self._record_trace_result(
                    times,
                    columns,
                    enabled_kind="cmp_delay",
                )
            except RustBackendError:
                self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
                self._perf_stats["rust_full_model_cmp_delay_rust_fallbacks"] += 1

        outp_transition = TransitionState()
        outn_transition = TransitionState()
        delay_transition = TransitionState()
        prev_clk_expr = self._wave_value(source_by_node, cmp_nodes[clk_port], 0.0) - edge_vth
        prev_outp_expr = -edge_vth
        t_start = 0.0
        armed = False
        clock_events = 0
        columns = {name: [] for name in self.recorded_signals}

        for raw_t in times:
            t = float(raw_t)
            clk_val = self._wave_value(source_by_node, cmp_nodes[clk_port], t)
            vinp_val = self._wave_value(source_by_node, cmp_nodes[vinp_port], t)
            vinn_val = self._wave_value(source_by_node, cmp_nodes[vinn_port], t)
            vdd_val = self._wave_value(source_by_node, cmp_nodes[vdd_port], t, 0.9)
            clk_expr = clk_val - edge_vth
            if self._crossed(prev_clk_expr, clk_expr, +1):
                t_start = t
                armed = True
                vdiff = vinp_val - vinn_val - voffset
                vdiff_eff = max(abs(vdiff), 1.0e-9)
                td = td0 + tau * math.log(vdd_val / vdiff_eff)
                td = min(td_max, max(td_min, td))
                outp_transition.evaluate(t)
                outn_transition.evaluate(t)
                if vdiff > 0.0:
                    outp_transition.set_target(t, vdd_val, td, tedge, tedge)
                    outn_transition.set_target(t, 0.0, td, tedge, tedge)
                else:
                    outp_transition.set_target(t, 0.0, td, tedge, tedge)
                    outn_transition.set_target(t, vdd_val, td, tedge, tedge)
                clock_events += 1
            elif self._crossed(prev_clk_expr, clk_expr, -1):
                outp_transition.evaluate(t)
                outn_transition.evaluate(t)
                outp_transition.set_target(t, 0.0, 0.0, tedge, tedge)
                outn_transition.set_target(t, 0.0, 0.0, tedge, tedge)
                clock_events += 1
            outp_val = outp_transition.evaluate(t)
            outn_val = outn_transition.evaluate(t)
            outp_expr = outp_val - edge_vth
            if armed and self._crossed(prev_outp_expr, outp_expr, +1):
                delay_ps = (t - t_start) * 1.0e12
                delay_transition.evaluate(t)
                delay_transition.set_target(t, delay_ps, 0.0, 10.0e-12, 10.0e-12)
                armed = False
            delay_val = delay_transition.evaluate(t)
            values_by_node = {
                cmp_nodes[clk_port]: clk_val,
                cmp_nodes[vinp_port]: vinp_val,
                cmp_nodes[vinn_port]: vinn_val,
                cmp_nodes[outp_port]: outp_val,
                cmp_nodes[outn_port]: outn_val,
                edge_nodes[delay_port]: delay_val,
            }
            for name in columns:
                columns[name].append(float(values_by_node.get(name, 0.0)))
            prev_clk_expr = clk_expr
            prev_outp_expr = outp_expr

        self._perf_stats["rust_full_model_cmp_delay_clock_events"] = clock_events
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="cmp_delay",
        )

    def _try_weighted_sar_loop_fastpath(
        self,
        *,
        rust_backend=None,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        sar_model = dac_model = sh_model = None
        sar_candidate = dac_candidate = sh_candidate = None
        for model in self.models:
            candidate = self._model_candidate(model, "weighted_sar_adc_v1")
            if candidate is not None:
                sar_model = model
                sar_candidate = candidate
            candidate = self._model_candidate(model, "weighted_dac_v1")
            if candidate is not None:
                dac_model = model
                dac_candidate = candidate
            candidate = self._model_candidate(model, "sample_hold_rising_v1")
            if candidate is not None:
                sh_model = model
                sh_candidate = candidate
        if sar_model is None or dac_model is None or sh_model is None:
            return None
        try:
            (
                _sar_kind,
                vin_port,
                clks_port,
                rst_port,
                dout_ports,
                bit_index_port,
                trial_code_port,
                trial_vdac_port,
                cmp_decision_port,
                conv_done_port,
                vin_sample_port,
                vdd_param,
                vth_param,
                width,
            ) = sar_candidate
            (_dac_kind, dac_din_ports, dac_vout_port, dac_vdd_param, dac_vth_param) = dac_candidate
            (_sh_kind, sh_vin_port, sh_clk_port, sh_vdd_port, sh_vss_port, sh_rst_port, sh_vout_port, sh_tr_param) = sh_candidate
        except (TypeError, ValueError):
            return None
        width = int(width)
        if width <= 0 or len(tuple(dout_ports)) != width or len(tuple(dac_din_ports)) != width:
            return None
        sar_nodes = {
            port: self._external_node(sar_model, port)
            for port in (
                vin_port, clks_port, rst_port,
                bit_index_port, trial_code_port, trial_vdac_port,
                cmp_decision_port, conv_done_port, vin_sample_port,
                *tuple(dout_ports),
            )
        }
        dac_nodes = {
            port: self._external_node(dac_model, port)
            for port in (*tuple(dac_din_ports), dac_vout_port)
        }
        sh_nodes = {
            port: self._external_node(sh_model, port)
            for port in (sh_vin_port, sh_clk_port, sh_vdd_port, sh_vss_port, sh_rst_port, sh_vout_port)
        }
        if (
            sar_nodes[vin_port] != sh_nodes[sh_vout_port]
            or sar_nodes[clks_port] != sh_nodes[sh_clk_port]
            or sar_nodes[rst_port] != sh_nodes[sh_rst_port]
            or any(dac_nodes[dac_port] != sar_nodes[dout_port] for dac_port, dout_port in zip(tuple(dac_din_ports), tuple(dout_ports)))
        ):
            return None
        generated_nodes = {
            sh_nodes[sh_vin_port],
            sar_nodes[vin_port],
            sar_nodes[clks_port],
            sar_nodes[rst_port],
            dac_nodes[dac_vout_port],
            sar_nodes[bit_index_port],
            sar_nodes[trial_code_port],
            sar_nodes[trial_vdac_port],
            sar_nodes[cmp_decision_port],
            sar_nodes[conv_done_port],
            sar_nodes[vin_sample_port],
            *[sar_nodes[port] for port in tuple(dout_ports)],
        }
        if any(name not in generated_nodes for name in self.recorded_signals):
            return None
        source_by_node = {src.node: src for src in self.sources}
        for node in (
            sh_nodes[sh_vin_port],
            sar_nodes[clks_port],
            sar_nodes[rst_port],
            sh_nodes[sh_vdd_port],
            sh_nodes[sh_vss_port],
        ):
            if node not in source_by_node:
                return None
        try:
            vdd = float(sar_model.params[vdd_param])
            vth = float(sar_model.params[vth_param])
            dac_vdd = float(dac_model.params[dac_vdd_param])
            dac_vth = float(dac_model.params[dac_vth_param])
            sh_tr = float(sh_model.params[sh_tr_param])
        except Exception:
            return None
        if abs(dac_vdd - vdd) > 1.0e-12 or abs(dac_vth - vth) > 1.0e-12:
            return None
        clk_src = source_by_node[sar_nodes[clks_port]]
        clk_meta = self._waveform_metadata(clk_src.waveform)
        if not clk_meta or clk_meta.get("kind") != "pulse":
            return None
        default_tr = float(getattr(sar_model, "default_transition", 30.0e-12) or 30.0e-12)
        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )
        for src in source_by_node.values():
            self._add_source_breakpoint_times(raw_times, src, tstop)
        clk_rises = self._pulse_threshold_cross_times(clk_meta, tstop, vth, +1)
        clk_falls = self._pulse_threshold_cross_times(clk_meta, tstop, vth, -1)
        for event_t in (*clk_rises, *clk_falls):
            raw_times.append(event_t)
            for offset in (0.25 * default_tr, 0.5 * default_tr, 0.75 * default_tr, default_tr, 1.5 * default_tr):
                raw_times.append(event_t + offset)
        times = self._dedupe_times(raw_times, tstop)

        rust_sar_trace_enabled = (
            os.environ.get("EVAS_RUST_SAR_LOOP_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off", "disabled"}
        )
        self._perf_stats["rust_full_model_sar_loop_rust_requested"] = int(
            bool(rust_backend is not None and rust_sar_trace_enabled)
        )
        if rust_backend is not None and rust_sar_trace_enabled:
            try:
                def values_for(node: str) -> array:
                    return array(
                        "d",
                        (
                            self._wave_value(source_by_node, node, float(time))
                            for time in times
                        ),
                    )

                trace_values, clock_events = rust_backend.sar_loop_trace(
                    times,
                    point_vin=values_for(sh_nodes[sh_vin_port]),
                    point_clk=values_for(sar_nodes[clks_port]),
                    point_rst=values_for(sar_nodes[rst_port]),
                    vdd=vdd,
                    vth=vth,
                    sh_tr=sh_tr,
                    default_tr=default_tr,
                    width=width,
                )
                signal_count = 11 + width
                column_index = {
                    sh_nodes[sh_vin_port]: 0,
                    sar_nodes[vin_port]: 1,
                    sar_nodes[clks_port]: 2,
                    sar_nodes[rst_port]: 3,
                    dac_nodes[dac_vout_port]: 4,
                    sar_nodes[bit_index_port]: 5,
                    sar_nodes[trial_code_port]: 6,
                    sar_nodes[trial_vdac_port]: 7,
                    sar_nodes[cmp_decision_port]: 8,
                    sar_nodes[conv_done_port]: 9,
                    sar_nodes[vin_sample_port]: 10,
                }
                for idx, port in enumerate(tuple(dout_ports)):
                    column_index[sar_nodes[port]] = 11 + idx
                columns = {name: [] for name in self.recorded_signals}
                for name in columns:
                    idx = column_index.get(name)
                    if idx is None:
                        raise RuntimeError(f"Rust SAR trace has no column for {name}")
                    columns[name] = [
                        float(trace_values[row_idx * signal_count + idx])
                        for row_idx in range(len(times))
                    ]
                self._perf_stats["rust_full_model_sar_loop_enabled"] = 1
                self._perf_stats["rust_full_model_sar_loop_clock_events"] = clock_events
                self._perf_stats["rust_full_model_sar_loop_rust_enabled"] = 1
                self._perf_stats["rust_full_model_sar_loop_rust_points"] = len(times)
                return self._record_trace_result(
                    times,
                    columns,
                    enabled_kind="sar_loop",
                )
            except Exception:
                self._perf_stats["rust_full_model_sar_loop_rust_fallbacks"] += 1

        dout_bits = [0 for _ in range(width)]
        trial_bits = [0 for _ in range(width)]
        bit_idx = 0
        busy = 0
        vsampled = 0.0
        trial_vdac_state = 0.0
        bit_index_v = 0.0
        trial_code_v = 0.0
        cmp_decision_v = 0.0
        conv_done_v = 0.0
        total_sum = float((1 << width) - 1)
        weights = [float(1 << (width - 1 - idx)) for idx in range(width)]
        transitions = {
            "vin_sh": TransitionState(),
            "vout": TransitionState(),
            "bit_index": TransitionState(),
            "trial_code": TransitionState(),
            "trial_vdac": TransitionState(),
            "cmp_decision": TransitionState(),
            "conv_done": TransitionState(),
            "vin_sample": TransitionState(),
        }
        dout_transitions = [TransitionState() for _ in range(width)]

        def current_code(bits: List[int]) -> int:
            return int(sum(bit * int(weight) for bit, weight in zip(bits, weights)))

        def drive_outputs(event_t: float) -> None:
            code = current_code(dout_bits)
            transitions["vout"].evaluate(event_t)
            transitions["vout"].set_target(event_t, code / total_sum * vdd, 0.0, default_tr, default_tr)
            scalar_targets = {
                "bit_index": bit_index_v,
                "trial_code": trial_code_v,
                "trial_vdac": trial_vdac_state,
                "cmp_decision": cmp_decision_v,
                "conv_done": conv_done_v,
                "vin_sample": vsampled,
            }
            for key, target in scalar_targets.items():
                transitions[key].evaluate(event_t)
                transitions[key].set_target(event_t, float(target), 0.0, default_tr, default_tr)
            for idx, bit in enumerate(dout_bits):
                dout_transitions[idx].evaluate(event_t)
                dout_transitions[idx].set_target(event_t, vdd if bit else 0.0, 0.0, default_tr, default_tr)

        prev_clk_expr = self._wave_value(source_by_node, sar_nodes[clks_port], 0.0) - vth
        clock_events = 0
        columns = {name: [] for name in self.recorded_signals}
        drive_outputs(0.0)

        for raw_t in times:
            t = float(raw_t)
            vin_val = self._wave_value(source_by_node, sh_nodes[sh_vin_port], t)
            clk_val = self._wave_value(source_by_node, sar_nodes[clks_port], t)
            rst_val = self._wave_value(source_by_node, sar_nodes[rst_port], t)
            clk_expr = clk_val - vth
            if self._crossed(prev_clk_expr, clk_expr, +1):
                if rst_val > vth:
                    transitions["vin_sh"].evaluate(t)
                    transitions["vin_sh"].set_target(t, vin_val, 0.0, sh_tr, sh_tr)
                if rst_val < vth:
                    dout_bits = [0 for _ in range(width)]
                    trial_bits = [0 for _ in range(width)]
                    bit_idx = 0
                    busy = 0
                    trial_vdac_state = 0.0
                    bit_index_v = 0.0
                    trial_code_v = 0.0
                    cmp_decision_v = 0.0
                    conv_done_v = 0.0
                elif busy == 1 and bit_idx > 0:
                    bit_pos = width - bit_idx
                    if cmp_decision_v > vth and 0 <= bit_pos < width:
                        dout_bits[bit_pos] = 1
                    bit_idx -= 1
                    if bit_idx > 0:
                        trial_bits = list(dout_bits)
                        next_pos = width - bit_idx
                        if 0 <= next_pos < width:
                            trial_bits[next_pos] = 1
                        trial_code_int = current_code(trial_bits)
                        trial_vdac_state = trial_code_int / total_sum * vdd
                        cmp_decision_v = vdd if vsampled >= trial_vdac_state else 0.0
                        bit_index_v = bit_idx / float(width) * vdd
                        trial_code_v = trial_code_int / total_sum * vdd
                        conv_done_v = 0.0
                    else:
                        final_code = current_code(dout_bits)
                        trial_vdac_state = final_code / total_sum * vdd
                        trial_code_v = trial_vdac_state
                        bit_index_v = 0.0
                        cmp_decision_v = vdd if vsampled >= trial_vdac_state else 0.0
                        conv_done_v = vdd
                        busy = 0
                drive_outputs(t)
                clock_events += 1
            elif self._crossed(prev_clk_expr, clk_expr, -1):
                if rst_val >= vth and busy == 0:
                    vsampled = transitions["vin_sh"].evaluate(t)
                    dout_bits = [0 for _ in range(width)]
                    trial_bits = [0 for _ in range(width)]
                    trial_bits[0] = 1
                    bit_idx = width
                    busy = 1
                    conv_done_v = 0.0
                    trial_code_int = current_code(trial_bits)
                    trial_vdac_state = trial_code_int / total_sum * vdd
                    cmp_decision_v = vdd if vsampled >= trial_vdac_state else 0.0
                    bit_index_v = bit_idx / float(width) * vdd
                    trial_code_v = trial_code_int / total_sum * vdd
                    drive_outputs(t)
                clock_events += 1
            vin_sh_val = transitions["vin_sh"].evaluate(t)
            vout_val = transitions["vout"].evaluate(t)
            values_by_node = {
                sh_nodes[sh_vin_port]: vin_val,
                sar_nodes[vin_port]: vin_sh_val,
                sar_nodes[clks_port]: clk_val,
                sar_nodes[rst_port]: rst_val,
                dac_nodes[dac_vout_port]: vout_val,
                sar_nodes[bit_index_port]: transitions["bit_index"].evaluate(t),
                sar_nodes[trial_code_port]: transitions["trial_code"].evaluate(t),
                sar_nodes[trial_vdac_port]: transitions["trial_vdac"].evaluate(t),
                sar_nodes[cmp_decision_port]: transitions["cmp_decision"].evaluate(t),
                sar_nodes[conv_done_port]: transitions["conv_done"].evaluate(t),
                sar_nodes[vin_sample_port]: transitions["vin_sample"].evaluate(t),
            }
            for port, trans in zip(tuple(dout_ports), dout_transitions):
                values_by_node[sar_nodes[port]] = trans.evaluate(t)
            for name in columns:
                columns[name].append(float(values_by_node.get(name, 0.0)))
            prev_clk_expr = clk_expr

        self._perf_stats["rust_full_model_sar_loop_clock_events"] = clock_events
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="sar_loop",
        )

    def _try_cppll_reacquire_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        ref_model = None
        pll_model = None
        ref_candidate = None
        pll_candidate = None
        for model in self.models:
            candidate = self._model_candidate(model, "ref_step_clock_v1")
            if candidate is not None:
                ref_model = model
                ref_candidate = candidate
            candidate = self._model_candidate(model, "cppll_timer_v1")
            if candidate is not None:
                pll_model = model
                pll_candidate = candidate
        if ref_model is None or pll_model is None or ref_candidate is None or pll_candidate is None:
            return None
        try:
            (
                _ref_kind,
                ref_vdd_port,
                ref_vss_port,
                ref_clk_port,
                period_pre_param,
                period_post_param,
                t_switch_param,
                ref_tedge_param,
            ) = ref_candidate
            (
                _pll_kind,
                pll_vdd_port,
                pll_vss_port,
                pll_ref_port,
                fb_port,
                dco_port,
                vctrl_port,
                lock_port,
                div_ratio_param,
                f_center_param,
                kvco_param,
                f_min_param,
                f_max_param,
                kp_param,
                ki_param,
                integ_min_param,
                integ_max_param,
                vctrl_init_param,
                pll_tedge_param,
                lock_tol_param,
                lock_count_target_param,
            ) = pll_candidate
        except (TypeError, ValueError):
            return None

        ref_nodes = {
            port: self._external_node(ref_model, port)
            for port in (ref_vdd_port, ref_vss_port, ref_clk_port)
        }
        pll_nodes = {
            port: self._external_node(pll_model, port)
            for port in (
                pll_vdd_port, pll_vss_port, pll_ref_port,
                fb_port, dco_port, vctrl_port, lock_port,
            )
        }
        if (
            ref_nodes[ref_vdd_port] != pll_nodes[pll_vdd_port]
            or ref_nodes[ref_vss_port] != pll_nodes[pll_vss_port]
            or ref_nodes[ref_clk_port] != pll_nodes[pll_ref_port]
        ):
            return None
        source_by_node = {src.node: src for src in self.sources}
        vdd_node = pll_nodes[pll_vdd_port]
        vss_node = pll_nodes[pll_vss_port]
        if vdd_node not in source_by_node or vss_node not in source_by_node:
            return None
        generated_nodes = {
            ref_nodes[ref_clk_port],
            pll_nodes[fb_port],
            pll_nodes[dco_port],
            pll_nodes[vctrl_port],
            pll_nodes[lock_port],
            vdd_node,
            vss_node,
        }
        if any(name not in generated_nodes for name in self.recorded_signals):
            return None

        vdd_meta = self._waveform_metadata(source_by_node[vdd_node].waveform)
        vss_meta = self._waveform_metadata(source_by_node[vss_node].waveform)
        if not vdd_meta or not vss_meta or vdd_meta.get("kind") != "dc" or vss_meta.get("kind") != "dc":
            return None
        try:
            vh = float(vdd_meta["voltage"])
            vl = float(vss_meta["voltage"])
            period_pre = float(ref_model.params[period_pre_param])
            period_post = float(ref_model.params[period_post_param])
            t_switch = float(ref_model.params[t_switch_param])
            ref_tedge = float(ref_model.params[ref_tedge_param])
            div_ratio = int(pll_model.params[div_ratio_param])
            f_center = float(pll_model.params[f_center_param])
            kvco = float(pll_model.params[kvco_param])
            f_min = float(pll_model.params[f_min_param])
            f_max = float(pll_model.params[f_max_param])
            kp = float(pll_model.params[kp_param])
            ki = float(pll_model.params[ki_param])
            integ_min = float(pll_model.params[integ_min_param])
            integ_max = float(pll_model.params[integ_max_param])
            vctrl_init = float(pll_model.params[vctrl_init_param])
            pll_tedge = float(pll_model.params[pll_tedge_param])
            lock_tol = float(pll_model.params[lock_tol_param])
            lock_count_target = int(pll_model.params[lock_count_target_param])
        except Exception:
            return None
        if (
            period_pre <= 0.0
            or period_post <= 0.0
            or ref_tedge <= 0.0
            or pll_tedge <= 0.0
            or div_ratio <= 0
            or f_center <= 0.0
            or kvco <= 0.0
            or f_min <= 0.0
            or f_max <= 0.0
            or lock_count_target <= 0
        ):
            return None

        vcm = 0.5 * (vh + vl)

        def clamp(value: float, lo: float, hi: float) -> float:
            return min(hi, max(lo, value))

        def dco_freq_from_vctrl(value: float) -> float:
            return clamp(f_center + kvco * (value - vcm), f_min, f_max)

        import heapq

        seq = 0
        event_heap: List[Tuple[float, int, int, str]] = []

        def push_event(event_time: float, priority: int, kind: str) -> None:
            nonlocal seq
            if event_time > tstop + max(ref_tedge, pll_tedge) + 1.0e-18:
                return
            heapq.heappush(event_heap, (float(event_time), int(priority), seq, kind))
            seq += 1

        ref_transition_events: List[Tuple[float, float]] = [(0.0, vl)]
        dco_transition_events: List[Tuple[float, float]] = [(0.0, vl)]
        fb_transition_events: List[Tuple[float, float]] = [(0.0, vl)]
        lock_transition_events: List[Tuple[float, float]] = [(0.0, vl)]
        vctrl_events: List[Tuple[float, float]] = []
        raw_times = self._whole_segment_uniform_times(
            tstop=tstop,
            record_step=record_step,
            tstep=tstep,
        )

        ref_state = 0
        dco_state = 0
        fb_state = 0
        div_cnt = 0
        lock_state = 0
        lock_streak = 0
        t_ref_prev = -1.0
        t_ref_last = -1.0
        t_fb_last = -1.0
        ref_period = period_pre
        phase_err = 0.0
        integ = 0.0
        vctrl = clamp(vctrl_init, vl, vh)
        vctrl_events.append((0.0, vctrl))
        dco_freq = dco_freq_from_vctrl(vctrl)

        push_event(0.5 * period_pre, 0, "ref_toggle")
        push_event(0.5 / dco_freq, 1, "dco_toggle")
        ref_toggles = 0
        dco_toggles = 0
        fb_rise_crosses = 0
        ref_rise_crosses = 0

        def add_transition_times(event_time: float, edge: float) -> None:
            raw_times.append(event_time)
            for frac in (0.25, 0.5, 0.75, 1.0):
                raw_times.append(event_time + frac * edge)

        while event_heap:
            event_t, _priority, _seq, kind = heapq.heappop(event_heap)
            if event_t > tstop + max(ref_tedge, pll_tedge) + 1.0e-18:
                break
            raw_times.append(event_t)
            if kind == "ref_toggle":
                ref_state = 1 - ref_state
                ref_transition_events.append((event_t, vh if ref_state else vl))
                add_transition_times(event_t, ref_tedge)
                if ref_state == 1:
                    push_event(event_t + 0.5 * ref_tedge, 2, "ref_rise_cross")
                half_period = 0.5 * (period_post if event_t >= t_switch else period_pre)
                push_event(event_t + half_period, 0, "ref_toggle")
                ref_toggles += 1
            elif kind == "dco_toggle":
                dco_state = 1 - dco_state
                dco_transition_events.append((event_t, vh if dco_state else vl))
                add_transition_times(event_t, pll_tedge)
                if dco_state == 1:
                    div_cnt += 1
                    if div_cnt >= div_ratio:
                        div_cnt = 0
                        fb_state = 1 - fb_state
                        fb_transition_events.append((event_t, vh if fb_state else vl))
                        add_transition_times(event_t, pll_tedge)
                        if fb_state == 1:
                            push_event(event_t + 0.5 * pll_tedge, 2, "fb_rise_cross")
                dco_freq = dco_freq_from_vctrl(vctrl)
                push_event(event_t + 0.5 / dco_freq, 1, "dco_toggle")
                dco_toggles += 1
            elif kind == "fb_rise_cross":
                t_fb_last = event_t
                fb_rise_crosses += 1
            elif kind == "ref_rise_cross":
                t_ref_prev = t_ref_last
                t_ref_last = event_t
                if t_fb_last >= 0.0:
                    phase_err = t_ref_last - t_fb_last
                    if t_ref_prev >= 0.0:
                        ref_period = t_ref_last - t_ref_prev
                        if ref_period > 1.0e-15:
                            while phase_err > 0.5 * ref_period:
                                phase_err -= ref_period
                            while phase_err < -0.5 * ref_period:
                                phase_err += ref_period
                    integ = clamp(integ + ki * phase_err, integ_min, integ_max)
                    vctrl = clamp(vcm + kp * phase_err + integ, vl, vh)
                    vctrl_events.append((event_t, vctrl))
                    if abs(phase_err) <= lock_tol:
                        lock_streak += 1
                    else:
                        lock_streak = 0
                    new_lock_state = 1 if lock_streak >= lock_count_target else 0
                    if new_lock_state != lock_state:
                        lock_state = new_lock_state
                        lock_transition_events.append((event_t, vh if lock_state else vl))
                        add_transition_times(event_t, pll_tedge)
                ref_rise_crosses += 1

        times = self._dedupe_times(raw_times, tstop)
        transitions = {
            ref_nodes[ref_clk_port]: TransitionState(),
            pll_nodes[dco_port]: TransitionState(),
            pll_nodes[fb_port]: TransitionState(),
            pll_nodes[lock_port]: TransitionState(),
        }
        event_targets = sorted(
            [
                (t, ref_nodes[ref_clk_port], value, ref_tedge)
                for t, value in ref_transition_events
            ]
            + [
                (t, pll_nodes[dco_port], value, pll_tedge)
                for t, value in dco_transition_events
            ]
            + [
                (t, pll_nodes[fb_port], value, pll_tedge)
                for t, value in fb_transition_events
            ]
            + [
                (t, pll_nodes[lock_port], value, pll_tedge)
                for t, value in lock_transition_events
            ],
            key=lambda item: item[0],
        )
        vctrl_events.sort(key=lambda item: item[0])
        rust_cppll_trace_enabled = (
            os.environ.get("EVAS_RUST_CPPLL_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off", "disabled"}
        )
        self._perf_stats["rust_full_model_cppll_reacquire_rust_requested"] = int(
            bool(rust_backend is not None and rust_cppll_trace_enabled)
        )
        if rust_backend is not None and rust_cppll_trace_enabled:
            try:
                def event_arrays(items: List[Tuple[float, float]]) -> Tuple[array, array]:
                    return (
                        array("d", (float(t) for t, _value in items)),
                        array("d", (float(value) for _t, value in items)),
                    )

                flat_values = rust_backend.cppll_reacquire_trace(
                    times,
                    ref_events=event_arrays(ref_transition_events),
                    dco_events=event_arrays(dco_transition_events),
                    fb_events=event_arrays(fb_transition_events),
                    lock_events=event_arrays(lock_transition_events),
                    vctrl_events=event_arrays(vctrl_events),
                    vh=vh,
                    vl=vl,
                    ref_tedge=ref_tedge,
                    pll_tedge=pll_tedge,
                )
                matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(
                    len(times),
                    7,
                )
                column_by_node = {
                    vdd_node: 0,
                    vss_node: 1,
                    ref_nodes[ref_clk_port]: 2,
                    pll_nodes[dco_port]: 3,
                    pll_nodes[fb_port]: 4,
                    pll_nodes[vctrl_port]: 5,
                    pll_nodes[lock_port]: 6,
                }
                columns = {
                    name: np.array(matrix[:, column_by_node[name]], copy=True)
                    for name in self.recorded_signals
                }
                self._perf_stats["rust_full_model_cppll_reacquire_ref_toggles"] = ref_toggles
                self._perf_stats["rust_full_model_cppll_reacquire_dco_toggles"] = dco_toggles
                self._perf_stats["rust_full_model_cppll_reacquire_ref_crosses"] = ref_rise_crosses
                self._perf_stats["rust_full_model_cppll_reacquire_fb_crosses"] = fb_rise_crosses
                self._perf_stats["rust_full_model_cppll_reacquire_rust_enabled"] = 1
                self._perf_stats["rust_full_model_cppll_reacquire_rust_points"] = len(times)
                return self._record_trace_result(
                    times,
                    columns,
                    enabled_kind="cppll_reacquire",
                )
            except (KeyError, RustBackendError, ValueError):
                self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
                self._perf_stats["rust_full_model_cppll_reacquire_rust_fallbacks"] += 1

        target_idx = 0
        vctrl_idx = 0
        vctrl_current = vctrl_events[0][1] if vctrl_events else vctrl
        columns = {name: [] for name in self.recorded_signals}

        for raw_t in times:
            t = float(raw_t)
            while target_idx < len(event_targets) and event_targets[target_idx][0] <= t + 1.0e-18:
                event_t, node, value, edge = event_targets[target_idx]
                trans = transitions[node]
                trans.evaluate(event_t)
                trans.set_target(event_t, value, 0.0, edge, edge)
                target_idx += 1
            while vctrl_idx < len(vctrl_events) and vctrl_events[vctrl_idx][0] <= t + 1.0e-18:
                vctrl_current = vctrl_events[vctrl_idx][1]
                vctrl_idx += 1
            values_by_node = {
                vdd_node: vh,
                vss_node: vl,
                ref_nodes[ref_clk_port]: transitions[ref_nodes[ref_clk_port]].evaluate(t),
                pll_nodes[dco_port]: transitions[pll_nodes[dco_port]].evaluate(t),
                pll_nodes[fb_port]: transitions[pll_nodes[fb_port]].evaluate(t),
                pll_nodes[lock_port]: transitions[pll_nodes[lock_port]].evaluate(t),
                pll_nodes[vctrl_port]: vctrl_current,
            }
            for name in columns:
                columns[name].append(float(values_by_node.get(name, 0.0)))

        self._perf_stats["rust_full_model_cppll_reacquire_ref_toggles"] = ref_toggles
        self._perf_stats["rust_full_model_cppll_reacquire_dco_toggles"] = dco_toggles
        self._perf_stats["rust_full_model_cppll_reacquire_ref_crosses"] = ref_rise_crosses
        self._perf_stats["rust_full_model_cppll_reacquire_fb_crosses"] = fb_rise_crosses
        return self._record_trace_result(
            times,
            columns,
            enabled_kind="cppll_reacquire",
        )

    def _try_rust_prbs7_full_model_fastpath(
        self,
        *,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        if rust_backend is None or not self.models or len(self.models) != 1:
            return None
        model = self.models[0]
        model_cls = getattr(model, "__class__", type(model))
        if getattr(model, "_child_models", []):
            return None
        for raw_candidate in getattr(model_cls, "_whole_segment_candidates", ()) or ():
            if not raw_candidate or raw_candidate[0] != "cross_scalar_lfsr_transition_bus_v1":
                continue
            result = self._try_rust_lfsr_transition_full_model_fastpath(
                model=model,
                candidate=raw_candidate,
                rust_backend=rust_backend,
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
            )
            if result is not None:
                return result
        return None

    def _try_rust_lfsr_transition_full_model_fastpath(
        self,
        *,
        model,
        candidate: tuple,
        rust_backend,
        tstop: float,
        record_step: Optional[float],
        tstep: float,
    ) -> Optional[SimResult]:
        try:
            (
                _kind,
                clock_port,
                reset_port,
                enable_port,
                threshold_param,
                seed_param,
                vdd_param,
                td_param,
                trf_param,
                state_names,
                taps,
                shift_sources,
                output_ports,
                output_bits,
                zero_guard_index,
            ) = candidate
        except (TypeError, ValueError):
            return None

        required_ports = (clock_port, reset_port, enable_port, *tuple(output_ports))
        node_map = getattr(model, "node_map", {}) or {}
        external = {port: node_map.get(port, port) for port in required_ports}
        canonical_nodes = tuple(external[port] for port in required_ports)
        if any(node is None for node in canonical_nodes):
            return None
        recorded = tuple(self.recorded_signals.keys())
        if any(name not in canonical_nodes for name in recorded):
            return None

        sources = {src.node: src for src in self.sources}
        clk_src = sources.get(external[clock_port])
        rst_src = sources.get(external[reset_port])
        en_src = sources.get(external[enable_port])
        if clk_src is None or rst_src is None or en_src is None:
            return None
        clk_meta = self._waveform_metadata(clk_src.waveform)
        rst_meta = self._waveform_metadata(rst_src.waveform)
        en_meta = self._waveform_metadata(en_src.waveform)
        if (
            not clk_meta
            or not rst_meta
            or not en_meta
            or clk_meta.get("kind") != "pulse"
            or rst_meta.get("kind") != "pulse"
            or en_meta.get("kind") != "dc"
        ):
            return None

        try:
            vdd = float(model.params.get(str(vdd_param), 0.9))
            vth = float(model.params.get(str(threshold_param), 0.45))
            trf = float(model.params.get(str(trf_param), 1.0e-12))
            td = float(model.params.get(str(td_param), 0.0))
            seed = int(model.params.get(str(seed_param), 127))
            en_voltage = float(en_meta.get("voltage", 0.0))
            width = len(tuple(state_names))
            tap_values = tuple(int(tap) for tap in taps)
            shift_values = tuple(int(source) for source in shift_sources)
            output_bit_values = tuple(int(bit) for bit in output_bits)
        except Exception:
            return None
        if (
            width <= 0
            or len(shift_values) != width
            or not tap_values
            or len(tuple(output_ports)) != len(output_bit_values)
        ):
            return None

        sample_step = float(record_step or tstep)
        if sample_step <= 0.0:
            return None
        raw_times: List[float] = []
        uniform_count = int(math.floor(tstop / sample_step + 1.0e-9)) + 1
        for idx in range(max(1, uniform_count)):
            raw_times.append(min(tstop, idx * sample_step))
        if not raw_times or raw_times[-1] < tstop - 1.0e-18:
            raw_times.append(tstop)

        transition_offsets = tuple(
            offset
            for offset in (
                0.25 * trf,
                0.5 * trf,
                0.75 * trf,
                trf,
            )
            if offset > 0.0
        )
        cross_count = self._add_pulse_schedule_times(
            raw_times,
            clk_meta,
            tstop,
            vth=vth,
            transition_offsets=transition_offsets,
        )
        self._add_pulse_schedule_times(raw_times, rst_meta, tstop)
        times = self._dedupe_times(raw_times, tstop)
        if not times:
            return None

        try:
            flat_values, event_count = rust_backend.lfsr_transition_trace(
                times,
                clk=clk_meta,
                rst_n=rst_meta,
                en_voltage=en_voltage,
                vdd=vdd,
                vth=vth,
                trf=trf,
                td=td,
                seed=seed,
                width=width,
                taps=tap_values,
                shift_sources=shift_values,
                output_bits=output_bit_values,
                zero_guard_index=int(zero_guard_index),
            )
        except RustBackendError:
            self._perf_stats["rust_full_model_fastpath_fallbacks_total"] += 1
            return None

        point_count = len(times)
        signal_count = 3 + len(output_bit_values)
        matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(
            point_count,
            signal_count,
        )
        column_by_node = {
            external[clock_port]: 0,
            external[reset_port]: 1,
            external[enable_port]: 2,
        }
        for idx, port in enumerate(tuple(output_ports)):
            column_by_node[external[port]] = 3 + idx
        signals = {
            name: np.array(matrix[:, column_by_node[name]], copy=True)
            for name in recorded
        }
        time_arr = np.frombuffer(times, dtype=np.float64).copy()
        step_sizes = np.empty(point_count, dtype=np.float64)
        if point_count:
            step_sizes[0] = 0.0
        if point_count > 1:
            step_sizes[1:] = np.diff(time_arr)

        self.time_points = time_arr.tolist()
        for name, values in signals.items():
            self.recorded_signals[name] = values.tolist()
            self.node_voltages[name] = float(values[-1]) if len(values) else 0.0
        self._step_sizes = step_sizes.tolist()
        self._perf_stats["rust_full_model_fastpath_enabled"] = 1
        self._perf_stats["rust_full_model_prbs7_points"] = point_count
        self._perf_stats["rust_full_model_prbs7_cross_events"] = int(event_count)
        self._perf_stats["rust_full_model_prbs7_scheduled_crosses"] = int(cross_count)
        self._perf_stats["rust_full_model_lfsr_transition_points"] = point_count
        self._perf_stats["rust_full_model_lfsr_transition_cross_events"] = int(event_count)
        self._perf_stats["rust_full_model_lfsr_transition_outputs"] = len(output_bit_values)
        self._perf_stats["cross_event_steps"] = int(event_count)
        self._perf_stats["cross_refine_triggers"] = int(event_count)
        self._perf_stats["steps_total"] = max(0, point_count - 1)
        return SimResult(time=time_arr, signals=signals, step_sizes=step_sizes)

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
            profile_model_eval: bool = False,
            profile_model_io: bool = False,
            indexed_snapshot_profile: bool = False,
            indexed_arrays: bool = False,
            indexed_state_storage: bool = False,
            static_branch_fastpath: bool = False,
            static_lifecycle_fastpath: bool = True,
            transition_unchanged_fastpath: bool = False,
            rust_static_eval: bool = False,
            rust_static_fast_sync: bool = False,
            rust_transition_shadow: bool = False,
            rust_transition_production: bool = False,
            rust_event_interpolation: bool = False,
            rust_timer_event: bool = False,
            rust_event_due_shadow: bool = False,
            rust_cross_above_production: bool = False,
            generic_executor: bool = False,
            rust_event_write_shadow: bool = False,
            rust_event_write_production: bool = False,
            rust_body_ir: bool = False,
            rust_event_transition_shadow: bool = False,
            rust_event_transition_production: bool = False,
            rust_full_model_fastpath: bool = False,
            rust_full_model_required: bool = False,
            event_trace_audit: bool = False,
            cross_acceptance_slack_factor: float = 0.0,
            rust_required: bool = False) -> SimResult:
        """Run transient simulation with adaptive step control near cross events."""
        if tstep is None:
            tstep = tstop / 10000
        if max_step is None:
            max_step = tstep
        min_step_defaulted = min_step is None
        if min_step_defaulted:
            min_step = tstep / 4096.0
        adaptive_step_floor = min_step
        if min_step_defaulted:
            adaptive_step_floor = max(min_step, tstep / 64.0)
        indexed_arrays_requested = bool(indexed_arrays)
        indexed_state_storage_requested = bool(indexed_state_storage)
        if not rust_static_eval:
            rust_static_fast_sync = False
        if rust_event_transition_production:
            rust_event_transition_shadow = False

        def _model_tree_has_event_write_ir() -> bool:
            def _visit(model) -> bool:
                if (
                    getattr(model.__class__, "_event_lfsr_shift_ir_ops", ()) or ()
                    or getattr(model.__class__, "_event_static_linear_ir_ops", ()) or ()
                ):
                    return True
                return any(
                    _visit(child)
                    for child in getattr(model, "_child_models", []) or []
                )

            return any(_visit(model) for model in self.models)

        def _model_tree_has_event_linear_ir() -> bool:
            def _visit(model) -> bool:
                if getattr(model.__class__, "_event_static_linear_ir_ops", ()) or ():
                    return True
                return any(
                    _visit(child)
                    for child in getattr(model, "_child_models", []) or []
                )

            return any(_visit(model) for model in self.models)

        rust_event_write_has_candidate = (
            (rust_event_write_shadow or rust_event_write_production)
            and _model_tree_has_event_write_ir()
        )
        rust_event_write_has_linear_candidate = (
            (rust_event_write_shadow or rust_event_write_production)
            and _model_tree_has_event_linear_ir()
        )
        if not rust_event_write_has_candidate:
            rust_event_write_shadow = False
            rust_event_write_production = False
        if rust_static_eval or rust_transition_shadow:
            indexed_arrays = True
            indexed_state_storage = True
        if rust_event_transition_production:
            indexed_state_storage = True
        if rust_event_write_shadow or rust_event_write_production:
            indexed_state_storage = True
        if rust_event_write_has_linear_candidate:
            indexed_arrays = True
        if rust_timer_event and rust_event_write_production:
            indexed_arrays = True

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
            "transition_target_breakpoint_scan_calls": 0,
            "transition_target_breakpoint_clamps": 0,
            "bound_step_clamps": 0,
            "min_step_clamps": 0,
            "adaptive_step_floor": adaptive_step_floor,
            "adaptive_step_floor_clamps": 0,
            "adaptive_step_floor_defaulted": int(min_step_defaulted),
            "adaptive_step_floor_model_bypasses": 0,
            "adaptive_step_floor_sensitive_models": 0,
            "adaptive_step_floor_source_bypasses": 0,
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
            "indexed_array_record_id_reads": 0,
            "indexed_array_record_reads": 0,
            "indexed_array_snapshots": 0,
            "indexed_array_source_updates": 0,
            "indexed_array_syncs": 0,
            "indexed_array_values_checked": 0,
            "indexed_array_dirty_validation_enabled": 0,
            "indexed_array_dirty_syncs": 0,
            "indexed_array_dirty_nodes_checked": 0,
            "indexed_array_prev_snapshot_dirty_skips": 0,
            "indexed_model_io_mapped_ports": 0,
            "indexed_model_io_models": 0,
            "indexed_model_io_outputs": 0,
            "indexed_model_io_refreshes": 0,
            "indexed_output_write_through_nodes": 0,
            "indexed_output_write_throughs": 0,
            "indexed_post_model_sync_repairs": 0,
            "indexed_state_storage_enabled": int(bool(indexed_state_storage)),
            "indexed_state_storage_models": 0,
            "indexed_state_storage_scalar_slots": 0,
            "indexed_state_storage_integer_slots": 0,
            "indexed_state_storage_array_slots": 0,
            "indexed_state_scalar_reads_total": 0,
            "indexed_state_scalar_writes_total": 0,
            "indexed_state_array_reads_total": 0,
            "indexed_state_array_writes_total": 0,
            "indexed_state_array_oob_writes_total": 0,
            "indexed_voltage_probe_event_skips": 0,
            "indexed_voltage_probe_max_abs_diff": 0.0,
            "indexed_voltage_probe_mismatches": 0,
            "indexed_voltage_probe_missing_nodes": 0,
            "indexed_voltage_probes": 0,
            "indexed_voltage_read_fallbacks": 0,
            "indexed_voltage_read_nodes": 0,
            "indexed_voltage_reads": 0,
            "static_branch_fastpath_codegen_models": 0,
            "static_branch_direct_array_models": 0,
            "static_branch_direct_array_read_nodes": 0,
            "static_branch_direct_array_write_nodes": 0,
            "static_branch_fastpath_fallbacks_total": 0,
            "static_branch_fastpath_static_read_nodes": 0,
            "static_branch_fastpath_static_write_nodes": 0,
            "rust_static_eval_requested": int(bool(rust_static_eval)),
            "rust_static_eval_available": 0,
            "rust_static_eval_candidate_models": 0,
            "rust_static_eval_models": 0,
            "rust_static_eval_ops": 0,
            "rust_static_eval_segments": 0,
            "rust_static_eval_max_segment_models": 0,
            "rust_static_eval_calls": 0,
            "rust_static_eval_output_syncs": 0,
            "rust_static_eval_node_voltage_syncs": 0,
            "rust_static_eval_deferred_output_syncs": 0,
            "rust_static_eval_lifecycle_model_skips": 0,
            "rust_static_eval_runtime_param_ops": 0,
            "rust_static_eval_coeff_eval_fallbacks": 0,
            "rust_static_eval_fallback_models": 0,
            "rust_static_eval_mixed_small_fallbacks": 0,
            "rust_static_eval_no_candidate_models": 0,
            "rust_static_eval_gated_models": 0,
            "rust_static_eval_gated_ops": 0,
            "rust_static_eval_gated_segments": 0,
            "rust_static_eval_terms": 0,
            "rust_static_eval_state_read_terms": 0,
            "rust_static_eval_state_write_ops": 0,
            "rust_static_eval_integer_state_write_ops": 0,
            "rust_static_eval_errors": 0,
            "rust_static_eval_no_segment_fallbacks": 0,
            "rust_static_fast_sync_requested": int(bool(rust_static_fast_sync)),
            "rust_static_fast_sync_enabled": 0,
            "rust_static_fast_sync_node_voltage_sync_skips": 0,
            "rust_static_fast_sync_validation_skips": 0,
            "rust_transition_shadow_requested": int(bool(rust_transition_shadow)),
            "rust_transition_shadow_available": 0,
            "rust_transition_shadow_candidate_models": 0,
            "rust_transition_shadow_models": 0,
            "rust_transition_shadow_static_ops": 0,
            "rust_transition_shadow_target_ops": 0,
            "rust_transition_shadow_segments": 0,
            "rust_transition_shadow_calls": 0,
            "rust_transition_shadow_matches": 0,
            "rust_transition_shadow_mismatches": 0,
            "rust_transition_shadow_skips": 0,
            "rust_transition_shadow_errors": 0,
            "rust_transition_shadow_max_abs_diff": 0.0,
            "rust_transition_shadow_state_matches": 0,
            "rust_transition_shadow_state_mismatches": 0,
            "rust_transition_shadow_state_max_abs_diff": 0.0,
            "rust_transition_production_requested": int(bool(rust_transition_production)),
            "rust_transition_production_available": 0,
            "rust_transition_production_enabled": 0,
            "rust_transition_state_production_calls_total": 0,
            "rust_transition_state_production_outputs_total": 0,
            "rust_transition_state_production_fallbacks_total": 0,
            "rust_transition_state_buffer_reuse_calls_total": 0,
            "rust_transition_state_buffer_alloc_grand_total": 0,
            "rust_transition_batch_flushes_total": 0,
            "rust_transition_batch_slot_total_total": 0,
            "rust_transition_batch_fallbacks_total": 0,
            "rust_transition_batch_max_slots_total": 0,
            "rust_transition_lazy_enqueues_total": 0,
            "rust_cross_above_production_requested": int(bool(rust_cross_above_production)),
            "rust_cross_above_production_available": 0,
            "rust_cross_above_production_enabled": 0,
            # Audit 091c: generic executor dispatch gate inspection
            "generic_executor_models_with_candidate": 0,
            "generic_executor_dispatchable_runs": 0,
            "generic_executor_blocked_runs": 0,
            # Audit 091d: generic executor body counters
            "generic_executor_runs": 0,
            "generic_executor_runtime_fallbacks": 0,
            "rust_cross_production_calls_total": 0,
            "rust_cross_production_fires_total": 0,
            "rust_cross_production_fallbacks_total": 0,
            "rust_above_production_calls_total": 0,
            "rust_above_production_fires_total": 0,
            "rust_above_production_fallbacks_total": 0,
            "rust_event_interpolation_requested": int(bool(rust_event_interpolation)),
            "rust_event_interpolation_available": 0,
            "rust_event_interpolation_enabled": 0,
            "rust_event_interpolation_batches_total": 0,
            "rust_event_interpolation_nodes_total": 0,
            "rust_event_interpolation_cache_hits_total": 0,
            "rust_event_interpolation_fallbacks_total": 0,
            "rust_event_due_shadow_requested": int(bool(rust_event_due_shadow)),
            "rust_event_due_shadow_available": 0,
            "rust_event_due_shadow_enabled": 0,
            "rust_event_due_shadow_cross_checks_total": 0,
            "rust_event_due_shadow_above_checks_total": 0,
            "rust_event_due_shadow_timer_periodic_checks_total": 0,
            "rust_event_due_shadow_timer_absolute_checks_total": 0,
            "rust_event_due_shadow_matches_total": 0,
            "rust_event_due_shadow_mismatches_total": 0,
            "rust_event_due_shadow_errors_total": 0,
            "rust_event_due_shadow_max_time_diff_total": 0.0,
            "rust_timer_event_requested": int(bool(rust_timer_event)),
            "rust_timer_event_available": 0,
            "rust_timer_event_enabled": 0,
            "rust_timer_event_production_periodic_calls_total": 0,
            "rust_timer_event_production_absolute_calls_total": 0,
            "rust_timer_event_production_fires_total": 0,
            "rust_timer_event_production_skips_total": 0,
            "rust_timer_event_production_expirations_total": 0,
            "rust_timer_event_production_fallbacks_total": 0,
            "rust_event_write_shadow_requested": int(bool(rust_event_write_shadow)),
            "rust_event_write_shadow_available": 0,
            "rust_event_write_shadow_enabled": 0,
            "rust_event_write_shadow_checks_total": 0,
            "rust_event_write_shadow_matches_total": 0,
            "rust_event_write_shadow_mismatches_total": 0,
            "rust_event_write_shadow_errors_total": 0,
            "rust_event_write_production_requested": int(bool(rust_event_write_production)),
            "rust_event_write_production_available": 0,
            "rust_event_write_production_enabled": 0,
            "rust_event_write_production_calls_total": 0,
            "rust_event_write_production_executed_total": 0,
            "rust_event_write_production_fallbacks_total": 0,
            "rust_event_linear_write_batches_total": 0,
            "rust_event_linear_write_shadow_checks_total": 0,
            "rust_event_linear_write_shadow_matches_total": 0,
            "rust_event_linear_write_shadow_mismatches_total": 0,
            "rust_event_linear_write_shadow_errors_total": 0,
            "rust_event_linear_write_production_calls_total": 0,
            "rust_event_linear_write_production_executed_total": 0,
            "rust_event_linear_write_production_fallbacks_total": 0,
            "rust_body_ir_requested": int(bool(rust_body_ir)),
            "rust_body_ir_available": 0,
            "rust_body_ir_enabled": 0,
            "rust_body_ir_candidate_models": 0,
            "rust_body_ir_models": 0,
            "rust_body_ir_stmt_ops": 0,
            "rust_body_ir_expr_ops": 0,
            "rust_body_ir_production_batches_total": 0,
            "rust_body_ir_production_calls_total": 0,
            "rust_body_ir_production_executed_total": 0,
            "rust_body_ir_production_fallbacks_total": 0,
            "rust_body_ir_production_node_writes_total": 0,
            "rust_body_ir_production_state_writes_total": 0,
            "rust_event_transition_shadow_requested": int(
                bool(rust_event_transition_shadow)
            ),
            "rust_event_transition_shadow_available": 0,
            "rust_event_transition_shadow_enabled": 0,
            "rust_event_transition_shadow_candidate_models": 0,
            "rust_event_transition_shadow_models": 0,
            "rust_event_transition_shadow_calls_total": 0,
            "rust_event_transition_shadow_matches_total": 0,
            "rust_event_transition_shadow_mismatches_total": 0,
            "rust_event_transition_shadow_errors_total": 0,
            "rust_event_transition_shadow_value_checks_total": 0,
            "rust_event_transition_shadow_max_abs_diff": 0.0,
            "rust_event_transition_production_requested": int(
                bool(rust_event_transition_production)
            ),
            "rust_event_transition_production_available": 0,
            "rust_event_transition_production_enabled": 0,
            "rust_event_transition_production_candidate_models": 0,
            "rust_event_transition_production_models": 0,
            "rust_event_transition_production_calls_total": 0,
            "rust_event_transition_production_executed_total": 0,
            "rust_event_transition_production_fallbacks_total": 0,
            "rust_event_transition_production_state_writes_total": 0,
            "rust_event_transition_production_output_writes_total": 0,
            "rust_event_transition_production_fired_events_total": 0,
            "rust_event_transition_production_breakpoint_scans_total": 0,
            "rust_event_transition_production_breakpoint_clamps_total": 0,
            "rust_event_transition_plan_core_candidate_models": 0,
            "rust_event_transition_plan_core_event_statements": 0,
            "rust_event_transition_plan_core_due_triggers": 0,
            "rust_event_transition_plan_core_transitions": 0,
            "rust_event_transition_plan_core_output_writes": 0,
            "rust_event_transition_plan_ordered_v1_candidate_models": 0,
            "rust_event_transition_plan_ordered_v1_event_statements": 0,
            "rust_event_transition_plan_ordered_v1_due_triggers": 0,
            "rust_event_transition_plan_ordered_v1_transitions": 0,
            "rust_event_transition_plan_ordered_v1_output_writes": 0,
            "rust_event_transition_plan_side_effect_candidate_models": 0,
            "rust_event_transition_plan_side_effect_event_statements": 0,
            "rust_event_transition_plan_side_effect_due_triggers": 0,
            "rust_event_transition_plan_side_effect_transitions": 0,
            "rust_event_transition_plan_side_effect_output_writes": 0,
            "rust_full_model_fastpath_requested": int(bool(rust_full_model_fastpath)),
            "rust_full_model_required_requested": int(bool(rust_full_model_required)),
            "rust_full_model_fastpath_available": 0,
            "rust_full_model_fastpath_enabled": 0,
            "rust_full_model_required_failures": 0,
            "rust_full_model_fastpath_fallbacks_total": 0,
            "rust_sim_program_requested": 0,
            "rust_sim_program_available": 0,
            "rust_sim_program_enabled": 0,
            "rust_sim_program_source_record_enabled": 0,
            "rust_sim_program_event_transition_enabled": 0,
            "rust_sim_program_node_count": 0,
            "rust_sim_program_state_count": 0,
            "rust_sim_program_source_count": 0,
            "rust_sim_program_record_count": 0,
            "rust_sim_program_continuous_linear_ops": 0,
            "rust_sim_program_event_count": 0,
            "rust_sim_program_always_body_count": 0,
            "rust_sim_program_body_stmt_ops": 0,
            "rust_sim_program_body_expr_ops": 0,
            "rust_sim_program_transition_count": 0,
            "rust_sim_program_points": 0,
            "rust_sim_program_source_breakpoints": 0,
            "rust_sim_program_event_fires": 0,
            "rust_sim_program_transition_breakpoints": 0,
            "rust_sim_program_rejections": 0,
            "rust_sim_program_lower_elapsed_s": 0.0,
            "rust_sim_program_abi_build_elapsed_s": 0.0,
            "rust_sim_program_time_grid_elapsed_s": 0.0,
            "rust_sim_program_runtime_elapsed_s": 0.0,
            "rust_sim_program_runtime_attempts": 0,
            "rust_sim_program_final_capacity": 0,
            "rust_sim_program_record_replay_elapsed_s": 0.0,
            "rust_sim_program_state_sync_elapsed_s": 0.0,
            "rust_sim_program_fastpath_total_elapsed_s": 0.0,
            "rust_full_model_prbs7_points": 0,
            "rust_full_model_prbs7_cross_events": 0,
            "rust_full_model_prbs7_scheduled_crosses": 0,
            "rust_full_model_lfsr_transition_points": 0,
            "rust_full_model_lfsr_transition_cross_events": 0,
            "rust_full_model_lfsr_transition_outputs": 0,
            "rust_full_model_whole_segment_points": 0,
            "rust_full_model_timer_static_linear_enabled": 0,
            "rust_full_model_timer_static_linear_events": 0,
            "rust_full_model_timer_static_linear_rust_requested": 0,
            "rust_full_model_timer_static_linear_rust_enabled": 0,
            "rust_full_model_timer_static_linear_rust_fallbacks": 0,
            "rust_full_model_timer_static_linear_rust_points": 0,
            "rust_full_model_timer_static_linear_default_trace_enabled": 0,
            "rust_full_model_timer_static_linear_timer_count": 0,
            "rust_full_model_gain_timer_reduction_enabled": 0,
            "rust_full_model_gain_timer_reduction_samples": 0,
            "rust_full_model_gain_timer_reduction_rust_requested": 0,
            "rust_full_model_gain_timer_reduction_rust_enabled": 0,
            "rust_full_model_gain_timer_reduction_rust_fallbacks": 0,
            "rust_full_model_gain_timer_reduction_rust_points": 0,
            "rust_full_model_gain_measurement_flow_enabled": 0,
            "rust_full_model_gain_measurement_flow_vin_events": 0,
            "rust_full_model_gain_measurement_flow_lfsr_events": 0,
            "rust_full_model_gain_measurement_flow_rust_requested": 0,
            "rust_full_model_gain_measurement_flow_rust_enabled": 0,
            "rust_full_model_gain_measurement_flow_rust_fallbacks": 0,
            "rust_full_model_gain_measurement_flow_rust_points": 0,
            "rust_full_model_cmp_delay_enabled": 0,
            "rust_full_model_cmp_delay_clock_events": 0,
            "rust_full_model_cmp_delay_rust_requested": 0,
            "rust_full_model_cmp_delay_rust_enabled": 0,
            "rust_full_model_cmp_delay_rust_fallbacks": 0,
            "rust_full_model_cmp_delay_rust_points": 0,
            "rust_full_model_sar_loop_enabled": 0,
            "rust_full_model_sar_loop_clock_events": 0,
            "rust_full_model_sar_loop_rust_requested": 0,
            "rust_full_model_sar_loop_rust_enabled": 0,
            "rust_full_model_sar_loop_rust_fallbacks": 0,
            "rust_full_model_sar_loop_rust_points": 0,
            "rust_full_model_cppll_reacquire_enabled": 0,
            "rust_full_model_cppll_reacquire_ref_toggles": 0,
            "rust_full_model_cppll_reacquire_dco_toggles": 0,
            "rust_full_model_cppll_reacquire_ref_crosses": 0,
            "rust_full_model_cppll_reacquire_fb_crosses": 0,
            "rust_full_model_cppll_reacquire_rust_requested": 0,
            "rust_full_model_cppll_reacquire_rust_enabled": 0,
            "rust_full_model_cppll_reacquire_rust_fallbacks": 0,
            "rust_full_model_cppll_reacquire_rust_points": 0,
            "rust_full_model_event_transition_core_enabled": 0,
            "rust_full_model_event_transition_core_rust_requested": 0,
            "rust_full_model_event_transition_core_rust_enabled": 0,
            "rust_full_model_event_transition_core_rust_points": 0,
            "rust_full_model_event_transition_core_calls_total": 0,
            "rust_full_model_event_transition_core_fired_events": 0,
            "rust_full_model_event_transition_core_transition_breakpoints": 0,
            "rust_full_model_event_transition_core_source_breakpoints": 0,
            "rust_full_model_event_transition_core_build_errors": 0,
            "rust_full_model_event_transition_core_runtime_fallbacks": 0,
            "rust_full_model_event_transition_core_native_trace_requested": 0,
            "rust_full_model_event_transition_core_native_trace_enabled": 0,
            "rust_full_model_event_transition_core_native_trace_points": 0,
            "rust_full_model_event_transition_core_native_trace_fired_events": 0,
            "rust_full_model_event_transition_core_native_trace_transition_breakpoints": 0,
            "rust_full_model_event_transition_core_native_trace_source_breakpoints": 0,
            "rust_full_model_event_transition_core_native_trace_fallbacks": 0,
            "rust_timer_lfsr_output_batches_total": 0,
            "rust_timer_lfsr_output_calls_total": 0,
            "rust_timer_lfsr_output_due_total": 0,
            "rust_timer_lfsr_output_skips_total": 0,
            "rust_timer_lfsr_output_executed_total": 0,
            "rust_timer_lfsr_output_writes_total": 0,
            "rust_timer_lfsr_output_fallbacks_total": 0,
            "event_trace_audit_requested": int(bool(event_trace_audit)),
            "event_trace_audit_enabled": 0,
            "event_trace_audit_events_total": 0,
            "event_trace_audit_body_entries_total": 0,
            "event_trace_audit_cross_events_total": 0,
            "event_trace_audit_above_events_total": 0,
            "event_trace_audit_timer_events_total": 0,
            "event_trace_audit_initial_step_events_total": 0,
            "event_trace_audit_final_step_events_total": 0,
            "event_trace_audit_combined_events_total": 0,
            "event_trace_audit_state_writes_total": 0,
            "event_trace_audit_array_writes_total": 0,
            "event_trace_audit_output_writes_total": 0,
            "event_trace_audit_timer_state_writes_total": 0,
            "event_trace_audit_timer_last_fired_writes_total": 0,
            "event_trace_audit_transition_writes_total": 0,
            "event_trace_audit_transition_output_writes_total": 0,
            "event_trace_audit_in_event_writes_total": 0,
            "event_trace_audit_records_dropped_total": 0,
            "rust_array_snapshot_copies": 0,
            "rust_array_snapshot_fallbacks": 0,
            "rust_array_err_ratio_scans": 0,
            "rust_array_err_ratio_fallbacks": 0,
            "rust_array_err_ratio_reads": 0,
            "rust_array_record_scans": 0,
            "rust_array_record_fallbacks": 0,
            "rust_array_record_values": 0,
            "rust_array_loop_enabled": 0,
            "rust_transition_breakpoint_enabled": 0,
            "rust_transition_breakpoint_scans_total": 0,
            "rust_transition_breakpoint_state_scans_total": 0,
            "rust_transition_breakpoint_fallbacks_total": 0,
            "rust_timer_breakpoint_enabled": 0,
            "rust_timer_breakpoint_scans_total": 0,
            "rust_timer_breakpoint_state_scans_total": 0,
            "rust_timer_breakpoint_fallbacks_total": 0,
            "timer_array_sidecar_updates_total": 0,
            "timer_array_sidecar_rebuilds_total": 0,
            "timer_array_sidecar_scans_total": 0,
            "static_lifecycle_fastpath_enabled": int(bool(static_lifecycle_fastpath)),
            "transition_unchanged_fastpath_enabled": int(bool(transition_unchanged_fastpath)),
            "model_prepare_step_calls": 0,
            "model_prepare_step_skips": 0,
            "model_timer_expire_calls": 0,
            "model_timer_expire_skips": 0,
            "model_post_update_calls": 0,
            "model_post_update_skips": 0,
            "node_resolution_cache_entries": 0,
            "node_resolution_cache_models": 0,
            "dynamic_node_cache_hits_total": 0,
            "dynamic_node_cache_misses_total": 0,
            "dynamic_node_cache_bypasses_total": 0,
            "dynamic_node_cache_entries": 0,
            "dynamic_node_cache_models": 0,
            "source_breakpoint_scan_calls": 0,
            "model_breakpoint_scan_calls": 0,
            "bound_step_scan_calls": 0,
            "timer_breakpoint_cache_hits_total": 0,
            "timer_breakpoint_hits_total": 0,
            "timer_breakpoint_scans_total": 0,
            "timer_state_updates_total": 0,
            "timer_batch_due_calls_total": 0,
            "timer_batch_due_events_total": 0,
            "timer_batch_due_fires_total": 0,
            "timer_batch_due_fallbacks_total": 0,
            "timer_state_owned_checks_total": 0,
            "timer_state_owned_fast_skips_total": 0,
            "timer_state_owned_target_reads_total": 0,
            "timer_state_owned_fires_total": 0,
            "timer_state_owned_fallbacks_total": 0,
            "transition_unchanged_target_fastpath_total": 0,
            "transition_calls_total": 0,
            "transition_output_fastpath_calls_total": 0,
            "transition_set_target_calls_total": 0,
            "transition_evaluate_calls_total": 0,
            "transition_breakpoint_scans_total": 0,
            "transition_breakpoint_active_scans_total": 0,
            "transition_breakpoint_inactive_skips_total": 0,
            "steps_total": 0,
        }
        self._profile_times: Dict[str, float] = {}
        self._indexed_snapshot_stats: Dict[str, object] = {}
        self._indexed_array_stats: Dict[str, object] = {}
        self._indexed_model_io_stats: Dict[str, object] = {}
        self._indexed_voltage_probe_stats: Dict[str, object] = {}
        self._indexed_voltage_read_stats: Dict[str, object] = {}
        self._model_io_profile_stats: Dict[str, object] = {}
        self._model_profile_stats: Dict[str, Dict[str, float]] = {}
        profile_clock = (
            _wall_time.perf_counter
            if (
                profile_sections
                or profile_model_eval
                or indexed_snapshot_profile
                or indexed_arrays
            )
            else None
        )
        model_profile_enabled = bool(profile_model_eval and profile_clock is not None)

        def _add_profile_time(name: str, start: float) -> float:
            if profile_clock is not None:
                elapsed = profile_clock() - start
                self._profile_times[name] = self._profile_times.get(name, 0.0) + elapsed
                return elapsed
            return 0.0

        def _add_model_profile_time(index: int, model, name: str, elapsed: float):
            if not model_profile_enabled:
                return
            model_name = getattr(getattr(model, "__class__", type(model)), "__name__", "model")
            key = f"model[{index}] {model_name}"
            stats = self._model_profile_stats.setdefault(
                key,
                {
                    "evaluate_calls": 0.0,
                    "evaluate_s": 0.0,
                    "post_update_s": 0.0,
                    "prepare_step_s": 0.0,
                },
            )
            stats[name] = stats.get(name, 0.0) + elapsed

        model_io_profile_enabled = bool(profile_model_io)
        model_io_stats: Dict[str, int] = {
            "voltage_reads": 0,
            "voltage_read_event_contexts": 0,
            "voltage_read_missing_nodes": 0,
            "output_writes": 0,
        }
        model_io_read_local_nodes_seen: set[str] = set()
        model_io_read_external_nodes_seen: set[str] = set()
        model_io_output_nodes_seen: set[str] = set()

        def _refresh_model_io_profile_stats():
            if not model_io_profile_enabled:
                return
            self._model_io_profile_stats = {
                **model_io_stats,
                "voltage_read_local_nodes": len(model_io_read_local_nodes_seen),
                "voltage_read_external_nodes": len(model_io_read_external_nodes_seen),
                "output_write_nodes": len(model_io_output_nodes_seen),
            }

        def _record_model_io_voltage_read(
            local_node: str,
            external_node: str,
            _value: float,
            event_context: bool,
        ):
            if not model_io_profile_enabled:
                return
            model_io_stats["voltage_reads"] += 1
            if event_context:
                model_io_stats["voltage_read_event_contexts"] += 1
            if isinstance(local_node, str):
                model_io_read_local_nodes_seen.add(local_node)
            if isinstance(external_node, str):
                model_io_read_external_nodes_seen.add(external_node)
                if not event_context and external_node not in self.node_voltages:
                    model_io_stats["voltage_read_missing_nodes"] += 1
            _refresh_model_io_profile_stats()

        def _record_model_io_output_write(node: str, _value: float):
            if not model_io_profile_enabled:
                return
            model_io_stats["output_writes"] += 1
            if isinstance(node, str):
                model_io_output_nodes_seen.add(node)
            _refresh_model_io_profile_stats()

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

        def _set_model_indexed_voltage_reader(reader):
            for model in self.models:
                setter = getattr(model, "_set_indexed_voltage_reader", None)
                if setter is not None:
                    setter(reader)

        def _set_model_node_resolution_cache_enabled(enabled: bool):
            for model in self.models:
                setter = getattr(model, "_set_node_resolution_cache_enabled", None)
                if setter is not None:
                    setter(enabled)

        def _set_model_static_branch_fastpath_enabled(enabled: bool):
            for model in self.models:
                setter = getattr(model, "_set_static_branch_fastpath_enabled", None)
                if setter is not None:
                    setter(enabled)

        def _set_model_transition_unchanged_fastpath_enabled(enabled: bool):
            for model in self.models:
                setter = getattr(model, "_set_transition_unchanged_fastpath_enabled", None)
                if setter is not None:
                    setter(enabled)

        def _set_model_rust_transition_breakpoint_scanner(scanner):
            for model in self.models:
                setter = getattr(model, "_set_rust_transition_breakpoint_scanner", None)
                if setter is not None:
                    setter(scanner)

        def _set_model_rust_transition_state_backend(backend, *, production: bool = False):
            for model in self.models:
                setter = getattr(model, "_set_rust_transition_state_backend", None)
                if setter is not None:
                    setter(backend, production=production)

        def _set_model_rust_event_interpolation_backend(backend):
            for model in self.models:
                setter = getattr(model, "_set_rust_event_interpolation_backend", None)
                if setter is not None:
                    setter(backend)

        def _set_model_rust_timer_breakpoint_scanner(scanner):
            for model in self.models:
                setter = getattr(model, "_set_rust_timer_breakpoint_scanner", None)
                if setter is not None:
                    setter(scanner)

        def _set_model_rust_timer_event_backend(backend, *, production: bool = False):
            for model in self.models:
                setter = getattr(model, "_set_rust_timer_event_backend", None)
                if setter is not None:
                    setter(backend, production=production)

        def _set_model_rust_event_due_shadow_backend(backend):
            for model in self.models:
                setter = getattr(model, "_set_rust_event_due_shadow_backend", None)
                if setter is not None:
                    setter(backend)

        def _set_model_rust_cross_above_production_backend(backend, *, production: bool = False):
            for model in self.models:
                setter = getattr(model, "_set_rust_cross_above_production_backend", None)
                if setter is not None:
                    setter(backend, production=production)

        def _set_model_rust_event_write_backend(
            backend,
            node_index=None,
            node_values=None,
            *,
            shadow: bool = False,
            production: bool = False,
        ):
            for model in self.models:
                setter = getattr(model, "_set_rust_event_write_backend", None)
                if setter is not None:
                    setter(
                        backend,
                        node_index,
                        node_values,
                        shadow=shadow,
                        production=production,
                    )

        def _set_model_rust_body_ir_backend(backend, *, production: bool = False):
            for model in self.models:
                setter = getattr(model, "_set_rust_body_ir_backend", None)
                if setter is not None:
                    setter(backend, production=production)

        def _set_model_event_trace_audit_enabled(enabled: bool):
            for model in self.models:
                setter = getattr(model, "_set_event_trace_audit_enabled", None)
                if setter is not None:
                    setter(enabled)

        def _set_model_indexed_state_storage_empty():
            def _visit(model):
                setter = getattr(model, "_set_indexed_state_storage", None)
                if setter is not None:
                    setter(None)
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

        def _install_indexed_state_storage():
            models = 0
            scalar_slots = 0
            integer_slots = 0
            array_slots = 0

            def _visit(model):
                nonlocal models, scalar_slots, integer_slots, array_slots
                setter = getattr(model, "_set_indexed_state_storage", None)
                model_cls = getattr(model, "__class__", type(model))
                scalar_names = tuple(
                    str(name)
                    for name in getattr(model_cls, "_state_scalar_names", ()) or ()
                )
                array_ranges = tuple(
                    (
                        str(name),
                        int(lo),
                        int(hi),
                        bool(integer),
                    )
                    for name, lo, hi, integer in (
                        getattr(model_cls, "_state_array_ranges", ()) or ()
                    )
                )
                if setter is not None:
                    scalar_ids = {}
                    for name in scalar_names:
                        scalar_ids[name] = len(scalar_ids)
                    slot_name_fn = getattr(model, "_state_array_slot_name", None)
                    for array_name, lo, hi, _ in array_ranges:
                        for idx in range(lo, hi + 1):
                            if slot_name_fn is not None:
                                slot_name = slot_name_fn(array_name, idx)
                            else:
                                slot_name = f"{array_name}[{idx}]"
                            scalar_ids[slot_name] = len(scalar_ids)
                    integer_names = tuple(
                        str(name)
                        for name in getattr(model_cls, "_integer_state_names", ()) or ()
                    )
                    setter(scalar_ids, integer_names, array_ranges)
                    if scalar_names or array_ranges:
                        models += 1
                        scalar_slots += len(scalar_names)
                        integer_slots += len(integer_names)
                        array_slots += sum(
                            max(0, hi - lo + 1)
                            for _, lo, hi, _ in array_ranges
                        )
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

            self._perf_stats["indexed_state_storage_models"] = models
            self._perf_stats["indexed_state_storage_scalar_slots"] = scalar_slots
            self._perf_stats["indexed_state_storage_integer_slots"] = integer_slots
            self._perf_stats["indexed_state_storage_array_slots"] = array_slots

        def _set_model_static_branch_indexed_io_empty():
            def _visit(model):
                setter = getattr(model, "_set_static_branch_indexed_io", None)
                if setter is not None:
                    setter()
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

        _set_model_indexed_output_writer(None)
        _set_model_indexed_voltage_probe(None)
        _set_model_indexed_voltage_reader(None)
        _set_model_indexed_state_storage_empty()
        _set_model_node_resolution_cache_enabled(False)
        _set_model_static_branch_fastpath_enabled(False)
        _set_model_transition_unchanged_fastpath_enabled(transition_unchanged_fastpath)
        _set_model_rust_transition_breakpoint_scanner(None)
        _set_model_rust_transition_state_backend(None, production=False)
        _set_model_rust_event_interpolation_backend(None)
        _set_model_rust_timer_breakpoint_scanner(None)
        _set_model_rust_timer_event_backend(None, production=False)
        _set_model_rust_event_due_shadow_backend(None)
        _set_model_rust_event_write_backend(None)
        _set_model_rust_body_ir_backend(None, production=False)
        _set_model_event_trace_audit_enabled(False)
        _set_model_static_branch_indexed_io_empty()
        _set_model_node_resolution_cache_enabled(True)
        rust_event_interpolation_for_event_write = bool(
            rust_event_write_shadow or rust_event_write_production
        )
        if event_trace_audit:
            self._perf_stats["event_trace_audit_enabled"] = 1
            _set_model_event_trace_audit_enabled(True)
        rust_backend = (
            load_optional_rust_backend()
            if (
                rust_static_eval
                or rust_transition_shadow
                or rust_transition_production
                or rust_event_transition_shadow
                or rust_event_transition_production
                or rust_event_interpolation
                or rust_timer_event
                or rust_event_due_shadow
                or rust_cross_above_production
                or rust_event_write_shadow
                or rust_event_write_production
                or rust_body_ir
                or rust_full_model_fastpath
                or indexed_arrays
            )
            else None
        )
        if (
            rust_required
            and (
                rust_static_eval
                or rust_transition_shadow
                or rust_transition_production
                or rust_event_transition_shadow
                or rust_event_transition_production
                or rust_event_interpolation
                or rust_timer_event
                or rust_event_due_shadow
                or rust_cross_above_production
                or rust_event_write_shadow
                or rust_event_write_production
                or rust_body_ir
                or rust_full_model_fastpath
                or indexed_arrays
            )
            and rust_backend is None
        ):
            raise RuntimeError(
                "EVAS Rust backend was required but could not be loaded; "
                "build EVAS/evas/rust_core target/release/libevas_rust_core.so "
                "on this host or unset evas_rust_required."
            )
        if rust_backend is not None:
            if rust_static_eval:
                self._perf_stats["rust_static_eval_available"] = 1
            if rust_transition_shadow:
                self._perf_stats["rust_transition_shadow_available"] = 1
            if rust_event_transition_shadow:
                self._perf_stats["rust_event_transition_shadow_available"] = 1
            if rust_event_transition_production:
                self._perf_stats["rust_event_transition_production_available"] = 1
            if rust_transition_production:
                self._perf_stats["rust_transition_production_available"] = 1
                self._perf_stats["rust_transition_production_enabled"] = 1
                _set_model_rust_transition_state_backend(
                    rust_backend,
                    production=True,
                )
            if rust_timer_event:
                self._perf_stats["rust_timer_event_available"] = 1
                self._perf_stats["rust_timer_event_enabled"] = 1
                self._perf_stats["rust_timer_breakpoint_enabled"] = 1
                _set_model_rust_timer_breakpoint_scanner(
                    rust_backend.next_timer_breakpoint
                )
                _set_model_rust_timer_event_backend(
                    rust_backend,
                    production=True,
                )
            if rust_event_interpolation or rust_event_interpolation_for_event_write:
                self._perf_stats["rust_event_interpolation_available"] = 1
                self._perf_stats["rust_event_interpolation_enabled"] = 1
                _set_model_rust_event_interpolation_backend(rust_backend)
            if rust_event_due_shadow:
                self._perf_stats["rust_event_due_shadow_available"] = 1
                self._perf_stats["rust_event_due_shadow_enabled"] = 1
                _set_model_rust_event_due_shadow_backend(rust_backend)
            if rust_cross_above_production:
                self._perf_stats["rust_cross_above_production_available"] = 1
                self._perf_stats["rust_cross_above_production_enabled"] = 1
                _set_model_rust_cross_above_production_backend(
                    rust_backend, production=True,
                )
            if rust_event_write_shadow:
                self._perf_stats["rust_event_write_shadow_available"] = 1
                self._perf_stats["rust_event_write_shadow_enabled"] = 1
            if rust_event_write_production:
                self._perf_stats["rust_event_write_production_available"] = 1
                self._perf_stats["rust_event_write_production_enabled"] = 1
            if rust_body_ir:
                self._perf_stats["rust_body_ir_available"] = 1
                self._perf_stats["rust_body_ir_enabled"] = 1
                _set_model_rust_body_ir_backend(rust_backend, production=True)
            if rust_full_model_fastpath:
                self._perf_stats["rust_full_model_fastpath_available"] = 1

        def _aggregate_rust_body_ir_plan_stats():
            candidate_models = 0
            enabled_models = 0
            stmt_ops = 0
            expr_ops = 0

            def _visit(model):
                nonlocal candidate_models, enabled_models, stmt_ops, expr_ops
                cls = getattr(model, "__class__", type(model))
                model_stmt_ops = tuple(getattr(cls, "_rust_body_ir_stmt_ops", ()) or ())
                model_expr_ops = tuple(getattr(cls, "_rust_body_ir_expr_ops", ()) or ())
                if model_stmt_ops:
                    candidate_models += 1
                    stmt_ops += len(model_stmt_ops)
                    expr_ops += len(model_expr_ops)
                    if getattr(model, "_rust_body_ir_batch", None) is not None:
                        enabled_models += 1
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)
            self._perf_stats["rust_body_ir_candidate_models"] = candidate_models
            self._perf_stats["rust_body_ir_models"] = enabled_models
            self._perf_stats["rust_body_ir_stmt_ops"] = stmt_ops
            self._perf_stats["rust_body_ir_expr_ops"] = expr_ops

        _aggregate_rust_body_ir_plan_stats()

        def _aggregate_event_transition_plan_stats():
            profile_suffixes = {
                "event_transition_core": "core",
                "event_transition_ordered_v1": "ordered_v1",
                "event_transition_with_side_effect_boundary": "side_effect",
            }

            def _visit(model):
                cls = getattr(model, "__class__", type(model))
                profiles = tuple(
                    getattr(cls, "_event_transition_plan_profiles", ()) or ()
                )
                for profile in profiles:
                    suffix = profile_suffixes.get(str(profile))
                    if suffix is None:
                        continue
                    self._perf_stats[
                        f"rust_event_transition_plan_{suffix}_candidate_models"
                    ] += 1
                    self._perf_stats[
                        f"rust_event_transition_plan_{suffix}_event_statements"
                    ] += int(
                        getattr(cls, "_event_transition_plan_event_count", 0) or 0
                    )
                    self._perf_stats[
                        f"rust_event_transition_plan_{suffix}_due_triggers"
                    ] += int(
                        getattr(cls, "_event_transition_plan_due_trigger_count", 0)
                        or 0
                    )
                    self._perf_stats[
                        f"rust_event_transition_plan_{suffix}_transitions"
                    ] += int(
                        getattr(cls, "_event_transition_plan_transition_count", 0)
                        or 0
                    )
                    self._perf_stats[
                        f"rust_event_transition_plan_{suffix}_output_writes"
                    ] += int(
                        getattr(cls, "_event_transition_plan_output_write_count", 0)
                        or 0
                    )
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

        _aggregate_event_transition_plan_stats()

        rust_event_transition_plans: Dict[int, tuple] = {}
        rust_event_transition_production_buffers: Dict[int, tuple] = {}

        def _build_event_transition_plans():
            if (
                not (rust_event_transition_shadow or rust_event_transition_production)
                or rust_backend is None
            ):
                return
            candidates = 0
            planned = 0
            for model_index, model in enumerate(self.models):
                cls = getattr(model, "__class__", type(model))
                profiles = tuple(
                    getattr(cls, "_event_transition_plan_profiles", ()) or ()
                )
                if "event_transition_core" not in profiles:
                    continue
                candidates += 1
                module = getattr(cls, "_module_ast", None)
                node_names = tuple(getattr(cls, "_rust_body_ir_node_names", ()) or ())
                if module is None or not node_names:
                    continue
                node_slots = {name: idx for idx, name in enumerate(node_names)}
                try:
                    runtime = try_build_event_then_transition_shadow_runtime(
                        module,
                        rust_backend,
                        node_slots,
                        default_transition=float(
                            getattr(model, "default_transition", 1e-12) or 1e-12
                        ),
                    )
                except Exception:
                    self._perf_stats["rust_event_transition_shadow_errors_total"] += 1
                    continue
                if runtime is None:
                    continue
                if rust_event_transition_production:
                    event_body_reads = tuple(
                        getattr(cls, "_event_body_voltage_read_nodes", ()) or ()
                    )
                    if event_body_reads:
                        continue
                    event_runtimes = tuple(
                        getattr(
                            getattr(runtime, "event_runtime", None),
                            "event_runtimes",
                            (),
                        )
                        or ()
                    )
                    uses_timer = any(
                        str(getattr(trigger, "kind", "")) == "timer"
                        for event_runtime in event_runtimes
                        for trigger in getattr(
                            getattr(
                                getattr(event_runtime, "due_runtime", None),
                                "program",
                                None,
                            ),
                            "triggers",
                            (),
                        )
                    )
                    if uses_timer:
                        continue
                rust_event_transition_plans[model_index] = (
                    model,
                    runtime,
                    node_names,
                    tuple(getattr(cls, "_state_scalar_names", ()) or ()),
                    tuple(
                        str(param.name)
                        for param in getattr(module, "parameters", ()) or ()
                    ),
                )
                planned += 1
            if rust_event_transition_shadow:
                self._perf_stats[
                    "rust_event_transition_shadow_candidate_models"
                ] = candidates
                self._perf_stats["rust_event_transition_shadow_models"] = planned
            if rust_event_transition_production:
                self._perf_stats[
                    "rust_event_transition_production_candidate_models"
                ] = candidates
                self._perf_stats["rust_event_transition_production_models"] = planned
            if planned and rust_event_transition_shadow:
                self._perf_stats["rust_event_transition_shadow_enabled"] = 1
            if planned and rust_event_transition_production:
                self._perf_stats["rust_event_transition_production_enabled"] = 1

        def _fill_event_transition_node_array(model, node_names, values) -> None:
            for idx, local_name in enumerate(node_names):
                external_name = model.node_map.get(local_name, local_name)
                if external_name in self.node_voltages:
                    values[idx] = float(self.node_voltages.get(external_name, 0.0))
                else:
                    values[idx] = float(model.output_nodes.get(local_name, 0.0))

        def _event_transition_node_array(model, node_names: Tuple[str, ...]) -> array:
            values = array("d", [0.0] * len(node_names))
            _fill_event_transition_node_array(model, node_names, values)
            return values

        def _fill_event_transition_state_array(model, state_names, values) -> None:
            for idx, name in enumerate(state_names):
                values[idx] = float(model.state.get(name, 0.0))

        def _event_transition_state_array(model, state_names: Tuple[str, ...]) -> array:
            values = array("d", [0.0] * len(state_names))
            _fill_event_transition_state_array(model, state_names, values)
            return values

        def _event_transition_param_array(model, param_names: Tuple[str, ...]) -> array:
            values = array("d")
            for name in param_names:
                value = model.params.get(name, 0.0)
                values.append(float(value) if isinstance(value, (int, float)) else 0.0)
            return values

        def _run_rust_event_transition_shadow_pre(
            model_index: int,
            eval_time: float,
            *,
            initial_step: bool = False,
        ):
            if not rust_event_transition_shadow:
                return None
            plan = rust_event_transition_plans.get(model_index)
            if plan is None:
                return None
            model, runtime, node_names, state_names, param_names = plan
            node_values = _event_transition_node_array(model, node_names)
            state_values = _event_transition_state_array(model, state_names)
            param_values = _event_transition_param_array(model, param_names)
            try:
                runtime.step(
                    time=eval_time,
                    node_values=node_values,
                    state_values=state_values,
                    param_values=param_values,
                    initial_step=initial_step,
                )
            except Exception:
                self._perf_stats["rust_event_transition_shadow_errors_total"] += 1
                return None
            self._perf_stats["rust_event_transition_shadow_calls_total"] += 1
            return (model, node_names, state_names, node_values, state_values)

        def _compare_rust_event_transition_shadow(shadow_result) -> None:
            if shadow_result is None:
                return
            model, node_names, state_names, node_values, state_values = shadow_result
            node_slots = {name: idx for idx, name in enumerate(node_names)}
            check_nodes = set(getattr(model.__class__, "_static_output_write_nodes", ()) or ())
            check_nodes.update(model.output_nodes.keys())
            max_diff = 0.0
            checks = 0
            for local_name in sorted(check_nodes):
                slot = node_slots.get(local_name)
                if slot is None:
                    continue
                external_name = model.node_map.get(local_name, local_name)
                if external_name in self.node_voltages:
                    python_value = float(self.node_voltages.get(external_name, 0.0))
                else:
                    python_value = float(model.output_nodes.get(local_name, 0.0))
                diff = abs(float(node_values[slot]) - python_value)
                max_diff = max(max_diff, diff)
                checks += 1
            for slot, name in enumerate(state_names):
                if slot >= len(state_values):
                    break
                diff = abs(float(state_values[slot]) - float(model.state.get(name, 0.0)))
                max_diff = max(max_diff, diff)
                checks += 1
            self._perf_stats[
                "rust_event_transition_shadow_value_checks_total"
            ] += checks
            if max_diff > self._perf_stats["rust_event_transition_shadow_max_abs_diff"]:
                self._perf_stats["rust_event_transition_shadow_max_abs_diff"] = max_diff
            if max_diff <= 1e-9:
                self._perf_stats["rust_event_transition_shadow_matches_total"] += 1
            else:
                self._perf_stats["rust_event_transition_shadow_mismatches_total"] += 1

        def _sync_rust_event_transition_state(model, state_names, state_values) -> int:
            integer_names = set(getattr(model.__class__, "_integer_state_names", ()) or ())
            writes = 0
            for slot, name in enumerate(state_names):
                if slot >= len(state_values):
                    break
                value = float(state_values[slot])
                if name in integer_names:
                    value = model._to_integer(value)
                model._state_set_by_slot(slot, name, value)
                writes += 1
            return writes

        def _sync_rust_event_transition_outputs(
            model,
            runtime,
            node_names,
            node_values,
        ) -> int:
            node_slots = {name: idx for idx, name in enumerate(node_names)}
            transition_program = getattr(
                getattr(runtime, "transition_runtime", None),
                "program",
                None,
            )
            output_slots = set(
                int(slot)
                for slot in getattr(transition_program, "output_node_slots", ()) or ()
            )
            for local_name in getattr(model.__class__, "_static_output_write_nodes", ()) or ():
                slot = node_slots.get(local_name)
                if slot is not None:
                    output_slots.add(slot)
            writes = 0
            for slot in sorted(output_slots):
                if slot < 0 or slot >= len(node_names) or slot >= len(node_values):
                    continue
                model._set_output(
                    node_names[slot],
                    float(node_values[slot]),
                    self.node_voltages,
                )
                writes += 1
            return writes

        def _run_rust_event_transition_production(
            model_index: int,
            eval_time: float,
        ) -> bool:
            if not rust_event_transition_production or event_trace_audit:
                return False
            plan = rust_event_transition_plans.get(model_index)
            if plan is None:
                return False
            model, runtime, node_names, state_names, param_names = plan
            if getattr(model, "_child_models", None):
                self._perf_stats[
                    "rust_event_transition_production_fallbacks_total"
                ] += 1
                return False
            if getattr(model, "_transition_pending_count", 0):
                self._perf_stats[
                    "rust_event_transition_production_fallbacks_total"
                ] += 1
                return False
            model._event_trace_audit_phase = "evaluate"
            model._event_time = eval_time
            model._bound_step = 0.0
            buffers = rust_event_transition_production_buffers.get(model_index)
            if buffers is None:
                node_values = _event_transition_node_array(model, node_names)
                state_values = _event_transition_state_array(model, state_names)
                param_values = _event_transition_param_array(model, param_names)
                rust_event_transition_production_buffers[model_index] = (
                    node_values,
                    state_values,
                    param_values,
                )
            else:
                node_values, state_values, param_values = buffers
                _fill_event_transition_node_array(model, node_names, node_values)
                _fill_event_transition_state_array(model, state_names, state_values)
            try:
                result = runtime.step(
                    time=eval_time,
                    node_values=node_values,
                    state_values=state_values,
                    param_values=param_values,
                    initial_step=False,
                )
            except Exception:
                self._perf_stats[
                    "rust_event_transition_production_fallbacks_total"
                ] += 1
                return False

            self._perf_stats["rust_event_transition_production_calls_total"] += 1
            self._perf_stats["rust_event_transition_production_executed_total"] += 1
            state_writes = _sync_rust_event_transition_state(
                model,
                state_names,
                state_values,
            )
            output_writes = _sync_rust_event_transition_outputs(
                model,
                runtime,
                node_names,
                node_values,
            )
            fired_events = len(getattr(result, "fired_event_statements", ()) or ())
            if fired_events:
                model._step_event_fired = True
            self._perf_stats[
                "rust_event_transition_production_state_writes_total"
            ] += state_writes
            self._perf_stats[
                "rust_event_transition_production_output_writes_total"
            ] += output_writes
            self._perf_stats[
                "rust_event_transition_production_fired_events_total"
            ] += fired_events
            return True

        def _rust_event_transition_next_breakpoint(time_value: float) -> Optional[float]:
            if not rust_event_transition_production:
                return None
            best: Optional[float] = None
            for _model, runtime, _node_names, _state_names, _param_names in (
                rust_event_transition_plans.values()
            ):
                self._perf_stats[
                    "rust_event_transition_production_breakpoint_scans_total"
                ] += 1
                try:
                    bp = runtime.next_breakpoint(
                        time_value,
                        transition_breakpoint_min_ramp,
                    )
                except Exception:
                    bp = None
                if bp is not None and bp > time_value and (
                    best is None or bp < best
                ):
                    best = bp
            return best

        _build_event_transition_plans()
        if indexed_state_storage:
            _install_indexed_state_storage()
        if static_branch_fastpath:
            _set_model_static_branch_fastpath_enabled(True)
        if model_io_profile_enabled:
            _set_model_indexed_output_writer(_record_model_io_output_write)
            _set_model_indexed_voltage_probe(_record_model_io_voltage_read)

        source_nodes = {src.node for src in self.sources}
        source_future_waveforms = {src.node: src.waveform for src in self.sources}
        source_breakpoint_sources = [src for src in self.sources if src.breakpoint_fn is not None]
        source_breakpoint_sources.extend(
            src for _pos, _neg, src in self.current_sources if src.breakpoint_fn is not None
        )
        self._generic_executor_enabled = bool(generic_executor)
        if rust_full_model_fastpath:
            sim_program_result = self._try_rust_sim_program_fastpath(
                rust_backend=rust_backend,
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
                max_step=max_step,
                cross_acceptance_slack_factor=cross_acceptance_slack_factor,
            )
            if sim_program_result is not None:
                return sim_program_result
            full_model_result = self._try_compiler_whole_segment_fastpath(
                rust_backend=rust_backend,
                tstop=tstop,
                record_step=record_step,
                tstep=tstep,
                max_step=max_step,
                min_step=min_step,
                rust_full_model_required=rust_full_model_required,
            )
            if full_model_result is not None:
                return full_model_result
            if rust_full_model_required:
                self._perf_stats["rust_full_model_required_failures"] = 1
                rejection = getattr(
                    self,
                    "_rust_sim_program_last_rejection",
                    "unknown",
                )
                raise RuntimeError(
                    "evas-rust full-model path was required but no "
                    "supported whole-segment Rust runtime matched this design. "
                    "Use the python engine for fallback or extend the evas-rust "
                    "Rust lowering coverage. "
                    f"RustSimProgram rejection: {rejection}"
                )
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
        model_has_dynamic_breakpoints = tuple(
            bool(getattr(model, "_has_dynamic_breakpoints_tree", lambda: True)())
            for model in self.models
        )
        breakpoint_models = [
            model
            for model, has_dynamic_breakpoints in zip(
                self.models,
                model_has_dynamic_breakpoints,
            )
            if has_dynamic_breakpoints
        ]
        transition_target_breakpoint_models = [
            model
            for model in self.models
            if bool(
                getattr(model, "_has_transition_target_probes_tree", lambda: False)()
            )
        ]
        model_uses_bound_step = tuple(
            bool(getattr(model, "_uses_bound_step_tree", lambda: True)())
            for model in self.models
        )
        bound_step_models = [
            model
            for model, uses_bound_step in zip(self.models, model_uses_bound_step)
            if uses_bound_step
        ]
        model_has_post_update_events = tuple(
            bool(getattr(model, "_has_post_update_events", True))
            for model in self.models
        )
        adaptive_step_floor_model_sensitive = bool(
            breakpoint_models
            or transition_target_breakpoint_models
            or bound_step_models
            or any(model_has_post_update_events)
        )
        self._perf_stats["adaptive_step_floor_sensitive_models"] = int(
            adaptive_step_floor_model_sensitive
        )
        model_needs_future_node_voltages = tuple(
            bool(getattr(model, "_needs_future_node_voltages_tree", lambda: False)())
            for model in self.models
        )
        models_need_future_node_voltages = any(model_needs_future_node_voltages)
        model_needs_step_context = tuple(
            (
                not static_lifecycle_fastpath
                or needs_future
                or has_dynamic_breakpoints
                or has_post_update_events
            )
            for needs_future, has_dynamic_breakpoints, has_post_update_events in zip(
                model_needs_future_node_voltages,
                model_has_dynamic_breakpoints,
                model_has_post_update_events,
            )
        )
        model_needs_timer_expire = tuple(
            (
                not static_lifecycle_fastpath
                or has_dynamic_breakpoints
                or has_post_update_events
            )
            for has_dynamic_breakpoints, has_post_update_events in zip(
                model_has_dynamic_breakpoints,
                model_has_post_update_events,
            )
        )

        def _set_initial_condition_mode(model, enabled: bool):
            if hasattr(model, "_set_initial_condition_mode"):
                model._set_initial_condition_mode(enabled)

        def _set_current_source_values(nv: Dict[str, float], t: float) -> None:
            for pos, neg, src in self.current_sources:
                value = src.waveform(t)
                nv[f"@I:{pos}:{neg}"] = value
                nv[f"@I:{neg}:{pos}"] = -value

        # Initialize node voltages
        for src in self.sources:
            self.node_voltages[src.node] = src.waveform(0.0)
        _set_current_source_values(self.node_voltages, 0.0)
        for model in self.models:
            model._prepare_step(self.node_voltages, self.node_voltages, 0.0, 0.0)

        # Spectre solves initial_step and coincident t=0 events before the
        # first saved transient point.  During that operating-point-like pass,
        # transition() contributes its settled target instead of a ramp from an
        # implicit zero.
        for model in self.models:
            _set_initial_condition_mode(model, True)

        # Fire initial_step events
        for model_index, model in enumerate(self.models):
            shadow_result = _run_rust_event_transition_shadow_pre(
                model_index,
                0.0,
                initial_step=True,
            )
            model.initial_step(self.node_voltages, 0.0)
            _compare_rust_event_transition_shadow(shadow_result)

        def _evaluate_model_with_optional_body_ir(
            model_index: int,
            model,
            eval_time: float,
        ) -> bool:
            shadow_result = _run_rust_event_transition_shadow_pre(
                model_index,
                eval_time,
                initial_step=False,
            )
            if rust_body_ir:
                runner = getattr(model, "_try_evaluate_rust_body_ir", None)
                if runner is not None and runner(self.node_voltages, eval_time):
                    _compare_rust_event_transition_shadow(shadow_result)
                    return True
            if _run_rust_event_transition_production(model_index, eval_time):
                _compare_rust_event_transition_shadow(shadow_result)
                return True
            model.evaluate(self.node_voltages, eval_time)
            _compare_rust_event_transition_shadow(shadow_result)
            return False

        # Evaluate models at t=0 so output nodes are assigned before recording.
        # Without this, output nodes default to 0, producing spurious values in
        # post-processing (e.g. noise = vout_o - vin_i = 0 - 1 = -1 V).
        for model_index, (model, has_post_update_events) in enumerate(
            zip(self.models, model_has_post_update_events)
        ):
            _evaluate_model_with_optional_body_ir(model_index, model, 0.0)
            model._expire_absolute_timers(0.0)
            if has_post_update_events and model.post_update_events(self.node_voltages, 0.0):
                model.refresh_outputs(self.node_voltages, 0.0)

        for model in self.models:
            _set_initial_condition_mode(model, False)

        model_output_nodes: set[str] = set()
        model_output_versions: Optional[tuple[int, ...]] = None
        err_ratio_nodes: tuple[str, ...] = ()
        err_ratio_cache_key = None
        err_ratio_node_id_batch = None
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
        indexed_record_node_ids = None
        rust_record_node_batch = None
        indexed_model_io_versions = None
        indexed_model_io_plan = None
        indexed_output_nodes_seen: set[str] = set()
        indexed_voltage_probe_max_node = ""
        indexed_voltage_read_nodes_seen: set[str] = set()
        indexed_dirty_nodes: set[str] = set()
        indexed_dirty_validation_nodes: Tuple[str, ...] = ()
        indexed_dirty_validation_enabled = False
        rust_static_eval_segments_by_start: Dict[int, tuple] = {}
        rust_static_eval_segment_members: set[int] = set()
        rust_transition_shadow_plans: Dict[int, tuple] = {}
        rust_static_fast_sync_enabled = False
        rust_array_loop_enabled = False
        if indexed_arrays:
            indexed_array = IndexedVoltageArray.from_names(
                sorted(set(self.node_voltages) | set(self.recorded_signals) | source_nodes | model_output_nodes)
            )
            indexed_array.update_from_mapping(self.node_voltages)
            indexed_record_node_ids = tuple(
                indexed_array.node_index.id_of(name)
                for name in self.recorded_signals
            )
            rust_array_loop_enabled = (
                rust_backend is not None
                and indexed_array.node_count >= 64
            )
            if rust_array_loop_enabled and not isinstance(indexed_array.values, array):
                indexed_array.values = array("d", indexed_array.values)
            if rust_array_loop_enabled:
                rust_record_node_batch = rust_backend.make_node_id_batch(
                    indexed_record_node_ids
                )
                self._perf_stats["rust_array_loop_enabled"] = 1
                self._perf_stats["rust_transition_breakpoint_enabled"] = 1
                self._perf_stats["rust_timer_breakpoint_enabled"] = 1
                _set_model_rust_transition_breakpoint_scanner(
                    rust_backend.next_transition_breakpoint
                )
                _set_model_rust_timer_breakpoint_scanner(
                    rust_backend.next_timer_breakpoint
                )
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
        rust_array_prev_values = None

        def _model_tree_output_versions() -> tuple[int, ...]:
            versions: List[int] = []

            def _visit(model):
                versions.append(int(getattr(model, "_output_nodes_version", len(model.output_nodes))))
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)
            return tuple(versions)

        def _model_at_path(path: tuple[int, ...]):
            if not path:
                return None
            try:
                model = self.models[path[0]]
                for child_idx in path[1:]:
                    model = getattr(model, "_child_models", [])[child_idx]
                return model
            except (IndexError, TypeError):
                return None

        def _install_static_branch_indexed_io():
            if (
                indexed_array is None
                or indexed_model_io_plan is None
                or not static_branch_fastpath
            ):
                return
            models = 0
            read_nodes = 0
            write_nodes = 0
            for model_io in indexed_model_io_plan.model_ios:
                model = _model_at_path(model_io.model_path)
                setter = getattr(model, "_set_static_branch_indexed_io", None)
                if model is None or setter is None:
                    continue
                write_external_nodes = tuple(
                    indexed_model_io_plan.node_index.name_of(node_id)
                    for node_id in model_io.static_output_write_node_ids
                )
                setter(
                    model_io.static_voltage_read_node_ids,
                    model_io.static_output_write_node_ids,
                    write_external_nodes,
                    indexed_array.values,
                )
                if (
                    model_io.static_voltage_read_node_ids
                    or model_io.static_output_write_node_ids
                ):
                    models += 1
                    read_nodes += len(model_io.static_voltage_read_node_ids)
                    write_nodes += len(model_io.static_output_write_node_ids)
            self._perf_stats["static_branch_direct_array_models"] = models
            self._perf_stats["static_branch_direct_array_read_nodes"] = read_nodes
            self._perf_stats["static_branch_direct_array_write_nodes"] = write_nodes

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
            _install_static_branch_indexed_io()
            indexed_model_io_versions = versions
            self._perf_stats["indexed_model_io_refreshes"] += 1
            self._perf_stats["indexed_model_io_models"] = indexed_model_io_plan.model_count
            self._perf_stats["indexed_model_io_mapped_ports"] = (
                indexed_model_io_plan.mapped_port_count
            )
            self._perf_stats["indexed_model_io_outputs"] = indexed_model_io_plan.output_count
            self._indexed_model_io_stats = {
                "dynamic_output_write_count": indexed_model_io_plan.dynamic_output_write_count,
                "dynamic_branch_access_count": indexed_model_io_plan.dynamic_branch_access_count,
                "dynamic_voltage_read_count": indexed_model_io_plan.dynamic_voltage_read_count,
                "event_body_voltage_read_count": indexed_model_io_plan.event_body_voltage_read_count,
                "event_trigger_voltage_count": indexed_model_io_plan.event_trigger_voltage_count,
                "event_voltage_read_count": indexed_model_io_plan.event_voltage_read_count,
                "node_count": indexed_model_io_plan.node_count,
                "model_count": indexed_model_io_plan.model_count,
                "mapped_port_count": indexed_model_io_plan.mapped_port_count,
                "output_count": indexed_model_io_plan.output_count,
                "scalar_state_count": indexed_model_io_plan.scalar_state_count,
                "integer_state_count": indexed_model_io_plan.integer_state_count,
                "state_array_count": indexed_model_io_plan.state_array_count,
                "state_array_slot_count": indexed_model_io_plan.state_array_slot_count,
                "static_output_write_count": indexed_model_io_plan.static_output_write_count,
                "static_voltage_read_count": indexed_model_io_plan.static_voltage_read_count,
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
            self._indexed_array_stats["record_id_reads"] = self._perf_stats[
                "indexed_array_record_id_reads"
            ]
            self._indexed_array_stats["err_ratio_reads"] = self._perf_stats[
                "indexed_array_err_ratio_reads"
            ]
            self._indexed_array_stats["checked_values"] = self._perf_stats[
                "indexed_array_values_checked"
            ]
            self._indexed_array_stats["dirty_validation_enabled"] = self._perf_stats[
                "indexed_array_dirty_validation_enabled"
            ]
            self._indexed_array_stats["dirty_syncs"] = self._perf_stats[
                "indexed_array_dirty_syncs"
            ]
            self._indexed_array_stats["dirty_nodes_checked"] = self._perf_stats[
                "indexed_array_dirty_nodes_checked"
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
            self._indexed_voltage_read_stats = {
                "reads": self._perf_stats["indexed_voltage_reads"],
                "read_nodes": self._perf_stats["indexed_voltage_read_nodes"],
                "fallbacks": self._perf_stats["indexed_voltage_read_fallbacks"],
            }

        def _install_rust_event_write_batches():
            if (
                rust_backend is None
                or not (rust_event_write_shadow or rust_event_write_production)
            ):
                return
            if indexed_array is not None:
                required_nodes: set[str] = set()

                def _collect_required_nodes(model):
                    getter = getattr(model, "_rust_timer_lfsr_required_external_nodes", None)
                    if getter is not None:
                        required_nodes.update(str(node) for node in getter())
                    getter = getattr(model, "_rust_event_linear_required_external_nodes", None)
                    if getter is not None:
                        required_nodes.update(str(node) for node in getter())

                for model in self.models:
                    _collect_required_nodes(model)
                if required_nodes:
                    indexed_array.ensure_nodes(required_nodes)
            _set_model_rust_event_write_backend(
                rust_backend,
                indexed_array.node_index if indexed_array is not None else None,
                indexed_array.values if indexed_array is not None else None,
                shadow=rust_event_write_shadow,
                production=rust_event_write_production,
            )
            models = 0
            batches = 0
            linear_batches = 0

            def _visit(model):
                nonlocal models, batches, linear_batches
                current = len(getattr(model, "_rust_event_write_batches", {}) or {})
                current_linear = len(
                    getattr(model, "_rust_event_linear_write_batches", {}) or {}
                )
                current += current_linear
                if current:
                    models += 1
                    batches += current
                    linear_batches += current_linear
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)
            self._perf_stats["rust_event_write_models"] = models
            self._perf_stats["rust_event_write_batches"] = batches
            self._perf_stats["rust_event_linear_write_batches"] = linear_batches

        def _build_rust_static_eval_plans():
            nonlocal rust_static_eval_segments_by_start, rust_static_eval_segment_members
            rust_static_eval_segments_by_start = {}
            rust_static_eval_segment_members = set()
            if not rust_static_eval or indexed_array is None:
                return

            candidates = 0
            planned_models = 0
            planned_ops = 0
            fallback_models = 0
            planned_terms = 0
            state_read_terms = 0
            state_write_ops = 0
            integer_state_write_ops = 0
            per_model_ops: Dict[int, Tuple[List[LinearOp], Tuple[tuple, ...], Optional[List[float]], Tuple[tuple, ...], bool]] = {}

            def _count_static_fallback(reason: str) -> None:
                key = f"rust_static_eval_fallback_{reason}"
                self._perf_stats[key] = int(self._perf_stats.get(key, 0) or 0) + 1

            def _count_static_no_candidate(reason: str, count: int = 1) -> None:
                token = "".join(
                    ch if ch.isalnum() else "_"
                    for ch in str(reason).lower()
                )
                token = "_".join(part for part in token.split("_") if part) or "unknown"
                key = f"rust_static_eval_no_candidate_{token}"
                self._perf_stats[key] = int(self._perf_stats.get(key, 0) or 0) + int(
                    count
                )

            for model_index, model in enumerate(self.models):
                raw_ir_metadata = tuple(
                    getattr(model.__class__, "_evaluate_ir_static_linear_ops", ()) or ()
                )
                if not raw_ir_metadata:
                    self._perf_stats["rust_static_eval_no_candidate_models"] += 1
                    rejections = tuple(
                        getattr(
                            model.__class__,
                            "_evaluate_ir_static_linear_rejections",
                            (),
                        )
                        or ()
                    )
                    if not rejections:
                        _count_static_no_candidate("unknown")
                    for reason, count in rejections:
                        _count_static_no_candidate(reason, count)
                    continue
                candidates += 1
                if rust_backend is None:
                    _count_static_fallback("backend_unavailable")
                    fallback_models += 1
                    continue
                if model_has_post_update_events[model_index]:
                    _count_static_fallback("post_update_events")
                    fallback_models += 1
                    continue
                if model_needs_future_node_voltages[model_index]:
                    _count_static_fallback("future_node_voltages")
                    fallback_models += 1
                    continue
                if getattr(model, "_child_models", []):
                    _count_static_fallback("child_models")
                    fallback_models += 1
                    continue

                try:
                    ir_ops = normalize_linear_ops(raw_ir_metadata)
                except (TypeError, ValueError):
                    _count_static_fallback("normalize_failed")
                    fallback_models += 1
                    continue

                ops: List[LinearOp] = []
                sync_entries = []
                state_io_entries = []
                model_fallback_reason = "ir_conversion_failed"
                model_uses_state = False
                model_integer_state_write_ops = 0
                state_values = getattr(model, "_indexed_state_values", None)
                state_ids = getattr(model, "_indexed_state_ids", {}) or {}

                def _convert_linear_terms(ir_terms) -> Tuple[Optional[List[LinearTerm]], int]:
                    nonlocal model_uses_state, model_fallback_reason
                    converted_terms: List[LinearTerm] = []
                    state_reads = 0
                    for term in ir_terms:
                        try:
                            gain_value = model._evaluate_rust_static_affine_scalar(
                                term.gain,
                                model.params,
                            )
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            self._perf_stats["rust_static_eval_coeff_eval_fallbacks"] += 1
                            return None, 0

                        if term.source_kind == SOURCE_NODE:
                            external = model._resolve_external_node(term.source_name)
                            indexed_array.ensure_nodes((external,))
                            source_id = indexed_array.node_index.id_of(external)
                            converted_terms.append(
                                LinearTerm(
                                    source_kind=SOURCE_NODE,
                                    source_id=source_id,
                                    gain=gain_value,
                                )
                            )
                        elif term.source_kind == SOURCE_STATE:
                            model_uses_state = True
                            if state_values is None:
                                model_fallback_reason = "state_storage_missing"
                                return None, 0
                            source_id = state_ids.get(term.source_name)
                            if source_id is None:
                                model_fallback_reason = "state_id_missing"
                                return None, 0
                            converted_terms.append(
                                LinearTerm(
                                    source_kind=SOURCE_STATE,
                                    source_id=source_id,
                                    gain=gain_value,
                                )
                            )
                            state_reads += 1
                        else:
                            model_fallback_reason = "unsupported_source"
                            return None, 0
                    return converted_terms, state_reads

                def _convert_condition(ir_condition) -> Tuple[Optional[LinearCondition], int]:
                    nonlocal model_fallback_reason
                    if ir_condition is None:
                        return None, 0
                    try:
                        left_bias = model._evaluate_rust_static_affine_scalar(
                            ir_condition.left_bias,
                            model.params,
                        )
                        right_bias = model._evaluate_rust_static_affine_scalar(
                            ir_condition.right_bias,
                            model.params,
                        )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        self._perf_stats["rust_static_eval_coeff_eval_fallbacks"] += 1
                        model_fallback_reason = "coeff_eval_failed"
                        return None, 0
                    left_terms, left_reads = _convert_linear_terms(
                        ir_condition.left_terms
                    )
                    right_terms, right_reads = _convert_linear_terms(
                        ir_condition.right_terms
                    )
                    if left_terms is None or right_terms is None:
                        return None, 0
                    return (
                        LinearCondition(
                            op_kind=ir_condition.op_kind,
                            left_bias=left_bias,
                            left_terms=tuple(left_terms),
                            right_bias=right_bias,
                            right_terms=tuple(right_terms),
                        ),
                        left_reads + right_reads,
                    )

                for ir_op in ir_ops:
                    try:
                        bias_value = model._evaluate_rust_static_affine_scalar(
                            ir_op.bias,
                            model.params,
                        )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        self._perf_stats["rust_static_eval_coeff_eval_fallbacks"] += 1
                        model_fallback_reason = "coeff_eval_failed"
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        break

                    if linear_op_uses_params(ir_op):
                        self._perf_stats["rust_static_eval_runtime_param_ops"] += 1

                    terms, state_reads_for_op = _convert_linear_terms(ir_op.terms)
                    if terms is None:
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        model_fallback_reason = "term_conversion_failed"
                        break
                    false_terms, false_reads = _convert_linear_terms(ir_op.false_terms)
                    if false_terms is None:
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        model_fallback_reason = "term_conversion_failed"
                        break
                    condition, condition_reads = _convert_condition(ir_op.condition)
                    if ir_op.condition is not None and condition is None:
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        model_fallback_reason = "condition_conversion_failed"
                        break
                    state_reads_for_op += false_reads + condition_reads
                    try:
                        false_bias_value = model._evaluate_rust_static_affine_scalar(
                            ir_op.false_bias,
                            model.params,
                        )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        self._perf_stats["rust_static_eval_coeff_eval_fallbacks"] += 1
                        model_fallback_reason = "coeff_eval_failed"
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        break

                    if ir_op.target_kind == TARGET_NODE:
                        write_external = model._resolve_external_node(ir_op.target_name)
                        indexed_array.ensure_nodes((write_external,))
                        target_id = indexed_array.node_index.id_of(write_external)
                        sync_entries.append(
                            (model, ir_op.target_name, write_external, target_id)
                        )
                    elif ir_op.target_kind == TARGET_STATE:
                        model_uses_state = True
                        if state_values is None:
                            ops = []
                            sync_entries = []
                            state_io_entries = []
                            model_fallback_reason = "state_storage_missing"
                            break
                        target_id = state_ids.get(ir_op.target_name)
                        if target_id is None:
                            ops = []
                            sync_entries = []
                            state_io_entries = []
                            model_fallback_reason = "state_id_missing"
                            break
                        state_io_entries.append((model, state_reads_for_op, 1))
                        if getattr(ir_op, "target_integer", False):
                            model_integer_state_write_ops += 1
                    else:
                        ops = []
                        sync_entries = []
                        state_io_entries = []
                        model_fallback_reason = "unsupported_target"
                        break

                    if ir_op.target_kind != TARGET_STATE and state_reads_for_op:
                        state_io_entries.append((model, state_reads_for_op, 0))

                    ops.append(
                        LinearOp(
                            target_kind=ir_op.target_kind,
                            target_id=target_id,
                            bias=bias_value,
                            terms=tuple(terms),
                            condition=condition,
                            false_bias=false_bias_value,
                            false_terms=tuple(false_terms),
                            target_integer=bool(
                                getattr(ir_op, "target_integer", False)
                            ),
                        )
                    )

                if not ops:
                    _count_static_fallback(model_fallback_reason)
                    fallback_models += 1
                    continue
                per_model_ops[model_index] = (
                    ops,
                    tuple(sync_entries),
                    state_values if model_uses_state else None,
                    tuple(state_io_entries),
                    bool(model_uses_state),
                )
                planned_models += 1
                planned_ops += len(ops)
                planned_terms += sum(
                    len(op.terms)
                    + len(op.false_terms)
                    + (
                        0
                        if op.condition is None
                        else len(op.condition.left_terms)
                        + len(op.condition.right_terms)
                    )
                    for op in ops
                )
                state_read_terms += sum(
                    reads for _, reads, _ in state_io_entries
                )
                state_write_ops += sum(
                    writes for _, _, writes in state_io_entries
                )
                integer_state_write_ops += model_integer_state_write_ops

            def _flush_segment(model_indices: List[int]) -> None:
                if not model_indices or rust_backend is None:
                    return
                combined_ops: List[LinearOp] = []
                combined_sync_entries = []
                combined_state_io_entries = []
                state_values = None
                for idx in model_indices:
                    ops, sync_entries, model_state_values, state_io_entries, _ = per_model_ops[idx]
                    combined_ops.extend(ops)
                    combined_sync_entries.extend(sync_entries)
                    combined_state_io_entries.extend(state_io_entries)
                    if model_state_values is not None:
                        state_values = model_state_values
                rust_static_eval_segments_by_start[model_indices[0]] = (
                    tuple(model_indices),
                    rust_backend.make_static_linear_batch(combined_ops),
                    tuple(combined_sync_entries),
                    state_values,
                    tuple(combined_state_io_entries),
                )
                rust_static_eval_segment_members.update(model_indices)

            current_segment: List[int] = []
            for idx in sorted(per_model_ops):
                model_uses_state = per_model_ops[idx][4]
                if model_uses_state:
                    _flush_segment(current_segment)
                    current_segment = []
                    _flush_segment([idx])
                    continue
                if current_segment and idx != current_segment[-1] + 1:
                    _flush_segment(current_segment)
                    current_segment = []
                current_segment.append(idx)
            _flush_segment(current_segment)

            full_static_coverage = bool(self.models) and len(
                rust_static_eval_segment_members
            ) == len(self.models)
            mixed_small_gate = (
                bool(rust_static_fast_sync)
                and not rust_required
                and planned_models > 0
                and not full_static_coverage
                and planned_ops < 64
            )
            if mixed_small_gate:
                self._perf_stats["rust_static_eval_mixed_small_fallbacks"] = 1
                self._perf_stats["rust_static_eval_gated_models"] = planned_models
                self._perf_stats["rust_static_eval_gated_ops"] = planned_ops
                self._perf_stats["rust_static_eval_gated_segments"] = len(
                    rust_static_eval_segments_by_start
                )
                fallback_models += planned_models
                planned_models = 0
                planned_ops = 0
                planned_terms = 0
                state_read_terms = 0
                state_write_ops = 0
                integer_state_write_ops = 0
                rust_static_eval_segments_by_start = {}
                rust_static_eval_segment_members = set()

            self._perf_stats["rust_static_eval_candidate_models"] = candidates
            self._perf_stats["rust_static_eval_models"] = planned_models
            self._perf_stats["rust_static_eval_ops"] = planned_ops
            self._perf_stats["rust_static_eval_fallback_models"] = fallback_models
            self._perf_stats["rust_static_eval_terms"] = planned_terms
            self._perf_stats["rust_static_eval_state_read_terms"] = state_read_terms
            self._perf_stats["rust_static_eval_state_write_ops"] = state_write_ops
            self._perf_stats["rust_static_eval_integer_state_write_ops"] = (
                integer_state_write_ops
            )
            self._perf_stats["rust_static_eval_segments"] = len(
                rust_static_eval_segments_by_start
            )
            self._perf_stats["rust_static_eval_max_segment_models"] = max(
                (len(segment[0]) for segment in rust_static_eval_segments_by_start.values()),
                default=0,
            )

        def _build_rust_transition_shadow_plans():
            nonlocal rust_transition_shadow_plans
            rust_transition_shadow_plans = {}
            if not rust_transition_shadow or indexed_array is None:
                return

            candidates = 0
            planned_models = 0
            planned_static_ops = 0
            planned_target_ops = 0

            for model_index, model in enumerate(self.models):
                raw_segment = getattr(
                    model.__class__,
                    "_ordered_transition_segment_ir_ops",
                    ((), ()),
                ) or ((), ())
                try:
                    raw_linear, raw_transition = raw_segment
                except (TypeError, ValueError):
                    continue
                raw_linear = tuple(raw_linear or ())
                raw_transition = tuple(raw_transition or ())
                if not raw_transition:
                    continue
                candidates += 1
                if rust_backend is None or getattr(model, "_child_models", []):
                    continue

                try:
                    linear_ir_ops = normalize_linear_ops(raw_linear)
                    transition_ir_ops = normalize_transition_target_ops(raw_transition)
                except (TypeError, ValueError):
                    continue

                state_values = getattr(model, "_indexed_state_values", None)
                state_ids = getattr(model, "_indexed_state_ids", {}) or {}

                def _eval_scalar(value):
                    try:
                        return model._evaluate_rust_static_affine_scalar(
                            value,
                            model.params,
                        )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        return None

                def _convert_terms(ir_terms):
                    converted_terms: List[LinearTerm] = []
                    for term in ir_terms:
                        gain_value = _eval_scalar(term.gain)
                        if gain_value is None:
                            return None
                        if term.source_kind == SOURCE_NODE:
                            external = model._resolve_external_node(term.source_name)
                            indexed_array.ensure_nodes((external,))
                            source_id = indexed_array.node_index.id_of(external)
                            converted_terms.append(
                                LinearTerm(
                                    source_kind=SOURCE_NODE,
                                    source_id=source_id,
                                    gain=float(gain_value),
                                )
                            )
                        elif term.source_kind == SOURCE_STATE:
                            if state_values is None:
                                return None
                            source_id = state_ids.get(term.source_name)
                            if source_id is None:
                                return None
                            converted_terms.append(
                                LinearTerm(
                                    source_kind=SOURCE_STATE,
                                    source_id=source_id,
                                    gain=float(gain_value),
                                )
                            )
                        else:
                            return None
                    return tuple(converted_terms)

                def _convert_condition(ir_condition):
                    if ir_condition is None:
                        return None
                    left_bias = _eval_scalar(ir_condition.left_bias)
                    right_bias = _eval_scalar(ir_condition.right_bias)
                    if left_bias is None or right_bias is None:
                        return False
                    left_terms = _convert_terms(ir_condition.left_terms)
                    right_terms = _convert_terms(ir_condition.right_terms)
                    if left_terms is None or right_terms is None:
                        return False
                    return LinearCondition(
                        op_kind=ir_condition.op_kind,
                        left_bias=float(left_bias),
                        left_terms=left_terms,
                        right_bias=float(right_bias),
                        right_terms=right_terms,
                    )

                linear_ops: List[LinearOp] = []
                transition_ops: List[TransitionTargetOp] = []
                transition_keys: List[Optional[str]] = []
                failed = False

                for ir_op in linear_ir_ops:
                    bias_value = _eval_scalar(ir_op.bias)
                    false_bias_value = _eval_scalar(ir_op.false_bias)
                    if bias_value is None or false_bias_value is None:
                        failed = True
                        break
                    terms = _convert_terms(ir_op.terms)
                    false_terms = _convert_terms(ir_op.false_terms)
                    if terms is None or false_terms is None:
                        failed = True
                        break
                    condition = _convert_condition(ir_op.condition)
                    if condition is False:
                        failed = True
                        break
                    if ir_op.target_kind == TARGET_NODE:
                        external = model._resolve_external_node(ir_op.target_name)
                        indexed_array.ensure_nodes((external,))
                        target_id = indexed_array.node_index.id_of(external)
                    elif ir_op.target_kind == TARGET_STATE:
                        if state_values is None:
                            failed = True
                            break
                        target_id = state_ids.get(ir_op.target_name)
                        if target_id is None:
                            failed = True
                            break
                    else:
                        failed = True
                        break
                    linear_ops.append(
                        LinearOp(
                            target_kind=ir_op.target_kind,
                            target_id=target_id,
                            bias=float(bias_value),
                            terms=terms,
                            condition=condition,
                            false_bias=float(false_bias_value),
                            false_terms=false_terms,
                            target_integer=bool(
                                getattr(ir_op, "target_integer", False)
                            ),
                        )
                    )

                if failed:
                    continue

                for target_id, ir_op in enumerate(transition_ir_ops):
                    bias_value = _eval_scalar(ir_op.bias)
                    false_bias_value = _eval_scalar(ir_op.false_bias)
                    delay_value = _eval_scalar(ir_op.delay)
                    rise_value = _eval_scalar(ir_op.rise)
                    fall_value = _eval_scalar(ir_op.fall)
                    if (
                        bias_value is None
                        or false_bias_value is None
                        or delay_value is None
                        or rise_value is None
                        or fall_value is None
                    ):
                        failed = True
                        break
                    terms = _convert_terms(ir_op.terms)
                    false_terms = _convert_terms(ir_op.false_terms)
                    if terms is None or false_terms is None:
                        failed = True
                        break
                    condition = _convert_condition(ir_op.condition)
                    if condition is False:
                        failed = True
                        break
                    transition_ops.append(
                        TransitionTargetOp(
                            target_id=target_id,
                            bias=float(bias_value),
                            terms=terms,
                            condition=condition,
                            false_bias=float(false_bias_value),
                            false_terms=false_terms,
                            delay=float(delay_value),
                            rise=float(rise_value),
                            fall=float(fall_value),
                        )
                    )
                    transition_keys.append(ir_op.transition_key)

                if failed or not transition_ops:
                    continue

                rust_transition_shadow_plans[model_index] = (
                    model,
                    rust_backend.make_static_linear_batch(linear_ops),
                    rust_backend.make_transition_target_batch(transition_ops),
                    state_values,
                    tuple(transition_keys),
                )
                planned_models += 1
                planned_static_ops += len(linear_ops)
                planned_target_ops += len(transition_ops)

            self._perf_stats["rust_transition_shadow_candidate_models"] = candidates
            self._perf_stats["rust_transition_shadow_models"] = planned_models
            self._perf_stats["rust_transition_shadow_static_ops"] = planned_static_ops
            self._perf_stats["rust_transition_shadow_target_ops"] = planned_target_ops
            self._perf_stats["rust_transition_shadow_segments"] = len(
                rust_transition_shadow_plans
            )

        def _sync_rust_static_outputs(
            sync_entries: Tuple[tuple, ...],
            *,
            sync_node_voltages: bool,
            sync_output_nodes: bool,
        ) -> None:
            if indexed_array is None:
                return
            for model, local_node, external_node, write_node_id in sync_entries:
                value = float(indexed_array.values[write_node_id])
                if sync_output_nodes:
                    if local_node not in model.output_nodes:
                        model._output_nodes_version += 1
                    model.output_nodes[local_node] = value
                    audit_write = getattr(model, "_event_trace_audit_note_write", None)
                    if audit_write is not None:
                        audit_write("output", local_node)
                    self._perf_stats["rust_static_eval_output_syncs"] += 1
                else:
                    self._perf_stats["rust_static_eval_deferred_output_syncs"] += 1
                if sync_node_voltages:
                    self.node_voltages[external_node] = value
                    _mark_indexed_dirty_node(external_node)
                    self._perf_stats["rust_static_eval_node_voltage_syncs"] += 1

        def _snapshot_rust_transition_states(model_index: int):
            plan = rust_transition_shadow_plans.get(model_index)
            if plan is None:
                return None
            model = plan[0]
            transition_keys = plan[4]
            snapshot = []
            for key in transition_keys:
                if not key:
                    snapshot.append(None)
                    continue
                transition_state = model.transitions.get(key)
                if transition_state is None:
                    snapshot.append(None)
                    continue
                snapshot.append(
                    (
                        float(transition_state.current_val),
                        float(transition_state.target_val),
                        float(transition_state.start_time),
                        float(transition_state.start_val),
                        float(transition_state.delay),
                        float(transition_state.rise_time),
                        float(transition_state.fall_time),
                        1 if transition_state.active else 0,
                    )
                )
            return tuple(snapshot)

        def _run_rust_transition_shadow(
            model_index: int,
            pre_transition_snapshot=None,
        ) -> None:
            if indexed_array is None or rust_backend is None:
                return
            plan = rust_transition_shadow_plans.get(model_index)
            if plan is None:
                return
            (
                model,
                linear_batch,
                transition_batch,
                state_values,
                transition_keys,
            ) = plan
            target_count = len(transition_keys)
            if target_count == 0:
                return

            node_snapshot = array(
                "d",
                (float(value) for value in indexed_array.values),
            )
            state_snapshot = array(
                "d",
                (float(value) for value in (state_values or ())),
            )
            target_values = array("d", [0.0] * target_count)
            delay_values = array("d", [0.0] * target_count)
            rise_values = array("d", [0.0] * target_count)
            fall_values = array("d", [0.0] * target_count)
            try:
                rust_backend.evaluate_ordered_transition_segment(
                    linear_batch,
                    transition_batch,
                    node_snapshot,
                    state_snapshot,
                    target_values,
                    delay_values,
                    rise_values,
                    fall_values,
                )
            except RustBackendError:
                self._perf_stats["rust_transition_shadow_errors"] += 1
                return

            self._perf_stats["rust_transition_shadow_calls"] += 1
            default_transition = float(
                getattr(model, "default_transition", 1e-12) or 1e-12
            )
            for idx, key in enumerate(transition_keys):
                if not key:
                    self._perf_stats["rust_transition_shadow_skips"] += 1
                    continue
                transition_state = model.transitions.get(key)
                if transition_state is None:
                    self._perf_stats["rust_transition_shadow_skips"] += 1
                    continue
                rise = float(rise_values[idx])
                fall = float(fall_values[idx])
                effective_rise = rise if rise > 0.0 else default_transition
                effective_fall = fall if fall > 0.0 else default_transition
                max_diff = max(
                    abs(float(target_values[idx]) - float(transition_state.target_val)),
                    abs(float(delay_values[idx]) - float(transition_state.delay)),
                    abs(effective_rise - float(transition_state.rise_time)),
                    abs(effective_fall - float(transition_state.fall_time)),
                )
                if max_diff > self._perf_stats["rust_transition_shadow_max_abs_diff"]:
                    self._perf_stats["rust_transition_shadow_max_abs_diff"] = max_diff
                if max_diff <= 1e-12:
                    self._perf_stats["rust_transition_shadow_matches"] += 1
                else:
                    self._perf_stats["rust_transition_shadow_mismatches"] += 1

            if pre_transition_snapshot is None:
                return

            current_values = array("d")
            target_state_values = array("d")
            start_times = array("d")
            start_values = array("d")
            state_delays = array("d")
            state_rise_times = array("d")
            state_fall_times = array("d")
            active_flags: List[int] = []
            initialized_flags: List[int] = []
            comparable_indices: List[int] = []
            for idx, snapshot in enumerate(pre_transition_snapshot):
                if idx >= target_count:
                    break
                key = transition_keys[idx]
                if not key:
                    continue
                comparable_indices.append(idx)
                if snapshot is None:
                    current_values.append(0.0)
                    target_state_values.append(0.0)
                    start_times.append(0.0)
                    start_values.append(0.0)
                    state_delays.append(0.0)
                    state_rise_times.append(default_transition)
                    state_fall_times.append(default_transition)
                    active_flags.append(0)
                    initialized_flags.append(0)
                else:
                    (
                        current_value,
                        target_state_value,
                        start_time,
                        start_value,
                        state_delay,
                        state_rise,
                        state_fall,
                        active_flag,
                    ) = snapshot
                    current_values.append(current_value)
                    target_state_values.append(target_state_value)
                    start_times.append(start_time)
                    start_values.append(start_value)
                    state_delays.append(state_delay)
                    state_rise_times.append(state_rise)
                    state_fall_times.append(state_fall)
                    active_flags.append(active_flag)
                    initialized_flags.append(1)
            if not comparable_indices:
                return

            shadow_targets = array("d", (float(target_values[idx]) for idx in comparable_indices))
            shadow_delays = array("d", (float(delay_values[idx]) for idx in comparable_indices))
            shadow_rises = array("d", (float(rise_values[idx]) for idx in comparable_indices))
            shadow_falls = array("d", (float(fall_values[idx]) for idx in comparable_indices))
            output_values = array("d", [0.0] * len(comparable_indices))
            try:
                rust_backend.transition_state_step(
                    current_values,
                    target_state_values,
                    start_times,
                    start_values,
                    state_delays,
                    state_rise_times,
                    state_fall_times,
                    active_flags,
                    initialized_flags,
                    shadow_targets,
                    shadow_delays,
                    shadow_rises,
                    shadow_falls,
                    output_values,
                    time,
                    default_transition,
                    bool(getattr(model, "_initial_condition_mode", False)),
                )
            except RustBackendError:
                self._perf_stats["rust_transition_shadow_errors"] += 1
                return

            for local_idx, transition_idx in enumerate(comparable_indices):
                key = transition_keys[transition_idx]
                transition_state = model.transitions.get(key)
                if transition_state is None:
                    self._perf_stats["rust_transition_shadow_state_mismatches"] += 1
                    continue
                state_diff = max(
                    abs(float(current_values[local_idx]) - float(transition_state.current_val)),
                    abs(float(target_state_values[local_idx]) - float(transition_state.target_val)),
                    abs(float(start_times[local_idx]) - float(transition_state.start_time)),
                    abs(float(start_values[local_idx]) - float(transition_state.start_val)),
                    abs(float(state_delays[local_idx]) - float(transition_state.delay)),
                    abs(float(state_rise_times[local_idx]) - float(transition_state.rise_time)),
                    abs(float(state_fall_times[local_idx]) - float(transition_state.fall_time)),
                    abs(int(active_flags[local_idx]) - int(bool(transition_state.active))),
                )
                if state_diff > self._perf_stats["rust_transition_shadow_state_max_abs_diff"]:
                    self._perf_stats["rust_transition_shadow_state_max_abs_diff"] = state_diff
                if state_diff <= 1e-12:
                    self._perf_stats["rust_transition_shadow_state_matches"] += 1
                else:
                    self._perf_stats["rust_transition_shadow_state_mismatches"] += 1

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

        def _mark_indexed_dirty_node(node: str) -> None:
            if (
                indexed_dirty_validation_enabled
                and not indexed_dirty_validation_nodes
                and node
            ):
                indexed_dirty_nodes.add(node)

        def _validate_indexed_array_mapping(
            voltages: Dict[str, float],
            *,
            force_full: bool = False,
        ) -> float:
            if indexed_array is None:
                return 0.0
            if indexed_dirty_validation_enabled and not force_full:
                dirty_names = (
                    indexed_dirty_validation_nodes
                    if indexed_dirty_validation_nodes
                    else tuple(indexed_dirty_nodes)
                )
                max_diff, max_node, checked = indexed_array.max_abs_diff_names(
                    voltages,
                    dirty_names,
                )
                self._perf_stats["indexed_array_dirty_syncs"] += 1
                self._perf_stats["indexed_array_dirty_nodes_checked"] += checked
                _record_indexed_array_diff(max_diff, max_node, checked)
                indexed_dirty_nodes.clear()
                return max_diff
            max_diff, max_node, checked = indexed_array.max_abs_diff_mapping(voltages)
            _record_indexed_array_diff(max_diff, max_node, checked)
            if indexed_dirty_validation_enabled:
                indexed_dirty_nodes.clear()
            return max_diff

        _install_rust_event_write_batches()

        if indexed_array is not None:
            _refresh_indexed_model_io_plan(force=True)
            _build_rust_static_eval_plans()
            _build_rust_transition_shadow_plans()
            if (
                rust_static_eval
                and not indexed_arrays_requested
                and not rust_static_eval_segments_by_start
            ):
                indexed_array = None
                indexed_model_io_plan = None
                indexed_model_io_versions = None
                indexed_dirty_validation_nodes = ()
                indexed_dirty_validation_enabled = False
                self._indexed_array_stats = {}
                self._indexed_model_io_stats = {}
                self._indexed_voltage_probe_stats = {}
                self._indexed_voltage_read_stats = {}
                self._perf_stats["rust_static_eval_no_segment_fallbacks"] = 1
                if not (
                    profile_sections
                    or profile_model_eval
                    or indexed_snapshot_profile
                    or indexed_arrays_requested
                ):
                    profile_clock = None
                    model_profile_enabled = False
                _set_model_indexed_output_writer(
                    _record_model_io_output_write if model_io_profile_enabled else None
                )
                _set_model_indexed_voltage_probe(
                    _record_model_io_voltage_read if model_io_profile_enabled else None
                )
                _set_model_indexed_voltage_reader(None)
                _set_model_static_branch_indexed_io_empty()
                if not indexed_state_storage_requested:
                    _set_model_indexed_state_storage_empty()
                    self._perf_stats["indexed_state_storage_enabled"] = 0
                    self._perf_stats["indexed_state_storage_models"] = 0
                    self._perf_stats["indexed_state_storage_scalar_slots"] = 0
                    self._perf_stats["indexed_state_storage_integer_slots"] = 0
                    self._perf_stats["indexed_state_storage_array_slots"] = 0

        if indexed_array is not None:
            rust_static_fast_sync_enabled = bool(
                rust_static_fast_sync
                and rust_static_eval
                and self.models
                and len(rust_static_eval_segment_members) == len(self.models)
            )
            indexed_dirty_validation_enabled = bool(
                rust_static_eval
                and not rust_static_fast_sync_enabled
                and self.models
                and len(rust_static_eval_segment_members) == len(self.models)
            )
            if indexed_dirty_validation_enabled:
                dirty_names = set(source_nodes)
                for segment in rust_static_eval_segments_by_start.values():
                    sync_entries = segment[2]
                    for _, _, external_node, _ in sync_entries:
                        dirty_names.add(external_node)
                indexed_dirty_validation_nodes = tuple(sorted(dirty_names))
            self._perf_stats["indexed_array_dirty_validation_enabled"] = int(
                indexed_dirty_validation_enabled
            )
            self._perf_stats["rust_static_fast_sync_enabled"] = int(
                rust_static_fast_sync_enabled
            )

            def _indexed_output_write_through(node: str, value: float):
                _record_model_io_output_write(node, value)
                indexed_array.set(node, value)
                _mark_indexed_dirty_node(node)
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
                _record_model_io_voltage_read(
                    local_node,
                    external_node,
                    dict_value,
                    event_context,
                )
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

            def _indexed_voltage_read(local_node: str, external_node: str) -> Optional[float]:
                if not indexed_array.node_index.has(external_node):
                    self._perf_stats["indexed_voltage_read_fallbacks"] += 1
                    return None
                indexed_voltage_read_nodes_seen.add(external_node)
                self._perf_stats["indexed_voltage_reads"] += 1
                self._perf_stats["indexed_voltage_read_nodes"] = len(
                    indexed_voltage_read_nodes_seen
                )
                value = indexed_array.get(external_node, 0.0)
                _record_model_io_voltage_read(local_node, external_node, value, False)
                return value

            _set_model_indexed_output_writer(_indexed_output_write_through)
            _set_model_indexed_voltage_probe(_indexed_voltage_probe)
            _set_model_indexed_voltage_reader(_indexed_voltage_read)
            _validate_indexed_array_mapping(self.node_voltages, force_full=True)

        # Record initial state
        initial_record_reads = self._record_point(
            0.0,
            indexed_array,
            indexed_record_node_ids,
            rust_backend=rust_backend if rust_array_loop_enabled else None,
            rust_record_node_batch=rust_record_node_batch,
        )
        if indexed_array is not None:
            self._perf_stats["indexed_array_record_reads"] += initial_record_reads
            if indexed_record_node_ids is not None:
                self._perf_stats["indexed_array_record_id_reads"] += initial_record_reads
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
            self._perf_stats["source_breakpoint_scan_calls"] += len(source_breakpoint_sources)
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
            self._perf_stats["model_breakpoint_scan_calls"] += len(breakpoint_models)
            for model in breakpoint_models:
                bp = model.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    force_record_point = True
                    self._perf_stats["model_breakpoint_clamps"] += 1
                    if dt < 1e-18:
                        dt = 1e-18
            bp = _rust_event_transition_next_breakpoint(time)
            if bp is not None and bp > time and bp < time + dt:
                dt = bp - time
                force_record_point = True
                self._perf_stats["model_breakpoint_clamps"] += 1
                self._perf_stats[
                    "rust_event_transition_production_breakpoint_clamps_total"
                ] += 1
                if dt < 1e-18:
                    dt = 1e-18
            if profile_clock is not None:
                _add_profile_time("model_breakpoint_scan_s", _section_start)

            _section_start = profile_clock() if profile_clock is not None else 0.0
            self._perf_stats["transition_target_breakpoint_scan_calls"] += len(
                transition_target_breakpoint_models
            )
            for model in transition_target_breakpoint_models:
                bp = model.transition_target_breakpoint(self.node_voltages, time, dt)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    force_record_point = True
                    self._perf_stats["transition_target_breakpoint_clamps"] += 1
                    if dt < 1e-18:
                        dt = 1e-18
            if profile_clock is not None:
                _add_profile_time("transition_target_breakpoint_scan_s", _section_start)

            # Respect $bound_step from models
            _section_start = profile_clock() if profile_clock is not None else 0.0
            self._perf_stats["bound_step_scan_calls"] += len(bound_step_models)
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
                remaining = tstop - time
                if remaining > 0.0 and remaining < min_step:
                    dt = remaining
                else:
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
                if (
                    rust_array_loop_enabled
                    and rust_backend is not None
                    and isinstance(indexed_array.values, array)
                ):
                    if (
                        rust_array_prev_values is None
                        or len(rust_array_prev_values) != len(indexed_array.values)
                    ):
                        rust_array_prev_values = array(
                            "d",
                            [0.0] * len(indexed_array.values),
                        )
                    try:
                        rust_backend.copy_f64(
                            indexed_array.values,
                            rust_array_prev_values,
                        )
                        prev_indexed_values = rust_array_prev_values
                        self._perf_stats["rust_array_snapshot_copies"] += 1
                    except RustBackendError:
                        prev_indexed_values = indexed_array.snapshot()
                        self._perf_stats["rust_array_snapshot_fallbacks"] += 1
                else:
                    prev_indexed_values = indexed_array.snapshot()
                self._perf_stats["indexed_array_snapshots"] += 1
                if indexed_dirty_validation_enabled:
                    self._perf_stats["indexed_array_prev_snapshot_dirty_skips"] += 1
                    _refresh_indexed_array_stats()
                else:
                    _validate_indexed_array_mapping(prev_nv)
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
                    _mark_indexed_dirty_node(src.node)
                    self._perf_stats["indexed_array_source_updates"] += 1
            _set_current_source_values(self.node_voltages, time)
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
            for model_index, (model, model_needs_future, has_post_update_events) in enumerate(
                zip(self.models, model_needs_future_node_voltages, model_has_post_update_events)
            ):
                if (
                    model_index in rust_static_eval_segment_members
                    and model_index not in rust_static_eval_segments_by_start
                ):
                    continue
                rust_segment = rust_static_eval_segments_by_start.get(model_index)
                if rust_segment is not None and indexed_array is not None and rust_backend is not None:
                    (
                        segment_model_indices,
                        batch,
                        sync_entries,
                        state_values,
                        state_io_entries,
                    ) = rust_segment
                    segment_models = [self.models[idx] for idx in segment_model_indices]
                    _section_start = profile_clock() if profile_clock is not None else 0.0
                    rust_segment_succeeded = False
                    try:
                        rust_backend.evaluate_static_linear(
                            batch,
                            indexed_array.values,
                            state_values,
                        )
                        if rust_static_fast_sync_enabled:
                            self._perf_stats[
                                "rust_static_fast_sync_node_voltage_sync_skips"
                            ] += len(sync_entries)
                            self._perf_stats[
                                "rust_static_eval_deferred_output_syncs"
                            ] += len(sync_entries)
                        else:
                            _sync_rust_static_outputs(
                                sync_entries,
                                sync_node_voltages=True,
                                sync_output_nodes=False,
                            )
                        for state_model, reads, writes in state_io_entries:
                            state_model._perf_stats["indexed_state_scalar_reads"] += int(reads)
                            state_model._perf_stats["indexed_state_scalar_writes"] += int(writes)
                        self._perf_stats["rust_static_eval_calls"] += 1
                        rust_segment_succeeded = True
                    except RustBackendError:
                        self._perf_stats["rust_static_eval_errors"] += 1

                    if rust_segment_succeeded:
                        self._perf_stats["model_post_update_skips"] += len(segment_models)
                        self._perf_stats["rust_static_eval_lifecycle_model_skips"] += len(
                            segment_models
                        )
                        if profile_clock is not None:
                            elapsed = _add_profile_time(
                                "rust_static_eval_s",
                                _section_start,
                            )
                            per_model_elapsed = elapsed / max(1, len(segment_models))
                            for segment_model_index, segment_model in zip(
                                segment_model_indices,
                                segment_models,
                            ):
                                _add_model_profile_time(
                                    segment_model_index,
                                    segment_model,
                                    "evaluate_s",
                                    per_model_elapsed,
                                )
                                _add_model_profile_time(
                                    segment_model_index,
                                    segment_model,
                                    "evaluate_calls",
                                    1.0,
                                )
                    else:
                        for segment_model_index, segment_model in zip(
                            segment_model_indices,
                            segment_models,
                        ):
                            _section_start = profile_clock() if profile_clock is not None else 0.0
                            segment_model_needs_future = model_needs_future_node_voltages[
                                segment_model_index
                            ]
                            segment_model_future_nv = (
                                future_nv
                                if future_nv is not None
                                and segment_model_needs_future
                                else None
                            )
                            if model_needs_step_context[segment_model_index]:
                                segment_model._prepare_step(
                                    prev_nv,
                                    self.node_voltages,
                                    prev_time,
                                    time,
                                    segment_model_future_nv,
                                )
                                self._perf_stats["model_prepare_step_calls"] += 1
                                if profile_clock is not None:
                                    elapsed = _add_profile_time(
                                        "model_prepare_step_s",
                                        _section_start,
                                    )
                                    _add_model_profile_time(
                                        segment_model_index,
                                        segment_model,
                                        "prepare_step_s",
                                        elapsed,
                                    )
                            else:
                                self._perf_stats["model_prepare_step_skips"] += 1
                            segment_model._event_time = time
                            segment_model._bound_step = 0.0
                            _section_start = profile_clock() if profile_clock is not None else 0.0
                            transition_snapshot = _snapshot_rust_transition_states(
                                segment_model_index
                            )
                            _evaluate_model_with_optional_body_ir(
                                segment_model_index,
                                segment_model,
                                time,
                            )
                            _run_rust_transition_shadow(
                                segment_model_index,
                                transition_snapshot,
                            )
                            if profile_clock is not None:
                                elapsed = _add_profile_time(
                                    "model_evaluate_s",
                                    _section_start,
                                )
                                _add_model_profile_time(
                                    segment_model_index,
                                    segment_model,
                                    "evaluate_s",
                                    elapsed,
                                )
                                _add_model_profile_time(
                                    segment_model_index,
                                    segment_model,
                                    "evaluate_calls",
                                    1.0,
                                )
                        indexed_array.update_from_mapping(self.node_voltages)
                        self._perf_stats["indexed_post_model_sync_repairs"] += 1
                        _refresh_indexed_array_stats()

                        for segment_model_index, segment_model in zip(
                            segment_model_indices,
                            segment_models,
                        ):
                            _section_start = profile_clock() if profile_clock is not None else 0.0
                            segment_has_post_update_events = model_has_post_update_events[
                                segment_model_index
                            ]
                            if model_needs_timer_expire[segment_model_index]:
                                segment_model._expire_absolute_timers(time)
                                self._perf_stats["model_timer_expire_calls"] += 1
                            else:
                                self._perf_stats["model_timer_expire_skips"] += 1
                            if segment_has_post_update_events:
                                self._perf_stats["model_post_update_calls"] += 1
                                if segment_model.post_update_events(self.node_voltages, time):
                                    segment_model.refresh_outputs(self.node_voltages, time)
                            else:
                                self._perf_stats["model_post_update_skips"] += 1
                            if profile_clock is not None:
                                elapsed = _add_profile_time(
                                    "model_post_update_s",
                                    _section_start,
                                )
                                _add_model_profile_time(
                                    segment_model_index,
                                    segment_model,
                                    "post_update_s",
                                    elapsed,
                                )
                            if getattr(segment_model, "_step_event_fired", False):
                                cross_fired = True
                    continue

                _section_start = profile_clock() if profile_clock is not None else 0.0
                model_future_nv = (
                    future_nv
                    if future_nv is not None
                    and model_needs_future
                    else None
                )
                if model_needs_step_context[model_index]:
                    model._prepare_step(prev_nv, self.node_voltages, prev_time, time, model_future_nv)
                    self._perf_stats["model_prepare_step_calls"] += 1
                    if profile_clock is not None:
                        elapsed = _add_profile_time("model_prepare_step_s", _section_start)
                        _add_model_profile_time(model_index, model, "prepare_step_s", elapsed)
                else:
                    self._perf_stats["model_prepare_step_skips"] += 1
                _section_start = profile_clock() if profile_clock is not None else 0.0
                transition_snapshot = _snapshot_rust_transition_states(model_index)
                _evaluate_model_with_optional_body_ir(model_index, model, time)
                _run_rust_transition_shadow(model_index, transition_snapshot)
                if profile_clock is not None:
                    elapsed = _add_profile_time("model_evaluate_s", _section_start)
                    _add_model_profile_time(model_index, model, "evaluate_s", elapsed)
                    _add_model_profile_time(model_index, model, "evaluate_calls", 1.0)
                _section_start = profile_clock() if profile_clock is not None else 0.0
                if model_needs_timer_expire[model_index]:
                    model._expire_absolute_timers(time)
                    self._perf_stats["model_timer_expire_calls"] += 1
                else:
                    self._perf_stats["model_timer_expire_skips"] += 1
                if has_post_update_events:
                    self._perf_stats["model_post_update_calls"] += 1
                    if model.post_update_events(self.node_voltages, time):
                        model.refresh_outputs(self.node_voltages, time)
                else:
                    self._perf_stats["model_post_update_skips"] += 1
                if profile_clock is not None:
                    elapsed = _add_profile_time("model_post_update_s", _section_start)
                    _add_model_profile_time(model_index, model, "post_update_s", elapsed)
                if getattr(model, "_step_event_fired", False):
                    cross_fired = True

            if indexed_array is not None:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                if rust_static_fast_sync_enabled:
                    self._perf_stats["rust_static_fast_sync_validation_skips"] += 1
                    _refresh_indexed_array_stats()
                else:
                    _refresh_indexed_model_io_plan()
                    self._perf_stats["indexed_array_syncs"] += 1
                    max_diff = _validate_indexed_array_mapping(self.node_voltages)
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
            err_ratio_from_source = False
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
                err_ratio_node_id_batch = None
                if (
                    rust_array_loop_enabled
                    and indexed_array is not None
                    and rust_backend is not None
                ):
                    node_ids = []
                    for node in err_ratio_nodes:
                        if not indexed_array.node_index.has(node):
                            node_ids = []
                            break
                        node_ids.append(indexed_array.node_index.id_of(node))
                    if len(node_ids) == len(err_ratio_nodes):
                        err_ratio_node_id_batch = rust_backend.make_node_id_batch(
                            node_ids
                        )
            if err_ratio_skipped_outputs_per_step:
                self._perf_stats["err_ratio_skipped_outputs"] += err_ratio_skipped_outputs_per_step
            if err_ratio_skipped_sources_per_step:
                self._perf_stats["err_ratio_skipped_sources"] += err_ratio_skipped_sources_per_step
            used_rust_err_ratio = False
            if (
                indexed_array is not None
                and rust_array_loop_enabled
                and rust_backend is not None
                and err_ratio_node_id_batch is not None
                and prev_indexed_values is not None
                and len(prev_indexed_values) == len(indexed_array.values)
            ):
                try:
                    err_ratio = rust_backend.max_err_ratio(
                        indexed_array.values,
                        prev_indexed_values,
                        err_ratio_node_id_batch,
                        reltol,
                        vabstol,
                    )
                    self._perf_stats["rust_array_err_ratio_scans"] += 1
                    self._perf_stats["rust_array_err_ratio_reads"] += len(
                        err_ratio_node_id_batch
                    )
                    self._perf_stats["indexed_array_err_ratio_reads"] += len(
                        err_ratio_node_id_batch
                    )
                    err_ratio_from_source = any(
                        node in source_nodes for node in err_ratio_nodes
                    )
                    used_rust_err_ratio = True
                except RustBackendError:
                    self._perf_stats["rust_array_err_ratio_fallbacks"] += 1
            if not used_rust_err_ratio:
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
                            err_ratio_from_source = node in source_nodes
            if indexed_array is not None:
                _refresh_indexed_array_stats()
            if profile_clock is not None:
                _add_profile_time("err_ratio_node_scan_s", _section_start)
            if err_ratio > 1.0:
                scale = min(4.0, max(1.2, math.sqrt(err_ratio)))
                next_dynamic_step = dynamic_step / scale
                adaptive_floor_allowed = not (
                    err_ratio_from_source or adaptive_step_floor_model_sensitive
                )
                shrink_floor = adaptive_step_floor if adaptive_floor_allowed else min_step
                if (
                    err_ratio_from_source
                    and adaptive_step_floor > min_step
                    and next_dynamic_step < adaptive_step_floor
                ):
                    self._perf_stats["adaptive_step_floor_source_bypasses"] += 1
                if (
                    adaptive_step_floor_model_sensitive
                    and not err_ratio_from_source
                    and adaptive_step_floor > min_step
                    and next_dynamic_step < adaptive_step_floor
                ):
                    self._perf_stats["adaptive_step_floor_model_bypasses"] += 1
                if next_dynamic_step < shrink_floor:
                    dynamic_step = shrink_floor
                    if adaptive_floor_allowed:
                        self._perf_stats["adaptive_step_floor_clamps"] += 1
                else:
                    dynamic_step = next_dynamic_step
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
                    or time >= tstop
                )

            if should_record:
                _section_start = profile_clock() if profile_clock is not None else 0.0
                record_reads = self._record_point(
                    time,
                    indexed_array,
                    indexed_record_node_ids,
                    rust_backend=rust_backend if rust_array_loop_enabled else None,
                    rust_record_node_batch=rust_record_node_batch,
                )
                if indexed_array is not None:
                    self._perf_stats["indexed_array_record_reads"] += record_reads
                    if indexed_record_node_ids is not None:
                        self._perf_stats["indexed_array_record_id_reads"] += record_reads
                    _refresh_indexed_array_stats()
                self._step_sizes.append(dt)
                if next_record_time is not None:
                    while next_record_time <= time + 1e-18:
                        next_record_time += record_step
                if profile_clock is not None:
                    _add_profile_time("record_point_s", _section_start)
            self._perf_stats["steps_total"] += 1

        if rust_static_eval_segments_by_start and indexed_array is not None:
            for segment in rust_static_eval_segments_by_start.values():
                sync_entries = segment[2]
                _sync_rust_static_outputs(
                    sync_entries,
                    sync_node_voltages=rust_static_fast_sync_enabled,
                    sync_output_nodes=True,
                )

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

        def _aggregate_model_perf_stats():
            keys = {
                "timer_breakpoint_cache_hits": "timer_breakpoint_cache_hits_total",
                "timer_breakpoint_hits": "timer_breakpoint_hits_total",
                "timer_breakpoint_scans": "timer_breakpoint_scans_total",
                "timer_state_updates": "timer_state_updates_total",
                "timer_batch_due_calls": "timer_batch_due_calls_total",
                "timer_batch_due_events": "timer_batch_due_events_total",
                "timer_batch_due_fires": "timer_batch_due_fires_total",
                "timer_batch_due_fallbacks": "timer_batch_due_fallbacks_total",
                "timer_state_owned_checks": "timer_state_owned_checks_total",
                "timer_state_owned_fast_skips": "timer_state_owned_fast_skips_total",
                "timer_state_owned_target_reads": "timer_state_owned_target_reads_total",
                "timer_state_owned_fires": "timer_state_owned_fires_total",
                "timer_state_owned_fallbacks": "timer_state_owned_fallbacks_total",
                "transition_unchanged_target_fastpath": "transition_unchanged_target_fastpath_total",
                "transition_calls": "transition_calls_total",
                "transition_output_fastpath_calls": "transition_output_fastpath_calls_total",
                "transition_set_target_calls": "transition_set_target_calls_total",
                "transition_evaluate_calls": "transition_evaluate_calls_total",
                "rust_transition_state_production_calls": "rust_transition_state_production_calls_total",
                "rust_transition_state_production_outputs": "rust_transition_state_production_outputs_total",
                "rust_transition_state_production_fallbacks": "rust_transition_state_production_fallbacks_total",
                "rust_transition_state_buffer_reuse_calls": "rust_transition_state_buffer_reuse_calls_total",
                "rust_transition_state_buffer_alloc_total": "rust_transition_state_buffer_alloc_grand_total",
                "rust_transition_batch_flushes": "rust_transition_batch_flushes_total",
                "rust_transition_batch_slot_total": "rust_transition_batch_slot_total_total",
                "rust_transition_batch_fallbacks": "rust_transition_batch_fallbacks_total",
                "rust_transition_batch_max_slots": "rust_transition_batch_max_slots_total",
                "rust_transition_lazy_enqueues": "rust_transition_lazy_enqueues_total",
                "rust_cross_production_calls": "rust_cross_production_calls_total",
                "rust_cross_production_fires": "rust_cross_production_fires_total",
                "rust_cross_production_fallbacks": "rust_cross_production_fallbacks_total",
                "rust_above_production_calls": "rust_above_production_calls_total",
                "rust_above_production_fires": "rust_above_production_fires_total",
                "rust_above_production_fallbacks": "rust_above_production_fallbacks_total",
                "rust_event_interpolation_batches": "rust_event_interpolation_batches_total",
                "rust_event_interpolation_nodes": "rust_event_interpolation_nodes_total",
                "rust_event_interpolation_cache_hits": "rust_event_interpolation_cache_hits_total",
                "rust_event_interpolation_fallbacks": "rust_event_interpolation_fallbacks_total",
                "transition_breakpoint_scans": "transition_breakpoint_scans_total",
                "transition_breakpoint_active_scans": "transition_breakpoint_active_scans_total",
                "transition_breakpoint_inactive_skips": "transition_breakpoint_inactive_skips_total",
                "rust_transition_breakpoint_scans": "rust_transition_breakpoint_scans_total",
                "rust_transition_breakpoint_state_scans": "rust_transition_breakpoint_state_scans_total",
                "rust_transition_breakpoint_fallbacks": "rust_transition_breakpoint_fallbacks_total",
                "rust_timer_breakpoint_scans": "rust_timer_breakpoint_scans_total",
                "rust_timer_breakpoint_state_scans": "rust_timer_breakpoint_state_scans_total",
                "rust_timer_breakpoint_fallbacks": "rust_timer_breakpoint_fallbacks_total",
                "rust_timer_event_production_periodic_calls": "rust_timer_event_production_periodic_calls_total",
                "rust_timer_event_production_absolute_calls": "rust_timer_event_production_absolute_calls_total",
                "rust_timer_event_production_fires": "rust_timer_event_production_fires_total",
                "rust_timer_event_production_skips": "rust_timer_event_production_skips_total",
                "rust_timer_event_production_expirations": "rust_timer_event_production_expirations_total",
                "rust_timer_event_production_fallbacks": "rust_timer_event_production_fallbacks_total",
                "rust_event_due_shadow_cross_checks": "rust_event_due_shadow_cross_checks_total",
                "rust_event_due_shadow_above_checks": "rust_event_due_shadow_above_checks_total",
                "rust_event_due_shadow_timer_periodic_checks": "rust_event_due_shadow_timer_periodic_checks_total",
                "rust_event_due_shadow_timer_absolute_checks": "rust_event_due_shadow_timer_absolute_checks_total",
                "rust_event_due_shadow_matches": "rust_event_due_shadow_matches_total",
                "rust_event_due_shadow_mismatches": "rust_event_due_shadow_mismatches_total",
                "rust_event_due_shadow_errors": "rust_event_due_shadow_errors_total",
                "rust_event_write_shadow_checks": "rust_event_write_shadow_checks_total",
                "rust_event_write_shadow_matches": "rust_event_write_shadow_matches_total",
                "rust_event_write_shadow_mismatches": "rust_event_write_shadow_mismatches_total",
                "rust_event_write_shadow_errors": "rust_event_write_shadow_errors_total",
                "rust_event_write_production_calls": "rust_event_write_production_calls_total",
                "rust_event_write_production_executed": "rust_event_write_production_executed_total",
                "rust_event_write_production_fallbacks": "rust_event_write_production_fallbacks_total",
                "rust_event_linear_write_batches": "rust_event_linear_write_batches_total",
                "rust_event_linear_write_shadow_checks": "rust_event_linear_write_shadow_checks_total",
                "rust_event_linear_write_shadow_matches": "rust_event_linear_write_shadow_matches_total",
                "rust_event_linear_write_shadow_mismatches": "rust_event_linear_write_shadow_mismatches_total",
                "rust_event_linear_write_shadow_errors": "rust_event_linear_write_shadow_errors_total",
                "rust_event_linear_write_production_calls": "rust_event_linear_write_production_calls_total",
                "rust_event_linear_write_production_executed": "rust_event_linear_write_production_executed_total",
                "rust_event_linear_write_production_fallbacks": "rust_event_linear_write_production_fallbacks_total",
                "rust_body_ir_production_batches": "rust_body_ir_production_batches_total",
                "rust_body_ir_production_calls": "rust_body_ir_production_calls_total",
                "rust_body_ir_production_executed": "rust_body_ir_production_executed_total",
                "rust_body_ir_production_fallbacks": "rust_body_ir_production_fallbacks_total",
                "rust_body_ir_production_node_writes": "rust_body_ir_production_node_writes_total",
                "rust_body_ir_production_state_writes": "rust_body_ir_production_state_writes_total",
                "rust_timer_lfsr_output_batches": "rust_timer_lfsr_output_batches_total",
                "rust_timer_lfsr_output_calls": "rust_timer_lfsr_output_calls_total",
                "rust_timer_lfsr_output_due": "rust_timer_lfsr_output_due_total",
                "rust_timer_lfsr_output_skips": "rust_timer_lfsr_output_skips_total",
                "rust_timer_lfsr_output_executed": "rust_timer_lfsr_output_executed_total",
                "rust_timer_lfsr_output_writes": "rust_timer_lfsr_output_writes_total",
                "rust_timer_lfsr_output_fallbacks": "rust_timer_lfsr_output_fallbacks_total",
                "event_trace_audit_events": "event_trace_audit_events_total",
                "event_trace_audit_body_entries": "event_trace_audit_body_entries_total",
                "event_trace_audit_cross_events": "event_trace_audit_cross_events_total",
                "event_trace_audit_above_events": "event_trace_audit_above_events_total",
                "event_trace_audit_timer_events": "event_trace_audit_timer_events_total",
                "event_trace_audit_initial_step_events": "event_trace_audit_initial_step_events_total",
                "event_trace_audit_final_step_events": "event_trace_audit_final_step_events_total",
                "event_trace_audit_combined_events": "event_trace_audit_combined_events_total",
                "event_trace_audit_state_writes": "event_trace_audit_state_writes_total",
                "event_trace_audit_array_writes": "event_trace_audit_array_writes_total",
                "event_trace_audit_output_writes": "event_trace_audit_output_writes_total",
                "event_trace_audit_timer_state_writes": "event_trace_audit_timer_state_writes_total",
                "event_trace_audit_timer_last_fired_writes": "event_trace_audit_timer_last_fired_writes_total",
                "event_trace_audit_transition_writes": "event_trace_audit_transition_writes_total",
                "event_trace_audit_transition_output_writes": "event_trace_audit_transition_output_writes_total",
                "event_trace_audit_in_event_writes": "event_trace_audit_in_event_writes_total",
                "event_trace_audit_records_dropped": "event_trace_audit_records_dropped_total",
                "timer_array_sidecar_updates": "timer_array_sidecar_updates_total",
                "timer_array_sidecar_rebuilds": "timer_array_sidecar_rebuilds_total",
                "timer_array_sidecar_scans": "timer_array_sidecar_scans_total",
                "static_branch_fastpath_fallbacks": "static_branch_fastpath_fallbacks_total",
                "dynamic_node_cache_hits": "dynamic_node_cache_hits_total",
                "dynamic_node_cache_misses": "dynamic_node_cache_misses_total",
                "dynamic_node_cache_bypasses": "dynamic_node_cache_bypasses_total",
                "indexed_state_scalar_reads": "indexed_state_scalar_reads_total",
                "indexed_state_scalar_writes": "indexed_state_scalar_writes_total",
                "indexed_state_array_reads": "indexed_state_array_reads_total",
                "indexed_state_array_writes": "indexed_state_array_writes_total",
                "indexed_state_array_oob_writes": "indexed_state_array_oob_writes_total",
            }

            def _visit(model):
                perf = getattr(model, "_perf_stats", {}) or {}
                for source_key, dest_key in keys.items():
                    self._perf_stats[dest_key] += int(perf.get(source_key, 0) or 0)
                for source_key, value in perf.items():
                    if not (
                        source_key.startswith("event_trace_audit_body::")
                        or source_key.startswith("event_trace_audit_target::")
                    ):
                        continue
                    self._perf_stats[source_key] = int(
                        self._perf_stats.get(source_key, 0) or 0
                    ) + int(value or 0)
                self._perf_stats["rust_event_due_shadow_max_time_diff_total"] = max(
                    self._perf_stats["rust_event_due_shadow_max_time_diff_total"],
                    float(perf.get("rust_event_due_shadow_max_time_diff", 0.0) or 0.0),
                )
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

        _aggregate_model_perf_stats()

        def _aggregate_static_branch_fastpath_plan_stats():
            if not static_branch_fastpath:
                return

            models = 0
            read_nodes = 0
            write_nodes = 0

            def _visit(model):
                nonlocal models, read_nodes, write_nodes
                model_cls = getattr(model, "__class__", type(model))
                if bool(getattr(model_cls, "_static_branch_fastpath_codegen", False)):
                    models += 1
                    read_nodes += len(
                        getattr(model_cls, "_static_voltage_read_nodes", ()) or ()
                    )
                    write_nodes += len(
                        getattr(model_cls, "_static_output_write_nodes", ()) or ()
                    )
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

            self._perf_stats["static_branch_fastpath_codegen_models"] = models
            self._perf_stats["static_branch_fastpath_static_read_nodes"] = read_nodes
            self._perf_stats["static_branch_fastpath_static_write_nodes"] = write_nodes

        _aggregate_static_branch_fastpath_plan_stats()

        def _aggregate_node_resolution_cache_stats():
            entries = 0
            models_with_entries = 0

            def _visit(model):
                nonlocal entries, models_with_entries
                cache = getattr(model, "_node_resolution_cache", {}) or {}
                size = len(cache)
                if size:
                    models_with_entries += 1
                    entries += size
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

            self._perf_stats["node_resolution_cache_entries"] = entries
            self._perf_stats["node_resolution_cache_models"] = models_with_entries

        _aggregate_node_resolution_cache_stats()

        def _aggregate_dynamic_node_cache_stats():
            entries = 0
            models_with_entries = 0

            def _visit(model):
                nonlocal entries, models_with_entries
                cache = getattr(model, "_dynamic_node_cache", {}) or {}
                size = len(cache)
                if size:
                    models_with_entries += 1
                    entries += size
                for child in getattr(model, "_child_models", []) or []:
                    _visit(child)

            for model in self.models:
                _visit(model)

            self._perf_stats["dynamic_node_cache_entries"] = entries
            self._perf_stats["dynamic_node_cache_models"] = models_with_entries

        _aggregate_dynamic_node_cache_stats()
        _refresh_model_io_profile_stats()

        _set_model_indexed_output_writer(None)
        _set_model_indexed_voltage_probe(None)
        _set_model_indexed_voltage_reader(None)
        _set_model_indexed_state_storage_empty()
        _set_model_node_resolution_cache_enabled(False)
        _set_model_static_branch_fastpath_enabled(False)
        _set_model_transition_unchanged_fastpath_enabled(False)
        _set_model_rust_transition_breakpoint_scanner(None)
        _set_model_rust_transition_state_backend(None, production=False)
        _set_model_rust_event_interpolation_backend(None)
        _set_model_rust_timer_breakpoint_scanner(None)
        _set_model_rust_timer_event_backend(None, production=False)
        _set_model_rust_event_due_shadow_backend(None)
        _set_model_rust_event_write_backend(None)
        _set_model_rust_body_ir_backend(None, production=False)
        _set_model_event_trace_audit_enabled(False)
        _set_model_static_branch_indexed_io_empty()

        return SimResult(time=time_arr, signals=signals,
                         step_sizes=np.array(self._step_sizes))

    def _record_point(
        self,
        time: float,
        indexed_voltages: Optional[IndexedVoltageArray] = None,
        indexed_record_node_ids: Optional[Tuple[int, ...]] = None,
        *,
        rust_backend=None,
        rust_record_node_batch=None,
    ) -> int:
        replace_last = bool(
            self.time_points and abs(time - self.time_points[-1]) <= 1.0e-18
        )
        if not replace_last:
            self.time_points.append(time)
        reads = 0
        if indexed_voltages is not None and indexed_record_node_ids is not None:
            values = None
            if rust_backend is not None and rust_record_node_batch is not None:
                try:
                    values = rust_backend.record_values_for_ids(
                        indexed_voltages.values,
                        rust_record_node_batch,
                        default=0.0,
                    )
                    self._perf_stats["rust_array_record_scans"] += 1
                    self._perf_stats["rust_array_record_values"] += len(
                        indexed_record_node_ids
                    )
                except RustBackendError:
                    self._perf_stats["rust_array_record_fallbacks"] += 1
            if values is None:
                values = indexed_voltages.values_for_ids(indexed_record_node_ids, 0.0)
            for name, val in zip(self.recorded_signals, values):
                if replace_last and self.recorded_signals[name]:
                    self.recorded_signals[name][-1] = val
                else:
                    self.recorded_signals[name].append(val)
            return len(indexed_record_node_ids)

        for name in self.recorded_signals:
            if indexed_voltages is not None:
                val = indexed_voltages.get(name, 0.0)
                reads += 1
            else:
                val = self.node_voltages.get(name, 0.0)
            if replace_last and self.recorded_signals[name]:
                self.recorded_signals[name][-1] = val
            else:
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
    wfn._evas_waveform = {
        "kind": "pulse",
        "v_lo": float(v_lo),
        "v_hi": float(v_hi),
        "period": float(period),
        "duty": float(duty),
        "rise": float(rise),
        "fall": float(fall),
        "delay": float(delay),
        "width": float(width) if width is not None else 0.0,
        "has_width": width is not None,
        "one_shot": bool(one_shot),
    }
    return wfn


def dc(voltage):
    """Create a DC voltage waveform."""
    voltage = float(voltage)

    def wfn(_t):
        return voltage

    wfn._evas_waveform = {
        "kind": "dc",
        "voltage": voltage,
    }
    return wfn


def sine(offset, amplitude, freq, phase=0.0):
    """Create a sine waveform."""
    offset = float(offset)
    amplitude = float(amplitude)
    freq = float(freq)
    phase = float(phase)

    def wfn(t):
        return offset + amplitude * math.sin(2 * math.pi * freq * t + phase)

    wfn._evas_waveform = {
        "kind": "sine",
        "offset": offset,
        "amplitude": amplitude,
        "freq": freq,
        "phase": phase,
    }
    return wfn


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
    wfn._evas_waveform = {
        "kind": "pwl",
        "times": tuple(float(t) for t in times),
        "values": tuple(float(v) for v in values),
    }
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


def square(v_lo, v_hi, period, delay=0.0, rise=1e-12, fall=1e-12, width=None):
    """Create a square waveform.

    Mirrors Spectre ``vsource type=square`` semantics: a periodic waveform
    that sits at ``v_lo`` until ``delay``, ramps linearly to ``v_hi`` over
    ``rise``, holds at ``v_hi`` for ``width`` (the high plateau after the rise
    ramp), then ramps back to ``v_lo`` over ``fall``.  When ``width`` is
    omitted it defaults to roughly 50% duty (``period/2 - rise/2 - fall/2``);
    when ``period`` is non-positive the wave is a single one-shot pulse that
    stays at ``v_hi`` after the rise.

    The returned callable carries ``_next_breakpoint`` (the edge schedule) so
    that source-driven ``@cross`` events on a ``type=square`` node are detected
    at each edge, matching the behaviour already provided for ``pulse`` and
    ``pwl`` sources.  Without it the source-breakpoint scan in
    ``TransientSim.run`` would have no edges to align to and ``@cross`` would
    appear never to fire.
    """
    period = float(period)
    rise = float(rise)
    fall = float(fall)
    delay = float(delay)
    one_shot = period <= 0.0
    if width is not None:
        width = float(width)
    elif one_shot:
        width = float("inf")
    else:
        # ~50% duty: split the period around the rise/fall ramps.
        width = max(0.0, period / 2.0 - rise / 2.0 - fall / 2.0)
    fall_start = rise + width
    fall_end = fall_start + fall
    # Include edge interior breakpoints so source-driven @cross events are
    # scheduled in chronological order, same idiom as pulse().
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
    wfn._evas_waveform = {
        "kind": "square",
        "v_lo": float(v_lo),
        "v_hi": float(v_hi),
        "period": float(period),
        "delay": float(delay),
        "rise": float(rise),
        "fall": float(fall),
        "width": float(width) if np.isfinite(width) else 0.0,
        "has_width": width is not None,
        "one_shot": bool(one_shot),
    }
    return wfn
