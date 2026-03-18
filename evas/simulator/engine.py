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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


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

        if abs(target - self.target_val) > 1e-15 or not self.active:
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

    def next_breakpoint(self, time: float) -> Optional[float]:
        """Return next important time for this transition, or None."""
        if not self.active:
            return None
        t_begin = self.start_time + self.delay
        going_up = self.target_val > self.start_val
        ramp_time = self.rise_time if going_up else self.fall_time
        t_end = t_begin + ramp_time

        breakpoints = []
        if t_begin > time:
            breakpoints.append(t_begin)
        if t_end > time:
            breakpoints.append(t_end)
        return min(breakpoints) if breakpoints else None


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

    def check(self, time: float, val: float) -> bool:
        """Check if a zero crossing occurred. Returns True if triggered."""
        if not self.initialized:
            self.pprev_val = self.prev_val = val
            self.pprev_time = self.prev_time = time
            self.initialized = True
            self.last_triggered = False
            return False

        _tol = 1e-12  # guard against floating-point rounding at exact crossings
        triggered = False
        if self.direction >= 0 and self.prev_val < 0 and val >= -_tol:
            triggered = True  # Rising
        if self.direction <= 0 and self.prev_val > 0 and val <= _tol:
            if self.direction == 0 or self.direction == -1:
                triggered = True  # Falling

        if triggered:
            dv = val - self.prev_val
            frac = max(0.0, min(1.0, -self.prev_val / dv)) if abs(dv) > 1e-30 else 0.0
            self.t_cross = self.prev_time + frac * (time - self.prev_time)
            # Clamp val to the post-crossing side to prevent immediate re-trigger.
            # Falling: prev_val was >0, val near 0 — ensure stored val is <= 0.
            # Rising:  prev_val was <0, val near 0 — ensure stored val is >= 0.
            if self.prev_val > 0:
                val = min(val, 0.0)
            else:
                val = max(val, 0.0)

        self.pprev_val = self.prev_val
        self.pprev_time = self.prev_time
        self.prev_time = time
        self.prev_val = val
        self.last_triggered = triggered
        return triggered

    def next_breakpoint(self) -> Optional[float]:
        """Predict the next zero-crossing time by linear extrapolation."""
        dt = self.prev_time - self.pprev_time
        if dt <= 1e-30:
            return None
        rate = (self.prev_val - self.pprev_val) / dt
        if self.direction >= 0 and self.prev_val < 0 and rate > 0:
            return self.prev_time + (-self.prev_val / rate)
        if self.direction <= 0 and self.prev_val > 0 and rate < 0:
            return self.prev_time + (-self.prev_val / rate)
        return None

    def would_cross(self, val: float) -> bool:
        """Check if a crossing would occur without updating state."""
        if not self.initialized:
            return False
        if self.direction >= 0 and self.prev_val < 0 and val >= -1e-12:
            return True
        if self.direction <= 0 and self.prev_val > 0 and val <= 1e-12:
            if self.direction == 0 or self.direction == -1:
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
            self.last_triggered = False
            return False

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

    def next_breakpoint(self, time: float) -> Optional[float]:
        bpfn = getattr(self.waveform, '_next_breakpoint', None)
        if bpfn is None:
            return None
        return bpfn(time)


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
            refine_steps: int = 8) -> SimResult:
        """Run transient simulation with adaptive step control near cross events."""
        if tstep is None:
            tstep = tstop / 10000
        if max_step is None:
            max_step = tstep

        time = 0.0
        self.time_points = []
        self._step_sizes = []
        refine_steps_left = 0  # countdown of refined steps after cross
        refine_dt = tstep  # current refined step size

        # Initialize node voltages
        for src in self.sources:
            self.node_voltages[src.node] = src.waveform(0.0)

        # Fire initial_step events
        for model in self.models:
            model.initial_step(self.node_voltages, 0.0)

        # Evaluate models at t=0 so output nodes are assigned before recording.
        # Without this, output nodes default to 0, producing spurious values in
        # post-processing (e.g. noise = vout_o - vin_i = 0 - 1 = -1 V).
        for model in self.models:
            model.evaluate(self.node_voltages, 0.0)

        # Record initial state
        self._record_point(0.0)
        self._step_sizes.append(0.0)

        # Main simulation loop
        while time < tstop:
            if refine_steps_left > 0:
                dt = min(refine_dt, tstop - time)
                refine_steps_left -= 1
            else:
                dt = min(tstep, tstop - time)

            # Check for breakpoints from sources (PWL knees, pulse edges)
            for src in self.sources:
                bp = src.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    if dt < 1e-18:
                        dt = 1e-18

            # Check for breakpoints from transition operators
            for model in self.models:
                bp = model.next_breakpoint(time)
                if bp is not None and bp > time and bp < time + dt:
                    dt = bp - time
                    if dt < 1e-18:
                        dt = 1e-18

            # Respect $bound_step from models
            for model in self.models:
                bs = model._bound_step
                if bs > 0 and dt > bs:
                    dt = bs

            time += dt

            # Update source voltages
            for src in self.sources:
                self.node_voltages[src.node] = src.waveform(time)

            # Evaluate all models
            for model in self.models:
                model.evaluate(self.node_voltages, time)

            # Check if any cross/above event fired this step
            cross_fired = False
            for model in self.models:
                for cd in model.cross_detectors.values():
                    if cd.last_triggered:
                        cross_fired = True
                        break
                if not cross_fired:
                    for ad in model.above_detectors.values():
                        if ad.last_triggered:
                            cross_fired = True
                            break
                if cross_fired:
                    break

            if cross_fired and refine_steps_left == 0 and dt > tstep / refine_factor:
                refine_dt = dt / refine_factor
                refine_steps_left = refine_steps

            self._record_point(time)
            self._step_sizes.append(dt)

        # Fire final_step events
        for model in self.models:
            model.final_step(self.node_voltages, time)

        # Close any open file handles
        for model in self.models:
            model._cleanup_files()

        # Convert to arrays
        time_arr = np.array(self.time_points)
        signals = {}
        for name, data in self.recorded_signals.items():
            signals[name] = np.array(data)

        return SimResult(time=time_arr, signals=signals,
                         step_sizes=np.array(self._step_sizes))

    def _record_point(self, time: float):
        self.time_points.append(time)
        for name in self.recorded_signals:
            val = self.node_voltages.get(name, 0.0)
            self.recorded_signals[name].append(val)


# ─── Waveform helpers ───

def pulse(v_lo, v_hi, period, duty=0.5, rise=1e-12, fall=1e-12, delay=0.0):
    """Create a pulse waveform function."""
    t_hi = period * duty
    knees = sorted([0.0, rise, t_hi, t_hi + fall])

    def wfn(t):
        t_eff = t - delay
        if t_eff < 0:
            return v_lo
        t_mod = t_eff % period
        if t_mod < rise:
            frac = t_mod / rise if rise > 0 else 1.0
            return v_lo + frac * (v_hi - v_lo)
        elif t_mod < t_hi:
            return v_hi
        elif t_mod < t_hi + fall:
            frac = (t_mod - t_hi) / fall if fall > 0 else 1.0
            return v_hi - frac * (v_hi - v_lo)
        else:
            return v_lo

    def _bpfn(t):
        if period <= 0:
            return None
        if t < delay:
            return delay
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

    sorted_t = sorted(set(times))

    def wfn(t):
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
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
