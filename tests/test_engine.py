"""Unit tests for evas.simulator.engine — core simulation primitives.

Covers:
  - TransitionState  (transition() operator)
  - CrossDetector    (@cross event)
  - AboveDetector    (@above event)
  - Waveform helpers (dc, pulse, pwl, ramp, sine)
  - Simulator        (end-to-end with hand-built models)
  - CompiledModel    (base-class helpers)
"""
import math
import pytest

from evas.simulator.engine import (
    AboveDetector,
    CrossDetector,
    Simulator,
    TransitionState,
    dc,
    pulse,
    pwl,
    ramp,
    sine,
)
from evas.simulator.backend import CompiledModel


# ===========================================================================
# TransitionState
# ===========================================================================

class TestTransitionState:

    def test_inactive_returns_current_val(self):
        ts = TransitionState(current_val=0.5)
        assert ts.evaluate(0.0) == 0.5
        assert ts.evaluate(1e-9) == 0.5

    def test_rising_ramp_midpoint(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        assert ts.evaluate(5e-9) == pytest.approx(0.5)

    def test_rising_ramp_start(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        assert ts.evaluate(0.0) == pytest.approx(0.0)

    def test_rising_ramp_end_deactivates(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        result = ts.evaluate(10e-9)
        assert result == pytest.approx(1.0)
        assert not ts.active

    def test_rising_ramp_beyond_end(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        ts.evaluate(10e-9)           # deactivate
        assert ts.evaluate(20e-9) == pytest.approx(1.0)

    def test_falling_ramp_uses_fall_time(self):
        ts = TransitionState(current_val=1.0)
        ts.set_target(time=0.0, target=0.0, fall=20e-9)
        assert ts.evaluate(10e-9) == pytest.approx(0.5)
        assert ts.evaluate(20e-9) == pytest.approx(0.0)
        assert not ts.active

    def test_delay_holds_start_val_during_delay(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, delay=5e-9, rise=5e-9)
        # During delay window (0 < t < 5ns): should stay at start_val=0
        assert ts.evaluate(2e-9) == pytest.approx(0.0)
        assert ts.evaluate(4.9e-9) == pytest.approx(0.0)

    def test_delay_ramp_begins_after_delay(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, delay=5e-9, rise=10e-9)
        # Ramp runs from t=5ns to t=15ns
        assert ts.evaluate(10e-9) == pytest.approx(0.5)
        assert ts.evaluate(15e-9) == pytest.approx(1.0)

    def test_set_target_same_as_current_noop(self):
        ts = TransitionState(current_val=0.7)
        ts.set_target(time=0.0, target=0.7, rise=10e-9)
        assert not ts.active
        assert ts.evaluate(0.0) == pytest.approx(0.7)

    def test_default_transition_used_when_rise_zero(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=0.0, default_transition=5e-9)
        # rise_time should be default_transition=5ns
        assert ts.evaluate(2.5e-9) == pytest.approx(0.5)
        assert ts.evaluate(5e-9) == pytest.approx(1.0)

    # next_breakpoint ---

    def test_next_breakpoint_inactive_is_none(self):
        ts = TransitionState(current_val=0.5)
        assert ts.next_breakpoint(0.0) is None

    def test_next_breakpoint_before_delay_returns_t_begin(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, delay=5e-9, rise=5e-9)
        bp = ts.next_breakpoint(0.0)
        assert bp == pytest.approx(5e-9)

    def test_next_breakpoint_during_ramp_returns_t_end(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        bp = ts.next_breakpoint(3e-9)
        assert bp == pytest.approx(10e-9)

    def test_next_breakpoint_after_ramp_is_none(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        ts.evaluate(10e-9)   # finish ramp → inactive
        assert ts.next_breakpoint(10e-9) is None

    def test_non_zero_start_time(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=5e-9, target=1.0, rise=10e-9)
        # ramp from 5ns to 15ns
        assert ts.evaluate(10e-9) == pytest.approx(0.5)
        assert ts.evaluate(15e-9) == pytest.approx(1.0)


# ===========================================================================
# CrossDetector
# ===========================================================================

class TestCrossDetector:

    def test_first_call_returns_false(self):
        cd = CrossDetector(direction=0)
        assert cd.check(-0.5) is False
        assert cd.initialized

    def test_rising_edge_both_direction(self):
        cd = CrossDetector(direction=0)
        cd.check(-0.5)
        assert cd.check(0.5) is True

    def test_falling_edge_both_direction(self):
        cd = CrossDetector(direction=0)
        cd.check(0.5)
        assert cd.check(-0.5) is True

    def test_no_crossing_same_sign(self):
        cd = CrossDetector(direction=0)
        cd.check(-0.3)
        assert cd.check(-0.1) is False   # still negative, no crossing

    def test_rising_only_ignores_falling(self):
        cd = CrossDetector(direction=1)
        cd.check(0.5)                  # init positive
        assert cd.check(-0.5) is False  # falling → should NOT fire
        assert cd.check(0.5) is True   # rising → should fire

    def test_falling_only_ignores_rising(self):
        cd = CrossDetector(direction=-1)
        cd.check(-0.5)                 # init negative
        assert cd.check(0.5) is False  # rising → should NOT fire
        assert cd.check(-0.5) is True  # falling → should fire

    def test_exact_zero_crossing_rising(self):
        cd = CrossDetector(direction=0)
        cd.check(-1e-10)
        assert cd.check(0.0) is True   # val == 0 counts as "above zero"

    def test_last_triggered_reflects_result(self):
        cd = CrossDetector(direction=0)
        cd.check(-1.0)
        cd.check(1.0)
        assert cd.last_triggered is True
        cd.check(0.5)                  # no crossing
        assert cd.last_triggered is False

    def test_would_cross_does_not_update_state(self):
        cd = CrossDetector(direction=0)
        cd.check(-0.5)
        prev_val = cd.prev_val
        result = cd.would_cross(0.5)
        assert result is True
        assert cd.prev_val == prev_val   # state unchanged

    def test_would_cross_before_init_returns_false(self):
        cd = CrossDetector(direction=0)
        assert cd.would_cross(0.5) is False


# ===========================================================================
# AboveDetector
# ===========================================================================

class TestAboveDetector:

    def test_first_call_returns_false(self):
        ad = AboveDetector()
        assert ad.check(-0.5) is False

    def test_negative_to_positive_triggers(self):
        ad = AboveDetector()
        ad.check(-0.5)
        assert ad.check(0.5) is True

    def test_positive_to_negative_does_not_trigger(self):
        ad = AboveDetector()
        ad.check(0.5)
        assert ad.check(-0.5) is False

    def test_stays_positive_does_not_trigger(self):
        ad = AboveDetector()
        ad.check(0.3)
        assert ad.check(0.7) is False

    def test_last_triggered_updates(self):
        ad = AboveDetector()
        ad.check(-1.0)
        ad.check(1.0)
        assert ad.last_triggered is True
        ad.check(0.5)
        assert ad.last_triggered is False


# ===========================================================================
# Waveform helpers
# ===========================================================================

class TestDc:
    def test_constant(self):
        fn = dc(1.8)
        assert fn(0.0) == 1.8
        assert fn(1e-9) == 1.8


class TestPulse:
    def _make(self):
        # period=10ns, 50% duty, rise=1ns, fall=1ns, no delay
        return pulse(v_lo=0.0, v_hi=1.0, period=10e-9, duty=0.5,
                     rise=1e-9, fall=1e-9, delay=0.0)

    def test_at_zero(self):
        fn = self._make()
        assert fn(0.0) == pytest.approx(0.0)

    def test_mid_rise(self):
        fn = self._make()
        assert fn(0.5e-9) == pytest.approx(0.5)

    def test_high_plateau(self):
        fn = self._make()
        assert fn(3e-9) == pytest.approx(1.0)

    def test_mid_fall(self):
        fn = self._make()
        assert fn(5.5e-9) == pytest.approx(0.5)

    def test_low_plateau(self):
        fn = self._make()
        assert fn(8e-9) == pytest.approx(0.0)

    def test_second_period(self):
        fn = self._make()
        assert fn(10e-9) == pytest.approx(0.0)  # period boundary = low
        assert fn(13e-9) == pytest.approx(1.0)  # high in second period

    def test_before_delay(self):
        fn = pulse(v_lo=0.0, v_hi=1.0, period=10e-9, delay=5e-9)
        assert fn(2e-9) == pytest.approx(0.0)

    def test_next_breakpoint_advances(self):
        fn = self._make()
        # First knee after t=0 is the end of the rise (rise=1ns)
        bp = fn._next_breakpoint(0.0)
        assert bp == pytest.approx(1e-9)


class TestPwl:
    def _make(self):
        return pwl([0.0, 5e-9, 10e-9], [0.0, 1.0, 0.0])

    def test_at_first_knot(self):
        fn = self._make()
        assert fn(0.0) == pytest.approx(0.0)

    def test_midpoint_rising(self):
        fn = self._make()
        assert fn(2.5e-9) == pytest.approx(0.5)

    def test_at_peak(self):
        fn = self._make()
        assert fn(5e-9) == pytest.approx(1.0)

    def test_midpoint_falling(self):
        fn = self._make()
        assert fn(7.5e-9) == pytest.approx(0.5)

    def test_before_start_clamps(self):
        fn = self._make()
        assert fn(-1e-9) == pytest.approx(0.0)

    def test_after_end_clamps(self):
        fn = self._make()
        assert fn(20e-9) == pytest.approx(0.0)

    def test_next_breakpoint_returns_next_knot(self):
        fn = self._make()
        bp = fn._next_breakpoint(0.0)
        assert bp == pytest.approx(5e-9)

    def test_next_breakpoint_past_last_knot_is_none(self):
        fn = self._make()
        assert fn._next_breakpoint(10e-9) is None


class TestRamp:
    def _make(self):
        return ramp(v_start=0.0, v_end=1.0, t_start=0.0, t_end=10e-9)

    def test_before_start(self):
        fn = self._make()
        assert fn(-1e-9) == pytest.approx(0.0)

    def test_midpoint(self):
        fn = self._make()
        assert fn(5e-9) == pytest.approx(0.5)

    def test_at_end(self):
        fn = self._make()
        assert fn(10e-9) == pytest.approx(1.0)

    def test_after_end(self):
        fn = self._make()
        assert fn(20e-9) == pytest.approx(1.0)


class TestSine:
    def test_offset_only(self):
        fn = sine(offset=0.5, amplitude=0.0, freq=1e9)
        assert fn(0.0) == pytest.approx(0.5)
        assert fn(1e-9) == pytest.approx(0.5)

    def test_amplitude_at_quarter_period(self):
        fn = sine(offset=0.0, amplitude=1.0, freq=1e9)
        # sin(2π * 1e9 * 0.25e-9) = sin(π/2) = 1.0
        assert fn(0.25e-9) == pytest.approx(1.0, abs=1e-9)

    def test_phase_shift(self):
        fn = sine(offset=0.0, amplitude=1.0, freq=1e9, phase=math.pi / 2)
        # sin(π/2) at t=0 → 1.0
        assert fn(0.0) == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# Simulator — end-to-end with hand-built models
# ===========================================================================

class TestSimulator:

    def test_dc_source_recorded(self):
        sim = Simulator()
        sim.add_source("vdd", dc(1.8))
        sim.record("vdd")
        result = sim.run(tstop=10e-9, tstep=1e-9)
        assert result.signals["vdd"].min() == pytest.approx(1.8)
        assert result.signals["vdd"].max() == pytest.approx(1.8)

    def test_time_array_starts_at_zero(self):
        sim = Simulator()
        sim.add_source("v", dc(0.0))
        sim.record("v")
        result = sim.run(tstop=5e-9, tstep=1e-9)
        assert result.time[0] == pytest.approx(0.0)

    def test_time_array_ends_at_tstop(self):
        sim = Simulator()
        sim.add_source("v", dc(0.0))
        sim.record("v")
        result = sim.run(tstop=10e-9, tstep=1e-9)
        assert result.time[-1] == pytest.approx(10e-9)

    def test_pulse_source_toggles(self):
        sim = Simulator()
        sim.add_source("clk", pulse(0.0, 1.0, period=2e-9, rise=10e-12, fall=10e-12))
        sim.record("clk")
        result = sim.run(tstop=10e-9, tstep=0.1e-9)
        assert result.signals["clk"].max() == pytest.approx(1.0, abs=0.01)
        assert result.signals["clk"].min() == pytest.approx(0.0, abs=0.01)

    def test_unrecorded_node_absent_from_result(self):
        sim = Simulator()
        sim.add_source("v", dc(1.0))
        sim.record("v")
        result = sim.run(tstop=1e-9, tstep=0.1e-9)
        assert "other" not in result.signals

    def test_model_drives_output_node(self):
        """A hand-built model that drives 'out' to VDD/2 via CompiledModel helpers."""

        class HalfVddModel(CompiledModel):
            def evaluate(self, nv, time):
                vdd = self._get_voltage("vdd", nv)
                self._set_output("out", vdd * 0.5, nv)

        sim = Simulator()
        sim.add_source("vdd", dc(1.8))
        model = HalfVddModel()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=5e-9, tstep=1e-9)
        # t=0 is recorded before evaluate() runs, so skip initial point
        assert result.signals["out"][1:].mean() == pytest.approx(0.9, abs=1e-6)

    def test_simresult_step_sizes_length_matches_time(self):
        sim = Simulator()
        sim.add_source("v", dc(0.0))
        sim.record("v")
        result = sim.run(tstop=5e-9, tstep=1e-9)
        assert len(result.step_sizes) == len(result.time)


# ===========================================================================
# CompiledModel base-class helpers
# ===========================================================================

class TestCompiledModelHelpers:

    def setup_method(self):
        self.model = CompiledModel()

    def test_get_voltage_from_node_voltages(self):
        nv = {"vdd": 1.8}
        assert self.model._get_voltage("vdd", nv) == pytest.approx(1.8)

    def test_get_voltage_missing_node_returns_zero(self):
        assert self.model._get_voltage("x", {}) == pytest.approx(0.0)

    def test_get_voltage_via_node_map(self):
        self.model.node_map["out_internal"] = "out_external"
        nv = {"out_external": 0.9}
        assert self.model._get_voltage("out_internal", nv) == pytest.approx(0.9)

    def test_set_output_writes_node_voltages(self):
        nv = {}
        self.model._set_output("out", 1.2, nv)
        assert nv["out"] == pytest.approx(1.2)
        assert self.model.output_nodes["out"] == pytest.approx(1.2)

    def test_set_output_respects_node_map(self):
        self.model.node_map["q"] = "q_ext"
        nv = {}
        self.model._set_output("q", 0.9, nv)
        assert nv["q_ext"] == pytest.approx(0.9)

    def test_transition_first_call_returns_target(self):
        nv = {}
        val = self.model._transition("t0", time=0.0, target=1.0, rise=10e-9)
        assert val == pytest.approx(1.0)

    def test_transition_ramps_over_time(self):
        # First call: registers key, returns target immediately (no prior state)
        self.model._transition("t1", time=0.0, target=0.0, rise=10e-9)
        # Second call: set new target, transition should begin
        self.model._transition("t1", time=0.0, target=1.0, rise=10e-9)
        val_mid = self.model._transition("t1", time=5e-9, target=1.0, rise=10e-9)
        assert 0.0 < val_mid < 1.0

    def test_check_cross_first_call_false(self):
        assert self.model._check_cross("c0", -0.5) is False

    def test_check_cross_rising_triggers(self):
        self.model._check_cross("c1", -0.5)
        assert self.model._check_cross("c1", 0.5) is True

    def test_check_above_first_call_false(self):
        assert self.model._check_above("a0", -0.5) is False

    def test_check_above_negative_to_positive_triggers(self):
        self.model._check_above("a1", -0.5)
        assert self.model._check_above("a1", 0.5) is True

    def test_array_get_unset_returns_zero(self):
        assert self.model._array_get("arr", 0) == 0

    def test_array_set_and_get(self):
        self.model._array_set("arr", 3, 42)
        assert self.model._array_get("arr", 3) == 42

    def test_strobe_appends_message(self):
        self.model._strobe(1e-9, "val=%d", 7)
        assert len(self.model._strobe_log) == 1
        assert "7" in self.model._strobe_log[0]

    def test_next_breakpoint_no_active_transitions(self):
        assert self.model.next_breakpoint(0.0) is None

    def test_next_breakpoint_with_active_transition(self):
        # Plant an active TransitionState directly
        ts = TransitionState(current_val=0.0, target_val=1.0,
                             start_time=0.0, start_val=0.0,
                             rise_time=10e-9, active=True)
        self.model.transitions["t"] = ts
        bp = self.model.next_breakpoint(0.0)
        assert bp == pytest.approx(10e-9)
