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
import re
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
        assert cd.check(0.0, -0.5) is False
        assert cd.initialized

    def test_rising_edge_both_direction(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -0.5)
        assert cd.check(0.0, 0.5) is True

    def test_falling_edge_both_direction(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, 0.5)
        assert cd.check(0.0, -0.5) is True

    def test_no_crossing_same_sign(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -0.3)
        assert cd.check(0.0, -0.1) is False   # still negative, no crossing

    def test_rising_only_ignores_falling(self):
        cd = CrossDetector(direction=1)
        cd.check(0.0, 0.5)                  # init positive
        assert cd.check(0.0, -0.5) is False  # falling → should NOT fire
        assert cd.check(0.0, 0.5) is True   # rising → should fire

    def test_falling_only_ignores_rising(self):
        cd = CrossDetector(direction=-1)
        cd.check(0.0, -0.5)                 # init negative
        assert cd.check(0.0, 0.5) is False  # rising → should NOT fire
        assert cd.check(0.0, -0.5) is True  # falling → should fire

    def test_exact_zero_crossing_rising(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -1e-10)
        assert cd.check(0.0, 0.0) is True   # val == 0 counts as "above zero"

    def test_last_triggered_reflects_result(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -1.0)
        cd.check(0.0, 1.0)
        assert cd.last_triggered is True
        cd.check(0.0, 0.5)                  # no crossing
        assert cd.last_triggered is False

    def test_would_cross_does_not_update_state(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -0.5)
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
        assert ad.check(0.0, -0.5) is False

    def test_negative_to_positive_triggers(self):
        ad = AboveDetector()
        ad.check(0.0, -0.5)
        assert ad.check(0.0, 0.5) is True

    def test_positive_to_negative_does_not_trigger(self):
        ad = AboveDetector()
        ad.check(0.0, 0.5)
        assert ad.check(0.0, -0.5) is False

    def test_stays_positive_does_not_trigger(self):
        ad = AboveDetector()
        ad.check(0.0, 0.3)
        assert ad.check(0.0, 0.7) is False

    def test_last_triggered_updates(self):
        ad = AboveDetector()
        ad.check(0.0, -1.0)
        ad.check(0.0, 1.0)
        assert ad.last_triggered is True
        ad.check(0.0, 0.5)
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

    def test_empty_wave_raises_clear_error(self):
        with pytest.raises(ValueError, match="at least one time/value pair"):
            pwl([], [])

    def test_mismatched_lengths_raise_clear_error(self):
        with pytest.raises(ValueError, match=re.escape("same length")):
            pwl([0.0], [0.0, 1.0])


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
        assert self.model._check_cross("c0", 0.0, -0.5) is False

    def test_check_cross_rising_triggers(self):
        self.model._check_cross("c1", 0.0, -0.5)
        assert self.model._check_cross("c1", 0.0, 0.5) is True

    def test_check_above_first_call_false(self):
        assert self.model._check_above("a0", 0.0, -0.5) is False

    def test_check_above_negative_to_positive_triggers(self):
        self.model._check_above("a1", 0.0, -0.5)
        assert self.model._check_above("a1", 0.0, 0.5) is True

    def test_array_get_unset_returns_zero(self):
        assert self.model._array_get("arr", 0) == 0

    def test_array_set_and_get(self):
        self.model._array_set("arr", 3, 42)
        assert self.model._array_get("arr", 3) == 42

    def test_strobe_appends_message(self):
        self.model._strobe(1e-9, "val=%d", 7)
        assert len(self.model._strobe_log) == 1
        assert "7" in self.model._strobe_log[0][1]

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

    def test_check_timer_first_fire(self):
        fired = self.model._check_timer("t0", 10e-9, 10e-9)
        assert fired is True

    def test_check_timer_before_period(self):
        fired = self.model._check_timer("t1", 5e-9, 10e-9)
        assert fired is False

    def test_check_timer_advances_next_fire(self):
        self.model._check_timer("t2", 10e-9, 10e-9)  # fires, sets next=20ns
        assert not self.model._check_timer("t2", 15e-9, 10e-9)  # before 20ns
        assert self.model._check_timer("t2", 20e-9, 10e-9)  # fires at 20ns

    def test_next_breakpoint_includes_timer(self):
        self.model.timer_states["t0"] = 10e-9
        bp = self.model.next_breakpoint(0.0)
        assert bp == pytest.approx(10e-9)

    def test_idtmod_trapezoid_and_wrap(self):
        y0 = self.model._idtmod("i0", time=0.0, x=2.0, ic=0.0, mod=1.0)
        assert y0 == pytest.approx(0.0)

        # +2 * 0.1 = 0.2
        y1 = self.model._idtmod("i0", time=0.1, x=2.0, ic=0.0, mod=1.0)
        assert y1 == pytest.approx(0.2)

        # +2 * 0.5 = +1.0, wrapped by mod=1.0 -> back to 0.2
        y2 = self.model._idtmod("i0", time=0.6, x=2.0, ic=0.0, mod=1.0)
        assert y2 == pytest.approx(0.2)

    def test_idtmod_same_time_no_double_integrate(self):
        self.model._idtmod("i1", time=0.0, x=1.0, ic=0.0, mod=1.0)
        y1 = self.model._idtmod("i1", time=1e-9, x=1.0, ic=0.0, mod=1.0)
        y2 = self.model._idtmod("i1", time=1e-9, x=1.0, ic=0.0, mod=1.0)
        assert y2 == pytest.approx(y1)

    def test_temperature_default(self):
        assert self.model._temperature == pytest.approx(27.0)

    def test_final_step_base_is_noop(self):
        # base class final_step should not raise
        self.model.final_step({}, 0.0)


# ===========================================================================
# Compiled model features: timer, final_step, $temperature, $vt
# ===========================================================================

class TestTimerEvent:
    """Test @(timer(period)) via a compiled VA module."""

    VA_SRC = """\
`include "disciplines.vams"
module timer_test(out);
    output voltage out;
    integer count;
    analog begin
        @(initial_step) begin
            count = 0;
        end
        @(timer(10e-9)) begin
            count = count + 1;
        end
        V(out) <+ count;
    end
endmodule
"""

    def test_timer_fires_periodically(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=50e-9, tstep=1e-9)
        # At 50ns, timer should have fired at 10, 20, 30, 40, 50 → count=5
        final_val = result.signals["out"][-1]
        assert final_val == pytest.approx(5.0, abs=1.0)


class TestIdtmodEvent:
    """Smoke test for idtmod() compilation and runtime behavior."""

    VA_SRC = """\
`include "disciplines.vams"
module idtmod_test(out);
    output voltage out;
    real f;
    real ph;
    analog begin
        f = 1.0e9;
        ph = idtmod(f, 0.0, 1.0);
        V(out) <+ ph;
    end
endmodule
"""

    def test_idtmod_compiles_and_changes_over_time(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=2e-9, tstep=0.5e-9)

        out = result.signals["out"]
        assert out.max() > out.min()
        assert out.min() >= -1e-12
        assert out.max() <= 1.0 + 1e-12


class TestFinalStep:
    """Test @(final_step) via a compiled VA module."""

    VA_SRC = """\
`include "disciplines.vams"
module final_test(out);
    output voltage out;
    integer flag;
    analog begin
        @(initial_step) begin
            flag = 0;
        end
        @(final_step) begin
            flag = 99;
        end
        V(out) <+ flag;
    end
endmodule
"""

    def test_final_step_fires_after_sim(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        result = sim.run(tstop=10e-9, tstep=1e-9)
        # After simulation, final_step should have set flag=99
        assert model.state['flag'] == 99


class TestTemperatureVt:
    """Test $temperature and $vt via compiled VA modules."""

    VA_TEMP = """\
`include "disciplines.vams"
module temp_test(out);
    output voltage out;
    real t_val;
    analog begin
        t_val = $temperature;
        V(out) <+ t_val;
    end
endmodule
"""

    VA_VT = """\
`include "disciplines.vams"
module vt_test(out);
    output voltage out;
    real vt_val;
    analog begin
        vt_val = $vt;
        V(out) <+ vt_val;
    end
endmodule
"""

    def test_temperature_returns_kelvin(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_TEMP)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=0.5e-9)
        # Default temp = 27C → 300.15 K
        assert result.signals["out"][-1] == pytest.approx(300.15, abs=0.01)

    def test_vt_at_room_temperature(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_VT)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=0.5e-9)
        # kT/q at 300.15K ≈ 0.025875 V
        expected_vt = 1.380649e-23 * 300.15 / 1.602176634e-19
        assert result.signals["out"][-1] == pytest.approx(expected_vt, rel=1e-4)


class TestCaseStatement:
    """Test case/endcase via compiled VA modules."""

    VA_SRC = """\
`include "disciplines.vams"
module case_test(out, sel);
    input voltage sel;
    output voltage out;
    integer code;
    real result;
    analog begin
        code = V(sel) > 1.5 ? 2 : (V(sel) > 0.5 ? 1 : 0);
        case (code)
            0: result = 0.0;
            1: result = 0.5;
            2: result = 1.0;
            default: result = -1.0;
        endcase
        V(out) <+ result;
    end
endmodule
"""

    def test_case_selects_branch(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)

        # sel=0.0 → code=0 → result=0.0
        model = ModelCls()
        sim = Simulator()
        sim.add_source("sel", dc(0.0))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=0.5e-9)
        assert result.signals["out"][-1] == pytest.approx(0.0)

    def test_case_selects_middle(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)

        # sel=1.0 → code=1 → result=0.5
        model = ModelCls()
        sim = Simulator()
        sim.add_source("sel", dc(1.0))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=0.5e-9)
        assert result.signals["out"][-1] == pytest.approx(0.5)

    def test_case_selects_high(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)

        # sel=2.0 → code=2 → result=1.0
        model = ModelCls()
        sim = Simulator()
        sim.add_source("sel", dc(2.0))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=0.5e-9)
        assert result.signals["out"][-1] == pytest.approx(1.0)


class TestBoundStep:
    """Test $bound_step() via compiled VA module."""

    VA_SRC = """\
`include "disciplines.vams"
module bound_test(out);
    output voltage out;
    analog begin
        $bound_step(1e-9);
        V(out) <+ 1.0;
    end
endmodule
"""

    def test_bound_step_limits_dt(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        # tstep=10ns but $bound_step limits to 1ns
        result = sim.run(tstop=10e-9, tstep=10e-9)
        # Should have at least 10 steps (10ns / 1ns)
        assert len(result.time) >= 10
        # All step sizes (after first) should be <= 1ns + tolerance
        for dt in result.step_sizes[1:]:
            assert dt <= 1e-9 + 1e-15


class TestFileIO:
    """Test $fopen/$fstrobe/$fclose via compiled VA module."""

    VA_SRC = """\
`include "disciplines.vams"
module fileio_test(clk);
    input voltage clk;
    integer fd;
    integer count;
    analog begin
        @(initial_step) begin
            fd = $fopen("{filepath}", "w");
            count = 0;
        end
        @(cross(V(clk) - 0.5, 1)) begin
            count = count + 1;
            $fstrobe(fd, "edge %d at %e", count, $abstime);
        end
        @(final_step) begin
            $fclose(fd);
        end
    end
endmodule
"""

    def test_fopen_fstrobe_fclose(self, tmp_path):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        outfile = tmp_path / "test_output.txt"
        src = self.VA_SRC.replace("{filepath}", str(outfile).replace("\\", "/"))
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        # 5 rising edges: pulse period=20ns, tstop=100ns
        sim.add_source("clk", pulse(v_lo=0.0, v_hi=1.0, period=20e-9,
                                     rise=0.1e-9, fall=0.1e-9, duty=0.5))
        sim.add_model(model)
        result = sim.run(tstop=100e-9, tstep=1e-9)

        # File should exist and have lines
        assert outfile.exists()
        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 5
        assert lines[0].startswith("edge 1 at")
        assert lines[4].startswith("edge 5 at")

    def test_fclose_cleans_handle(self, tmp_path):
        """After simulation, file handles should be cleaned up."""
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        outfile = tmp_path / "cleanup_test.txt"
        src = self.VA_SRC.replace("{filepath}", str(outfile).replace("\\", "/"))
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("clk", pulse(v_lo=0.0, v_hi=1.0, period=20e-9,
                                     rise=0.1e-9, fall=0.1e-9, duty=0.5))
        sim.add_model(model)
        sim.run(tstop=50e-9, tstep=1e-9)

        # After sim, all file handles should be closed
        assert len(model._file_handles) == 0


class Test2DNodeArray:
    """Backend compilation and runtime for 2-D electrical node arrays."""

    VA_1D = """\
`include "disciplines.vams"
module clk_array_1d(VDD, VSS);
    inout electrical VDD, VSS;
    electrical [0:3] clk_nodes;
    genvar N;
    analog begin
        for (N = 0; N <= 3; N = N + 1)
            V(clk_nodes[N], VSS) <+ (N == 0) ? 1.0 : 0.0;
    end
endmodule
"""

    VA_2D = """\
`include "disciplines.vams"
module dbus_2d(VDD, VSS);
    inout electrical VDD, VSS;
    electrical [1:0] dbus [0:3];
    genvar ch, j;
    analog begin
        for (ch = 0; ch <= 3; ch = ch + 1)
            for (j = 0; j <= 1; j = j + 1)
                V(dbus[ch][j], VSS) <+ (ch * 2.0 + j);
    end
endmodule
"""

    def test_1d_array_contribution_compiles(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_1D)
        ModelCls = compile_module(mod)
        model = ModelCls()
        nv = {}
        model.evaluate(nv, 0.0)
        # clk_nodes[0] should be 1.0, others 0.0
        assert model.output_nodes.get("clk_nodes[0]") == pytest.approx(1.0)
        assert model.output_nodes.get("clk_nodes[1]") == pytest.approx(0.0)
        assert model.output_nodes.get("clk_nodes[2]") == pytest.approx(0.0)
        assert model.output_nodes.get("clk_nodes[3]") == pytest.approx(0.0)

    def test_2d_array_contribution_compiles(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_2D)
        ModelCls = compile_module(mod)
        model = ModelCls()
        nv = {}
        model.evaluate(nv, 0.0)
        # V(dbus[ch][j]) <+ ch*2 + j
        for ch in range(4):
            for j in range(2):
                key = f"dbus[{ch}][{j}]"
                expected = ch * 2.0 + j
                assert model.output_nodes.get(key) == pytest.approx(expected), \
                    f"{key} expected {expected}, got {model.output_nodes.get(key)}"

    def test_2d_array_read_voltage(self):
        """V(dbus[ch][j], VSS) used as a read expression (cross event etc.)"""
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        va_src = """\
`include "disciplines.vams"
module dbus_read(VDD, VSS);
    inout electrical VDD, VSS;
    electrical [1:0] dbus [0:3];
    real v00;
    genvar ch, j;
    analog begin
        for (ch = 0; ch <= 3; ch = ch + 1)
            for (j = 0; j <= 1; j = j + 1)
                V(dbus[ch][j], VSS) <+ (ch * 2.0 + j);
        v00 = V(dbus[0][0], VSS);
    end
endmodule
"""
        mod = parse(va_src)
        ModelCls = compile_module(mod)
        model = ModelCls()
        # Pre-set dbus[0][0] in output_nodes so the read can see it
        nv = {}
        model.evaluate(nv, 0.0)
        # v00 = V(dbus[0][0]) = 0*2+0 = 0.0
        assert model.state.get("v00") == pytest.approx(0.0)
        # v01 check: dbus[1][1] = 1*2+1 = 3.0
        assert model.output_nodes.get("dbus[1][1]") == pytest.approx(3.0)


class TestHierarchicalInstantiation:
    VA_CHILD = """\
`include "disciplines.vams"
module child_inv(out, inp, vdd, vss);
output out;
electrical out;
input inp;
electrical inp;
inout vdd, vss;
electrical vdd, vss;
analog begin
    V(out, vss) <+ (V(inp, vss) > 0.5*V(vdd, vss) ? 0.0 : V(vdd, vss));
end
endmodule
"""

    VA_TOP = """\
`include "disciplines.vams"
module top_wrap(out, inp, vdd, vss);
output out;
electrical out;
input inp;
electrical inp;
inout vdd, vss;
electrical vdd, vss;
child_inv u0 (
    .out(out),
    .inp(inp),
    .vdd(vdd),
    .vss(vss)
);
endmodule
"""

    def test_named_port_instance_parses(self):
        from evas.compiler.parser import parse
        mod = parse(self.VA_TOP)
        assert len(mod.instances) == 1
        inst = mod.instances[0]
        assert inst.module_name == "child_inv"
        assert inst.instance_name == "u0"
        assert len(inst.connections) == 4
        assert inst.connections[0].port_name == "out"

    def test_parent_calls_child_evaluate(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        child_mod = parse(self.VA_CHILD)
        top_mod = parse(self.VA_TOP)
        ChildCls = compile_module(child_mod)
        TopCls = compile_module(top_mod)
        registry = {
            "child_inv": (ChildCls, child_mod),
            "top_wrap": (TopCls, top_mod),
        }
        ChildCls._module_registry = registry
        TopCls._module_registry = registry

        top = TopCls()
        top.node_map = {"out": "OUT", "inp": "INP", "vdd": "VDD", "vss": "VSS"}
        nv = {"INP": 0.0, "VDD": 1.0, "VSS": 0.0}
        top.initial_step(nv, 0.0)
        top.evaluate(nv, 0.0)
        assert nv["OUT"] == pytest.approx(1.0)
        nv["INP"] = 1.0
        top.evaluate(nv, 1e-9)
        assert nv["OUT"] == pytest.approx(0.0)
