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
from evas.simulator.backend import CompilationError


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

    def test_next_breakpoint_during_ramp_returns_inner_point_first(self):
        # Long transition() ramps expose interior points so cross() events and
        # output threshold checks can observe the edge before the ramp completes.
        # At time=3ns the midpoint (5ns) is the nearest upcoming interior point.
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-9)
        bp = ts.next_breakpoint(3e-9)
        assert bp == pytest.approx(5e-9)

    def test_next_breakpoint_skips_short_ramp_inner_points(self):
        ts = TransitionState(current_val=0.0)
        ts.set_target(time=0.0, target=1.0, rise=10e-12)
        bp = ts.next_breakpoint(0.0, min_ramp_time=20e-12)
        assert bp == pytest.approx(10e-12)

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

    def test_repeated_falling_interruptions_readjust_original_transition(self):
        ts = TransitionState(current_val=0.8317)
        events = [
            (44.55e-9, 0.7901),
            (46.55e-9, 0.7506),
            (48.55e-9, 0.7131),
            (50.55e-9, 0.6774),
            (52.55e-9, 0.6435),
            (54.55e-9, 0.6114),
            (56.55e-9, 0.5808),
            (58.55e-9, 0.5517),
            (60.55e-9, 0.5242),
            (62.55e-9, 0.4979),
            (64.55e-9, 0.5379),
        ]
        samples = {
            54.5e-9: 0.715881,
            60.0e-9: 0.580100,
            64.0e-9: 0.5203865,
        }

        sample_items = iter(sorted(samples.items()))
        next_sample = next(sample_items, None)
        for event_time, target in events:
            while next_sample is not None and next_sample[0] < event_time:
                sample_time, expected = next_sample
                assert ts.evaluate(sample_time) == pytest.approx(expected)
                next_sample = next(sample_items, None)
            ts.evaluate(event_time)
            ts.set_target(event_time, target, rise=150e-12, fall=10e-9)

        while next_sample is not None:
            sample_time, expected = next_sample
            assert ts.evaluate(sample_time) == pytest.approx(expected)
            next_sample = next(sample_items, None)


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

    def test_exact_zero_touch_from_one_side_triggers_without_post_side_nudge(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -1e-10)
        assert cd.check(1e-9, 0.0) is True
        assert cd.t_cross == pytest.approx(1e-9)
        assert cd.last_trigger_direction == 1
        assert cd.last_trigger_went_beyond is False

    def test_exact_zero_touch_returning_same_side_does_not_retrigger(self):
        cd = CrossDetector(direction=1)
        cd.check(0.0, -1.0)
        assert cd.check(1e-9, 0.0, expr_tol=0.0) is True
        assert cd.last_trigger_went_beyond is False
        assert cd.check(2e-9, -1.0, expr_tol=0.0) is False
        assert cd.pending_touch_direction == 0
        assert cd.last_triggered is False

    def test_exact_zero_touch_falling_triggers_at_touch_time(self):
        cd = CrossDetector(direction=-1)
        cd.check(0.0, 1.0)
        assert cd.check(1e-9, 0.0, expr_tol=0.0) is True
        assert cd.t_cross == pytest.approx(1e-9)
        assert cd.last_trigger_direction == -1
        assert cd.last_trigger_went_beyond is False

    def test_last_triggered_reflects_result(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -1.0)
        cd.check(0.0, 1.0)
        assert cd.last_triggered is True
        assert cd.last_trigger_went_beyond is True
        cd.check(0.0, 0.5)                  # no crossing
        assert cd.last_triggered is False
        assert cd.last_trigger_went_beyond is False

    def test_would_cross_does_not_update_state(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -0.5)
        prev_val = cd.prev_val
        result = cd.would_cross(0.5)
        assert result is True
        assert cd.prev_val == prev_val   # state unchanged

    def test_would_cross_exact_touch_counts_as_cross(self):
        cd = CrossDetector(direction=1)
        cd.check(0.0, -0.5)
        assert cd.would_cross(0.0, expr_tol=0.0) is True
        assert cd.would_cross(0.5, expr_tol=0.0) is True

    def test_would_cross_before_init_returns_false(self):
        cd = CrossDetector(direction=0)
        assert cd.would_cross(0.5) is False

    def test_expr_tol_affects_cross_decision(self):
        cd = CrossDetector(direction=1)
        cd.check(0.0, -1e-4)
        # With expr_tol=1e-3 both samples are inside tolerance band -> no trigger.
        assert cd.check(1e-9, 2e-4, expr_tol=1e-3) is False
        # Tightening tolerance allows trigger.
        cd = CrossDetector(direction=1)
        cd.check(0.0, -1e-4)
        assert cd.check(1e-9, 2e-4, expr_tol=1e-5) is True

    def test_time_tol_debounces_back_to_back_crosses(self):
        cd = CrossDetector(direction=0)
        cd.check(0.0, -1.0)
        assert cd.check(1e-9, 1.0, time_tol=2e-9) is True
        # Immediate opposite crossing within time_tol should be suppressed.
        assert cd.check(1.5e-9, -1.0, time_tol=2e-9) is False


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
        # Edge interiors are breakpoints so @cross events on source ramps are
        # ordered before later same-step events.
        bp = fn._next_breakpoint(0.0)
        assert bp == pytest.approx(0.5e-9)

    def test_spectre_width_excludes_rise_time(self):
        fn = pulse(
            v_lo=0.0, v_hi=0.9, period=1e-9, delay=100e-12,
            rise=20e-12, fall=20e-12, width=500e-12,
        )

        assert fn(100e-12) == pytest.approx(0.0)
        assert fn(110e-12) == pytest.approx(0.45)
        assert fn(120e-12) == pytest.approx(0.9)
        assert fn(620e-12) == pytest.approx(0.9)
        assert fn(630e-12) == pytest.approx(0.45)
        assert fn(640e-12) == pytest.approx(0.0)

    def test_spectre_width_breakpoints_include_fall_after_rise_plus_width(self):
        fn = pulse(
            v_lo=0.0, v_hi=0.9, period=1e-9, delay=100e-12,
            rise=20e-12, fall=20e-12, width=500e-12,
        )

        assert fn._next_breakpoint(619e-12) == pytest.approx(620e-12)
        assert fn._next_breakpoint(620e-12) == pytest.approx(630e-12)


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

    def test_record_step_keeps_event_breakpoints_on_output_grid(self):
        class InternalBreakpointModel(CompiledModel):
            def next_breakpoint(self, time):
                return 0.5e-9 if time < 0.5e-9 else None

            def evaluate(self, nv, time):
                self._set_output("out", time, nv)

        sim = Simulator()
        sim.add_model(InternalBreakpointModel())
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-9, record_step=1e-9)

        assert len(result.time) == 3
        assert result.time[0] == pytest.approx(0.0)
        assert result.time[1] == pytest.approx(0.5e-9)
        assert result.time[-1] == pytest.approx(1e-9)
        assert len(result.step_sizes) == len(result.time)

    def test_skip_source_error_control_only_removes_source_dynamic_shrinks(self):
        sim = Simulator()
        sim.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.record("vin")
        sim.run(tstop=2e-9, tstep=1e-9, reltol=1e-3, vabstol=1e-6)
        default_stats = dict(sim._perf_stats)

        fast = Simulator()
        fast.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        fast.record("vin")
        fast.run(
            tstop=2e-9,
            tstep=1e-9,
            reltol=1e-3,
            vabstol=1e-6,
            skip_source_error_control=True,
        )

        assert default_stats["dynamic_step_shrinks"] > 0
        assert default_stats["err_ratio_skipped_sources"] == 0
        assert fast._perf_stats["dynamic_step_shrinks"] == 0
        assert fast._perf_stats["err_ratio_skipped_sources"] > 0


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

    def test_get_voltage_interpolates_inside_event_context(self):
        self.model._prepare_step({"vin": 0.0, "rst": 0.0}, {"vin": 1.0, "rst": 1.0}, 0.0, 10e-9)
        self.model._event_time = 4e-9
        self.model._event_context_active = True
        self.model._event_interpolated_nodes = {"vin"}
        assert self.model._get_voltage("vin", {"vin": 1.0}) == pytest.approx(0.4)
        assert self.model._get_voltage("rst", {"rst": 1.0}) == pytest.approx(0.4)
        self.model._event_context_active = False
        assert self.model._get_voltage("vin", {"vin": 1.0}) == pytest.approx(1.0)

    def test_integer_assignment_rounds_real_values_like_spectre(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module integer_round_probe(out, neg);
    output voltage out;
    output voltage neg;
    integer ipos;
    integer ineg;
    analog begin
        @(initial_step) begin
            ipos = 0.51;
            ineg = -0.51;
        end
        V(out) <+ ipos;
        V(neg) <+ ineg;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        sim = Simulator()
        sim.add_model(ModelCls())
        sim.record("out")
        sim.record("neg")
        result = sim.run(tstop=1e-9, tstep=1e-9)

        assert result.signals["out"][0] == pytest.approx(1.0)
        assert result.signals["neg"][0] == pytest.approx(-1.0)

    def test_module_scope_real_initializer_can_reference_prior_constants(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module initializer_probe(out);
    output voltage out;
    parameter real vin_min = 0.25;
    parameter real vin_max = 0.90;
    real vmin_out = 0.28;
    real vmax_out = 0.82;
    real slope = (vmax_out - vmin_out) / (vin_max - vin_min);
    real target = vmin_out + slope * (0.75 - vin_min);
    analog begin
        V(out) <+ target;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        assert model.state["slope"] == pytest.approx((0.82 - 0.28) / (0.90 - 0.25))
        assert model.state["target"] == pytest.approx(0.6953846153846154)

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-9)

        assert result.signals["out"][-1] == pytest.approx(0.6953846153846154)

    def test_plain_real_comparisons_do_not_use_hidden_epsilon(self):
        assert self.model._cmp_ge(0.5 - 1e-16, 0.5) is False
        assert self.model._cmp_le(0.5 + 1e-16, 0.5) is False


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

    def test_check_cross_with_tolerances(self):
        self.model._check_cross("ct", 0.0, -1e-4, direction=1, expr_tol=1e-3)
        assert self.model._check_cross("ct", 1e-9, 2e-4, direction=1, expr_tol=1e-3) is False
        # Re-arm by going sufficiently below -expr_tol, then cross again.
        assert self.model._check_cross("ct", 2e-9, -2e-2, direction=1, expr_tol=1e-3) is False
        assert self.model._check_cross("ct", 3e-9, 2e-2, direction=1, expr_tol=1e-3) is True

    def test_seeded_random_streams_are_reproducible(self):
        seq_a = [self.model._rand_normal(7, 0.0, 1.0) for _ in range(4)]
        seq_b = [self.model._rand_normal(7, 0.0, 1.0) for _ in range(4)]
        # Same seeded stream continues deterministically.
        assert seq_a != seq_b

        m2 = CompiledModel()
        seq_c = [m2._rand_normal(7, 0.0, 1.0) for _ in range(4)]
        # Fresh model with same seed reproduces the initial sequence.
        assert seq_a == pytest.approx(seq_c)

    def test_unseeded_random_stream_is_deterministic_per_model(self):
        seq_a = [self.model._rand_uniform(None, -1.0, 1.0) for _ in range(3)]
        m2 = CompiledModel()
        seq_b = [m2._rand_uniform(None, -1.0, 1.0) for _ in range(3)]
        assert seq_a == pytest.approx(seq_b)

    def test_slew_first_call_returns_target(self):
        val = self.model._slew("s0", time=0.0, target=1.0, maxrise=1e8, maxfall=1e8)
        assert val == pytest.approx(1.0)

    def test_slew_limits_rising_slope(self):
        self.model._slew("s1", time=0.0, target=0.0, maxrise=1e8, maxfall=1e8)
        # 1e8 V/s over 1ns -> +0.1V max increment
        val = self.model._slew("s1", time=1e-9, target=1.0, maxrise=1e8, maxfall=1e8)
        assert val == pytest.approx(0.1, abs=1e-12)

    def test_slew_limits_falling_slope(self):
        self.model._slew("s2", time=0.0, target=1.0, maxrise=1e8, maxfall=2e8)
        # 2e8 V/s over 1ns -> -0.2V max decrement
        val = self.model._slew("s2", time=1e-9, target=0.0, maxrise=1e8, maxfall=2e8)
        assert val == pytest.approx(0.8, abs=1e-12)

    def test_slew_zero_limit_means_unlimited(self):
        self.model._slew("s3", time=0.0, target=0.0, maxrise=0.0, maxfall=0.0)
        val = self.model._slew("s3", time=1e-9, target=1.0, maxrise=0.0, maxfall=0.0)
        assert val == pytest.approx(1.0, abs=1e-12)

    def test_check_above_first_call_false(self):
        assert self.model._check_above("a0", 0.0, -0.5) is False

    def test_check_above_first_call_positive_triggers(self):
        assert self.model._check_above("a0_pos", 0.0, 0.5) is True

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
        # Plant an active TransitionState directly.
        # At time=0.0 the first interior point (2.5ns) is returned first.
        ts = TransitionState(current_val=0.0, target_val=1.0,
                             start_time=0.0, start_val=0.0,
                             rise_time=10e-9, active=True)
        self.model.transitions["t"] = ts
        bp = self.model.next_breakpoint(0.0)
        assert bp == pytest.approx(2.5e-9)

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

    def test_check_timer_rejects_non_finite_period(self):
        assert not self.model._check_timer("tnf", 1e-9, float("nan"))
        assert not self.model._check_timer("tnf2", 1e-9, float("inf"))

    def test_check_timer_at_fires_once_per_target(self):
        assert not self.model._check_timer_at("ta", 5e-9, 10e-9)
        assert self.model._check_timer_at("ta", 10e-9, 10e-9)
        assert not self.model._check_timer_at("ta", 11e-9, 10e-9)
        assert not self.model._check_timer_at("ta", 11e-9, 20e-9)
        assert self.model._check_timer_at("ta", 20e-9, 20e-9)

    def test_check_timer_at_rejects_non_finite_target(self):
        assert not self.model._check_timer_at("ta_nf", 1e-9, float("nan"))

    def test_next_breakpoint_includes_timer(self):
        self.model.timer_states["t0"] = 10e-9
        bp = self.model.next_breakpoint(0.0)
        assert bp == pytest.approx(10e-9)

    def test_next_breakpoint_ignores_consumed_absolute_timer(self):
        self.model.timer_states["ta"] = 10e-9
        self.model.timer_last_fired["ta"] = 10e-9
        assert self.model.next_breakpoint(10e-9) is None

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
# Compiled event scheduling parity
# ===========================================================================

class TestCompiledEventScheduling:

    def test_cross_event_body_samples_other_nodes_at_crossing_time(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module crossing_sample_probe(clk, vin, out);
    input voltage clk;
    input voltage vin;
    output voltage out;
    real sampled;
    analog begin
        @(initial_step) begin
            sampled = 0.0;
        end
        @(cross(V(clk) - 0.5, +1)) begin
            sampled = V(vin);
        end
        V(out) <+ sampled;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("clk", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_source("vin", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(0.5, abs=1e-12)

    def test_combined_cross_evaluates_all_detectors_before_or(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module combined_cross_probe(a, b, out);
    input voltage a;
    input voltage b;
    output voltage out;
    integer count;
    analog begin
        @(initial_step) begin
            count = 0;
        end
        @(cross(V(a) - 0.5, +1) or cross(V(b) - 0.5, -1)) begin
            count = count + 1;
        end
        V(out) <+ count;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("a", lambda t: 0.0 if t < 10e-9 else 1.0)
        sim.add_source("b", lambda t: 1.0 if t < 10e-9 else 0.0)
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=20e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(1.0, abs=1e-12)

    def test_cross_event_body_sees_trigger_node_on_post_cross_side(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module strict_window_probe(vin, vss, out);
    input voltage vin;
    input voltage vss;
    output voltage out;
    real target;
    analog begin
        @(initial_step) begin
            target = 0.0;
        end
        @(cross(V(vin,vss) - 0.3, +1) or cross(V(vin,vss) - 0.6, +1) or
          cross(V(vin,vss) - 0.3, -1) or cross(V(vin,vss) - 0.6, -1)) begin
            if ((V(vin,vss) > 0.3) && (V(vin,vss) < 0.6))
                target = 1.0;
            else
                target = 0.0;
        end
        V(out) <+ target;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("vin", ramp(0.0, 0.9, 0.0, 90e-9))
        sim.add_source("vss", dc(0.0))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=90e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"].max() == pytest.approx(1.0, abs=1e-12)

    def test_cross_event_body_nudges_subtracted_branch_nodes(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module strict_diff_comparator(vinp, vinn, out);
    input voltage vinp;
    input voltage vinn;
    output voltage out;
    real target;
    analog begin
        @(initial_step) begin
            target = 0.0;
        end
        @(cross(V(vinp) - V(vinn), 0)) begin
            if (V(vinp) > V(vinn))
                target = 1.0;
            else
                target = 0.0;
        end
        V(out) <+ target;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("vinp", ramp(0.0, 0.9, 0.0, 10e-9))
        sim.add_source("vinn", dc(0.45))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"].max() == pytest.approx(1.0, abs=1e-12)

    def test_cross_direction_zero_float_detects_any_direction(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module any_cross_probe(vin, out);
    input voltage vin;
    output voltage out;
    integer count;
    analog begin
        @(initial_step) count = 0;
        @(cross(V(vin) - 0.5, 0.0, 1p)) count = count + 1;
        V(out) <+ count;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("vin", pwl([0.0, 10e-9, 20e-9], [0.0, 1.0, 0.0]))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=20e-9,
            tstep=5e-9,
            max_step=5e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(2.0, abs=1e-12)

    def test_same_step_cross_event_bodies_follow_chronological_order(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module ordered_threshold_probe(vin, out);
    input voltage vin;
    output voltage out;
    real state;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(vin) - 0.65, 0)) state = 2.0;
        @(cross(V(vin) - 0.50, 0)) state = 1.0;
        V(out) <+ state;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("vin", ramp(0.2, 0.7, 0.0, 10e-9))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(2.0, abs=1e-12)

    def test_cross_event_samples_simultaneous_nontrigger_source_post_side(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module simultaneous_sample(clk, err, out);
    input voltage clk;
    input voltage err;
    output voltage out;
    real target;
    analog begin
        @(initial_step) begin
            target = 0.0;
        end
        @(cross(V(clk) - 0.5, +1)) begin
            if (V(err) > 0.5)
                target = 1.0;
            else
                target = 0.0;
        end
        V(out) <+ target;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("clk", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_source("err", ramp(1.0, 0.0, 0.0, 10e-9))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"].max() == pytest.approx(0.0, abs=1e-12)

    def test_combined_cross_merges_simultaneous_trigger_directions(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module simultaneous_thermometer_step(b0, b1, out);
    input voltage b0;
    input voltage b1;
    output voltage out;
    integer code;
    analog begin
        @(initial_step) begin
            code = 0;
        end
        @(cross(V(b0) - 0.5, 0) or cross(V(b1) - 0.5, 0)) begin
            code = ((V(b1) > 0.5) ? 2 : 0) + ((V(b0) > 0.5) ? 1 : 0);
        end
        V(out) <+ code;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("b0", ramp(1.0, 0.0, 0.0, 10e-9))
        sim.add_source("b1", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(2.0, abs=1e-12)

    def test_exact_threshold_touch_does_not_force_strict_post_side(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module exact_touch_brownout(vin, out);
    input voltage vin;
    output voltage out;
    real target;
    analog begin
        @(initial_step) begin
            target = 1.0;
        end
        @(cross(V(vin) - 0.5, -1)) begin
            if (V(vin) < 0.5)
                target = 0.0;
            else
                target = 1.0;
        end
        V(out) <+ target;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        sim = Simulator()
        sim.add_source("vin", ramp(1.0, 0.5, 0.0, 10e-9))
        sim.add_model(ModelCls())
        sim.record("out")
        result = sim.run(
            tstop=10e-9,
            tstep=10e-9,
            max_step=10e-9,
            skip_source_error_control=True,
        )

        assert result.signals["out"][-1] == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# Compiled model features: timer, final_step, $temperature, $vt
# ===========================================================================

class TestTimerEvent:
    """Test timer event compilation/runtime for periodic and absolute-time forms."""

    VA_SRC = """\
`include "disciplines.vams"
module timer_test(out);
    output voltage out;
    integer count;
    analog begin
        @(initial_step) begin
            count = 0;
        end
        @(timer(0.0, 10e-9)) begin
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

    VA_SRC_ABSOLUTE = """\
`include "disciplines.vams"
module timer_abs_test(out);
    output voltage out;
    real next_t;
    integer count;
    analog begin
        @(initial_step) begin
            count = 0;
            next_t = 10e-9;
        end
        @(timer(next_t)) begin
            count = count + 1;
            next_t = next_t + 10e-9;
        end
        V(out) <+ count;
    end
endmodule
"""

    def test_timer_absolute_time_rearms_from_state(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC_ABSOLUTE)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=50e-9, tstep=1e-9)
        final_val = result.signals["out"][-1]
        assert final_val == pytest.approx(5.0, abs=1.0)

    VA_SRC_DIVISION_TYPES = """\
`include "disciplines.vams"
module division_type_probe(out_real, out_int);
    output voltage out_real;
    output voltage out_int;
    parameter integer navg = 5;
    integer code;
    real ratio;
    analog begin
        @(initial_step) begin
            code = 512;
        end
        ratio = 1.0 * code / 1023.0;
        V(out_real) <+ ratio;
        V(out_int) <+ navg / 2;
    end
endmodule
"""

    def test_integer_division_does_not_override_real_literals(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC_DIVISION_TYPES)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out_real", "out_int")
        result = sim.run(tstop=1e-9, tstep=1e-9)

        assert result.signals["out_real"][-1] == pytest.approx(512.0 / 1023.0)
        assert result.signals["out_int"][-1] == pytest.approx(2.0)

    def test_dollar_floor_alias_matches_spectre_math_function(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
module dollar_floor_probe(vin, out);
    input voltage vin;
    output voltage out;
    integer code;
    analog begin
        code = $floor(V(vin) / 0.25);
        V(out) <+ code;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("vin", dc(0.62))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-9)

        assert "math.floor" in ModelCls._generated_code
        assert result.signals["out"][-1] == pytest.approx(2.0)

    def test_tanh_math_function_matches_spectre_alias(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
module tanh_probe(vin, out);
    input voltage vin;
    output voltage out;
    analog begin
        V(out) <+ tanh(V(vin));
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("vin", dc(0.5))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-9)

        assert "math.tanh" in ModelCls._generated_code
        assert result.signals["out"][-1] == pytest.approx(math.tanh(0.5))

    def test_transition_three_arg_form_uses_rise_for_fall(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
module transition_three_arg_probe(out);
    output voltage out;
    analog begin
        V(out) <+ transition(1.0, 0.0, 10p);
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)

        assert "1e-11, 1e-11" in ModelCls._generated_code

    VA_SRC_TIMER_TO_TRANSITION_CROSS = """\
`include "disciplines.vams"
module timer_cross_probe(out, seen);
    output voltage out;
    output voltage seen;
    integer state;
    real next_t;
    real seen_t;
    analog begin
        @(initial_step) begin
            state = 0;
            next_t = 10e-9;
            seen_t = -1.0;
        end
        @(timer(next_t)) begin
            state = 1 - state;
            next_t = next_t + 10e-9;
        end
        @(cross(V(out) - 0.5, +1)) begin
            seen_t = $abstime;
        end
        V(out) <+ transition(state ? 1.0 : 0.0, 0.0, 2e-9, 2e-9);
        V(seen) <+ seen_t;
    end
endmodule
"""

    def test_timer_driven_transition_cross_hits_edge_midpoint(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC_TIMER_TO_TRANSITION_CROSS)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("seen")
        result = sim.run(tstop=15e-9, tstep=1e-9)

        seen = result.signals["seen"][-1]
        assert seen == pytest.approx(11e-9, abs=1e-12)

    VA_SRC_VAR_PERIOD = """\
`include "disciplines.vams"
module timer_var_period(out);
    output voltage out;
    integer count;
    real p;
    analog begin
        @(initial_step) begin
            count = 0;
            p = 2e-9;
        end
        @(timer(1e-9, p)) begin
            count = count + 1;
            p = (count & 1) ? 1e-9 : 2e-9;
        end
        V(out) <+ count;
    end
endmodule
"""

    def test_timer_variable_period_is_stable(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC_VAR_PERIOD)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=10e-9, tstep=1e-9)

        # Expected fire times: 1,2,4,5,7,8,10 ns => 7 fires.
        assert result.signals["out"][-1] == pytest.approx(7.0, abs=1.0)


class TestCrossToleranceAndLastCrossing:
    VA_SRC = """\
`include "disciplines.vams"
module last_crossing_probe(vin, out, tlast);
    input voltage vin;
    output voltage out;
    output voltage tlast;
    real lc;
    integer seen;
    analog begin
        @(initial_step) seen = 0;
        lc = last_crossing(V(vin) - 0.5, +1, 0.0, 1e-12);
        @(cross(V(vin) - 0.5, +1, 0.0, 1e-12))
            seen = 1;
        V(out) <+ seen;
        V(tlast) <+ lc;
    end
endmodule
"""

    def test_last_crossing_tracks_rising_edge(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("vin", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_model(model)
        sim.record("tlast")
        result = sim.run(tstop=12e-9, tstep=1e-9)

        # Ramp crosses 0.5 near 5ns.
        assert result.signals["tlast"][-1] == pytest.approx(5e-9, abs=2e-9)


class TestWhileStatement:
    VA_SRC = """\
`include "disciplines.vams"
module while_wrap_test(out);
    output voltage out;
    real x;
    analog begin
        @(initial_step) begin
            x = 12.0;
            while (x > 5.0) x = x - 10.0;
        end
        V(out) <+ x;
    end
endmodule
"""

    def test_while_in_event_body_executes_until_condition_clears(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-10)

        assert result.signals["out"][-1] == pytest.approx(2.0, abs=1e-12)


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


class TestDifferentialContribution:
    """Test V(a,b) <+ expr semantics through a compiled VA module."""

    VA_SRC = """\
`include "disciplines.vams"
module diff_contrib_test(vss, outp, outn);
    inout voltage vss, outp, outn;
    analog begin
        V(outn, vss) <+ 0.2;
        V(outp, outn) <+ 0.5;
    end
endmodule
"""

    def test_differential_contribution_references_node2_voltage(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module
        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.record("outp")
        sim.record("outn")
        result = sim.run(tstop=1e-9, tstep=0.1e-9)
        assert result.signals["outn"][-1] == pytest.approx(0.2)
        assert result.signals["outp"][-1] == pytest.approx(0.7)


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

    def test_direct_filename_fstrobe_does_not_crash_evas(self, tmp_path):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        outfile = tmp_path / "direct_output.txt"
        outfile_s = str(outfile).replace("\\", "/")
        src = f"""\
`include "disciplines.vams"
module direct_fileio_test(clk);
    input voltage clk;
    integer count;
        analog begin
            @(initial_step) count = 0;
            @(cross(V(clk) - 0.5, 1)) begin
                count = count + 1;
            $fstrobe("{outfile_s}", "direct %d", count);
            end
        end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("clk", pulse(v_lo=0.0, v_hi=1.0, period=20e-9,
                                     rise=0.1e-9, fall=0.1e-9, duty=0.5))
        sim.add_model(model)
        sim.run(tstop=45e-9, tstep=1e-9)

        assert outfile.exists()
        assert outfile.read_text().strip().splitlines() == ["direct 1", "direct 2", "direct 3"]

    def test_fopen_accepts_string_parameter_filename(self, tmp_path):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        outfile = tmp_path / "param_filename.txt"
        outfile_s = str(outfile).replace("\\", "/")
        src = f"""\
`include "disciplines.vams"
module param_fileio_test(clk);
    input voltage clk;
    parameter string filename = "{outfile_s}";
    integer fd;
    analog begin
        @(final_step) begin
            fd = $fopen(filename, "w");
            $fstrobe(fd, "ok");
            $fclose(fd);
        end
    end
endmodule
"""
        mod = parse(src)
        assert [p.name for p in mod.parameters] == ["filename"]

        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.run(tstop=1e-9, tstep=1e-9)

        assert outfile.exists()
        assert outfile.read_text().strip() == "ok"

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


class TestSpectreRestrictedOperators:
    def test_conditional_idtmod_is_rejected(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module bad_idt(out, rst, vss);
output out;
electrical out;
input rst;
electrical rst;
inout vss;
electrical vss;
real x;
analog begin
    if (V(rst, vss) > 0.5)
        x = idtmod(1.0, 0.0, 1.0);
    else
        x = 0.0;
    V(out, vss) <+ x;
end
endmodule
"""
        mod = parse(src)
        with pytest.raises(CompilationError, match="idtmod"):
            compile_module(mod)

    def test_conditional_transition_is_rejected(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module bad_trans(out, sel, vss);
output out;
electrical out;
input sel;
electrical sel;
inout vss;
electrical vss;
analog begin
    if (V(sel, vss) > 0.5)
        V(out, vss) <+ transition(1.0, 0.0, 1p, 1p);
    else
        V(out, vss) <+ 0.0;
end
endmodule
"""
        mod = parse(src)
        with pytest.raises(CompilationError, match="transition"):
            compile_module(mod)

    def test_event_body_contribution_is_rejected(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module bad_event_contrib(out, clk, vss);
output out;
electrical out;
input clk;
electrical clk;
inout vss;
electrical vss;
analog begin
    @(cross(V(clk, vss) - 0.5, +1))
        V(out, vss) <+ 1.0;
end
endmodule
"""
        mod = parse(src)
        with pytest.raises(CompilationError, match="contribution"):
            compile_module(mod)

    def test_unknown_function_is_rejected(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module bad_unknown_func(out);
output out;
electrical out;
real x;
analog begin
    x = $itor(3);
    V(out) <+ x;
end
endmodule
"""
        mod = parse(src)
        with pytest.raises(CompilationError, match=r"\$itor"):
            compile_module(mod)

    def test_transition_of_continuous_signal_is_accepted(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module bad_transition_continuous(out, vdd, vss);
output out;
electrical out;
inout vdd, vss;
electrical vdd, vss;
real vh;
analog begin
    vh = V(vdd, vss);
    V(out, vss) <+ transition(vh, 0.0, 1p, 1p);
end
endmodule
"""
        mod = parse(src)
        # Spectre/Virtuoso accepts continuous-affine transition() targets such as
        # transition(V(vres_p) + offset * 0.5, 0). EVAS evaluates transition()
        # dynamically, so the static piecewise-constant guard is removed.
        # compile_module must succeed without raising.
        compile_module(mod)


class TestInitialStepVisibility:
    VA_SRC = """\
`include "disciplines.vams"
module initial_step_precompute(out, vdd, vss);
    output voltage out;
    input voltage vdd;
    input voltage vss;
    real vh;
    real vl;
    real x;
    analog begin
        vh = V(vdd);
        vl = V(vss);
        @(initial_step) begin
            x = 0.5;
            if (x > vh) x = vh;
            if (x < vl) x = vl;
        end
        V(out) <+ x;
    end
endmodule
"""

    def test_initial_step_sees_preceding_continuous_assignments(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.add_source("vdd", dc(0.9))
        sim.add_source("vss", dc(0.0))
        sim.record("out")
        result = sim.run(tstop=1e-9, tstep=1e-10)

        assert result.signals["out"][0] == pytest.approx(0.5, abs=1e-12)


class TestSlewOperator:
    VA_SRC = """\
`include "disciplines.vams"
module slew_test(out, vss);
    output voltage out;
    inout voltage vss;
    real target;
    analog begin
        @(initial_step) target = 0.0;
        @(timer(5e-9)) target = 1.0;
        V(out, vss) <+ slew(target, 1e8, 1e8);
    end
endmodule
"""

    def test_slew_compiles_and_rate_limits(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        mod = parse(self.VA_SRC)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_model(model)
        sim.add_source("vss", dc(0.0))
        sim.record("out")
        result = sim.run(tstop=15e-9, tstep=1e-9)

        # Before timer at 5ns, output stays near 0.
        idx_4ns = min(range(len(result.time)), key=lambda i: abs(result.time[i] - 4e-9))
        assert result.signals["out"][idx_4ns] == pytest.approx(0.0, abs=1e-9)

        # Near 6ns (about 1ns after target step), slew should have moved ~0.1V.
        idx_6ns = min(range(len(result.time)), key=lambda i: abs(result.time[i] - 6e-9))
        assert 0.0 < result.signals["out"][idx_6ns] < 0.5

        # By 15ns, with 1e8 V/s rise limit and 10ns elapsed since 5ns step, output nears 1V.
        assert result.signals["out"][-1] == pytest.approx(1.0, abs=0.1)
