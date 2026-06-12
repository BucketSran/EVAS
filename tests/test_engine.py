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
import shutil
import subprocess
from pathlib import Path

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
from evas.simulator.backend import compile_module
from evas.simulator.rust_coverage import (
    audit_veriloga_paths,
    estimate_event_transition_plan_profiles,
    estimate_event_transition_profiles,
)
from evas.compiler.parser import parse


RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core_or_skip():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


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

    def test_transition_target_condition_inserts_breakpoint(self):
        src = r"""
`include "disciplines.vams"
module transition_target_guard(VDD, VSS, guard_out, phase_out);
    input VDD, VSS;
    output guard_out, phase_out;
    electrical VDD, VSS, guard_out, phase_out;
    parameter real period = 8n;
    parameter real pulse_w = 1.5n;
    parameter integer points_per_period = 16;
    parameter real tedge = 40p;
    real cycle_start, next_cycle, phase, phase_norm, guard_target, vh, vl;
    analog begin
        vh = V(VDD);
        vl = V(VSS);
        @(initial_step) begin
            cycle_start = 0.0;
            next_cycle = period;
        end
        @(timer(next_cycle)) begin
            cycle_start = next_cycle;
            next_cycle = next_cycle + period;
        end
        $bound_step(period / points_per_period);
        phase = $abstime - cycle_start;
        if (phase < 0.0) phase = 0.0;
        if (phase > period) phase = period;
        guard_target = (phase <= pulse_w) ? 1.0 : 0.0;
        phase_norm = phase / period;
        V(guard_out) <+ vl + (vh - vl) * transition(guard_target, 0.0, tedge, tedge);
        V(phase_out) <+ vl + (vh - vl) * phase_norm;
    end
endmodule
"""
        Model = compile_module(parse(src))
        model = Model()
        sim = Simulator()
        sim.add_source("VDD", dc(1.0))
        sim.add_source("VSS", dc(0.0))
        sim.add_model(model)
        sim.record("guard_out", "phase_out")

        result = sim.run(
            tstop=2.2e-9,
            tstep=0.5e-9,
            max_step=0.5e-9,
            skip_source_error_control=True,
        )

        assert Model._transition_target_probe_count == 1
        assert sim._perf_stats["transition_target_breakpoint_clamps"] >= 1
        low_times = [
            t
            for t, guard in zip(result.time, result.signals["guard_out"])
            if guard < 0.2
        ]
        assert low_times
        assert min(low_times) < 1.7e-9

    def test_rust_sim_program_dc_source_record_matches_default(self):
        _build_rust_core_or_skip()

        def run_model(use_rust: bool):
            sim = Simulator()
            sim.add_source("vdd", dc(1.8))
            sim.record("vdd")
            result = sim.run(
                tstop=5e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["vdd"].tolist() == pytest.approx(
            default_result.signals["vdd"].tolist()
        )
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_source_record_enabled"] == 1
        assert rust_sim._perf_stats["rust_full_model_fastpath_enabled"] == 1

    def test_rust_sim_program_pulse_breakpoints_match_default(self):
        _build_rust_core_or_skip()

        def run_model(use_rust: bool):
            sim = Simulator()
            sim.add_source(
                "clk",
                pulse(
                    0.0,
                    1.0,
                    period=4e-9,
                    rise=1e-9,
                    fall=1e-9,
                    width=1e-9,
                ),
            )
            sim.record("clk")
            result = sim.run(
                tstop=5e-9,
                tstep=2e-9,
                record_step=2e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["clk"].tolist() == pytest.approx(
            default_result.signals["clk"].tolist()
        )
        assert rust_sim._perf_stats["rust_sim_program_source_breakpoints"] > 0
        assert rust_sim._perf_stats["rust_sim_program_source_record_enabled"] == 1

    def test_rust_sim_program_pwl_source_record_matches_default(self):
        _build_rust_core_or_skip()

        def run_model(use_rust: bool):
            sim = Simulator()
            sim.add_source(
                "vin",
                pwl(
                    [0.0, 1e-9, 3e-9, 4e-9],
                    [0.0, 0.5, 1.5, 0.25],
                ),
            )
            sim.record("vin")
            result = sim.run(
                tstop=4e-9,
                tstep=2e-9,
                record_step=2e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["vin"].tolist() == pytest.approx(
            default_result.signals["vin"].tolist()
        )
        assert rust_sim._perf_stats["rust_sim_program_source_breakpoints"] > 0
        assert rust_sim._perf_stats["rust_sim_program_source_record_enabled"] == 1

    def test_rust_sim_program_continuous_linear_model_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_gain(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    analog begin
        V(vout) <+ gain * V(vin) + 0.1;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", dc(0.4))
            sim.add_model(model)
            sim.record("VOUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust_sim._perf_stats["rust_sim_program_continuous_linear_ops"] == 1
        assert rust_sim._perf_stats["rust_sim_program_source_record_enabled"] == 1

    def test_rust_sim_program_continuous_state_write_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_stateassign(vin, vout);
    input voltage vin;
    output voltage vout;
    real acc = 0.0;
    analog begin
        acc = V(vin) + 0.1;
        V(vout) <+ acc;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", dc(0.25))
            sim.add_model(model)
            sim.record("VOUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim, model

        default_result, _default_sim, default_model = run_model(False)
        rust_result, rust_sim, rust_model = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust_model.state["acc"] == pytest.approx(default_model.state["acc"])
        assert rust_sim._perf_stats["rust_sim_program_state_count"] == 1
        assert rust_sim._perf_stats["rust_sim_program_continuous_linear_ops"] == 2
        assert rust_sim._perf_stats["rust_sim_program_source_record_enabled"] == 1

    def test_rust_sim_program_body_differential_contribution_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_diff_contrib(ref, out);
    input voltage ref;
    output voltage out;
    integer ready = 0;
    analog begin
        @(initial_step) ready = 1;
        V(out, ref) <+ 0.25;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"ref": "REF", "out": "OUT"}
            sim = Simulator()
            sim.add_source("REF", dc(0.5))
            sim.add_model(model)
            sim.record("OUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim, model

        default_result, _default_sim, default_model = run_model(False)
        rust_result, rust_sim, rust_model = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["OUT"].tolist() == pytest.approx(
            default_result.signals["OUT"].tolist()
        )
        assert rust_result.signals["OUT"][-1] == pytest.approx(0.75)
        assert rust_model.state["ready"] == pytest.approx(default_model.state["ready"])
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_always_body_count"] == 1

    def test_rust_sim_program_event_body_case_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_case_body(out);
    output voltage out;
    integer code = 0;
    real y = 0.0;
    analog begin
        @(initial_step) begin
            code = 2;
            case (code)
                0: y = 0.1;
                1, 2: y = 0.8;
                default: y = 0.2;
            endcase
        end
        V(out) <+ y;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim, model

        default_result, _default_sim, default_model = run_model(False)
        rust_result, rust_sim, rust_model = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert rust_result.signals["out"][-1] == pytest.approx(0.8)
        assert rust_model.state["code"] == pytest.approx(default_model.state["code"])
        assert rust_model.state["y"] == pytest.approx(default_model.state["y"])
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1

    def test_rust_sim_program_final_step_updates_state_after_trace(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_final_step(out);
    output voltage out;
    integer flag = 0;
    analog begin
        @(initial_step) flag = 1;
        @(final_step) flag = 7;
        V(out) <+ flag;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim, model

        default_result, _default_sim, default_model = run_model(False)
        rust_result, rust_sim, rust_model = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert rust_result.signals["out"][-1] == pytest.approx(1.0)
        assert default_model.state["flag"] == 7
        assert rust_model.state["flag"] == 7
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_event_fires"] >= 2

    def test_rust_sim_program_body_tan_tanh_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_tan_tanh(vin, out);
    input voltage vin;
    output voltage out;
    integer ready = 0;
    analog begin
        @(initial_step) ready = 1;
        V(out) <+ tan(V(vin) - 0.5) + tanh(V(vin));
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"vin": "VIN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("VIN", dc(0.25))
            sim.add_model(model)
            sim.record("OUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["OUT"].tolist() == pytest.approx(
            default_result.signals["OUT"].tolist()
        )
        assert rust_result.signals["OUT"][-1] == pytest.approx(
            math.tan(-0.25) + math.tanh(0.25)
        )
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1

    def test_rust_sim_program_pure_continuous_nonlinear_body_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_multitone_body(out);
    output voltage out;
    parameter real a1 = 0.2;
    parameter real a2 = 0.05;
    parameter real f1 = 1.0e9;
    parameter real f2 = 2.0e9;
    analog begin
        V(out) <+ a1 * sin(6.283185307179586 * f1 * $abstime)
                + a2 * sin(6.283185307179586 * f2 * $abstime);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            result = sim.run(
                tstop=2e-9,
                tstep=0.25e-9,
                record_step=0.25e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        expected_times = [idx * 0.25e-9 for idx in range(9)]
        assert rust_result.time.tolist() == pytest.approx(expected_times)
        for time_value, signal_value in zip(
            rust_result.time.tolist(),
            rust_result.signals["OUT"].tolist(),
        ):
            expected = 0.2 * math.sin(6.283185307179586 * 1.0e9 * time_value)
            expected += 0.05 * math.sin(6.283185307179586 * 2.0e9 * time_value)
            assert signal_value == pytest.approx(expected, abs=1e-12)
        for time_value, signal_value in zip(
            default_result.time.tolist(),
            default_result.signals["OUT"].tolist(),
        ):
            expected = 0.2 * math.sin(6.283185307179586 * 1.0e9 * time_value)
            expected += 0.05 * math.sin(6.283185307179586 * 2.0e9 * time_value)
            assert signal_value == pytest.approx(expected, abs=1e-12)
        assert rust_sim._perf_stats["rust_sim_program_enabled"] == 1
        assert rust_sim._perf_stats["rust_sim_program_always_body_count"] == 1
        assert rust_sim._perf_stats["rust_sim_program_continuous_linear_ops"] == 0

    def test_rust_sim_program_rdist_normal_noise_behavior(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module rustsim_noise_body(vin, out);
    input voltage vin;
    output voltage out;
    parameter real sigma = 0.1;
    analog begin
        V(out) <+ V(vin) + sigma * $rdist_normal(0, 0, 1);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        def run_model(use_rust: bool):
            model = ModelCls()
            model.node_map = {"vin": "VIN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("VIN", dc(1.0))
            sim.add_model(model)
            sim.record("VIN")
            sim.record("OUT")
            result = sim.run(
                tstop=100e-9,
                tstep=1e-9,
                record_step=1e-9,
                rust_full_model_fastpath=use_rust,
                rust_full_model_required=use_rust,
                rust_required=use_rust,
                skip_source_error_control=True,
            )
            return result, sim

        default_result, _default_sim = run_model(False)
        result, sim = run_model(True)
        assert result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert result.signals["VIN"].tolist() == pytest.approx(
            default_result.signals["VIN"].tolist()
        )

        noises = result.signals["OUT"] - result.signals["VIN"]
        mean = sum(noises.tolist()) / len(noises)
        var = sum((float(value) - mean) ** 2 for value in noises.tolist()) / len(noises)
        assert var**0.5 > 0.01
        assert max(abs(float(value)) for value in noises.tolist()) > 0.05
        assert sim._perf_stats["rust_sim_program_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_always_body_count"] == 1

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

    def test_rust_full_model_required_rejects_python_fallback(self):
        sim = Simulator()
        sim.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.record("vin")

        with pytest.raises(RuntimeError, match="evas-rust full-model path"):
            sim.run(
                tstop=1e-9,
                tstep=1e-9,
                rust_full_model_fastpath=True,
                rust_full_model_required=True,
            )

        assert sim._perf_stats["rust_full_model_required_failures"] == 1
        assert sim._perf_stats["rust_full_model_fastpath_enabled"] == 0
        assert sim._perf_stats["rust_sim_program_rejections"] == 1

    def test_indexed_snapshot_profile_records_sidecar_timing_without_changing_result(self):
        sim = Simulator()
        sim.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.record("vin")

        result = sim.run(tstop=2e-9, tstep=1e-9, indexed_snapshot_profile=True)

        assert result.signals["vin"][-1] == pytest.approx(1.0)
        assert sim._perf_stats["indexed_prev_snapshots"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["indexed_snapshot_mismatches"] == 0
        assert sim._indexed_snapshot_stats["max_abs_diff"] == pytest.approx(0.0)
        assert sim._indexed_snapshot_stats["snapshots"] == sim._perf_stats["steps_total"]
        assert "dict_prev_snapshot_s" in sim._profile_times
        assert "indexed_prev_snapshot_s" in sim._profile_times

    def test_model_eval_profile_is_explicitly_opted_in(self):
        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "vin", "out": "vout"}

            def evaluate(self, node_voltages, time):
                self._set_output("out", self._get_voltage("inp", node_voltages), node_voltages)

        default = Simulator()
        default.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        default.add_model(MirrorModel())
        default.record("vout")
        default_result = default.run(tstop=2e-9, tstep=1e-9, profile_sections=True)

        profiled = Simulator()
        profiled.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        profiled.add_model(MirrorModel())
        profiled.record("vout")
        profiled_result = profiled.run(tstop=2e-9, tstep=1e-9, profile_model_eval=True)

        assert default_result.signals["vout"].tolist() == pytest.approx(
            profiled_result.signals["vout"].tolist()
        )
        assert default._model_profile_stats == {}
        assert "model[0] MirrorModel" in profiled._model_profile_stats
        stats = profiled._model_profile_stats["model[0] MirrorModel"]
        assert stats["evaluate_calls"] == profiled._perf_stats["steps_total"]
        assert stats["prepare_step_s"] >= 0.0
        assert stats["evaluate_s"] >= 0.0
        assert stats["post_update_s"] >= 0.0

    def test_model_io_profile_counts_voltage_reads_and_output_writes(self):
        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "vin", "out": "vout"}

            def evaluate(self, node_voltages, time):
                self._set_output("out", self._get_voltage("inp", node_voltages), node_voltages)

        default = Simulator()
        default.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        default.add_model(MirrorModel())
        default.record("vout")
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        profiled = Simulator()
        profiled.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        profiled.add_model(MirrorModel())
        profiled.record("vout")
        profiled_result = profiled.run(tstop=2e-9, tstep=1e-9, profile_model_io=True)

        assert default_result.signals["vout"].tolist() == pytest.approx(
            profiled_result.signals["vout"].tolist()
        )
        assert default._model_io_profile_stats == {}
        stats = profiled._model_io_profile_stats
        assert stats["voltage_reads"] >= profiled._perf_stats["steps_total"]
        assert stats["output_writes"] == stats["voltage_reads"]
        assert stats["voltage_read_local_nodes"] == 1
        assert stats["voltage_read_external_nodes"] == 1
        assert stats["output_write_nodes"] == 1
        assert stats["voltage_read_event_contexts"] == 0
        assert stats["voltage_read_missing_nodes"] == 0

    def test_indexed_arrays_preserve_source_record_and_error_scan_results(self):
        default = Simulator()
        default.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        default.record("vin")
        default_result = default.run(tstop=2e-9, tstep=1e-9, reltol=1e-3, vabstol=1e-6)

        indexed = Simulator()
        indexed.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        indexed.record("vin")
        indexed_result = indexed.run(
            tstop=2e-9,
            tstep=1e-9,
            reltol=1e-3,
            vabstol=1e-6,
            indexed_arrays=True,
        )

        assert indexed_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert indexed_result.step_sizes.tolist() == pytest.approx(default_result.step_sizes.tolist())
        assert indexed_result.signals["vin"].tolist() == pytest.approx(
            default_result.signals["vin"].tolist()
        )
        assert indexed._perf_stats["indexed_array_source_updates"] > 0
        assert indexed._perf_stats["indexed_array_record_reads"] == len(indexed_result.time)
        assert indexed._perf_stats["indexed_array_record_id_reads"] == len(indexed_result.time)
        assert indexed._perf_stats["indexed_array_err_ratio_reads"] > 0
        assert indexed._perf_stats["indexed_array_mismatches"] == 0
        assert indexed._indexed_array_stats["max_abs_diff"] == pytest.approx(0.0)
        assert indexed._indexed_array_stats["record_id_reads"] == len(indexed_result.time)
        assert "indexed_array_prev_snapshot_s" in indexed._profile_times
        assert "indexed_array_sync_s" in indexed._profile_times

    def test_indexed_arrays_use_rust_record_scan_when_array_loop_enabled(self):
        _build_rust_core_or_skip()

        def build_sim(*, indexed_arrays=False):
            sim = Simulator()
            for idx in range(80):
                sim.add_source(f"n{idx}", dc(float(idx)))
            sim.record("n0")
            sim.record("n17")
            sim.record("n79")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                indexed_arrays=indexed_arrays,
            )
            return sim, result

        default, default_result = build_sim(indexed_arrays=False)
        rust, rust_result = build_sim(indexed_arrays=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        for name in ("n0", "n17", "n79"):
            assert rust_result.signals[name].tolist() == pytest.approx(
                default_result.signals[name].tolist()
            )
        assert rust._perf_stats["rust_array_loop_enabled"] == 1
        assert rust._perf_stats["rust_array_record_scans"] == len(rust_result.time)
        assert rust._perf_stats["rust_array_record_values"] == (
            len(rust_result.time) * 3
        )
        assert rust._perf_stats["rust_array_record_fallbacks"] == 0

    def test_indexed_arrays_build_model_io_plan_without_changing_mapped_output(self):
        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "vin", "out": "vout"}

            def evaluate(self, node_voltages, time):
                self._set_output("out", self._get_voltage("inp", node_voltages), node_voltages)

        default = Simulator()
        default.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        default.add_model(MirrorModel())
        default.record("vout")
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        indexed = Simulator()
        indexed.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
        indexed.add_model(MirrorModel())
        indexed.record("vout")
        indexed_result = indexed.run(tstop=2e-9, tstep=1e-9, indexed_arrays=True)

        assert indexed_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert indexed_result.step_sizes.tolist() == pytest.approx(default_result.step_sizes.tolist())
        assert indexed_result.signals["vout"].tolist() == pytest.approx(
            default_result.signals["vout"].tolist()
        )
        assert indexed._perf_stats["indexed_model_io_models"] == 1
        assert indexed._perf_stats["indexed_model_io_mapped_ports"] == 2
        assert indexed._perf_stats["indexed_model_io_outputs"] == 1
        assert indexed._perf_stats["indexed_model_io_refreshes"] >= 1
        assert indexed._perf_stats["indexed_output_write_throughs"] > 0
        assert indexed._perf_stats["indexed_output_write_through_nodes"] == 1
        assert indexed._perf_stats["indexed_post_model_sync_repairs"] == 0
        assert indexed._perf_stats["indexed_voltage_reads"] > 0
        assert indexed._perf_stats["indexed_voltage_read_nodes"] == 1
        assert indexed._perf_stats["indexed_voltage_read_fallbacks"] == 0
        assert indexed._perf_stats["indexed_voltage_probe_mismatches"] == 0
        assert indexed._perf_stats["indexed_voltage_probe_missing_nodes"] == 0
        assert indexed._indexed_model_io_stats["output_count"] == 1
        assert indexed._indexed_array_stats["max_abs_diff"] == pytest.approx(0.0)
        assert indexed._indexed_array_stats["output_write_through_nodes"] == 1
        assert indexed._indexed_voltage_read_stats["reads"] > 0
        assert indexed._indexed_voltage_read_stats["fallbacks"] == 0
        assert indexed._indexed_voltage_probe_stats["mismatches"] == 0

    def test_indexed_state_storage_preserves_stateful_waveform_and_counts_writes(self):
        src = """\
`include "disciplines.vams"
module stateful(out);
    output voltage out;
    real x = 0.0;
    integer code = 0;
    real accum[0:1];
    analog begin
        x = x + 0.25;
        code = code + 1;
        accum[1] = x + code;
        V(out) <+ accum[1];
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(indexed_state_storage=False):
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                indexed_state_storage=indexed_state_storage,
            )
            return result, sim

        default_result, default_sim = run_model(False)
        indexed_result, indexed_sim = run_model(True)

        assert indexed_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert indexed_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert default_sim._perf_stats["indexed_state_storage_enabled"] == 0
        assert indexed_sim._perf_stats["indexed_state_storage_enabled"] == 1
        assert indexed_sim._perf_stats["indexed_state_storage_models"] == 1
        assert indexed_sim._perf_stats["indexed_state_storage_scalar_slots"] == 2
        assert indexed_sim._perf_stats["indexed_state_storage_integer_slots"] == 1
        assert indexed_sim._perf_stats["indexed_state_storage_array_slots"] == 2
        assert indexed_sim._perf_stats["indexed_state_scalar_writes_total"] > 0
        assert indexed_sim._perf_stats["indexed_state_array_writes_total"] > 0
        assert indexed_sim._perf_stats["indexed_state_array_oob_writes_total"] == 0

    def test_compiler_emits_094_body_ir_metadata_for_general_body(self):
        src = """\
`include "disciplines.vams"
module body_ir_gain(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    real acc = 0.0;
    integer code = 0;
    analog begin
        acc = gain * V(vin);
        if (acc > 0.5) code = 1; else code = 0;
        V(vout) <+ code ? acc : 0.0;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        assert ModelCls._rust_body_ir_rejection_reason == "ok"
        assert len(ModelCls._rust_body_ir_stmt_ops) == 7
        assert len(ModelCls._rust_body_ir_expr_ops) > 0
        assert ModelCls._rust_body_ir_node_names == ("vin", "vout")
        assert ModelCls._rust_body_ir_param_names == ("gain",)
        assert ModelCls._rust_body_ir_state_names == ("acc", "code")
        assert ModelCls._rust_body_ir_target_node_slots == (1,)
        assert ModelCls._rust_body_ir_target_state_slots == (0, 1)

    def test_current_rust_coverage_audit_reports_body_ir_candidates(self, tmp_path):
        body_path = tmp_path / "body.va"
        body_path.write_text(
            """\
`include "disciplines.vams"
module body_ir_gain(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    real acc = 0.0;
    analog begin
        acc = acc + gain * V(vin);
        V(vout) <+ acc;
    end
endmodule
"""
        )
        event_path = tmp_path / "event.va"
        event_path.write_text(
            """\
`include "disciplines.vams"
module event_only(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ state;
    end
endmodule
"""
        )
        event_transition_path = tmp_path / "event_transition.va"
        event_transition_path.write_text(
            """\
`include "disciplines.vams"
module event_transition(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ transition(state, 0.0, 1n, 1n);
    end
endmodule
"""
        )

        summary = audit_veriloga_paths((body_path, event_path, event_transition_path))

        assert summary.total_files == 3
        assert summary.compile_ok == 3
        assert summary.rust_body_ir_candidates == 1
        reasons = {
            row.path: row.rust_body_ir_rejection_reason
            for row in summary.rows
        }
        tags = {
            row.path: set(row.rust_body_ir_rejection_tags)
            for row in summary.rows
        }
        assert reasons[str(body_path)] == "ok"
        assert reasons[str(event_path)] in {"event_cross", "event_statement"}
        assert {"event_cross", "event_statement"} <= tags[str(event_path)]

        estimates = estimate_event_transition_profiles(summary.rows)
        assert estimates["event_transition_core"]["candidate_count"] == 2
        assert str(event_path) in set(
            estimates["event_transition_core"]["candidate_paths"]
        )
        assert str(event_transition_path) in set(
            estimates["event_transition_core"]["candidate_paths"]
        )
        plan_estimates = estimate_event_transition_plan_profiles(summary.rows)
        assert plan_estimates["event_transition_core"]["candidate_count"] == 2
        assert str(event_path) in set(
            plan_estimates["event_transition_core"]["candidate_paths"]
        )
        assert str(event_transition_path) in set(
            plan_estimates["event_transition_core"]["candidate_paths"]
        )
        event_transition_cls = compile_module(parse(event_transition_path.read_text()))
        assert "event_transition_core" in (
            event_transition_cls._event_transition_plan_profiles
        )
        assert event_transition_cls._event_transition_plan_event_count == 2
        assert event_transition_cls._event_transition_plan_transition_count == 1

    def test_engine_reports_event_transition_plan_metadata(self):
        src = """\
`include "disciplines.vams"
module event_transition(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ transition(state, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        model.node_map = {"clk": "CLK", "out": "OUT"}
        sim = Simulator()
        sim.add_source("CLK", dc(0.0))
        sim.add_model(model)
        sim.record("OUT")
        sim.run(tstop=1e-9, tstep=1e-9)

        assert sim._perf_stats["rust_event_transition_plan_core_candidate_models"] == 1
        assert sim._perf_stats["rust_event_transition_plan_core_event_statements"] == 2
        assert sim._perf_stats["rust_event_transition_plan_core_due_triggers"] == 2
        assert sim._perf_stats["rust_event_transition_plan_core_transitions"] == 1
        assert sim._perf_stats["rust_event_transition_plan_ordered_v1_candidate_models"] == 1
        assert sim._perf_stats["rust_event_transition_plan_side_effect_candidate_models"] == 1

    def test_rust_event_transition_shadow_executes_and_matches_python(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_transition(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ transition(state, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        model.node_map = {"clk": "CLK", "out": "OUT"}
        sim = Simulator()
        sim.add_source("CLK", pulse(0.0, 1.0, period=2e-9, width=1e-9))
        sim.add_model(model)
        sim.record("OUT")
        sim.run(
            tstop=3e-9,
            tstep=1e-9,
            rust_event_transition_shadow=True,
            rust_required=True,
        )

        assert sim._perf_stats["rust_event_transition_shadow_available"] == 1
        assert sim._perf_stats["rust_event_transition_shadow_enabled"] == 1
        assert sim._perf_stats["rust_event_transition_shadow_models"] == 1
        assert sim._perf_stats["rust_event_transition_shadow_calls_total"] > 0
        assert sim._perf_stats["rust_event_transition_shadow_matches_total"] > 0
        assert sim._perf_stats["rust_event_transition_shadow_mismatches_total"] == 0
        assert sim._perf_stats["rust_event_transition_shadow_errors_total"] == 0

    def test_rust_event_transition_production_matches_default_waveform(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_transition_prod(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ transition(state, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"clk": "CLK", "out": "OUT"}
            sim = Simulator()
            sim.add_source("CLK", pulse(0.0, 1.0, period=2e-9, width=1e-9))
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(tstop=3e-9, tstep=1e-9, record_step=250e-12)

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=3e-9,
            tstep=1e-9,
            record_step=250e-12,
            rust_event_transition_production=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust_model.state["state"] == pytest.approx(ref_model.state["state"])
        assert rust._perf_stats["rust_event_transition_production_requested"] == 1
        assert rust._perf_stats["rust_event_transition_production_available"] == 1
        assert rust._perf_stats["rust_event_transition_production_enabled"] == 1
        assert rust._perf_stats["rust_event_transition_production_models"] == 1
        assert rust._perf_stats["rust_event_transition_production_calls_total"] > 0
        assert rust._perf_stats["rust_event_transition_production_executed_total"] > 0
        assert rust._perf_stats["rust_event_transition_production_fallbacks_total"] == 0
        assert rust._perf_stats["rust_event_transition_production_state_writes_total"] > 0
        assert rust._perf_stats["rust_event_transition_production_output_writes_total"] > 0
        assert rust._perf_stats["rust_event_transition_production_fired_events_total"] > 0
        assert ref._perf_stats["rust_event_transition_production_requested"] == 0

    def test_rust_sim_program_event_transition_strict_full_model_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_transition_evas2(clk, out);
    input voltage clk;
    output voltage out;
    real state = 0.0;
    analog begin
        @(initial_step) state = 0.0;
        @(cross(V(clk)-0.5, +1)) state = 1.0;
        V(out) <+ transition(state, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"clk": "CLK", "out": "OUT"}
            sim = Simulator()
            sim.add_source("CLK", pulse(0.0, 1.0, period=2e-9, width=1e-9))
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(tstop=3e-9, tstep=1e-9, record_step=250e-12)

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=3e-9,
            tstep=1e-9,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
        )

        time_deltas = [
            abs(float(rust_t) - float(ref_t))
            for ref_t, rust_t in zip(ref_result.time, rust_result.time)
        ]
        signal_deltas = [
            abs(float(rust_v) - float(ref_v))
            for ref_v, rust_v in zip(
                ref_result.signals["OUT"],
                rust_result.signals["OUT"],
            )
        ]
        max_time_delta = max(time_deltas)
        max_signal_delta = max(signal_deltas)
        assert len(rust_result.time) == len(ref_result.time)
        assert max_time_delta < 1.0e-13
        # The strict Rust path owns its scheduler, so it can differ from the
        # Python adaptive scheduler by tens of femtoseconds. For this fixture
        # the transition slope is 1 / 1ns, so the observed output delta must be
        # explained by the time-grid delta rather than event/body semantics.
        assert max_signal_delta <= max_time_delta / 1.0e-9 + 1.0e-9
        assert rust_model.state["state"] == pytest.approx(ref_model.state["state"])
        assert rust._perf_stats["rust_full_model_required_failures"] == 0
        assert rust._perf_stats["rust_full_model_fastpath_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_count"] == 2
        assert rust._perf_stats["rust_sim_program_body_stmt_ops"] == 2
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["rust_sim_program_event_fires"] > 0
        assert rust._perf_stats["rust_sim_program_transition_breakpoints"] > 0
        assert rust._perf_stats["rust_event_transition_production_requested"] == 0
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_post_update_cross_refreshes_outputs(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module post_cross_evas2(inp, mid, out);
    input voltage inp;
    output voltage mid, out;
    integer seen = 0;
    analog begin
        V(mid) <+ V(inp);
        @(cross(V(mid)-0.5, +1)) seen = 1;
        V(out) <+ seen;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"inp": "IN", "mid": "MID", "out": "OUT"}
            sim = Simulator()
            sim.add_source(
                "IN",
                pulse(0.0, 1.0, delay=0.5e-9, period=4e-9, width=2e-9),
            )
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust_model.state["seen"] == pytest.approx(ref_model.state["seen"])
        assert rust_model.state["seen"] == pytest.approx(1.0)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_count"] == 3
        assert rust._perf_stats["rust_sim_program_event_fires"] >= 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_transition_breakpoint_drains_post_cross(self):
        _build_rust_core_or_skip()
        driver_src = """\
`include "disciplines.vams"
module transition_breakpoint_driver(clk, sig);
    input voltage clk;
    output voltage sig;
    integer q;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk)-0.5, +1)) q = 1;
        V(sig) <+ transition(q ? 1.0 : 0.0, 0.0, 10p, 10p);
    end
endmodule
"""
        observer_src = """\
`include "disciplines.vams"
module transition_breakpoint_observer(sig, seen_time, out);
    input voltage sig;
    output voltage seen_time, out;
    real seen_t;
    integer seen;
    analog begin
        @(initial_step) begin
            seen_t = -1.0;
            seen = 0;
        end
        @(cross(V(sig)-0.5, +1)) begin
            seen_t = $abstime;
            seen = 1;
        end
        V(seen_time) <+ seen_t;
        V(out) <+ seen;
    end
endmodule
"""
        DriverCls = compile_module(parse(driver_src))
        ObserverCls = compile_module(parse(observer_src))

        driver = DriverCls()
        driver.node_map = {"clk": "CLK", "sig": "SIG"}
        observer = ObserverCls()
        observer.node_map = {
            "sig": "SIG",
            "seen_time": "SEEN_T",
            "out": "OUT",
        }
        sim = Simulator()
        sim.add_source("CLK", pwl([0.0, 1e-9], [0.0, 1.0]))
        sim.add_model(driver)
        sim.add_model(observer)
        sim.record("SIG")
        sim.record("SEEN_T")
        sim.record("OUT")

        result = sim.run(
            tstop=1.2e-9,
            tstep=1e-9,
            record_step=1e-9,
            max_step=1e-9,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert sim._perf_stats["rust_full_model_required_failures"] == 0
        assert sim._perf_stats["rust_sim_program_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_transition_breakpoints"] >= 1
        assert sim._perf_stats["generic_executor_runs"] == 0
        assert observer.state["seen"] == pytest.approx(1.0)
        assert observer.state["seen_t"] == pytest.approx(510e-12, abs=1.0e-15)
        assert all(
            float(right) + 1.0e-18 >= float(left)
            for left, right in zip(result.time, result.time[1:])
        )
        assert result.signals["OUT"][2] == pytest.approx(1.0)
        assert result.signals["SEEN_T"][2] == pytest.approx(510e-12, abs=1.0e-15)

    def test_rust_sim_program_cross_acceptance_uses_accepted_event_time(self):
        _build_rust_core_or_skip()
        driver_src = """\
`include "disciplines.vams"
module accepted_cross_driver(clk, sig);
    input voltage clk;
    output voltage sig;
    integer q;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk)-0.5, +1)) q = 1;
        V(sig) <+ transition(q ? 1.0 : 0.0, 0.0, 10p, 10p);
    end
endmodule
"""
        observer_src = """\
`include "disciplines.vams"
module accepted_cross_observer(sig, seen_time);
    input voltage sig;
    output voltage seen_time;
    real seen_t;
    analog begin
        @(initial_step) seen_t = -1.0;
        @(cross(V(sig)-0.5, +1)) seen_t = $abstime;
        V(seen_time) <+ seen_t;
    end
endmodule
"""
        DriverCls = compile_module(parse(driver_src))
        ObserverCls = compile_module(parse(observer_src))

        def run_with_factor(factor):
            driver = DriverCls()
            driver.node_map = {"clk": "CLK", "sig": "SIG"}
            observer = ObserverCls()
            observer.node_map = {"sig": "SIG", "seen_time": "SEEN_T"}
            sim = Simulator()
            sim.add_source("CLK", pwl([0.0, 1e-9], [0.0, 1.0]))
            sim.add_model(driver)
            sim.add_model(observer)
            sim.record("SIG")
            sim.record("SEEN_T")
            result = sim.run(
                tstop=1.2e-9,
                tstep=1e-9,
                record_step=1e-9,
                max_step=1e-9,
                rust_full_model_fastpath=True,
                rust_full_model_required=True,
                rust_required=True,
                skip_source_error_control=True,
                cross_acceptance_slack_factor=factor,
            )
            assert sim._perf_stats["rust_sim_program_enabled"] == 1
            assert sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
            return result, observer

        default_result, default_observer = run_with_factor(0.0)
        accepted_result, accepted_observer = run_with_factor(0.25)

        assert default_observer.state["seen_t"] == pytest.approx(510e-12, abs=1e-15)
        assert default_result.signals["SEEN_T"][2] == pytest.approx(510e-12, abs=1e-15)
        assert accepted_observer.state["seen_t"] == pytest.approx(512.5e-12, abs=1e-15)
        assert accepted_result.signals["SEEN_T"][2] == pytest.approx(512.5e-12, abs=1e-15)
        assert accepted_result.time[1] == pytest.approx(502.5e-12, abs=1e-15)

    def test_rust_sim_program_rdist_event_body_preserves_transition_ramp(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_noise_transition(vin, out);
    input voltage vin;
    output voltage out;
    parameter real sigma = 0.0;
    parameter real dt = 0.5n;
    real noise, vout;
    analog begin
        @(initial_step) begin
            noise = 0.0;
            vout = 0.0;
        end
        @(timer(dt)) begin
            noise = sigma * $rdist_normal(0, 0, 1);
            vout = V(vin) + noise;
        end
        V(out) <+ transition(vout, 0, dt/10, dt/10);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        model.node_map = {"vin": "VIN", "out": "OUT"}
        sim = Simulator()
        sim.add_source("VIN", dc(1.0))
        sim.add_model(model)
        sim.record("OUT")

        result = sim.run(
            tstop=0.6e-9,
            tstep=0.5e-9,
            max_step=0.5e-9,
            record_step=0.5e-9,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert sim._perf_stats["rust_sim_program_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert list(result.time[:3]) == pytest.approx([0.0, 0.5e-9, 0.55e-9])
        assert result.signals["OUT"][1] == pytest.approx(0.0)
        assert result.signals["OUT"][2] == pytest.approx(1.0)

    def test_rust_sim_program_transition_uses_model_default_transition(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module default_transition_driver(clk, out);
    input voltage clk;
    output voltage out;
    integer q;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk)-0.5, +1)) q = 1;
        V(out) <+ transition(q ? 1.0 : 0.0, 0.0);
    end
endmodule
"""
        ModelCls = compile_module(parse(src), default_transition=30e-12)

        def build_sim():
            model = ModelCls()
            model.node_map = {"clk": "CLK", "out": "OUT"}
            sim = Simulator()
            sim.add_source("CLK", pwl([0.0, 1e-9], [0.0, 1.0]))
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(
            tstop=0.6e-9,
            tstep=0.1e-9,
            record_step=0.1e-9,
            max_step=0.1e-9,
            skip_source_error_control=True,
        )

        rust = build_sim()
        rust_result = rust.run(
            tstop=0.6e-9,
            tstep=0.1e-9,
            record_step=0.1e-9,
            max_step=0.1e-9,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0
        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"]), abs=5e-4
        )
        assert any(
            math.isclose(float(t), 530e-12, abs_tol=1.0e-15)
            and math.isclose(float(v), 1.0, abs_tol=1.0e-12)
            for t, v in zip(rust_result.time, rust_result.signals["OUT"])
        )

    def test_rust_sim_program_cross_model_transition_output_uses_post_phase(self):
        _build_rust_core_or_skip()
        driver_src = """\
`include "disciplines.vams"
module transition_driver(clk, sig);
    input voltage clk;
    output voltage sig;
    integer state;
    analog begin
        @(initial_step) state = 0;
        @(cross(V(clk)-0.5, +1)) state = 1;
        V(sig) <+ transition(state ? 1.0 : 0.0, 20p, 30p, 30p);
    end
endmodule
"""
        observer_src = """\
`include "disciplines.vams"
module transition_observer(clk, sig, delay_ps, seen_time);
    input voltage clk, sig;
    output voltage delay_ps, seen_time;
    real t_start;
    real measured_ps;
    real seen_t;
    integer armed;
    analog begin
        @(initial_step) begin
            t_start = 0.0;
            measured_ps = -1.0;
            seen_t = -1.0;
            armed = 0;
        end
        @(cross(V(clk)-0.5, +1)) begin
            t_start = $abstime;
            armed = 1;
        end
        @(cross(V(sig)-0.5, +1)) begin
            if (armed == 1) begin
                seen_t = $abstime;
                measured_ps = ($abstime - t_start) * 1e12;
                armed = 0;
            end
        end
        V(delay_ps) <+ measured_ps;
        V(seen_time) <+ seen_t;
    end
endmodule
"""
        DriverCls = compile_module(parse(driver_src))
        ObserverCls = compile_module(parse(observer_src))

        def build_sim():
            driver = DriverCls()
            driver.node_map = {"clk": "CLK", "sig": "SIG"}
            observer = ObserverCls()
            observer.node_map = {
                "clk": "CLK",
                "sig": "SIG",
                "delay_ps": "DELAY_PS",
                "seen_time": "SEEN_TIME",
            }
            sim = Simulator()
            sim.add_source(
                "CLK",
                pulse(
                    0.0,
                    1.0,
                    period=10e-9,
                    width=5e-9,
                    rise=1e-9,
                    fall=1e-9,
                    delay=0.0,
                ),
            )
            sim.add_model(driver)
            sim.add_model(observer)
            sim.record("DELAY_PS")
            sim.record("SEEN_TIME")
            return sim, observer

        ref, ref_observer = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            max_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_observer = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            max_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert rust._perf_stats["rust_full_model_required_failures"] == 0
        assert rust._perf_stats["rust_full_model_fastpath_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0
        assert rust_observer.state["seen_t"] == pytest.approx(
            ref_observer.state["seen_t"],
            abs=1.0e-15,
        )
        assert rust_observer.state["measured_ps"] == pytest.approx(
            ref_observer.state["measured_ps"],
            abs=1.0e-6,
        )
        assert rust_result.signals["SEEN_TIME"][-1] == pytest.approx(
            ref_result.signals["SEEN_TIME"][-1],
            abs=1.0e-15,
        )
        assert rust_result.signals["DELAY_PS"][-1] == pytest.approx(
            35.0,
            abs=1.0e-6,
        )

    def test_rust_sim_program_continuous_transition_target_does_not_microstep(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module continuous_transition_target(vin, out);
    input voltage vin;
    output voltage out;
    parameter real tr = 200p;
    real y;
    analog begin
        y = V(vin);
        V(out) <+ transition(y, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("VIN", pwl([0.0, 2e-9, 4e-9], [0.1, 0.8, 0.2]))
            sim.add_model(model)
            sim.record("VIN")
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(
            tstop=4e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust = build_sim()
        rust_result = rust.run(
            tstop=4e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert rust._perf_stats["rust_full_model_required_failures"] == 0
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["rust_sim_program_points"] < 1000
        assert len(rust_result.time) == rust._perf_stats["rust_sim_program_points"]

        def interp(times, values, sample_time):
            time_list = list(times)
            value_list = list(values)
            for idx in range(1, len(time_list)):
                t0 = time_list[idx - 1]
                t1 = time_list[idx]
                if sample_time <= t1:
                    if t1 == t0:
                        return value_list[idx]
                    frac = (sample_time - t0) / (t1 - t0)
                    return value_list[idx - 1] + frac * (value_list[idx] - value_list[idx - 1])
            return value_list[-1]

        sample_times = [idx * 250e-12 for idx in range(17)]
        assert [interp(rust_result.time, rust_result.signals["VIN"], t) for t in sample_times] == pytest.approx(
            [interp(ref_result.time, ref_result.signals["VIN"], t) for t in sample_times],
            abs=2.0e-5,
        )
        rust_out = [interp(rust_result.time, rust_result.signals["OUT"], t) for t in sample_times]
        assert all(math.isfinite(value) for value in rust_out)
        assert min(rust_out) >= 0.09
        assert max(rust_out) <= 0.81
        assert rust_out[8] > rust_out[0] + 0.45
        assert rust_out[-1] < rust_out[8] - 0.35

    def test_rust_sim_program_conditional_integer_weighted_transition_target(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module weighted_transition_target(en, b0, b1, b2, b3, out);
    input voltage en, b0, b1, b2, b3;
    output voltage out;
    parameter real vth = 0.5;
    parameter real vlo = 0.0;
    parameter real vhi = 1.0;
    parameter real tr = 1n;
    integer code = 0;
    real y = 0.0;
    analog begin
        if (V(en) > vth) begin
            code = (V(b0) > vth ? 1 : 0)
                 + (V(b1) > vth ? 2 : 0)
                 + (V(b2) > vth ? 4 : 0)
                 + (V(b3) > vth ? 8 : 0);
        end else begin
            code = 0;
        end
        y = vlo + (vhi - vlo) * code / 15.0;
        V(out) <+ transition(y, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {
                "en": "EN",
                "b0": "B0",
                "b1": "B1",
                "b2": "B2",
                "b3": "B3",
                "out": "OUT",
            }
            sim = Simulator()
            sim.add_source("B2", dc(1.0))
            sim.add_source("EN", dc(1.0))
            sim.add_source("B3", dc(1.0))
            sim.add_source("B0", dc(1.0))
            sim.add_source("B1", dc(0.0))
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        time_deltas = [
            abs(float(rust_t) - float(ref_t))
            for ref_t, rust_t in zip(ref_result.time, rust_result.time)
        ]
        signal_deltas = [
            abs(float(rust_v) - float(ref_v))
            for ref_v, rust_v in zip(
                ref_result.signals["OUT"],
                rust_result.signals["OUT"],
            )
        ]
        max_time_delta = max(time_deltas)
        max_signal_delta = max(signal_deltas)
        assert len(rust_result.time) == len(ref_result.time)
        assert max_time_delta < 1.0e-13
        assert max_signal_delta <= max_time_delta / 100e-12 + 1.0e-9
        assert rust_result.signals["OUT"][-1] > 0.8
        assert rust_model.state["code"] == pytest.approx(ref_model.state["code"])
        assert rust_model.state["y"] == pytest.approx(ref_model.state["y"])
        assert rust_model.state["code"] == pytest.approx(13.0)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_always_body_count"] == 1
        assert rust._perf_stats["rust_sim_program_body_stmt_ops"] == 6
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["rust_sim_program_event_fires"] > 0
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_threshold_transition_target_skips_zero_width_code(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module threshold_weighted_transition_target(code_0, code_1, code_2, code_3, vref, vss, aout);
    input code_0, code_1, code_2, code_3, vref, vss;
    output aout;
    electrical code_0, code_1, code_2, code_3, vref, vss, aout;
    parameter real vth = 0.45;
    parameter real tr = 500p;
    integer code;
    real y;
    analog begin
        code = (V(code_0) > vth ? 1 : 0)
             + (V(code_1) > vth ? 2 : 0)
             + (V(code_2) > vth ? 4 : 0)
             + (V(code_3) > vth ? 8 : 0);
        y = V(vss) + (V(vref) - V(vss)) * code / 15.0;
        V(aout, vss) <+ transition(y - V(vss), 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {
                "code_0": "B0",
                "code_1": "B1",
                "code_2": "B2",
                "code_3": "B3",
                "vref": "VREF",
                "vss": "VSS",
                "aout": "OUT",
            }
            sim = Simulator()
            sim.add_source("VREF", dc(0.9))
            sim.add_source("VSS", dc(0.0))
            sim.add_source("B0", pwl([39.5e-9, 40.0e-9], [0.9, 0.0]))
            sim.add_source("B1", pwl([39.5e-9, 40.0e-9], [0.9, 0.0]))
            sim.add_source("B2", pwl([39.5e-9, 40.0e-9], [0.0, 0.9]))
            sim.add_source("B3", dc(0.0))
            sim.add_model(model)
            sim.record("OUT")
            return sim

        rust = build_sim()
        rust_result = rust.run(
            tstop=41e-9,
            tstep=500e-12,
            record_step=500e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        def interp(times, values, sample_time):
            time_list = list(times)
            value_list = list(values)
            for idx in range(1, len(time_list)):
                t0 = time_list[idx - 1]
                t1 = time_list[idx]
                if sample_time <= t1:
                    if t1 == t0:
                        return value_list[idx]
                    frac = (sample_time - t0) / (t1 - t0)
                    return value_list[idx - 1] + frac * (value_list[idx] - value_list[idx - 1])
            return value_list[-1]

        out_39p500 = interp(rust_result.time, rust_result.signals["OUT"], 39.500e-9)
        out_39p875 = interp(rust_result.time, rust_result.signals["OUT"], 39.875e-9)
        out_40p250 = interp(rust_result.time, rust_result.signals["OUT"], 40.250e-9)
        assert rust._perf_stats["rust_full_model_required_failures"] == 0
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert out_39p500 == pytest.approx(0.18, abs=1.0e-9)
        assert out_39p875 < 0.225
        assert out_40p250 == pytest.approx(0.24, abs=1.0e-6)

    def test_rust_sim_program_static_state_array_transition_target(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module state_array_transition_target(en, b0, b1, out);
    input voltage en, b0, b1;
    output voltage out;
    parameter real vth = 0.5;
    parameter real tr = 100p;
    integer bits[0:1];
    real code = 0.0;
    analog begin
        if (V(en) > vth) begin
            bits[0] = V(b0) > vth ? 1 : 0;
            bits[1] = V(b1) > vth ? 1 : 0;
        end else begin
            bits[0] = 0;
            bits[1] = 0;
        end
        code = bits[0] + 2.0 * bits[1];
        V(out) <+ transition(code / 3.0, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"en": "EN", "b0": "B0", "b1": "B1", "out": "OUT"}
            sim = Simulator()
            sim.add_source("EN", dc(1.0))
            sim.add_source("B0", dc(1.0))
            sim.add_source("B1", dc(0.0))
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        time_deltas = [
            abs(float(rust_t) - float(ref_t))
            for ref_t, rust_t in zip(ref_result.time, rust_result.time)
        ]
        signal_deltas = [
            abs(float(rust_v) - float(ref_v))
            for ref_v, rust_v in zip(
                ref_result.signals["OUT"],
                rust_result.signals["OUT"],
            )
        ]
        max_time_delta = max(time_deltas)
        max_signal_delta = max(signal_deltas)
        assert len(rust_result.time) == len(ref_result.time)
        assert max_time_delta < 1.0e-13
        assert max_signal_delta <= max_time_delta / 100e-12 + 1.0e-9
        assert rust_model.arrays["bits"][0] == ref_model.arrays["bits"][0] == 1
        assert rust_model.arrays["bits"][1] == ref_model.arrays["bits"][1] == 0
        assert rust_model.state["code"] == pytest.approx(ref_model.state["code"])
        assert rust_model.state["code"] == pytest.approx(1.0)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_always_body_count"] == 1
        assert rust._perf_stats["rust_sim_program_body_stmt_ops"] == 8
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_static_bus_bitshift_transition_targets(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module bus_bitshift_transition_target(din, out);
    input voltage [0:3] din;
    output voltage [0:1] out;
    parameter real vth = 0.5;
    parameter real tr = 100p;
    integer code = 0;
    integer inv = 0;
    real y = 0.0;
    analog begin
        if (V(din[3]) > vth) begin
            code = ((V(din[0]) > vth ? 1 : 0) << 0)
                 | ((V(din[1]) > vth ? 1 : 0) << 1)
                 | ((V(din[2]) > vth ? 1 : 0) << 2);
        end else begin
            inv = 0;
        end
        inv = (~(code >> 1)) & 3;
        y = (code + inv) / 15.0;
        V(out[0]) <+ transition(y, 0.0, tr, tr);
        V(out[1]) <+ transition(code & 1, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_source("din[0]", dc(1.0))
            sim.add_source("din[1]", dc(0.0))
            sim.add_source("din[2]", dc(1.0))
            sim.add_source("din[3]", dc(1.0))
            sim.add_model(model)
            sim.record("out[0]")
            sim.record("out[1]")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["out[0]"]) == pytest.approx(
            list(ref_result.signals["out[0]"])
        )
        assert list(rust_result.signals["out[1]"]) == pytest.approx(
            list(ref_result.signals["out[1]"])
        )
        assert rust_model.state["code"] == pytest.approx(ref_model.state["code"])
        assert rust_model.state["inv"] == pytest.approx(ref_model.state["inv"])
        assert rust_model.state["y"] == pytest.approx(ref_model.state["y"])
        assert rust_model.state["code"] == pytest.approx(5.0)
        assert rust_model.state["inv"] == pytest.approx(1.0)
        assert rust_model.state["y"] == pytest.approx(0.4)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_always_body_count"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 2
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_static_for_bus_state_transition_target(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module for_bus_state_transition_target(din, out);
    input voltage [0:3] din;
    output voltage out;
    parameter real vth = 0.5;
    parameter real tr = 100p;
    integer i = 0;
    integer bits[0:3];
    integer code = 0;
    real y = 0.0;
    analog begin
        code = 0;
        for (i = 0; i < 4; i = i + 1) begin
            bits[i] = V(din[i]) > vth ? 1 : 0;
            code = code + (bits[i] << i);
        end
        y = code / 15.0;
        V(out) <+ transition(y, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_source("din[0]", dc(1.0))
            sim.add_source("din[1]", dc(0.0))
            sim.add_source("din[2]", dc(1.0))
            sim.add_source("din[3]", dc(1.0))
            sim.add_model(model)
            sim.record("out")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["out"]) == pytest.approx(
            list(ref_result.signals["out"])
        )
        assert rust_model.state["code"] == pytest.approx(ref_model.state["code"])
        assert rust_model.state["code"] == pytest.approx(13.0)
        assert rust_model.state["y"] == pytest.approx(ref_model.state["y"])
        assert rust_model.arrays["bits"][0] == ref_model.arrays["bits"][0] == 1
        assert rust_model.arrays["bits"][1] == ref_model.arrays["bits"][1] == 0
        assert rust_model.arrays["bits"][2] == ref_model.arrays["bits"][2] == 1
        assert rust_model.arrays["bits"][3] == ref_model.arrays["bits"][3] == 1
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_always_body_count"] == 1
        assert rust._perf_stats["rust_sim_program_body_stmt_ops"] >= 12
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_top_level_static_for_transition_outputs(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module top_level_static_for_transition_outputs(out);
    output voltage [0:3] out;
    parameter real tr = 100p;
    integer i = 0;
    real val[0:3];
    analog begin
        @(initial_step) begin
            for (i = 0; i < 4; i = i + 1) begin
                val[i] = i == 2 ? 0.8 : 0.2;
            end
        end
        for (i = 0; i < 4; i = i + 1) begin
            V(out[i]) <+ transition(val[i], 0.0, tr, tr);
        end
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            for idx in range(4):
                sim.record(f"out[{idx}]")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        for idx in range(4):
            assert list(rust_result.signals[f"out[{idx}]"]) == pytest.approx(
                list(ref_result.signals[f"out[{idx}]"])
            )
            assert rust_model.arrays["val"][idx] == pytest.approx(
                ref_model.arrays["val"][idx]
            )
        assert rust_model.arrays["val"][2] == pytest.approx(0.8)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 4
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_one_based_state_array_transition_target(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module one_based_state_array_transition_target(out);
    output voltage out;
    parameter real vdd = 0.9;
    parameter real tr = 100p;
    integer bits[8:1];
    analog begin
        @(initial_step) begin
            bits[8] = 1;
            bits[1] = 0;
        end
        V(out) <+ transition(bits[8] ? vdd : 0.0, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["out"]) == pytest.approx(
            list(ref_result.signals["out"])
        )
        assert rust_model.arrays["bits"][8] == ref_model.arrays["bits"][8] == 1
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_event_body_ignores_display_strobe_for_waveform(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_strobe_noop(out);
    output voltage out;
    parameter real tr = 100p;
    integer code = 0;
    analog begin
        @(initial_step) begin
            code = 3;
            $strobe("code=%d", code);
            $display("code=%d", code);
        end
        V(out) <+ transition(code, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["out"]) == pytest.approx(
            list(ref_result.signals["out"])
        )
        assert rust_model.state["code"] == pytest.approx(ref_model.state["code"])
        assert rust_model.state["code"] == pytest.approx(3.0)
        assert rust_model._strobe_log == ref_model._strobe_log == []
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_count"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_timer_strobe_matches_python_side_effect(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_strobe(out);
    output voltage out;
    integer code = 0;
    analog begin
        @(timer(1n)) begin
            code = 5;
            $strobe("code=%d", code);
        end
        V(out) <+ code;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            return sim, model

        ref, ref_model = build_sim()
        ref.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert rust_model._strobe_log == ref_model._strobe_log
        assert rust_model._strobe_log == [(pytest.approx(1e-9), "code=5")]
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_side_effects"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_bound_step_scaled_transition_and_direct_output(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module bound_scaled_transition(vdd, vss, out, mon);
    input voltage vdd, vss;
    output voltage out, mon;
    parameter real tr = 100p;
    real vh;
    real vl;
    real tnext = 0.0;
    integer state = 0;
    analog begin
        vh = V(vdd);
        vl = V(vss);
        @(initial_step) begin
            state = 0;
            tnext = 0.5n;
        end
        @(timer(tnext)) begin
            state = 1 - state;
            $bound_step(50p);
            tnext = tnext + 0.5n;
        end
        V(out) <+ vl + (vh - vl) * transition(state ? 1.0 : 0.0, 0.0, tr, tr);
        V(mon) <+ vl + (vh - vl) * (state ? 0.75 : 0.25);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            sim = Simulator()
            sim.add_source("vdd", dc(0.9))
            sim.add_source("vss", dc(0.0))
            sim.add_model(model)
            sim.record("out")
            sim.record("mon")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=1.5e-9,
            tstep=250e-12,
            record_step=50e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=1.5e-9,
            tstep=250e-12,
            record_step=50e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["out"]) == pytest.approx(
            list(ref_result.signals["out"]),
            abs=1.0e-9,
        )
        assert list(rust_result.signals["mon"]) == pytest.approx(
            list(ref_result.signals["mon"]),
            abs=1.0e-12,
        )
        assert rust_model.state["state"] == pytest.approx(ref_model.state["state"])
        assert max(rust_result.signals["out"]) > 0.8
        assert min(rust_result.signals["out"]) < 0.1
        assert max(rust_result.signals["mon"]) == pytest.approx(0.675)
        assert min(rust_result.signals["mon"]) == pytest.approx(0.225)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_nested_state_machine_guard_write_order(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module nested_state_machine(clk, cmp, out, rdy);
    input voltage clk, cmp;
    output voltage out, rdy;
    parameter real vth = 0.5;
    parameter real tr = 100p;
    integer state = 0;
    integer bit = 0;
    real ready = 0.0;
    analog begin
        @(initial_step) begin
            state = 0;
            bit = 1;
            ready = 0.0;
        end
        @(cross(V(clk) - vth, +1)) begin
            if (state == 0) begin
                if (V(cmp) < vth) bit = 0;
                state = 1;
                ready = 0.0;
            end else if (state == 1) begin
                bit = bit ? 0 : 1;
                ready = 1.0;
                state = 2;
            end else begin
                bit = 1;
                ready = 0.0;
                state = 0;
            end
        end
        V(out) <+ transition(bit, 0.0, tr, tr);
        V(rdy) <+ transition(ready, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"clk": "CLK", "cmp": "CMP", "out": "OUT", "rdy": "RDY"}
            sim = Simulator()
            sim.add_source("CLK", pulse(0.0, 1.0, period=1e-9, width=0.5e-9))
            sim.add_source("CMP", dc(0.0))
            sim.add_model(model)
            sim.record("OUT")
            sim.record("RDY")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=4e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=4e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"]),
            abs=1.0e-3,
        )
        assert list(rust_result.signals["RDY"]) == pytest.approx(
            list(ref_result.signals["RDY"]),
            abs=1.0e-3,
        )
        assert rust_model.state["state"] == pytest.approx(ref_model.state["state"])
        assert rust_model.state["bit"] == pytest.approx(ref_model.state["bit"])
        assert rust_model.state["ready"] == pytest.approx(ref_model.state["ready"])
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_count"] == 2
        assert rust._perf_stats["rust_sim_program_transition_count"] == 2
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_sim_program_state_owned_timer_rearms_with_abstime(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module state_owned_timer(out);
    output voltage out;
    parameter real period = 1n;
    parameter real tr = 100p;
    integer bit = 0;
    real next_t = 0.0;
    analog begin
        @(initial_step) begin
            bit = 0;
            next_t = period;
        end
        @(timer(next_t)) begin
            bit = bit ? 0 : 1;
            next_t = $abstime + period;
        end
        V(out) <+ transition(bit, 0.0, tr, tr);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, ref_model = build_sim()
        ref_result = ref.run(
            tstop=3e-9,
            tstep=250e-12,
            record_step=250e-12,
            skip_source_error_control=True,
        )

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=3e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        time_deltas = [
            abs(float(rust_t) - float(ref_t))
            for ref_t, rust_t in zip(ref_result.time, rust_result.time)
        ]
        signal_deltas = [
            abs(float(rust_v) - float(ref_v))
            for ref_v, rust_v in zip(
                ref_result.signals["OUT"],
                rust_result.signals["OUT"],
            )
        ]
        max_time_delta = max(time_deltas)
        max_signal_delta = max(signal_deltas)
        assert len(rust_result.time) == len(ref_result.time)
        assert max_time_delta < 1.0e-13
        assert max_signal_delta <= max_time_delta / 100e-12 + 1.0e-9
        assert rust_model.state["bit"] == pytest.approx(ref_model.state["bit"])
        assert rust_model.state["next_t"] == pytest.approx(ref_model.state["next_t"])
        assert rust_model.state["next_t"] == pytest.approx(4e-9)
        assert rust._perf_stats["rust_sim_program_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert rust._perf_stats["rust_sim_program_event_count"] == 2
        assert rust._perf_stats["rust_sim_program_transition_count"] == 1
        assert rust._perf_stats["rust_sim_program_event_fires"] >= 4
        assert rust._perf_stats["generic_executor_runs"] == 0

    def test_rust_body_ir_production_matches_python_evaluate_and_counts_calls(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module body_ir_gain(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    real acc = 0.0;
    integer code = 0;
    analog begin
        acc = gain * V(vin);
        if (acc > 0.5) code = 1; else code = 0;
        V(vout) <+ code ? acc : 0.0;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_model(rust_body_ir=False):
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", dc(0.4))
            sim.add_model(model)
            sim.record("VOUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                rust_body_ir=rust_body_ir,
                rust_required=rust_body_ir,
            )
            return result, sim

        default_result, default_sim = run_model(False)
        rust_result, rust_sim = run_model(True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust_sim._perf_stats["rust_body_ir_requested"] == 1
        assert rust_sim._perf_stats["rust_body_ir_available"] == 1
        assert rust_sim._perf_stats["rust_body_ir_candidate_models"] == 1
        assert rust_sim._perf_stats["rust_body_ir_models"] == 1
        assert rust_sim._perf_stats["rust_body_ir_production_executed_total"] > 0
        assert rust_sim._perf_stats["rust_body_ir_production_node_writes_total"] > 0
        assert rust_sim._perf_stats["rust_body_ir_production_state_writes_total"] > 0
        assert default_sim._perf_stats["rust_body_ir_requested"] == 0

    def test_indexed_state_fastpath_reads_state_slots_without_changing_waveform(self):
        src = """\
`include "disciplines.vams"
module stateful(out);
    output voltage out;
    real x = 0.0;
    real cold = 7.0;
    integer code = 0;
    real accum[0:1];
    analog begin
        @(initial_step) cold = 11.0;
        x = x + 0.25;
        code = code + 1;
        accum[1] = x + code;
        V(out) <+ accum[1] + x;
    end
endmodule
"""
        DefaultModel = compile_module(parse(src))
        FastModel = compile_module(parse(src), indexed_state_fastpath_codegen=True)

        def run_model(model_cls, indexed_state_storage=False):
            model = model_cls()
            sim = Simulator()
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                indexed_state_storage=indexed_state_storage,
            )
            return result, sim, model

        default_result, _, _ = run_model(DefaultModel, False)
        fallback_result, fallback_sim, fallback_model = run_model(FastModel, False)
        indexed_result, indexed_sim, indexed_model = run_model(FastModel, True)

        assert "_st_x" in FastModel._generated_code
        assert "_st_cold" not in FastModel._generated_code
        assert "_state_values[0]" in FastModel._generated_code
        assert fallback_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert indexed_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert indexed_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert indexed_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert fallback_sim._perf_stats["indexed_state_scalar_reads_total"] == 0
        assert indexed_sim._perf_stats["indexed_state_scalar_reads_total"] > 0
        assert indexed_sim._perf_stats["indexed_state_array_reads_total"] > 0
        assert indexed_sim._perf_stats["indexed_state_scalar_writes_total"] > 0
        assert indexed_sim._perf_stats["indexed_state_array_writes_total"] > 0
        assert indexed_model.state["x"] == pytest.approx(fallback_model.state["x"])
        assert indexed_model.state["code"] == fallback_model.state["code"]
        assert indexed_model.state["cold"] == pytest.approx(fallback_model.state["cold"])

    def test_static_branch_fastpath_matches_default_and_counts_hits(self):
        src = """\
`include "disciplines.vams"
module pass_through(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ V(vin);
    end
endmodule
"""
        DefaultModel = compile_module(parse(src))
        FastModel = compile_module(parse(src), static_branch_fastpath_codegen=True)

        def run_model(model_cls, enabled):
            model = model_cls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            result = sim.run(
                tstop=2e-9,
                tstep=1e-9,
                static_branch_fastpath=enabled,
            )
            return result, sim, model

        default_result, default_sim, _ = run_model(DefaultModel, False)
        fast_result, fast_sim, fast_model = run_model(FastModel, True)

        assert fast_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert fast_result.step_sizes.tolist() == pytest.approx(default_result.step_sizes.tolist())
        assert fast_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert default_sim._perf_stats["static_branch_fastpath_codegen_models"] == 0
        assert fast_sim._perf_stats["static_branch_fastpath_codegen_models"] == 1
        assert fast_sim._perf_stats["static_branch_fastpath_static_read_nodes"] == 1
        assert fast_sim._perf_stats["static_branch_fastpath_static_write_nodes"] == 1
        assert fast_sim._perf_stats["static_branch_fastpath_fallbacks_total"] == 0
        assert fast_model._static_branch_fastpath_enabled is False

    def test_static_branch_fastpath_uses_indexed_node_ids_when_array_enabled(self):
        src = """\
`include "disciplines.vams"
module pass_through(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ V(vin);
    end
endmodule
"""
        DefaultModel = compile_module(parse(src))
        FastModel = compile_module(parse(src), static_branch_fastpath_codegen=True)

        default_model = DefaultModel()
        default_model.node_map = {"vin": "VIN", "vout": "VOUT"}
        default = Simulator()
        default.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        default.add_model(default_model)
        default.record("VOUT")
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        fast_model = FastModel()
        fast_model.node_map = {"vin": "VIN", "vout": "VOUT"}
        fast = Simulator()
        fast.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        fast.add_model(fast_model)
        fast.record("VOUT")
        fast_result = fast.run(
            tstop=2e-9,
            tstep=1e-9,
            indexed_arrays=True,
            static_branch_fastpath=True,
        )

        assert fast_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert fast_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert fast._perf_stats["static_branch_direct_array_models"] == 1
        assert fast._perf_stats["static_branch_direct_array_read_nodes"] == 1
        assert fast._perf_stats["static_branch_direct_array_write_nodes"] == 1
        assert fast._perf_stats["static_branch_fastpath_fallbacks_total"] == 0
        assert fast._perf_stats["indexed_voltage_reads"] == 0
        assert fast._perf_stats["indexed_output_write_throughs"] == 0
        assert fast._perf_stats["indexed_post_model_sync_repairs"] == 0

    def test_rust_static_eval_matches_default_for_static_affine_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 2.0 * V(vin) + 0.125;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        default_model = ModelCls()
        default_model.node_map = {"vin": "VIN", "vout": "VOUT"}
        default = Simulator()
        default.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        default.add_model(default_model)
        default.record("VOUT")
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust_model = ModelCls()
        rust_model.node_map = {"vin": "VIN", "vout": "VOUT"}
        rust = Simulator()
        rust.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        rust.add_model(rust_model)
        rust.record("VOUT")
        rust_result = rust.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
        )

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_requested"] == 1
        assert rust._perf_stats["rust_static_eval_available"] == 1
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 1
        assert rust._perf_stats["rust_static_eval_segments"] == 1
        assert rust._perf_stats["rust_static_eval_max_segment_models"] == 1
        assert rust._perf_stats["rust_static_eval_calls"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["rust_static_eval_node_voltage_syncs"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["rust_static_eval_deferred_output_syncs"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["rust_static_eval_lifecycle_model_skips"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["rust_static_eval_output_syncs"] == 1
        assert rust._perf_stats["model_post_update_skips"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["indexed_array_dirty_validation_enabled"] == 1
        assert rust._perf_stats["indexed_array_dirty_syncs"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["indexed_array_prev_snapshot_dirty_skips"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["indexed_array_dirty_nodes_checked"] == (
            rust._perf_stats["steps_total"] * 2
        )
        assert rust._perf_stats["rust_static_eval_errors"] == 0
        assert rust._perf_stats["indexed_post_model_sync_repairs"] == 0
        assert rust._indexed_array_stats["max_abs_diff"] == pytest.approx(0.0)
        assert rust_model.output_nodes["vout"] == pytest.approx(
            rust_result.signals["VOUT"][-1]
        )

    def test_rust_required_raises_when_backend_is_unavailable(self, monkeypatch):
        import evas.simulator.engine as engine

        monkeypatch.setattr(engine, "load_optional_rust_backend", lambda: None)
        sim = Simulator()
        sim.add_source("VIN", dc(1.0))
        sim.record("VIN")

        with pytest.raises(RuntimeError, match="Rust backend was required"):
            sim.run(
                tstop=1e-9,
                tstep=1e-9,
                rust_static_eval=True,
                rust_required=True,
            )

    def test_rust_transition_shadow_matches_ordered_state_target_segment(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module trans_target_shadow(inp, out);
    input voltage inp;
    output voltage out;
    integer q = 0;
    analog begin
        q = V(inp) > 0.45 ? 1 : 0;
        V(out) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        model.node_map = {"inp": "IN", "out": "OUT"}
        sim = Simulator()
        sim.add_source("IN", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.add_model(model)
        sim.record("OUT")

        sim.run(
            tstop=3e-9,
            tstep=1e-9,
            rust_transition_shadow=True,
        )

        assert sim._perf_stats["rust_transition_shadow_requested"] == 1
        assert sim._perf_stats["rust_transition_shadow_available"] == 1
        assert sim._perf_stats["rust_transition_shadow_candidate_models"] == 1
        assert sim._perf_stats["rust_transition_shadow_models"] == 1
        assert sim._perf_stats["rust_transition_shadow_static_ops"] == 1
        assert sim._perf_stats["rust_transition_shadow_target_ops"] == 1
        assert sim._perf_stats["rust_transition_shadow_segments"] == 1
        assert sim._perf_stats["rust_transition_shadow_calls"] == sim._perf_stats[
            "steps_total"
        ]
        assert sim._perf_stats["rust_transition_shadow_matches"] >= sim._perf_stats[
            "steps_total"
        ]
        assert sim._perf_stats["rust_transition_shadow_mismatches"] == 0
        assert sim._perf_stats["rust_transition_shadow_errors"] == 0
        assert sim._perf_stats["rust_transition_shadow_max_abs_diff"] == pytest.approx(
            0.0
        )
        assert sim._perf_stats["rust_transition_shadow_state_matches"] >= sim._perf_stats[
            "steps_total"
        ]
        assert sim._perf_stats["rust_transition_shadow_state_mismatches"] == 0
        assert sim._perf_stats[
            "rust_transition_shadow_state_max_abs_diff"
        ] == pytest.approx(0.0)
        assert model.transitions["trans_0"].rise_time == pytest.approx(1e-9)
        assert model.transitions["trans_0"].fall_time == pytest.approx(2e-9)

    def test_rust_transition_production_matches_default_waveform(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module trans_state_prod(inp, out);
    input voltage inp;
    output voltage out;
    integer q = 0;
    analog begin
        q = V(inp) > 0.45 ? 1 : 0;
        V(out) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"inp": "IN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("IN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("OUT")
            return sim, model

        ref, _ = build_sim()
        ref_result = ref.run(tstop=3e-9, tstep=500e-12, record_step=500e-12)

        rust, rust_model = build_sim()
        rust_result = rust.run(
            tstop=3e-9,
            tstep=500e-12,
            record_step=500e-12,
            rust_transition_production=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust._perf_stats["rust_transition_production_requested"] == 1
        assert rust._perf_stats["rust_transition_production_available"] == 1
        assert rust._perf_stats["rust_transition_production_enabled"] == 1
        assert rust._perf_stats["rust_transition_state_production_calls_total"] > 0
        assert rust._perf_stats["rust_transition_state_production_outputs_total"] == (
            rust._perf_stats["rust_transition_state_production_calls_total"]
        )
        assert rust._perf_stats["rust_transition_state_production_fallbacks_total"] == 0
        assert rust_model.transitions["trans_0"].rise_time == pytest.approx(1e-9)
        assert rust_model.transitions["trans_0"].fall_time == pytest.approx(2e-9)

    def test_rust_event_interpolation_matches_default_cross_body_read(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module cross_interp_sample(inp, out);
    input voltage inp;
    output voltage out;
    real sampled = 0.0;
    analog begin
        @(cross(V(inp) - 0.5, +1)) begin
            sampled = V(inp);
        end
        V(out) <+ sampled;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"inp": "IN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("IN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=2e-9, tstep=1e-9, record_step=500e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=1e-9,
            record_step=500e-12,
            rust_event_interpolation=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust._perf_stats["rust_event_interpolation_requested"] == 1
        assert rust._perf_stats["rust_event_interpolation_available"] == 1
        assert rust._perf_stats["rust_event_interpolation_enabled"] == 1
        assert rust._perf_stats["rust_event_interpolation_batches_total"] > 0
        assert rust._perf_stats["rust_event_interpolation_nodes_total"] > 0
        assert rust._perf_stats["rust_event_interpolation_cache_hits_total"] > 0
        assert rust._perf_stats["rust_event_interpolation_fallbacks_total"] == 0

    def test_rust_event_interpolation_matches_default_above_body_read(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module above_interp_sample(inp, out);
    input voltage inp;
    output voltage out;
    real sampled = 0.0;
    analog begin
        @(above(V(inp) - 0.5)) begin
            sampled = V(inp);
        end
        V(out) <+ sampled;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"inp": "IN", "out": "OUT"}
            sim = Simulator()
            sim.add_source("IN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=2e-9, tstep=1e-9, record_step=500e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=1e-9,
            record_step=500e-12,
            rust_event_interpolation=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust._perf_stats["rust_event_interpolation_batches_total"] > 0
        assert rust._perf_stats["rust_event_interpolation_cache_hits_total"] > 0
        assert rust._perf_stats["rust_event_interpolation_fallbacks_total"] == 0

    def test_rust_event_due_shadow_matches_cross_above_and_timer_checks(self):
        _build_rust_core_or_skip()

        class ShadowEventModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.state["count"] = 0.0

            def evaluate(self, node_voltages, time):
                val = float(node_voltages["VIN"]) - 0.5
                if self._check_cross("cross_vin", time, val, direction=1):
                    self.state["count"] += 1.0
                if self._check_above("above_vin", time, val, direction=1):
                    self.state["count"] += 1.0
                if self._check_timer("periodic", time, 1e-9, 0.0):
                    self.state["count"] += 1.0
                if self._check_timer_at("absolute", time, 2e-9):
                    self.state["count"] += 1.0
                self.output_nodes["OUT"] = float(self.state["count"])

        model = ShadowEventModel()
        sim = Simulator()
        sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 2e-9))
        sim.add_model(model)
        sim.record("VIN")

        sim.run(
            tstop=3e-9,
            tstep=1e-9,
            rust_event_due_shadow=True,
        )

        stats = sim._perf_stats
        assert stats["rust_event_due_shadow_requested"] == 1
        assert stats["rust_event_due_shadow_available"] == 1
        assert stats["rust_event_due_shadow_enabled"] == 1
        assert stats["indexed_state_storage_enabled"] == 0
        assert stats["rust_event_due_shadow_cross_checks_total"] > 0
        assert stats["rust_event_due_shadow_above_checks_total"] > 0
        assert stats["rust_event_due_shadow_timer_periodic_checks_total"] > 0
        assert stats["rust_event_due_shadow_timer_absolute_checks_total"] > 0
        total_checks = (
            stats["rust_event_due_shadow_cross_checks_total"]
            + stats["rust_event_due_shadow_above_checks_total"]
            + stats["rust_event_due_shadow_timer_periodic_checks_total"]
            + stats["rust_event_due_shadow_timer_absolute_checks_total"]
        )
        assert stats["rust_event_due_shadow_matches_total"] == total_checks
        assert stats["rust_event_due_shadow_mismatches_total"] == 0
        assert stats["rust_event_due_shadow_errors_total"] == 0
        assert stats["rust_event_due_shadow_max_time_diff_total"] == pytest.approx(0.0)

    def test_event_trace_audit_records_generated_event_body_writes(self):
        src = """\
`include "disciplines.vams"
module event_audit(inp, out);
    input voltage inp;
    output voltage out;
    real x = 0.0;
    real acc[0:1];
    analog begin
        @(initial_step) begin
            x = 1.0;
            acc[0] = x;
        end
        @(cross(V(inp) - 0.5, +1)) begin
            x = x + 1.0;
            acc[1] = x;
        end
        @(timer(2n)) begin
            x = x + 1.0;
        end
        @(final_step) begin
            x = x + 1.0;
        end
        V(out) <+ x;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        sim = Simulator()
        sim.add_source("inp", ramp(0.0, 1.0, 0.0, 2e-9))
        sim.add_model(model)
        sim.record("out")

        sim.run(tstop=3e-9, tstep=1e-9, event_trace_audit=True)

        stats = sim._perf_stats
        assert stats["event_trace_audit_requested"] == 1
        assert stats["event_trace_audit_enabled"] == 1
        assert stats["event_trace_audit_events_total"] > 0
        assert stats["event_trace_audit_body_entries_total"] > 0
        assert stats["event_trace_audit_cross_events_total"] > 0
        assert stats["event_trace_audit_timer_events_total"] > 0
        assert stats["event_trace_audit_initial_step_events_total"] == 1
        assert stats["event_trace_audit_final_step_events_total"] == 1
        assert stats["event_trace_audit_state_writes_total"] > 0
        assert stats["event_trace_audit_array_writes_total"] > 0
        assert stats["event_trace_audit_output_writes_total"] > 0
        assert stats["event_trace_audit_in_event_writes_total"] > 0
        assert stats["event_trace_audit_records_dropped_total"] == 0
        record_types = {record["type"] for record in model._event_trace_audit_records}
        assert {"event", "body_enter", "body_exit", "write"} <= record_types

    def test_rust_event_write_shadow_and_production_match_lfsr_body(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_lfsr(clk, rstb, vdd, vss, out);
    input voltage clk, rstb, vdd, vss;
    output voltage out;
    integer i;
    real lfsr_r[0:3];
    real tmp_lfsr_r[0:4];
    real dpn_level;
    analog begin
        @(initial_step) begin
            lfsr_r[0] = 1;
            lfsr_r[1] = 0;
            lfsr_r[2] = 1;
            lfsr_r[3] = 0;
            dpn_level = V(vss);
        end
        @(cross(V(clk) - 0.5, +1)) begin
            if (V(rstb) > 0.5) begin
                for (i = 0; i < 4; i = i + 1) begin
                    tmp_lfsr_r[i + 1] = lfsr_r[i];
                end
                tmp_lfsr_r[0] = lfsr_r[3] ^ lfsr_r[1] ^ lfsr_r[0];
                for (i = 0; i < 4; i = i + 1) begin
                    lfsr_r[i] = tmp_lfsr_r[i];
                end
                dpn_level = lfsr_r[3] > 0 ? V(vdd) : V(vss);
            end
        end
        V(out) <+ dpn_level;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_lfsr(**kwargs):
            model = ModelCls()
            sim = Simulator()
            sim.add_source("clk", pulse(0.0, 1.0, 1e-9, width=0.4e-9))
            sim.add_source("rstb", dc(1.0))
            sim.add_source("vdd", dc(1.2))
            sim.add_source("vss", dc(0.0))
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=3e-9,
                tstep=0.25e-9,
                max_step=0.25e-9,
                rust_required=True,
                **kwargs,
            )
            return result, sim, model

        default_result, _, default_model = run_lfsr()
        shadow_result, shadow_sim, _ = run_lfsr(rust_event_write_shadow=True)
        production_result, production_sim, production_model = run_lfsr(
            rust_event_write_production=True
        )

        assert shadow_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert shadow_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        shadow_stats = shadow_sim._perf_stats
        assert shadow_stats["rust_event_write_shadow_requested"] == 1
        assert shadow_stats["rust_event_write_shadow_enabled"] == 1
        assert shadow_stats["rust_event_write_batches"] == 1
        assert shadow_stats["rust_event_write_shadow_checks_total"] > 0
        assert shadow_stats["rust_event_write_shadow_matches_total"] == shadow_stats[
            "rust_event_write_shadow_checks_total"
        ]
        assert shadow_stats["rust_event_write_shadow_mismatches_total"] == 0
        assert shadow_stats["rust_event_write_shadow_errors_total"] == 0

        assert production_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert production_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert production_model.arrays["lfsr_r"] == pytest.approx(
            default_model.arrays["lfsr_r"]
        )
        production_stats = production_sim._perf_stats
        assert production_stats["rust_event_write_production_requested"] == 1
        assert production_stats["rust_event_write_production_enabled"] == 1
        assert production_stats["rust_event_write_batches"] == 1
        assert production_stats["rust_event_write_production_calls_total"] > 0
        assert production_stats["rust_event_write_production_executed_total"] > 0
        assert production_stats["rust_event_write_production_fallbacks_total"] == 0
        assert production_stats["indexed_array_syncs"] == 0

    def test_rust_event_linear_write_batches_cross_state_body(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_linear_cross(clk, out);
    input voltage clk;
    output voltage out;
    integer count;
    real toggle;
    analog begin
        @(initial_step) begin
            count = 0;
            toggle = 0.0;
        end
        @(cross(V(clk) - 0.5, +1)) begin
            count = count + 1;
            toggle = 1.0 - toggle;
        end
        V(out) <+ count + 0.01 * toggle;
    end
endmodule
"""
        ModelCls = compile_module(parse(src), indexed_state_fastpath_codegen=True)

        def run_counter(**kwargs):
            model = ModelCls()
            sim = Simulator()
            sim.add_source("clk", pulse(0.0, 1.0, 1e-9, width=0.4e-9))
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=3e-9,
                tstep=0.25e-9,
                max_step=0.25e-9,
                rust_required=True,
                **kwargs,
            )
            return result, sim, model

        default_result, _, default_model = run_counter()
        shadow_result, shadow_sim, _ = run_counter(rust_event_write_shadow=True)
        production_result, production_sim, production_model = run_counter(
            rust_event_write_production=True
        )

        assert shadow_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert shadow_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        shadow_stats = shadow_sim._perf_stats
        assert shadow_stats["rust_event_linear_write_batches_total"] == 1
        assert shadow_stats["rust_event_linear_write_shadow_checks_total"] > 0
        assert shadow_stats["rust_event_linear_write_shadow_matches_total"] == shadow_stats[
            "rust_event_linear_write_shadow_checks_total"
        ]
        assert shadow_stats["rust_event_linear_write_shadow_mismatches_total"] == 0

        assert production_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert production_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert production_model.state["count"] == default_model.state["count"]
        assert production_model.state["toggle"] == pytest.approx(
            default_model.state["toggle"]
        )
        production_stats = production_sim._perf_stats
        assert production_stats["rust_event_write_batches"] == 1
        assert production_stats["rust_event_linear_write_batches_total"] == 1
        assert production_stats["rust_event_linear_write_production_calls_total"] > 0
        assert production_stats["rust_event_linear_write_production_executed_total"] > 0
        assert production_stats["rust_event_linear_write_production_fallbacks_total"] == 0

    def test_rust_event_linear_write_batches_above_node_body(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_linear_above(inp, out);
    input voltage inp;
    output voltage out;
    real sampled;
    analog begin
        @(initial_step) sampled = 0.0;
        @(above(V(inp) - 0.5)) begin
            sampled = V(inp) > 0.7 ? 2.0 : 1.0;
        end
        V(out) <+ sampled;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_above(**kwargs):
            model = ModelCls()
            sim = Simulator()
            sim.add_source("inp", ramp(0.0, 1.0, 0.0, 2e-9))
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=3e-9,
                tstep=0.25e-9,
                max_step=0.25e-9,
                rust_required=True,
                **kwargs,
            )
            return result, sim, model

        default_result, _, default_model = run_above()
        production_result, production_sim, production_model = run_above(
            rust_event_write_production=True
        )

        assert production_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert production_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert production_model.state["sampled"] == pytest.approx(
            default_model.state["sampled"]
        )
        stats = production_sim._perf_stats
        assert stats["rust_event_linear_write_batches_total"] == 1
        assert stats["rust_event_linear_write_production_calls_total"] > 0
        assert stats["rust_event_linear_write_production_fallbacks_total"] == 0

    def test_rust_event_linear_write_cross_node_body_falls_back_for_interpolation(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module event_linear_cross_sample(clk, inp, out);
    input voltage clk, inp;
    output voltage out;
    real sampled;
    analog begin
        @(initial_step) sampled = 0.0;
        @(cross(V(clk) - 0.5, +1)) begin
            sampled = V(inp);
        end
        V(out) <+ sampled;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_sample(**kwargs):
            model = ModelCls()
            sim = Simulator()
            sim.add_source("clk", ramp(0.0, 1.0, 0.0, 2e-9))
            sim.add_source("inp", ramp(0.0, 2.0, 0.0, 2e-9))
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=3e-9,
                tstep=0.5e-9,
                max_step=0.5e-9,
                rust_required=True,
                **kwargs,
            )
            return result, sim, model

        default_result, _, default_model = run_sample()
        production_result, production_sim, production_model = run_sample(
            rust_event_write_production=True
        )

        assert production_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert production_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert production_model.state["sampled"] == pytest.approx(
            default_model.state["sampled"]
        )
        stats = production_sim._perf_stats
        assert stats["rust_event_linear_write_batches_total"] == 1
        assert stats["rust_event_linear_write_production_calls_total"] == 0

    def test_rust_fuses_timer_lfsr_output_and_record_path(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_lfsr(rstb, vdd, vss, out);
    input voltage rstb, vdd, vss;
    output voltage out;
    integer i;
    real lfsr_r[0:3];
    real tmp_lfsr_r[0:4];
    real dpn_level;
    analog begin
        @(initial_step) begin
            lfsr_r[0] = 1;
            lfsr_r[1] = 0;
            lfsr_r[2] = 1;
            lfsr_r[3] = 0;
            dpn_level = V(vss);
        end
        @(timer(1n, 1n)) begin
            if (V(rstb) > 0.5) begin
                for (i = 0; i < 4; i = i + 1) begin
                    tmp_lfsr_r[i + 1] = lfsr_r[i];
                end
                tmp_lfsr_r[0] = lfsr_r[3] ^ lfsr_r[1] ^ lfsr_r[0];
                for (i = 0; i < 4; i = i + 1) begin
                    lfsr_r[i] = tmp_lfsr_r[i];
                end
                dpn_level = lfsr_r[3] > 0 ? V(vdd) : V(vss);
            end
        end
        V(out) <+ dpn_level;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def run_timer_lfsr(**kwargs):
            model = ModelCls()
            sim = Simulator()
            sim.add_source("rstb", dc(1.0))
            sim.add_source("vdd", dc(1.2))
            sim.add_source("vss", dc(0.0))
            sim.add_model(model)
            sim.record("out")
            result = sim.run(
                tstop=4e-9,
                tstep=0.25e-9,
                max_step=0.25e-9,
                rust_required=True,
                **kwargs,
            )
            return result, sim, model

        default_result, _, default_model = run_timer_lfsr()
        fused_result, fused_sim, fused_model = run_timer_lfsr(
            rust_timer_event=True,
            rust_event_write_production=True,
        )

        assert fused_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert fused_result.signals["out"].tolist() == pytest.approx(
            default_result.signals["out"].tolist()
        )
        assert fused_model.arrays["lfsr_r"] == pytest.approx(
            default_model.arrays["lfsr_r"]
        )
        stats = fused_sim._perf_stats
        assert stats["rust_timer_event_enabled"] == 1
        assert stats["rust_event_write_production_enabled"] == 1
        assert stats["rust_timer_lfsr_output_batches_total"] == 1
        assert stats["rust_timer_lfsr_output_calls_total"] > 0
        assert stats["rust_timer_lfsr_output_due_total"] > 0
        assert stats["rust_timer_lfsr_output_executed_total"] > 0
        assert stats["rust_timer_lfsr_output_writes_total"] > 0
        assert stats["rust_timer_lfsr_output_fallbacks_total"] == 0
        assert stats["indexed_array_record_reads"] > 0

    def test_rust_static_eval_applies_runtime_parameter_overrides(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain_param(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    parameter real offset = 0.125;
    analog begin
        V(vout) <+ gain * V(vin) + offset;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            model.params["gain"] = 3.0
            model.params["offset"] = -0.25
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust_result.signals["VOUT"][-1] == pytest.approx(2.75)
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_runtime_param_ops"] == 1
        assert rust._perf_stats["rust_static_eval_coeff_eval_fallbacks"] == 0
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_handles_simple_state_linear_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module state_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    real sample = 0.0;
    analog begin
        sample = 2.0 * V(vin) + 0.1;
        V(vout) <+ sample + 0.2;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim, model

        default, _ = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust, rust_model = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 2
        assert rust._perf_stats["rust_static_eval_state_read_terms"] == 1
        assert rust._perf_stats["rust_static_eval_state_write_ops"] == 1
        assert rust._perf_stats["indexed_state_storage_enabled"] == 1
        assert rust._perf_stats["indexed_state_scalar_writes_total"] >= (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["indexed_state_scalar_reads_total"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust_model.state["sample"] == pytest.approx(2.1)
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_handles_fixed_index_state_array_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module state_array_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter integer N = 2;
    real tap[0:1];
    analog begin
        tap[0] = 2.0 * V(vin) + 0.1;
        tap[N-1] = tap[0] + 0.25;
        V(vout) <+ tap[1];
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._evaluate_ir_static_linear_rejections == ()
        assert len(ModelCls._evaluate_ir_static_linear_ops) == 3

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim, model

        default, default_model = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust, rust_model = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 3
        assert rust._perf_stats["rust_static_eval_state_write_ops"] == 2
        assert rust._perf_stats["rust_static_eval_state_read_terms"] == 2
        assert rust._perf_stats["indexed_state_storage_array_slots"] == 2
        assert rust_model.arrays["tap"][0] == pytest.approx(
            default_model.arrays["tap"][0]
        )
        assert rust_model.arrays["tap"][1] == pytest.approx(
            default_model.arrays["tap"][1]
        )
        assert rust_model.arrays["tap"][1] == pytest.approx(2.35)
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_unrolls_static_for_state_array_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module state_array_loop_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    integer i;
    real tap[0:2];
    analog begin
        tap[0] = V(vin);
        for (i = 0; i < 2; i = i + 1) begin
            tap[i + 1] = tap[i] + 0.25;
        end
        V(vout) <+ tap[2] + 0.1 * i;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._evaluate_ir_static_linear_rejections == ()
        assert len(ModelCls._evaluate_ir_static_linear_ops) == 5

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim, model

        default, default_model = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust, rust_model = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 5
        assert rust._perf_stats["rust_static_eval_state_write_ops"] == 4
        assert rust._perf_stats["rust_static_eval_state_read_terms"] == 4
        assert rust._perf_stats["indexed_state_storage_array_slots"] == 3
        assert rust_model.state["i"] == default_model.state["i"] == 2
        assert rust_model.arrays["tap"][0] == pytest.approx(
            default_model.arrays["tap"][0]
        )
        assert rust_model.arrays["tap"][1] == pytest.approx(
            default_model.arrays["tap"][1]
        )
        assert rust_model.arrays["tap"][2] == pytest.approx(
            default_model.arrays["tap"][2]
        )
        assert rust_model.arrays["tap"][2] == pytest.approx(1.5)
        assert rust_result.signals["VOUT"][-1] == pytest.approx(1.7)
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_rejects_dynamic_state_array_index(self):
        src = """\
`include "disciplines.vams"
module dynamic_state_array(vin, vout);
    input voltage vin;
    output voltage vout;
    integer i = 0;
    real tap[0:1];
    analog begin
        i = V(vin) > 0.5 ? 1 : 0;
        tap[i] = V(vin);
        V(vout) <+ tap[i];
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        reasons = dict(ModelCls._evaluate_ir_static_linear_rejections)

        assert ModelCls._evaluate_ir_static_linear_ops == ()
        assert reasons["assignment_array_target"] == 1
        assert reasons["assignment_dynamic_array_index"] == 1
        assert reasons["expr_array_access"] >= 1
        assert reasons["expr_dynamic_array_index"] >= 1

    def test_rust_static_eval_rejects_dynamic_for_state_array_bounds(self):
        src = """\
`include "disciplines.vams"
module dynamic_for_state_array(vin, vout);
    input voltage vin;
    output voltage vout;
    integer i;
    integer limit = 0;
    real tap[0:2];
    analog begin
        limit = V(vin) > 0.5 ? 2 : 1;
        for (i = 0; i < limit; i = i + 1) begin
            tap[i] = V(vin);
        end
        V(vout) <+ tap[0];
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        reasons = dict(ModelCls._evaluate_ir_static_linear_rejections)

        assert ModelCls._evaluate_ir_static_linear_ops == ()
        assert reasons["for_statement"] == 1
        assert reasons["for_dynamic_bounds"] == 1
        assert reasons["assignment_dynamic_array_index"] == 1

    def test_rust_static_eval_handles_integer_state_linear_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module integer_state_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    integer code = 0;
    analog begin
        code = 1.6 + V(vin);
        V(vout) <+ code;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim, model

        default, default_model = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust, rust_model = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 2
        assert rust._perf_stats["rust_static_eval_integer_state_write_ops"] == 1
        assert rust._perf_stats["indexed_state_storage_integer_slots"] == 1
        assert rust_model.state["code"] == default_model.state["code"]
        assert rust_model.state["code"] == 3
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_handles_initial_step_and_conditional_state_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module dither_like(vres_p, vres_n, dpn, vout_p, vout_n);
    input voltage vres_p;
    input voltage vres_n;
    input voltage dpn;
    output voltage vout_p;
    output voltage vout_n;
    parameter real vth = 0.45;
    parameter real amp = 0.014;
    real dither_diff;
    analog begin
        @(initial_step)
            $strobe("init only");
        dither_diff = (V(dpn) > vth) ? amp : -amp;
        V(vout_p) <+ V(vres_p) + dither_diff * 0.5;
        V(vout_n) <+ V(vres_n) - dither_diff * 0.5;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.node_map = {
                "vres_p": "VRESP",
                "vres_n": "VRESN",
                "dpn": "DPN",
                "vout_p": "VOUTP",
                "vout_n": "VOUTN",
            }
            sim = Simulator()
            sim.add_source("VRESP", dc(0.55))
            sim.add_source("VRESN", dc(0.45))
            sim.add_source("DPN", ramp(0.2, 0.8, 0.0, 2e-9))
            sim.add_model(model)
            sim.record("VOUTP")
            sim.record("VOUTN")
            return sim, model

        default, _ = build_sim()
        default_result = default.run(tstop=3e-9, tstep=1e-9)

        rust, rust_model = build_sim()
        rust_result = rust.run(tstop=3e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUTP"].tolist() == pytest.approx(
            default_result.signals["VOUTP"].tolist()
        )
        assert rust_result.signals["VOUTN"].tolist() == pytest.approx(
            default_result.signals["VOUTN"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 3
        assert rust._perf_stats["rust_static_eval_state_write_ops"] == 1
        assert rust._perf_stats["rust_static_eval_state_read_terms"] >= 2
        assert rust._perf_stats["rust_static_eval_errors"] == 0
        assert rust_model.state["dither_diff"] == pytest.approx(0.014)

    def test_rust_static_eval_batches_consecutive_affine_models_in_order(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 2.0 * V(vin) + 0.125;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            for vin, vout in (("VIN", "N0"), ("N0", "N1"), ("N1", "VOUT")):
                model = ModelCls()
                model.node_map = {"vin": vin, "vout": vout}
                sim.add_model(model)
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_models"] == 3
        assert rust._perf_stats["rust_static_eval_ops"] == 3
        assert rust._perf_stats["rust_static_eval_segments"] == 1
        assert rust._perf_stats["rust_static_eval_max_segment_models"] == 3
        assert rust._perf_stats["rust_static_eval_calls"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["rust_static_eval_node_voltage_syncs"] == (
            rust._perf_stats["steps_total"] * 3
        )
        assert rust._perf_stats["rust_static_eval_deferred_output_syncs"] == (
            rust._perf_stats["steps_total"] * 3
        )
        assert rust._perf_stats["rust_static_eval_lifecycle_model_skips"] == (
            rust._perf_stats["steps_total"] * 3
        )
        assert rust._perf_stats["rust_static_eval_output_syncs"] == 3
        assert rust._perf_stats["model_post_update_skips"] == (
            rust._perf_stats["steps_total"] * 3
        )
        assert rust._perf_stats["indexed_array_dirty_validation_enabled"] == 1
        assert rust._perf_stats["indexed_array_dirty_syncs"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["indexed_array_prev_snapshot_dirty_skips"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["indexed_array_dirty_nodes_checked"] > 0
        assert rust._perf_stats["rust_static_eval_errors"] == 0
        assert rust._perf_stats["indexed_post_model_sync_repairs"] == 0
        assert rust.models[-1].output_nodes["vout"] == pytest.approx(
            rust_result.signals["VOUT"][-1]
        )

    def test_rust_static_fast_sync_skips_per_step_dict_validation(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 2.0 * V(vin) + 0.125;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            for vin, vout in (("VIN", "N0"), ("N0", "N1"), ("N1", "VOUT")):
                model = ModelCls()
                model.node_map = {"vin": vin, "vout": vout}
                sim.add_model(model)
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        fast = build_sim()
        fast_result = fast.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
            rust_static_fast_sync=True,
        )

        assert fast_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert fast_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert fast_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert fast._perf_stats["rust_static_fast_sync_requested"] == 1
        assert fast._perf_stats["rust_static_fast_sync_enabled"] == 1
        assert fast._perf_stats["rust_static_eval_models"] == 3
        assert fast._perf_stats["rust_static_eval_calls"] == fast._perf_stats["steps_total"]
        assert fast._perf_stats["indexed_array_dirty_validation_enabled"] == 0
        assert fast._perf_stats["indexed_array_dirty_syncs"] == 0
        assert fast._perf_stats["indexed_array_syncs"] == 0
        assert fast._perf_stats["rust_static_fast_sync_validation_skips"] == (
            fast._perf_stats["steps_total"]
        )
        assert fast._perf_stats["rust_static_fast_sync_node_voltage_sync_skips"] == (
            fast._perf_stats["steps_total"] * 3
        )
        assert fast._perf_stats["rust_static_eval_deferred_output_syncs"] == (
            fast._perf_stats["steps_total"] * 3
        )
        assert fast._perf_stats["rust_static_eval_node_voltage_syncs"] == 3
        assert fast._perf_stats["rust_static_eval_output_syncs"] == 3
        assert fast._perf_stats["indexed_post_model_sync_repairs"] == 0
        assert fast.models[-1].output_nodes["vout"] == pytest.approx(
            fast_result.signals["VOUT"][-1]
        )

    def test_rust_static_fast_sync_no_segment_falls_back_to_python_runtime(self):
        _build_rust_core_or_skip()

        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "VIN", "out": "VOUT"}

            def evaluate(self, node_voltages, time):
                value = self._get_voltage("inp", node_voltages)
                self._set_output("out", value, node_voltages)

        def build_sim():
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(MirrorModel())
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        requested = build_sim()
        requested_result = requested.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
            rust_static_fast_sync=True,
        )

        assert requested_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert requested._perf_stats["rust_static_eval_requested"] == 1
        assert requested._perf_stats["rust_static_fast_sync_requested"] == 1
        assert requested._perf_stats["rust_static_eval_candidate_models"] == 0
        assert requested._perf_stats["rust_static_eval_models"] == 0
        assert requested._perf_stats["rust_static_eval_no_segment_fallbacks"] == 1
        assert requested._perf_stats["rust_static_fast_sync_enabled"] == 0
        assert requested._perf_stats["indexed_array_syncs"] == 0
        assert requested._perf_stats["indexed_array_snapshots"] == 0
        assert requested._perf_stats["indexed_array_record_reads"] == 0
        assert requested._perf_stats["indexed_state_storage_enabled"] == 0

    def test_rust_static_eval_reports_transition_rejection_reason(self):
        src = """\
`include "disciplines.vams"
module transition_driver(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ transition(V(vin) > 0.5 ? 1.0 : 0.0, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        reasons = dict(ModelCls._evaluate_ir_static_linear_rejections)
        assert reasons["contribution_non_static_linear_expr"] == 1
        assert reasons["expr_function_transition"] == 1

        model = ModelCls()
        model.node_map = {"vin": "VIN", "vout": "VOUT"}
        sim = Simulator()
        sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.add_model(model)
        sim.record("VOUT")

        result = sim.run(tstop=2e-9, tstep=1e-9, rust_static_eval=True)

        assert result.signals["VOUT"][-1] == pytest.approx(1.0)
        assert sim._perf_stats["rust_static_eval_requested"] == 1
        assert sim._perf_stats["rust_static_eval_candidate_models"] == 0
        assert sim._perf_stats["rust_static_eval_no_candidate_models"] == 1
        assert (
            sim._perf_stats[
                "rust_static_eval_no_candidate_contribution_non_static_linear_expr"
            ]
            == 1
        )
        assert sim._perf_stats["rust_static_eval_no_candidate_expr_function_transition"] == 1

    def test_rust_static_eval_reports_event_rejection_reason(self):
        src = """\
`include "disciplines.vams"
module event_driver(vin, vout);
    input voltage vin;
    output voltage vout;
    real q = 0.0;
    analog begin
        @(cross(V(vin) - 0.5, +1)) begin
            q = 1.0;
        end
        V(vout) <+ q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        reasons = dict(ModelCls._evaluate_ir_static_linear_rejections)
        assert reasons["event_statement"] == 1
        assert reasons["event_cross"] == 1

        model = ModelCls()
        model.node_map = {"vin": "VIN", "vout": "VOUT"}
        sim = Simulator()
        sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
        sim.add_model(model)
        sim.record("VOUT")

        sim.run(tstop=2e-9, tstep=0.5e-9, rust_static_eval=True)

        assert sim._perf_stats["rust_static_eval_candidate_models"] == 0
        assert sim._perf_stats["rust_static_eval_no_candidate_models"] == 1
        assert sim._perf_stats["rust_static_eval_no_candidate_event_statement"] == 1
        assert sim._perf_stats["rust_static_eval_no_candidate_event_cross"] == 1

    def test_rust_static_eval_lowers_simple_if_else_contribution(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module if_driver(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        if (V(vin) > 0.5)
            V(vout) <+ 1.0;
        else
            V(vout) <+ 0.0;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._evaluate_ir_static_linear_rejections == ()
        assert len(ModelCls._evaluate_ir_static_linear_ops) == 1

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=0.5e-9)

        rust = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=0.5e-9, rust_static_eval=True)

        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 1
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_lowers_if_else_state_assignment(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module if_state_driver(vin, vout);
    input voltage vin;
    output voltage vout;
    real q = 0.0;
    analog begin
        if (V(vin) > 0.5)
            q = 1.0;
        else
            q = 0.0;
        V(vout) <+ q + 0.25;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._evaluate_ir_static_linear_rejections == ()
        assert len(ModelCls._evaluate_ir_static_linear_ops) == 2

        def build_sim():
            model = ModelCls()
            model.node_map = {"vin": "VIN", "vout": "VOUT"}
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=0.5e-9)

        rust = build_sim()
        rust_result = rust.run(tstop=2e-9, tstep=0.5e-9, rust_static_eval=True)

        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_candidate_models"] == 1
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_ops"] == 2
        assert rust._perf_stats["rust_static_eval_state_write_ops"] == 1
        assert rust._perf_stats["rust_static_eval_state_read_terms"] == 1
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_rust_static_eval_deferred_output_sync_preserves_unmapped_model(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 0.5 * V(vin) + 0.25;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            sim = Simulator()
            sim.add_source("vin", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(ModelCls())
            sim.record("vout")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
        )

        assert rust_result.time.tolist() == pytest.approx(default_result.time.tolist())
        assert rust_result.step_sizes.tolist() == pytest.approx(
            default_result.step_sizes.tolist()
        )
        assert rust_result.signals["vout"].tolist() == pytest.approx(
            default_result.signals["vout"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_output_syncs"] == 1
        assert rust._perf_stats["rust_static_eval_node_voltage_syncs"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["rust_static_eval_lifecycle_model_skips"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["model_post_update_skips"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["indexed_array_dirty_validation_enabled"] == 1
        assert rust.models[0].output_nodes["vout"] == pytest.approx(
            rust_result.signals["vout"][-1]
        )

    def test_rust_static_eval_keeps_full_indexed_validation_for_mixed_models(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 0.5 * V(vin) + 0.25;
    end
endmodule
"""
        RustModelCls = compile_module(parse(src))

        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "MID", "out": "VOUT"}

            def evaluate(self, node_voltages, time):
                value = self._get_voltage("inp", node_voltages)
                self._set_output("out", value, node_voltages)

        def build_sim():
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            rust_model = RustModelCls()
            rust_model.node_map = {"vin": "VIN", "vout": "MID"}
            sim.add_model(rust_model)
            sim.add_model(MirrorModel())
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        rust = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
        )

        assert rust_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_fast_sync_requested"] == 0
        assert rust._perf_stats["rust_static_fast_sync_enabled"] == 0
        assert rust._perf_stats["rust_static_eval_lifecycle_model_skips"] == (
            rust._perf_stats["steps_total"]
        )
        assert rust._perf_stats["model_post_update_skips"] == (
            rust._perf_stats["rust_static_eval_lifecycle_model_skips"]
        )
        assert rust._perf_stats["model_post_update_calls"] == rust._perf_stats["steps_total"]
        assert rust._perf_stats["indexed_array_dirty_validation_enabled"] == 0
        assert rust._perf_stats["indexed_array_dirty_syncs"] == 0
        assert rust._perf_stats["indexed_post_model_sync_repairs"] == 0

    def test_rust_static_fast_sync_gates_small_mixed_segments_to_python(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 0.5 * V(vin) + 0.25;
    end
endmodule
"""
        RustModelCls = compile_module(parse(src))

        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "MID", "out": "VOUT"}

            def evaluate(self, node_voltages, time):
                value = self._get_voltage("inp", node_voltages)
                self._set_output("out", value, node_voltages)

        def build_sim():
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            rust_model = RustModelCls()
            rust_model.node_map = {"vin": "VIN", "vout": "MID"}
            sim.add_model(rust_model)
            sim.add_model(MirrorModel())
            sim.record("VOUT")
            return sim

        default = build_sim()
        default_result = default.run(tstop=2e-9, tstep=1e-9)

        requested = build_sim()
        requested_result = requested.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
            rust_static_fast_sync=True,
        )

        assert requested_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert requested._perf_stats["rust_static_eval_candidate_models"] == 1
        assert requested._perf_stats["rust_static_eval_models"] == 0
        assert requested._perf_stats["rust_static_eval_ops"] == 0
        assert requested._perf_stats["rust_static_eval_segments"] == 0
        assert requested._perf_stats["rust_static_eval_gated_models"] == 1
        assert requested._perf_stats["rust_static_eval_gated_ops"] == 1
        assert requested._perf_stats["rust_static_eval_mixed_small_fallbacks"] == 1
        assert requested._perf_stats["rust_static_eval_no_segment_fallbacks"] == 1
        assert requested._perf_stats["rust_static_fast_sync_requested"] == 1
        assert requested._perf_stats["rust_static_fast_sync_enabled"] == 0
        assert requested._perf_stats["rust_static_eval_calls"] == 0

        required = build_sim()
        required_result = required.run(
            tstop=2e-9,
            tstep=1e-9,
            rust_static_eval=True,
            rust_static_fast_sync=True,
            rust_required=True,
        )

        assert required_result.signals["VOUT"].tolist() == pytest.approx(
            default_result.signals["VOUT"].tolist()
        )
        assert required._perf_stats["rust_static_eval_candidate_models"] == 1
        assert required._perf_stats["rust_static_eval_models"] == 1
        assert required._perf_stats["rust_static_eval_ops"] == 1
        assert required._perf_stats["rust_static_eval_segments"] == 1
        assert required._perf_stats["rust_static_eval_calls"] > 0
        assert required._perf_stats["rust_static_eval_gated_models"] == 0
        assert required._perf_stats["rust_static_eval_mixed_small_fallbacks"] == 0
        assert required._perf_stats["rust_static_fast_sync_enabled"] == 0
        assert requested._perf_stats["indexed_array_syncs"] == 0
        assert requested._perf_stats["indexed_array_snapshots"] == 0
        assert requested._perf_stats["indexed_state_storage_enabled"] == 0


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

    def test_get_voltage_non_event_prefers_indexed_reader(self):
        self.model._set_indexed_voltage_reader(
            lambda local, ext: 1.2 if ext == "vin" else None
        )

        assert self.model._get_voltage("vin", {"vin": 0.1}) == pytest.approx(1.2)

    def test_get_voltage_mapped_non_event_prefers_indexed_reader(self):
        self.model.node_map["inp"] = "VIN"
        self.model._set_indexed_voltage_reader(
            lambda local, ext: 1.3 if ext == "VIN" else None
        )

        assert self.model._get_voltage("inp", {"VIN": 0.7}) == pytest.approx(1.3)

    def test_node_resolution_cache_resolves_mapped_reads_and_writes(self):
        self.model.node_map["inp"] = "VIN"
        self.model.node_map["out"] = "VOUT"
        self.model._set_node_resolution_cache_enabled(True)
        nv = {"VIN": 0.7}

        assert self.model._get_voltage("inp", nv) == pytest.approx(0.7)
        self.model._set_output("out", 1.1, nv)

        assert nv["VOUT"] == pytest.approx(1.1)
        assert self.model._node_resolution_cache == {"inp": "VIN", "out": "VOUT"}

    def test_node_resolution_cache_resolves_parent_mapped_nodes(self):
        parent = CompiledModel()
        parent.node_map["out"] = "OUT"
        child = CompiledModel()
        child.node_map["z"] = "@parent:out"
        child._parent_model = parent
        parent._child_models = [child]
        parent._set_node_resolution_cache_enabled(True)
        nv = {}

        child._set_output("z", 1.1, nv)

        assert nv["OUT"] == pytest.approx(1.1)
        assert child._node_resolution_cache == {"z": "OUT"}

    def test_node_resolution_cache_disable_clears_stale_mapping(self):
        self.model.node_map["inp"] = "VIN"
        self.model._set_node_resolution_cache_enabled(True)
        assert self.model._get_voltage("inp", {"VIN": 0.7}) == pytest.approx(0.7)

        self.model.node_map["inp"] = "VIN2"
        assert self.model._get_voltage("inp", {"VIN": 0.7, "VIN2": 0.9}) == pytest.approx(0.7)
        self.model._set_node_resolution_cache_enabled(False)

        assert self.model._node_resolution_cache == {}
        assert self.model._get_voltage("inp", {"VIN": 0.7, "VIN2": 0.9}) == pytest.approx(0.9)

    def test_get_voltage_interpolates_inside_event_context(self):
        self.model._prepare_step({"vin": 0.0, "rst": 0.0}, {"vin": 1.0, "rst": 1.0}, 0.0, 10e-9)
        self.model._event_time = 4e-9
        self.model._event_context_active = True
        self.model._event_interpolated_nodes = {"vin"}
        assert self.model._get_voltage("vin", {"vin": 1.0}) == pytest.approx(0.4)
        assert self.model._get_voltage("rst", {"rst": 1.0}) == pytest.approx(0.4)
        self.model._event_context_active = False
        assert self.model._get_voltage("vin", {"vin": 1.0}) == pytest.approx(1.0)

    def test_get_voltage_event_context_ignores_indexed_reader(self):
        calls = []
        self.model._set_indexed_voltage_reader(
            lambda local, ext: calls.append((local, ext)) or 1.2
        )
        self.model._prepare_step({"vin": 0.0}, {"vin": 1.0}, 0.0, 10e-9)
        self.model._event_time = 4e-9
        self.model._event_context_active = True

        assert self.model._get_voltage("vin", {"vin": 1.0}) == pytest.approx(0.4)
        assert calls == []

    def test_static_branch_voltage_falls_back_to_event_interpolation(self):
        calls = []
        self.model._set_static_branch_fastpath_enabled(True)
        self.model._set_indexed_voltage_reader(
            lambda local, ext: calls.append((local, ext)) or 1.2
        )
        self.model._prepare_step({"vin": 0.0}, {"vin": 1.0}, 0.0, 10e-9)
        self.model._event_time = 4e-9
        self.model._event_context_active = True

        assert self.model._get_static_branch_voltage("vin", {"vin": 1.0}) == pytest.approx(0.4)
        assert calls == []
        assert self.model._perf_stats["static_branch_fastpath_fallbacks"] == 1

    def test_static_branch_slot_voltage_falls_back_to_event_interpolation(self):
        class SlotModel(CompiledModel):
            _static_voltage_read_nodes = ("vin",)

        model = SlotModel()
        model._set_static_branch_indexed_io((0,), (), (), [1.2])
        model._prepare_step({"vin": 0.0}, {"vin": 1.0}, 0.0, 10e-9)
        model._event_time = 4e-9
        model._event_context_active = True

        assert model._get_static_branch_voltage_by_slot(0, {"vin": 1.0}) == pytest.approx(0.4)
        assert model._perf_stats["static_branch_fastpath_fallbacks"] == 1

    def test_prepare_step_skips_future_snapshot_when_unneeded(self):
        self.model._prepare_step(
            {"vin": 0.0},
            {"vin": 1.0},
            0.0,
            10e-9,
            {"vin": 2.0},
        )

        assert self.model._step_future_node_voltages is self.model._step_curr_node_voltages
        assert self.model._step_future_node_voltages["vin"] == pytest.approx(1.0)

    def test_prepare_step_copies_future_snapshot_when_needed(self):
        self.model._needs_future_node_voltages = True
        future = {"vin": 2.0}
        self.model._prepare_step({"vin": 0.0}, {"vin": 1.0}, 0.0, 10e-9, future)
        future["vin"] = 3.0

        assert self.model._step_future_node_voltages is not self.model._step_curr_node_voltages
        assert self.model._step_future_node_voltages["vin"] == pytest.approx(2.0)

    def test_prepare_step_accepts_lazy_future_snapshot(self):
        class LazyFuture:
            def _with_fallback(self, fallback):
                self.fallback = fallback
                return self

            def get(self, node, default=None):
                if node == "src":
                    return 2.0
                return self.fallback.get(node, default)

        self.model._needs_future_node_voltages = True
        lazy = LazyFuture()
        self.model._prepare_step(
            {"src": 0.0, "held": 0.0},
            {"src": 1.0, "held": 0.5},
            0.0,
            10e-9,
            lazy,
        )

        assert self.model._step_future_node_voltages is lazy
        assert self.model._step_future_node_voltages.get("src") == pytest.approx(2.0)
        assert self.model._step_future_node_voltages.get("held") == pytest.approx(0.5)

    def test_voltage_cross_event_marks_future_snapshot_need(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module cross_future_probe(vin, out);
    input voltage vin;
    output voltage out;
    real seen;
    analog begin
        @(cross(V(vin) - 0.5, +1)) seen = 1.0;
        V(out) <+ seen;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()

        assert ModelCls._needs_future_node_voltages is True
        assert model._needs_future_node_voltages_tree() is True

    def test_timer_only_event_does_not_mark_future_snapshot_need(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module timer_future_probe(out);
    output voltage out;
    real seen;
    analog begin
        @(timer(1n)) seen = seen + 1.0;
        V(out) <+ seen;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()

        assert ModelCls._needs_future_node_voltages is False
        assert model._needs_future_node_voltages_tree() is False

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

    def test_simulator_clears_node_resolution_cache_after_run(self):
        class MirrorModel(CompiledModel):
            def __init__(self):
                super().__init__()
                self.node_map = {"inp": "vin", "out": "vout"}

            def evaluate(self, node_voltages, time):
                self._set_output("out", self._get_voltage("inp", node_voltages), node_voltages)

        model = MirrorModel()
        sim = Simulator()
        sim.add_source("vin", dc(0.25))
        sim.add_model(model)
        sim.record("vout")

        result = sim.run(tstop=2e-9, tstep=1e-9)

        assert result.signals["vout"][-1] == pytest.approx(0.25)
        assert sim._perf_stats["node_resolution_cache_entries"] >= 2
        assert sim._perf_stats["node_resolution_cache_models"] == 1
        assert model._node_resolution_cache_enabled is False
        assert model._node_resolution_cache == {}

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

    def test_indexed_output_writer_sees_resolved_output_node(self):
        calls = []
        self.model.node_map["q"] = "q_ext"
        self.model._set_indexed_output_writer(lambda node, value: calls.append((node, value)))

        self.model._set_output("q", 0.9, {})

        assert calls == [("q_ext", pytest.approx(0.9))]

    def test_indexed_output_writer_resolves_parent_mapped_node(self):
        parent = CompiledModel()
        parent.node_map["out"] = "OUT"
        child = CompiledModel()
        child.node_map["z"] = "@parent:out"
        child._parent_model = parent
        parent._child_models = [child]
        calls = []
        parent._set_indexed_output_writer(lambda node, value: calls.append((node, value)))

        child._set_output("z", 1.1, {})

        assert calls == [("OUT", pytest.approx(1.1))]

    def test_indexed_voltage_probe_sees_resolved_read_node(self):
        calls = []
        self.model.node_map["inp"] = "VIN"
        self.model._set_indexed_voltage_probe(
            lambda local, ext, value, event: calls.append((local, ext, value, event))
        )

        value = self.model._get_voltage("inp", {"VIN": 0.7})

        assert value == pytest.approx(0.7)
        assert calls == [("inp", "VIN", pytest.approx(0.7), False)]

    def test_transition_first_call_returns_target(self):
        nv = {}
        val = self.model._transition("t0", time=0.0, target=1.0, rise=10e-9)
        assert val == pytest.approx(1.0)

    def test_transition_active_set_skips_inactive_breakpoint_scan(self):
        self.model._transition("idle", time=0.0, target=0.0, rise=10e-9)

        assert self.model.next_breakpoint(0.0) is None
        assert self.model._perf_stats["transition_breakpoint_scans"] == 0
        assert self.model._perf_stats["transition_breakpoint_inactive_skips"] == 1

    def test_transition_ramps_over_time(self):
        # First call: registers key, returns target immediately (no prior state)
        self.model._transition("t1", time=0.0, target=0.0, rise=10e-9)
        # Second call: set new target, transition should begin
        self.model._transition("t1", time=0.0, target=1.0, rise=10e-9)
        val_mid = self.model._transition("t1", time=5e-9, target=1.0, rise=10e-9)
        assert 0.0 < val_mid < 1.0

    def test_transition_unchanged_target_fastpath_preserves_interruption(self):
        self.model._set_transition_unchanged_fastpath_enabled(True)
        self.model._transition("tf", time=0.0, target=0.0, rise=10e-9, fall=10e-9)
        self.model._transition("tf", time=0.0, target=1.0, rise=10e-9, fall=10e-9)

        fastpath_before = self.model._perf_stats["transition_unchanged_target_fastpath"]
        val_mid = self.model._transition("tf", time=5e-9, target=1.0, rise=10e-9, fall=10e-9)

        assert val_mid == pytest.approx(0.5)
        assert self.model._perf_stats["transition_unchanged_target_fastpath"] == fastpath_before + 1
        assert self.model.transitions["tf"].active is True

        val_interrupted = self.model._transition("tf", time=6e-9, target=0.0, rise=10e-9, fall=10e-9)

        assert val_interrupted == pytest.approx(0.6)
        assert self.model.transitions["tf"].target_val == pytest.approx(0.0)
        assert self.model.transitions["tf"].active is True
        assert self.model._perf_stats["transition_unchanged_target_fastpath"] == fastpath_before + 1

    def test_transition_unchanged_target_fastpath_is_default_disabled(self):
        model = CompiledModel()
        model._transition("tf_off", time=0.0, target=0.0, rise=10e-9, fall=10e-9)
        model._transition("tf_off", time=0.0, target=1.0, rise=10e-9, fall=10e-9)

        val_mid = model._transition("tf_off", time=5e-9, target=1.0, rise=10e-9, fall=10e-9)

        assert val_mid == pytest.approx(0.5)
        assert model._perf_stats["transition_unchanged_target_fastpath"] == 0

    def test_transition_initial_state_records_target_and_parameters_for_fastpath(self):
        self.model._set_transition_unchanged_fastpath_enabled(True)
        val_initial = self.model._transition(
            "tf_initial",
            time=0.0,
            target=1.0,
            rise=10e-9,
            fall=10e-9,
        )

        assert val_initial == pytest.approx(1.0)
        fastpath_before = self.model._perf_stats["transition_unchanged_target_fastpath"]

        val_same = self.model._transition(
            "tf_initial",
            time=1e-9,
            target=1.0,
            rise=10e-9,
            fall=10e-9,
        )

        assert val_same == pytest.approx(1.0)
        assert self.model.transitions["tf_initial"].target_val == pytest.approx(1.0)
        assert self.model.transitions["tf_initial"].rise_time == pytest.approx(10e-9)
        assert self.model._perf_stats["transition_unchanged_target_fastpath"] == fastpath_before + 1

    def test_transition_unchanged_target_fastpath_checks_transition_parameters(self):
        self.model._set_transition_unchanged_fastpath_enabled(True)
        self.model._transition("tf_params", time=0.0, target=0.0, rise=10e-9, fall=10e-9)
        self.model._transition("tf_params", time=0.0, target=1.0, rise=10e-9, fall=10e-9)

        fastpath_before = self.model._perf_stats["transition_unchanged_target_fastpath"]
        val_mid = self.model._transition(
            "tf_params",
            time=5e-9,
            target=1.0,
            rise=5e-9,
            fall=10e-9,
        )

        assert val_mid == pytest.approx(0.5)
        assert self.model._perf_stats["transition_unchanged_target_fastpath"] == fastpath_before

        val_same_params = self.model._transition(
            "tf_params",
            time=6e-9,
            target=1.0,
            rise=10e-9,
            fall=10e-9,
        )

        assert val_same_params == pytest.approx(0.6)
        assert self.model._perf_stats["transition_unchanged_target_fastpath"] == fastpath_before + 1

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

    def test_seeded_normal_random_is_a_deterministic_per_seed_stream(self):
        # Sequential-stream semantics: draws are hashed on (seed, draw index),
        # not wall-clock time, so the sequence is schedule-independent.
        seq_a = [self.model._rand_normal(7, 0.0, 1.0, t) for t in (0.0, 1e-9, 2e-9)]
        seq_b = [self.model._rand_normal(7, 0.0, 1.0, t) for t in (0.0, 1e-9, 2e-9)]
        # Same instance keeps drawing from the stream: no repetition.
        assert all(a != b for a, b in zip(seq_a, seq_b))

        # A fresh instance with the same seed reproduces the sequence exactly,
        # regardless of the times passed at the call sites.
        m2 = CompiledModel()
        seq_c = [m2._rand_normal(7, 0.0, 1.0, t) for t in (5e-9, 6e-9, 7e-9)]
        assert seq_a == pytest.approx(seq_c)

        # Distinct seeds give independent streams.
        m3 = CompiledModel()
        seq_d = [m3._rand_normal(8, 0.0, 1.0, 0.0) for _ in range(3)]
        assert all(c != d for c, d in zip(seq_c, seq_d))

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

    def test_next_breakpoint_uses_rust_transition_scanner_hook(self):
        ts = TransitionState(
            current_val=0.0,
            target_val=1.0,
            start_time=0.0,
            start_val=0.0,
            rise_time=10e-9,
            fall_time=10e-9,
            active=True,
        )
        self.model.transitions["t"] = ts
        self.model._active_transition_keys.add("t")
        self.model._transition_active_keys_known = True
        calls = []

        def scanner(
            start_times,
            start_values,
            target_values,
            delays,
            rise_times,
            fall_times,
            active_flags,
            time,
            min_ramp,
        ):
            calls.append(
                (
                    list(start_times),
                    list(start_values),
                    list(target_values),
                    list(active_flags),
                    time,
                    min_ramp,
                )
            )
            return 2.5e-9

        self.model._set_rust_transition_breakpoint_scanner(scanner)

        bp = self.model.next_breakpoint(0.0)

        assert bp == pytest.approx(2.5e-9)
        assert calls == [([0.0], [0.0], [1.0], [1], 0.0, 0.0)]
        assert self.model._perf_stats["rust_transition_breakpoint_scans"] == 1
        assert self.model._perf_stats["rust_transition_breakpoint_state_scans"] == 1
        assert self.model._perf_stats["rust_transition_breakpoint_fallbacks"] == 0

    def test_next_breakpoint_uses_rust_timer_scanner_hook(self):
        self.model.timer_states["t0"] = 10e-9
        self.model.timer_states["t1"] = 4e-9
        self.model.timer_states["t2"] = 7e-9
        self.model.timer_last_fired["t1"] = 4e-9
        calls = []

        def scanner(next_fire_times, last_fired_times, has_last_fired_flags, time):
            calls.append(
                (
                    list(next_fire_times),
                    list(last_fired_times),
                    list(has_last_fired_flags),
                    time,
                )
            )
            return 7e-9

        self.model._set_rust_timer_breakpoint_scanner(scanner)

        bp = self.model.next_breakpoint(1e-9)

        assert bp == pytest.approx(7e-9)
        assert calls == [
            (
                [10e-9, 4e-9, 7e-9],
                [0.0, 4e-9, 0.0],
                [0, 1, 0],
                1e-9,
            )
        ]
        assert self.model._perf_stats["rust_timer_breakpoint_scans"] == 1
        assert self.model._perf_stats["rust_timer_breakpoint_state_scans"] == 3
        assert self.model._perf_stats["rust_timer_breakpoint_fallbacks"] == 0
        assert self.model._perf_stats["timer_array_sidecar_rebuilds"] == 1
        assert self.model._perf_stats["timer_array_sidecar_scans"] == 1

    def test_timer_setters_maintain_array_sidecar_for_rust_scan(self):
        calls = []

        def scanner(next_fire_times, last_fired_times, has_last_fired_flags, time):
            calls.append(
                (
                    type(next_fire_times).__name__,
                    type(last_fired_times).__name__,
                    type(has_last_fired_flags).__name__,
                    list(next_fire_times),
                    list(last_fired_times),
                    list(has_last_fired_flags),
                    time,
                )
            )
            return 10e-9

        self.model._set_rust_timer_breakpoint_scanner(scanner)
        self.model._set_timer_state("t0", 10e-9)
        self.model._set_timer_state("t1", 4e-9)
        self.model._set_timer_last_fired("t1", 4e-9)

        bp = self.model.next_breakpoint(1e-9)

        assert bp == pytest.approx(10e-9)
        assert calls == [
            (
                "array",
                "array",
                "array",
                [10e-9, 4e-9],
                [0.0, 4e-9],
                [0, 1],
                1e-9,
            )
        ]
        assert list(self.model._timer_next_fire_values) == pytest.approx(
            [10e-9, 4e-9]
        )
        assert list(self.model._timer_last_fired_values) == pytest.approx(
            [0.0, 4e-9]
        )
        assert list(self.model._timer_has_last_fired_flags) == [0, 1]
        assert self.model._perf_stats["timer_array_sidecar_updates"] == 3
        assert self.model._perf_stats["timer_array_sidecar_rebuilds"] == 0
        assert self.model._perf_stats["timer_array_sidecar_scans"] == 1

    def test_simulator_installs_rust_transition_breakpoint_scanner(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module trans_cmp(inp, out);
    input voltage inp;
    output voltage out;
    analog begin
        V(out) <+ transition((V(inp) > 0.45) ? 1.0 : 0.0, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        model = ModelCls()
        model.node_map = {"inp": "IN", "out": "OUT"}
        sim = Simulator()
        sim.add_source(
            "IN",
            pulse(
                0.0,
                0.9,
                period=4e-9,
                width=2e-9,
                rise=100e-12,
                fall=100e-12,
            ),
        )
        sim.add_model(model)
        sim.record("OUT")
        for idx in range(80):
            sim.record(f"DUMMY{idx}")

        sim.run(tstop=3e-9, tstep=250e-12, rust_static_eval=True)

        assert sim._perf_stats["rust_array_loop_enabled"] == 1
        assert sim._perf_stats["rust_transition_breakpoint_enabled"] == 1
        assert sim._perf_stats["rust_transition_breakpoint_scans_total"] > 0
        assert sim._perf_stats["rust_transition_breakpoint_state_scans_total"] > 0
        assert sim._perf_stats["rust_transition_breakpoint_fallbacks_total"] == 0

    def test_simulator_installs_rust_timer_breakpoint_scanner(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_toggle(out);
    output voltage out;
    integer p;
    integer q;
    analog begin
        @(initial_step) begin
            p = 0;
            q = 0;
        end
        @(timer(0, 1n)) p = 1 - p;
        @(timer(1n)) q = 1 - q;
        V(out) <+ p + 2 * q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert "_check_timer_event_batch" in ModelCls._generated_code
        model = ModelCls()
        model.node_map = {"out": "OUT"}
        sim = Simulator()
        sim.add_model(model)
        sim.record("OUT")
        for idx in range(80):
            sim.record(f"DUMMY{idx}")

        sim.run(tstop=3e-9, tstep=250e-12, rust_static_eval=True)

        assert sim._perf_stats["rust_array_loop_enabled"] == 1
        assert sim._perf_stats["rust_timer_breakpoint_enabled"] == 1
        assert sim._perf_stats["rust_timer_breakpoint_scans_total"] > 0
        assert sim._perf_stats["rust_timer_breakpoint_state_scans_total"] > 0
        assert sim._perf_stats["rust_timer_breakpoint_fallbacks_total"] == 0
        assert sim._perf_stats["timer_batch_due_calls_total"] > 0
        assert sim._perf_stats["timer_batch_due_events_total"] >= (
            2 * sim._perf_stats["timer_batch_due_calls_total"]
        )
        assert sim._perf_stats["timer_batch_due_fallbacks_total"] == 0

    def test_timer_event_batch_uses_rust_production_when_enabled(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_toggle(out);
    output voltage out;
    integer p;
    integer q;
    analog begin
        @(initial_step) begin
            p = 0;
            q = 0;
        end
        @(timer(0, 1n)) p = 1 - p;
        @(timer(1n)) q = 1 - q;
        V(out) <+ p + 2 * q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert "_check_timer_event_batch" in ModelCls._generated_code

        def build_sim():
            model = ModelCls()
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=3e-9, tstep=250e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=3e-9,
            tstep=250e-12,
            rust_timer_event=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust._perf_stats["rust_timer_event_requested"] == 1
        assert rust._perf_stats["rust_timer_event_enabled"] == 1
        assert rust._perf_stats["timer_batch_due_calls_total"] > 0
        assert rust._perf_stats["rust_timer_event_production_periodic_calls_total"] > 0
        assert rust._perf_stats["rust_timer_event_production_absolute_calls_total"] > 0
        assert rust._perf_stats["rust_timer_event_production_fallbacks_total"] == 0

    def test_rust_full_model_timer_static_linear_trace_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_counter(out);
    output voltage out;
    parameter real start_t = 0;
    parameter real period = 1n;
    real acc;
    integer toggle;
    analog begin
        @(initial_step) begin
            acc = 1.0;
            toggle = 0;
        end
        @(timer(start_t, period)) begin
            acc = acc + 2.0;
            toggle = 1 - toggle;
        end
        V(out) <+ acc + toggle;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._event_timer_static_linear_ir_ops
        assert ModelCls._evaluate_ir_static_linear_non_event_ops

        def build_sim():
            model = ModelCls()
            model.params["period"] = 700e-12
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=4e-9, tstep=250e-12, record_step=250e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=4e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert any(abs(time - 700e-12) < 1e-18 for time in rust_result.time)
        assert rust._perf_stats["rust_full_model_fastpath_enabled"] == 1
        assert rust._perf_stats["rust_full_model_required_failures"] == 0
        assert rust._perf_stats["rust_full_model_timer_static_linear_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_rust_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_events"] == 6
        assert rust._perf_stats["rust_full_model_timer_static_linear_rust_fallbacks"] == 0

    def test_rust_full_model_timer_static_linear_default_trace_matches_default(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_counter_default_trace(out);
    output voltage out;
    parameter real start_t = 0;
    parameter real period = 1n;
    real acc;
    integer toggle;
    analog begin
        @(initial_step) begin
            acc = 1.0;
            toggle = 0;
        end
        @(timer(start_t, period)) begin
            acc = acc + 2.0;
            toggle = 1 - toggle;
        end
        V(out) <+ acc + toggle;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))

        def build_sim():
            model = ModelCls()
            model.params["period"] = 700e-12
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=4e-9, tstep=250e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=4e-9,
            tstep=250e-12,
            rust_full_model_fastpath=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert any(abs(time - 700e-12) < 1e-18 for time in rust_result.time)
        assert rust._perf_stats["rust_full_model_fastpath_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_rust_enabled"] == 1
        assert rust._perf_stats[
            "rust_full_model_timer_static_linear_default_trace_enabled"
        ] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_rust_fallbacks"] == 0

    def test_rust_full_model_multi_timer_static_linear_queue_preserves_order(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_queue_order(out);
    output voltage out;
    real acc;
    analog begin
        @(initial_step) acc = 1.0;
        @(timer(0, 1n)) acc = acc + 1.0;
        @(timer(0, 1n)) acc = 2.0 * acc;
        V(out) <+ acc;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert len(ModelCls._event_timer_static_linear_ir_ops) == 2
        assert ModelCls._evaluate_ir_static_linear_non_event_ops

        def build_sim():
            model = ModelCls()
            model.node_map = {"out": "OUT"}
            sim = Simulator()
            sim.add_model(model)
            sim.record("OUT")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=2e-9, tstep=250e-12, record_step=250e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=2e-9,
            tstep=250e-12,
            record_step=250e-12,
            rust_full_model_fastpath=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust_result.signals["OUT"][0] == pytest.approx(4.0)
        assert rust_result.signals["OUT"][-1] == pytest.approx(22.0)
        assert rust._perf_stats["rust_full_model_fastpath_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_enabled"] == 1
        assert rust._perf_stats["rust_full_model_timer_static_linear_timer_count"] == 2
        assert rust._perf_stats["rust_full_model_timer_static_linear_events"] == 6
        assert rust._perf_stats["rust_full_model_timer_static_linear_rust_fallbacks"] == 0

    def test_rust_static_eval_executes_piecewise_abs_min_max_ir(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module piecewise_eval(vin, vrect, vclip);
    input voltage vin;
    output voltage vrect, vclip;
    real hi;
    real clip;
    analog begin
        V(vrect) <+ 0.45 + abs(V(vin) - 0.45);
        hi = min(V(vin), 0.8);
        clip = max(hi, 0.2);
        V(vclip) <+ clip;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert ModelCls._evaluate_ir_static_linear_rejections == ()

        def build_sim():
            model = ModelCls()
            model.node_map = {
                "vin": "VIN",
                "vrect": "VRECT",
                "vclip": "VCLIP",
            }
            sim = Simulator()
            sim.add_source("VIN", ramp(0.0, 1.0, 0.0, 1e-9))
            sim.add_model(model)
            sim.record("VRECT")
            sim.record("VCLIP")
            return sim

        ref = build_sim()
        ref_result = ref.run(tstop=1e-9, tstep=250e-12)

        rust = build_sim()
        rust_result = rust.run(
            tstop=1e-9,
            tstep=250e-12,
            rust_static_eval=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["VRECT"]) == pytest.approx(
            list(ref_result.signals["VRECT"])
        )
        assert list(rust_result.signals["VCLIP"]) == pytest.approx(
            list(ref_result.signals["VCLIP"])
        )
        assert rust._perf_stats["rust_static_eval_models"] == 1
        assert rust._perf_stats["rust_static_eval_calls"] > 0
        assert rust._perf_stats["rust_static_eval_errors"] == 0

    def test_state_owned_absolute_timer_fastpath_skips_target_reads(self):
        src = """\
`include "disciplines.vams"
module state_timer(out);
    output voltage out;
    integer q;
    real next_t;
    analog begin
        @(initial_step) begin
            q = 0;
            next_t = 1n;
        end
        @(timer(next_t)) begin
            q = 1 - q;
            next_t = next_t + 1n;
        end
        V(out) <+ q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert "_check_state_owned_timer_at" in ModelCls._generated_code
        assert ModelCls._state_owned_timer_targets == (("timer_0", "next_t"),)

        model = ModelCls()
        model.node_map = {"out": "OUT"}
        sim = Simulator()
        sim.add_model(model)
        sim.record("OUT")

        result = sim.run(tstop=3e-9, tstep=250e-12)

        values = list(result.signals["OUT"])
        assert max(values) == pytest.approx(1.0)
        assert min(values) == pytest.approx(0.0)
        assert sim._perf_stats["timer_state_owned_checks_total"] > 0
        assert sim._perf_stats["timer_state_owned_fires_total"] >= 2
        assert sim._perf_stats["timer_state_owned_fast_skips_total"] > 0
        assert sim._perf_stats["timer_state_owned_target_reads_total"] < (
            sim._perf_stats["timer_state_owned_checks_total"]
        )

    def test_state_owned_absolute_timer_fastpath_rejects_external_target_writes(self):
        src = """\
`include "disciplines.vams"
module unsafe_state_timer(clk, out);
    input voltage clk;
    output voltage out;
    integer q;
    real next_t;
    analog begin
        @(initial_step) begin
            q = 0;
            next_t = 1n;
        end
        @(cross(V(clk) - 0.45, +1)) next_t = next_t + 1n;
        @(timer(next_t)) begin
            q = 1 - q;
            next_t = next_t + 1n;
        end
        V(out) <+ q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert "_check_state_owned_timer_at" not in ModelCls._generated_code
        assert ModelCls._state_owned_timer_targets == ()

    def test_rust_timer_event_path_matches_default_without_indexed_arrays(self):
        _build_rust_core_or_skip()
        src = """\
`include "disciplines.vams"
module timer_mix(out);
    output voltage out;
    integer p;
    integer q;
    real next_t;
    analog begin
        @(initial_step) begin
            p = 0;
            q = 0;
            next_t = 1n;
        end
        @(timer(0, 1n)) p = 1 - p;
        @(timer(next_t)) begin
            q = 1 - q;
            next_t = next_t + 1n;
        end
        V(out) <+ p + 2 * q;
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        assert "_check_timer_event_batch" not in ModelCls._generated_code

        ref_model = ModelCls()
        ref_model.node_map = {"out": "OUT"}
        ref_sim = Simulator()
        ref_sim.add_model(ref_model)
        ref_sim.record("OUT")
        ref_result = ref_sim.run(tstop=3e-9, tstep=250e-12)

        rust_model = ModelCls()
        rust_model.node_map = {"out": "OUT"}
        rust_sim = Simulator()
        rust_sim.add_model(rust_model)
        rust_sim.record("OUT")
        rust_result = rust_sim.run(
            tstop=3e-9,
            tstep=250e-12,
            rust_timer_event=True,
            rust_required=True,
        )

        assert list(rust_result.time) == pytest.approx(list(ref_result.time))
        assert list(rust_result.signals["OUT"]) == pytest.approx(
            list(ref_result.signals["OUT"])
        )
        assert rust_sim._perf_stats["rust_array_loop_enabled"] == 0
        assert rust_sim._perf_stats["rust_timer_event_requested"] == 1
        assert rust_sim._perf_stats["rust_timer_event_available"] == 1
        assert rust_sim._perf_stats["rust_timer_event_enabled"] == 1
        assert rust_sim._perf_stats["rust_timer_breakpoint_enabled"] == 1
        assert rust_sim._perf_stats["rust_timer_breakpoint_scans_total"] > 0
        assert rust_sim._perf_stats["rust_timer_event_production_absolute_calls_total"] > 0
        assert rust_sim._perf_stats["rust_timer_event_production_periodic_calls_total"] > 0
        assert rust_sim._perf_stats["rust_timer_event_production_fallbacks_total"] == 0

    def test_fused_transition_contribution_matches_default_output(self):
        src = """\
`include "disciplines.vams"
module trans_out(clk, out);
    input voltage clk;
    output voltage out;
    real vh = 0.9;
    real vl = 0.0;
    integer q;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk) - 0.45, +1)) q = 1 - q;
        V(out) <+ vl + (vh - vl) * transition(q ? 1.0 : 0.0, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        # Audit 088 may emit the lazy (deferred) form; both are valid fastpath.
        assert (
            "_transition_output(" in ModelCls._generated_code
            or "_transition_output_lazy(" in ModelCls._generated_code
        )

        model = ModelCls()
        model.node_map = {"clk": "CLK", "out": "OUT"}
        sim = Simulator()
        sim.add_source(
            "CLK",
            pulse(
                0.0,
                0.9,
                period=4e-9,
                width=2e-9,
                rise=100e-12,
                fall=100e-12,
            ),
        )
        sim.add_model(model)
        sim.record("OUT")

        result = sim.run(tstop=6e-9, tstep=250e-12)

        assert result.signals["OUT"][0] == pytest.approx(0.0)
        assert max(result.signals["OUT"]) == pytest.approx(0.9, abs=1e-12)
        assert sim._perf_stats["transition_output_fastpath_calls_total"] > 0
        assert sim._perf_stats["transition_calls_total"] == sim._perf_stats[
            "transition_output_fastpath_calls_total"
        ]

    def test_fused_transition_contribution_handles_differential_output(self):
        src = """\
`include "disciplines.vams"
module trans_diff(clk, vdd, vss, out);
    input voltage clk;
    input voltage vdd;
    input voltage vss;
    output voltage out;
    integer q;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk, vss) - 0.45, +1)) q = 1 - q;
        V(out, vss) <+ V(vdd, vss) * transition(q ? 1.0 : 0.0, 0.0, 1n, 1n);
    end
endmodule
"""
        ModelCls = compile_module(parse(src))
        # Audit 088 may emit the lazy (deferred) form; both are valid fastpath.
        assert (
            "_transition_output(" in ModelCls._generated_code
            or "_transition_output_lazy(" in ModelCls._generated_code
        )

        model = ModelCls()
        model.node_map = {"clk": "CLK", "vdd": "VDD", "vss": "VSS", "out": "OUT"}
        sim = Simulator()
        sim.add_source("VDD", dc(0.9))
        sim.add_source("VSS", dc(0.0))
        sim.add_source(
            "CLK",
            pulse(
                0.0,
                0.9,
                period=4e-9,
                width=2e-9,
                rise=100e-12,
                fall=100e-12,
            ),
        )
        sim.add_model(model)
        sim.record("OUT")

        result = sim.run(tstop=6e-9, tstep=250e-12)

        assert result.signals["OUT"][0] == pytest.approx(0.0)
        assert max(result.signals["OUT"]) == pytest.approx(0.9, abs=1e-12)
        assert sim._perf_stats["transition_output_fastpath_calls_total"] > 0

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

    def test_next_breakpoint_uses_timer_cache_until_state_changes(self):
        self.model._set_timer_state("t0", 10e-9)

        assert self.model.next_breakpoint(0.0) == pytest.approx(10e-9)
        scans = self.model._perf_stats["timer_breakpoint_scans"]
        assert self.model.next_breakpoint(1e-9) == pytest.approx(10e-9)

        assert self.model._perf_stats["timer_breakpoint_scans"] == scans
        assert self.model._perf_stats["timer_breakpoint_cache_hits"] == 1

    def test_next_breakpoint_timer_cache_invalidates_on_reschedule(self):
        self.model._set_timer_state("t0", 10e-9)
        assert self.model.next_breakpoint(0.0) == pytest.approx(10e-9)

        self.model._reschedule_timer("t0", 10e-9, 10e-9)

        assert self.model.next_breakpoint(10e-9) == pytest.approx(20e-9)

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

    def test_realtime_alias_tracks_cross_event_time(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module realtime_probe(clk, out);
    input voltage clk;
    output voltage out;
    real sampled_t;
    analog begin
        @(initial_step) sampled_t = -1.0;
        @(cross(V(clk) - 0.5, +1)) begin
            sampled_t = $realtime;
        end
        V(out) <+ sampled_t;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source("clk", ramp(0.0, 1.0, 0.0, 10e-9))
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=10e-9, tstep=1e-9)

        assert result.signals["out"][-1] == pytest.approx(5e-9, abs=1e-12)

    def test_self_referential_real_update_uses_nominal_step_not_refine_steps(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        src = """\
`include "disciplines.vams"
module self_decay_probe(clk, out);
    input voltage clk;
    output voltage out;
    real env;
    analog begin
        @(initial_step) begin
            env = 1.0;
        end
        @(cross(V(clk) - 0.5, +1)) begin
            env = 1.0;
        end
        if (env > 0.0) begin
            env = env - 0.1;
        end
        V(out) <+ env;
    end
endmodule
"""
        mod = parse(src)
        ModelCls = compile_module(mod)
        model = ModelCls()

        sim = Simulator()
        sim.add_source(
            "clk",
            pulse(v_lo=0.0, v_hi=1.0, delay=1e-9, period=10e-9,
                  width=5e-9, rise=100e-12, fall=100e-12),
        )
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=2e-9, tstep=1e-9, refine_factor=32, refine_steps=32)

        assert result.signals["out"][-1] == pytest.approx(0.8, abs=0.11)

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
        assert sim._perf_stats["model_breakpoint_scan_calls"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["timer_breakpoint_scans_total"] == model._perf_stats[
            "timer_breakpoint_scans"
        ]
        assert sim._perf_stats["timer_breakpoint_hits_total"] == model._perf_stats[
            "timer_breakpoint_hits"
        ]
        assert sim._perf_stats["timer_breakpoint_scans_total"] > 0

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
        assert ModelCls._uses_bound_step is True
        model = ModelCls()
        assert model._uses_bound_step_tree() is True

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        # tstep=10ns but $bound_step limits to 1ns
        result = sim.run(tstop=10e-9, tstep=10e-9)
        # Should have at least 10 steps (10ns / 1ns)
        assert len(result.time) >= 10
        assert sim._perf_stats["bound_step_scan_calls"] == sim._perf_stats["steps_total"]
        # All step sizes (after first) should be <= 1ns + tolerance
        for dt in result.step_sizes[1:]:
            assert dt <= 1e-9 + 1e-15


class TestCompiledModelCapabilityFlags:
    """Regression coverage for simulator scan prefilters."""

    def test_static_contribution_model_skips_dynamic_scans(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        mod = parse("""\
`include "disciplines.vams"
module static_out(out);
    output voltage out;
    analog begin
        V(out) <+ 1.0;
    end
endmodule
""")
        ModelCls = compile_module(mod)
        model = ModelCls()

        assert ModelCls._has_dynamic_breakpoints is False
        assert model._has_dynamic_breakpoints_tree() is False
        assert ModelCls._has_post_update_events is False
        assert ModelCls._uses_bound_step is False
        assert model._uses_bound_step_tree() is False

        sim = Simulator()
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=2e-9, tstep=1e-9)

        assert result.signals["out"][-1] == pytest.approx(1.0)
        assert sim._perf_stats["source_breakpoint_scan_calls"] == 0
        assert sim._perf_stats["model_breakpoint_scan_calls"] == 0
        assert sim._perf_stats["bound_step_scan_calls"] == 0
        assert sim._perf_stats["timer_breakpoint_scans_total"] == 0
        assert sim._perf_stats["model_post_update_calls"] == 0
        assert sim._perf_stats["model_post_update_skips"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["static_lifecycle_fastpath_enabled"] == 1
        assert sim._perf_stats["model_prepare_step_calls"] == 0
        assert sim._perf_stats["model_prepare_step_skips"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["model_timer_expire_calls"] == 0
        assert sim._perf_stats["model_timer_expire_skips"] == sim._perf_stats["steps_total"]

        legacy_sim = Simulator()
        legacy_model = ModelCls()
        legacy_sim.add_model(legacy_model)
        legacy_sim.record("out")
        legacy_result = legacy_sim.run(
            tstop=2e-9,
            tstep=1e-9,
            static_lifecycle_fastpath=False,
        )

        assert legacy_result.signals["out"][-1] == pytest.approx(result.signals["out"][-1])
        assert legacy_sim._perf_stats["static_lifecycle_fastpath_enabled"] == 0
        assert (
            legacy_sim._perf_stats["model_prepare_step_calls"]
            == legacy_sim._perf_stats["steps_total"]
        )
        assert legacy_sim._perf_stats["model_prepare_step_skips"] == 0
        assert (
            legacy_sim._perf_stats["model_timer_expire_calls"]
            == legacy_sim._perf_stats["steps_total"]
        )
        assert legacy_sim._perf_stats["model_timer_expire_skips"] == 0

    def test_cross_event_model_keeps_dynamic_breakpoint_scan(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        mod = parse("""\
`include "disciplines.vams"
module edge_seen(in, out);
    input voltage in;
    output voltage out;
    integer seen;
    analog begin
        @(cross(V(in) - 0.5, +1)) seen = 1;
        V(out) <+ seen;
    end
endmodule
""")
        ModelCls = compile_module(mod)
        model = ModelCls()

        assert ModelCls._has_dynamic_breakpoints is True
        assert model._has_dynamic_breakpoints_tree() is True

        sim = Simulator()
        sim.add_source(
            "in",
            pulse(
                v_lo=0.0,
                v_hi=1.0,
                delay=0.5e-9,
                period=4e-9,
                rise=0.1e-9,
                fall=0.1e-9,
            ),
        )
        sim.add_model(model)
        sim.record("out")
        result = sim.run(tstop=2e-9, tstep=0.25e-9)

        assert result.signals["out"][-1] == pytest.approx(1.0)
        assert sim._perf_stats["model_breakpoint_scan_calls"] > 0
        assert sim._perf_stats["model_prepare_step_calls"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["model_prepare_step_skips"] == 0
        assert sim._perf_stats["model_timer_expire_calls"] == sim._perf_stats["steps_total"]
        assert sim._perf_stats["model_timer_expire_skips"] == 0

    def test_output_dependent_cross_keeps_post_update_path(self):
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        mod = parse("""\
`include "disciplines.vams"
module output_cross(out);
    output voltage out;
    integer seen;
    analog begin
        V(out) <+ seen;
        @(cross(V(out) - 0.5, +1)) seen = 1;
    end
endmodule
""")
        ModelCls = compile_module(mod)

        assert ModelCls._has_dynamic_breakpoints is True
        assert ModelCls._has_post_update_events is True


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

    def test_rust_sim_program_fopen_fstrobe_fclose(self, tmp_path):
        _build_rust_core_or_skip()
        from evas.compiler.parser import parse
        from evas.simulator.backend import compile_module

        outfile = tmp_path / "rust_test_output.txt"
        outfile_s = str(outfile).replace("\\", "/")
        src = f"""\
`include "disciplines.vams"
module rust_fileio_metric(clk, done);
    input voltage clk;
    output voltage done;
    parameter string filename = "{outfile_s}";
    parameter real tr = 100p;
    integer fd;
    integer count;
    analog begin
        @(initial_step) begin
            fd = $fopen(filename, "w");
            count = 0;
        end
        @(cross(V(clk) - 0.5, 1)) begin
            count = count + 1;
            $fwrite(fd, "edge %d at %e", count, $abstime);
        end
        V(done) <+ transition(count > 0, 0.0, tr, tr);
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
        sim.record("done")
        sim.run(
            tstop=100e-9,
            tstep=1e-9,
            record_step=1e-9,
            rust_full_model_fastpath=True,
            rust_full_model_required=True,
            rust_required=True,
            skip_source_error_control=True,
        )

        assert sim._perf_stats["rust_sim_program_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_event_transition_enabled"] == 1
        assert sim._perf_stats["rust_sim_program_side_effects"] == 6
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
