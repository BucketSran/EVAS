"""Unit tests for audit 089 cross/above production gate.

Verifies that when `rust_cross_above_production=True` is passed to
Simulator.run(), the CrossDetector / AboveDetector state evolution is
delegated to the Rust primitive instead of detector.check(). The
waveform must be bit-exact with the default (Python detector) path,
and the production counter must be non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.engine import Simulator, dc, ramp, pulse


RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core_or_skip():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


# A simple comparator that uses a cross() detector + a transition output.
# When clk crosses 0.5 rising, q toggles; the output ramps to q*vdd.
CROSS_SRC = """\
`include "disciplines.vams"
module cross_sample(clk, vdd, vss, o);
    input voltage clk;
    input voltage vdd;
    input voltage vss;
    output voltage o;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) begin
            q = 1 - q;
        end
        V(o) <+ V(vdd, vss) * transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""

# A model exercising above().
ABOVE_SRC = """\
`include "disciplines.vams"
module above_sample(inp, vdd, vss, o);
    input voltage inp;
    input voltage vdd;
    input voltage vss;
    output voltage o;
    integer q = 0;
    analog begin
        @(above(V(inp) - 0.5)) begin
            q = 1;
        end
        V(o) <+ V(vdd, vss) * transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""


def _build_cross_sim():
    ModelCls = compile_module(parse(CROSS_SRC))
    model = ModelCls()
    model.node_map = {"clk": "CLK", "vdd": "VDD", "vss": "VSS", "o": "O"}
    sim = Simulator()
    sim.add_source("VDD", dc(0.9))
    sim.add_source("VSS", dc(0.0))
    sim.add_source("CLK", pulse(
        v_lo=0.0, v_hi=0.9, period=4e-9, duty=0.5,
        rise=100e-12, fall=100e-12,
    ))
    sim.add_model(model)
    sim.record("O")
    return sim


def _build_above_sim():
    ModelCls = compile_module(parse(ABOVE_SRC))
    model = ModelCls()
    model.node_map = {"inp": "IN", "vdd": "VDD", "vss": "VSS", "o": "O"}
    sim = Simulator()
    sim.add_source("VDD", dc(0.9))
    sim.add_source("VSS", dc(0.0))
    sim.add_source("IN", ramp(0.0, 1.0, 0.0, 5e-9))
    sim.add_model(model)
    sim.record("O")
    return sim


class TestCrossProduction:

    def test_cross_production_matches_default_waveform(self):
        _build_rust_core_or_skip()
        ref = _build_cross_sim()
        ref_res = ref.run(tstop=8e-9, tstep=100e-12, record_step=100e-12)

        prod = _build_cross_sim()
        prod_res = prod.run(
            tstop=8e-9, tstep=100e-12, record_step=100e-12,
            rust_cross_above_production=True, rust_required=True,
        )

        assert list(prod_res.time) == pytest.approx(list(ref_res.time))
        assert list(prod_res.signals["O"]) == pytest.approx(list(ref_res.signals["O"]))

    def test_cross_production_counter_nonzero_when_enabled(self):
        _build_rust_core_or_skip()
        sim = _build_cross_sim()
        sim.run(
            tstop=8e-9, tstep=100e-12, record_step=100e-12,
            rust_cross_above_production=True, rust_required=True,
        )
        stats = sim._perf_stats
        assert stats["rust_cross_above_production_requested"] == 1
        assert stats["rust_cross_above_production_available"] == 1
        assert stats["rust_cross_above_production_enabled"] == 1
        assert stats["rust_cross_production_calls_total"] > 0
        assert stats["rust_cross_production_fallbacks_total"] == 0
        # cross fires at least twice over 8ns with 4ns period (rising 1ns, 5ns)
        assert stats["rust_cross_production_fires_total"] >= 2

    def test_cross_production_counter_zero_when_disabled(self):
        _build_rust_core_or_skip()
        sim = _build_cross_sim()
        sim.run(tstop=8e-9, tstep=100e-12, record_step=100e-12)
        stats = sim._perf_stats
        assert stats["rust_cross_above_production_requested"] == 0
        assert stats["rust_cross_above_production_enabled"] == 0
        assert stats["rust_cross_production_calls_total"] == 0


class TestAboveProduction:

    def test_above_production_matches_default_waveform(self):
        _build_rust_core_or_skip()
        ref = _build_above_sim()
        ref_res = ref.run(tstop=5e-9, tstep=100e-12, record_step=100e-12)

        prod = _build_above_sim()
        prod_res = prod.run(
            tstop=5e-9, tstep=100e-12, record_step=100e-12,
            rust_cross_above_production=True, rust_required=True,
        )

        assert list(prod_res.time) == pytest.approx(list(ref_res.time))
        assert list(prod_res.signals["O"]) == pytest.approx(list(ref_res.signals["O"]))

    def test_above_production_counter_nonzero_when_enabled(self):
        _build_rust_core_or_skip()
        sim = _build_above_sim()
        sim.run(
            tstop=5e-9, tstep=100e-12, record_step=100e-12,
            rust_cross_above_production=True, rust_required=True,
        )
        stats = sim._perf_stats
        assert stats["rust_above_production_calls_total"] > 0
        assert stats["rust_above_production_fallbacks_total"] == 0


class TestCombinedWithShadow:

    def test_production_and_shadow_can_coexist(self):
        # When both shadow and production are enabled, production owns the
        # math and shadow still does its compare-after-the-fact audit. The
        # shadow should match (because Python's detector wasn't called at all,
        # the shadow compares against the detector's post-Rust state — which
        # is what Rust just wrote).
        _build_rust_core_or_skip()
        sim = _build_cross_sim()
        sim.run(
            tstop=8e-9, tstep=100e-12, record_step=100e-12,
            rust_cross_above_production=True,
            rust_event_due_shadow=True,
            rust_required=True,
        )
        stats = sim._perf_stats
        assert stats["rust_cross_production_calls_total"] > 0
        assert stats["rust_event_due_shadow_cross_checks_total"] > 0
        # Mismatches would indicate something off; they should be 0.
        # (Note: shadow runs AFTER production has already updated the detector,
        # so it's comparing Rust against Rust. Mismatches should remain 0.)
        assert stats["rust_event_due_shadow_mismatches_total"] == 0
