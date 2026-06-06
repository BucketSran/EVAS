"""Unit tests for audit 091d generic-executor body.

091d added a legacy opt-in fixed-grid executor. Current EVAS2 RustSimProgram
coverage accepts this fixture before the legacy executor, so the activation
tests assert that production prefers the strict RustSimProgram path instead of
forcing the older proof-of-concept executor.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import time as _time
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.engine import Simulator, dc, pulse


RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core_or_skip():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


GEN_SRC = """\
`include "disciplines.vams"
module gen_exec_sample(clk, vdd, vss, o1, o2);
    input voltage clk;
    input voltage vdd;
    input voltage vss;
    output voltage o1;
    output voltage o2;
    integer state = 0;
    integer b1 = 0;
    integer b2 = 0;
    analog begin
        @(initial_step) begin
            state = 0;
            b1 = 0;
            b2 = 0;
        end
        @(cross(V(clk) - 0.45, +1)) begin
            if (state == 0) begin
                b1 = 1;
                state = 1;
            end else if (state == 1) begin
                b2 = 1;
                state = 2;
            end else begin
                state = 0;
                b1 = 0;
                b2 = 0;
            end
        end
        V(o1) <+ V(vdd, vss) * transition(b1 ? 1.0 : 0.0, 0.0, 1n, 2n);
        V(o2) <+ V(vdd, vss) * transition(b2 ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""


def _build_sim():
    ModelCls = compile_module(parse(GEN_SRC))
    model = ModelCls()
    model.node_map = {"clk": "CLK", "vdd": "VDD", "vss": "VSS",
                      "o1": "O1", "o2": "O2"}
    sim = Simulator()
    sim.add_source("VDD", dc(0.9))
    sim.add_source("VSS", dc(0.0))
    sim.add_source("CLK", pulse(
        v_lo=0.0, v_hi=0.9, period=4e-9, duty=0.5,
        rise=100e-12, fall=100e-12,
    ))
    sim.add_model(model)
    sim.record("O1")
    sim.record("O2")
    return sim


class TestExecutorActivation:

    def test_rust_sim_program_supersedes_executor_when_supported(self):
        _build_rust_core_or_skip()
        sim = _build_sim()
        sim.run(
            tstop=12e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            generic_executor=True,
        )
        stats = sim._perf_stats
        assert stats["rust_sim_program_enabled"] == 1
        assert stats["rust_sim_program_event_transition_enabled"] == 1
        # The legacy executor is below RustSimProgram and should not steal a
        # design that the strict Rust path can already own.
        assert stats["generic_executor_models_with_candidate"] == 0
        assert stats["generic_executor_runs"] == 0
        assert stats["generic_executor_runtime_fallbacks"] == 0

    def test_executor_does_not_run_without_flag(self):
        _build_rust_core_or_skip()
        sim = _build_sim()
        sim.run(
            tstop=4e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            # generic_executor=False default
        )
        stats = sim._perf_stats
        assert stats["rust_sim_program_enabled"] == 1
        assert stats["rust_sim_program_event_transition_enabled"] == 1
        assert stats["generic_executor_models_with_candidate"] == 0
        assert stats["generic_executor_runs"] == 0


class TestExecutorProducesSensibleOutput:

    def test_executor_output_in_valid_range(self):
        # The output is transition() of 0/1 state * VDD (0.9).
        # So any recorded value must be within [0, 0.9].
        _build_rust_core_or_skip()
        sim = _build_sim()
        result = sim.run(
            tstop=20e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            generic_executor=True,
        )
        for sig_name in ("O1", "O2"):
            values = result.signals[sig_name]
            for v in values:
                assert -0.01 <= v <= 0.91, f"{sig_name}={v} outside [0, 0.9]"

    def test_executor_responds_to_clock_edges(self):
        # After several clock edges, b1 should eventually be 1, which means
        # O1 should reach a non-trivial high value.
        _build_rust_core_or_skip()
        sim = _build_sim()
        result = sim.run(
            tstop=20e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            generic_executor=True,
        )
        # O1 should reach > 0.4 V at some point (transitions to VDD when b1=1)
        assert max(result.signals["O1"]) > 0.4
        assert max(result.signals["O2"]) > 0.4


class TestExecutorVsPython:

    def test_terminal_values_match_within_tolerance(self):
        # Python's adaptive stepper vs 091d's fixed grid will differ in
        # transient details but should converge to the same logical state
        # after several full clock cycles. Compare last value in record.
        _build_rust_core_or_skip()
        ref = _build_sim()
        ref_res = ref.run(
            tstop=40e-9, tstep=100e-12, record_step=100e-12,
        )
        gen = _build_sim()
        gen_res = gen.run(
            tstop=40e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            generic_executor=True,
        )
        # FSM cycles every 3 clock edges (period 4ns). After 40ns we have
        # ~10 edges, FSM cycled multiple times; b1/b2/state ends in a
        # predictable state. Both implementations should agree on the
        # rough waveform shape: max(O1), max(O2), and mean(O1+O2).
        for sig in ("O1", "O2"):
            ref_max = max(ref_res.signals[sig])
            gen_max = max(gen_res.signals[sig])
            assert ref_max > 0.4 and gen_max > 0.4, (
                f"signal {sig}: ref_max={ref_max} gen_max={gen_max}"
            )
            ref_mean = sum(ref_res.signals[sig]) / len(ref_res.signals[sig])
            gen_mean = sum(gen_res.signals[sig]) / len(gen_res.signals[sig])
            assert abs(ref_mean - gen_mean) / max(ref_mean, 0.1) < 0.15, (
                f"signal {sig}: ref_mean={ref_mean} gen_mean={gen_mean}"
            )


class TestExecutorWallSmoke:

    def test_executor_run_completes_in_reasonable_time(self):
        # Smoke check: with the executor enabled, the run should complete
        # in similar (or better) wall time than Python evaluate. We don't
        # assert a speedup — just that the executor doesn't make things
        # dramatically slower (regression guard).
        _build_rust_core_or_skip()
        ref = _build_sim()
        t0 = _time.perf_counter()
        ref.run(tstop=20e-9, tstep=100e-12, record_step=100e-12)
        ref_wall = _time.perf_counter() - t0
        gen = _build_sim()
        t0 = _time.perf_counter()
        gen.run(
            tstop=20e-9, tstep=100e-12, record_step=100e-12,
            rust_full_model_fastpath=True, rust_required=True,
            generic_executor=True,
        )
        gen_wall = _time.perf_counter() - t0
        # Allow generous 5x slowdown (091d is opt-in proof-of-concept).
        # Real performance characterization is for 091e.
        assert gen_wall < ref_wall * 5.0, (
            f"executor too slow: ref={ref_wall:.4f}s gen={gen_wall:.4f}s"
        )
