#!/usr/bin/env python3
"""Audit 091d wall benchmark.

Measures wall time of running gen_exec_sample (a 091b candidate model)
under three modes:
  - Python adaptive evaluate (default)
  - 091d fixed-grid executor (model.evaluate at fixed grid)

Repeats 15 times, drops top/bottom 2 outliers, reports trimmed mean.

Run:
    PYTHONPATH=. python3 prototypes/audit_091d_wall_bench.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.engine import Simulator, dc, pulse


REPEATS = 15
TSTOP = 80e-9
TSTEP = 100e-12
RECORD_STEP = 100e-12

SRC = """\
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


def _trimmed_mean(values, trim=2):
    if len(values) <= 2 * trim:
        return statistics.mean(values)
    return statistics.mean(sorted(values)[trim:-trim])


def build_sim():
    ModelCls = compile_module(parse(SRC))
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


def run_once(use_executor):
    sim = build_sim()
    kw = dict(tstop=TSTOP, tstep=TSTEP, record_step=RECORD_STEP)
    if use_executor:
        kw.update(
            rust_full_model_fastpath=True,
            rust_required=True,
            generic_executor=True,
        )
    t0 = time.perf_counter()
    result = sim.run(**kw)
    wall = time.perf_counter() - t0
    stats = sim._perf_stats
    return {
        "wall_s": wall,
        "executor_runs": stats.get("generic_executor_runs", 0),
        "fallbacks": stats.get("generic_executor_runtime_fallbacks", 0),
        "point_count": len(result.time),
        "o1_mean": sum(result.signals["O1"]) / max(len(result.signals["O1"]), 1),
        "o2_mean": sum(result.signals["O2"]) / max(len(result.signals["O2"]), 1),
    }


def summarize(label, samples):
    walls = [s["wall_s"] for s in samples]
    s = samples[-1]
    print(f"=== {label} ===")
    print(f"  repeats           = {len(samples)}")
    print(f"  wall median_s     = {statistics.median(walls):.6f}")
    print(f"  wall trimmed_mean_s = {_trimmed_mean(walls):.6f}")
    print(f"  wall stdev_s      = {statistics.stdev(walls):.6f}")
    print(f"  wall min/max_s    = {min(walls):.6f} / {max(walls):.6f}")
    print(f"  executor_runs     = {s['executor_runs']}")
    print(f"  runtime_fallbacks = {s['fallbacks']}")
    print(f"  point_count       = {s['point_count']}")
    print(f"  o1_mean / o2_mean = {s['o1_mean']:.4f} / {s['o2_mean']:.4f}")
    return _trimmed_mean(walls), s


def main():
    print(f"workload: gen_exec_sample (synthetic FSM, 1 cross + 2 transitions)")
    print(f"tstop / tstep / record = {TSTOP:.0e} / {TSTEP:.0e} / {RECORD_STEP:.0e}")
    print(f"repeats  : {REPEATS}\n")

    # Warm up
    run_once(use_executor=False)

    py = [run_once(use_executor=False) for _ in range(REPEATS)]
    ge = [run_once(use_executor=True) for _ in range(REPEATS)]

    py_trim, _ = summarize("Python adaptive (default)", py)
    print()
    ge_trim, _ = summarize("091d generic executor (fixed grid)", ge)
    print()
    speedup = py_trim / ge_trim if ge_trim > 0 else float("nan")
    delta_pct = (py_trim - ge_trim) / py_trim * 100 if py_trim > 0 else 0.0
    print(f"speedup (Python / 091d) = {speedup:.3f}x  ({delta_pct:+.2f}% faster)")


if __name__ == "__main__":
    main()
