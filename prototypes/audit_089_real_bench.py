#!/usr/bin/env python3
"""Audit 089 real-workload benchmark: cross/above detector production gate.

Uses the bundled `comparator/cmp_delay` example (same as vabench
top-wall `vbr1_l1_propagation_delay_comparator`). cmp_delay has TWO
cross() detectors (rising-edge and falling-edge clk crossings) plus
two transition() outputs. This bench measures whether routing the
CrossDetector state evolution through the Rust primitive gives a
measurable wall improvement on a real top-wall row.

Two modes:
- **Python detectors**: default, CrossDetector.check() runs in Python.
- **089 production**: rust_cross_above_production=True, Rust primitive
  drives detector state evolution.

Run:
    PYTHONPATH=. python3 prototypes/audit_089_real_bench.py
"""
from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from evas.netlist.runner import evas_simulate
from evas.simulator.engine import Simulator


REPEATS = 15
HERE = Path(__file__).resolve().parent
SCS_PATH = HERE.parent / "evas" / "examples" / "comparator" / "tb_cmp_delay.scs"


def _trimmed_mean(values, trim=2):
    if len(values) <= 2 * trim:
        return statistics.mean(values)
    cut = sorted(values)[trim:-trim]
    return statistics.mean(cut)


def run_once(use_production: bool):
    captured = []
    orig_run = Simulator.run

    def wrap_run(self, *a, **kw):
        captured.append(self)
        if use_production:
            kw["rust_cross_above_production"] = True
            kw["rust_transition_production"] = True
        kw["rust_required"] = True
        return orig_run(self, *a, **kw)

    Simulator.run = wrap_run
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            t0 = time.perf_counter()
            ok = evas_simulate(
                str(SCS_PATH),
                log_path=str(Path(tmpd) / "sim.log"),
                output_dir=tmpd,
            )
            wall = time.perf_counter() - t0
            if not ok:
                raise RuntimeError("evas_simulate returned False")
    finally:
        Simulator.run = orig_run

    sim = captured[0]
    stats = sim._perf_stats
    return {
        "wall_s": wall,
        "cross_fires": stats.get("cross_fires_total", 0),
        "cross_prod_calls": stats.get("rust_cross_production_calls_total", 0),
        "cross_prod_fires": stats.get("rust_cross_production_fires_total", 0),
        "cross_prod_fallbacks": stats.get("rust_cross_production_fallbacks_total", 0),
        "above_prod_calls": stats.get("rust_above_production_calls_total", 0),
    }


def summarize(label, samples):
    walls = [s["wall_s"] for s in samples]
    s = samples[-1]
    print(f"=== {label} ===")
    print(f"  repeats               = {len(samples)}")
    print(f"  wall median_s         = {statistics.median(walls):.6f}")
    print(f"  wall trimmed_mean_s   = {_trimmed_mean(walls):.6f}  (drop top/bottom 2)")
    print(f"  wall stdev_s          = {statistics.stdev(walls):.6f}")
    print(f"  wall min/max_s        = {min(walls):.6f} / {max(walls):.6f}")
    print(f"  cross_fires           = {s['cross_fires']}")
    print(f"  cross_prod_calls      = {s['cross_prod_calls']}")
    print(f"  cross_prod_fires      = {s['cross_prod_fires']}")
    print(f"  cross_prod_fallbacks  = {s['cross_prod_fallbacks']}")
    print(f"  above_prod_calls      = {s['above_prod_calls']}")
    return statistics.median(walls), _trimmed_mean(walls), s


def main():
    if not SCS_PATH.exists():
        raise SystemExit(f"missing testbench: {SCS_PATH}")
    print(f"workload : {SCS_PATH}")
    print(f"module   : cmp_delay (2 cross() detectors, 2 transition outputs)")
    print(f"repeats  : {REPEATS}\n")

    # Warm up
    run_once(use_production=False)

    py = [run_once(use_production=False) for _ in range(REPEATS)]
    rs = [run_once(use_production=True) for _ in range(REPEATS)]

    py_med, py_trim, py_last = summarize("Python detector (default)", py)
    print()
    rs_med, rs_trim, rs_last = summarize("089 production (Rust detector)", rs)
    print()

    spd_med = py_med / rs_med if rs_med > 0 else float("nan")
    spd_trim = py_trim / rs_trim if rs_trim > 0 else float("nan")
    print(f"speedup median   = {spd_med:.3f}x  ({(py_med-rs_med)/py_med*100:+.2f}%)")
    print(f"speedup trimmed  = {spd_trim:.3f}x  ({(py_trim-rs_trim)/py_trim*100:+.2f}%)")


if __name__ == "__main__":
    main()
