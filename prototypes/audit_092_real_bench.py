#!/usr/bin/env python3
"""Audit 092 real-row validation of 091d generic executor.

Picks pipeline_stage (vbr1_l1_pipeline_adc_stage/forms/tb) — a real
vabench row that 091a confirmed matches the generic candidate and
has no $strobe in event bodies. Runs the full testbench through
evas_simulate with and without `generic_executor=True`, measures
wall + output parity over 15 trimmed-mean repeats.

Run:
    PYTHONPATH=. python3 prototypes/audit_092_real_bench.py
"""
from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from evas.netlist.runner import evas_simulate
from evas.simulator.engine import Simulator


REPEATS = 15
REPO = Path(__file__).resolve().parents[2]
SCS_PATH = (
    REPO / "behavioral-veriloga-eval/benchmark-vabench-release-v1/tasks"
    / "CT01_data_converter_models/vbr1_l1_pipeline_adc_stage/forms/tb/gold"
    / "tb_pipeline_stage_ref.scs"
)


def _trimmed_mean(values, trim=2):
    if len(values) <= 2 * trim:
        return statistics.mean(values)
    return statistics.mean(sorted(values)[trim:-trim])


def run_once(use_executor: bool):
    captured = []
    orig_run = Simulator.run

    def wrap_run(self, *a, **kw):
        captured.append(self)
        kw["rust_required"] = True
        if use_executor:
            kw["rust_full_model_fastpath"] = True
            kw["generic_executor"] = True
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
                raise RuntimeError("evas_simulate failed")
            # Capture CSV output for parity comparison
            csv_files = list(Path(tmpd).glob("*.csv"))
            csv_text = csv_files[0].read_text() if csv_files else ""
    finally:
        Simulator.run = orig_run

    sim = captured[0]
    stats = sim._perf_stats
    return {
        "wall_s": wall,
        "executor_runs": stats.get("generic_executor_runs", 0),
        "fallbacks": stats.get("generic_executor_runtime_fallbacks", 0),
        "models_with_candidate": stats.get(
            "generic_executor_models_with_candidate", 0
        ),
        "csv_text": csv_text,
        "csv_len": len(csv_text),
    }


def summarize(label, samples):
    walls = [s["wall_s"] for s in samples]
    s = samples[-1]
    print(f"=== {label} ===")
    print(f"  repeats             = {len(samples)}")
    print(f"  wall median_s       = {statistics.median(walls):.4f}")
    print(f"  wall trimmed_mean_s = {_trimmed_mean(walls):.4f}")
    print(f"  wall stdev_s        = {statistics.stdev(walls):.4f}")
    print(f"  wall min/max_s      = {min(walls):.4f} / {max(walls):.4f}")
    print(f"  executor_runs       = {s['executor_runs']}")
    print(f"  fallbacks           = {s['fallbacks']}")
    print(f"  candidate_models    = {s['models_with_candidate']}")
    print(f"  csv_size_bytes      = {s['csv_len']}")
    return _trimmed_mean(walls), s


def csv_first_diff_line(a: str, b: str) -> str:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    for i, (la, lb) in enumerate(zip(a_lines, b_lines)):
        if la != lb:
            return f"line {i}: {la[:60]!r} vs {lb[:60]!r}"
    if len(a_lines) != len(b_lines):
        return f"length: {len(a_lines)} vs {len(b_lines)} lines"
    return "(identical)"


def main():
    if not SCS_PATH.exists():
        raise SystemExit(f"missing testbench: {SCS_PATH}")
    print(f"workload : {SCS_PATH.relative_to(REPO)}")
    print(f"repeats  : {REPEATS}\n")

    # Warm up
    run_once(use_executor=False)

    py = [run_once(use_executor=False) for _ in range(REPEATS)]
    ge = [run_once(use_executor=True) for _ in range(REPEATS)]

    py_trim, py_last = summarize("Python adaptive (default)", py)
    print()
    ge_trim, ge_last = summarize("091d generic executor", ge)
    print()

    if ge_trim > 0:
        speedup = py_trim / ge_trim
        delta_pct = (py_trim - ge_trim) / py_trim * 100
        print(f"speedup (Python / 091d) = {speedup:.3f}x  ({delta_pct:+.2f}% faster)")
    print()

    # Parity check on CSV output
    parity_status = csv_first_diff_line(py_last["csv_text"], ge_last["csv_text"])
    print(f"CSV parity (first diff): {parity_status}")
    if py_last["csv_len"] != ge_last["csv_len"]:
        print(f"CSV size differs: {py_last['csv_len']} vs {ge_last['csv_len']} bytes")
    else:
        print(f"CSV size identical: {py_last['csv_len']} bytes")


if __name__ == "__main__":
    main()
