#!/usr/bin/env python3
"""Audit 093 Phase A: Large-sweep validation of 091d generic executor.

Iterates over every vabench tb-form .va file in the rust coverage
manifest that (a) matches the 091b generic candidate and (b) has a
testbench .scs nearby. For each, runs evas_simulate with and without
`generic_executor=True` (3 repeats each), records wall + fallback +
CSV size delta. Outputs ranked table + summary stats.

Run:
    PYTHONPATH=. python3 prototypes/audit_093_sweep.py
"""
from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
import traceback
from pathlib import Path

from evas.compiler.parser import parse
from evas.netlist.runner import evas_simulate
from evas.simulator.backend import compile_module
from evas.simulator.engine import Simulator


REPO = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO / "behavioral-veriloga-eval/speed-optimization/reports"
    / "current_release_rust_coverage_manifest_20260604.json"
)
PER_ROW_REPEATS = 3
TIMEOUT_PER_RUN_S = 60.0  # skip rows that take longer than this in Python mode
OUTPUT = Path(__file__).parent / "audit_093_sweep_results.json"


def _trimmed_mean(values, trim=0):
    if not values:
        return 0.0
    if len(values) <= 2 * trim:
        return statistics.mean(values)
    return statistics.mean(sorted(values)[trim:-trim])


def run_once(scs_path: Path, use_executor: bool, deadline_s: float):
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
    error = None
    csv_size = 0
    wall = 0.0
    stats = {}
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            t0 = time.perf_counter()
            try:
                ok = evas_simulate(
                    str(scs_path),
                    log_path=str(Path(tmpd) / "sim.log"),
                    output_dir=tmpd,
                )
            except Exception as e:
                error = f"sim_error: {e}"[:120]
                ok = False
            wall = time.perf_counter() - t0
            if wall > deadline_s:
                error = error or f"timeout {wall:.1f}s"
            if ok:
                csv_files = list(Path(tmpd).glob("*.csv"))
                if csv_files:
                    csv_size = csv_files[0].stat().st_size
    finally:
        Simulator.run = orig_run

    if captured:
        sim = captured[0]
        stats = {
            "executor_runs": sim._perf_stats.get("generic_executor_runs", 0),
            "fallbacks": sim._perf_stats.get("generic_executor_runtime_fallbacks", 0),
            "candidate_models": sim._perf_stats.get(
                "generic_executor_models_with_candidate", 0
            ),
        }
    return {
        "wall_s": wall,
        "csv_size": csv_size,
        "error": error,
        **stats,
    }


def find_candidates():
    """Return list of (entry_id, scs_path) for tb-form models that match the
    091b generic candidate and have a tb_*.scs nearby."""
    manifest = json.loads(MANIFEST.read_text())
    found = []
    seen_sha = set()
    for m in manifest["models"]:
        if m.get("form") != "tb":
            continue
        if m.get("sha256") in seen_sha:
            continue
        sigs = set(m.get("rust_signals", []))
        if "whole_segment_candidate" in sigs:
            continue
        if "transition_target_ir" not in sigs:
            continue
        try:
            va_path = REPO / "behavioral-veriloga-eval" / m["path"]
            text = va_path.read_text(errors="ignore")
            if any(tok in text for tok in ("$strobe", "$display", "$write")):
                continue
            mod = parse(text)
            Cls = compile_module(mod)
            kinds = [c[0] for c in Cls._whole_segment_candidates]
            if "generic_event_state_transition_v1" not in kinds:
                continue
            tb_dir = va_path.parent
            scs_files = sorted(tb_dir.glob("tb_*.scs")) + sorted(tb_dir.glob("*.scs"))
            if not scs_files:
                continue
            seen_sha.add(m["sha256"])
            found.append((m["entry_id"], scs_files[0]))
        except Exception:
            continue
    return found


def measure_row(scs_path: Path):
    """Run 3 repeats each of Python and 091d, return aggregated stats."""
    py_walls, py_csvs, py_errors = [], [], None
    for _ in range(PER_ROW_REPEATS):
        result = run_once(scs_path, use_executor=False, deadline_s=TIMEOUT_PER_RUN_S)
        if result.get("error"):
            py_errors = result["error"]
            return {"py_error": py_errors}
        py_walls.append(result["wall_s"])
        py_csvs.append(result["csv_size"])

    ge_walls, ge_csvs, ge_errors, ge_fallbacks = [], [], None, 0
    ge_runs = 0
    for _ in range(PER_ROW_REPEATS):
        result = run_once(scs_path, use_executor=True, deadline_s=TIMEOUT_PER_RUN_S)
        if result.get("error"):
            ge_errors = result["error"]
            return {
                "py_wall_med": statistics.median(py_walls),
                "py_csv": py_csvs[-1],
                "ge_error": ge_errors,
            }
        ge_walls.append(result["wall_s"])
        ge_csvs.append(result["csv_size"])
        ge_fallbacks += result.get("fallbacks", 0)
        ge_runs += result.get("executor_runs", 0)

    py_med = statistics.median(py_walls)
    ge_med = statistics.median(ge_walls)
    speedup = py_med / ge_med if ge_med > 0 else float("nan")
    csv_diff_pct = (
        (ge_csvs[-1] - py_csvs[-1]) / py_csvs[-1] * 100 if py_csvs[-1] else 0.0
    )
    return {
        "py_wall_med": py_med,
        "py_wall_min": min(py_walls),
        "ge_wall_med": ge_med,
        "ge_wall_min": min(ge_walls),
        "speedup": speedup,
        "py_csv": py_csvs[-1],
        "ge_csv": ge_csvs[-1],
        "csv_diff_pct": csv_diff_pct,
        "fallbacks": ge_fallbacks,
        "executor_runs": ge_runs,
    }


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"missing manifest: {MANIFEST}")
    candidates = find_candidates()
    print(f"Found {len(candidates)} tb-form 091b-candidate models with .scs\n")

    results = []
    for i, (entry_id, scs_path) in enumerate(candidates, 1):
        rel = scs_path.relative_to(REPO)
        print(f"[{i}/{len(candidates)}] {entry_id}")
        try:
            row = measure_row(scs_path)
        except Exception as e:
            row = {"unexpected": str(e)[:120]}
            traceback.print_exc()
        row["entry_id"] = entry_id
        row["scs_path"] = str(rel)
        results.append(row)
        if "speedup" in row:
            print(f"     py={row['py_wall_med']:.4f}s  ge={row['ge_wall_med']:.4f}s  speedup={row['speedup']:.1f}x  fb={row['fallbacks']}  csv_diff={row['csv_diff_pct']:+.1f}%")
        else:
            print(f"     SKIPPED: {row.get('py_error') or row.get('ge_error') or row.get('unexpected')}")

    # Aggregate
    valid = [r for r in results if "speedup" in r and math.isfinite(r["speedup"])]
    speedups = [r["speedup"] for r in valid]
    fallbacks_total = sum(r["fallbacks"] for r in valid)

    summary = {
        "candidate_count": len(candidates),
        "measured": len(valid),
        "skipped": len(results) - len(valid),
        "speedup_min": min(speedups) if speedups else None,
        "speedup_max": max(speedups) if speedups else None,
        "speedup_median": statistics.median(speedups) if speedups else None,
        "speedup_geomean": (
            math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            if speedups else None
        ),
        "total_fallbacks": fallbacks_total,
        "py_wall_total_s": sum(r["py_wall_med"] for r in valid),
        "ge_wall_total_s": sum(r["ge_wall_med"] for r in valid),
    }

    print("\n=== Aggregate ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:24s} = {v:.4f}")
        else:
            print(f"  {k:24s} = {v}")

    OUTPUT.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
