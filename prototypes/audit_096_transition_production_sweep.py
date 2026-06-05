#!/usr/bin/env python3
"""Audit 096: full-release sweep of rust_transition_production effect.

Audit 086/088 reported "+2.8% real" on transition-heavy single-model
benches but never validated rust_transition_production=True across the
full release. cmp_strongarm regression (audit 097 finding) shows ~2x
SLOWDOWN on transition-light-per-call rows. This sweep settles the
question: across all TB-form rows with transition_target_ir, is the
flag net positive or net negative?

Each row: 3 repeats Python default vs 3 repeats with the flag, median
wall, classify win/neutral/loss.

Run:
    PYTHONPATH=. python3 prototypes/audit_096_transition_production_sweep.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

from evas.netlist.runner import evas_simulate
from evas.simulator.engine import Simulator


REPO = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO / "behavioral-veriloga-eval/speed-optimization/reports"
    / "current_release_rust_coverage_manifest_20260604.json"
)
REPEATS = 3
TIMEOUT_PER_RUN_S = 30.0
OUTPUT = Path(__file__).parent / "audit_096_sweep_results.json"


class NullWriter:
    def write(self, _): pass
    def flush(self): pass


def run_once(scs_path: Path, opt_flags: dict, deadline_s: float):
    captured = []
    orig = Simulator.run
    def wrap(self, *a, **kw):
        captured.append(self)
        kw.update(opt_flags)
        return orig(self, *a, **kw)
    Simulator.run = wrap
    error = None
    wall = 0.0
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            t0 = time.perf_counter()
            real_out = sys.stdout
            sys.stdout = NullWriter()
            try:
                ok = evas_simulate(
                    str(scs_path),
                    log_path=str(Path(tmpd) / "sim.log"),
                    output_dir=tmpd,
                )
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:80]}"
                ok = False
            finally:
                sys.stdout = real_out
            wall = time.perf_counter() - t0
            if wall > deadline_s:
                error = error or f"timeout {wall:.1f}s"
    finally:
        Simulator.run = orig
    return {"wall_s": wall, "error": error}


def find_transition_targets():
    """Return list of (entry_id, scs_path) for tb-form models that
    have transition_target_ir rust signal (i.e., the matcher knows
    there's transition activity)."""
    manifest = json.loads(MANIFEST.read_text())
    found = []
    seen = set()
    for m in manifest["models"]:
        if m.get("form") != "tb":
            continue
        if m.get("sha256") in seen:
            continue
        sigs = set(m.get("rust_signals", []))
        if "transition_target_ir" not in sigs:
            continue
        va_path = REPO / "behavioral-veriloga-eval" / m["path"]
        if not va_path.exists():
            continue
        tb_dir = va_path.parent
        scs_files = sorted(tb_dir.glob("tb_*.scs")) + sorted(tb_dir.glob("*.scs"))
        if not scs_files:
            continue
        seen.add(m["sha256"])
        found.append((m["entry_id"], scs_files[0]))
    return found


def measure_row(scs_path: Path):
    py_walls = []
    for _ in range(REPEATS):
        r = run_once(scs_path, {}, TIMEOUT_PER_RUN_S)
        if r.get("error"):
            return {"py_error": r["error"]}
        py_walls.append(r["wall_s"])

    rust_walls = []
    rust_flags = {"rust_required": True, "rust_transition_production": True}
    for _ in range(REPEATS):
        r = run_once(scs_path, rust_flags, TIMEOUT_PER_RUN_S)
        if r.get("error"):
            return {
                "py_wall_med": statistics.median(py_walls),
                "rust_error": r["error"],
            }
        rust_walls.append(r["wall_s"])

    py_med = statistics.median(py_walls)
    rust_med = statistics.median(rust_walls)
    return {
        "py_wall_med": py_med,
        "rust_wall_med": rust_med,
        "speedup": py_med / rust_med if rust_med > 0 else float("nan"),
    }


def main():
    candidates = find_transition_targets()
    print(f"Found {len(candidates)} unique tb-form rows with transition_target_ir\n",
          flush=True)
    results = []
    for i, (entry_id, scs_path) in enumerate(candidates, 1):
        try:
            row = measure_row(scs_path)
        except Exception as e:
            row = {"unexpected": str(e)[:120]}
        row["entry_id"] = entry_id
        row["scs"] = str(scs_path.relative_to(REPO))
        results.append(row)
        if "speedup" in row:
            print(f"[{i}/{len(candidates)}] {entry_id:60s} py={row['py_wall_med']:.4f}s "
                  f"rust={row['rust_wall_med']:.4f}s  speedup={row['speedup']:.2f}x",
                  flush=True)
        else:
            err = row.get("py_error") or row.get("rust_error") or row.get("unexpected")
            print(f"[{i}/{len(candidates)}] {entry_id:60s} SKIP: {err}", flush=True)

    # Aggregate
    valid = [r for r in results if "speedup" in r and math.isfinite(r["speedup"])]
    speedups = [r["speedup"] for r in valid]
    wins = [r for r in valid if r["speedup"] > 1.05]
    neutral = [r for r in valid if 0.95 <= r["speedup"] <= 1.05]
    losses = [r for r in valid if r["speedup"] < 0.95]
    big_losses = [r for r in valid if r["speedup"] < 0.50]

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
        "wins_gt_1.05x": len(wins),
        "neutral_0.95_to_1.05x": len(neutral),
        "losses_lt_0.95x": len(losses),
        "big_losses_lt_0.5x": len(big_losses),
        "py_wall_total_s": sum(r["py_wall_med"] for r in valid),
        "rust_wall_total_s": sum(r["rust_wall_med"] for r in valid),
    }

    print("\n=== Aggregate ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}", flush=True)
        else:
            print(f"  {k:30s} = {v}", flush=True)

    if big_losses:
        print("\n=== Big losses (rust_transition_production < 0.5x) ===", flush=True)
        for r in sorted(big_losses, key=lambda x: x["speedup"])[:10]:
            print(f"  {r['entry_id']:60s} speedup={r['speedup']:.2f}x  "
                  f"py={r['py_wall_med']:.3f}s  rust={r['rust_wall_med']:.3f}s",
                  flush=True)

    if wins:
        print("\n=== Top 10 wins (rust_transition_production > 1.05x) ===", flush=True)
        for r in sorted(wins, key=lambda x: -x["speedup"])[:10]:
            print(f"  {r['entry_id']:60s} speedup={r['speedup']:.2f}x  "
                  f"py={r['py_wall_med']:.3f}s  rust={r['rust_wall_med']:.3f}s",
                  flush=True)

    OUTPUT.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(f"\nWrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
