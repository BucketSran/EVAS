#!/usr/bin/env python3
"""Profile cmp_delay to find where wall time actually goes.

Uses Simulator.run(profile_sections=True) to enable per-section timing.
After the run, prints all `*_s` perf stats sorted by elapsed time so we
can see which phases dominate.
"""
from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from evas.netlist.runner import evas_simulate
from evas.simulator.engine import Simulator


HERE = Path(__file__).resolve().parent
SCS_PATH = HERE.parent / "evas" / "examples" / "comparator" / "tb_cmp_delay.scs"
REPEATS = 5


def run_once():
    captured = []
    orig_run = Simulator.run
    def wrap_run(self, *a, **kw):
        captured.append(self)
        kw["profile_sections"] = True
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
            log_text = Path(tmpd, "sim.log").read_text(errors="ignore")
    finally:
        Simulator.run = orig_run
    sim = captured[0]
    # Section times live on sim._profile_times, not _perf_stats.
    sections = {
        k: v for k, v in sim._profile_times.items()
        if isinstance(v, (int, float)) and v > 0
    }
    # CSV write time comes from the runner log
    csv_write_s = 0.0
    for line in log_text.splitlines():
        if "csv_write_s" in line:
            try:
                csv_write_s = float(line.split("=")[1].split("s")[0].strip())
            except (IndexError, ValueError):
                pass
    return wall, sections, csv_write_s


def main():
    if not SCS_PATH.exists():
        raise SystemExit(f"missing testbench: {SCS_PATH}")
    print(f"workload: {SCS_PATH}")
    print(f"repeats : {REPEATS}\n")

    walls = []
    section_runs = []
    csv_runs = []
    for _ in range(REPEATS):
        wall, sec, csv = run_once()
        walls.append(wall)
        section_runs.append(sec)
        csv_runs.append(csv)

    # Aggregate: median per section name
    all_keys = set()
    for run in section_runs:
        all_keys |= set(run.keys())
    median_sections = {
        k: statistics.median([run.get(k, 0.0) for run in section_runs])
        for k in all_keys
    }
    csv_med = statistics.median(csv_runs)
    wall_med = statistics.median(walls)

    print(f"=== cmp_delay profile (median of {REPEATS} runs) ===\n")
    print(f"Total wall (evas_simulate full path): {wall_med:.4f} s")
    print(f"CSV write (from runner log)         : {csv_med:.4f} s\n")

    # Sort by elapsed descending
    sorted_secs = sorted(median_sections.items(), key=lambda kv: kv[1], reverse=True)
    sec_sum = sum(median_sections.values())
    print(f"Profile sections (sum {sec_sum:.4f} s, {sec_sum/wall_med*100:.1f}% of wall):")
    print(f"  {'section':40s} {'time_s':>10s}  {'% wall':>8s}  {'% sec':>8s}")
    print(f"  {'-'*40} {'-'*10}  {'-'*8}  {'-'*8}")
    for name, t in sorted_secs:
        pct_wall = t / wall_med * 100
        pct_sec = t / sec_sum * 100
        print(f"  {name:40s} {t:10.4f}  {pct_wall:7.2f}%  {pct_sec:7.2f}%")

    # Add CSV write and "unaccounted" rows
    unaccounted = wall_med - sec_sum - csv_med
    print(f"\n  {'csv_write_s (runner-side)':40s} {csv_med:10.4f}  "
          f"{csv_med/wall_med*100:7.2f}%  (outside Simulator.run)")
    print(f"  {'unaccounted (engine outer + import)':40s} {unaccounted:10.4f}  "
          f"{unaccounted/wall_med*100:7.2f}%")


if __name__ == "__main__":
    main()
