#!/usr/bin/env python3
"""Audit current compiler-visible EVAS Rust coverage.

This is P0 for the 094 production work: it reports which Verilog-A files expose
current Rust candidates, especially the new 094 body-IR production candidate.
It does not run EVAS/Spectre timing and must not be used as speed evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evas.simulator.rust_coverage import (
    audit_veriloga_paths,
    discover_veriloga_files,
    estimate_event_transition_plan_profiles,
    estimate_event_transition_profiles,
)


def _default_tasks_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "behavioral-veriloga-eval"
        / "benchmark-vabench-release-v1"
        / "tasks"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_tasks_root(),
        help="Root directory to scan for .va files.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    paths = list(discover_veriloga_files(args.root))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    summary = audit_veriloga_paths(paths)
    data = summary.to_dict()
    data["rust_body_ir_rejection_reasons"] = dict(
        Counter(
            row.rust_body_ir_rejection_reason or "unknown"
            for row in summary.rows
            if row.compile_ok and not row.rust_body_ir_candidate
        )
    )
    data["rust_body_ir_rejection_tags"] = dict(
        Counter(
            tag
            for row in summary.rows
            if row.compile_ok and not row.rust_body_ir_candidate
            for tag in (row.rust_body_ir_rejection_tags or ("unknown",))
        )
    )
    data["whole_segment_kind_counts"] = dict(
        Counter(kind for row in summary.rows for kind in row.whole_segment_kinds)
    )
    data["event_transition_estimates"] = estimate_event_transition_profiles(
        summary.rows
    )
    data["event_transition_plan_estimates"] = (
        estimate_event_transition_plan_profiles(summary.rows)
    )

    text = json.dumps(data, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
