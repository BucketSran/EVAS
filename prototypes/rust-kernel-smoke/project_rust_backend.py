#!/usr/bin/env python3
"""Project likely EVAS subprocess speedups from targeted Rust-kernel work.

This reads the r14 same-slice timing artifact and applies a deliberately small
scenario: only the known slow classes are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "behavioral-veriloga-eval"
    / "speed-optimization"
    / "reports"
    / "e2e_wall_unified_full_20260602_r14_core_fastpath_exactrows.json"
)

MEASUREMENT_ROWS = {
    "vbr1_l1_gain_estimator",
    "vbr1_l2_gain_extraction_convergence_measurement_flow",
}

PFD_EVENT_ROWS = {
    "vbr1_l1_pfd_small_phase_error_response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--measurement-us-per-step", type=float, default=50.0)
    parser.add_argument("--pfd-event-target-s", type=float, default=0.70)
    parser.add_argument("--min-row-s", type=float, default=0.30)
    return parser.parse_args()


def key(row: dict) -> tuple[str, str]:
    return (row["entry_id"], row["form"])


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.artifact.read_text())
    rows = artifact["results"]
    by_key_mode = {(key(row), row["mode"]): row for row in rows}
    evas_rows = [
        row
        for row in rows
        if row["mode"] == "profile_fast_skip_source_error_control"
    ]

    current_evas = sum(row["simulator_subprocess_wall_s"] for row in evas_rows)
    current_ax = sum(
        row["simulator_subprocess_wall_s"]
        for row in rows
        if row["mode"] == "spectre_ax_equalized_precision"
    )
    current_strict = sum(
        row["simulator_subprocess_wall_s"]
        for row in rows
        if row["mode"] == "spectre_reference_strict_primary"
    )

    projected_evas = 0.0
    changes = []
    for row in evas_rows:
        original = row["simulator_subprocess_wall_s"]
        projected = original
        reason = "unchanged"
        steps = row.get("timing", {}).get("accepted_tran_steps")

        if row["entry_id"] in MEASUREMENT_ROWS and steps:
            candidate = max(args.min_row_s, steps * args.measurement_us_per_step / 1_000_000.0)
            projected = min(original, candidate)
            reason = f"measurement_per_step->{args.measurement_us_per_step:g}us"
        elif row["entry_id"] in PFD_EVENT_ROWS:
            projected = min(original, max(args.min_row_s, args.pfd_event_target_s))
            reason = f"pfd_event_target->{args.pfd_event_target_s:g}s"

        projected_evas += projected
        if projected < original:
            ax_row = by_key_mode.get((key(row), "spectre_ax_equalized_precision"), {})
            changes.append(
                {
                    "entry_id": row["entry_id"],
                    "form": row["form"],
                    "reason": reason,
                    "evas_current_s": original,
                    "evas_projected_s": projected,
                    "saved_s": original - projected,
                    "ax_s": ax_row.get("simulator_subprocess_wall_s"),
                    "steps": steps,
                }
            )

    print(json.dumps(
        {
            "artifact": str(args.artifact),
            "scenario": {
                "measurement_us_per_step": args.measurement_us_per_step,
                "pfd_event_target_s": args.pfd_event_target_s,
                "min_row_s": args.min_row_s,
                "changed_rows": len(changes),
            },
            "totals": {
                "evas_current_s": current_evas,
                "evas_projected_s": projected_evas,
                "evas_saved_s": current_evas - projected_evas,
                "spectre_ax_s": current_ax,
                "spectre_strict_s": current_strict,
                "current_ax_over_evas": current_ax / current_evas,
                "projected_ax_over_evas": current_ax / projected_evas,
                "current_strict_over_evas": current_strict / current_evas,
                "projected_strict_over_evas": current_strict / projected_evas,
            },
            "changed_rows": sorted(changes, key=lambda item: item["saved_s"], reverse=True),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
