#!/usr/bin/env python3
"""Audit 094p smoke: replay a real pipeline_stage row through Rust shadow runtime.

This prototype deliberately does not modify engine.py.  It uses the current
EVAS simulator to produce the reference time/source grid, then replays that
grid through the 094o Rust analog-block shadow runtime and compares output
signals on the same timestamps.

Run:
    PYTHONPATH=EVAS python3 EVAS/prototypes/audit_094p_pipeline_stage_shadow_sweep.py
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.netlist import runner as netlist_runner
from evas.simulator.analog_block_runtime import (
    RustAnalogBlockShadowRuntime,
    try_build_event_then_transition_shadow_runtime,
)
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import default_rust_core_library_path, load_rust_backend


REPO = Path(__file__).resolve().parents[2]
RUST_CORE = REPO / "EVAS" / "evas" / "rust_core"
BENCH_DIR = (
    REPO
    / "behavioral-veriloga-eval/benchmark-vabench-release-v1/tasks"
    / "CT01_data_converter_models/vbr1_l1_pipeline_adc_stage/forms/tb/gold"
)
SCS_PATH = BENCH_DIR / "tb_pipeline_stage_ref.scs"
VA_PATH = BENCH_DIR / "pipeline_stage.va"
SIGNALS = ("vres", "d1", "d0")


def _build_rust_core() -> None:
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


def _run_reference() -> dict[str, Any]:
    out_dir = Path(tempfile.mkdtemp(prefix="vaevas_094p_ref_"))
    t0 = time.perf_counter()
    ok = netlist_runner.evas_simulate(
        str(SCS_PATH),
        log_path=str(out_dir / "sim.log"),
        output_dir=str(out_dir),
    )
    wall_s = time.perf_counter() - t0
    csv_path = out_dir / "tran.csv"
    return {
        "ok": bool(ok),
        "wall_s": wall_s,
        "out_dir": out_dir,
        "csv_path": csv_path,
        "csv_exists": csv_path.exists(),
        "csv_size": csv_path.stat().st_size if csv_path.exists() else 0,
    }


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
        return reader.fieldnames or [], rows


def _write_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["time", "phi1", "phi2", "vin", "vres", "d1", "d0"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_shadow_runtime() -> tuple[RustAnalogBlockShadowRuntime, object, dict[str, int], float]:
    source = VA_PATH.read_text(encoding="utf-8")
    preprocessed_source, _defines, default_transition = preprocess(
        source,
        source_dir=str(VA_PATH.parent),
    )
    module = parse(preprocessed_source)
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    backend = load_rust_backend(default_rust_core_library_path())
    runtime = try_build_event_then_transition_shadow_runtime(
        module,
        backend,
        node_slots,
        default_transition=default_transition or 0.0,
    )
    if runtime is None:
        raise RuntimeError("094o shadow runtime rejected pipeline_stage")
    return runtime, module, node_slots, default_transition or 0.0


def _run_shadow_replay(
    ref_rows: list[dict[str, float]],
) -> dict[str, Any]:
    runtime, module, node_slots, default_transition = _build_shadow_runtime()
    bindings = build_state_binding_ir(module)
    node_values = array("d", [0.0] * len(module.ports))
    state_values = array("d", [0.0] * 7)
    param_values = array("d", [0.45, 0.9, 200e-12])
    shadow_rows: list[dict[str, float]] = []
    fired_count = 0

    t0 = time.perf_counter()
    for idx, row in enumerate(ref_rows):
        node_values[node_slots["VDD"]] = 0.9
        node_values[node_slots["VSS"]] = 0.0
        node_values[node_slots["PHI1"]] = row["phi1"]
        node_values[node_slots["PHI2"]] = row["phi2"]
        node_values[node_slots["VIN"]] = row["vin"]
        node_values[node_slots["VREF"]] = 0.9
        result = runtime.step(
            time=row["time"],
            node_values=node_values,
            state_values=state_values,
            param_values=param_values,
            initial_step=(idx == 0),
        )
        fired_count += len(result.fired_event_statements)
        shadow_rows.append(
            {
                "time": row["time"],
                "phi1": row["phi1"],
                "phi2": row["phi2"],
                "vin": row["vin"],
                "vres": node_values[node_slots["VRES"]],
                "d1": node_values[node_slots["D1"]],
                "d0": node_values[node_slots["D0"]],
            }
        )
    wall_s = time.perf_counter() - t0
    return {
        "wall_s": wall_s,
        "rows": shadow_rows,
        "fired_count": fired_count,
        "default_transition": default_transition,
        "final_state": {
            name: state_values[bindings.resolve(name).slot]
            for name in ("vin_s", "vres_level", "d1_level", "d0_level")
        },
    }


def _compare_outputs(
    ref_rows: list[dict[str, float]],
    shadow_rows: list[dict[str, float]],
) -> dict[str, Any]:
    max_abs = 0.0
    max_rel_1v = 0.0
    worst = None
    per_signal = {name: 0.0 for name in SIGNALS}
    for idx, (ref, got) in enumerate(zip(ref_rows, shadow_rows)):
        for name in SIGNALS:
            diff = abs(ref[name] - got[name])
            rel = diff / max(1.0, abs(ref[name]))
            per_signal[name] = max(per_signal[name], diff)
            if rel > max_rel_1v:
                worst = (idx, ref["time"], name, ref[name], got[name], diff, rel)
            max_abs = max(max_abs, diff)
            max_rel_1v = max(max_rel_1v, rel)
    return {
        "rows_compared": min(len(ref_rows), len(shadow_rows)),
        "row_count_equal": len(ref_rows) == len(shadow_rows),
        "max_signal_abs": max_abs,
        "max_signal_rel_1v": max_rel_1v,
        "per_signal_max_abs": per_signal,
        "worst_signal": worst,
        "close_le_1pct": len(ref_rows) == len(shadow_rows) and max_rel_1v <= 0.01,
        "close_le_1e_6": len(ref_rows) == len(shadow_rows) and max_rel_1v <= 1.0e-6,
    }


def main() -> int:
    if not SCS_PATH.exists() or not VA_PATH.exists():
        raise SystemExit("missing pipeline_stage benchmark files")

    print(f"workload: {SCS_PATH.relative_to(REPO)}")
    _build_rust_core()
    ref = _run_reference()
    print("\n=== reference EVAS ===")
    print(f"ok          : {ref['ok']}")
    print(f"wall_s      : {ref['wall_s']:.6f}")
    print(f"out_dir     : {ref['out_dir']}")
    print(f"csv_exists  : {ref['csv_exists']}")
    print(f"csv_size    : {ref['csv_size']}")
    if not ref["ok"] or not ref["csv_exists"]:
        print("\nDECISION: DO_NOT_WIRE_ENGINE")
        print("reason  : reference EVAS run did not produce tran.csv")
        return 1

    _header, ref_rows = _read_csv(ref["csv_path"])
    shadow = _run_shadow_replay(ref_rows)
    shadow_csv = Path(tempfile.mkdtemp(prefix="vaevas_094p_shadow_")) / "tran.csv"
    _write_csv(shadow_csv, shadow["rows"])
    comparison = _compare_outputs(ref_rows, shadow["rows"])

    print("\n=== rust shadow replay ===")
    print(f"wall_s             : {shadow['wall_s']:.6f}")
    print(f"rows               : {len(shadow['rows'])}")
    print(f"fired_event_count  : {shadow['fired_count']}")
    print(f"default_transition : {shadow['default_transition']}")
    print(f"shadow_csv         : {shadow_csv}")
    print(f"final_state        : {shadow['final_state']}")

    print("\n=== same-grid output comparison ===")
    for key, value in comparison.items():
        print(f"{key:20}: {value}")

    if comparison["close_le_1e_6"]:
        print("\nDECISION: SHADOW_REPLAY_PASSED_BUT_DO_NOT_DIRECT_WIRE_ENGINE")
        print("reason  : 094o Rust shadow runtime matches the reference output on")
        print("          the reference time/source grid, but adaptive scheduling,")
        print("          breakpoint ownership, CSV/record, and engine.py dispatch")
        print("          are still Python-owned.")
        return 0
    if comparison["close_le_1pct"]:
        print("\nDECISION: PARITY_SMOKE_PASSED_WITH_TOLERANCE_BUT_DO_NOT_WIRE_ENGINE")
        print("reason  : output is within 1% but not tight enough for production wiring")
        return 0

    print("\nDECISION: DO_NOT_WIRE_ENGINE")
    print("reason  : Rust shadow replay does not match reference outputs")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
