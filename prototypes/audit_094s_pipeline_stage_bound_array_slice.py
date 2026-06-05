#!/usr/bin/env python3
"""Audit 094s smoke: bound indexed-array slice for pipeline_stage.

This prototype does not modify engine.py.  It compares two replay adapters on
the same reference EVAS time/source grid:

1. dict-pack replay, mirroring the 094q wrapper cost shape where node values
   are copied from a dict into runtime arrays and outputs are synced back;
2. bound-array replay, where module ports are lowered directly to global
   IndexedVoltageArray ids and the Rust runtime reads/writes that persistent
   array in place.

Run:
    PYTHONPATH=EVAS python3 EVAS/prototypes/audit_094s_pipeline_stage_bound_array_slice.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.netlist import runner as netlist_runner
from evas.simulator.analog_block_runtime import (
    RustAnalogBlockShadowRuntime,
    try_build_event_then_transition_shadow_runtime,
)
from evas.simulator.indexed import IndexedVoltageArray
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
PORT_TO_NODE = {
    "VDD": "vdd",
    "VSS": "vss",
    "PHI1": "phi1",
    "PHI2": "phi2",
    "VIN": "vin",
    "VREF": "vref",
    "VRES": "vres",
    "D1": "d1",
    "D0": "d0",
}
SOURCE_PORTS = ("VDD", "VSS", "PHI1", "PHI2", "VIN", "VREF")
OUTPUT_PORTS = ("VRES", "D1", "D0")
SIGNALS = ("vres", "d1", "d0")


def _build_rust_core() -> None:
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


def _run_reference() -> dict[str, Any]:
    out_dir = Path(tempfile.mkdtemp(prefix="vaevas_094s_ref_"))
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


def _build_module():
    source = VA_PATH.read_text(encoding="utf-8")
    preprocessed_source, _defines, default_transition = preprocess(
        source,
        source_dir=str(VA_PATH.parent),
    )
    return parse(preprocessed_source), default_transition or 0.0


def _build_runtime(
    node_slots: dict[str, int],
) -> RustAnalogBlockShadowRuntime:
    module, default_transition = _build_module()
    backend = load_rust_backend(default_rust_core_library_path())
    runtime = try_build_event_then_transition_shadow_runtime(
        module,
        backend,
        node_slots,
        default_transition=default_transition,
    )
    if runtime is None:
        raise RuntimeError("094s runtime rejected pipeline_stage")
    return runtime


def _row_source_values(row: dict[str, float]) -> dict[str, float]:
    return {
        "VDD": 0.9,
        "VSS": 0.0,
        "PHI1": row["phi1"],
        "PHI2": row["phi2"],
        "VIN": row["vin"],
        "VREF": 0.9,
    }


def _iter_replay_rows(
    ref_rows: list[dict[str, float]],
    repeat: int,
) -> Iterable[tuple[int, int, float, dict[str, float]]]:
    if not ref_rows:
        return
    span = float(ref_rows[-1]["time"]) + 1.0e-9
    for pass_idx in range(repeat):
        offset = pass_idx * span
        for row_idx, row in enumerate(ref_rows):
            yield pass_idx, row_idx, offset + float(row["time"]), row


def _run_dict_pack_replay(
    ref_rows: list[dict[str, float]],
    *,
    repeat: int,
) -> dict[str, Any]:
    module, _default_transition = _build_module()
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    runtime = _build_runtime(node_slots)
    node_values = array("d", [0.0] * len(module.ports))
    state_values = array("d", [0.0] * 7)
    param_values = array("d", [0.45, 0.9, 200e-12])
    nv = {name: 0.0 for name in module.ports}
    output_slots = {port: node_slots[port] for port in OUTPUT_PORTS}
    first_pass_rows: list[dict[str, float]] = []
    fired_count = 0
    breakpoint_hits = 0
    steps = 0

    t0 = time.perf_counter()
    for pass_idx, row_idx, time_value, row in _iter_replay_rows(ref_rows, repeat):
        nv.update(_row_source_values(row))
        for name, slot in node_slots.items():
            node_values[slot] = float(nv.get(name, 0.0))
        result = runtime.step(
            time=time_value,
            node_values=node_values,
            state_values=state_values,
            param_values=param_values,
            initial_step=(pass_idx == 0 and row_idx == 0),
        )
        fired_count += len(result.fired_event_statements)
        if runtime.next_breakpoint(time_value) is not None:
            breakpoint_hits += 1
        for port, slot in output_slots.items():
            nv[port] = float(node_values[slot])
        if pass_idx == 0:
            first_pass_rows.append(
                {
                    "time": row["time"],
                    "vres": nv["VRES"],
                    "d1": nv["D1"],
                    "d0": nv["D0"],
                }
            )
        steps += 1
    wall_s = time.perf_counter() - t0
    return {
        "wall_s": wall_s,
        "steps": steps,
        "first_pass_rows": first_pass_rows,
        "fired_count": fired_count,
        "breakpoint_hits": breakpoint_hits,
    }


def _run_bound_indexed_replay(
    ref_rows: list[dict[str, float]],
    *,
    repeat: int,
) -> dict[str, Any]:
    indexed = IndexedVoltageArray.from_names(PORT_TO_NODE.values())
    indexed.values = array("d", indexed.values)
    node_slots = {
        port: indexed.node_index.id_of(node)
        for port, node in PORT_TO_NODE.items()
    }
    runtime = _build_runtime(node_slots)
    state_values = array("d", [0.0] * 7)
    param_values = array("d", [0.45, 0.9, 200e-12])
    source_ids = {
        port: indexed.node_index.id_of(PORT_TO_NODE[port])
        for port in SOURCE_PORTS
    }
    output_ids = {
        port: indexed.node_index.id_of(PORT_TO_NODE[port])
        for port in OUTPUT_PORTS
    }
    first_pass_rows: list[dict[str, float]] = []
    fired_count = 0
    breakpoint_hits = 0
    steps = 0

    t0 = time.perf_counter()
    for pass_idx, row_idx, time_value, row in _iter_replay_rows(ref_rows, repeat):
        values = indexed.values
        for port, value in _row_source_values(row).items():
            values[source_ids[port]] = value
        result = runtime.step(
            time=time_value,
            node_values=values,
            state_values=state_values,
            param_values=param_values,
            initial_step=(pass_idx == 0 and row_idx == 0),
        )
        fired_count += len(result.fired_event_statements)
        if runtime.next_breakpoint(time_value) is not None:
            breakpoint_hits += 1
        if pass_idx == 0:
            first_pass_rows.append(
                {
                    "time": row["time"],
                    "vres": values[output_ids["VRES"]],
                    "d1": values[output_ids["D1"]],
                    "d0": values[output_ids["D0"]],
                }
            )
        steps += 1
    wall_s = time.perf_counter() - t0
    return {
        "wall_s": wall_s,
        "steps": steps,
        "first_pass_rows": first_pass_rows,
        "fired_count": fired_count,
        "breakpoint_hits": breakpoint_hits,
        "node_count": indexed.node_count,
    }


def _compare_outputs(
    ref_rows: list[dict[str, float]],
    got_rows: list[dict[str, float]],
) -> dict[str, Any]:
    max_abs = 0.0
    max_rel_1v = 0.0
    per_signal = {name: 0.0 for name in SIGNALS}
    worst = None
    for idx, (ref, got) in enumerate(zip(ref_rows, got_rows)):
        for name in SIGNALS:
            diff = abs(float(ref[name]) - float(got[name]))
            rel = diff / max(1.0, abs(float(ref[name])))
            per_signal[name] = max(per_signal[name], diff)
            if rel > max_rel_1v:
                worst = (idx, ref["time"], name, ref[name], got[name], diff, rel)
            max_abs = max(max_abs, diff)
            max_rel_1v = max(max_rel_1v, rel)
    return {
        "rows_compared": min(len(ref_rows), len(got_rows)),
        "row_count_equal": len(ref_rows) == len(got_rows),
        "max_signal_abs": max_abs,
        "max_signal_rel_1v": max_rel_1v,
        "per_signal_max_abs": per_signal,
        "worst_signal": worst,
        "close_le_1e_6": len(ref_rows) == len(got_rows) and max_rel_1v <= 1.0e-6,
    }


def main() -> int:
    if not SCS_PATH.exists() or not VA_PATH.exists():
        raise SystemExit("missing pipeline_stage benchmark files")

    repeat = int(os.environ.get("EVAS_094S_REPEAT", "32"))
    repeat = max(1, repeat)
    print(f"workload: {SCS_PATH.relative_to(REPO)}")
    print(f"repeat_passes: {repeat}")
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
    dict_pack = _run_dict_pack_replay(ref_rows, repeat=repeat)
    bound = _run_bound_indexed_replay(ref_rows, repeat=repeat)
    dict_cmp = _compare_outputs(ref_rows, dict_pack["first_pass_rows"])
    bound_cmp = _compare_outputs(ref_rows, bound["first_pass_rows"])
    speedup = (
        dict_pack["wall_s"] / bound["wall_s"]
        if bound["wall_s"] > 0.0
        else float("inf")
    )

    print("\n=== dict-pack replay ===")
    print(f"wall_s          : {dict_pack['wall_s']:.6f}")
    print(f"steps           : {dict_pack['steps']}")
    print(f"fired_events    : {dict_pack['fired_count']}")
    print(f"breakpoint_hits : {dict_pack['breakpoint_hits']}")

    print("\n=== bound indexed-array replay ===")
    print(f"wall_s          : {bound['wall_s']:.6f}")
    print(f"steps           : {bound['steps']}")
    print(f"node_count      : {bound['node_count']}")
    print(f"fired_events    : {bound['fired_count']}")
    print(f"breakpoint_hits : {bound['breakpoint_hits']}")
    print(f"speedup_vs_dict_pack: {speedup:.3f}x")

    print("\n=== first-pass comparison vs reference ===")
    print("dict_pack:")
    for key, value in dict_cmp.items():
        print(f"  {key:18}: {value}")
    print("bound_indexed:")
    for key, value in bound_cmp.items():
        print(f"  {key:18}: {value}")

    if bound_cmp["close_le_1e_6"]:
        print("\nDECISION: BOUND_ARRAY_SLICE_PASSED_BUT_NOT_ENGINE_DISPATCH")
        print("reason  : direct global node-id binding preserves same-grid outputs and")
        print("          isolates node dict pack/sync cost, but adaptive scheduling,")
        print("          record/CSV, and default engine dispatch are still out of scope.")
        return 0

    print("\nDECISION: DO_NOT_WIRE_ENGINE")
    print("reason  : bound indexed-array replay failed same-grid output parity")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
