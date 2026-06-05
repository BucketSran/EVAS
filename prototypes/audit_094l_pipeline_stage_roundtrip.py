#!/usr/bin/env python3
"""Audit 094l smoke: IR emit -> compile -> sim on pipeline_stage.

This is intentionally a prototype-level gate.  It proves whether the 094
statement/event emitter can drive a real EVAS simulation far enough to produce
``tran.csv`` and stay close to the default Python adaptive path.  It does not
modify ``engine.py`` or the production dispatcher.

Run:
    PYTHONPATH=EVAS python3 EVAS/prototypes/audit_094l_pipeline_stage_roundtrip.py
"""

from __future__ import annotations

import csv
import math
import bisect
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from evas.netlist import runner as netlist_runner
from evas.simulator.backend import compile_module as default_compile_module
from evas.simulator.engine import Simulator
from evas.simulator.stmt_ir import (
    StatementLoweringContext,
    emit_python_statement,
    lower_stmt,
)


REPO = Path(__file__).resolve().parents[2]
SCS_PATH = (
    REPO
    / "behavioral-veriloga-eval/benchmark-vabench-release-v1/tasks"
    / "CT01_data_converter_models/vbr1_l1_pipeline_adc_stage/forms/tb/gold"
    / "tb_pipeline_stage_ref.scs"
)


def _compile_emitted_body(stmt_ir: object):
    body_lines = emit_python_statement(stmt_ir)
    source_lines = [
        "def _run_emit_094_body(self, nv, time, phase):",
        "    _rt = _Emit094Runtime(self, nv, time, phase)",
        "    _rt.reset_call_counters()",
        "    var = _rt.var",
        "    array_value = _rt.array_value",
        "    set_var = _rt.set_var",
        "    set_array = _rt.set_array",
        "    voltage = _rt.voltage",
        "    current = _rt.current",
        "    node_ref = _rt.node_ref",
        "    branch_target = _rt.branch_target",
        "    contribute = _rt.contribute",
        "    event_trigger = _rt.event_trigger",
        "    combined_event = _rt.combined_event",
        "    event_due = _rt.event_due",
        "    system_task = _rt.system_task",
        "    method_substr = _rt.method_substr",
        "    method_call = _rt.method_call",
        "    fn_transition = _rt.fn_transition",
        "    fn_cross = _rt.fn_cross",
        "    fn_last_crossing = _rt.fn_last_crossing",
        "    fn_slew = _rt.fn_slew",
        "    fn_idtmod = _rt.fn_idtmod",
        "    fn_random = _rt.fn_random",
        "    fn_dist_uniform = _rt.fn_dist_uniform",
        "    fn_rdist_normal = _rt.fn_rdist_normal",
        "    fn_fopen = _rt.fn_fopen",
        "    time_value = _rt.time_value",
        "    temperature_c = _rt.temperature_c",
    ]
    source_lines.extend(body_lines)
    source_lines.append("    return _rt.post_body()")
    source = "\n".join(source_lines) + "\n"
    namespace = {
        "_Emit094Runtime": _Emit094Runtime,
        "math": math,
        "abs": abs,
        "pow": pow,
        "min": min,
        "max": max,
        "int": int,
        "float": float,
    }
    exec(compile(source, "<audit-094l-emit-body>", "exec"), namespace)
    return namespace["_run_emit_094_body"], source


def compile_module_emit_094_roundtrip(
    module,
    default_transition: float = None,
    static_branch_fastpath_codegen: bool = False,
    indexed_state_fastpath_codegen: bool = False,
):
    """Compile a normal model class, then override its body with 094 emitted IR."""

    base_cls = default_compile_module(
        module,
        default_transition,
        static_branch_fastpath_codegen=static_branch_fastpath_codegen,
        indexed_state_fastpath_codegen=indexed_state_fastpath_codegen,
    )
    if module.analog_block is None:
        return base_cls

    stmt_ir = lower_stmt(
        module.analog_block.body,
        StatementLoweringContext.veriloga_body(),
    )
    if stmt_ir is None:
        raise RuntimeError(f"094 statement lowering rejected module {module.name}")
    emitted_fn, emitted_source = _compile_emitted_body(stmt_ir)

    class Emit094RoundTripModel(base_cls):
        _emit_094_roundtrip_enabled = True
        _emit_094_roundtrip_source = emitted_source
        _has_post_update_events = bool(getattr(base_cls, "_has_post_update_events", True))

        def _run_emit_094(self, nv, time_value: float, phase: str):
            return emitted_fn(self, nv, time_value, phase)

        def initial_step(self, nv, time_value):
            self._event_trace_audit_phase = "initial_step"
            if self._initial_step_done:
                return
            self._initial_step_done = True
            for child in self._child_models:
                child.initial_step(nv, time_value)
            self._run_emit_094(nv, time_value, "initial_step")
            if self._transition_pending_count > 0:
                self._flush_transitions(nv, time_value)

        def evaluate(self, nv, time_value):
            self._event_trace_audit_phase = "evaluate"
            self._event_time = time_value
            self._bound_step = 0.0
            if self._transition_pending_count > 0:
                self._reset_transition_pending()
            for child in self._child_models:
                child.evaluate(nv, time_value)
            self._run_emit_094(nv, time_value, "evaluate")
            if self._transition_pending_count > 0:
                self._flush_transitions(nv, time_value)
            for child in self._child_models:
                bound = child._bound_step
                if bound > 0.0 and (self._bound_step <= 0.0 or bound < self._bound_step):
                    self._bound_step = bound

        def post_update_events(self, nv, time_value):
            if not self.__class__._has_post_update_events:
                return False
            # The 094 statement emitter currently has no phase-pruned body
            # emitter.  Running the whole emitted body here would duplicate
            # continuous contributions, so this smoke only supports rows whose
            # base compiler says post-update replay is unnecessary.
            self._perf_stats["rust_event_due_shadow_errors"] += 1
            return False

        def refresh_outputs(self, nv, time_value):
            # See post_update_events(): do not replay the whole emitted body in
            # refresh until 094 gains a phase-specific emitter.
            return None

    Emit094RoundTripModel.__name__ = f"{module.name}_Emit094RoundTripModel"
    return Emit094RoundTripModel


class _Emit094Runtime:
    def __init__(self, model, nv: Dict[str, float], time_value: float, phase: str):
        self.model = model
        self.nv = nv
        self.time_value = float(time_value)
        self.temperature_c = float(getattr(model, "_temperature", 27.0))
        self.phase = phase
        self._transition_call_index = 0
        self._cross_call_index = 0
        self._last_crossing_call_index = 0
        self._slew_call_index = 0
        self._idtmod_call_index = 0
        self._event_indices = {
            "INITIAL_STEP": 0,
            "FINAL_STEP": 0,
            "CROSS": 0,
            "ABOVE": 0,
            "TIMER": 0,
        }
        self._post_event_fired = False

    def reset_call_counters(self) -> None:
        self._transition_call_index = 0
        self._cross_call_index = 0
        self._last_crossing_call_index = 0
        self._slew_call_index = 0
        self._idtmod_call_index = 0
        for key in self._event_indices:
            self._event_indices[key] = 0
        self._post_event_fired = False

    def post_body(self) -> bool:
        return self._post_event_fired

    def var(self, name: str) -> Any:
        if name in self.model.params:
            return self.model.params[name]
        if name == "$abstime" or name == "$realtime":
            return self.time_value
        if name == "$temperature":
            return self.temperature_c + 273.15
        if name == "$vt":
            return 1.380649e-23 * (self.temperature_c + 273.15) / 1.602176634e-19
        return self.model.state.get(name, 0)

    def array_value(self, name: str, idx: int) -> Any:
        return self.model._array_get(name, int(idx))

    def set_var(self, name: str, value: Any) -> None:
        if name in getattr(self.model.__class__, "_integer_state_names", ()):
            value = self.model._to_integer(value)
        self.model._state_set(name, value)

    def set_array(self, name: str, idx: int, value: Any) -> None:
        if name in getattr(self.model.__class__, "_integer_state_names", ()):
            value = self.model._to_integer(value)
        self.model._array_set(name, int(idx), value)

    def voltage(self, node: str) -> float:
        return self.model._get_voltage(str(node), self.nv)

    def current(self, *_nodes: str) -> float:
        return 0.0

    def node_ref(self, name: str, idx1: int, idx2: int = None) -> str:
        if idx2 is None:
            return self.model._resolve_dynamic_node(str(name), int(idx1))
        return self.model._resolve_dynamic_node(str(name), int(idx1), int(idx2))

    def branch_target(self, access_type: str, node1: str, node2: str = None):
        return (access_type, node1, node2)

    def contribute(self, target, value: float) -> None:
        access_type, node1, node2 = target
        if access_type != "V":
            return
        out = float(value)
        if node2 is not None:
            out += self.voltage(node2)
        self.model._set_output(str(node1), out, self.nv)

    def event_trigger(
        self,
        event_type: str,
        args: Tuple[Any, ...],
        direction: int,
        time_tol: Any,
        expr_tol: Any,
    ):
        kind = str(event_type).upper()
        index = self._event_indices.get(kind, 0)
        self._event_indices[kind] = index + 1
        return {
            "kind": kind,
            "index": index,
            "args": tuple(args),
            "direction": direction,
            "time_tol": time_tol,
            "expr_tol": expr_tol,
        }

    def combined_event(self, events: Iterable[dict]):
        return {"kind": "COMBINED", "events": tuple(events)}

    def event_due(self, spec) -> bool:
        kind = spec["kind"]
        if kind == "COMBINED":
            return any(self.event_due(child) for child in spec["events"])
        if kind == "INITIAL_STEP":
            return self.phase == "initial_step"
        if kind == "FINAL_STEP":
            return self.phase == "final_step"
        if self.phase != "evaluate":
            return False
        if kind == "CROSS":
            return self._cross_event_due(spec)
        if kind == "ABOVE":
            return self._above_event_due(spec)
        if kind == "TIMER":
            return self._timer_event_due(spec)
        return False

    def _cross_event_due(self, spec) -> bool:
        args = spec["args"]
        val = float(args[0]) if args else 0.0
        direction = int(spec["direction"] or 0)
        time_tol = 0.0 if spec["time_tol"] is None else float(spec["time_tol"])
        expr_tol = 1e-12 if spec["expr_tol"] is None else float(spec["expr_tol"])
        key = f"emit094_cross_{spec['index']}"
        fired = self.model._check_cross(
            key,
            self.time_value,
            val,
            direction,
            time_tol,
            expr_tol,
        )
        if fired:
            self._post_event_fired = True
            self.model._event_context_active = True
        return fired

    def _above_event_due(self, spec) -> bool:
        args = spec["args"]
        val = float(args[0]) if args else 0.0
        direction = 1 if spec["direction"] is None else int(spec["direction"])
        key = f"emit094_above_{spec['index']}"
        fired = self.model._check_above(key, self.time_value, val, direction)
        if fired:
            self._post_event_fired = True
            self.model._event_context_active = True
        return fired

    def _timer_event_due(self, spec) -> bool:
        args = spec["args"]
        key = f"emit094_timer_{spec['index']}"
        if len(args) >= 2:
            fired = self.model._check_timer_due(
                key,
                self.time_value,
                float(args[1]),
                float(args[0]),
            )
            if fired:
                self.model._reschedule_timer(key, self.time_value, float(args[1]))
            return fired
        target = float(args[0]) if args else 0.0
        fired = self.model._check_timer_at(key, self.time_value, target)
        if fired:
            self.model._set_timer_state(key, target)
        return fired

    def system_task(self, name: str, *args: Any) -> None:
        if name in {"$strobe", "$display"}:
            if args:
                self.model._strobe(self.time_value, args[0], *args[1:])
            else:
                self.model._strobe(self.time_value, "")
        elif name == "$bound_step" and args:
            self.model._bound_step = float(args[0])

    def method_substr(self, obj: str, start: int, end: int) -> str:
        value = str(self.model.params.get(obj, ""))
        return value[int(start) : int(end) + 1]

    def method_call(self, _obj: str, _method: str, *_args: Any) -> str:
        return ""

    def fn_transition(
        self,
        target: float,
        delay: float = 0.0,
        rise: float = 0.0,
        fall: float = None,
    ) -> float:
        if fall is None:
            fall = rise
        key = f"emit094_trans_{self._transition_call_index}"
        self._transition_call_index += 1
        return self.model._transition(
            key,
            self.time_value,
            float(target),
            float(delay),
            float(rise),
            float(fall),
        )

    def fn_cross(
        self,
        val: float,
        direction: int = 0,
        time_tol: float = 0.0,
        expr_tol: float = 1e-12,
    ) -> bool:
        key = f"emit094_cross_fn_{self._cross_call_index}"
        self._cross_call_index += 1
        return self.model._check_cross(
            key,
            self.time_value,
            float(val),
            int(direction),
            float(time_tol),
            float(expr_tol),
        )

    def fn_last_crossing(
        self,
        val: float,
        direction: int = 0,
        time_tol: float = 0.0,
        expr_tol: float = 1e-12,
    ) -> float:
        key = f"emit094_last_cross_{self._last_crossing_call_index}"
        self._last_crossing_call_index += 1
        return self.model._last_crossing(
            key,
            self.time_value,
            float(val),
            int(direction),
            float(time_tol),
            float(expr_tol),
        )

    def fn_slew(self, target: float, maxrise: float = 0.0, maxfall: float = None) -> float:
        if maxfall is None:
            maxfall = maxrise
        key = f"emit094_slew_{self._slew_call_index}"
        self._slew_call_index += 1
        return self.model._slew(
            key,
            self.time_value,
            float(target),
            float(maxrise),
            float(maxfall),
        )

    def fn_idtmod(self, x: float, ic: float = 0.0, mod: float = 1.0) -> float:
        key = f"emit094_idtmod_{self._idtmod_call_index}"
        self._idtmod_call_index += 1
        return self.model._idtmod(key, self.time_value, float(x), float(ic), float(mod))

    def fn_random(self, seed: float = None) -> int:
        return self.model._rand_int32(seed)

    def fn_dist_uniform(self, *args: float) -> float:
        if len(args) >= 3:
            return self.model._rand_uniform(args[0], args[1], args[2])
        if len(args) == 2:
            return self.model._rand_uniform(None, args[0], args[1])
        return self.model._rand_uniform(None, 0.0, args[0] if args else 1.0)

    def fn_rdist_normal(self, *args: float) -> float:
        if len(args) >= 3:
            return self.model._rand_normal(args[0], args[1], args[2])
        if len(args) == 2:
            return self.model._rand_normal(None, args[0], args[1])
        return self.model._rand_normal(None, args[0] if args else 0.0, 1.0)

    def fn_fopen(self, filename: str, mode: str = "w") -> int:
        return self.model._fopen(filename, mode)


def run_sim(label: str, use_emit_094: bool):
    out_dir = Path(tempfile.mkdtemp(prefix=f"vaevas_094l_{label}_"))
    captured = []
    orig_run = Simulator.run
    orig_compile_module = netlist_runner.compile_module

    def wrap_run(self, *args, **kwargs):
        captured.append(self)
        kwargs["rust_required"] = True
        return orig_run(self, *args, **kwargs)

    Simulator.run = wrap_run
    if use_emit_094:
        netlist_runner.compile_module = compile_module_emit_094_roundtrip
    try:
        t0 = time.perf_counter()
        ok = netlist_runner.evas_simulate(
            str(SCS_PATH),
            log_path=str(out_dir / "sim.log"),
            output_dir=str(out_dir),
        )
        wall_s = time.perf_counter() - t0
    finally:
        Simulator.run = orig_run
        netlist_runner.compile_module = orig_compile_module

    csv_path = out_dir / "tran.csv"
    return {
        "label": label,
        "ok": bool(ok),
        "wall_s": wall_s,
        "out_dir": out_dir,
        "csv_path": csv_path,
        "csv_exists": csv_path.exists(),
        "csv_size": csv_path.stat().st_size if csv_path.exists() else 0,
        "sim": captured[0] if captured else None,
    }


def _read_csv(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]
        return reader.fieldnames or [], rows


def compare_csv_rowwise(ref_path: Path, test_path: Path):
    ref_header, ref_rows = _read_csv(ref_path)
    test_header, test_rows = _read_csv(test_path)
    common = [name for name in ref_header if name in test_header]
    signals = [name for name in common if name != "time"]
    n = min(len(ref_rows), len(test_rows))
    max_time_abs = 0.0
    max_signal_abs = 0.0
    max_signal_rel_1v = 0.0
    worst_signal = None

    for idx in range(n):
        if "time" in common:
            max_time_abs = max(
                max_time_abs,
                abs(ref_rows[idx]["time"] - test_rows[idx]["time"]),
            )
        for name in signals:
            ref = ref_rows[idx][name]
            got = test_rows[idx][name]
            diff = abs(ref - got)
            rel = diff / max(1.0, abs(ref))
            if rel > max_signal_rel_1v:
                worst_signal = (idx, name, ref, got, diff, rel)
            max_signal_abs = max(max_signal_abs, diff)
            max_signal_rel_1v = max(max_signal_rel_1v, rel)

    close = (
        ref_header == test_header
        and len(ref_rows) == len(test_rows)
        and max_signal_rel_1v <= 0.01
    )
    return {
        "ref_rows": len(ref_rows),
        "test_rows": len(test_rows),
        "headers_equal": ref_header == test_header,
        "common_columns": common,
        "max_time_abs": max_time_abs,
        "max_signal_abs": max_signal_abs,
        "max_signal_rel_1v": max_signal_rel_1v,
        "worst_signal": worst_signal,
        "close_le_1pct": close,
    }


def compare_csv_time_aligned(ref_path: Path, test_path: Path):
    """Compare waveforms at reference timestamps using linear interpolation."""

    ref_header, ref_rows = _read_csv(ref_path)
    test_header, test_rows = _read_csv(test_path)
    common = [name for name in ref_header if name in test_header]
    signals = [name for name in common if name != "time"]
    test_times = [row["time"] for row in test_rows]
    max_signal_abs = 0.0
    max_signal_rel_1v = 0.0
    worst_signal = None
    max_nearest_time_delta = 0.0

    for idx, ref_row in enumerate(ref_rows):
        t = ref_row["time"]
        insert_at = bisect.bisect_left(test_times, t)
        nearest_candidates = []
        if insert_at > 0:
            nearest_candidates.append(insert_at - 1)
        if insert_at < len(test_rows):
            nearest_candidates.append(insert_at)
        if nearest_candidates:
            nearest = min(
                nearest_candidates,
                key=lambda row_index: abs(test_times[row_index] - t),
            )
            max_nearest_time_delta = max(
                max_nearest_time_delta,
                abs(test_times[nearest] - t),
            )

        for name in signals:
            got = _interpolate_signal(test_rows, test_times, insert_at, t, name)
            ref = ref_row[name]
            diff = abs(ref - got)
            rel = diff / max(1.0, abs(ref))
            if rel > max_signal_rel_1v:
                worst_signal = (idx, t, name, ref, got, diff, rel)
            max_signal_abs = max(max_signal_abs, diff)
            max_signal_rel_1v = max(max_signal_rel_1v, rel)

    return {
        "ref_rows": len(ref_rows),
        "test_rows": len(test_rows),
        "headers_equal": ref_header == test_header,
        "common_columns": common,
        "max_nearest_time_delta": max_nearest_time_delta,
        "max_signal_abs_interp": max_signal_abs,
        "max_signal_rel_1v_interp": max_signal_rel_1v,
        "worst_signal_interp": worst_signal,
        "close_le_1pct_interp": (
            ref_header == test_header
            and max_signal_rel_1v <= 0.01
        ),
    }


def _interpolate_signal(
    rows: List[Dict[str, float]],
    times: List[float],
    insert_at: int,
    t: float,
    signal: str,
) -> float:
    if not rows:
        return 0.0
    if insert_at <= 0:
        return rows[0][signal]
    if insert_at >= len(rows):
        return rows[-1][signal]
    t0 = times[insert_at - 1]
    t1 = times[insert_at]
    v0 = rows[insert_at - 1][signal]
    v1 = rows[insert_at][signal]
    if t1 == t0:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def _perf_summary(sim) -> Dict[str, Any]:
    if sim is None:
        return {}
    stats = dict(getattr(sim, "_perf_stats", {}) or {})
    keys = [
        "generic_executor_runs",
        "generic_executor_runtime_fallbacks",
        "generic_executor_models_with_candidate",
        "cross_fires",
        "transition_calls",
        "transition_output_fastpath_calls",
        "rust_event_due_shadow_errors",
    ]
    return {key: stats.get(key, 0) for key in keys}


def main() -> int:
    if not SCS_PATH.exists():
        raise SystemExit(f"missing testbench: {SCS_PATH}")

    print(f"workload: {SCS_PATH.relative_to(REPO)}")
    baseline = run_sim("python_adaptive", use_emit_094=False)
    emitted = run_sim("emit094_roundtrip", use_emit_094=True)

    for sample in (baseline, emitted):
        print(f"\n=== {sample['label']} ===")
        print(f"ok          : {sample['ok']}")
        print(f"wall_s      : {sample['wall_s']:.6f}")
        print(f"out_dir     : {sample['out_dir']}")
        print(f"csv_exists  : {sample['csv_exists']}")
        print(f"csv_size    : {sample['csv_size']}")
        print(f"perf        : {_perf_summary(sample['sim'])}")

    if not baseline["csv_exists"] or not emitted["csv_exists"]:
        print("\nDECISION: DO_NOT_WIRE_ENGINE")
        print("reason  : one side did not produce tran.csv")
        return 1

    rowwise = compare_csv_rowwise(baseline["csv_path"], emitted["csv_path"])
    aligned = compare_csv_time_aligned(baseline["csv_path"], emitted["csv_path"])
    print("\n=== Rowwise CSV comparison: Python adaptive vs emit094 round-trip ===")
    for key, value in rowwise.items():
        print(f"{key:18}: {value}")

    print("\n=== Time-aligned CSV comparison: Python adaptive vs emit094 round-trip ===")
    for key, value in aligned.items():
        print(f"{key:28}: {value}")

    if aligned["close_le_1pct_interp"]:
        print("\nDECISION: PARITY_SMOKE_PASSED_BUT_DO_NOT_DIRECT_WIRE_ENGINE")
        print("reason  : emit094 produces a close waveform after time alignment,")
        print("          but this prototype is helper-based Python and is slower than")
        print("          the default path; engine.py should wait for a guarded Rust")
        print("          batch dispatcher plus phase-specific emitted bodies.")
    else:
        print("\nDECISION: DO_NOT_WIRE_ENGINE")
        print("reason  : emit094 waveform is not within the <=1% time-aligned smoke gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
