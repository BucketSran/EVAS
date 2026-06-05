#!/usr/bin/env python3
"""Audit 094q smoke: opt-in full simulation through the 094o Rust wrapper.

This prototype monkey-patches netlist compilation in this process only.  It
does not modify engine.py and it does not change the default simulator path.

Run:
    PYTHONPATH=EVAS python3 EVAS/prototypes/audit_094q_pipeline_stage_fullsim_wrapper.py
"""

from __future__ import annotations

import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Tuple

from audit_094l_pipeline_stage_roundtrip import (
    compare_csv_rowwise,
    compare_csv_time_aligned,
)

from evas.netlist import runner as netlist_runner
from evas.simulator.analog_block_runtime import (
    RustAnalogBlockShadowRuntime,
    try_build_event_then_transition_shadow_runtime,
)
from evas.simulator.backend import compile_module as default_compile_module
from evas.simulator.engine import Simulator, TransitionState
from evas.simulator.expr_ir import (
    SYMBOL_PARAMETER,
    SYMBOL_STATE_SCALAR,
    build_state_binding_ir,
)
from evas.simulator.rust_backend import default_rust_core_library_path, load_rust_backend


REPO = Path(__file__).resolve().parents[2]
SCS_PATH = (
    REPO
    / "behavioral-veriloga-eval/benchmark-vabench-release-v1/tasks"
    / "CT01_data_converter_models/vbr1_l1_pipeline_adc_stage/forms/tb/gold"
    / "tb_pipeline_stage_ref.scs"
)


def compile_module_rust_094q_wrapper(
    module,
    default_transition: float = None,
    static_branch_fastpath_codegen: bool = False,
    indexed_state_fastpath_codegen: bool = False,
):
    base_cls = default_compile_module(
        module,
        default_transition,
        static_branch_fastpath_codegen=static_branch_fastpath_codegen,
        indexed_state_fastpath_codegen=indexed_state_fastpath_codegen,
    )
    if module.analog_block is None:
        return base_cls

    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    bindings = build_state_binding_ir(module)
    state_bindings = tuple(
        binding
        for binding in bindings.bindings
        if binding.kind == SYMBOL_STATE_SCALAR
    )
    param_bindings = tuple(
        binding
        for binding in bindings.bindings
        if binding.kind == SYMBOL_PARAMETER
    )
    output_names_by_slot = tuple(module.ports)

    class Rust094QWrapperModel(base_cls):
        _rust_094q_wrapper_enabled = True
        _has_post_update_events = False

        def __init__(self):
            super().__init__()
            backend = load_rust_backend(default_rust_core_library_path())
            runtime = try_build_event_then_transition_shadow_runtime(
                module,
                backend,
                node_slots,
                default_transition=default_transition or self.default_transition,
            )
            if runtime is None:
                raise RuntimeError(f"094q runtime rejected module {module.name}")
            self._rust094q_runtime = runtime
            self._rust094q_node_values = array("d", [0.0] * len(module.ports))
            self._rust094q_state_values = array("d", [0.0] * len(state_bindings))
            self._rust094q_param_values = array("d", [0.0] * len(param_bindings))
            transition_runtime = runtime.transition_runtime
            self._rust094q_output_slots = tuple(
                transition_runtime.program.output_node_slots
            )
            self._rust094q_transition_keys = tuple(
                f"rust094q_transition_{idx}"
                for idx in range(len(self._rust094q_output_slots))
            )
            self._rust094q_steps = 0
            self._rust094q_fired_events = 0

        def _rust094q_sync_inputs(self, nv) -> None:
            for name, slot in node_slots.items():
                self._rust094q_node_values[slot] = self._get_voltage(name, nv)
            for binding in state_bindings:
                self._rust094q_state_values[binding.slot] = float(
                    self.state.get(binding.name, 0.0)
                )
            for binding in param_bindings:
                self._rust094q_param_values[binding.slot] = float(
                    self.params.get(binding.name, 0.0)
                )

        def _rust094q_sync_state_and_outputs(self, nv) -> None:
            for binding in state_bindings:
                value = float(self._rust094q_state_values[binding.slot])
                if binding.name in getattr(self.__class__, "_integer_state_names", ()):
                    value = self._to_integer(value)
                self._state_set(binding.name, value)
            for slot in self._rust094q_output_slots:
                self._set_output(
                    output_names_by_slot[slot],
                    float(self._rust094q_node_values[slot]),
                    nv,
                )
            self._rust094q_sync_transition_states()

        def _rust094q_sync_transition_states(self) -> None:
            rt = self._rust094q_runtime.transition_runtime
            for idx, key in enumerate(self._rust094q_transition_keys):
                ts = self.transitions.get(key)
                if ts is None:
                    ts = TransitionState()
                    self.transitions[key] = ts
                was_active = bool(ts.active)
                ts.current_val = float(rt.current_values[idx])
                ts.target_val = float(rt.target_values[idx])
                ts.start_time = float(rt.start_times[idx])
                ts.start_val = float(rt.start_values[idx])
                ts.delay = float(rt.delays[idx])
                ts.rise_time = float(rt.rise_times[idx])
                ts.fall_time = float(rt.fall_times[idx])
                ts.active = bool(rt.active_flags[idx])
                if ts.active:
                    self._track_transition_active_change(key, was_active, True)
                else:
                    self._mark_transition_known_inactive(key, was_active)

        def _rust094q_step(self, nv, time_value: float, *, initial_step: bool) -> bool:
            self._rust094q_sync_inputs(nv)
            result = self._rust094q_runtime.step(
                time=float(time_value),
                node_values=self._rust094q_node_values,
                state_values=self._rust094q_state_values,
                param_values=self._rust094q_param_values,
                initial_step=initial_step,
            )
            self._rust094q_steps += 1
            self._rust094q_fired_events += len(result.fired_event_statements)
            self._rust094q_sync_state_and_outputs(nv)
            return bool(result.fired_event_statements)

        def initial_step(self, nv, time_value):
            self._event_trace_audit_phase = "initial_step"
            if self._initial_step_done:
                return
            self._initial_step_done = True
            for child in self._child_models:
                child.initial_step(nv, time_value)
            self._rust094q_step(nv, time_value, initial_step=True)

        def evaluate(self, nv, time_value):
            self._event_trace_audit_phase = "evaluate"
            self._event_time = time_value
            self._bound_step = 0.0
            for child in self._child_models:
                child.evaluate(nv, time_value)
            fired = self._rust094q_step(nv, time_value, initial_step=False)
            self._step_event_fired = bool(fired)
            for child in self._child_models:
                bound = child._bound_step
                if bound > 0.0 and (self._bound_step <= 0.0 or bound < self._bound_step):
                    self._bound_step = bound

        def post_update_events(self, nv, time_value):
            return False

        def refresh_outputs(self, nv, time_value):
            return None

    Rust094QWrapperModel.__name__ = f"{module.name}_Rust094QWrapperModel"
    return Rust094QWrapperModel


def run_sim(label: str, use_rust_094q: bool) -> Dict[str, Any]:
    out_dir = Path(tempfile.mkdtemp(prefix=f"vaevas_094q_{label}_"))
    captured = []
    orig_run = Simulator.run
    orig_compile_module = netlist_runner.compile_module

    def wrap_run(self, *args, **kwargs):
        captured.append(self)
        kwargs["rust_required"] = True
        return orig_run(self, *args, **kwargs)

    Simulator.run = wrap_run
    if use_rust_094q:
        netlist_runner.compile_module = compile_module_rust_094q_wrapper
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


def _rust094q_perf(sim) -> Dict[str, Any]:
    if sim is None:
        return {}
    models = getattr(sim, "models", ()) or ()
    result = {
        "wrapper_models": 0,
        "wrapper_steps": 0,
        "wrapper_fired_events": 0,
    }
    for model in models:
        if getattr(model.__class__, "_rust_094q_wrapper_enabled", False):
            result["wrapper_models"] += 1
            result["wrapper_steps"] += int(getattr(model, "_rust094q_steps", 0))
            result["wrapper_fired_events"] += int(
                getattr(model, "_rust094q_fired_events", 0)
            )
    return result


def main() -> int:
    if not SCS_PATH.exists():
        raise SystemExit(f"missing testbench: {SCS_PATH}")

    print(f"workload: {SCS_PATH.relative_to(REPO)}")
    baseline = run_sim("python_adaptive", use_rust_094q=False)
    rust = run_sim("rust094q_wrapper", use_rust_094q=True)

    for sample in (baseline, rust):
        print(f"\n=== {sample['label']} ===")
        print(f"ok          : {sample['ok']}")
        print(f"wall_s      : {sample['wall_s']:.6f}")
        print(f"out_dir     : {sample['out_dir']}")
        print(f"csv_exists  : {sample['csv_exists']}")
        print(f"csv_size    : {sample['csv_size']}")
        print(f"rust094q    : {_rust094q_perf(sample['sim'])}")

    if not baseline["csv_exists"] or not rust["csv_exists"]:
        print("\nDECISION: DO_NOT_WIRE_ENGINE")
        print("reason  : one side did not produce tran.csv")
        return 1

    rowwise = compare_csv_rowwise(baseline["csv_path"], rust["csv_path"])
    aligned = compare_csv_time_aligned(baseline["csv_path"], rust["csv_path"])
    print("\n=== Rowwise CSV comparison: Python adaptive vs rust094q wrapper ===")
    for key, value in rowwise.items():
        print(f"{key:18}: {value}")

    print("\n=== Time-aligned CSV comparison: Python adaptive vs rust094q wrapper ===")
    for key, value in aligned.items():
        print(f"{key:28}: {value}")

    if rowwise["close_le_1pct"]:
        print("\nDECISION: FULLSIM_WRAPPER_PASSED_BUT_KEEP_OPT_IN")
        print("reason  : 094q wrapper matches the current EVAS path on this row,")
        print("          but this is a prototype monkey patch and not a release-wide")
        print("          engine dispatch/sweep.")
        return 0
    if aligned["close_le_1pct_interp"]:
        print("\nDECISION: FULLSIM_WRAPPER_TIME_ALIGNED_PASS_BUT_DO_NOT_DIRECT_WIRE")
        print("reason  : waveform is close only after time alignment; scheduler or")
        print("          breakpoint ownership still differs.")
        return 0

    print("\nDECISION: DO_NOT_WIRE_ENGINE")
    print("reason  : 094q full-sim wrapper waveform is outside the smoke gate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
