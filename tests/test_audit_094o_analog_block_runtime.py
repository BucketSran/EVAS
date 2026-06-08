"""Audit 094o tests for combined event + transition shadow runtime."""

from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.simulator.analog_block_runtime import (
    RustAnalogBlockShadowRuntime,
    try_build_event_then_transition_shadow_runtime,
)
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import (
    default_rust_core_library_path,
    load_rust_backend,
)

RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"
PIPELINE_STAGE_VA = (
    Path(__file__).resolve().parents[2]
    / "behavioral-veriloga-eval"
    / "benchmark-vabench-release-v1"
    / "tasks"
    / "CT01_data_converter_models"
    / "vbr1_l1_pipeline_adc_stage"
    / "forms"
    / "tb"
    / "gold"
    / "pipeline_stage.va"
)


def _pipeline_stage_va() -> Path:
    if not PIPELINE_STAGE_VA.exists():
        pytest.skip(
            "pipeline_stage fixture lives in the behavioral-veriloga-eval "
            f"sibling checkout: {PIPELINE_STAGE_VA}"
        )
    return PIPELINE_STAGE_VA


def _build_rust_core():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_CORE,
        check=True,
    )


def test_pipeline_stage_event_then_transition_runtime_updates_state_and_outputs():
    _build_rust_core()
    pipeline_stage_va = _pipeline_stage_va()
    source = pipeline_stage_va.read_text(encoding="utf-8")
    preprocessed_source, _defines, default_transition = preprocess(
        source,
        source_dir=str(pipeline_stage_va.parent),
    )
    module = parse(preprocessed_source)
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    bindings = build_state_binding_ir(module)
    backend = load_rust_backend(default_rust_core_library_path())
    runtime = try_build_event_then_transition_shadow_runtime(
        module,
        backend,
        node_slots,
        default_transition=default_transition or 0.0,
    )
    assert isinstance(runtime, RustAnalogBlockShadowRuntime)

    node_values = array("d", [0.9, 0.0, 0.0, 0.0, 0.72, 0.9, 0.0, 0.0, 0.0])
    state_values = array("d", [0.0] * 7)
    param_values = array("d", [0.45, 0.9, 200e-12])

    result = runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
        initial_step=True,
    )
    assert result.fired_event_statements == (0,)
    assert result.transition_outputs == pytest.approx((0.45, 0.0, 0.0))
    assert node_values[node_slots["VRES"]] == pytest.approx(0.45)
    assert node_values[node_slots["D1"]] == pytest.approx(0.0)
    assert node_values[node_slots["D0"]] == pytest.approx(0.0)

    node_values[node_slots["PHI1"]] = 0.9
    result = runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )
    assert result.fired_event_statements == (1,)
    assert state_values[bindings.resolve("vin_s").slot] == pytest.approx(0.72)
    assert node_values[node_slots["VRES"]] == pytest.approx(0.45)

    node_values[node_slots["PHI2"]] = 0.9
    result = runtime.step(
        time=2.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )
    assert result.fired_event_statements == (2,)
    assert state_values[bindings.resolve("vres_level").slot] == pytest.approx(0.54)
    assert state_values[bindings.resolve("d1_level").slot] == pytest.approx(0.9)
    assert node_values[node_slots["VRES"]] == pytest.approx(0.45)
    assert node_values[node_slots["D1"]] == pytest.approx(0.0)
    assert runtime.next_breakpoint(2.0e-9) == pytest.approx(2.05e-9)

    result = runtime.step(
        time=2.2e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )
    assert result.fired_event_statements == ()
    assert result.transition_outputs == pytest.approx((0.54, 0.9, 0.0))
    assert node_values[node_slots["VRES"]] == pytest.approx(0.54)
    assert node_values[node_slots["D1"]] == pytest.approx(0.9)
    assert node_values[node_slots["D0"]] == pytest.approx(0.0)
