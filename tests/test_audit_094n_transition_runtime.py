"""Audit 094n tests for continuous transition contribution runtime."""

from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import (
    default_rust_core_library_path,
    load_rust_backend,
)
from evas.simulator.stmt_ir import lower_stmt
from evas.simulator.transition_runtime import (
    RustTransitionContributionRuntime,
    try_build_rust_transition_contribution_runtime,
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


def test_pipeline_stage_transition_contribution_runtime_updates_output_nodes():
    runtime, bindings, node_slots = _build_pipeline_stage_transition_runtime()

    node_values = array("d", [0.9, 0.0, 0.0, 0.0, 0.72, 0.9, 0.0, 0.0, 0.0])
    state_values = array("d", [0.72, 0.45, 0.27, 0.225, 0.54, 0.9, 0.0])
    param_values = array("d", [0.45, 0.9, 200e-12])

    assert runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
        initial_condition_mode=True,
    ) == pytest.approx((0.54, 0.9, 0.0))
    assert node_values[node_slots["VRES"]] == pytest.approx(0.54)
    assert node_values[node_slots["D1"]] == pytest.approx(0.9)
    assert node_values[node_slots["D0"]] == pytest.approx(0.0)

    state_values[bindings.resolve("vres_level").slot] = 0.45
    state_values[bindings.resolve("d1_level").slot] = 0.0
    state_values[bindings.resolve("d0_level").slot] = 0.9

    runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )
    assert node_values[node_slots["VRES"]] == pytest.approx(0.54)
    assert node_values[node_slots["D1"]] == pytest.approx(0.9)
    assert node_values[node_slots["D0"]] == pytest.approx(0.0)

    runtime.step(
        time=1.2e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )
    assert node_values[node_slots["VRES"]] == pytest.approx(0.45)
    assert node_values[node_slots["D1"]] == pytest.approx(0.0)
    assert node_values[node_slots["D0"]] == pytest.approx(0.9)


def test_pipeline_stage_transition_runtime_exposes_next_breakpoint():
    runtime, bindings, _node_slots = _build_pipeline_stage_transition_runtime()

    node_values = array("d", [0.9, 0.0, 0.0, 0.0, 0.72, 0.9, 0.0, 0.0, 0.0])
    state_values = array("d", [0.72, 0.45, 0.27, 0.225, 0.54, 0.9, 0.0])
    param_values = array("d", [0.45, 0.9, 200e-12])

    runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
        initial_condition_mode=True,
    )
    assert runtime.next_breakpoint(0.0) is None

    state_values[bindings.resolve("vres_level").slot] = 0.45
    state_values[bindings.resolve("d1_level").slot] = 0.0
    state_values[bindings.resolve("d0_level").slot] = 0.9
    runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    )

    bp1 = runtime.next_breakpoint(1.0e-9)
    assert bp1 == pytest.approx(1.05e-9)
    bp2 = runtime.next_breakpoint(bp1)
    assert bp2 == pytest.approx(1.1e-9)
    bp3 = runtime.next_breakpoint(bp2)
    assert bp3 == pytest.approx(1.15e-9)
    bp4 = runtime.next_breakpoint(bp3)
    assert bp4 == pytest.approx(1.2e-9)
    assert runtime.next_breakpoint(1.0e-9, min_ramp_time=200e-12) == pytest.approx(
        1.2e-9
    )


def _build_pipeline_stage_transition_runtime():
    _build_rust_core()
    pipeline_stage_va = _pipeline_stage_va()
    source = pipeline_stage_va.read_text(encoding="utf-8")
    preprocessed_source, _defines, default_transition = preprocess(
        source,
        source_dir=str(pipeline_stage_va.parent),
    )
    module = parse(preprocessed_source)
    stmt_ir = lower_stmt(module.analog_block.body)
    assert stmt_ir is not None
    bindings = build_state_binding_ir(module)
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    backend = load_rust_backend(default_rust_core_library_path())
    runtime = try_build_rust_transition_contribution_runtime(
        stmt_ir,
        bindings,
        node_slots,
        backend,
        default_transition=default_transition or 0.0,
    )
    assert isinstance(runtime, RustTransitionContributionRuntime)
    return runtime, bindings, node_slots
