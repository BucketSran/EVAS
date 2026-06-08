"""Audit 094h tests for event trigger due-program lowering."""

from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.simulator.event_due_runtime import (
    RustAnalogBlockEventRuntime,
    RustEventDueRuntime,
    RustEventStatementRuntime,
    try_build_rust_event_only_analog_block_runtime,
)
from evas.simulator.expr_ir import build_state_binding_ir
from evas.simulator.rust_backend import (
    default_rust_core_library_path,
    load_rust_backend,
)
from evas.simulator.schedule_ir import (
    EVENT_DUE_ABOVE,
    EVENT_DUE_CROSS,
    EVENT_DUE_INITIAL_STEP,
    EVENT_DUE_TIMER,
    EventDueProgram,
    encode_event_due_program,
    event_due_expr_segments,
    event_due_timer_segments,
)
from evas.simulator.stmt_ir import (
    EventStatementIR,
    encode_event_body_program,
    lower_stmt,
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


def test_event_due_program_encodes_combined_cross_above_and_batch_values():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_due_sample(clk, inp);
    input voltage clk;
    input voltage inp;
    parameter real vth = 0.5;
    parameter real vtol = 1m;
    integer q = 0;
    analog begin
        @(initial_step or cross(V(clk) - vth, +1, 1p, vtol) or above(V(inp) - 0.2))
            q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)

    program = encode_event_due_program(stmt_ir.event, bindings, {"clk": 0, "inp": 1})

    assert isinstance(program, EventDueProgram)
    assert [trigger.kind for trigger in program.triggers] == [
        EVENT_DUE_INITIAL_STEP,
        EVENT_DUE_CROSS,
        EVENT_DUE_ABOVE,
    ]
    assert program.triggers[1].direction == 1
    assert program.triggers[1].time_tol_ops
    assert program.triggers[1].expr_tol_ops

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_expr_batch(event_due_expr_segments(program))
    values = backend.evaluate_body_expr_batch(
        batch,
        node_values=array("d", [0.75, 0.1]),
        state_values=array("d", [0.0]),
        param_values=array("d", [0.5, 1.0e-3]),
    )

    assert values == pytest.approx((0.25, -0.1))


def test_event_due_program_encodes_static_timer_segments():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_due_timer_sample();
    parameter real t0 = 1n;
    parameter real per = 2n;
    integer q = 0;
    analog begin
        @(timer(t0, per)) q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)

    program = encode_event_due_program(stmt_ir.event, bindings, {})

    assert isinstance(program, EventDueProgram)
    assert [trigger.kind for trigger in program.triggers] == [EVENT_DUE_TIMER]
    assert program.triggers[0].timer_start_ops
    assert program.triggers[0].timer_period_ops

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_expr_batch(event_due_timer_segments(program))
    values = backend.evaluate_body_expr_batch(
        batch,
        node_values=array("d"),
        state_values=array("d", [0.0]),
        param_values=array("d", [1.0e-9, 2.0e-9]),
    )

    assert values == pytest.approx((1.0e-9, 2.0e-9))


def test_event_due_program_rejects_dynamic_timer_expression():
    module = parse(
        """\
`include "disciplines.vams"
module dynamic_timer_sample(clk);
    input voltage clk;
    integer q = 0;
    analog begin
        @(timer(V(clk), 1n)) q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)

    assert encode_event_due_program(stmt_ir.event, bindings, {"clk": 0}) is None


def test_event_due_runtime_returns_fired_indices_in_source_order():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module mixed_due_runtime_sample(clk, inp);
    input voltage clk;
    input voltage inp;
    integer q = 0;
    analog begin
        @(initial_step or cross(V(clk) - 0.5, +1) or above(V(inp) - 0.2) or timer(1n, 2n))
            q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)
    program = encode_event_due_program(stmt_ir.event, bindings, {"clk": 0, "inp": 1})
    assert isinstance(program, EventDueProgram)

    backend = load_rust_backend(default_rust_core_library_path())
    runtime = RustEventDueRuntime(program, backend)
    state_values = array("d", [0.0])
    param_values = array("d")

    assert runtime.step(
        time=0.0,
        node_values=array("d", [0.0, 0.0]),
        state_values=state_values,
        param_values=param_values,
        initial_step=True,
    ) == (0,)

    assert runtime.step(
        time=1.0e-9,
        node_values=array("d", [0.75, 0.1]),
        state_values=state_values,
        param_values=param_values,
    ) == (1, 3)

    assert runtime.step(
        time=2.0e-9,
        node_values=array("d", [0.75, 0.5]),
        state_values=state_values,
        param_values=param_values,
    ) == (2,)

    assert runtime.step(
        time=3.0e-9,
        node_values=array("d", [0.75, 0.5]),
        state_values=state_values,
        param_values=param_values,
    ) == (3,)


def test_event_statement_runtime_executes_body_once_for_simultaneous_triggers():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_stmt_runtime_sample(clk, out);
    input voltage clk;
    output voltage out;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1) or timer(1n, 2n)) begin
            q = q + 1;
            V(out) <+ q;
        end
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)
    node_slots = {"clk": 0, "out": 1}
    due_program = encode_event_due_program(stmt_ir.event, bindings, node_slots)
    body_program = encode_event_body_program(stmt_ir, bindings, node_slots)
    assert isinstance(due_program, EventDueProgram)
    assert body_program is not None

    backend = load_rust_backend(default_rust_core_library_path())
    runtime = RustEventStatementRuntime(
        due_program=due_program,
        body_program=body_program,
        backend=backend,
    )
    node_values = array("d", [0.0, 0.0])
    state_values = array("d", [0.0])
    param_values = array("d")

    assert runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == ()
    assert state_values.tolist() == pytest.approx([0.0])
    assert node_values.tolist() == pytest.approx([0.0, 0.0])

    node_values[0] = 0.75
    assert runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == (0, 1)
    assert state_values.tolist() == pytest.approx([1.0])
    assert node_values.tolist() == pytest.approx([0.75, 1.0])

    assert runtime.step(
        time=3.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == (1,)
    assert state_values.tolist() == pytest.approx([2.0])
    assert node_values.tolist() == pytest.approx([0.75, 2.0])


def test_analog_block_event_runtime_executes_event_statements_in_source_order():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module analog_block_runtime_sample(clk, out);
    input voltage clk;
    output voltage out;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) q = q + 1;
        @(timer(1n, 2n)) begin
            q = q + 1;
            V(out) <+ q;
        end
    end
endmodule
"""
    )
    bindings = build_state_binding_ir(module)
    node_slots = {"clk": 0, "out": 1}
    backend = load_rust_backend(default_rust_core_library_path())
    event_runtimes = []
    for stmt in module.analog_block.body.statements:
        stmt_ir = lower_stmt(stmt)
        assert isinstance(stmt_ir, EventStatementIR)
        due_program = encode_event_due_program(stmt_ir.event, bindings, node_slots)
        body_program = encode_event_body_program(stmt_ir, bindings, node_slots)
        assert isinstance(due_program, EventDueProgram)
        assert body_program is not None
        event_runtimes.append(
            RustEventStatementRuntime(
                due_program=due_program,
                body_program=body_program,
                backend=backend,
            )
        )

    runtime = RustAnalogBlockEventRuntime(tuple(event_runtimes))
    node_values = array("d", [0.0, 0.0])
    state_values = array("d", [0.0])
    param_values = array("d")

    assert runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == ()

    node_values[0] = 0.75
    assert runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == (0, 1)
    assert state_values.tolist() == pytest.approx([2.0])
    assert node_values.tolist() == pytest.approx([0.75, 2.0])


def test_pipeline_stage_event_only_runtime_executes_initial_phi1_phi2_sequence():
    _build_rust_core()
    pipeline_stage_va = _pipeline_stage_va()
    source = pipeline_stage_va.read_text(encoding="utf-8")
    preprocessed_source, _defines, _default_transition = preprocess(
        source,
        source_dir=str(pipeline_stage_va.parent),
    )
    module = parse(preprocessed_source)
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    bindings = build_state_binding_ir(module)
    backend = load_rust_backend(default_rust_core_library_path())
    runtime = try_build_rust_event_only_analog_block_runtime(
        module,
        backend,
        node_slots,
    )
    assert isinstance(runtime, RustAnalogBlockEventRuntime)

    node_values = array("d", [0.9, 0.0, 0.0, 0.0, 0.72, 0.9, 0.0, 0.0, 0.0])
    state_values = array("d", [0.0] * 7)
    param_values = array("d", [0.45, 0.9, 200e-12])

    assert runtime.step(
        time=0.0,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
        initial_step=True,
    ) == (0,)
    assert state_values[bindings.resolve("vres_level").slot] == pytest.approx(0.45)

    node_values[node_slots["PHI1"]] = 0.9
    assert runtime.step(
        time=1.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == (1,)
    assert state_values[bindings.resolve("vin_s").slot] == pytest.approx(0.72)

    node_values[node_slots["PHI2"]] = 0.9
    assert runtime.step(
        time=2.0e-9,
        node_values=node_values,
        state_values=state_values,
        param_values=param_values,
    ) == (2,)
    assert state_values[bindings.resolve("d1_level").slot] == pytest.approx(0.9)
    assert state_values[bindings.resolve("d0_level").slot] == pytest.approx(0.0)
    assert state_values[bindings.resolve("vres_level").slot] == pytest.approx(0.54)
