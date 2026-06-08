"""Audit 094f tests for ExprIR to Rust body-op encoding."""

from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from evas.compiler.ast_nodes import BranchAccess, Identifier
from evas.compiler.parser import parse
from evas.compiler.preprocessor import preprocess
from evas.simulator.expr_ir import (
    BinaryExprIR,
    BranchAccessIR,
    LoweringContext,
    build_state_binding_ir,
    encode_body_expr_ops,
    lower_expr,
)
from evas.simulator.rust_backend import (
    BODY_STMT_WHILE,
    BODY_TARGET_STATE,
    BodyStmtOp,
    default_rust_core_library_path,
    load_rust_backend,
)
from evas.simulator.stmt_ir import (
    BodyStmtProgram,
    EventBodyProgram,
    EventStatementIR,
    encode_body_stmt_ops,
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


SAMPLE = """\
`include "disciplines.vams"
module body_encoder_sample(vin, vref, out);
    input voltage vin, vref;
    output voltage out;
    parameter real gain = -2.0;
    parameter real thresh = 0.3;
    real acc = 0.0;
    analog begin
        acc = ((V(vin, vref) * abs(gain)) > thresh) ? acc + 1.0 : 0.0;
    end
endmodule
"""


STMT_SAMPLE = """\
`include "disciplines.vams"
module stmt_encoder_sample(vin, out);
    input voltage vin;
    output voltage out;
    parameter real gain = 2.0;
    parameter real offset = 0.1;
    real acc = 0.0;
    analog begin
        acc = V(vin) * gain;
        V(out) <+ acc + offset;
    end
endmodule
"""


def _build_rust_core():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_CORE,
        check=True,
    )


def test_expr_ir_encodes_to_rust_body_ops_and_executes_state_write():
    _build_rust_core()
    module = parse(SAMPLE)
    assignment = module.analog_block.body.statements[0]
    expr_ir = lower_expr(assignment.value, LoweringContext.veriloga_body())
    assert expr_ir is not None
    bindings = build_state_binding_ir(module)
    acc_binding = bindings.resolve("acc")
    assert acc_binding is not None
    expr_ops = encode_body_expr_ops(expr_ir, bindings, {"vin": 0, "vref": 1})
    assert expr_ops is not None

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=[
            BodyStmtOp(
                target_kind=BODY_TARGET_STATE,
                target_id=acc_binding.slot,
                expr_start=0,
                expr_count=len(expr_ops),
            )
        ],
        expr_ops=expr_ops,
    )
    node_values = array("d", [0.5, 0.3])
    state_values = array("d", [3.5])
    param_values = array("d", [-2.0, 0.3])

    backend.evaluate_body_ir(batch, node_values, state_values, param_values)

    assert state_values.tolist() == pytest.approx([4.5])

    node_values = array("d", [0.4, 0.3])
    state_values = array("d", [3.5])
    backend.evaluate_body_ir(batch, node_values, state_values, param_values)

    assert state_values.tolist() == pytest.approx([0.0])


def test_expr_ir_encoder_rejects_dynamic_indexed_voltage_read():
    module = parse(SAMPLE)
    bindings = build_state_binding_ir(module)
    dynamic_expr = lower_expr(
        BranchAccess("V", "vin", node1_index=Identifier("acc")),
        LoweringContext.veriloga_body(),
    )
    assert isinstance(dynamic_expr, BranchAccessIR)

    assert encode_body_expr_ops(dynamic_expr, bindings, {"vin": 0}) is None


def test_stmt_ir_encodes_ordered_state_and_output_writes_to_rust_batch():
    _build_rust_core()
    module = parse(STMT_SAMPLE)
    stmt_ir = lower_stmt(module.analog_block.body)
    assert stmt_ir is not None
    bindings = build_state_binding_ir(module)
    program = encode_body_stmt_ops(stmt_ir, bindings, {"vin": 0, "out": 1})
    assert isinstance(program, BodyStmtProgram)
    assert len(program.stmt_ops) == 2

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=program.stmt_ops,
        expr_ops=program.expr_ops,
    )
    node_values = array("d", [0.25, 0.0])
    state_values = array("d", [0.0])
    param_values = array("d", [2.0, 0.1])

    backend.evaluate_body_ir(batch, node_values, state_values, param_values)

    assert state_values.tolist() == pytest.approx([0.5])
    assert node_values.tolist() == pytest.approx([0.25, 0.6])


def test_stmt_ir_encoder_rejects_event_body_until_scheduler_owns_ordering():
    module = parse(
        """\
`include "disciplines.vams"
module event_stmt_sample(clk);
    input voltage clk;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body)
    assert stmt_ir is not None
    bindings = build_state_binding_ir(module)

    assert encode_body_stmt_ops(stmt_ir, bindings, {"clk": 0}) is None


def test_event_body_program_encodes_cross_body_write_set_for_future_scheduler():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_body_sample(clk, out);
    input voltage clk;
    output voltage out;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) begin
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
    program = encode_event_body_program(stmt_ir, bindings, {"clk": 0, "out": 1})
    assert isinstance(program, EventBodyProgram)
    assert len(program.body_program.stmt_ops) == 2

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=program.body_program.stmt_ops,
        expr_ops=program.body_program.expr_ops,
    )
    node_values = array("d", [0.0, 0.0])
    state_values = array("d", [2.0])
    param_values = array("d")

    backend.evaluate_body_ir(batch, node_values, state_values, param_values)

    assert state_values.tolist() == pytest.approx([3.0])
    assert node_values.tolist() == pytest.approx([0.0, 3.0])


def test_event_body_program_encodes_if_else_body_to_rust_batch():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_body_if_sample(clk);
    input voltage clk;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) begin
            if (q > 0) q = 0;
            else q = 1;
        end
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    bindings = build_state_binding_ir(module)
    program = encode_event_body_program(stmt_ir, bindings, {"clk": 0})
    assert isinstance(program, EventBodyProgram)

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=program.body_program.stmt_ops,
        expr_ops=program.body_program.expr_ops,
    )
    node_values = array("d", [0.0])
    param_values = array("d")

    state_values = array("d", [2.0])
    backend.evaluate_body_ir(batch, node_values, state_values, param_values)
    assert state_values.tolist() == pytest.approx([0.0])

    state_values = array("d", [0.0])
    backend.evaluate_body_ir(batch, node_values, state_values, param_values)
    assert state_values.tolist() == pytest.approx([1.0])


def test_body_ir_while_loop_executes_with_guarded_rust_opcode():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module body_while_sample;
    real phase_err = 0.0;
    real ref_period = 1.0;
    analog begin
        while (phase_err > 0.5 * ref_period) phase_err = phase_err - ref_period;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body)
    assert stmt_ir is not None
    bindings = build_state_binding_ir(module)
    program = encode_body_stmt_ops(stmt_ir, bindings, {})
    assert isinstance(program, BodyStmtProgram)
    assert any(op.target_kind == BODY_STMT_WHILE for op in program.stmt_ops)

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=program.stmt_ops,
        expr_ops=program.expr_ops,
    )
    node_values = array("d")
    param_values = array("d")
    phase_slot = bindings.resolve("phase_err").slot
    period_slot = bindings.resolve("ref_period").slot

    state_values = array("d", [0.0, 0.0])
    state_values[phase_slot] = 3.2
    state_values[period_slot] = 1.0
    backend.evaluate_body_ir(batch, node_values, state_values, param_values)
    assert state_values[phase_slot] == pytest.approx(0.2)

    state_values[phase_slot] = 0.3
    backend.evaluate_body_ir(batch, node_values, state_values, param_values)
    assert state_values[phase_slot] == pytest.approx(0.3)


def test_pipeline_stage_phi2_if_else_and_clamp_event_body_executes_in_rust_batch():
    _build_rust_core()
    pipeline_stage_va = _pipeline_stage_va()
    source = pipeline_stage_va.read_text(encoding="utf-8")
    preprocessed_source, _defines, _default_transition = preprocess(
        source,
        source_dir=str(pipeline_stage_va.parent),
    )
    module = parse(preprocessed_source)
    stmt_ir = lower_stmt(module.analog_block.body.statements[2])
    assert isinstance(stmt_ir, EventStatementIR)

    bindings = build_state_binding_ir(module)
    node_slots = {name: idx for idx, name in enumerate(module.ports)}
    program = encode_event_body_program(stmt_ir, bindings, node_slots)
    assert isinstance(program, EventBodyProgram)

    backend = load_rust_backend(default_rust_core_library_path())
    batch = backend.make_body_ir_batch(
        stmt_ops=program.body_program.stmt_ops,
        expr_ops=program.body_program.expr_ops,
    )
    param_values = array("d", [0.45, 0.9, 200e-12])

    def run_phi2(vin_s: float) -> dict[str, float]:
        node_values = array("d", [0.9, 0.0, 0.0, 0.9, vin_s, 0.9, 0.0, 0.0, 0.0])
        state_values = array("d", [vin_s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        backend.evaluate_body_ir(batch, node_values, state_values, param_values)
        return {
            name: state_values[bindings.resolve(name).slot]
            for name in (
                "vin_s",
                "vcm",
                "vin_rel",
                "vref_qtr",
                "vres_level",
                "d1_level",
                "d0_level",
            )
        }

    upper = run_phi2(0.72)
    assert upper["d1_level"] == pytest.approx(0.9)
    assert upper["d0_level"] == pytest.approx(0.0)
    assert upper["vres_level"] == pytest.approx(0.54)

    middle = run_phi2(0.45)
    assert middle["d1_level"] == pytest.approx(0.0)
    assert middle["d0_level"] == pytest.approx(0.9)
    assert middle["vres_level"] == pytest.approx(0.45)

    lower = run_phi2(0.18)
    assert lower["d1_level"] == pytest.approx(0.0)
    assert lower["d0_level"] == pytest.approx(0.0)
    assert lower["vres_level"] == pytest.approx(0.36)

    assert run_phi2(1.2)["vres_level"] == pytest.approx(0.9)
    assert run_phi2(-0.2)["vres_level"] == pytest.approx(0.0)


def test_event_trigger_expression_uses_standalone_rust_expr_eval():
    _build_rust_core()
    module = parse(
        """\
`include "disciplines.vams"
module event_trigger_expr_sample(clk);
    input voltage clk;
    integer q = 0;
    analog begin
        @(cross(V(clk) - 0.5, +1)) q = q + 1;
    end
endmodule
"""
    )
    stmt_ir = lower_stmt(module.analog_block.body.statements[0])
    assert isinstance(stmt_ir, EventStatementIR)
    trigger_expr = stmt_ir.event.args[0]
    assert isinstance(trigger_expr, BinaryExprIR)
    bindings = build_state_binding_ir(module)
    expr_ops = encode_body_expr_ops(trigger_expr, bindings, {"clk": 0})
    assert expr_ops is not None

    backend = load_rust_backend(default_rust_core_library_path())
    value = backend.evaluate_body_expr(
        expr_ops,
        node_values=array("d", [0.7]),
        state_values=array("d", [0.0]),
        param_values=array("d"),
    )

    assert value == pytest.approx(0.2)
