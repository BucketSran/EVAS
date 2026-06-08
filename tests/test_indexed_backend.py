"""Tests for indexed simulator migration helpers."""

import numpy as np
import pytest

from evas.simulator.indexed import (
    DynamicBranchAccessIO,
    IndexedVoltages,
    IndexedStateArrayLayout,
    IndexedVoltageArray,
    IndexedVoltageSnapshotter,
    NodeIndex,
    StateIndex,
    build_indexed_model_io_plan,
    build_indexed_run_plan,
    build_node_index,
    check_indexed_trace_round_trip,
    copy_values_into,
)
from evas.simulator.evaluate_ir import (
    COND_GE,
    COND_GT,
    COND_LE,
    COND_NE,
    SOURCE_NODE,
    SOURCE_STATE,
    TARGET_NODE,
    TARGET_STATE,
    evaluate_linear_python,
    evaluate_transition_targets_python,
    normalize_linear_ops,
    normalize_transition_target_ops,
)
from evas.compiler.parser import parse
from evas.simulator.backend import CompiledModel, compile_module
from evas.simulator.engine import SimResult, Simulator, dc


def test_node_index_assigns_stable_ids_and_names():
    index = NodeIndex()

    assert index.intern("vin") == 0
    assert index.intern("vout") == 1
    assert index.intern("vin") == 0
    assert index.id_of("vout") == 1
    assert index.name_of(0) == "vin"
    assert index.names == ("vin", "vout")


def test_node_index_rejects_empty_names_and_unknown_lookup():
    index = NodeIndex()

    with pytest.raises(ValueError, match="non-empty"):
        index.intern("")
    with pytest.raises(KeyError, match="unknown node"):
        index.id_of("missing")
    with pytest.raises(IndexError, match="unknown node id"):
        index.name_of(0)


def test_build_node_index_preserves_first_seen_order_and_deduplicates():
    index = build_node_index(["vin", "clk"], ["clk", "vout"])

    assert index.names == ("vin", "clk", "vout")
    assert [index.id_of(name) for name in index.names] == [0, 1, 2]


def test_indexed_voltages_round_trip_mapping_and_snapshot():
    index = build_node_index(["vin", "vout", "clk"])
    voltages = IndexedVoltages.from_mapping(index, {"vin": 0.25, "clk": 0.9})

    assert voltages.get("vin") == pytest.approx(0.25)
    assert voltages.get("vout") == pytest.approx(0.0)
    assert voltages.get("clk") == pytest.approx(0.9)

    snapshot = voltages.snapshot()
    voltages.set("vout", 0.7)
    assert voltages.to_mapping() == pytest.approx({"vin": 0.25, "vout": 0.7, "clk": 0.9})

    voltages.restore(snapshot)
    assert voltages.to_mapping() == pytest.approx({"vin": 0.25, "vout": 0.0, "clk": 0.9})


def test_indexed_voltages_restore_validates_size():
    index = build_node_index(["a", "b"])
    voltages = IndexedVoltages.zeros(index)

    with pytest.raises(ValueError, match="snapshot length"):
        voltages.restore([1.0])


def test_state_index_tracks_model_state_names():
    states = StateIndex()

    assert states.intern("phase") == 0
    assert states.intern("last_clk") == 1
    assert states.intern("phase") == 0
    assert states.id_of("last_clk") == 1
    assert states.names == ("phase", "last_clk")


def test_copy_values_into_validates_lengths_and_copies_float_values():
    target = [0.0, 0.0]
    copy_values_into(target, [1, 2.5])

    assert target == pytest.approx([1.0, 2.5])
    with pytest.raises(ValueError, match="target length"):
        copy_values_into(target, [1.0])


def test_indexed_voltage_snapshotter_tracks_mapping_updates_and_diffs():
    snapshotter = IndexedVoltageSnapshotter.from_names(["vin", "vout"])

    first = snapshotter.snapshot_from_mapping({"vin": 0.25, "vout": 0.75})
    assert first == pytest.approx([0.25, 0.75])
    diff, node, checked = snapshotter.max_abs_diff(first, {"vin": 0.25, "vout": 0.75})
    assert diff == pytest.approx(0.0)
    assert node == ""
    assert checked == 2

    second = snapshotter.snapshot_from_mapping({"vin": 0.5, "vout": 0.9, "clk": 1.0})
    assert snapshotter.node_index.names == ("vin", "vout", "clk")
    assert snapshotter.dynamic_interns == 1
    assert second == pytest.approx([0.5, 0.9, 1.0])

    diff, node, checked = snapshotter.max_abs_diff([0.4, 0.9, 1.0], {"vin": 0.5, "clk": 1.0})
    assert diff == pytest.approx(0.1)
    assert node == "vin"
    assert checked == 2


def test_indexed_voltage_array_grows_and_reads_previous_snapshots():
    array = IndexedVoltageArray.from_names(["vin", "vout"])
    array.update_from_mapping({"vin": 0.25, "vout": 0.75})

    previous = array.snapshot()
    array.set("vout", 0.9)
    array.set("clk", 1.0)

    assert array.node_index.names == ("vin", "vout", "clk")
    assert array.dynamic_interns == 1
    assert array.get("vout") == pytest.approx(0.9)
    assert array.get("missing", -1.0) == pytest.approx(-1.0)
    assert array.get_from_snapshot(previous, "vout", 0.0) == pytest.approx(0.75)
    assert array.get_from_snapshot(previous, "clk", -2.0) == pytest.approx(-2.0)

    diff, node, checked = array.max_abs_diff_mapping({"vin": 0.25, "vout": 0.9, "clk": 1.0})
    assert diff == pytest.approx(0.0)
    assert node == ""
    assert checked == 3
    assert array.to_mapping() == pytest.approx({"vin": 0.25, "vout": 0.9, "clk": 1.0})


def test_indexed_voltage_array_reads_batch_by_node_ids_without_name_lookup():
    values = IndexedVoltageArray.from_names(["vin", "vout"])
    values.set("vin", 0.25)
    values.set("vout", 0.75)

    assert values.values_for_ids(
        [values.node_index.id_of("vout"), values.node_index.id_of("vin"), 99],
        default=-1.0,
    ) == pytest.approx([0.75, 0.25, -1.0])


def test_indexed_voltage_array_checks_named_dirty_subset():
    array = IndexedVoltageArray.from_names(["vin", "vout", "clk"])
    array.update_from_mapping({"vin": 0.25, "vout": 0.75, "clk": 1.0})

    array.values[array.node_index.id_of("vout")] = 0.8

    diff, node, checked = array.max_abs_diff_names(
        {"vin": 0.25, "vout": 0.75, "clk": 1.0},
        ("vin", "clk"),
    )
    assert diff == pytest.approx(0.0)
    assert node == ""
    assert checked == 2

    diff, node, checked = array.max_abs_diff_names(
        {"vin": 0.25, "vout": 0.75, "clk": 1.0},
        ("vout",),
    )
    assert diff == pytest.approx(0.05)
    assert node == "vout"
    assert checked == 1


def test_indexed_model_io_plan_resolves_mapped_ports_outputs_and_parent_nodes():
    parent = CompiledModel()
    parent.node_map = {"inp": "VIN", "out": "VOUT"}
    parent.output_nodes = {"out": 0.0}
    parent._output_nodes_version = 1

    child = CompiledModel()
    child.node_map = {"a": "@parent:inp", "z": "@parent:out"}
    child.output_nodes = {"z": 0.0}
    child._output_nodes_version = 1
    child._parent_model = parent
    parent._child_models = [child]

    sim = Simulator()
    sim.add_model(parent)

    plan = build_indexed_model_io_plan(sim, extra_nodes=["monitor"])

    assert plan.node_index.names == ("monitor", "VIN", "VOUT")
    assert plan.model_count == 2
    assert plan.mapped_port_count == 4
    assert plan.output_count == 2

    root_io, child_io = plan.model_ios
    assert root_io.model_path == (0,)
    assert child_io.model_path == (0, 0)
    assert root_io.mapped_port_node_ids == (
        plan.node_index.id_of("VIN"),
        plan.node_index.id_of("VOUT"),
    )
    assert child_io.mapped_port_node_ids == (
        plan.node_index.id_of("VIN"),
        plan.node_index.id_of("VOUT"),
    )
    assert child_io.output_node_ids == (plan.node_index.id_of("VOUT"),)


def test_compiled_model_records_static_branch_io_metadata():
    src = """\
`include "disciplines.vams"
module sample_hold(clk, inp, out);
    input voltage clk;
    input voltage inp;
    output voltage out;
    real sample;
    analog begin
        @(cross(V(clk) - 0.5, +1)) sample = V(inp);
        V(out) <+ sample;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._static_voltage_read_nodes == ("clk",)
    assert ModelCls._event_trigger_voltage_read_nodes == ("clk",)
    assert ModelCls._event_voltage_read_nodes == ("inp",)
    assert ModelCls._event_body_voltage_read_nodes == ("inp",)
    assert ModelCls._static_output_write_nodes == ("out",)
    assert ModelCls._dynamic_voltage_read_count == 0
    assert ModelCls._dynamic_output_write_count == 0


def test_compiled_model_counts_dynamic_branch_io_metadata():
    src = """\
`include "disciplines.vams"
module bus_drive(VSS);
    inout electrical VSS;
    electrical [0:3] dout;
    genvar i;
    analog begin
        for (i = 0; i <= 3; i = i + 1)
            V(dout[i], VSS) <+ i;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    FastModelCls = compile_module(parse(src), static_branch_fastpath_codegen=True)

    assert ModelCls._static_voltage_read_nodes == ("VSS",)
    assert ModelCls._event_voltage_read_nodes == ()
    assert ModelCls._static_output_write_nodes == ()
    assert ModelCls._dynamic_voltage_read_count == 0
    assert ModelCls._dynamic_output_write_count == 1
    assert ModelCls._dynamic_branch_accesses == (
        ("output_write", "dout", 1, "ordinary"),
    )
    assert "self._resolve_dynamic_node('dout'" in FastModelCls._generated_code
    assert "self._format_dynamic_node('dout'" not in FastModelCls._generated_code
    assert "_set_static_branch_output('dout'" not in FastModelCls._generated_code
    assert "_set_static_branch_output_by_slot" not in FastModelCls._generated_code


def test_compiled_model_records_rust_static_affine_ops_for_literal_linear_model():
    src = """\
`include "disciplines.vams"
module gain(vin, vout);
    input voltage vin;
    output voltage vout;
    analog begin
        V(vout) <+ 2.0 * V(vin) + 0.125;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._rust_static_affine_ops == (
        ("vin", "vout", 2.0, 0.125),
    )


def test_compiled_model_records_rust_static_affine_ops_for_parameterized_linear_model():
    src = """\
`include "disciplines.vams"
module gain_param(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    parameter real offset = 0.125;
    analog begin
        V(vout) <+ gain * V(vin) + offset;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._rust_static_affine_ops == (
        ("vin", "vout", ("param", "gain"), ("param", "offset")),
    )


def test_compiled_model_records_rust_static_affine_ops_for_parameter_expression_model():
    src = """\
`include "disciplines.vams"
module gain_param_expr(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real gain_num = 6.0;
    parameter real gain_den = 2.0;
    parameter real offset_hi = 0.375;
    parameter real offset_lo = 0.125;
    analog begin
        V(vout) <+ (gain_num / gain_den) * V(vin) + (offset_hi - offset_lo);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    _, _, gain, bias = ModelCls._rust_static_affine_ops[0]

    assert ModelCls._rust_static_affine_ops[0][0:2] == ("vin", "vout")
    assert model._evaluate_rust_static_affine_scalar(gain, model.params) == pytest.approx(3.0)
    assert model._evaluate_rust_static_affine_scalar(bias, model.params) == pytest.approx(0.25)


def test_compiled_model_records_static_linear_evaluate_ir_for_differential_model():
    src = """\
`include "disciplines.vams"
module diff_gain(vip, vin, vout);
    input voltage vip;
    input voltage vin;
    output voltage vout;
    parameter real gain = 2.0;
    parameter real offset = 0.125;
    analog begin
        V(vout) <+ gain * V(vip, vin) + offset;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert len(ops) == 1
    assert ops[0].target_kind == TARGET_NODE
    assert ops[0].target_name == "vout"
    assert len(ops[0].terms) == 2
    assert [(term.source_kind, term.source_name) for term in ops[0].terms] == [
        (SOURCE_NODE, "vip"),
        (SOURCE_NODE, "vin"),
    ]

    model = ModelCls()
    node_values = [0.75, 0.25, 0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=[],
        node_ids={"vip": 0, "vin": 1, "vout": 2},
        state_ids={},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )

    assert node_values[2] == pytest.approx(1.125)


def test_static_linear_evaluate_ir_executes_state_assignment_then_output():
    src = """\
`include "disciplines.vams"
module state_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    real sample = 0.0;
    analog begin
        sample = 2.0 * V(vin) + 0.1;
        V(vout) <+ sample + 0.2;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert [(op.target_kind, op.target_name) for op in ops] == [
        (TARGET_STATE, "sample"),
        (TARGET_NODE, "vout"),
    ]
    assert ops[1].terms[0].source_kind == SOURCE_STATE

    model = ModelCls()
    node_values = [0.75, 0.0]
    state_values = [0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids={"vin": 0, "vout": 1},
        state_ids={"sample": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )

    assert state_values[0] == pytest.approx(1.6)
    assert node_values[1] == pytest.approx(1.8)


def test_static_linear_evaluate_ir_coerces_integer_state_before_output():
    src = """\
`include "disciplines.vams"
module integer_state_linear(vin, vout);
    input voltage vin;
    output voltage vout;
    integer code = 0;
    analog begin
        code = 1.6 + V(vin);
        V(vout) <+ code;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert [(op.target_kind, op.target_name, op.target_integer) for op in ops] == [
        (TARGET_STATE, "code", True),
        (TARGET_NODE, "vout", False),
    ]
    assert ops[1].terms[0].source_kind == SOURCE_STATE

    model = ModelCls()
    node_values = [0.2, 0.0]
    state_values = [0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids={"vin": 0, "vout": 1},
        state_ids={"code": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )

    assert state_values[0] == pytest.approx(2.0)
    assert node_values[1] == pytest.approx(2.0)


def test_transition_target_ir_records_integer_truthy_target():
    src = """\
`include "disciplines.vams"
module transition_target_meta(vin, vout);
    input voltage vin;
    output voltage vout;
    integer q = 0;
    analog begin
        q = V(vin) > 0.5 ? 1 : 0;
        V(vout) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 1n);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = ModelCls._transition_target_ir_ops

    assert len(ops) == 1
    (
        node,
        node2,
        transition_key,
        bias,
        terms,
        condition,
        false_bias,
        false_terms,
        delay,
        rise,
        fall,
    ) = ops[0]

    assert node == "vout"
    assert node2 is None
    assert transition_key == "trans_0"
    assert bias == pytest.approx(1.0)
    assert terms == ()
    assert condition == (
        COND_NE,
        0.0,
        ((SOURCE_STATE, "q", 1.0),),
        0.0,
        (),
    )
    assert false_bias == pytest.approx(0.0)
    assert false_terms == ()
    assert delay == pytest.approx(0.0)
    assert rise == pytest.approx(1.0e-9)
    assert fall == pytest.approx(1.0e-9)


def test_ordered_transition_segment_ir_records_state_write_before_target():
    src = """\
`include "disciplines.vams"
module ordered_transition_segment_meta(vin, vout);
    input voltage vin;
    output voltage vout;
    integer q = 0;
    analog begin
        q = V(vin) > 0.5 ? 1 : 0;
        V(vout) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    raw_linear, raw_transition = ModelCls._ordered_transition_segment_ir_ops

    assert len(raw_linear) == 1
    assert len(raw_transition) == 1
    linear_ops = normalize_linear_ops(raw_linear)
    transition_ops = normalize_transition_target_ops(raw_transition)

    assert linear_ops[0].target_kind == TARGET_STATE
    assert linear_ops[0].target_name == "q"
    assert linear_ops[0].target_integer is True
    assert transition_ops[0].transition_key == "trans_0"
    assert transition_ops[0].condition is not None
    assert transition_ops[0].rise == pytest.approx(1.0e-9)
    assert transition_ops[0].fall == pytest.approx(2.0e-9)


def test_transition_target_ir_python_array_executor_matches_truthy_state():
    src = """\
`include "disciplines.vams"
module transition_target_meta(vin, vout);
    input voltage vin;
    output voltage vout;
    integer q = 0;
    analog begin
        q = V(vin) > 0.5 ? 1 : 0;
        V(vout) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_transition_target_ops(ModelCls._transition_target_ir_ops)
    model = ModelCls()
    target_values = [0.0]
    delay_values = [0.0]
    rise_values = [0.0]
    fall_values = [0.0]

    evaluate_transition_targets_python(
        ops,
        node_values=[0.0],
        state_values=[1.0],
        target_values=target_values,
        delay_values=delay_values,
        rise_values=rise_values,
        fall_values=fall_values,
        node_ids={"vin": 0},
        state_ids={"q": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert target_values[0] == pytest.approx(1.0)
    assert delay_values[0] == pytest.approx(0.0)
    assert rise_values[0] == pytest.approx(1.0e-9)
    assert fall_values[0] == pytest.approx(2.0e-9)

    evaluate_transition_targets_python(
        ops,
        node_values=[0.0],
        state_values=[0.0],
        target_values=target_values,
        delay_values=delay_values,
        rise_values=rise_values,
        fall_values=fall_values,
        node_ids={"vin": 0},
        state_ids={"q": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert target_values[0] == pytest.approx(0.0)


def test_static_linear_evaluate_ir_ignores_initial_step_event_body():
    src = """\
`include "disciplines.vams"
module init_plus_linear(vip, vin, voutp, voutn);
    input voltage vip;
    input voltage vin;
    output voltage voutp;
    output voltage voutn;
    parameter real gain = 8.64;
    real diff;
    analog begin
        @(initial_step)
            $strobe("init only");
        diff = gain * V(vip, vin);
        V(voutp) <+ 0.45 + diff * 0.5;
        V(voutn) <+ 0.45 - diff * 0.5;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert [(op.target_kind, op.target_name) for op in ops] == [
        (TARGET_STATE, "diff"),
        (TARGET_NODE, "voutp"),
        (TARGET_NODE, "voutn"),
    ]

    model = ModelCls()
    node_values = [0.6, 0.4, 0.0, 0.0]
    state_values = [0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids={"vip": 0, "vin": 1, "voutp": 2, "voutn": 3},
        state_ids={"diff": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )

    assert state_values[0] == pytest.approx(1.728)
    assert node_values[2] == pytest.approx(1.314)
    assert node_values[3] == pytest.approx(-0.414)


def test_static_linear_evaluate_ir_lowers_top_level_ternary_state_assignment():
    src = """\
`include "disciplines.vams"
module conditional_state(dpn, vout);
    input voltage dpn;
    output voltage vout;
    parameter real vth = 0.45;
    parameter real amp = 0.014;
    real dither;
    analog begin
        dither = (V(dpn) > vth) ? amp : -amp;
        V(vout) <+ 0.5 + dither;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert len(ops) == 2
    assert ops[0].target_kind == TARGET_STATE
    assert ops[0].condition is not None
    assert ops[0].condition.op_kind == COND_GT
    assert ops[0].condition.left_terms[0].source_name == "dpn"
    assert ops[1].terms[0].source_kind == SOURCE_STATE

    model = ModelCls()
    node_values = [0.8, 0.0]
    state_values = [0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids={"dpn": 0, "vout": 1},
        state_ids={"dither": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert state_values[0] == pytest.approx(0.014)
    assert node_values[1] == pytest.approx(0.514)

    node_values = [0.2, 0.0]
    state_values = [0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids={"dpn": 0, "vout": 1},
        state_ids={"dither": 0},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert state_values[0] == pytest.approx(-0.014)
    assert node_values[1] == pytest.approx(0.486)


def test_static_linear_evaluate_ir_lowers_ternary_differential_contribution():
    src = """\
`include "disciplines.vams"
module conditional_diff(sel, outp, outn);
    input voltage sel;
    output voltage outp, outn;
    parameter real vth = 0.45;
    parameter real amp = 0.5;
    analog begin
        V(outp, outn) <+ (V(sel) > vth) ? amp : -amp;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert len(ops) == 1
    assert ops[0].condition is not None
    assert [term.source_name for term in ops[0].terms] == ["outn"]
    assert [term.source_name for term in ops[0].false_terms] == ["outn"]

    model = ModelCls()
    node_ids = {"sel": 0, "outp": 1, "outn": 2}
    node_values = [0.8, 0.0, 0.3]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=[],
        node_ids=node_ids,
        state_ids={},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(0.8)

    node_values = [0.2, 0.0, 0.3]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=[],
        node_ids=node_ids,
        state_ids={},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(-0.2)


def test_static_linear_evaluate_ir_lowers_abs_as_conditional_select():
    src = """\
`include "disciplines.vams"
module rectifier(vin, vout);
    input voltage vin;
    output voltage vout;
    parameter real vcm = 0.45;
    analog begin
        V(vout) <+ vcm + abs(V(vin) - vcm);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert len(ops) == 1
    assert ops[0].condition is not None
    assert ops[0].condition.op_kind == COND_GE

    model = ModelCls()
    node_ids = {"vin": 0, "vout": 1}
    node_values = [0.25, 0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=[],
        node_ids=node_ids,
        state_ids={},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(0.65)

    node_values = [0.8, 0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=[],
        node_ids=node_ids,
        state_ids={},
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(0.8)


def test_static_linear_evaluate_ir_lowers_min_max_state_pipeline():
    src = """\
`include "disciplines.vams"
module limiter(vin, vout);
    input voltage vin;
    output voltage vout;
    real hi;
    real clip;
    analog begin
        hi = min(V(vin), 0.8);
        clip = max(hi, 0.2);
        V(vout) <+ clip;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    ops = normalize_linear_ops(ModelCls._evaluate_ir_static_linear_ops)

    assert len(ops) == 3
    assert ops[0].condition is not None
    assert ops[0].condition.op_kind == COND_LE
    assert ops[1].condition is not None
    assert ops[1].condition.op_kind == COND_GE

    model = ModelCls()
    node_ids = {"vin": 0, "vout": 1}
    state_ids = {"hi": 0, "clip": 1}
    node_values = [0.1, 0.0]
    state_values = [0.0, 0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids=node_ids,
        state_ids=state_ids,
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(0.2)

    node_values = [0.9, 0.0]
    state_values = [0.0, 0.0]
    evaluate_linear_python(
        ops,
        node_values=node_values,
        state_values=state_values,
        node_ids=node_ids,
        state_ids=state_ids,
        params=model.params,
        scalar_eval=model._evaluate_rust_static_affine_scalar,
    )
    assert node_values[1] == pytest.approx(0.8)


def test_static_linear_evaluate_ir_rejects_self_referential_state_update():
    src = """\
`include "disciplines.vams"
module accum(vout);
    output voltage vout;
    real sample = 0.0;
    analog begin
        sample = sample + 0.1;
        V(vout) <+ sample;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._evaluate_ir_static_linear_ops == ()


def test_compiled_model_rejects_rust_static_affine_ops_for_stateful_model():
    src = """\
`include "disciplines.vams"
module stateful(vin, vout);
    input voltage vin;
    output voltage vout;
    real sample;
    analog begin
        sample = V(vin);
        V(vout) <+ sample;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._rust_static_affine_ops == ()


def test_dynamic_branch_codegen_handles_state_index_expression_without_nested_fstring():
    src = """\
`include "disciplines.vams"
module bus_drive(vout);
    output electrical vout;
    electrical [0:3] dout;
    integer ch = 1;
    analog begin
        V(dout[ch]) <+ V(vout);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()

    model.evaluate({"vout": 0.75}, 0.0)
    model.evaluate({"vout": 0.5}, 1e-9)

    assert model.output_nodes["dout[1]"] == pytest.approx(0.5)
    assert "self._resolve_dynamic_node('dout'" in ModelCls._generated_code
    assert "self._format_dynamic_node('dout'" not in ModelCls._generated_code
    assert model._perf_stats["dynamic_node_cache_misses"] == 1
    assert model._perf_stats["dynamic_node_cache_hits"] == 1


def test_simulator_aggregates_dynamic_bus_node_cache_stats():
    src = """\
`include "disciplines.vams"
module bus_drive(vin);
    input voltage vin;
    electrical [0:3] dout;
    integer ch = 2;
    analog begin
        V(dout[ch]) <+ V(vin);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    model.node_map = {"vin": "VIN"}

    sim = Simulator()
    sim.add_source("VIN", dc(0.8))
    sim.add_model(model)
    sim.record("dout[2]")

    result = sim.run(tstop=3e-9, tstep=1e-9)

    assert result.signals["dout[2]"].tolist()[-1] == pytest.approx(0.8)
    assert sim._perf_stats["dynamic_node_cache_misses_total"] == 1
    assert sim._perf_stats["dynamic_node_cache_hits_total"] > 0
    assert sim._perf_stats["dynamic_node_cache_bypasses_total"] == 0
    assert sim._perf_stats["dynamic_node_cache_entries"] == 1
    assert sim._perf_stats["dynamic_node_cache_models"] == 1


def test_compiled_model_records_dynamic_branch_access_ir_for_2d_reads():
    src = """\
`include "disciplines.vams"
module bus_read(VSS);
    inout electrical VSS;
    electrical [1:0] dbus [0:3];
    real sample;
    analog begin
        sample = V(dbus[1][0], VSS);
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._dynamic_voltage_read_count == 1
    assert ModelCls._dynamic_output_write_count == 0
    assert ModelCls._dynamic_branch_accesses == (
        ("voltage_read", "dbus", 2, "ordinary"),
    )


def test_compiled_model_records_state_layout_metadata():
    src = """\
`include "disciplines.vams"
module state_probe(out);
    output electrical out;
    real x = 1.25;
    integer code = 2;
    genvar i;
    real accum[3:0];
    integer bins[0:2];
    analog begin
        V(out) <+ x + code;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))

    assert ModelCls._state_scalar_names == ("x", "code", "i")
    assert ModelCls._integer_state_names == ("code", "i")
    assert ModelCls._state_array_ranges == (
        ("accum", 0, 3, False),
        ("bins", 0, 2, True),
    )


def test_indexed_state_storage_mirrors_scalar_and_array_writes():
    src = """\
`include "disciplines.vams"
module state_probe(out);
    output electrical out;
    real x = 1.25;
    integer code = 2;
    genvar i;
    real accum[0:3];
    integer bins[0:2];
    analog begin
        x = 2.5;
        code = 3.6;
        accum[1] = x + 0.25;
        bins[2] = code + 0.6;
        V(out) <+ x + code + accum[1] + bins[2];
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    model._set_indexed_state_storage(
        {"x": 0, "code": 1, "i": 2},
        ("code", "i"),
        (
            ("accum", 0, 3, False),
            ("bins", 0, 2, True),
        ),
    )

    model.evaluate({}, 0.0)

    assert model.state["x"] == pytest.approx(2.5)
    assert model.state["code"] == 4
    assert model.arrays["accum"][1] == pytest.approx(2.75)
    assert model.arrays["bins"][2] == 5
    assert model._indexed_state_values == pytest.approx([2.5, 4.0, 0.0])
    assert model._indexed_state_array_values["accum"] == pytest.approx(
        [0.0, 2.75, 0.0, 0.0]
    )
    assert model._indexed_state_array_values["bins"] == pytest.approx(
        [0.0, 0.0, 5.0]
    )
    assert model._perf_stats["indexed_state_scalar_writes"] == 2
    assert model._perf_stats["indexed_state_array_writes"] == 2
    assert model._perf_stats["indexed_state_array_oob_writes"] == 0


def test_indexed_model_io_plan_includes_static_branch_io_nodes():
    src = """\
`include "disciplines.vams"
module sample_hold(clk, inp, out);
    input voltage clk;
    input voltage inp;
    output voltage out;
    real sample;
    analog begin
        @(cross(V(clk) - 0.5, +1)) sample = V(inp);
        V(out) <+ sample;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    model.node_map = {"clk": "CLK", "inp": "INP", "out": "OUT"}
    sim = Simulator()
    sim.add_model(model)

    plan = build_indexed_model_io_plan(sim)
    (model_io,) = plan.model_ios

    assert model_io.output_node_ids == ()
    assert model_io.static_voltage_read_node_ids == (plan.node_index.id_of("CLK"),)
    assert model_io.event_trigger_voltage_node_ids == (plan.node_index.id_of("CLK"),)
    assert model_io.event_voltage_read_node_ids == (plan.node_index.id_of("INP"),)
    assert model_io.event_body_voltage_read_node_ids == (plan.node_index.id_of("INP"),)
    assert model_io.static_output_write_node_ids == (plan.node_index.id_of("OUT"),)
    assert model_io.dynamic_voltage_read_count == 0
    assert model_io.dynamic_output_write_count == 0
    assert model_io.dynamic_branch_accesses == ()
    assert plan.static_voltage_read_count == 1
    assert plan.event_trigger_voltage_count == 1
    assert plan.event_voltage_read_count == 1
    assert plan.event_body_voltage_read_count == 1
    assert plan.static_output_write_count == 1


def test_indexed_model_io_plan_exposes_dynamic_branch_accesses():
    src = """\
`include "disciplines.vams"
module bus_drive(VSS);
    inout electrical VSS;
    electrical [0:3] dout;
    genvar i;
    analog begin
        for (i = 0; i <= 3; i = i + 1)
            V(dout[i], VSS) <+ i;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    sim = Simulator()
    sim.add_model(model)

    plan = build_indexed_model_io_plan(sim)
    (model_io,) = plan.model_ios

    assert model_io.dynamic_branch_accesses == (
        DynamicBranchAccessIO(
            role="output_write",
            base_node="dout",
            dimensions=1,
            context="ordinary",
        ),
    )
    assert plan.dynamic_branch_access_count == 1
    assert plan.dynamic_output_write_count == 1
    assert plan.dynamic_voltage_read_count == 0


def test_indexed_model_io_plan_exposes_state_layouts():
    src = """\
`include "disciplines.vams"
module state_probe(out);
    output electrical out;
    real x = 1.25;
    integer code = 2;
    genvar i;
    real accum[3:0];
    integer bins[0:2];
    analog begin
        V(out) <+ x + code;
    end
endmodule
"""
    ModelCls = compile_module(parse(src))
    model = ModelCls()
    sim = Simulator()
    sim.add_model(model)

    plan = build_indexed_model_io_plan(sim)
    (model_io,) = plan.model_ios

    assert model_io.state_scalar_names == ("x", "code", "i")
    assert model_io.state_scalar_ids == (0, 1, 2)
    assert model_io.integer_state_names == ("code", "i")
    assert model_io.state_array_layouts == (
        IndexedStateArrayLayout(
            name="accum",
            lo=0,
            hi=3,
            length=4,
            integer=False,
        ),
        IndexedStateArrayLayout(
            name="bins",
            lo=0,
            hi=2,
            length=3,
            integer=True,
        ),
    )
    assert plan.scalar_state_count == 3
    assert plan.integer_state_count == 2
    assert plan.state_array_count == 2
    assert plan.state_array_slot_count == 7


def test_indexed_run_plan_collects_sources_records_and_model_nodes():
    class MirrorModel(CompiledModel):
        def __init__(self):
            super().__init__()
            self.node_map = {"in": "vin", "out": "vout"}
            self.output_nodes = {"vout": 0.0}

    sim = Simulator()
    sim.add_source("vin", dc(0.25))
    sim.add_model(MirrorModel())
    sim.record("vout")

    plan = build_indexed_run_plan(sim, extra_nodes=["monitor"])

    assert plan.node_index.names == ("monitor", "vin", "vout")
    assert plan.source_node_ids == (plan.node_index.id_of("vin"),)
    assert plan.recorded_node_ids == (plan.node_index.id_of("vout"),)
    assert plan.model_node_ids == (
        plan.node_index.id_of("vin"),
        plan.node_index.id_of("vout"),
        plan.node_index.id_of("vout"),
    )


def test_indexed_trace_round_trip_is_lossless_for_simresult():
    result = SimResult(
        time=np.array([0.0, 1e-9]),
        signals={
            "vin": np.array([0.125, 0.25]),
            "vout": np.array([0.0, 0.9]),
        },
        step_sizes=np.array([0.0, 1e-9]),
    )
    index = build_node_index(["vin", "vout"])

    report = check_indexed_trace_round_trip(
        result,
        node_index=index,
        signal_names=["vin", "vout"],
    )

    assert report.passed
    assert report.checked_signals == 2
    assert report.checked_samples == 4
    assert report.max_abs_diff == 0.0
    assert report.summary().startswith("passed:")


def test_indexed_trace_reports_requested_signals_missing_from_result():
    result = SimResult(
        time=np.array([0.0]),
        signals={"vin": np.array([0.5])},
        step_sizes=np.array([0.0]),
    )

    report = check_indexed_trace_round_trip(
        result,
        signal_names=["vin", "missing"],
    )

    assert report.passed
    assert report.checked_signals == 1
    assert report.missing_signals == ("missing",)
