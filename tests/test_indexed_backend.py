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
    assert "self._format_dynamic_node('dout'" in FastModelCls._generated_code
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

    assert model.output_nodes["dout[1]"] == pytest.approx(0.75)
    assert "self._format_dynamic_node('dout'" in ModelCls._generated_code


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
