"""Tests for indexed simulator migration helpers."""

import numpy as np
import pytest

from evas.simulator.indexed import (
    IndexedVoltages,
    IndexedVoltageSnapshotter,
    NodeIndex,
    StateIndex,
    build_indexed_run_plan,
    build_node_index,
    check_indexed_trace_round_trip,
    copy_values_into,
)
from evas.simulator.backend import CompiledModel
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
