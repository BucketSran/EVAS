"""Tests for indexed simulator migration helpers."""

import pytest

from evas.simulator.indexed import (
    IndexedVoltages,
    NodeIndex,
    StateIndex,
    build_node_index,
    copy_values_into,
)


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
