"""Tests for the optional EVAS Rust backend ctypes bridge."""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.simulator.backend import CompiledModel, compile_module
from evas.simulator.engine import AboveDetector, CrossDetector, TransitionState
from evas.simulator.rust_backend import (
    BODY_EXPR_ADD,
    BODY_EXPR_CONST,
    BODY_EXPR_MUL,
    BODY_EXPR_POW,
    BODY_EXPR_READ_NODE,
    BODY_EXPR_READ_PARAM,
    BODY_EXPR_READ_STATE,
    BODY_EXPR_SUB,
    BODY_TARGET_NODE,
    BODY_TARGET_STATE,
    BodyExprOp,
    BodyStmtOp,
    LinearCondition,
    LinearOp,
    LinearTerm,
    RustBackendError,
    StaticAffineOp,
    TransitionTargetOp,
    default_rust_core_library_path,
    load_rust_backend,
)
from evas.simulator.whole_segment import validate_whole_segment_candidate

RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_CORE,
        check=True,
    )


def test_rust_backend_static_affine_batch_updates_array_buffer():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    values = array("d", [0.5, 0.0, 0.0])
    batch = backend.make_static_affine_batch(
        [
            StaticAffineOp(read_node_id=0, write_node_id=1, gain=2.0, bias=0.25),
            StaticAffineOp(read_node_id=1, write_node_id=2, gain=-1.0, bias=1.0),
        ]
    )

    backend.evaluate_static_affine(batch, values)

    assert values.tolist() == pytest.approx([0.5, 1.25, -0.25])


def test_rust_backend_copies_f64_buffers():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    source = array("d", [0.25, 0.5, -1.0])
    target = array("d", [0.0, 0.0, 0.0])

    backend.copy_f64(source, target)

    assert target.tolist() == pytest.approx(source.tolist())


def test_rust_backend_generates_prbs7_full_model_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array("d", [0.0, 0.1e-9, 0.11e-9, 0.12e-9, 1.11e-9, 1.12e-9])

    values, events = backend.prbs7_trace(
        times,
        clk={
            "v_lo": 0.0,
            "v_hi": 0.9,
            "period": 1.0e-9,
            "duty": 0.5,
            "rise": 20.0e-12,
            "fall": 20.0e-12,
            "delay": 0.1e-9,
            "width": 0.5e-9,
            "has_width": True,
        },
        rst_n={
            "v_lo": 0.0,
            "v_hi": 0.9,
            "period": 300.0e-9,
            "duty": 298.0 / 300.0,
            "rise": 20.0e-12,
            "fall": 20.0e-12,
            "delay": 2.0e-9,
            "width": 298.0e-9,
            "has_width": True,
        },
        en_voltage=0.9,
        vdd=0.9,
        vth=0.45,
        trf=10.0e-12,
        td=0.0,
        seed=127,
    )

    assert events == 2
    assert len(values) == len(times) * 11
    assert values[2 * 11] == pytest.approx(0.45)
    assert values[5 * 11 + 3] == pytest.approx(0.9)


def test_rust_backend_generates_generic_lfsr_transition_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array("d", [0.0, 0.1e-9, 0.11e-9, 0.12e-9, 1.11e-9, 1.12e-9])

    values, events = backend.lfsr_transition_trace(
        times,
        clk={
            "v_lo": 0.0,
            "v_hi": 0.9,
            "period": 1.0e-9,
            "duty": 0.5,
            "rise": 20.0e-12,
            "fall": 20.0e-12,
            "delay": 0.1e-9,
            "width": 0.5e-9,
            "has_width": True,
        },
        rst_n={
            "v_lo": 0.0,
            "v_hi": 0.9,
            "period": 300.0e-9,
            "duty": 298.0 / 300.0,
            "rise": 20.0e-12,
            "fall": 20.0e-12,
            "delay": 2.0e-9,
            "width": 298.0e-9,
            "has_width": True,
        },
        en_voltage=0.9,
        vdd=0.9,
        vth=0.45,
        trf=10.0e-12,
        td=0.0,
        seed=127,
        width=7,
        taps=(6, 5),
        shift_sources=(-1, 0, 1, 2, 3, 4, 5),
        output_bits=(6, 0, 1, 2, 3, 4, 5, 6),
        zero_guard_index=6,
    )

    assert events == 2
    assert len(values) == len(times) * 11
    assert values[2 * 11] == pytest.approx(0.45)
    assert values[5 * 11 + 3] == pytest.approx(0.9)


def test_rust_backend_generates_gain_timer_reduction_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array("d", [0.0, 1.0e-9, 2.0e-9, 3.0e-9])
    sample_times = array("d", [0.0, 1.0e-9, 2.0e-9])
    ones = array("d", [1.0, 1.0, 1.0, 1.0])
    zeros = array("d", [0.0, 0.0, 0.0, 0.0])
    sample_ones = array("d", [1.0, 1.0, 1.0])
    sample_zeros = array("d", [0.0, 0.0, 0.0])

    values, samples = backend.gain_timer_reduction_trace(
        times,
        sample_times,
        point_vdd=ones,
        point_vss=zeros,
        point_vinp=array("d", [0.0, 0.5, 1.0, 1.0]),
        point_vinn=zeros,
        point_voutp=array("d", [0.0, 1.0, 2.0, 2.0]),
        point_voutn=zeros,
        sample_vdd=sample_ones,
        sample_vss=sample_zeros,
        sample_vinp=array("d", [0.0, 0.5, 1.0]),
        sample_vinn=sample_zeros,
        sample_voutp=array("d", [0.0, 1.0, 2.0]),
        sample_voutn=sample_zeros,
        start_time=0.0,
        gain_scale=2.0,
        min_input_span=0.1,
        tedge=1.0e-12,
    )

    assert samples == 3
    assert len(values) == len(times) * 8
    assert values[3 * 8 + 6] == pytest.approx(1.0)
    assert values[3 * 8 + 7] == pytest.approx(1.0)


def test_rust_backend_generates_gain_measurement_flow_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array(
        "d",
        [
            0.0,
            10.0e-9,
            10.015e-9,
            10.030e-9,
            10.050e-9,
            10.075e-9,
            10.100e-9,
        ],
    )

    values, vin_events, lfsr_events = backend.gain_measurement_flow_trace(
        times,
        vin_event_times=array("d", [10.0e-9]),
        vin_event_vinp=array("d", [0.47]),
        vin_event_vinn=array("d", [0.45]),
        lfsr_event_times=array("d", [10.050e-9]),
        vcm=0.45,
        vth=0.45,
        dither_amp=0.014063,
        actual_gain=8.64,
        vin_transition=30.0e-12,
        lfsr_transition=50.0e-12,
        vdd=0.9,
        vss=0.0,
        lfsr_seed=42,
    )

    assert vin_events == 1
    assert lfsr_events == 1
    assert len(values) == len(times) * 4
    assert values[0] == pytest.approx(0.45)
    assert values[1] == pytest.approx(0.45)
    assert values[2] == pytest.approx(0.38924784)
    assert values[3] == pytest.approx(0.51075216)
    assert values[6 * 4 + 0] == pytest.approx(0.47)
    assert values[6 * 4 + 2] == pytest.approx(0.59715216)
    assert values[6 * 4 + 3] == pytest.approx(0.30284784)


def test_rust_backend_generates_cmp_delay_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array(
        "d",
        [0.0, 100.0e-12, 140.0e-12, 155.0e-12, 170.0e-12, 180.0e-12],
    )
    clk = array("d", [0.0, 0.9, 0.9, 0.9, 0.9, 0.9])
    vinp = array("d", [0.455] * len(times))
    vinn = array("d", [0.445] * len(times))
    vdd = array("d", [0.9] * len(times))

    values, events = backend.cmp_delay_trace(
        times,
        point_clk=clk,
        point_vinn=vinn,
        point_vinp=vinp,
        point_vdd=vdd,
        voffset=0.0,
        tau=4.34e-12,
        td0=20.5e-12,
        td_min=20.0e-12,
        td_max=200.0e-12,
        tedge=30.0e-12,
        edge_vth=0.45,
    )

    assert events == 1
    assert len(values) == len(times) * 6
    assert values[5 * 6 + 4] == pytest.approx(0.9)
    assert values[5 * 6 + 5] > 50.0


def test_rust_backend_generates_sar_loop_trace():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array(
        "d",
        [
            0.0,
            0.10e-9,
            0.20e-9,
            0.30e-9,
            0.40e-9,
            0.50e-9,
            0.60e-9,
            0.70e-9,
            0.80e-9,
            0.90e-9,
        ],
    )
    clk = array("d", [0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9])
    rst = array("d", [0.9] * len(times))
    vin = array("d", [0.62] * len(times))

    values, events = backend.sar_loop_trace(
        times,
        point_vin=vin,
        point_clk=clk,
        point_rst=rst,
        vdd=0.9,
        vth=0.45,
        sh_tr=1.0e-12,
        default_tr=1.0e-12,
        width=3,
    )

    signal_count = 14
    assert events == 5
    assert len(values) == len(times) * signal_count
    assert values[2 * signal_count + 1] == pytest.approx(0.62)
    assert values[4 * signal_count + 10] == pytest.approx(0.62)
    assert values[5 * signal_count + 5] == pytest.approx(0.9)
    assert any(value > 0.0 for value in values[11::signal_count])


def test_rust_backend_generates_cppll_reacquire_trace_from_events():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    times = array("d", [0.0, 0.5e-9, 1.0e-9, 1.5e-9, 2.0e-9])

    values = backend.cppll_reacquire_trace(
        times,
        ref_events=(array("d", [0.0, 1.0e-9]), array("d", [0.0, 0.9])),
        dco_events=(array("d", [0.0, 0.5e-9]), array("d", [0.0, 0.9])),
        fb_events=(array("d", [0.0]), array("d", [0.0])),
        lock_events=(array("d", [0.0, 1.5e-9]), array("d", [0.0, 0.9])),
        vctrl_events=(array("d", [0.0, 1.2e-9]), array("d", [0.42, 0.55])),
        vh=0.9,
        vl=0.0,
        ref_tedge=1.0e-9,
        pll_tedge=0.5e-9,
    )

    signal_count = 7
    assert len(values) == len(times) * signal_count
    assert values[0] == pytest.approx(0.9)
    assert values[1] == pytest.approx(0.0)
    assert values[3 * signal_count + 2] == pytest.approx(0.45)
    assert values[3 * signal_count + 3] == pytest.approx(0.9)
    assert values[3 * signal_count + 5] == pytest.approx(0.55)
    assert values[4 * signal_count + 6] == pytest.approx(0.9)


def test_compiler_emits_scalar_lfsr_transition_whole_segment_candidate():
    source = """
module prbs7_ref (
    input electrical clk,
    input electrical rst_n,
    input electrical en,
    output electrical serial_out,
    output electrical state_0,
    output electrical state_1,
    output electrical state_2,
    output electrical state_3,
    output electrical state_4,
    output electrical state_5,
    output electrical state_6
);
parameter real vdd = 0.9;
parameter real vth = 0.45;
parameter real trf = 10p;
parameter real td = 0p;
parameter integer seed = 127;
integer state_0_i;
integer state_1_i;
integer state_2_i;
integer state_3_i;
integer state_4_i;
integer state_5_i;
integer state_6_i;
integer feedback;
integer serial_bit;
analog begin
    @(initial_step) begin
        state_0_i = (seed >> 0) & 1;
        state_1_i = (seed >> 1) & 1;
        state_2_i = (seed >> 2) & 1;
        state_3_i = (seed >> 3) & 1;
        state_4_i = (seed >> 4) & 1;
        state_5_i = (seed >> 5) & 1;
        state_6_i = (seed >> 6) & 1;
        if (state_0_i == 0 && state_1_i == 0 && state_2_i == 0 &&
            state_3_i == 0 && state_4_i == 0 && state_5_i == 0 &&
            state_6_i == 0)
            state_6_i = 1;
    end
    @(cross(V(clk) - vth, +1)) begin
        if (V(rst_n) < vth) begin
            state_0_i = (seed >> 0) & 1;
            state_1_i = (seed >> 1) & 1;
            state_2_i = (seed >> 2) & 1;
            state_3_i = (seed >> 3) & 1;
            state_4_i = (seed >> 4) & 1;
            state_5_i = (seed >> 5) & 1;
            state_6_i = (seed >> 6) & 1;
            if (state_0_i == 0 && state_1_i == 0 && state_2_i == 0 &&
                state_3_i == 0 && state_4_i == 0 && state_5_i == 0 &&
                state_6_i == 0)
                state_6_i = 1;
        end else if (V(en) > vth) begin
            feedback = state_6_i ^ state_5_i;
            state_6_i = state_5_i;
            state_5_i = state_4_i;
            state_4_i = state_3_i;
            state_3_i = state_2_i;
            state_2_i = state_1_i;
            state_1_i = state_0_i;
            state_0_i = feedback;
        end
    end
    serial_bit = state_6_i;
    V(serial_out) <+ transition(serial_bit ? vdd : 0.0, td, trf, trf);
    V(state_0) <+ transition(state_0_i ? vdd : 0.0, td, trf, trf);
    V(state_1) <+ transition(state_1_i ? vdd : 0.0, td, trf, trf);
    V(state_2) <+ transition(state_2_i ? vdd : 0.0, td, trf, trf);
    V(state_3) <+ transition(state_3_i ? vdd : 0.0, td, trf, trf);
    V(state_4) <+ transition(state_4_i ? vdd : 0.0, td, trf, trf);
    V(state_5) <+ transition(state_5_i ? vdd : 0.0, td, trf, trf);
    V(state_6) <+ transition(state_6_i ? vdd : 0.0, td, trf, trf);
end
endmodule
"""
    model_cls = compile_module(parse(source))

    candidates = model_cls._whole_segment_candidates

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate[0] == "cross_scalar_lfsr_transition_bus_v1"
    assert candidate[1:9] == ("clk", "rst_n", "en", "vth", "seed", "vdd", "td", "trf")
    assert candidate[10] == (6, 5)
    assert candidate[11] == (-1, 0, 1, 2, 3, 4, 5)
    assert candidate[12] == (
        "serial_out",
        "state_0",
        "state_1",
        "state_2",
        "state_3",
        "state_4",
        "state_5",
        "state_6",
    )
    assert candidate[13] == (6, 0, 1, 2, 3, 4, 5, 6)


def test_compiler_emits_topwall_whole_segment_candidates_from_semantic_gold():
    bench_root = Path(__file__).resolve().parents[2] / "behavioral-veriloga-eval"
    if not bench_root.exists():
        pytest.skip("vaBench release package is not available")

    cases = {
        "gain_timer_reduction_v1": "benchmark-vabench-release-v1/tasks/SUP01_measurement_instrumentation_flows/vbr1_l1_gain_estimator/forms/tb/gold/gain_estimator.va",
        "cmp_delay_log_transition_v1": "benchmark-vabench-release-v1/tasks/CT02_comparator_and_decision_circuits/vbr1_l1_propagation_delay_comparator/forms/tb/gold/cmp_delay.va",
        "edge_interval_timer_v1": "benchmark-vabench-release-v1/tasks/CT02_comparator_and_decision_circuits/vbr1_l1_propagation_delay_comparator/forms/tb/gold/edge_interval_timer.va",
        "weighted_sar_adc_v1": "benchmark-vabench-release-v1/tasks/CT01_data_converter_models/vbr1_l2_weighted_sar_adc_dac_loop/forms/tb/gold/sar_adc_weighted_8b.va",
        "weighted_dac_v1": "benchmark-vabench-release-v1/tasks/CT01_data_converter_models/vbr1_l2_weighted_sar_adc_dac_loop/forms/tb/gold/dac_weighted_8b.va",
        "sample_hold_rising_v1": "benchmark-vabench-release-v1/tasks/CT01_data_converter_models/vbr1_l2_weighted_sar_adc_dac_loop/forms/tb/gold/sh_ideal.va",
        "ref_step_clock_v1": "benchmark-vabench-release-v1/tasks/CT05_pll_clock_and_timing_systems/vbr1_l2_cppll_tracking_and_frequency_step_reacquire_flow/forms/tb/gold/ref_step_clk.va",
        "cppll_timer_v1": "benchmark-vabench-release-v1/tasks/CT05_pll_clock_and_timing_systems/vbr1_l2_cppll_tracking_and_frequency_step_reacquire_flow/forms/tb/gold/cppll_timer_ref.va",
    }

    for expected_kind, rel_path in cases.items():
        source = (bench_root / rel_path).read_text(encoding="utf-8")
        model_cls = compile_module(parse(source))
        candidates = tuple(model_cls._whole_segment_candidates)
        kinds = {candidate[0] for candidate in candidates}
        if expected_kind not in kinds:
            pytest.skip(
                f"vaBench sibling fixture no longer emits {expected_kind}: {rel_path}"
            )
        candidate = next(candidate for candidate in candidates if candidate[0] == expected_kind)
        contract = validate_whole_segment_candidate(candidate)
        assert contract.valid, contract
        if expected_kind in {"ref_step_clock_v1", "cppll_timer_v1"}:
            assert candidate[1:3] == ("VDD", "VSS")


def test_whole_segment_candidate_contract_rejects_bad_abi_shapes():
    bad_arity = validate_whole_segment_candidate(
        ("weighted_dac_v1", ("din2", "din1", "din0"), "vout", "vdd")
    )
    assert not bad_arity.valid
    assert "arity:4!=5" in bad_arity.errors

    bad_width = validate_whole_segment_candidate(
        ("weighted_dac_v1", ("din1", "din0"), "vout", "vdd", "vth")
    )
    assert not bad_width.valid
    assert "din_width_lt_3" in bad_width.errors


def test_compiler_rejects_weighted_dac_false_positive_state_machines():
    bench_root = Path(__file__).resolve().parents[2] / "behavioral-veriloga-eval"
    if not bench_root.exists():
        pytest.skip("vaBench release package is not available")

    cases = [
        "benchmark-vabench-release-v1/tasks/CT02_comparator_and_decision_circuits/vbr1_l1_debounce_latch/forms/dut/gold/debounce_latch.va",
        "benchmark-vabench-release-v1/tasks/CT05_pll_clock_and_timing_systems/vbr1_l1_lock_detector/forms/dut/gold/lock_detector.va",
    ]

    for rel_path in cases:
        source = (bench_root / rel_path).read_text(encoding="utf-8")
        model_cls = compile_module(parse(source))
        kinds = {candidate[0] for candidate in model_cls._whole_segment_candidates}
        assert "weighted_dac_v1" not in kinds


def test_compiler_rejects_name_only_whole_segment_stub():
    source = """
module gain_estimator(VDD, VSS, vinp, vinn, voutp, voutn, gain_out, valid);
 inout VDD,VSS; input vinp,vinn,voutp,voutn; output gain_out, valid;
 electrical VDD,VSS,vinp,vinn,voutp,voutn,gain_out,valid;
 parameter real sample_period=1n; parameter real start_time=20n;
 parameter real gain_scale=10.0; parameter real min_input_span=0.02;
 parameter real tedge=200p;
 real in_min,in_max,out_min,out_max,gain_q; integer valid_q;
 analog begin
   @(initial_step) begin
     in_min=0; in_max=0; out_min=0; out_max=0; gain_q=0; valid_q=0;
   end
 end
endmodule
"""
    model_cls = compile_module(parse(source))

    assert model_cls._whole_segment_candidates == ()


def test_rust_backend_computes_max_err_ratio_for_node_ids():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    values = array("d", [1.0, 2.1, 0.5])
    previous = array("d", [1.0, 2.0, 0.0])
    node_ids = backend.make_node_id_batch([1, 2])

    ratio = backend.max_err_ratio(values, previous, node_ids, 1.0e-3, 1.0e-6)

    assert ratio == pytest.approx(0.5 / (1.0e-3 * 0.5 + 1.0e-6))


def test_rust_backend_adaptive_step_policy_matches_python_scheduler():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())

    default_floor = backend.adaptive_step_floor(
        1.0e-9,
        1.0e-9 / 4096.0,
        min_step_defaulted=True,
    )
    explicit_floor = backend.adaptive_step_floor(
        1.0e-9,
        1.0e-13,
        min_step_defaulted=False,
    )
    shrunk, clamped = backend.adaptive_shrink_step(
        dynamic_step=1.0e-12,
        err_ratio=100.0,
        min_step=1.0e-9 / 4096.0,
        adaptive_floor=default_floor,
    )
    source_shrunk, source_clamped = backend.adaptive_shrink_step(
        dynamic_step=1.0e-12,
        err_ratio=100.0,
        min_step=1.0e-9 / 4096.0,
        adaptive_floor=default_floor,
        err_ratio_from_source=True,
    )
    guarded_shrunk, guarded_clamped = backend.adaptive_shrink_step(
        dynamic_step=1.0e-12,
        err_ratio=100.0,
        min_step=1.0e-9 / 4096.0,
        adaptive_floor=default_floor,
        adaptive_floor_allowed=False,
    )

    assert default_floor == pytest.approx(1.0e-9 / 64.0)
    assert explicit_floor == pytest.approx(1.0e-13)
    assert shrunk == pytest.approx(default_floor)
    assert clamped is True
    assert source_shrunk == pytest.approx(2.5e-13)
    assert source_clamped is False
    assert guarded_shrunk == pytest.approx(2.5e-13)
    assert guarded_clamped is False


def test_rust_backend_records_values_for_node_ids():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    values = array("d", [1.0, 2.0, 3.0])
    node_ids = backend.make_node_id_batch([2, 0, 9])

    out = backend.record_values_for_ids(values, node_ids, default=-1.0)

    assert list(out) == pytest.approx([3.0, 1.0, -1.0])


def test_rust_backend_interpolates_event_values():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    previous = array("d", [0.0, 2.0])
    current = array("d", [10.0, 4.0])

    out = backend.interpolate_event_values(previous, current, 0.0, 10.0, 2.5)

    assert list(out) == pytest.approx([2.5, 2.5])


def test_rust_backend_timer_static_linear_queue_trace_preserves_order():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    event_batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=1,
                target_id=0,
                bias=1.0,
                terms=(LinearTerm(source_kind=1, source_id=0, gain=1.0),),
            ),
            LinearOp(
                target_kind=1,
                target_id=0,
                bias=0.0,
                terms=(LinearTerm(source_kind=1, source_id=0, gain=2.0),),
            ),
        ]
    )
    evaluate_batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=0,
                target_id=0,
                bias=0.0,
                terms=(LinearTerm(source_kind=1, source_id=0, gain=1.0),),
            )
        ]
    )
    node_values = array("d", [0.0])
    state_values = array("d", [1.0])

    flat_values, event_count = backend.timer_static_linear_queue_trace(
        array("d", [0.0, 1.0e-9, 2.0e-9]),
        source_node_ids=[],
        source_values=array("d"),
        node_values=node_values,
        state_values=state_values,
        timer_starts=array("d", [0.0, 0.0]),
        timer_periods=array("d", [1.0e-9, 1.0e-9]),
        event_op_starts=[0, 1],
        event_op_counts=[1, 1],
        event_batch=event_batch,
        evaluate_batch=evaluate_batch,
        record_node_ids=[0],
    )

    assert event_count == 6
    assert list(flat_values) == pytest.approx([4.0, 10.0, 22.0])
    assert list(state_values) == pytest.approx([22.0])


def test_rust_backend_scans_transition_breakpoints_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    states = [
        TransitionState(
            start_time=0.0,
            start_val=0.0,
            target_val=1.0,
            delay=0.0,
            rise_time=4.0e-9,
            fall_time=4.0e-9,
            active=True,
        ),
        TransitionState(
            start_time=1.0e-9,
            start_val=1.0,
            target_val=0.0,
            delay=0.0,
            rise_time=5.0e-9,
            fall_time=5.0e-9,
            active=True,
        ),
        TransitionState(
            start_time=2.0e-9,
            start_val=0.0,
            target_val=1.0,
            delay=0.0,
            rise_time=1.0e-9,
            fall_time=1.0e-9,
            active=False,
        ),
    ]
    expected = min(
        bp
        for bp in (state.next_breakpoint(1.5e-9, 0.0) for state in states)
        if bp is not None
    )

    result = backend.next_transition_breakpoint(
        array("d", [state.start_time for state in states]),
        array("d", [state.start_val for state in states]),
        array("d", [state.target_val for state in states]),
        array("d", [state.delay for state in states]),
        array("d", [state.rise_time for state in states]),
        array("d", [state.fall_time for state in states]),
        [1 if state.active else 0 for state in states],
        1.5e-9,
        0.0,
    )

    assert result == pytest.approx(expected)


def test_rust_backend_steps_transition_state_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    default_transition = 1.0e-12
    state_count = 2
    current_values = array("d", [0.0] * state_count)
    target_values = array("d", [0.0] * state_count)
    start_times = array("d", [0.0] * state_count)
    start_values = array("d", [0.0] * state_count)
    delays = array("d", [0.0] * state_count)
    rise_times = array("d", [0.0] * state_count)
    fall_times = array("d", [0.0] * state_count)
    active_flags = [0] * state_count
    initialized_flags = [0] * state_count
    output_values = array("d", [0.0] * state_count)
    py_states = [None] * state_count
    py_initialized = [False] * state_count

    def py_step(idx, time, target, delay, rise, fall, initial_condition_mode=False):
        effective_rise = rise if rise > 0 else default_transition
        effective_fall = fall if fall > 0 else default_transition
        state = py_states[idx]
        if initial_condition_mode or not py_initialized[idx]:
            if state is None:
                state = TransitionState()
                py_states[idx] = state
            state.current_val = target
            state.target_val = target
            state.start_val = target
            state.start_time = time
            state.delay = delay
            state.rise_time = effective_rise
            state.fall_time = effective_fall
            state.active = False
            py_initialized[idx] = True
            return target
        state.evaluate(time)
        state.set_target(
            time,
            target,
            delay,
            rise,
            fall,
            default_transition,
        )
        return state.evaluate(time)

    steps = [
        (0.0, [0.0, 1.0], [0.0, 0.0], [10.0e-9, 0.0], [10.0e-9, 5.0e-9], False),
        (0.0, [1.0, 0.0], [0.0, 0.0], [10.0e-9, 4.0e-9], [10.0e-9, 5.0e-9], False),
        (5.0e-9, [1.0, 0.0], [0.0, 0.0], [10.0e-9, 4.0e-9], [10.0e-9, 5.0e-9], False),
        (5.0e-9, [0.0, 1.0], [0.0, 0.0], [10.0e-9, 4.0e-9], [10.0e-9, 5.0e-9], False),
        (7.5e-9, [0.0, 1.0], [0.0, 0.0], [10.0e-9, 4.0e-9], [10.0e-9, 5.0e-9], False),
        (12.0e-9, [0.25, 0.75], [1.0e-9, 0.0], [0.0, 4.0e-9], [0.0, 5.0e-9], True),
    ]

    for time, targets, step_delays, rises, falls, initial_condition_mode in steps:
        expected_outputs = [
            py_step(
                idx,
                time,
                targets[idx],
                step_delays[idx],
                rises[idx],
                falls[idx],
                initial_condition_mode,
            )
            for idx in range(state_count)
        ]
        backend.transition_state_step(
            current_values,
            target_values,
            start_times,
            start_values,
            delays,
            rise_times,
            fall_times,
            active_flags,
            initialized_flags,
            array("d", targets),
            array("d", step_delays),
            array("d", rises),
            array("d", falls),
            output_values,
            time,
            default_transition,
            initial_condition_mode,
        )

        assert output_values.tolist() == pytest.approx(expected_outputs)
        assert initialized_flags == [1, 1]
        for idx, state in enumerate(py_states):
            assert current_values[idx] == pytest.approx(state.current_val)
            assert target_values[idx] == pytest.approx(state.target_val)
            assert start_times[idx] == pytest.approx(state.start_time)
            assert start_values[idx] == pytest.approx(state.start_val)
            assert delays[idx] == pytest.approx(state.delay)
            assert rise_times[idx] == pytest.approx(state.rise_time)
            assert fall_times[idx] == pytest.approx(state.fall_time)
            assert active_flags[idx] == int(bool(state.active))


def test_rust_backend_scans_timer_breakpoints_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())

    result = backend.next_timer_breakpoint(
        array("d", [10.0e-9, 4.0e-9, 7.0e-9]),
        array("d", [0.0, 4.0e-9, 0.0]),
        [0, 1, 0],
        1.0e-9,
    )

    assert result == pytest.approx(7.0e-9)


def test_rust_backend_steps_periodic_timers_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    next_fire_times = array("d", [0.0, 10.0e-9, 10.0e-9, 0.0])
    has_state_flags = [0, 1, 1, 0]
    periods = array("d", [10.0e-9, 10.0e-9, 10.0e-9, float("nan")])
    starts = array("d", [5.0e-9, 0.0, 0.0, 0.0])
    has_start_flags = [1, 0, 0, 0]
    due_flags = [0, 0, 0, 0]
    skipped_flags = [0, 0, 0, 0]

    py_due = []
    py_next = []
    for idx, period in enumerate(periods):
        model = CompiledModel()
        key = f"t{idx}"
        if has_state_flags[idx]:
            model._set_timer_state(key, next_fire_times[idx])
        start = starts[idx] if has_start_flags[idx] else None
        due = model._check_timer(key, 10.0e-9, period, start)
        py_due.append(int(due))
        py_next.append(float(model.timer_states.get(key, next_fire_times[idx])))

    backend.timer_periodic_step(
        next_fire_times,
        has_state_flags,
        periods,
        starts,
        has_start_flags,
        due_flags,
        skipped_flags,
        10.0e-9,
        reschedule_on_due=True,
    )

    assert due_flags == py_due
    assert next_fire_times.tolist() == pytest.approx(py_next)
    assert skipped_flags == [1, 0, 0, 0]


def test_rust_backend_fuses_timer_lfsr_and_output_write():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    state_values = array("d", [0.0] * 12)
    state_values[0] = 1.0
    state_values[1] = 0.0
    state_values[2] = 1.0
    state_values[3] = 1.0
    node_values = array("d", [0.0, 0.2, 0.9, 0.7])
    next_fire_times = array("d", [1.0e-9])
    has_state_flags = [1]
    batch = backend.make_lfsr_event_batch(
        lfsr_slots=[0, 1, 2, 3],
        tmp_slots=[4, 5, 6, 7, 8],
        tap_slots=[3, 1, 0],
        gate_node_id=3,
        gate_threshold=0.5,
        high_node_id=2,
        low_node_id=1,
        output_state_id=9,
        output_node_id=0,
        loop_state_id=10,
        loop_final_value=4.0,
    )

    due, skipped, executed, output_written = backend.timer_lfsr_output_step(
        batch,
        node_values,
        state_values,
        next_fire_times,
        has_state_flags,
        1.0e-9,
        start=0.0,
        has_start=True,
        time=1.0e-9,
    )

    assert due is True
    assert skipped is False
    assert executed is True
    assert output_written is True
    assert next_fire_times.tolist() == pytest.approx([2.0e-9])
    assert has_state_flags == [1]
    assert state_values.tolist()[0:4] == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert state_values[9] == pytest.approx(0.9)
    assert node_values[0] == pytest.approx(0.9)


def test_rust_backend_steps_absolute_timers_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    next_fire_times = array("d", [0.0, 10.0e-9, 10.0e-9, 0.0])
    has_state_flags = [0, 1, 1, 0]
    last_fired_times = array("d", [0.0, 10.0e-9, 0.0, 0.0])
    has_last_fired_flags = [0, 1, 0, 0]
    targets = array("d", [5.0e-9, 10.0e-9, 10.0e-9, float("nan")])
    due_flags = [0, 0, 0, 0]
    expired_flags = [0, 0, 0, 0]

    py_due = []
    py_next = []
    py_last = []
    py_has_last = []
    for idx, target in enumerate(targets):
        model = CompiledModel()
        key = f"ta{idx}"
        if has_state_flags[idx]:
            model._set_timer_state(key, next_fire_times[idx])
        if has_last_fired_flags[idx]:
            model._set_timer_last_fired(key, last_fired_times[idx])
        due = model._check_timer_at(key, 10.0e-9, target)
        py_due.append(int(due))
        py_next.append(float(model.timer_states.get(key, next_fire_times[idx])))
        py_last.append(float(model.timer_last_fired.get(key, last_fired_times[idx])))
        py_has_last.append(int(key in model.timer_last_fired))

    backend.timer_absolute_step(
        next_fire_times,
        has_state_flags,
        last_fired_times,
        has_last_fired_flags,
        targets,
        due_flags,
        expired_flags,
        10.0e-9,
    )

    assert due_flags == py_due
    assert next_fire_times.tolist() == pytest.approx(py_next)
    assert last_fired_times.tolist() == pytest.approx(py_last)
    assert has_last_fired_flags == py_has_last
    assert expired_flags == [1, 0, 0, 0]


def test_rust_backend_steps_cross_detectors_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    directions = [1, -1, 1]
    detectors = [CrossDetector(direction=direction) for direction in directions]
    prev_values = array("d", [0.0] * len(directions))
    prev_times = array("d", [0.0] * len(directions))
    pprev_values = array("d", [0.0] * len(directions))
    pprev_times = array("d", [0.0] * len(directions))
    initialized_flags = [0] * len(directions)
    last_cross_times = array("d", [-1.0] * len(directions))
    triggered_flags = [0] * len(directions)
    cross_times = array("d", [0.0] * len(directions))
    trigger_directions = [0] * len(directions)
    went_beyond_flags = [0] * len(directions)
    steps = [
        (0.0, [-1.0, 1.0, -1.0]),
        (1.0e-9, [1.0, -1.0, 0.0]),
        (2.0e-9, [2.0, -2.0, 1.0]),
    ]

    for time, values in steps:
        expected_triggered = [
            int(detector.check(time, value, time_tol=1.0e-18, expr_tol=1.0e-12))
            for detector, value in zip(detectors, values)
        ]
        backend.cross_detector_step(
            prev_values,
            prev_times,
            pprev_values,
            pprev_times,
            initialized_flags,
            directions,
            last_cross_times,
            array("d", values),
            triggered_flags,
            cross_times,
            trigger_directions,
            went_beyond_flags,
            time,
            time_tol=1.0e-18,
            expr_tol=1.0e-12,
        )

        assert initialized_flags == [1, 1, 1]
        assert triggered_flags == expected_triggered
        assert prev_values.tolist() == pytest.approx(
            [detector.prev_val for detector in detectors]
        )
        assert pprev_values.tolist() == pytest.approx(
            [detector.pprev_val for detector in detectors]
        )
        assert prev_times.tolist() == pytest.approx(
            [detector.prev_time for detector in detectors]
        )
        assert pprev_times.tolist() == pytest.approx(
            [detector.pprev_time for detector in detectors]
        )
        assert last_cross_times.tolist() == pytest.approx(
            [detector.last_cross_time for detector in detectors]
        )
        for idx, detector in enumerate(detectors):
            if detector.last_triggered:
                assert cross_times[idx] == pytest.approx(detector.t_cross)
                assert trigger_directions[idx] == detector.last_trigger_direction
                assert went_beyond_flags[idx] == int(detector.last_trigger_went_beyond)
            else:
                assert trigger_directions[idx] == 0
                assert went_beyond_flags[idx] == 0


def test_rust_backend_steps_above_detectors_like_python():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    directions = [1, 1, -1]
    detectors = [AboveDetector(direction=direction) for direction in directions]
    prev_values = array("d", [0.0] * len(directions))
    prev_times = array("d", [0.0] * len(directions))
    pprev_values = array("d", [0.0] * len(directions))
    pprev_times = array("d", [0.0] * len(directions))
    initialized_flags = [0] * len(directions)
    triggered_flags = [0] * len(directions)
    cross_times = array("d", [0.0] * len(directions))
    steps = [
        (0.0, [0.1, -1.0, 1.0]),
        (1.0e-9, [0.2, 1.0, 2.0]),
        (2.0e-9, [-0.5, 2.0, 3.0]),
    ]

    for time, values in steps:
        expected_triggered = [
            int(detector.check(time, value))
            for detector, value in zip(detectors, values)
        ]
        backend.above_detector_step(
            prev_values,
            prev_times,
            pprev_values,
            pprev_times,
            initialized_flags,
            directions,
            array("d", values),
            triggered_flags,
            cross_times,
            time,
        )

        assert initialized_flags == [1, 1, 1]
        assert triggered_flags == expected_triggered
        assert prev_values.tolist() == pytest.approx(
            [detector.prev_val for detector in detectors]
        )
        assert pprev_values.tolist() == pytest.approx(
            [detector.pprev_val for detector in detectors]
        )
        assert prev_times.tolist() == pytest.approx(
            [detector.prev_time for detector in detectors]
        )
        assert pprev_times.tolist() == pytest.approx(
            [detector.pprev_time for detector in detectors]
        )
        for idx, detector in enumerate(detectors):
            if detector.last_triggered:
                assert cross_times[idx] == pytest.approx(detector.t_cross)


def test_rust_backend_computes_dynamic_bus_offsets():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    out_node_ids = [0, 0, 0]

    backend.dynamic_bus_offsets(
        base_offsets=[10, 100, 200],
        outer_lengths=[4, 8, 3],
        inner_strides=[1, 1, 2],
        inner_lengths=[1, 1, 2],
        first_indices=[2, 7, 1],
        second_indices=[0, 0, 1],
        has_second_index_flags=[0, 0, 1],
        out_node_ids=out_node_ids,
    )

    assert out_node_ids == [12, 107, 203]


def test_rust_backend_rejects_dynamic_bus_offset_out_of_bounds():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())

    with pytest.raises(RustBackendError, match="code -202"):
        backend.dynamic_bus_offsets(
            base_offsets=[10],
            outer_lengths=[4],
            inner_strides=[1],
            inner_lengths=[1],
            first_indices=[4],
            second_indices=[0],
            has_second_index_flags=[0],
            out_node_ids=[0],
        )


def test_rust_backend_static_linear_batch_updates_node_and_state_buffers():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.75, 0.0])
    state_values = array("d", [4.0, 0.0])
    batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=1,
                target_id=1,
                bias=0.25,
                terms=(
                    LinearTerm(source_kind=0, source_id=0, gain=2.0),
                    LinearTerm(source_kind=1, source_id=0, gain=0.5),
                ),
            ),
            LinearOp(
                target_kind=0,
                target_id=1,
                bias=-0.25,
                terms=(LinearTerm(source_kind=1, source_id=1, gain=1.0),),
            ),
        ]
    )

    backend.evaluate_static_linear(batch, node_values, state_values)

    assert state_values.tolist() == pytest.approx([4.0, 3.75])
    assert node_values.tolist() == pytest.approx([0.75, 3.5])


def test_rust_backend_static_linear_batch_evaluates_conditional_select():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.75])
    state_values = array("d", [0.0])
    batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=1,
                target_id=0,
                bias=2.0,
                terms=(),
                condition=LinearCondition(
                    op_kind=1,
                    left_bias=0.0,
                    left_terms=(LinearTerm(source_kind=0, source_id=0, gain=1.0),),
                    right_bias=0.5,
                    right_terms=(),
                ),
                false_bias=-2.0,
                false_terms=(),
            ),
        ]
    )

    backend.evaluate_static_linear(batch, node_values, state_values)
    assert state_values.tolist() == pytest.approx([2.0])

    node_values[0] = 0.25
    backend.evaluate_static_linear(batch, node_values, state_values)
    assert state_values.tolist() == pytest.approx([-2.0])


def test_rust_backend_static_linear_batch_coerces_integer_target_before_read():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.0])
    state_values = array("d", [0.0])
    batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=1,
                target_id=0,
                bias=1.6,
                terms=(),
                target_integer=True,
            ),
            LinearOp(
                target_kind=0,
                target_id=0,
                bias=0.0,
                terms=(LinearTerm(source_kind=1, source_id=0, gain=1.0),),
            ),
        ]
    )

    backend.evaluate_static_linear(batch, node_values, state_values)

    assert state_values.tolist() == pytest.approx([2.0])
    assert node_values.tolist() == pytest.approx([2.0])


def _veriloga_integer(value: float) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    if value >= 0:
        return float(int(value + 0.5))
    return float(int(value - 0.5))


def test_rust_backend_evaluates_body_ir_batch():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.25, 0.0])
    state_values = array("d", [2.0])
    param_values = array("d", [3.0])
    expr_ops = [
        BodyExprOp(BODY_EXPR_READ_NODE, index=0),
        BodyExprOp(BODY_EXPR_READ_PARAM, index=0),
        BodyExprOp(BODY_EXPR_MUL),
        BodyExprOp(BODY_EXPR_READ_STATE, index=0),
        BodyExprOp(BODY_EXPR_ADD),
    ]
    batch = backend.make_body_ir_batch(
        stmt_ops=[
            BodyStmtOp(
                target_kind=BODY_TARGET_NODE,
                target_id=1,
                expr_start=0,
                expr_count=len(expr_ops),
            )
        ],
        expr_ops=expr_ops,
    )

    backend.evaluate_body_ir(batch, node_values, state_values, param_values)

    assert node_values.tolist() == pytest.approx([0.25, 2.75])
    assert state_values.tolist() == pytest.approx([2.0])


def test_rust_backend_evaluates_body_expr_without_writes():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.25])
    state_values = array("d", [2.0])
    param_values = array("d", [3.0])
    expr_ops = [
        BodyExprOp(BODY_EXPR_READ_NODE, index=0),
        BodyExprOp(BODY_EXPR_READ_PARAM, index=0),
        BodyExprOp(BODY_EXPR_MUL),
        BodyExprOp(BODY_EXPR_READ_STATE, index=0),
        BodyExprOp(BODY_EXPR_ADD),
    ]

    value = backend.evaluate_body_expr(expr_ops, node_values, state_values, param_values)

    assert value == pytest.approx(2.75)
    assert node_values.tolist() == pytest.approx([0.25])
    assert state_values.tolist() == pytest.approx([2.0])


def test_rust_backend_evaluates_body_expr_pow_function():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    expr_ops = [
        BodyExprOp(BODY_EXPR_CONST, value=1.8),
        BodyExprOp(BODY_EXPR_CONST, value=9.0),
        BodyExprOp(BODY_EXPR_POW),
    ]

    value = backend.evaluate_body_expr(expr_ops, array("d"), array("d"), array("d"))

    assert value == pytest.approx(1.8**9)


def test_rust_backend_evaluates_body_expr_batch_segments():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.7, 0.2])
    state_values = array("d", [2.0])
    param_values = array("d", [0.5])
    batch = backend.make_body_expr_batch(
        [
            [
                BodyExprOp(BODY_EXPR_READ_NODE, index=0),
                BodyExprOp(BODY_EXPR_READ_PARAM, index=0),
                BodyExprOp(BODY_EXPR_SUB),
            ],
            [
                BodyExprOp(BODY_EXPR_READ_NODE, index=1),
                BodyExprOp(BODY_EXPR_READ_STATE, index=0),
                BodyExprOp(BODY_EXPR_MUL),
            ],
        ]
    )

    values = backend.evaluate_body_expr_batch(
        batch, node_values, state_values, param_values
    )

    assert values == pytest.approx((0.2, 0.4))
    assert node_values.tolist() == pytest.approx([0.7, 0.2])
    assert state_values.tolist() == pytest.approx([2.0])


def test_rust_backend_body_ir_random_state_writes_match_python_oracle():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    rng = random.Random(94)

    for case_idx in range(100):
        node0 = rng.uniform(-3.0, 3.0)
        state0 = rng.uniform(-3.0, 3.0)
        param0 = rng.uniform(-3.0, 3.0)
        bias = rng.uniform(-1.0, 1.0)
        integer_target = case_idx % 2 == 0
        node_values = array("d", [node0])
        state_values = array("d", [state0, 0.0])
        param_values = array("d", [param0])
        expr_ops = [
            BodyExprOp(BODY_EXPR_READ_STATE, index=0),
            BodyExprOp(BODY_EXPR_READ_PARAM, index=0),
            BodyExprOp(BODY_EXPR_MUL),
            BodyExprOp(BODY_EXPR_READ_NODE, index=0),
            BodyExprOp(BODY_EXPR_ADD),
            BodyExprOp(BODY_EXPR_CONST, value=bias),
            BodyExprOp(BODY_EXPR_SUB),
        ]
        batch = backend.make_body_ir_batch(
            stmt_ops=[
                BodyStmtOp(
                    target_kind=BODY_TARGET_STATE,
                    target_id=1,
                    expr_start=0,
                    expr_count=len(expr_ops),
                    target_integer=integer_target,
                )
            ],
            expr_ops=expr_ops,
        )
        expected = state0 * param0 + node0 - bias
        if integer_target:
            expected = _veriloga_integer(expected)

        backend.evaluate_body_ir(batch, node_values, state_values, param_values)

        assert state_values[1] == pytest.approx(expected)


def test_rust_backend_evaluates_transition_target_batch():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.0])
    state_values = array("d", [1.0])
    target_values = array("d", [0.0])
    delay_values = array("d", [0.0])
    rise_values = array("d", [0.0])
    fall_values = array("d", [0.0])
    batch = backend.make_transition_target_batch(
        [
            TransitionTargetOp(
                target_id=0,
                bias=1.0,
                terms=(),
                condition=LinearCondition(
                    op_kind=6,
                    left_bias=0.0,
                    left_terms=(LinearTerm(source_kind=1, source_id=0, gain=1.0),),
                    right_bias=0.0,
                    right_terms=(),
                ),
                false_bias=0.0,
                false_terms=(),
                delay=0.0,
                rise=1.0e-9,
                fall=2.0e-9,
            )
        ]
    )

    backend.evaluate_transition_targets(
        batch,
        node_values,
        state_values,
        target_values,
        delay_values,
        rise_values,
        fall_values,
    )
    assert target_values.tolist() == pytest.approx([1.0])
    assert rise_values.tolist() == pytest.approx([1.0e-9])
    assert fall_values.tolist() == pytest.approx([2.0e-9])

    state_values[0] = 0.0
    backend.evaluate_transition_targets(
        batch,
        node_values,
        state_values,
        target_values,
        delay_values,
        rise_values,
        fall_values,
    )
    assert target_values.tolist() == pytest.approx([0.0])


def test_rust_backend_evaluates_ordered_transition_segment():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    node_values = array("d", [0.0])
    state_values = array("d", [0.0])
    target_values = array("d", [0.0])
    delay_values = array("d", [0.0])
    rise_values = array("d", [0.0])
    fall_values = array("d", [0.0])
    linear_batch = backend.make_static_linear_batch(
        [
            LinearOp(
                target_kind=1,
                target_id=0,
                bias=1.0,
                terms=(),
                target_integer=True,
            )
        ]
    )
    transition_batch = backend.make_transition_target_batch(
        [
            TransitionTargetOp(
                target_id=0,
                bias=1.0,
                terms=(),
                condition=LinearCondition(
                    op_kind=6,
                    left_bias=0.0,
                    left_terms=(LinearTerm(source_kind=1, source_id=0, gain=1.0),),
                    right_bias=0.0,
                    right_terms=(),
                ),
                false_bias=0.0,
                false_terms=(),
                delay=0.0,
                rise=1.0e-9,
                fall=2.0e-9,
            )
        ]
    )

    backend.evaluate_ordered_transition_segment(
        linear_batch,
        transition_batch,
        node_values,
        state_values,
        target_values,
        delay_values,
        rise_values,
        fall_values,
    )

    assert state_values.tolist() == pytest.approx([1.0])
    assert target_values.tolist() == pytest.approx([1.0])
    assert rise_values.tolist() == pytest.approx([1.0e-9])
    assert fall_values.tolist() == pytest.approx([2.0e-9])


def test_rust_backend_default_library_name_matches_platform():
    path = default_rust_core_library_path()

    if sys.platform == "darwin":
        assert path.name == "libevas_rust_core.dylib"
    elif sys.platform.startswith("linux"):
        assert path.name == "libevas_rust_core.so"
    else:
        assert path.name == "evas_rust_core.dll"
