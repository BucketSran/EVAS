"""Whole-segment candidate ABI contract helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Tuple

CANDIDATE_SCHEMA_VERSION = "evas-whole-segment-candidate-contract.v1"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str


@dataclass(frozen=True)
class CandidateContract:
    schema_version: str
    kind: str
    valid: bool
    errors: Tuple[str, ...]
    arity: int
    expected_arity: int
    field_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _field(name: str, kind: str = "str") -> FieldSpec:
    return FieldSpec(name=name, kind=kind)


WHOLE_SEGMENT_CANDIDATE_SCHEMAS: Dict[str, Tuple[FieldSpec, ...]] = {
    "cross_scalar_lfsr_transition_bus_v1": (
        _field("kind", "kind"),
        _field("clock_port"),
        _field("reset_port"),
        _field("enable_port"),
        _field("threshold_param"),
        _field("seed_param"),
        _field("vdd_param"),
        _field("td_param"),
        _field("trf_param"),
        _field("state_names", "str_tuple"),
        _field("taps", "int_tuple"),
        _field("shift_sources", "int_tuple"),
        _field("output_ports", "str_tuple"),
        _field("output_bits", "int_tuple"),
        _field("zero_guard_index", "int"),
    ),
    "gain_timer_reduction_v1": (
        _field("kind", "kind"),
        _field("vdd_port"),
        _field("vss_port"),
        _field("vinp_port"),
        _field("vinn_port"),
        _field("voutp_port"),
        _field("voutn_port"),
        _field("gain_port"),
        _field("valid_port"),
        _field("sample_period_param"),
        _field("start_time_param"),
        _field("gain_scale_param"),
        _field("min_input_span_param"),
        _field("tedge_param"),
    ),
    "cmp_delay_log_transition_v1": (
        _field("kind", "kind"),
        _field("clk_port"),
        _field("vinn_port"),
        _field("vinp_port"),
        _field("outn_port"),
        _field("outp_port"),
        _field("vss_port"),
        _field("vdd_port"),
        _field("voffset_param"),
        _field("tau_param"),
        _field("td0_param"),
        _field("tdmin_param"),
        _field("tdmax_param"),
        _field("tedge_state"),
    ),
    "edge_interval_timer_v1": (
        _field("kind", "kind"),
        _field("clk1_port"),
        _field("clk2_port"),
        _field("delay_port"),
        _field("edge_vth_param"),
    ),
    "weighted_sar_adc_v1": (
        _field("kind", "kind"),
        _field("vin_port"),
        _field("clks_port"),
        _field("rst_port"),
        _field("dout_ports", "str_tuple"),
        _field("bit_index_port"),
        _field("trial_code_port"),
        _field("trial_vdac_port"),
        _field("cmp_decision_port"),
        _field("conv_done_port"),
        _field("vin_sample_port"),
        _field("vdd_param"),
        _field("vth_param"),
        _field("width", "positive_int"),
    ),
    "weighted_dac_v1": (
        _field("kind", "kind"),
        _field("din_ports", "str_tuple"),
        _field("output_port"),
        _field("vdd_param"),
        _field("threshold_param"),
    ),
    "sample_hold_rising_v1": (
        _field("kind", "kind"),
        _field("vin_port"),
        _field("clk_port"),
        _field("vdd_port"),
        _field("vss_port"),
        _field("rst_port"),
        _field("vout_port"),
        _field("tr_param"),
    ),
    "ref_step_clock_v1": (
        _field("kind", "kind"),
        _field("vdd_port"),
        _field("vss_port"),
        _field("clk_port"),
        _field("period_pre_param"),
        _field("period_post_param"),
        _field("t_switch_param"),
        _field("tedge_param"),
    ),
    "cppll_timer_v1": (
        _field("kind", "kind"),
        _field("vdd_port"),
        _field("vss_port"),
        _field("ref_port"),
        _field("fb_port"),
        _field("dco_port"),
        _field("vctrl_port"),
        _field("lock_port"),
        _field("div_ratio_param"),
        _field("f_center_param"),
        _field("kvco_param"),
        _field("f_min_param"),
        _field("f_max_param"),
        _field("kp_param"),
        _field("ki_param"),
        _field("integ_min_param"),
        _field("integ_max_param"),
        _field("vctrl_init_param"),
        _field("tedge_param"),
        _field("lock_tol_param"),
        _field("lock_count_target_param"),
    ),
    # Audit 091b: generic event-driven state machine + transition outputs.
    # This is the catch-all metadata kind that captures the common shape
    # accounting for ~91% of stuck models in the rust coverage manifest.
    "generic_event_state_transition_v1": (
        _field("kind", "kind"),
        _field("module"),
        _field("event_count", "positive_int"),
        _field("transition_output_count", "positive_int"),
        _field("target_state_scalars", "str_tuple"),
        _field("transition_output_nodes", "str_tuple"),
    ),
}


def whole_segment_candidate_kinds() -> Tuple[str, ...]:
    return tuple(sorted(WHOLE_SEGMENT_CANDIDATE_SCHEMAS))


def whole_segment_candidate_field_names(kind: str) -> Tuple[str, ...]:
    return tuple(spec.name for spec in WHOLE_SEGMENT_CANDIDATE_SCHEMAS.get(kind, ()))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, str) and item for item in value)


def _is_int_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and all(_is_int(item) for item in value)


FIELD_VALIDATORS: Dict[str, Callable[[Any], bool]] = {
    "kind": lambda value: isinstance(value, str) and bool(value),
    "str": lambda value: isinstance(value, str) and bool(value),
    "int": _is_int,
    "positive_int": lambda value: _is_int(value) and value > 0,
    "str_tuple": _is_str_tuple,
    "int_tuple": _is_int_tuple,
}


def validate_whole_segment_candidate(candidate: Any) -> CandidateContract:
    if not isinstance(candidate, tuple) or not candidate:
        return CandidateContract(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            kind="",
            valid=False,
            errors=("candidate_not_nonempty_tuple",),
            arity=0,
            expected_arity=0,
            field_names=(),
        )

    kind = candidate[0] if isinstance(candidate[0], str) else ""
    schema = WHOLE_SEGMENT_CANDIDATE_SCHEMAS.get(kind)
    if schema is None:
        return CandidateContract(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            kind=str(candidate[0]),
            valid=False,
            errors=("unknown_kind",),
            arity=len(candidate),
            expected_arity=0,
            field_names=(),
        )

    errors: list[str] = []
    if len(candidate) != len(schema):
        errors.append(f"arity:{len(candidate)}!={len(schema)}")

    for idx, spec in enumerate(schema[: len(candidate)]):
        validator = FIELD_VALIDATORS[spec.kind]
        if not validator(candidate[idx]):
            errors.append(f"field:{spec.name}:{spec.kind}")

    if candidate[0] != kind:
        errors.append("kind_mismatch")

    errors.extend(_cross_field_errors(kind, candidate))
    return CandidateContract(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        kind=kind,
        valid=not errors,
        errors=tuple(errors),
        arity=len(candidate),
        expected_arity=len(schema),
        field_names=tuple(spec.name for spec in schema),
    )


def _cross_field_errors(kind: str, candidate: tuple) -> Tuple[str, ...]:
    errors: list[str] = []
    if kind == "cross_scalar_lfsr_transition_bus_v1" and len(candidate) == 15:
        state_names = candidate[9]
        taps = candidate[10]
        shift_sources = candidate[11]
        output_ports = candidate[12]
        output_bits = candidate[13]
        zero_guard_index = candidate[14]
        if isinstance(state_names, tuple) and isinstance(shift_sources, tuple):
            if len(state_names) == 0:
                errors.append("state_names_empty")
            if len(state_names) != len(shift_sources):
                errors.append("state_shift_width_mismatch")
        if isinstance(output_ports, tuple) and isinstance(output_bits, tuple):
            if len(output_ports) != len(output_bits):
                errors.append("output_width_mismatch")
        if isinstance(taps, tuple) and len(taps) == 0:
            errors.append("taps_empty")
        if _is_int(zero_guard_index) and isinstance(state_names, tuple):
            if not 0 <= zero_guard_index < len(state_names):
                errors.append("zero_guard_index_out_of_range")
    elif kind == "weighted_sar_adc_v1" and len(candidate) == 14:
        dout_ports = candidate[4]
        width = candidate[13]
        if isinstance(dout_ports, tuple) and _is_int(width) and len(dout_ports) != width:
            errors.append("dout_width_mismatch")
    elif kind == "weighted_dac_v1" and len(candidate) == 5:
        din_ports = candidate[1]
        if isinstance(din_ports, tuple) and len(din_ports) < 3:
            errors.append("din_width_lt_3")
    return tuple(errors)
