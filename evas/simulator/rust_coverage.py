"""Release/model-level Rust coverage audit helpers.

This module is intentionally read-only: it compiles Verilog-A models and
summarizes which Rust paths the current compiler can expose.  It does not run
simulation and it is not paper-facing speed evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.event_transition_plan import (
    EVENT_TRANSITION_PROFILE_SUPPORT,
)
from evas.simulator.rust_program import build_source_record_rust_program


@dataclass(frozen=True)
class RustCoverageRow:
    path: str
    compile_ok: bool
    compile_error: str = ""
    rust_body_ir_candidate: bool = False
    rust_body_ir_rejection_reason: str = ""
    rust_body_ir_rejection_tags: Sequence[str] = ()
    rust_body_ir_stmt_ops: int = 0
    rust_body_ir_expr_ops: int = 0
    rust_body_ir_node_slots: int = 0
    rust_body_ir_state_slots: int = 0
    rust_body_ir_param_slots: int = 0
    static_linear_ops: int = 0
    whole_segment_kinds: Sequence[str] = ()
    event_transition_plan_profiles: Sequence[str] = ()
    event_transition_plan_rejection_reasons: Mapping[str, str] = field(
        default_factory=dict
    )
    event_transition_plan_blocker_tags: Mapping[str, Sequence[str]] = field(
        default_factory=dict
    )
    event_transition_plan_event_count: int = 0
    event_transition_plan_due_trigger_count: int = 0
    event_transition_plan_output_write_count: int = 0
    event_transition_plan_transition_count: int = 0
    event_transition_plan_state_assignment_count: int = 0
    event_transition_plan_side_effect_count: int = 0
    event_transition_plan_control_flow_count: int = 0
    rust_sim_program_candidate: bool = False
    rust_sim_program_rejection_reasons: Sequence[str] = ()
    rust_sim_program_node_count: int = 0
    rust_sim_program_state_count: int = 0
    rust_sim_program_source_count: int = 0
    rust_sim_program_record_count: int = 0
    rust_sim_program_continuous_linear_ops: int = 0


@dataclass(frozen=True)
class RustCoverageSummary:
    total_files: int
    compile_ok: int
    compile_failed: int
    rust_body_ir_candidates: int
    static_linear_candidates: int
    rust_sim_program_candidates: int
    whole_segment_candidates: int
    rows: Sequence[RustCoverageRow]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rows"] = [asdict(row) for row in self.rows]
        return data


def audit_veriloga_paths(paths: Iterable[Path]) -> RustCoverageSummary:
    rows: List[RustCoverageRow] = []
    for path in sorted(Path(p) for p in paths):
        try:
            module = parse(path.read_text())
            model_cls = compile_module(module)
        except Exception as exc:
            rows.append(
                RustCoverageRow(
                    path=str(path),
                    compile_ok=False,
                    compile_error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        whole_segment_kinds = tuple(
            sorted(
                str(candidate[0])
                for candidate in (getattr(model_cls, "_whole_segment_candidates", ()) or ())
                if candidate
            )
        )
        body_stmt_ops = tuple(getattr(model_cls, "_rust_body_ir_stmt_ops", ()) or ())
        body_expr_ops = tuple(getattr(model_cls, "_rust_body_ir_expr_ops", ()) or ())
        node_names = tuple(getattr(model_cls, "_rust_body_ir_node_names", ()) or ())
        rust_sim_report = build_source_record_rust_program(
            sources=(),
            recorded_signals=("__rustsim_audit__",),
            models=(model_cls(),),
        )
        rust_sim_program = rust_sim_report.program
        rows.append(
            RustCoverageRow(
                path=str(path),
                compile_ok=True,
                rust_body_ir_candidate=bool(body_stmt_ops),
                rust_body_ir_rejection_reason=str(
                    getattr(model_cls, "_rust_body_ir_rejection_reason", "")
                ),
                rust_body_ir_rejection_tags=tuple(
                    str(tag)
                    for tag in (
                        getattr(model_cls, "_rust_body_ir_rejection_tags", ()) or ()
                    )
                ),
                rust_body_ir_stmt_ops=len(body_stmt_ops),
                rust_body_ir_expr_ops=len(body_expr_ops),
                rust_body_ir_node_slots=len(node_names),
                rust_body_ir_state_slots=len(
                    tuple(getattr(model_cls, "_rust_body_ir_state_names", ()) or ())
                ),
                rust_body_ir_param_slots=len(
                    tuple(getattr(model_cls, "_rust_body_ir_param_names", ()) or ())
                ),
                static_linear_ops=len(
                    tuple(getattr(model_cls, "_evaluate_ir_static_linear_ops", ()) or ())
                ),
                whole_segment_kinds=whole_segment_kinds,
                event_transition_plan_profiles=tuple(
                    getattr(model_cls, "_event_transition_plan_profiles", ()) or ()
                ),
                event_transition_plan_rejection_reasons=dict(
                    getattr(
                        model_cls,
                        "_event_transition_plan_rejection_reasons",
                        {},
                    )
                    or {}
                ),
                event_transition_plan_blocker_tags=dict(
                    getattr(model_cls, "_event_transition_plan_blocker_tags", {}) or {}
                ),
                event_transition_plan_event_count=int(
                    getattr(model_cls, "_event_transition_plan_event_count", 0) or 0
                ),
                event_transition_plan_due_trigger_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_due_trigger_count",
                        0,
                    )
                    or 0
                ),
                event_transition_plan_output_write_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_output_write_count",
                        0,
                    )
                    or 0
                ),
                event_transition_plan_transition_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_transition_count",
                        0,
                    )
                    or 0
                ),
                event_transition_plan_state_assignment_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_state_assignment_count",
                        0,
                    )
                    or 0
                ),
                event_transition_plan_side_effect_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_side_effect_count",
                        0,
                    )
                    or 0
                ),
                event_transition_plan_control_flow_count=int(
                    getattr(
                        model_cls,
                        "_event_transition_plan_control_flow_count",
                        0,
                    )
                    or 0
                ),
                rust_sim_program_candidate=bool(rust_sim_report.supported),
                rust_sim_program_rejection_reasons=tuple(rust_sim_report.reasons),
                rust_sim_program_node_count=(
                    int(rust_sim_program.node_count)
                    if rust_sim_program is not None
                    else 0
                ),
                rust_sim_program_state_count=(
                    len(rust_sim_program.states)
                    if rust_sim_program is not None
                    else 0
                ),
                rust_sim_program_source_count=(
                    len(rust_sim_program.sources)
                    if rust_sim_program is not None
                    else 0
                ),
                rust_sim_program_record_count=(
                    len(rust_sim_program.records)
                    if rust_sim_program is not None
                    else 0
                ),
                rust_sim_program_continuous_linear_ops=(
                    len(rust_sim_program.continuous_linear_ops)
                    if rust_sim_program is not None
                    else 0
                ),
            )
        )

    compile_ok = sum(1 for row in rows if row.compile_ok)
    return RustCoverageSummary(
        total_files=len(rows),
        compile_ok=compile_ok,
        compile_failed=len(rows) - compile_ok,
        rust_body_ir_candidates=sum(1 for row in rows if row.rust_body_ir_candidate),
        static_linear_candidates=sum(1 for row in rows if row.static_linear_ops > 0),
        rust_sim_program_candidates=sum(
            1 for row in rows if row.rust_sim_program_candidate
        ),
        whole_segment_candidates=sum(1 for row in rows if row.whole_segment_kinds),
        rows=tuple(rows),
    )


def discover_veriloga_files(root: Path) -> Sequence[Path]:
    return tuple(sorted(Path(root).rglob("*.va")))


def estimate_event_transition_profiles(
    rows: Iterable[RustCoverageRow],
) -> dict:
    """Estimate how many current rejects a future ordered segment could cover.

    This is a static planning estimate, not proof of correctness or speed.  A
    row is counted for a profile only when every current rejection tag belongs
    to that profile's planned support set.
    """

    rows_tuple = tuple(rows)
    compile_ok_rejects = tuple(
        row for row in rows_tuple if row.compile_ok and not row.rust_body_ir_candidate
    )
    estimates = {}
    for profile_name, supported_tags in EVENT_TRANSITION_PROFILE_SUPPORT.items():
        candidate_paths = []
        blocker_counts: dict[str, int] = {}
        for row in compile_ok_rejects:
            tags = set(row.rust_body_ir_rejection_tags or ())
            unsupported = tags - set(supported_tags)
            if not unsupported:
                candidate_paths.append(row.path)
                continue
            for tag in unsupported:
                blocker_counts[tag] = blocker_counts.get(tag, 0) + 1
        estimates[profile_name] = {
            "supported_tags": tuple(sorted(supported_tags)),
            "candidate_count": len(candidate_paths),
            "denominator": len(compile_ok_rejects),
            "candidate_paths": tuple(candidate_paths),
            "blocker_counts": dict(
                sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }
    return estimates


def estimate_event_transition_plan_profiles(
    rows: Iterable[RustCoverageRow],
) -> dict:
    """Count rows accepted by the source-order event/output/transition planner."""

    rows_tuple = tuple(rows)
    compile_ok_rows = tuple(
        row for row in rows_tuple if row.compile_ok and not row.rust_body_ir_candidate
    )
    estimates = {}
    for profile_name in EVENT_TRANSITION_PROFILE_SUPPORT:
        candidate_paths = [
            row.path
            for row in compile_ok_rows
            if profile_name in set(row.event_transition_plan_profiles or ())
        ]
        rejection_counts: dict[str, int] = {}
        blocker_counts: dict[str, int] = {}
        for row in compile_ok_rows:
            if row.path in set(candidate_paths):
                continue
            reason = row.event_transition_plan_rejection_reasons.get(
                profile_name, "unknown"
            )
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            for tag in row.event_transition_plan_blocker_tags.get(profile_name, ()):
                blocker_counts[str(tag)] = blocker_counts.get(str(tag), 0) + 1
        estimates[profile_name] = {
            "candidate_count": len(candidate_paths),
            "denominator": len(compile_ok_rows),
            "candidate_paths": tuple(candidate_paths),
            "rejection_counts": dict(
                sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "blocker_counts": dict(
                sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }
    return estimates
