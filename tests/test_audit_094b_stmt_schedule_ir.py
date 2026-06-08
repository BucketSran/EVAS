"""Audit 094b/094c tests for statement and event schedule IR lowering."""

from __future__ import annotations

from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.schedule_ir import emit_event_python, lower_event
from evas.simulator.stmt_ir import (
    StatementLoweringContext,
    emit_python_statement,
    lower_stmt,
)


SAMPLE = """\
`include "disciplines.vams"
module sample(clk, vdd, vss, out);
    input voltage clk;
    input voltage vdd;
    input voltage vss;
    output voltage out;
    integer q = 0;
    analog begin
        @(initial_step) q = 0;
        @(cross(V(clk) - 0.45, +1)) begin
            if (q == 0) q = 1;
            else q = 0;
        end
        V(out) <+ V(vdd, vss) * transition(q ? 1.0 : 0.0, 0, 1n);
    end
endmodule
"""


def _compile_emitted_body(stmt_ir):
    lines = emit_python_statement(stmt_ir)
    source = "def lowered_body():\n" + "\n".join(lines) + "\n"
    compile(source, "<audit-094b-stmt>", "exec")
    return source


def test_statement_ir_lowers_and_emits_sample_body():
    module = parse(SAMPLE)
    stmt_ir = lower_stmt(module.analog_block.body)

    assert stmt_ir is not None
    source = _compile_emitted_body(stmt_ir)
    assert "event_due" in source
    assert "contribute" in source
    assert "set_var('q'" in source


def test_schedule_ir_lowers_cross_event():
    module = parse(SAMPLE)
    event_stmt = module.analog_block.body.statements[1]

    event_ir = lower_event(event_stmt.event)
    assert event_ir is not None
    emitted = emit_event_python(event_ir)
    compile(emitted, "<audit-094c-event>", "eval")
    assert "CROSS" in emitted


def test_statement_ir_rejects_unknown_system_task():
    module = parse(
        """\
`include "disciplines.vams"
module bad(clk);
    input voltage clk;
    analog begin
        $unsupported_task(1);
    end
endmodule
"""
    )

    assert lower_stmt(module.analog_block.body) is None


def test_release_generic_candidate_statement_coverage_and_emit_compile():
    repo_root = Path(__file__).resolve().parents[2]
    tasks_root = repo_root / "behavioral-veriloga-eval" / "benchmark-vabench-release-v1" / "tasks"
    if not tasks_root.exists():
        pytest.skip(f"release benchmark tree missing: {tasks_root}")

    ctx = StatementLoweringContext.veriloga_body()
    candidate_count = 0
    failures = []

    for path in sorted(tasks_root.rglob("*.va")):
        try:
            module = parse(path.read_text())
            model_cls = compile_module(module)
        except Exception:
            continue
        candidates = getattr(model_cls, "_whole_segment_candidates", ()) or ()
        if not any(
            candidate and candidate[0] == "generic_event_state_transition_v1"
            for candidate in candidates
        ):
            continue

        candidate_count += 1
        stmt_ir = lower_stmt(module.analog_block.body, ctx)
        if stmt_ir is None:
            failures.append((path, "lower_stmt_none"))
            continue
        try:
            _compile_emitted_body(stmt_ir)
        except SyntaxError as exc:
            failures.append((path, str(exc)))

    assert candidate_count >= 234
    assert failures == []
