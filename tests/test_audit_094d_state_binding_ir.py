"""Audit 094d tests for state/parameter/port binding IR."""

from __future__ import annotations

from pathlib import Path

import pytest

from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.expr_ir import (
    BindingTableIR,
    LoweringContext,
    SYMBOL_PARAMETER,
    SYMBOL_PORT,
    SYMBOL_STATE_ARRAY,
    SYMBOL_STATE_SCALAR,
    build_state_binding_ir,
    iter_exprs_from_statement,
    iter_identifier_names,
    lower_expr,
)


SAMPLE = """\
`include "disciplines.vams"
module binding_sample(clk, out);
    input voltage clk;
    output voltage out;
    parameter integer width = 4;
    parameter real vth = 0.45;
    integer q = 0;
    real acc = 0.0;
    integer bits[3:0];
    analog begin
        @(cross(V(clk) - vth, +1)) begin
            q = q + 1;
            bits[0] = q;
        end
        V(out) <+ q > width ? 1.0 : 0.0;
    end
endmodule
"""


def test_state_binding_ir_assigns_stable_symbol_kinds():
    table = build_state_binding_ir(parse(SAMPLE))

    assert isinstance(table, BindingTableIR)
    assert table.resolve("vth").kind == SYMBOL_PARAMETER
    assert table.resolve("width").integer is True
    assert table.resolve("clk").kind == SYMBOL_PORT
    assert table.resolve("q").kind == SYMBOL_STATE_SCALAR
    assert table.resolve("q").integer is True
    assert table.resolve("acc").slot == 1
    assert table.resolve("bits").kind == SYMBOL_STATE_ARRAY
    assert table.resolve("bits").lo == 0
    assert table.resolve("bits").hi == 3


def test_release_generic_candidate_identifier_bindings_resolve():
    repo_root = Path(__file__).resolve().parents[2]
    tasks_root = repo_root / "behavioral-veriloga-eval" / "benchmark-vabench-release-v1" / "tasks"
    if not tasks_root.exists():
        pytest.skip(f"release benchmark tree missing: {tasks_root}")

    ctx = LoweringContext.veriloga_body()
    candidate_count = 0
    checked_identifiers = 0
    unresolved = []

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
        table = build_state_binding_ir(module)
        for expr in iter_exprs_from_statement(module.analog_block.body):
            expr_ir = lower_expr(expr, ctx)
            if expr_ir is None:
                unresolved.append((path, "<lowering_failed>", repr(expr)))
                continue
            for name in iter_identifier_names(expr_ir):
                checked_identifiers += 1
                if table.resolve(name) is None:
                    unresolved.append((path, name, type(expr).__name__))

    assert candidate_count >= 234
    assert checked_identifiers >= 7000
    assert unresolved == []
