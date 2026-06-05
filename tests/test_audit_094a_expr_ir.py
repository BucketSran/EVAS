"""Audit 094a tests for general Verilog-A expression IR lowering."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from evas.compiler.ast_nodes import (
    BinaryExpr,
    BranchAccess,
    FunctionCall,
    Identifier,
    NumberLiteral,
    UnaryExpr,
)
from evas.compiler.parser import parse
from evas.simulator.backend import compile_module
from evas.simulator.expr_ir import (
    LoweringContext,
    emit_python,
    iter_exprs_from_statement,
    lower_expr,
)


def _eval_env():
    values = {"vin": 0.25, "gain": -3.0, "i": 2.0, "bus[2]": 0.7}
    arrays = {"cfg": {2: 5.0}}

    def var(name):
        return values[name]

    def voltage(name):
        return values[name]

    def node_ref(name, index, index2=None):
        if index2 is None:
            return f"{name}[{index}]"
        return f"{name}[{index}][{index2}]"

    def array_value(name, index):
        return arrays[name][index]

    return {
        "array_value": array_value,
        "float": float,
        "math": math,
        "node_ref": node_ref,
        "time_value": 0.0,
        "temperature_c": 27.0,
        "var": var,
        "voltage": voltage,
    }


def test_lower_emit_python_expression_round_trip_runs():
    expr = BinaryExpr(
        "+",
        BinaryExpr(
            "*",
            BranchAccess("V", "vin"),
            FunctionCall("abs", [Identifier("gain")]),
        ),
        BranchAccess("V", "bus", node1_index=Identifier("i")),
    )

    ir = lower_expr(expr, LoweringContext.veriloga_body())
    assert ir is not None
    emitted = emit_python(ir)

    code = compile(emitted, "<audit-094a-expr>", "eval")
    value = eval(code, _eval_env())
    assert value == pytest.approx(0.25 * 3.0 + 0.7)


def test_default_context_rejects_unknown_function():
    expr = FunctionCall("not_a_veriloga_math_function", [NumberLiteral(1.0)])

    assert lower_expr(expr) is None


def test_default_context_allows_pure_math_only():
    assert lower_expr(FunctionCall("sqrt", [NumberLiteral(4.0)])) is not None
    assert lower_expr(FunctionCall("transition", [NumberLiteral(1.0)])) is None


def test_body_context_lowers_stateful_and_system_expressions():
    ctx = LoweringContext.veriloga_body()

    assert lower_expr(FunctionCall("transition", [NumberLiteral(1.0)]), ctx) is not None
    assert lower_expr(FunctionCall("$fopen", [NumberLiteral(1.0)]), ctx) is not None


def test_candidate_release_expression_coverage_and_emit_compile():
    repo_root = Path(__file__).resolve().parents[2]
    tasks_root = repo_root / "behavioral-veriloga-eval" / "benchmark-vabench-release-v1" / "tasks"
    if not tasks_root.exists():
        pytest.skip(f"release benchmark tree missing: {tasks_root}")

    ctx = LoweringContext.veriloga_body()
    candidate_count = 0
    expr_count = 0
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
        for expr in iter_exprs_from_statement(module.analog_block.body):
            expr_count += 1
            ir = lower_expr(expr, ctx)
            if ir is None:
                failures.append((path, type(expr).__name__, repr(expr)))
                continue
            emitted = emit_python(ir)
            try:
                compile(emitted, f"<audit-094a:{path.name}>", "eval")
            except SyntaxError as exc:
                failures.append((path, type(expr).__name__, emitted, str(exc)))

    assert candidate_count >= 234
    assert expr_count >= 8000
    assert failures == []


def test_unary_and_integer_operator_emit_compile():
    expr = BinaryExpr(
        "&&",
        UnaryExpr("!", Identifier("flag")),
        BinaryExpr("^", Identifier("a"), NumberLiteral(3.0)),
    )
    ir = lower_expr(expr, LoweringContext.veriloga_body())
    assert ir is not None

    compile(emit_python(ir), "<audit-094a-int-ops>", "eval")
