"""Unit tests for evas.compiler — Lexer and Parser.

Lexer tests (TestLexer):
  - Keywords, identifiers, system tasks
  - Numbers: integer, float, scientific notation, SI suffixes
  - Strings, operators (single- and two-char), comments, attributes
  - Line-number tracking, EOF handling

Parser tests (TestParser):
  - Module structure, port lists (non-ANSI / ANSI)
  - Parameter and variable declarations
  - Contributions, assignments
  - Control flow: if/else, for
  - Event statements: @cross, @above, @initial_step, combined
  - Expressions: binary, unary, ternary, function calls, array, method
"""
import math
import pytest

from evas.compiler.lexer import tokenize, TokenType, LexerError
from evas.compiler.parser import parse, ParseError
from evas.compiler.ast_nodes import (
    Assignment, AnalogBlock, BinaryExpr, Block, BranchAccess,
    CaseStatement, Contribution, CombinedEvent, Direction, EventExpr,
    EventStatement, EventType, ForStatement, FunctionCall, Identifier,
    IfStatement, MethodCall, Module, NumberLiteral, ParamType,
    ParameterDecl, StringLiteral, SystemTask, TernaryExpr, UnaryExpr,
    VariableDecl, ArrayAccess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toks(src: str):
    """Return all tokens (including EOF)."""
    return tokenize(src)

def _types(src: str):
    """Return token types, excluding the trailing EOF."""
    return [t.type for t in tokenize(src) if t.type != TokenType.EOF]

def _vals(src: str):
    """Return token values, excluding the trailing EOF."""
    return [t.value for t in tokenize(src) if t.type != TokenType.EOF]

def _first(src: str):
    """Return the first token."""
    return tokenize(src)[0]

def _parse(src: str) -> Module:
    return parse(src)

def _wrap(body: str) -> str:
    """Wrap body in a minimal module for statement-level tests."""
    return f"module m(a, b); electrical a, b; analog begin {body} end endmodule"

def _stmts(body: str):
    """Return the list of analog statements for a minimal module."""
    m = parse(_wrap(body))
    return m.analog_block.body.statements


# ===========================================================================
# Lexer
# ===========================================================================

class TestLexerKeywords:

    def test_module(self):
        assert _first("module").type == TokenType.MODULE

    def test_endmodule(self):
        assert _first("endmodule").type == TokenType.ENDMODULE

    def test_analog(self):
        assert _first("analog").type == TokenType.ANALOG

    def test_begin_end(self):
        types = _types("begin end")
        assert types == [TokenType.BEGIN, TokenType.END]

    def test_parameter(self):
        assert _first("parameter").type == TokenType.PARAMETER

    def test_real_integer_genvar(self):
        types = _types("real integer genvar")
        assert types == [TokenType.REAL, TokenType.INTEGER, TokenType.GENVAR]

    def test_electrical(self):
        assert _first("electrical").type == TokenType.ELECTRICAL

    def test_input_output_inout(self):
        types = _types("input output inout")
        assert types == [TokenType.INPUT, TokenType.OUTPUT, TokenType.INOUT]

    def test_if_else_for(self):
        types = _types("if else for")
        assert types == [TokenType.IF, TokenType.ELSE, TokenType.FOR]

    def test_or_keyword(self):
        assert _first("or").type == TokenType.OR

    def test_keyword_vs_identifier(self):
        # 'order' starts with 'or' but is not a keyword
        tok = _first("order")
        assert tok.type == TokenType.IDENT
        assert tok.value == "order"


class TestLexerIdentifiers:

    def test_simple_ident(self):
        tok = _first("myvar")
        assert tok.type == TokenType.IDENT
        assert tok.value == "myvar"

    def test_ident_with_underscore(self):
        tok = _first("my_var_1")
        assert tok.type == TokenType.IDENT
        assert tok.value == "my_var_1"

    def test_ident_starts_with_underscore(self):
        tok = _first("_priv")
        assert tok.type == TokenType.IDENT
        assert tok.value == "_priv"

    def test_system_task(self):
        tok = _first("$strobe")
        assert tok.type == TokenType.IDENT
        assert tok.value == "$strobe"


class TestLexerNumbers:

    def test_integer(self):
        tok = _first("42")
        assert tok.type == TokenType.NUMBER
        assert float(tok.value) == pytest.approx(42.0)

    def test_float(self):
        tok = _first("3.14")
        assert tok.type == TokenType.NUMBER
        assert float(tok.value) == pytest.approx(3.14)

    def test_scientific_positive_exp(self):
        tok = _first("1e9")
        assert float(tok.value) == pytest.approx(1e9)

    def test_scientific_negative_exp(self):
        tok = _first("1e-9")
        assert float(tok.value) == pytest.approx(1e-9)

    def test_scientific_capital_E(self):
        tok = _first("2.5E6")
        assert float(tok.value) == pytest.approx(2.5e6)

    def test_leading_dot(self):
        tok = _first(".5")
        assert float(tok.value) == pytest.approx(0.5)

    # SI suffixes

    def test_si_nano(self):
        tok = _first("1n;")
        assert float(tok.value) == pytest.approx(1e-9)

    def test_si_pico(self):
        tok = _first("5p;")
        assert float(tok.value) == pytest.approx(5e-12)

    def test_si_micro(self):
        tok = _first("2u;")
        assert float(tok.value) == pytest.approx(2e-6)

    def test_si_milli(self):
        tok = _first("10m;")
        assert float(tok.value) == pytest.approx(10e-3)

    def test_si_kilo_lowercase(self):
        tok = _first("1k;")
        assert float(tok.value) == pytest.approx(1e3)

    def test_si_mega(self):
        tok = _first("1M;")
        assert float(tok.value) == pytest.approx(1e6)

    def test_si_giga(self):
        tok = _first("2G;")
        assert float(tok.value) == pytest.approx(2e9)

    def test_si_femto(self):
        tok = _first("20f;")
        assert float(tok.value) == pytest.approx(20e-15)

    def test_si_atto(self):
        tok = _first("1a;")
        assert float(tok.value) == pytest.approx(1e-18)

    def test_si_float_with_suffix(self):
        tok = _first("0.5p;")
        assert float(tok.value) == pytest.approx(0.5e-12)

    def test_no_si_when_followed_by_letter(self):
        # '1ns' — 'n' followed by 's' (alpha, not 'e') → no SI
        toks = [t for t in _toks("1ns") if t.type != TokenType.EOF]
        assert toks[0].type == TokenType.NUMBER
        assert float(toks[0].value) == pytest.approx(1.0)
        assert toks[1].type == TokenType.IDENT
        assert toks[1].value == "ns"


class TestLexerStrings:

    def test_simple_string(self):
        tok = _first('"hello"')
        assert tok.type == TokenType.STRING
        assert tok.value == "hello"

    def test_empty_string(self):
        tok = _first('""')
        assert tok.type == TokenType.STRING
        assert tok.value == ""

    def test_string_with_escape(self):
        tok = _first(r'"a\"b"')
        assert tok.type == TokenType.STRING
        assert tok.value == 'a"b'

    def test_string_with_spaces(self):
        tok = _first('"hello world"')
        assert tok.value == "hello world"


class TestLexerOperators:

    def test_two_char_operators(self):
        mapping = {
            "<+": TokenType.CONTRIB,
            "<<": TokenType.LSHIFT,
            ">>": TokenType.RSHIFT,
            ">=": TokenType.GE,
            "<=": TokenType.LE,
            "==": TokenType.EQ,
            "!=": TokenType.NE,
            "&&": TokenType.LAND,
            "||": TokenType.LOR,
        }
        for src, expected in mapping.items():
            assert _first(src).type == expected, f"Failed for {src!r}"

    def test_lt_not_confused_with_contrib(self):
        # '<' alone should be LT, not CONTRIB
        types = _types("a < b")
        assert TokenType.LT in types
        assert TokenType.CONTRIB not in types

    def test_single_char_operators(self):
        mapping = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "^": TokenType.CARET,
            "&": TokenType.AMP,
            "|": TokenType.PIPE,
            "~": TokenType.TILDE,
            "!": TokenType.BANG,
            "=": TokenType.ASSIGN,
            "?": TokenType.QUESTION,
            ":": TokenType.COLON,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ";": TokenType.SEMI,
            ",": TokenType.COMMA,
            "@": TokenType.AT,
        }
        for ch, expected in mapping.items():
            assert _first(ch).type == expected, f"Failed for {ch!r}"

    def test_attribute_begin_end(self):
        types = _types("(* *)")
        assert types == [TokenType.ATTR_BEGIN, TokenType.ATTR_END]


class TestLexerComments:

    def test_line_comment_skipped(self):
        types = _types("a // this is a comment\nb")
        assert types == [TokenType.IDENT, TokenType.IDENT]

    def test_block_comment_skipped(self):
        types = _types("a /* block */ b")
        assert types == [TokenType.IDENT, TokenType.IDENT]

    def test_multiline_block_comment(self):
        src = "a /* line1\nline2\nline3 */ b"
        types = _types(src)
        assert types == [TokenType.IDENT, TokenType.IDENT]

    def test_block_comment_tracks_lines(self):
        src = "/* line1\nline2 */\nx"
        toks = _toks(src)
        x_tok = next(t for t in toks if t.type == TokenType.IDENT and t.value == "x")
        assert x_tok.line == 3


class TestLexerLineTracking:

    def test_first_line_is_1(self):
        tok = _first("x")
        assert tok.line == 1

    def test_newline_increments_line(self):
        toks = _toks("a\nb")
        b_tok = next(t for t in toks if t.value == "b")
        assert b_tok.line == 2

    def test_col_resets_after_newline(self):
        toks = _toks("abc\nxy")
        xy_tok = next(t for t in toks if t.value == "xy")
        assert xy_tok.col == 1


class TestLexerEdgeCases:

    def test_empty_string_yields_only_eof(self):
        toks = _toks("")
        assert len(toks) == 1
        assert toks[0].type == TokenType.EOF

    def test_eof_always_last(self):
        toks = _toks("module")
        assert toks[-1].type == TokenType.EOF

    def test_backtick_directive_skipped(self):
        # Survived preprocessor directives should be silently skipped
        vals = _vals("`include x")
        assert "x" in vals


# ===========================================================================
# Parser
# ===========================================================================

class TestParserModule:

    def test_module_name(self):
        m = _parse("module foo(); endmodule")
        assert m.name == "foo"

    def test_empty_port_list(self):
        m = _parse("module foo(); endmodule")
        assert m.ports == []

    def test_non_ansi_ports(self):
        m = _parse("module buf(a, b); endmodule")
        assert "a" in m.ports
        assert "b" in m.ports

    def test_ansi_input_port(self):
        m = _parse("module inv(input electrical in, output electrical out); endmodule")
        names = [p.name for p in m.port_decls]
        assert "in" in names
        assert "out" in names
        inp = next(p for p in m.port_decls if p.name == "in")
        assert inp.direction == Direction.INPUT

    def test_ansi_output_port(self):
        m = _parse("module inv(input electrical in, output electrical out); endmodule")
        out = next(p for p in m.port_decls if p.name == "out")
        assert out.direction == Direction.OUTPUT

    def test_module_skips_preamble(self):
        # Nature/discipline definitions before 'module' should be silently skipped
        src = "nature Voltage; endnature module m(); endmodule"
        m = _parse(src)
        assert m.name == "m"

    def test_no_analog_block(self):
        m = _parse("module m(); endmodule")
        assert m.analog_block is None


class TestParserParameters:

    def test_real_parameter_default(self):
        m = _parse("module m(); parameter real VDD = 1.8; endmodule")
        assert len(m.parameters) == 1
        p = m.parameters[0]
        assert p.name == "VDD"
        assert p.param_type == ParamType.REAL
        assert isinstance(p.default_value, NumberLiteral)
        assert p.default_value.value == pytest.approx(1.8)

    def test_parameter_with_si_suffix(self):
        m = _parse("module m(); parameter real T = 1n; endmodule")
        p = m.parameters[0]
        assert p.default_value.value == pytest.approx(1e-9)

    def test_integer_parameter(self):
        m = _parse("module m(); parameter integer N = 4; endmodule")
        p = m.parameters[0]
        assert p.param_type == ParamType.INTEGER
        assert p.default_value.value == pytest.approx(4)

    def test_string_parameter(self):
        m = _parse('module m(); parameter CONF = "hello"; endmodule')
        p = m.parameters[0]
        assert p.param_type == ParamType.STRING
        assert isinstance(p.default_value, StringLiteral)
        assert p.default_value.value == "hello"

    def test_parameter_negative_default(self):
        m = _parse("module m(); parameter real X = -1.0; endmodule")
        p = m.parameters[0]
        assert p.default_value.value == pytest.approx(-1.0)

    def test_parameter_from_range(self):
        m = _parse("module m(); parameter real X = 0.5 from [0:1]; endmodule")
        p = m.parameters[0]
        assert p.range_lo is not None
        assert p.range_hi is not None
        assert p.range_lo_inclusive is True
        assert p.range_hi_inclusive is True

    def test_multiple_parameters(self):
        m = _parse("module m(); parameter real A = 1.0; parameter real B = 2.0; endmodule")
        assert len(m.parameters) == 2
        assert m.parameters[0].name == "A"
        assert m.parameters[1].name == "B"


class TestParserVariables:

    def test_real_variable(self):
        m = _parse("module m(); real x; endmodule")
        assert any(v.name == "x" for v in m.variables)

    def test_integer_variable(self):
        m = _parse("module m(); integer i; endmodule")
        v = next(v for v in m.variables if v.name == "i")
        assert v.var_type == ParamType.INTEGER

    def test_array_variable(self):
        m = _parse("module m(); integer arr[7:0]; endmodule")
        v = next(v for v in m.variables if v.name == "arr")
        assert v.is_array
        assert v.array_hi == 7
        assert v.array_lo == 0

    def test_variable_with_initializer(self):
        m = _parse("module m(); real x = 0.5; endmodule")
        v = next(v for v in m.variables if v.name == "x")
        assert v.init_values is not None
        assert v.init_values[0].value == pytest.approx(0.5)

    def test_array_initializer(self):
        m = _parse("module m(); integer arr[1:0] = {1, 0}; endmodule")
        v = next(v for v in m.variables if v.name == "arr")
        assert v.is_array
        assert v.init_values is not None
        assert len(v.init_values) == 2


class TestParserContributions:

    def test_single_node_contribution(self):
        stmts = _stmts("V(a) <+ 1.8;")
        assert len(stmts) == 1
        stmt = stmts[0]
        assert isinstance(stmt, Contribution)
        assert stmt.branch.access_type == "V"
        assert stmt.branch.node1 == "a"
        assert stmt.branch.node2 is None

    def test_differential_contribution(self):
        stmts = _stmts("V(a, b) <+ 0.9;")
        stmt = stmts[0]
        assert isinstance(stmt, Contribution)
        assert stmt.branch.node1 == "a"
        assert stmt.branch.node2 == "b"

    def test_contribution_expression(self):
        stmts = _stmts("V(a) <+ 1.0 + 0.5;")
        stmt = stmts[0]
        assert isinstance(stmt, Contribution)
        assert isinstance(stmt.expr, BinaryExpr)
        assert stmt.expr.op == "+"

    def test_contribution_with_transition(self):
        stmts = _stmts("V(a) <+ transition(1.0, 0.0, 1n, 1n);")
        stmt = stmts[0]
        assert isinstance(stmt, Contribution)
        assert isinstance(stmt.expr, FunctionCall)
        assert stmt.expr.name == "transition"
        assert len(stmt.expr.args) == 4


class TestParserAssignments:

    def test_simple_assignment(self):
        stmts = _stmts("x = 1.0;")
        stmt = stmts[0]
        assert isinstance(stmt, Assignment)
        assert isinstance(stmt.target, Identifier)
        assert stmt.target.name == "x"

    def test_array_assignment(self):
        stmts = _stmts("arr[2] = 1;")
        stmt = stmts[0]
        assert isinstance(stmt, Assignment)
        assert isinstance(stmt.target, ArrayAccess)
        assert stmt.target.name == "arr"


class TestParserIfStatement:

    def test_if_no_else(self):
        stmts = _stmts("if (x > 0) y = 1;")
        stmt = stmts[0]
        assert isinstance(stmt, IfStatement)
        assert stmt.else_body is None

    def test_if_with_else(self):
        stmts = _stmts("if (x > 0) y = 1; else y = 0;")
        stmt = stmts[0]
        assert isinstance(stmt, IfStatement)
        assert stmt.else_body is not None

    def test_if_condition_is_binary(self):
        stmts = _stmts("if (x > 0) y = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.cond, BinaryExpr)
        assert stmt.cond.op == ">"

    def test_nested_if(self):
        stmts = _stmts("if (a) if (b) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.then_body, IfStatement)

    def test_if_with_block(self):
        stmts = _stmts("if (x) begin y = 1; z = 2; end")
        stmt = stmts[0]
        assert isinstance(stmt.then_body, Block)
        assert len(stmt.then_body.statements) == 2


class TestParserForLoop:

    def test_for_loop_structure(self):
        stmts = _stmts("for (i = 0; i < 4; i = i + 1) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt, ForStatement)

    def test_for_init(self):
        stmts = _stmts("for (i = 0; i < 4; i = i + 1) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.init, Assignment)
        assert isinstance(stmt.init.target, Identifier)
        assert stmt.init.target.name == "i"

    def test_for_condition(self):
        stmts = _stmts("for (i = 0; i < 4; i = i + 1) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.cond, BinaryExpr)
        assert stmt.cond.op == "<"

    def test_for_body(self):
        stmts = _stmts("for (i = 0; i < 4; i = i + 1) begin x = i; end")
        stmt = stmts[0]
        assert isinstance(stmt.body, Block)


class TestParserEventStatements:

    def test_cross_event(self):
        stmts = _stmts("@(cross(V(a) - 0.45)) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt, EventStatement)
        assert isinstance(stmt.event, EventExpr)
        assert stmt.event.event_type == EventType.CROSS

    def test_cross_event_direction_rising(self):
        stmts = _stmts("@(cross(V(a) - 0.45, 1)) x = 1;")
        stmt = stmts[0]
        assert stmt.event.direction == 1

    def test_cross_event_direction_falling(self):
        stmts = _stmts("@(cross(V(a) - 0.45, -1)) x = 1;")
        stmt = stmts[0]
        assert stmt.event.direction == -1

    def test_above_event(self):
        stmts = _stmts("@(above(V(a) - 0.45)) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.event, EventExpr)
        assert stmt.event.event_type == EventType.ABOVE

    def test_initial_step_event(self):
        stmts = _stmts("@(initial_step) x = 0;")
        stmt = stmts[0]
        assert isinstance(stmt.event, EventExpr)
        assert stmt.event.event_type == EventType.INITIAL_STEP

    def test_combined_event_cross_or_initial_step(self):
        stmts = _stmts("@(cross(V(a) - 0.45) or initial_step) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.event, CombinedEvent)
        types = [e.event_type for e in stmt.event.events]
        assert EventType.CROSS in types
        assert EventType.INITIAL_STEP in types

    def test_event_body_is_assignment(self):
        stmts = _stmts("@(cross(V(a))) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.body, Assignment)

    def test_event_body_is_block(self):
        stmts = _stmts("@(cross(V(a))) begin x = 1; y = 2; end")
        stmt = stmts[0]
        assert isinstance(stmt.body, Block)

    def test_timer_event(self):
        stmts = _stmts("@(timer(10e-9)) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.event, EventExpr)
        assert stmt.event.event_type == EventType.TIMER
        assert len(stmt.event.args) == 1

    def test_final_step_event(self):
        stmts = _stmts("@(final_step) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.event, EventExpr)
        assert stmt.event.event_type == EventType.FINAL_STEP

    def test_combined_timer_or_initial_step(self):
        stmts = _stmts("@(initial_step or timer(10e-9)) x = 1;")
        stmt = stmts[0]
        assert isinstance(stmt.event, CombinedEvent)
        types = [e.event_type for e in stmt.event.events]
        assert EventType.INITIAL_STEP in types
        assert EventType.TIMER in types


class TestParserCaseStatement:

    def test_case_parses(self):
        stmts = _stmts("case (x) 0: y = 0; 1: y = 1; endcase")
        stmt = stmts[0]
        assert isinstance(stmt, CaseStatement)
        assert len(stmt.items) == 2

    def test_case_with_default(self):
        stmts = _stmts("case (x) 0: y = 0; default: y = -1; endcase")
        stmt = stmts[0]
        assert isinstance(stmt, CaseStatement)
        assert len(stmt.items) == 2
        # default has empty values
        assert stmt.items[1].values == []

    def test_case_multi_value(self):
        stmts = _stmts("case (x) 0, 1: y = 0; 2: y = 1; endcase")
        stmt = stmts[0]
        assert len(stmt.items[0].values) == 2

    def test_case_with_block_body(self):
        stmts = _stmts("case (x) 0: begin y = 0; z = 1; end endcase")
        stmt = stmts[0]
        assert isinstance(stmt.items[0].body, Block)


class TestParserExpressions:

    def test_number_literal(self):
        stmts = _stmts("x = 3.14;")
        stmt = stmts[0]
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == pytest.approx(3.14)

    def test_negative_literal(self):
        stmts = _stmts("x = -1.0;")
        stmt = stmts[0]
        # Parser optimises -(NumberLiteral) → NumberLiteral(-val)
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == pytest.approx(-1.0)

    def test_identifier_expr(self):
        stmts = _stmts("x = y;")
        stmt = stmts[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "y"

    def test_binary_add(self):
        stmts = _stmts("x = a + b;")
        expr = stmts[0].value
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "+"

    def test_binary_subtract(self):
        stmts = _stmts("x = a - b;")
        assert stmts[0].value.op == "-"

    def test_binary_multiply(self):
        stmts = _stmts("x = a * b;")
        assert stmts[0].value.op == "*"

    def test_binary_divide(self):
        stmts = _stmts("x = a / b;")
        assert stmts[0].value.op == "/"

    def test_binary_lshift(self):
        stmts = _stmts("x = a << 2;")
        assert stmts[0].value.op == "<<"

    def test_binary_rshift(self):
        stmts = _stmts("x = a >> 1;")
        assert stmts[0].value.op == ">>"

    def test_binary_bitand(self):
        stmts = _stmts("x = a & b;")
        assert stmts[0].value.op == "&"

    def test_binary_bitor(self):
        stmts = _stmts("x = a | b;")
        assert stmts[0].value.op == "|"

    def test_binary_xor(self):
        stmts = _stmts("x = a ^ b;")
        assert stmts[0].value.op == "^"

    def test_binary_land(self):
        stmts = _stmts("x = a && b;")
        assert stmts[0].value.op == "&&"

    def test_binary_lor(self):
        stmts = _stmts("x = a || b;")
        assert stmts[0].value.op == "||"

    def test_binary_eq(self):
        stmts = _stmts("x = a == b;")
        assert stmts[0].value.op == "=="

    def test_binary_ne(self):
        stmts = _stmts("x = a != b;")
        assert stmts[0].value.op == "!="

    def test_operator_precedence_mul_over_add(self):
        # a + b * c → a + (b * c)
        stmts = _stmts("x = a + b * c;")
        expr = stmts[0].value
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "+"
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == "*"

    def test_unary_minus_ident(self):
        stmts = _stmts("x = -y;")
        expr = stmts[0].value
        assert isinstance(expr, UnaryExpr)
        assert expr.op == "-"

    def test_unary_not(self):
        stmts = _stmts("x = !y;")
        assert stmts[0].value.op == "!"

    def test_unary_bitnot(self):
        stmts = _stmts("x = ~y;")
        assert stmts[0].value.op == "~"

    def test_ternary_expr(self):
        stmts = _stmts("x = (a > 0) ? 1.0 : 0.0;")
        expr = stmts[0].value
        assert isinstance(expr, TernaryExpr)
        assert isinstance(expr.cond, BinaryExpr)

    def test_function_call_no_args(self):
        stmts = _stmts("x = foo();")
        expr = stmts[0].value
        assert isinstance(expr, FunctionCall)
        assert expr.name == "foo"
        assert expr.args == []

    def test_function_call_with_args(self):
        stmts = _stmts("x = transition(target, 0.0, 1n, 1n);")
        expr = stmts[0].value
        assert isinstance(expr, FunctionCall)
        assert expr.name == "transition"
        assert len(expr.args) == 4

    def test_branch_access_single_node(self):
        stmts = _stmts("x = V(a);")
        expr = stmts[0].value
        assert isinstance(expr, BranchAccess)
        assert expr.access_type == "V"
        assert expr.node1 == "a"
        assert expr.node2 is None

    def test_branch_access_differential(self):
        stmts = _stmts("x = V(a, b);")
        expr = stmts[0].value
        assert isinstance(expr, BranchAccess)
        assert expr.node1 == "a"
        assert expr.node2 == "b"

    def test_array_access_expr(self):
        stmts = _stmts("x = arr[3];")
        expr = stmts[0].value
        assert isinstance(expr, ArrayAccess)
        assert expr.name == "arr"

    def test_method_call(self):
        stmts = _stmts("x = conf.substr(0, 1);")
        expr = stmts[0].value
        assert isinstance(expr, MethodCall)
        assert expr.obj == "conf"
        assert expr.method == "substr"
        assert len(expr.args) == 2

    def test_parenthesized_expr(self):
        stmts = _stmts("x = (a + b) * c;")
        expr = stmts[0].value
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryExpr)
        assert expr.left.op == "+"


class TestParserSystemTask:

    def test_strobe_no_args(self):
        stmts = _stmts("$strobe;")
        stmt = stmts[0]
        assert isinstance(stmt, SystemTask)
        assert stmt.name == "$strobe"
        assert stmt.args == []

    def test_strobe_with_args(self):
        stmts = _stmts('$strobe("val=%d", x);')
        stmt = stmts[0]
        assert isinstance(stmt, SystemTask)
        assert len(stmt.args) == 2
        assert isinstance(stmt.args[0], StringLiteral)

    def test_display(self):
        stmts = _stmts('$display("hello");')
        stmt = stmts[0]
        assert isinstance(stmt, SystemTask)
        assert stmt.name == "$display"


class TestParserPortDecls:

    def test_non_ansi_electrical_decl(self):
        src = """
        module buf(in, out);
        electrical in, out;
        input in;
        output out;
        endmodule
        """
        m = _parse(src)
        names = [p.name for p in m.port_decls]
        assert "in" in names
        assert "out" in names

    def test_port_direction_input(self):
        src = "module m(a); input electrical a; endmodule"
        m = _parse(src)
        pd = next(p for p in m.port_decls if p.name == "a")
        assert pd.direction == Direction.INPUT

    def test_port_direction_output(self):
        src = "module m(z); output electrical z; endmodule"
        m = _parse(src)
        pd = next(p for p in m.port_decls if p.name == "z")
        assert pd.direction == Direction.OUTPUT

    def test_array_port(self):
        src = "module m(DOUT); output electrical [3:0] DOUT; endmodule"
        m = _parse(src)
        pd = next(p for p in m.port_decls if p.name == "DOUT")
        assert pd.is_array
        assert pd.array_hi == 3
        assert pd.array_lo == 0


class TestParser2DNodeArray:
    """Parser support for 1-D and 2-D electrical node arrays in V()/I()."""

    def test_1d_node_array_contribution(self):
        """V(clk_nodes[N], VSS) <+ 0.0  — 1-D dynamic index."""
        src = _wrap("V(clk_nodes[N], VSS) <+ 0.0;")
        stmts = _parse(src).analog_block.body.statements
        contrib = stmts[0]
        assert isinstance(contrib, Contribution)
        ba = contrib.branch
        assert ba.node1 == "clk_nodes"
        assert isinstance(ba.node1_index, Identifier)
        assert ba.node1_index.name == "N"
        assert ba.node1_index2 is None
        assert ba.node2 == "VSS"

    def test_2d_node_array_contribution(self):
        """V(dbus[ch][j], VSS) <+ 0.0  — 2-D dynamic index."""
        src = _wrap("V(dbus[ch][j], VSS) <+ 0.0;")
        stmts = _parse(src).analog_block.body.statements
        contrib = stmts[0]
        assert isinstance(contrib, Contribution)
        ba = contrib.branch
        assert ba.node1 == "dbus"
        assert isinstance(ba.node1_index, Identifier)
        assert ba.node1_index.name == "ch"
        assert isinstance(ba.node1_index2, Identifier)
        assert ba.node1_index2.name == "j"
        assert ba.node2 == "VSS"

    def test_2d_node_array_read(self):
        """x = V(dbus[ch][j])  — reading a 2-D array node."""
        src = _wrap("x = V(dbus[ch][j]);")
        stmts = _parse(src).analog_block.body.statements
        assign = stmts[0]
        assert isinstance(assign, Assignment)
        ba = assign.value
        assert isinstance(ba, BranchAccess)
        assert ba.node1 == "dbus"
        assert isinstance(ba.node1_index, Identifier)
        assert ba.node1_index.name == "ch"
        assert isinstance(ba.node1_index2, Identifier)
        assert ba.node1_index2.name == "j"

    def test_2d_electrical_decl_parsed_without_error(self):
        """electrical [1:0] dbus [0:3]; should not leave stray tokens."""
        src = """
        module m(VDD, VSS);
        inout electrical VDD, VSS;
        electrical [1:0] dbus [0:3];
        analog begin
            V(dbus[0][0], VSS) <+ 0.0;
        end
        endmodule
        """
        m = _parse(src)
        # dbus should be registered as an array port/node
        pd = next((p for p in m.port_decls if p.name == "dbus"), None)
        assert pd is not None
        assert pd.is_array


class TestAnsiSharedDisciplineWarning:
    """Parser warns when ANSI port list has shared direction/discipline (Cadence VACOMP pitfall)."""

    def test_shared_discipline_generates_warning(self):
        """inout electrical VDD, VSS  — VSS has no direction, must warn."""
        src = """
        module adc(
            inout electrical VDD, VSS,
            input electrical VIN,
            output electrical [9:0] DOUT
        );
        analog begin
            V(DOUT[0], VSS) <+ 0.0;
        end
        endmodule
        """
        m = _parse(src)
        # VSS has no direction → warning expected
        assert len(m.warnings) >= 1
        assert any("VSS" in w for w in m.warnings)
        assert any("Cadence" in w or "direction" in w for w in m.warnings)

    def test_warning_message_mentions_spectre(self):
        src = """
        module m(
            input electrical VIN, CLK,
            output electrical DOUT
        );
        endmodule
        """
        m = _parse(src)
        # CLK has no direction
        w_text = " ".join(m.warnings)
        assert "CLK" in w_text
        assert "Spectre" in w_text

    def test_correct_separate_declarations_no_warning(self):
        """One port per line — no warnings."""
        src = """
        module adc(
            inout  electrical VDD,
            inout  electrical VSS,
            input  electrical VIN,
            input  electrical CLK,
            output electrical [9:0] DOUT
        );
        endmodule
        """
        m = _parse(src)
        assert m.warnings == []

    def test_shared_discipline_still_parses_correctly(self):
        """EVAS must still simulate even with the warning — tolerant behaviour."""
        src = """
        module m(
            inout electrical VDD, VSS,
            output electrical OUT
        );
        analog begin
            V(OUT, VSS) <+ 0.0;
        end
        endmodule
        """
        m = _parse(src)
        # All three ports should be reachable
        all_ports = set(m.ports) | {pd.name for pd in m.port_decls}
        assert "VDD" in all_ports
        assert "VSS" in all_ports
        assert "OUT" in all_ports
