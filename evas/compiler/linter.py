"""Static EVAS/Spectre-style lint checks for Verilog-A inputs.

The linter is intentionally separate from simulation.  It reports model
quality warnings and EVAS/Spectre compatibility errors without changing the
normal compile/simulate pass/fail path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from evas.netlist.spectre_parser import parse_spectre

from . import ast_nodes as va_ast
from .lexer import Token, TokenType, tokenize
from .parser import ParseError, SpectreReservedIdentifierError, parse_all
from .preprocessor import PreprocessorError, preprocess

COMPAT_ERROR = "compat-error"
STATIC_WARNING = "static-warning"
DYNAMIC_WARNING = "dynamic-warning"


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    file: str
    line: Optional[int] = None
    column: Optional[int] = None
    module: Optional[str] = None
    rule: str = ""
    spectre_ids: List[str] = field(default_factory=list)
    source: str = "evas-lint"

    def to_dict(self) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v not in (None, [], "")}

    def format_text(self) -> str:
        loc = self.file
        if self.line is not None:
            loc += f":{self.line}"
            if self.column is not None:
                loc += f":{self.column}"
        module = f" [{self.module}]" if self.module else ""
        ids = f" ({', '.join(self.spectre_ids)})" if self.spectre_ids else ""
        return f"{loc}: {self.severity} {self.code}{ids}{module}: {self.message}"


def lint_file(path: str | Path, *, min_transition: float = 1e-12) -> List[Diagnostic]:
    """Lint a Verilog-A file or a Spectre netlist with ahdl_include entries."""
    src_path = Path(path).resolve()
    if src_path.suffix.lower() == ".scs":
        return lint_spectre_netlist(src_path, min_transition=min_transition)
    return lint_veriloga_file(src_path, min_transition=min_transition)


def lint_spectre_netlist(
    path: str | Path,
    *,
    min_transition: float = 1e-12,
) -> List[Diagnostic]:
    scs_path = Path(path).resolve()
    try:
        netlist = parse_spectre(str(scs_path))
    except Exception as exc:
        return [
            Diagnostic(
                code="EVAS-COMP-ENETLIST",
                severity=COMPAT_ERROR,
                message=f"failed to parse Spectre netlist: {exc}",
                file=str(scs_path),
                rule="netlist-parse",
            )
        ]

    diagnostics: List[Diagnostic] = []
    scs_dir = Path(netlist.source_dir)
    for inc in netlist.ahdl_includes:
        va_path = _resolve_ahdl_include(inc.path, scs_dir)
        if va_path is None:
            diagnostics.append(
                Diagnostic(
                    code="EVAS-COMP-EINCLUDE",
                    severity=COMPAT_ERROR,
                    message=f"cannot resolve ahdl_include {inc.path!r}",
                    file=str(scs_path),
                    rule="ahdl-include",
                )
            )
            continue
        diagnostics.extend(
            lint_veriloga_file(va_path, min_transition=min_transition)
        )
    return diagnostics


def lint_veriloga_file(
    path: str | Path,
    *,
    min_transition: float = 1e-12,
) -> List[Diagnostic]:
    va_path = Path(path).resolve()
    try:
        source = va_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [
            Diagnostic(
                code="EVAS-COMP-EFILE",
                severity=COMPAT_ERROR,
                message=f"cannot read Verilog-A file: {exc}",
                file=str(va_path),
                rule="file-read",
            )
        ]
    return lint_source(
        source,
        filename=str(va_path),
        source_dir=str(va_path.parent),
        min_transition=min_transition,
    )


def lint_source(
    source: str,
    *,
    filename: str = "<string>",
    source_dir: str = ".",
    min_transition: float = 1e-12,
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    try:
        pp_src, _defines, _default_transition = preprocess(source, source_dir=source_dir)
        range_diagnostics = _lint_nonconstant_discipline_ranges(pp_src, filename)
        modules = parse_all(pp_src)
    except PreprocessorError as exc:
        return [
            Diagnostic(
                code="EVAS-COMP-EPREPROC",
                severity=COMPAT_ERROR,
                message=str(exc),
                file=filename,
                rule="preprocess",
            )
        ]
    except SpectreReservedIdentifierError as exc:
        token = getattr(exc, "token", None)
        return [
            Diagnostic(
                code="EVAS-COMP-E2174",
                severity=COMPAT_ERROR,
                message=str(exc),
                file=filename,
                line=getattr(token, "line", None),
                column=getattr(token, "col", None),
                rule="reserved-identifier",
                spectre_ids=[getattr(exc, "spectre_code", "VACOMP-2174")],
            )
        ]
    except ParseError as exc:
        token = getattr(exc, "token", None)
        range_diagnostics = (
            _lint_nonconstant_discipline_ranges(pp_src, filename)
            if "pp_src" in locals()
            else []
        )
        if any(d.code == "EVAS-COMP-E2446" for d in range_diagnostics):
            return range_diagnostics
        return [
            Diagnostic(
                code="EVAS-COMP-EPARSE",
                severity=COMPAT_ERROR,
                message=str(exc),
                file=filename,
                line=getattr(token, "line", None),
                column=getattr(token, "col", None),
                rule="parse",
            )
        ]

    diagnostics.extend(range_diagnostics)
    for module in modules:
        diagnostics.extend(_lint_module(module, filename, min_transition))
    return diagnostics


def has_compat_errors(diagnostics: Sequence[Diagnostic]) -> bool:
    return any(d.severity == COMPAT_ERROR for d in diagnostics)


def _resolve_ahdl_include(path_text: str, scs_dir: Path) -> Optional[Path]:
    raw = Path(path_text)
    candidates = [raw, scs_dir / raw, scs_dir / raw.name]
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _lint_module(
    module: va_ast.Module,
    filename: str,
    min_transition: float,
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for warning in module.warnings:
        diagnostics.append(
            Diagnostic(
                code="EVAS-COMP-WPARSE",
                severity=STATIC_WARNING,
                message=warning,
                file=filename,
                module=module.name,
                rule="spectre-parse-compat",
            )
        )

    user_function_names = {fn.name for fn in getattr(module, "functions", [])}
    genvar_names = {
        v.name for v in getattr(module, "variables", [])
        if getattr(v, "is_genvar", False)
    }
    integer_vars = {
        v.name for v in getattr(module, "variables", [])
        if getattr(v, "var_type", None) == va_ast.ParamType.INTEGER
    }
    discrete_vars = set(integer_vars)
    if module.analog_block is not None:
        discrete_vars.update(_assigned_in_events(module.analog_block.body))
        for _ in range(8):
            newly_discrete = _assigned_from_discrete_expr(
                module.analog_block.body,
                discrete_vars,
            )
            if newly_discrete.issubset(discrete_vars):
                break
            discrete_vars.update(newly_discrete)
        _lint_statement(
            module.analog_block.body,
            diagnostics,
            filename,
            module.name,
            min_transition,
            discrete_vars,
            user_function_names,
            genvar_names,
            conditional_depth=0,
            in_event=False,
        )

    for function in getattr(module, "functions", []):
        _lint_statement(
            function.body,
            diagnostics,
            filename,
            module.name,
            min_transition,
            discrete_vars,
            user_function_names,
            genvar_names,
            conditional_depth=0,
            in_event=False,
        )
    for task in getattr(module, "tasks", []):
        _lint_statement(
            task.body,
            diagnostics,
            filename,
            module.name,
            min_transition,
            discrete_vars,
            user_function_names,
            genvar_names,
            conditional_depth=0,
            in_event=False,
        )
    return diagnostics


def _lint_statement(
    stmt: va_ast.Statement,
    diagnostics: List[Diagnostic],
    filename: str,
    module: str,
    min_transition: float,
    discrete_vars: Set[str],
    user_function_names: Set[str],
    genvar_names: Set[str],
    *,
    conditional_depth: int,
    in_event: bool,
) -> None:
    if stmt is None:
        return
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            _lint_statement(
                child, diagnostics, filename, module, min_transition,
                discrete_vars, user_function_names, genvar_names,
                conditional_depth=conditional_depth, in_event=in_event,
            )
        return

    if isinstance(stmt, va_ast.Contribution):
        if in_event:
            diagnostics.append(
                Diagnostic(
                    code="EVAS-COMP-E2157",
                    severity=COMPAT_ERROR,
                    message=(
                        "contribution statement is embedded in an analog event "
                        "body; Spectre VACOMP rejects event-local contributions"
                    ),
                    file=filename,
                    module=module,
                    rule="event-body-contribution",
                    spectre_ids=["VACOMP-2157"],
                )
            )
        if conditional_depth > 0:
            diagnostics.extend(
                _conditional_analog_operator_diagnostics(
                    stmt.expr,
                    filename,
                    module,
                )
            )
        if (
            not _expr_has_call(stmt.expr, "transition")
            and _expr_has_discrete_behavior(stmt.expr, discrete_vars)
        ):
            diagnostics.append(
                Diagnostic(
                    code="EVAS-AHDL-W5008",
                    severity=STATIC_WARNING,
                    message=(
                        "discrete-valued expression directly drives an analog "
                        "contribution; use transition() on the discrete target "
                        "when a discontinuity is intended"
                    ),
                    file=filename,
                    module=module,
                    rule="discrete-contribution-transition",
                    spectre_ids=["AHDLLINT-5008"],
                )
            )
        _lint_expr(
            stmt.expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(stmt, va_ast.Assignment):
        if conditional_depth > 0:
            diagnostics.extend(
                _conditional_analog_operator_diagnostics(
                    stmt.value,
                    filename,
                    module,
                )
            )
        _lint_expr(
            stmt.target, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            stmt.value, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(stmt, va_ast.EventStatement):
        if _event_has_timer(stmt.event) and conditional_depth > 0:
            diagnostics.append(
                Diagnostic(
                    code="EVAS-AHDL-W8007",
                    severity=DYNAMIC_WARNING,
                    message=(
                        "timer() event is inside runtime-controlled statement "
                        "flow; Cadence AHDL lint reports this as a scheduling risk"
                    ),
                    file=filename,
                    module=module,
                    rule="conditional-timer",
                    spectre_ids=["AHDLLINT-8007"],
                )
            )
        _lint_event_expr(
            stmt.event, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        event_is_initial_step = (
            isinstance(stmt.event, va_ast.EventExpr)
            and stmt.event.event_type == va_ast.EventType.INITIAL_STEP
        )
        _lint_statement(
            stmt.body,
            diagnostics,
            filename,
            module,
            min_transition,
            discrete_vars,
            user_function_names,
            genvar_names,
            conditional_depth=conditional_depth + 1,
            in_event=False if event_is_initial_step else True,
        )
        return

    if isinstance(stmt, va_ast.IfStatement):
        _lint_expr(
            stmt.cond, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_statement(
            stmt.then_body, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=conditional_depth + 1, in_event=in_event,
        )
        _lint_statement(
            stmt.else_body, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=conditional_depth + 1, in_event=in_event,
        )
        return

    if isinstance(stmt, va_ast.ForStatement):
        loop_var = _assignment_target_name(stmt.init)
        loop_depth = conditional_depth if loop_var in genvar_names else conditional_depth + 1
        _lint_statement(
            stmt.init, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=conditional_depth, in_event=in_event,
        )
        _lint_expr(
            stmt.cond, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_statement(
            stmt.update, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=conditional_depth, in_event=in_event,
        )
        _lint_statement(
            stmt.body, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=loop_depth, in_event=in_event,
        )
        return

    if isinstance(stmt, va_ast.WhileStatement):
        _lint_expr(
            stmt.cond, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_statement(
            stmt.body, diagnostics, filename, module, min_transition,
            discrete_vars, user_function_names, genvar_names,
            conditional_depth=conditional_depth + 1, in_event=in_event,
        )
        return

    if isinstance(stmt, va_ast.CaseStatement):
        _lint_expr(
            stmt.expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        for item in stmt.items:
            for value in item.values:
                _lint_expr(
                    value, diagnostics, filename, module, min_transition,
                    user_function_names,
                )
            _lint_statement(
                item.body, diagnostics, filename, module, min_transition,
                discrete_vars, user_function_names, genvar_names,
                conditional_depth=conditional_depth + 1, in_event=in_event,
            )
        return

    if isinstance(stmt, va_ast.SystemTask):
        for arg in stmt.args:
            _lint_expr(
                arg, diagnostics, filename, module, min_transition,
                user_function_names,
            )
        return

    if isinstance(stmt, va_ast.TaskCall):
        for arg in stmt.args:
            _lint_expr(
                arg, diagnostics, filename, module, min_transition,
                user_function_names,
            )


def _lint_event_expr(
    event: va_ast.EventExpr | va_ast.CombinedEvent,
    diagnostics: List[Diagnostic],
    filename: str,
    module: str,
    min_transition: float,
    user_function_names: Set[str],
) -> None:
    if isinstance(event, va_ast.CombinedEvent):
        for child in event.events:
            _lint_event_expr(
                child, diagnostics, filename, module, min_transition,
                user_function_names,
            )
        return
    for expr in event.args:
        _lint_expr(
            expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
    _lint_expr(
        event.time_tol_expr, diagnostics, filename, module, min_transition,
        user_function_names,
    )
    _lint_expr(
        event.expr_tol_expr, diagnostics, filename, module, min_transition,
        user_function_names,
    )


def _lint_expr(
    expr: Optional[va_ast.Expr],
    diagnostics: List[Diagnostic],
    filename: str,
    module: str,
    min_transition: float,
    user_function_names: Set[str],
) -> None:
    if expr is None:
        return

    if isinstance(expr, va_ast.BinaryExpr):
        if expr.op in {"==", "!="} and (
            _is_abstime_expr(expr.left) or _is_abstime_expr(expr.right)
        ):
            diagnostics.append(
                Diagnostic(
                    code="EVAS-AHDL-W5012",
                    severity=STATIC_WARNING,
                    message=(
                        "$abstime is compared for exact equality; use an event, "
                        "timer(), or a tolerance/windowed comparison instead"
                    ),
                    file=filename,
                    module=module,
                    rule="abstime-exact-equality",
                    spectre_ids=["AHDLLINT-5012"],
                )
            )
        _lint_expr(
            expr.left, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.right, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.UnaryExpr):
        _lint_expr(
            expr.operand, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.TernaryExpr):
        _lint_expr(
            expr.cond, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.true_expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.false_expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.ConcatExpr):
        for part in expr.parts:
            _lint_expr(
                part, diagnostics, filename, module, min_transition,
                user_function_names,
            )
        return

    if isinstance(expr, va_ast.ReplicateExpr):
        _lint_expr(
            expr.count, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.expr, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.ArrayAccess):
        _lint_expr(
            expr.index, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.index2, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.PartSelect):
        _lint_expr(
            expr.msb, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.lsb, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.BranchAccess):
        _lint_expr(
            expr.node1_index, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.node2_index, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.node1_index2, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        _lint_expr(
            expr.node2_index2, diagnostics, filename, module, min_transition,
            user_function_names,
        )
        return

    if isinstance(expr, va_ast.MethodCall):
        for arg in expr.args:
            _lint_expr(
                arg, diagnostics, filename, module, min_transition,
                user_function_names,
            )
        return

    if isinstance(expr, va_ast.FunctionCall):
        name = expr.name.lower()
        if name == "transition":
            _lint_transition_call(
                expr, diagnostics, filename, module, min_transition,
            )
        elif name == "ddt" and any(_expr_has_call(arg, "ddt") for arg in expr.args):
            diagnostics.append(
                Diagnostic(
                    code="EVAS-COMP-E1519",
                    severity=COMPAT_ERROR,
                    message="nested ddt(ddt(...)) is not a Spectre-compatible analog operator form",
                    file=filename,
                    module=module,
                    rule="nested-ddt",
                    spectre_ids=["VACOMP-1519"],
                )
            )
        if not _is_supported_function(expr.name, user_function_names):
            diagnostics.append(
                Diagnostic(
                    code="EVAS-COMP-EUNSUPPORTED",
                    severity=COMPAT_ERROR,
                    message=f"unsupported Verilog-A function/operator call: {expr.name}()",
                    file=filename,
                    module=module,
                    rule="unsupported-function",
                )
            )
        for arg in expr.args:
            _lint_expr(
                arg, diagnostics, filename, module, min_transition,
                user_function_names,
            )


def _lint_transition_call(
    expr: va_ast.FunctionCall,
    diagnostics: List[Diagnostic],
    filename: str,
    module: str,
    min_transition: float,
) -> None:
    if expr.args and _expr_contains_branch_access(expr.args[0]):
        diagnostics.append(
            Diagnostic(
                code="EVAS-AHDL-W5007",
                severity=STATIC_WARNING,
                message=(
                    "transition() input depends on a continuous branch value; "
                    "Cadence AHDL lint recommends feeding transition() with a "
                    "discrete target updated by events"
                ),
                file=filename,
                module=module,
                rule="transition-continuous-input",
                spectre_ids=["AHDLLINT-5007", "AHDLLINT-8004"],
            )
        )

    for idx, code in ((2, "EVAS-AHDL-W5004"), (3, "EVAS-AHDL-W5005")):
        if len(expr.args) <= idx:
            continue
        value = _numeric_value(expr.args[idx])
        if value is None:
            continue
        if value <= 0.0 or value < min_transition:
            diagnostics.append(
                Diagnostic(
                    code=code,
                    severity=DYNAMIC_WARNING,
                    message=(
                        "transition() rise/fall time is zero or smaller than "
                        f"the lint minimum {min_transition:g}s"
                    ),
                    file=filename,
                    module=module,
                    rule="tiny-transition-time",
                    spectre_ids=[
                        "AHDLLINT-5004",
                        "AHDLLINT-5005",
                        "AHDLLINT-8001",
                        "AHDLLINT-8002",
                        "AHDLLINT-8007",
                    ],
                )
            )


def _assigned_in_events(stmt: va_ast.Statement, in_event: bool = False) -> Set[str]:
    names: Set[str] = set()
    if stmt is None:
        return names
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            names.update(_assigned_in_events(child, in_event))
    elif isinstance(stmt, va_ast.EventStatement):
        event_is_initial_step = (
            isinstance(stmt.event, va_ast.EventExpr)
            and stmt.event.event_type == va_ast.EventType.INITIAL_STEP
        )
        names.update(_assigned_in_events(stmt.body, False if event_is_initial_step else True))
    elif isinstance(stmt, va_ast.Assignment):
        if in_event:
            target = _assignment_target_name(stmt)
            if target:
                names.add(target)
    elif isinstance(stmt, va_ast.IfStatement):
        names.update(_assigned_in_events(stmt.then_body, in_event))
        names.update(_assigned_in_events(stmt.else_body, in_event))
    elif isinstance(stmt, va_ast.ForStatement):
        names.update(_assigned_in_events(stmt.init, in_event))
        names.update(_assigned_in_events(stmt.update, in_event))
        names.update(_assigned_in_events(stmt.body, in_event))
    elif isinstance(stmt, va_ast.WhileStatement):
        names.update(_assigned_in_events(stmt.body, in_event))
    elif isinstance(stmt, va_ast.CaseStatement):
        for item in stmt.items:
            names.update(_assigned_in_events(item.body, in_event))
    return names


def _assigned_from_discrete_expr(
    stmt: va_ast.Statement,
    discrete_vars: Set[str],
) -> Set[str]:
    names: Set[str] = set()
    if stmt is None:
        return names
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            names.update(_assigned_from_discrete_expr(child, discrete_vars))
    elif isinstance(stmt, va_ast.Assignment):
        target = _assignment_target_name(stmt)
        if (
            target
            and not _expr_has_call(stmt.value, "transition")
            and _expr_has_discrete_behavior(stmt.value, discrete_vars)
        ):
            names.add(target)
    elif isinstance(stmt, va_ast.EventStatement):
        names.update(_assigned_from_discrete_expr(stmt.body, discrete_vars))
    elif isinstance(stmt, va_ast.IfStatement):
        names.update(_assigned_from_discrete_expr(stmt.then_body, discrete_vars))
        names.update(_assigned_from_discrete_expr(stmt.else_body, discrete_vars))
    elif isinstance(stmt, va_ast.ForStatement):
        names.update(_assigned_from_discrete_expr(stmt.init, discrete_vars))
        names.update(_assigned_from_discrete_expr(stmt.update, discrete_vars))
        names.update(_assigned_from_discrete_expr(stmt.body, discrete_vars))
    elif isinstance(stmt, va_ast.WhileStatement):
        names.update(_assigned_from_discrete_expr(stmt.body, discrete_vars))
    elif isinstance(stmt, va_ast.CaseStatement):
        for item in stmt.items:
            names.update(_assigned_from_discrete_expr(item.body, discrete_vars))
    return names


def _assignment_target_name(assign: Optional[va_ast.Assignment]) -> Optional[str]:
    if assign is None:
        return None
    target = getattr(assign, "target", None)
    if isinstance(target, va_ast.Identifier):
        return target.name
    if isinstance(target, va_ast.ArrayAccess):
        return target.name
    return None


def _event_has_timer(event: va_ast.EventExpr | va_ast.CombinedEvent) -> bool:
    if isinstance(event, va_ast.CombinedEvent):
        return any(_event_has_timer(child) for child in event.events)
    return event.event_type == va_ast.EventType.TIMER


def _conditional_analog_operator_diagnostics(
    expr: Optional[va_ast.Expr],
    filename: str,
    module: str,
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for call in _iter_function_calls(expr):
        spec = _CONDITIONAL_ANALOG_OPERATOR_CODES.get(call.name.lower())
        if spec is None:
            continue
        code, spectre_id = spec
        diagnostics.append(
            Diagnostic(
                code=code,
                severity=COMPAT_ERROR,
                message=(
                    f"{call.name}() analog operator is inside a runtime "
                    "conditional/event/loop/case statement"
                ),
                file=filename,
                module=module,
                rule="conditional-analog-operator",
                spectre_ids=[spectre_id],
            )
        )
    return diagnostics


def _iter_function_calls(expr: Optional[va_ast.Expr]) -> Iterable[va_ast.FunctionCall]:
    if expr is None:
        return
    if isinstance(expr, va_ast.FunctionCall):
        yield expr
    for child in _expr_children(expr):
        yield from _iter_function_calls(child)


def _expr_has_call(expr: Optional[va_ast.Expr], call_name: str) -> bool:
    if expr is None:
        return False
    target = call_name.lower()
    if isinstance(expr, va_ast.FunctionCall):
        return expr.name.lower() == target or any(
            _expr_has_call(arg, call_name) for arg in expr.args
        )
    return any(_expr_has_call(child, call_name) for child in _expr_children(expr))


def _expr_contains_branch_access(expr: Optional[va_ast.Expr]) -> bool:
    if expr is None:
        return False
    if isinstance(expr, va_ast.BranchAccess):
        return True
    return any(_expr_contains_branch_access(child) for child in _expr_children(expr))


def _expr_has_discrete_behavior(expr: Optional[va_ast.Expr], discrete_vars: Set[str]) -> bool:
    if expr is None:
        return False
    if isinstance(expr, va_ast.TernaryExpr):
        return True
    if isinstance(expr, va_ast.BinaryExpr):
        if expr.op in {"==", "!=", ">", "<", ">=", "<=", "&&", "||", "&", "|", "^", "<<", ">>"}:
            return True
        return (
            _expr_has_discrete_behavior(expr.left, discrete_vars)
            or _expr_has_discrete_behavior(expr.right, discrete_vars)
        )
    if isinstance(expr, va_ast.UnaryExpr):
        if expr.op in {"!", "~"}:
            return True
        return _expr_has_discrete_behavior(expr.operand, discrete_vars)
    if isinstance(expr, va_ast.Identifier):
        return expr.name in discrete_vars
    if isinstance(expr, va_ast.ArrayAccess):
        return expr.name in discrete_vars or any(
            _expr_has_discrete_behavior(child, discrete_vars)
            for child in _expr_children(expr)
        )
    return any(
        _expr_has_discrete_behavior(child, discrete_vars)
        for child in _expr_children(expr)
    )


_DISCIPLINE_RANGE_TOKENS = {
    TokenType.ELECTRICAL,
    TokenType.VOLTAGE,
    TokenType.CURRENT,
}

_DIRECTION_TOKENS = {
    TokenType.INPUT,
    TokenType.OUTPUT,
    TokenType.INOUT,
}

_CONSTANT_RANGE_OPERATORS = {
    TokenType.NUMBER,
    TokenType.IDENT,
    TokenType.PLUS,
    TokenType.MINUS,
    TokenType.STAR,
    TokenType.SLASH,
    TokenType.PERCENT,
    TokenType.LPAREN,
    TokenType.RPAREN,
    TokenType.COLON,
}


def _lint_nonconstant_discipline_ranges(
    source: str,
    filename: str,
) -> List[Diagnostic]:
    tokens = tokenize(source)
    diagnostics: List[Diagnostic] = []
    emitted: set[tuple[int, int]] = set()
    for module_tokens in _iter_module_token_slices(tokens):
        parameter_names = _module_parameter_names(module_tokens)
        for statement in _iter_semicolon_statements(module_tokens):
            if not _is_discipline_declaration_statement(statement):
                continue
            for lbracket, range_tokens in _iter_bracket_ranges(statement):
                key = (lbracket.line, lbracket.col)
                if key in emitted:
                    continue
                if _range_uses_nonconstant_identifier(range_tokens, parameter_names):
                    emitted.add(key)
                    diagnostics.append(
                        Diagnostic(
                            code="EVAS-COMP-E2446",
                            severity=COMPAT_ERROR,
                            message=(
                                "discipline vector range uses a non-constant "
                                "identifier; Spectre requires numeric or "
                                "parameter constant expressions"
                            ),
                            file=filename,
                            line=lbracket.line,
                            column=lbracket.col,
                            rule="nonconstant-discipline-range",
                            spectre_ids=["VACOMP-2446"],
                        )
                    )
    return diagnostics


def _iter_module_token_slices(tokens: Sequence[Token]) -> Iterable[Sequence[Token]]:
    idx = 0
    while idx < len(tokens):
        if tokens[idx].type not in {TokenType.MODULE, TokenType.CONNECTMODULE}:
            idx += 1
            continue
        end_type = (
            TokenType.ENDCONNECTMODULE
            if tokens[idx].type == TokenType.CONNECTMODULE
            else TokenType.ENDMODULE
        )
        start = idx
        idx += 1
        while idx < len(tokens) and tokens[idx].type != end_type:
            idx += 1
        if idx < len(tokens):
            idx += 1
        yield tokens[start:idx]


def _module_parameter_names(tokens: Sequence[Token]) -> set[str]:
    names: set[str] = set()
    idx = 0
    while idx < len(tokens):
        if tokens[idx].type != TokenType.PARAMETER:
            idx += 1
            continue
        idx += 1
        if idx < len(tokens) and tokens[idx].type in {
            TokenType.REAL,
            TokenType.INTEGER,
        }:
            idx += 1
        elif (
            idx < len(tokens)
            and tokens[idx].type == TokenType.IDENT
            and tokens[idx].value == "string"
        ):
            idx += 1
        if idx < len(tokens) and tokens[idx].type == TokenType.IDENT:
            names.add(tokens[idx].value)
        while idx < len(tokens) and tokens[idx].type != TokenType.SEMI:
            idx += 1
    return names


def _iter_semicolon_statements(tokens: Sequence[Token]) -> Iterable[Sequence[Token]]:
    start = 0
    for idx, token in enumerate(tokens):
        if token.type == TokenType.SEMI:
            yield tokens[start:idx + 1]
            start = idx + 1


def _is_discipline_declaration_statement(tokens: Sequence[Token]) -> bool:
    types = {token.type for token in tokens}
    return bool(types & _DISCIPLINE_RANGE_TOKENS)


def _iter_bracket_ranges(
    tokens: Sequence[Token],
) -> Iterable[tuple[Token, Sequence[Token]]]:
    idx = 0
    while idx < len(tokens):
        if tokens[idx].type != TokenType.LBRACKET:
            idx += 1
            continue
        start = idx
        depth = 1
        idx += 1
        while idx < len(tokens) and depth:
            if tokens[idx].type == TokenType.LBRACKET:
                depth += 1
            elif tokens[idx].type == TokenType.RBRACKET:
                depth -= 1
            idx += 1
        end = idx - 1
        if end > start:
            yield tokens[start], tokens[start + 1:end]


def _range_uses_nonconstant_identifier(
    tokens: Sequence[Token],
    parameter_names: set[str],
) -> bool:
    for token in tokens:
        if token.type not in _CONSTANT_RANGE_OPERATORS:
            return True
        if token.type == TokenType.IDENT and token.value not in parameter_names:
            return True
    return False


def _expr_children(expr: va_ast.Expr) -> Iterable[va_ast.Expr]:
    if isinstance(expr, va_ast.FunctionCall):
        yield from expr.args
    elif isinstance(expr, va_ast.MethodCall):
        yield from expr.args
    elif isinstance(expr, va_ast.BinaryExpr):
        yield expr.left
        yield expr.right
    elif isinstance(expr, va_ast.UnaryExpr):
        yield expr.operand
    elif isinstance(expr, va_ast.TernaryExpr):
        yield expr.cond
        yield expr.true_expr
        yield expr.false_expr
    elif isinstance(expr, va_ast.ConcatExpr):
        yield from expr.parts
    elif isinstance(expr, va_ast.ReplicateExpr):
        yield expr.count
        yield expr.expr
    elif isinstance(expr, va_ast.ArrayAccess):
        yield expr.index
        if expr.index2 is not None:
            yield expr.index2
    elif isinstance(expr, va_ast.PartSelect):
        yield expr.msb
        yield expr.lsb
    elif isinstance(expr, va_ast.BranchAccess):
        for child in (
            expr.node1_index,
            expr.node2_index,
            expr.node1_index2,
            expr.node2_index2,
        ):
            if child is not None:
                yield child


def _numeric_value(expr: va_ast.Expr) -> Optional[float]:
    if isinstance(expr, va_ast.NumberLiteral):
        return float(expr.value)
    if isinstance(expr, va_ast.UnaryExpr) and expr.op == "-":
        value = _numeric_value(expr.operand)
        return -value if value is not None else None
    return None


def _is_abstime_expr(expr: va_ast.Expr) -> bool:
    if isinstance(expr, va_ast.Identifier):
        return expr.name == "$abstime"
    return isinstance(expr, va_ast.FunctionCall) and expr.name == "$abstime"


_SUPPORTED_FUNCTIONS = {
    "transition", "slew", "ddt", "idt", "idtmod", "cross", "last_crossing",
    "limexp", "laplace_nd", "laplace_np", "laplace_zd", "laplace_zp",
    "zi_nd", "zi_np", "zi_zd", "zi_zp", "ln", "log", "exp", "sqrt",
    "abs", "pow", "min", "max", "sin", "cos", "tan", "tanh", "floor",
    "ceil", "$ln", "$log", "$exp", "$sqrt", "$abs", "$pow", "$min",
    "$max", "$sin", "$cos", "$tan", "$tanh", "$floor", "$ceil",
    "$temperature", "$vt", "$simparam", "$attribute", "potential", "flow",
    "$rtoi", "$param_given", "$port_connected", "$mfactor",
    "$analog_node_alias", "$analog_port_alias", "$cds_get_mc_trial_number",
    "$cds_set_rf_source_info", "$cds_violation", "$table_model",
    "$rdist_normal", "$rdist_exponential", "$rdist_poisson",
    "$rdist_chi_square", "$rdist_t", "$rdist_erlang", "$dist_normal",
    "$dist_exponential", "$dist_poisson", "$dist_chi_square", "$dist_t",
    "$dist_erlang", "$random", "$dist_uniform", "$rdist_uniform", "$fopen",
    "$fclose", "$fwrite", "$fstrobe", "$strobe", "$display", "$debug",
    "$warning", "$error", "$info", "$fscanf", "$fgets", "$feof", "$fseek",
    "$ftell", "$rewind", "$swrite", "$sformat", "analysis", "ac_stim",
    "white_noise", "flicker_noise", "noise_table", "$abstime", "$realtime",
    "$bound_step", "$discontinuity",
}


def _is_supported_function(name: str, user_function_names: Set[str]) -> bool:
    return name in _SUPPORTED_FUNCTIONS or name in user_function_names


_CONDITIONAL_ANALOG_OPERATOR_CODES = {
    "transition": ("EVAS-COMP-E2143", "VACOMP-2143"),
    "slew": ("EVAS-COMP-E2151", "VACOMP-2151"),
    "idt": ("EVAS-COMP-E2154", "VACOMP-2154"),
}
