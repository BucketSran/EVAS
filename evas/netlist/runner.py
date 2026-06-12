"""
runner.py -- EVAS runner: parse .scs, compile VA, simulate, produce log + CSV.
"""

import csv
import os
import re as _re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import numpy as np

from evas.compiler import ast_nodes as va_ast
from evas.compiler.parser import parse as parse_va
from evas.compiler.preprocessor import preprocess
from evas.simulator.backend import compile_module
from evas.simulator.engine import SimResult, Simulator, dc, pulse, pwl, sine
from evas.simulator.indexed import (
    build_indexed_run_plan,
    check_indexed_trace_round_trip,
)

from .spectre_parser import (
    SpectreNetlist,
    SpectreSource,
    _parse_suffix_number,
    has_transistors,
    parse_spectre,
)

try:
    from importlib.metadata import version as _package_version
except ImportError:  # pragma: no cover - Python < 3.8 compatibility fallback
    from importlib_metadata import version as _package_version

try:
    VERSION = _package_version("evas-sim")
except Exception:
    VERSION = "0.4.4"

DEFAULT_EVAS_ENGINE = "python"
RUST_EVAS_ENGINE = "evas-rust"
_RUST_ENGINE_ALIASES = {"evas-rust", "evas2", "rust2"}

_EVAS_PROFILE_PRESETS = {
    # Focus on runtime.
    "fast": {"refine_factor": 8, "refine_steps": 4, "reltol_min": 5e-3},
    # Keep current default behavior.
    "balanced": {"refine_factor": 16, "refine_steps": 8, "reltol_min": 1e-3},
    # Focus on edge timing / crossing precision.
    "precision": {"refine_factor": 32, "refine_steps": 16, "reltol_max": 1e-4},
}

_SUPPLY_PORT_NAMES = {
    "vdd", "vdda", "vddd", "vcc", "avdd", "dvdd",
    "vss", "vssa", "vssd", "gnd", "gnda", "gndd", "vee",
}


def _apply_evas_profile(profile: str, refine_factor: int, refine_steps: int, reltol: float):
    p = (profile or "").strip().lower()
    if p not in _EVAS_PROFILE_PRESETS:
        return refine_factor, refine_steps, reltol, ""
    cfg = _EVAS_PROFILE_PRESETS[p]
    rf = int(cfg["refine_factor"])
    rs = int(cfg["refine_steps"])
    rt = float(reltol)
    if "reltol_min" in cfg:
        rt = max(rt, float(cfg["reltol_min"]))
    if "reltol_max" in cfg:
        rt = min(rt, float(cfg["reltol_max"]))
    return rf, rs, rt, p


def _simopt_bool(simopt: Dict[str, object], key: str, default: bool = False) -> bool:
    value = simopt.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _configured_evas_engine(simopt: Dict[str, object]) -> str:
    """Resolve EVAS engine selection.

    The Python engine is the packaged default because it works from PyPI and a
    fresh source checkout without building the optional Rust shared library.
    evas-rust remains available through explicit simulatorOptions or EVAS_ENGINE.
    Legacy evas2/rust2 selectors are accepted as compatibility aliases.
    """

    explicit = str(simopt.get("evas_engine", "")).strip().lower()
    if explicit:
        return _normalize_evas_engine(explicit)
    env_engine = os.environ.get("EVAS_ENGINE", "").strip().lower()
    if env_engine:
        return _normalize_evas_engine(env_engine)
    return DEFAULT_EVAS_ENGINE


def _normalize_evas_engine(engine: str) -> str:
    text = engine.strip().lower()
    if text in _RUST_ENGINE_ALIASES:
        return RUST_EVAS_ENGINE
    return text


def _first_param(params: Dict[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in params:
            return params[key]
    return default


# ---------------------------------------------------------------------------
# VA model compilation
# ---------------------------------------------------------------------------

def _compile_va(
    va_path: str,
    source_dir: str = None,
    static_branch_fastpath_codegen: bool = False,
    indexed_state_fastpath_codegen: bool = False,
):
    """Compile a .va file. Returns (ModelClass, Module) tuple."""
    if source_dir is None:
        source_dir = str(Path(va_path).parent)
    source = Path(va_path).read_text(encoding='utf-8', errors='replace')
    pp_src, defines, default_trans = preprocess(source, source_dir=source_dir)
    module = parse_va(pp_src)
    module.defines = defines
    _validate_va_spectre_compat(module)
    if default_trans is None:
        default_trans = 1e-12
    cls = compile_module(
        module,
        default_trans,
        static_branch_fastpath_codegen=static_branch_fastpath_codegen,
        indexed_state_fastpath_codegen=indexed_state_fastpath_codegen,
    )
    return cls, module


def _expr_has_call(expr, call_name: str) -> bool:
    """Return True if an expression tree contains a call by name."""
    if expr is None:
        return False
    if isinstance(expr, va_ast.FunctionCall):
        if expr.name.lower() == call_name.lower():
            return True
        return any(_expr_has_call(arg, call_name) for arg in expr.args)
    if isinstance(expr, va_ast.MethodCall):
        return any(_expr_has_call(arg, call_name) for arg in expr.args)
    if isinstance(expr, va_ast.BinaryExpr):
        return (_expr_has_call(expr.left, call_name) or
                _expr_has_call(expr.right, call_name))
    if isinstance(expr, va_ast.UnaryExpr):
        return _expr_has_call(expr.operand, call_name)
    if isinstance(expr, va_ast.TernaryExpr):
        return (_expr_has_call(expr.cond, call_name) or
                _expr_has_call(expr.true_expr, call_name) or
                _expr_has_call(expr.false_expr, call_name))
    if isinstance(expr, va_ast.ArrayAccess):
        return _expr_has_call(expr.index, call_name)
    if isinstance(expr, va_ast.BranchAccess):
        return (_expr_has_call(expr.node1_index, call_name) or
                _expr_has_call(expr.node2_index, call_name) or
                _expr_has_call(expr.node1_index2, call_name) or
                _expr_has_call(expr.node2_index2, call_name))
    return False


_SUPPORTED_FUNCTION_CALLS = {
    'transition', 'slew', 'idtmod', 'cross', 'last_crossing',
    'ln', 'log', 'exp', 'sqrt', 'abs', 'pow', 'min', 'max',
    'sin', 'cos', 'tan', 'tanh', 'floor', 'ceil',
    '$ln', '$log', '$exp', '$sqrt', '$abs', '$pow', '$min', '$max',
    '$sin', '$cos', '$tan', '$tanh', '$floor', '$ceil',
    '$rdist_normal', '$random', '$dist_uniform', '$fopen',
}


def _iter_expr_calls(expr):
    """Yield FunctionCall nodes from an expression tree."""
    if expr is None:
        return
    if isinstance(expr, va_ast.FunctionCall):
        yield expr
        for arg in expr.args:
            yield from _iter_expr_calls(arg)
    elif isinstance(expr, va_ast.MethodCall):
        for arg in expr.args:
            yield from _iter_expr_calls(arg)
    elif isinstance(expr, va_ast.BinaryExpr):
        yield from _iter_expr_calls(expr.left)
        yield from _iter_expr_calls(expr.right)
    elif isinstance(expr, va_ast.UnaryExpr):
        yield from _iter_expr_calls(expr.operand)
    elif isinstance(expr, va_ast.TernaryExpr):
        yield from _iter_expr_calls(expr.cond)
        yield from _iter_expr_calls(expr.true_expr)
        yield from _iter_expr_calls(expr.false_expr)
    elif isinstance(expr, va_ast.ArrayAccess):
        yield from _iter_expr_calls(expr.index)
    elif isinstance(expr, va_ast.BranchAccess):
        yield from _iter_expr_calls(expr.node1_index)
        yield from _iter_expr_calls(expr.node2_index)
        yield from _iter_expr_calls(expr.node1_index2)
        yield from _iter_expr_calls(expr.node2_index2)


def _assignment_target_name(assign) -> Optional[str]:
    target = getattr(assign, "target", None)
    if isinstance(target, va_ast.Identifier):
        return target.name
    if isinstance(target, va_ast.ArrayAccess):
        return target.name
    return None


def _validate_transition_statement(stmt, conditional_depth: int = 0,
                                   genvar_names: Optional[set] = None,
                                   in_event: bool = False) -> None:
    """Reject Verilog-A structures known to diverge from Spectre VACOMP."""
    if genvar_names is None:
        genvar_names = set()
    if stmt is None:
        return
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            _validate_transition_statement(child, conditional_depth, genvar_names, in_event)
        return
    if isinstance(stmt, va_ast.Contribution):
        if in_event:
            raise ValueError(
                "Spectre-incompatible Verilog-A: contribution statement "
                "is embedded in an analog event body"
            )
        if conditional_depth > 0 and _expr_has_call(stmt.expr, "transition"):
            raise ValueError(
                "Spectre-incompatible Verilog-A: transition() contribution "
                "is inside a conditional/event/loop/case statement"
            )
        _validate_supported_function_calls(stmt.expr)
        return
    if isinstance(stmt, va_ast.Assignment):
        if isinstance(stmt.target, va_ast.FunctionCall):
            raise ValueError(
                "Spectre-incompatible Verilog-A: standalone function call "
                f"{stmt.target.name}() is not a supported procedural statement"
            )
        if conditional_depth > 0 and _expr_has_call(stmt.value, "transition"):
            raise ValueError(
                "Spectre-incompatible Verilog-A: transition() expression "
                "is inside a conditional/event/loop/case statement"
            )
        _validate_supported_function_calls(stmt.target)
        _validate_supported_function_calls(stmt.value)
        return
    if isinstance(stmt, va_ast.SystemTask):
        for arg in stmt.args:
            _validate_supported_function_calls(arg)
        return
    if isinstance(stmt, va_ast.EventStatement):
        event_is_initial_step = (
            isinstance(stmt.event, va_ast.EventExpr)
            and stmt.event.event_type == va_ast.EventType.INITIAL_STEP
        )
        _validate_transition_statement(
            stmt.body,
            conditional_depth + 1,
            genvar_names,
            False if event_is_initial_step else True,
        )
        return
    if isinstance(stmt, va_ast.IfStatement):
        _validate_supported_function_calls(stmt.cond)
        _validate_transition_statement(stmt.then_body, conditional_depth + 1, genvar_names, in_event)
        _validate_transition_statement(stmt.else_body, conditional_depth + 1, genvar_names, in_event)
        return
    if isinstance(stmt, va_ast.ForStatement):
        loop_var = _assignment_target_name(stmt.init)
        loop_depth = conditional_depth if loop_var in genvar_names else conditional_depth + 1
        _validate_transition_statement(stmt.init, conditional_depth, genvar_names, in_event)
        _validate_supported_function_calls(stmt.cond)
        _validate_transition_statement(stmt.update, conditional_depth, genvar_names, in_event)
        _validate_transition_statement(stmt.body, loop_depth, genvar_names, in_event)
        return
    if isinstance(stmt, va_ast.WhileStatement):
        _validate_supported_function_calls(stmt.cond)
        _validate_transition_statement(stmt.body, conditional_depth + 1, genvar_names, in_event)
        return
    if isinstance(stmt, va_ast.CaseStatement):
        _validate_supported_function_calls(stmt.expr)
        for item in stmt.items:
            for value in item.values:
                _validate_supported_function_calls(value)
            _validate_transition_statement(item.body, conditional_depth + 1, genvar_names, in_event)


def _validate_supported_function_calls(expr) -> None:
    for call in _iter_expr_calls(expr):
        if call.name not in _SUPPORTED_FUNCTION_CALLS:
            raise ValueError(
                "Spectre-incompatible/unsupported Verilog-A function call: "
                f"{call.name}()"
            )


def _iter_contributions(stmt):
    """Yield Contribution nodes from a statement tree."""
    if stmt is None:
        return
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            yield from _iter_contributions(child)
    elif isinstance(stmt, va_ast.Contribution):
        yield stmt
    elif isinstance(stmt, va_ast.EventStatement):
        yield from _iter_contributions(stmt.body)
    elif isinstance(stmt, va_ast.IfStatement):
        yield from _iter_contributions(stmt.then_body)
        yield from _iter_contributions(stmt.else_body)
    elif isinstance(stmt, va_ast.ForStatement):
        yield from _iter_contributions(stmt.body)
    elif isinstance(stmt, va_ast.WhileStatement):
        yield from _iter_contributions(stmt.body)
    elif isinstance(stmt, va_ast.CaseStatement):
        for item in stmt.items:
            yield from _iter_contributions(item.body)


def _contributed_voltage_ports(module) -> set:
    """Collect Verilog-A port names driven by V(port) <+ contributions."""
    if module.analog_block is None:
        return set()
    ports = set(module.ports)
    driven = set()
    for contrib in _iter_contributions(module.analog_block.body):
        branch = contrib.branch
        if branch.access_type.upper() != "V":
            continue
        if branch.node1 in ports:
            driven.add(branch.node1)
    return driven


def _validate_va_spectre_compat(module) -> None:
    """Run small Spectre-compatibility checks that EVAS can validate locally."""
    ports = set(module.ports)
    for param in module.parameters:
        if param.name in ports:
            raise ValueError(
                "Spectre-incompatible Verilog-A: parameter name "
                f"{param.name!r} collides with module port in {module.name!r}"
            )
    if module.analog_block is not None:
        genvar_names = {v.name for v in module.variables if getattr(v, "is_genvar", False)}
        _validate_transition_statement(module.analog_block.body, genvar_names=genvar_names)


def _source_constrained_nodes(netlist: SpectreNetlist) -> set:
    nodes = set()
    for src in netlist.sources:
        nodes.add(src.node_pos)
        if src.node_neg != netlist.ground:
            nodes.add(src.node_neg)
    nodes.discard(netlist.ground)
    return nodes


def _validate_supply_drive_conflicts(instance, module, node_map: Dict[str, str],
                                     source_nodes: set) -> None:
    """Reject a common Spectre rigid-branch-loop pattern.

    A behavioral module should not hard-drive supply-like ports that are already
    constrained by external voltage sources in the testbench.
    """
    for port in _contributed_voltage_ports(module):
        if port.lower() not in _SUPPLY_PORT_NAMES:
            continue
        ext_node = node_map.get(port)
        if ext_node in source_nodes:
            raise ValueError(
                f"instance {instance.name} of {module.name} drives supply port "
                f"{port!r} mapped to externally sourced node {ext_node!r}"
            )


def _expanded_port_count(module) -> int:
    """Count positional instance terminals after expanding vector ports."""
    decl_by_name = {pd.name: pd for pd in module.port_decls}
    count = 0
    for port_name in module.ports:
        pd = decl_by_name.get(port_name)
        if pd and pd.is_array:
            hi = pd.array_hi if pd.array_hi is not None else 0
            lo = pd.array_lo if pd.array_lo is not None else 0
            count += abs(hi - lo) + 1
        else:
            count += 1
    return count


def _validate_instance_arity(instance, module) -> None:
    expected = _expanded_port_count(module)
    actual = len(instance.nodes)
    if actual != expected:
        raise ValueError(
            f"terminal count mismatch for instance {instance.name} of {module.name}: "
            f"{actual} provided, {expected} expected"
        )


# ---------------------------------------------------------------------------
# Build node_map from instance and module port declarations
# ---------------------------------------------------------------------------

def _build_node_map(instance, module) -> Dict[str, str]:
    """Map VA port names to netlist node names using positional matching.

    Uses module.ports for ordering (matches the module declaration order),
    and port_decls for array info.

    For array ports (e.g. DOUT[17:0]), the netlist supplies one node per
    array element in order from high index to low index.
    """
    decl_by_name = {pd.name: pd for pd in module.port_decls}
    node_map = {}
    ni = 0
    netlist_nodes = list(instance.nodes)

    for port_name in module.ports:
        pd = decl_by_name.get(port_name)
        if pd and pd.is_array:
            hi = pd.array_hi if pd.array_hi is not None else 0
            lo = pd.array_lo if pd.array_lo is not None else 0
            indices = range(hi, lo - 1, -1) if hi >= lo else range(hi, lo + 1)
            for idx in indices:
                if ni < len(netlist_nodes):
                    node_map[f'{pd.name}[{idx}]'] = netlist_nodes[ni]
                    ni += 1
        else:
            if ni < len(netlist_nodes):
                node_map[port_name] = netlist_nodes[ni]
                ni += 1

    return node_map


# ---------------------------------------------------------------------------
# Source conversion: Spectre params -> simulator waveform
# ---------------------------------------------------------------------------

def _add_spectre_source(sim: Simulator, src: SpectreSource,
                        ground: str) -> List[str]:
    """Convert a SpectreSource to a simulator waveform and add it.

    Returns a (possibly empty) list of warning strings for degenerate cases.
    """
    node = src.node_pos if src.node_neg == ground else src.node_pos
    params = src.params
    stype = src.source_type
    warn: List[str] = []

    if stype in ('dc', ''):
        sim.add_source(node, dc(float(params.get('dc', 0.0))))

    elif stype == 'pulse':
        v0 = float(params.get('val0', 0.0))
        v1 = float(params.get('val1', 1.0))
        period = float(params.get('period', 0.0))

        if v0 == v1:
            warn.append(f"{src.name}: pulse val0 == val1 == {v0} "
                        f"— treated as DC {v0} V")
            sim.add_source(node, dc(v0))
            return warn

        delay = float(params.get('delay', 0.0))
        rise = float(params.get('rise', 1e-12))
        fall = float(params.get('fall', 1e-12))
        width = params.get('width', None)
        if period <= 0:
            warn.append(f"{src.name}: pulse period not set "
                        "- treated as nonperiodic one-shot pulse")
        duty = float(width) / period if width is not None and period > 0 else 0.5

        sim.add_source(node, pulse(
            v_lo=v0, v_hi=v1, period=period, duty=duty,
            rise=rise, fall=fall, delay=delay,
            width=float(width) if width is not None else None,
        ))

    elif stype == 'pwl':
        wave = params.get('wave', '')
        if isinstance(wave, list):
            vals = wave
        elif isinstance(wave, str) and wave:
            vals = [_parse_suffix_number(t) for t in wave.split()]
            vals = [v for v in vals if v is not None]
        else:
            vals = []
        if not vals:
            raise ValueError(f"{src.name}: PWL wave must contain at least one time/value pair")
        if len(vals) % 2 != 0:
            raise ValueError(f"{src.name}: PWL wave must contain an even number of values")
        times = vals[0::2]
        values = vals[1::2]
        sim.add_source(node, pwl(times, values))

    elif stype in ('sin', 'sine'):
        # Match Spectre transient vsource sine semantics for the supported
        # subset.  `sinedc`/`ampl` are the canonical transient parameters; do
        # not treat small-signal or schematic convenience names such as
        # vo/va/offset/amplitude as transient aliases.
        offset = float(_first_param(
            params,
            'sinedc', 'dc',
            default=0.0,
        ))
        ampl = float(_first_param(
            params,
            'ampl', 'mag',
            default=1.0,
        ))
        freq = float(_first_param(
            params,
            'freq',
            default=0.0,
        ))
        phase = float(_first_param(
            params,
            'phase', 'sinephase', 'phi',
            default=0.0,
        ))

        if freq <= 0:
            warn.append(f"{src.name}: sine freq not set "
                        f"— treated as DC {offset} V")
            sim.add_source(node, dc(offset))
            return warn

        if ampl == 0:
            warn.append(f"{src.name}: sine amplitude=0 — treated as DC {offset} V")
            sim.add_source(node, dc(offset))
            return warn

        sim.add_source(node, sine(offset=offset, amplitude=ampl, freq=freq, phase=phase))

    return warn


# ---------------------------------------------------------------------------
# Engineering number formatting
# ---------------------------------------------------------------------------

_ENG_PREFIXES = [
    (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
    (1, ''), (1e-3, 'm'), (1e-6, 'u'), (1e-9, 'n'),
    (1e-12, 'p'), (1e-15, 'f'),
]


def _eng_format(val: float, unit: str = '') -> str:
    """Format a number in engineering notation with unit."""
    if val == 0:
        return f"0 {unit}".strip()

    abs_val = abs(val)
    for scale, prefix in _ENG_PREFIXES:
        if abs_val >= scale * 0.999:
            scaled = val / scale
            if scaled == int(scaled):
                return f"{int(scaled)} {prefix}{unit}".strip()
            return f"{scaled:.3g} {prefix}{unit}".strip()

    return f"{val:.3g} {unit}".strip()


# ---------------------------------------------------------------------------
# Streamlined log output
# ---------------------------------------------------------------------------

class _Logger:
    """Write to both a file and optionally stdout."""

    def __init__(self, log_file: Optional[TextIO] = None, quiet: bool = False):
        self.log_file = log_file
        self.quiet = quiet

    def write(self, msg: str = '') -> None:
        if not self.quiet:
            print(msg)
        if self.log_file:
            self.log_file.write(msg + '\n')

    def flush(self) -> None:
        if self.log_file:
            self.log_file.flush()


# ---------------------------------------------------------------------------
# Bus detection and derived signals
# ---------------------------------------------------------------------------

def _find_buses(result: SimResult) -> Dict[str, Dict[int, str]]:
    """Find bus-like signal groups (name_N, name_N-1, ..., name_0).

    Returns {prefix: {index: signal_name}} for contiguous buses starting at 0.
    """
    groups: Dict[str, Dict[int, str]] = {}
    for name in result.signals:
        m = _re.match(r'^(.+?)_(\d+)$', name)
        if m:
            prefix, idx = m.group(1), int(m.group(2))
            groups.setdefault(prefix, {})[idx] = name

    buses: Dict[str, Dict[int, str]] = {}
    for prefix, bits in groups.items():
        indices = sorted(bits.keys())
        if len(indices) < 2:
            continue
        if indices[0] != 0 or indices != list(range(len(indices))):
            continue
        buses[prefix] = bits
    return buses


def _derive_bus_signals(result: SimResult) -> Dict[str, np.ndarray]:
    """Compute combined integer-valued code signals for detected buses."""
    derived: Dict[str, np.ndarray] = {}
    for prefix, bits in _find_buses(result).items():
        indices = sorted(bits.keys())
        vdd = max(float(np.max(result.signals[bits[idx]])) for idx in indices)
        if vdd == 0:
            continue
        combined = np.zeros_like(result.time)
        for idx in indices:
            combined += (result.signals[bits[idx]] / vdd) * (2 ** idx)
        derived[f'{prefix}_code'] = combined
    return derived


def _dedupe_signal_names(names: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _normalize_trace_signal_name(name: str) -> str:
    text = name.strip()
    vm = _re.match(r"(?i)^v\(\s*([^)]+)\s*\)$", text)
    if vm:
        text = vm.group(1).strip()
    return text


def _parse_required_trace_signals(simopt: Dict[str, object]) -> List[str]:
    """Return an optional harness-provided sparse trace contract.

    EVAS normally follows Spectre `save` statements, or records every node when
    no save list exists.  The benchmark harness can set this contract when the
    checker only needs a small observable subset; this keeps CLI/default EVAS
    behavior unchanged while allowing paper speed runs to avoid unnecessary
    trace arrays and CSV columns.
    """
    value = simopt.get("evas_required_trace_signals")
    if value is None:
        value = os.environ.get("EVAS_REQUIRED_TRACE_SIGNALS", "")
    if isinstance(value, (list, tuple, set)):
        raw_names = [str(item) for item in value]
    else:
        raw_names = [part for part in _re.split(r"[\s,;]+", str(value)) if part]
    normalized = [
        _normalize_trace_signal_name(name)
        for name in raw_names
    ]
    return _dedupe_signal_names([
        name
        for name in normalized
        if name and name.lower() != "time"
    ])


def _trace_nodes_for_signals(required_signals: List[str], all_nodes: set) -> List[str]:
    if not required_signals:
        return []
    lower_to_node = {str(node).lower(): node for node in all_nodes}
    nodes: List[str] = []
    for signal in required_signals:
        if signal in all_nodes:
            nodes.append(signal)
            continue
        node = lower_to_node.get(signal.lower())
        if node is not None:
            nodes.append(node)
    return _dedupe_signal_names(nodes)


def _trace_output_signals_for_request(required_signals: List[str], available_signals: set) -> List[str]:
    lower_to_signal = {str(signal).lower(): signal for signal in available_signals}
    selected: List[str] = []
    for signal in required_signals:
        if signal in available_signals:
            selected.append(signal)
            continue
        actual = lower_to_signal.get(signal.lower())
        if actual is not None:
            selected.append(actual)
    return _dedupe_signal_names(selected)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _fmt_value(v: float, fmt: str) -> str:
    """Format a single value according to format string ('6e', '4f', 'd', etc.)."""
    if fmt == 'd':
        return str(int(round(v)))
    if fmt.endswith('f') or fmt.endswith('F'):
        try:
            n = int(fmt[:-1])
        except ValueError:
            n = 6
        return f"{v:.{n}f}"
    try:
        n = int(fmt.rstrip('eE'))
    except ValueError:
        n = 6
    return f"{v:.{n}e}"


def _csv_numpy_format(fmt: str) -> str:
    if fmt == 'd':
        return '%d'
    if fmt.endswith('f') or fmt.endswith('F'):
        try:
            n = int(fmt[:-1])
        except ValueError:
            n = 6
        return f'%.{n}f'
    try:
        n = int(fmt.rstrip('eE'))
    except ValueError:
        n = 6
    return f'%.{n}e'


def _write_csv_python(csv_path: Path, result: SimResult, valid_signals: List[str],
                      signal_arrays: List[np.ndarray], signal_formats: List[str]) -> None:
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time'] + valid_signals)
        for i, t_value in enumerate(result.time):
            row = [f"{t_value:.12e}"]
            for values, fmt in zip(signal_arrays, signal_formats):
                row.append(_fmt_value(values[i], fmt))
            writer.writerow(row)


def _write_csv(csv_path: Path, result: SimResult, save_signals: List[str],
               save_formats: Dict[str, str] = None) -> None:
    """Write simulation results to CSV file."""
    if save_formats is None:
        save_formats = {}
    signal_names = save_signals if save_signals else sorted(result.signals.keys())
    valid_signals = [s for s in signal_names if s in result.signals]
    signal_arrays = [result.signals[s] for s in valid_signals]
    signal_formats = [
        save_formats.get(sig, 'd' if sig.endswith('_code') else '6e')
        for sig in valid_signals
    ]

    if os.environ.get("EVAS_CSV_WRITER", "").strip().lower() == "python":
        _write_csv_python(csv_path, result, valid_signals, signal_arrays, signal_formats)
        return

    columns = [result.time]
    for values, fmt in zip(signal_arrays, signal_formats):
        if fmt == 'd':
            columns.append(np.rint(values))
        else:
            columns.append(values)
    matrix = np.column_stack(columns)
    formats = ['%.12e'] + [_csv_numpy_format(fmt) for fmt in signal_formats]
    header = ','.join(['time'] + valid_signals)
    np.savetxt(
        csv_path,
        matrix,
        delimiter=',',
        fmt=formats,
        header=header,
        comments='',
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# Collect all nodes from the netlist
# ---------------------------------------------------------------------------

def _collect_nodes(netlist: SpectreNetlist) -> set:
    """Gather all non-ground nodes from sources and instances."""
    nodes = set()
    for src in netlist.sources:
        nodes.add(src.node_pos)
        nodes.add(src.node_neg)
    for inst in netlist.instances:
        nodes.update(inst.nodes)
    nodes.discard(netlist.ground)
    return nodes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evas_simulate(scs_file: str, log_path: Optional[str] = None,
                output_dir: str = './output',
                strobe_log_path: Optional[str] = None) -> bool:
    """Run an EVAS .scs netlist. Returns True on success.

    Args:
        scs_file:        Path to the .scs netlist file.
        log_path:        Optional path for the simulation log. If None, log goes to stdout.
        output_dir:      Directory for output files (CSV, strobe log).
        strobe_log_path: Path for $strobe/$display output. Defaults to <output_dir>/strobe.txt.
    """
    scs_path = Path(scs_file).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, 'w', encoding='utf-8') if log_path else None
    log = _Logger(log_file, quiet=(log_path is not None))
    errors = 0
    warnings = 0

    t_total_start = time.time()

    # Banner
    now = datetime.now()
    timestamp = now.strftime("%I:%M:%S %p, %a %b %d, %Y").lstrip('0')
    log.write("EVAS -- Event-driven Verilog-A Simulator")
    log.write(f"Version {VERSION} -- {now.strftime('%b %Y')}")
    log.write("")
    log.write(f"Simulating `{scs_path.name}' at {timestamp}.")
    log.write("Command line:")
    log.write(f"    {' '.join(sys.argv)}")
    log.write("")

    # 1. Parse netlist
    log.write(f"Reading file: {scs_path.name}")
    try:
        netlist = parse_spectre(str(scs_path))
    except Exception as e:
        log.write(f"ERROR: Failed to parse {scs_path.name}: {e}")
        if log_file:
            log_file.close()
        return False

    if has_transistors(netlist):
        log.write("ERROR: Netlist contains transistor-level devices.")
        log.write("       EVAS only supports behavioral Verilog-A models.")
        if log_file:
            log_file.close()
        return False

    simopt = netlist.simulator_options or {}
    static_branch_fastpath = _simopt_bool(
        simopt,
        'evas_static_branch_fastpath',
        False,
    ) or os.environ.get("EVAS_STATIC_BRANCH_FASTPATH", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_state_storage_requested = _simopt_bool(
        simopt,
        'evas_indexed_state_storage',
        False,
    ) or os.environ.get("EVAS_INDEXED_STATE_STORAGE", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    state_local_fastpath = _simopt_bool(
        simopt,
        'evas_state_local_fastpath',
        False,
    ) or _simopt_bool(
        simopt,
        'evas_indexed_state_fastpath',
        False,
    ) or os.environ.get("EVAS_STATE_LOCAL_FASTPATH", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_state_fastpath_env = os.environ.get(
        "EVAS_INDEXED_STATE_FASTPATH",
        "",
    ).strip().lower()
    if indexed_state_fastpath_env:
        state_local_fastpath = indexed_state_fastpath_env in {
            "1", "true", "yes", "on", "enabled"
        }

    # 2. Compile VA models
    models_by_name = {}
    for inc in netlist.ahdl_includes:
        # Three-level path search so Virtuoso-exported absolute paths work
        # even when the netlist is used on a different machine:
        #   1. Path as written (works if absolute and reachable, or relative to cwd)
        #   2. Relative to the .scs source directory
        #   3. Just the filename in the .scs source directory (cross-machine fallback)
        p = Path(inc.path)
        scs_dir = Path(netlist.source_dir)
        candidates: List[Path] = []
        for c in [p, scs_dir / p, scs_dir / p.name]:
            r = c.resolve()
            if r not in candidates:
                candidates.append(r)

        va_path = next((c for c in candidates if c.exists()), None)
        if va_path is None:
            searched = ', '.join(str(c) for c in candidates)
            log.write(f"ERROR: Cannot find VA file: {inc.path!r}")
            log.write(f"       Searched: {searched}")
            errors += 1
            continue

        original = p.resolve()
        if va_path != original:
            log.write(f"WARNING: ahdl_include resolved to '{va_path.name}' "
                      f"(original path not found, used scs directory fallback)")
            warnings += 1

        try:
            cls, module = _compile_va(
                str(va_path),
                static_branch_fastpath_codegen=static_branch_fastpath,
                indexed_state_fastpath_codegen=state_local_fastpath,
            )
        except Exception as e:
            log.write(f"ERROR: Failed to compile Verilog-A file {va_path.name}: {e}")
            errors += 1
            continue
        models_by_name[module.name] = (cls, module)
        log.write(f"Compiled Verilog-A module: {module.name}")
        for w in module.warnings:
            log.write(f"WARNING ({module.name}): {w}")
            warnings += 1

    # Provide compiled-module registry for hierarchical Verilog-A instances.
    # Each class can instantiate child modules from this table at runtime.
    for _mname, (_cls, _module) in models_by_name.items():
        _cls._module_registry = models_by_name

    if errors > 0:
        log.write(f"\nevas completes with {errors} errors, {warnings} warnings.")
        if log_file:
            log_file.close()
        return False

    # 3. Build simulator
    sim = Simulator()

    for src in netlist.sources:
        try:
            src_warnings = _add_spectre_source(sim, src, netlist.ground)
        except ValueError as e:
            log.write(f"ERROR: Invalid source {src.name}: {e}")
            errors += 1
            continue
        for w in src_warnings:
            log.write(f"WARNING: {w}")
            warnings += 1

    if errors > 0:
        log.write(f"\nevas completes with {errors} errors, {warnings} warnings.")
        if log_file:
            log_file.close()
        return False

    # 4. Instantiate models
    instance_counts: Dict[str, int] = {}
    source_nodes = _source_constrained_nodes(netlist)
    for inst in netlist.instances:
        if inst.model_name not in models_by_name:
            log.write(f"ERROR: Model {inst.model_name} not found "
                      f"(available: {list(models_by_name.keys())})")
            errors += 1
            continue

        cls, module = models_by_name[inst.model_name]
        model = cls()
        try:
            _validate_instance_arity(inst, module)
            node_map = _build_node_map(inst, module)
            _validate_supply_drive_conflicts(inst, module, node_map, source_nodes)
        except ValueError as e:
            log.write(f"ERROR: Spectre-incompatible instance {inst.name}: {e}")
            errors += 1
            continue
        model.node_map = node_map
        # Case-insensitive param update: netlist keys are lowercased, model keys
        # preserve the original VA case. Match by lowercase to update correctly.
        lower_to_model_key = {k.lower(): k for k in model.params}
        param_types = {p.name.lower(): p.param_type for p in module.parameters}
        for k, v in inst.params.items():
            model_key = lower_to_model_key.get(k.lower(), k)
            if param_types.get(model_key.lower()) == va_ast.ParamType.INTEGER:
                v = int(float(v))
            elif param_types.get(model_key.lower()) == va_ast.ParamType.STRING and isinstance(v, str):
                if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
                    v = v[1:-1]
            model.params[model_key] = v
        sim.add_model(model)
        instance_counts[inst.model_name] = instance_counts.get(inst.model_name, 0) + 1

    # Circuit inventory
    all_nodes = _collect_nodes(netlist)
    log.write("")
    log.write("Circuit inventory:")
    log.write(f"{'nodes':>20s} {len(all_nodes)}")
    for mname, cnt in instance_counts.items():
        log.write(f"{mname:>20s} {cnt}")
    log.write(f"{'vsource':>20s} {len(netlist.sources)}")

    # 5. Record signals
    required_trace_signals = _parse_required_trace_signals(simopt)
    required_trace_nodes = _trace_nodes_for_signals(required_trace_signals, all_nodes)
    if required_trace_nodes:
        record_nodes = set(required_trace_nodes)
    else:
        record_nodes = set(netlist.save_signals) if netlist.save_signals else all_nodes
    if record_nodes:
        sim.record(*sorted(record_nodes))
    if required_trace_signals:
        record_node_lower = {node.lower() for node in required_trace_nodes}
        missing_trace = [
            signal
            for signal in required_trace_signals
            if signal not in required_trace_nodes
            and signal.lower() not in record_node_lower
        ]
        log.write("Trace counters:")
        log.write(f"    required_trace_signal_count = {len(required_trace_signals)}")
        log.write(f"    required_trace_record_node_count = {len(required_trace_nodes)}")
        log.write(f"    required_trace_missing_node_count = {len(missing_trace)}")

    # 6. Run simulation
    if netlist.tran is None:
        log.write("ERROR: No transient analysis found")
        if log_file:
            log_file.close()
        return False

    tstop = netlist.tran.stop
    tstep = netlist.tran.step
    reltol = float(simopt.get('reltol', 1e-3))
    vabstol = float(simopt.get('vabstol', 1e-6))
    iabstol = float(simopt.get('iabstol', 1e-12))
    maxstep_opt = simopt.get('maxstep', None)
    if maxstep_opt is not None:
        try:
            tstep = min(float(tstep), float(maxstep_opt))
        except Exception:
            pass

    refine_factor = netlist.tran.refine_factor
    refine_steps = netlist.tran.refine_steps
    errpreset = str(netlist.tran.__dict__.get('errpreset', simopt.get('errpreset', ''))).lower()
    if errpreset == 'conservative':
        refine_factor = max(refine_factor, 32)
        refine_steps = max(refine_steps, 16)
    elif errpreset == 'liberal':
        refine_factor = min(refine_factor, 8)
        refine_steps = min(refine_steps, 4)
    # Spectre-compatible "accepted" cross event timing is an explicit opt-in
    # experiment (2026-04 closure decision: exact/analytic stays the default
    # and benchmark flows must not enable tolerance-compatible behavior).
    # The lateness model is the measured Spectre law (cross-lateness DOE,
    # 2026-06-12):  delta = factor * 0.5 * reltol * |V_cross| / |slope|,
    # with factor=1.0 reproducing Spectre's observed behavior.
    # Example: simulatorOptions options evas_cross_acceptance_slack_factor=1.0
    try:
        cross_acceptance_user_factor = max(
            0.0, float(simopt.get('evas_cross_acceptance_slack_factor', 0.0) or 0.0)
        )
    except (TypeError, ValueError):
        cross_acceptance_user_factor = 0.0

    evas_profile = str(simopt.get('evas_profile', '')).lower()
    refine_factor, refine_steps, reltol, applied_profile = _apply_evas_profile(
        evas_profile, refine_factor, refine_steps, reltol
    )
    skip_source_error_control = _simopt_bool(
        simopt,
        'evas_skip_source_error_control',
        False,
    )
    profile_sections = _simopt_bool(
        simopt,
        'evas_profile_sections',
        False,
    ) or os.environ.get("EVAS_PROFILE_SECTIONS", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    profile_model_eval = _simopt_bool(
        simopt,
        'evas_profile_model_eval',
        False,
    ) or os.environ.get("EVAS_PROFILE_MODEL_EVAL", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    profile_model_io = _simopt_bool(
        simopt,
        'evas_profile_model_io',
        False,
    ) or os.environ.get("EVAS_PROFILE_MODEL_IO", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_parity = _simopt_bool(
        simopt,
        'evas_indexed_parity',
        False,
    ) or os.environ.get("EVAS_INDEXED_PARITY", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_snapshot_profile = _simopt_bool(
        simopt,
        'evas_indexed_snapshot_profile',
        False,
    ) or os.environ.get("EVAS_INDEXED_SNAPSHOT_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_arrays = _simopt_bool(
        simopt,
        'evas_indexed_arrays',
        False,
    ) or os.environ.get("EVAS_INDEXED_ARRAYS", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    indexed_state_storage = indexed_state_storage_requested
    static_lifecycle_fastpath_env = (
        os.environ.get("EVAS_STATIC_LIFECYCLE_FASTPATH", "").strip().lower()
    )
    static_lifecycle_fastpath = _simopt_bool(
        simopt,
        'evas_static_lifecycle_fastpath',
        True,
    )
    if static_lifecycle_fastpath_env:
        static_lifecycle_fastpath = static_lifecycle_fastpath_env in {
            "1", "true", "yes", "on", "enabled"
        }
    transition_unchanged_fastpath_env = (
        os.environ.get("EVAS_TRANSITION_UNCHANGED_FASTPATH", "").strip().lower()
    )
    transition_unchanged_fastpath = _simopt_bool(
        simopt,
        'evas_transition_unchanged_fastpath',
        False,
    )
    if transition_unchanged_fastpath_env:
        transition_unchanged_fastpath = transition_unchanged_fastpath_env in {
            "1", "true", "yes", "on", "enabled"
        }
    rust_static_eval = _simopt_bool(
        simopt,
        'evas_rust_static_eval',
        False,
    ) or os.environ.get("EVAS_RUST_STATIC_EVAL", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_static_fast_sync = _simopt_bool(
        simopt,
        'evas_rust_static_fast_sync',
        False,
    ) or os.environ.get("EVAS_RUST_STATIC_FAST_SYNC", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    if rust_static_fast_sync:
        rust_static_eval = True
    rust_transition_shadow = _simopt_bool(
        simopt,
        'evas_rust_transition_shadow',
        False,
    ) or os.environ.get("EVAS_RUST_TRANSITION_SHADOW", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_event_due_shadow = _simopt_bool(
        simopt,
        'evas_rust_event_due_shadow',
        False,
    ) or os.environ.get("EVAS_RUST_EVENT_DUE_SHADOW", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_event_write_shadow = _simopt_bool(
        simopt,
        'evas_rust_event_write_shadow',
        False,
    ) or os.environ.get("EVAS_RUST_EVENT_WRITE_SHADOW", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_event_write_production = _simopt_bool(
        simopt,
        'evas_rust_event_write_production',
        False,
    ) or os.environ.get("EVAS_RUST_EVENT_WRITE_PRODUCTION", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_timer_event = _simopt_bool(
        simopt,
        'evas_rust_timer_event',
        False,
    ) or os.environ.get("EVAS_RUST_TIMER_EVENT", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_full_model_fastpath = _simopt_bool(
        simopt,
        'evas_rust_full_model_fastpath',
        False,
    ) or os.environ.get("EVAS_RUST_FULL_MODEL_FASTPATH", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    evas_engine = _configured_evas_engine(simopt)
    evas_rust_engine = (
        evas_engine == RUST_EVAS_ENGINE
        or _simopt_bool(simopt, "evas2", False)
    )
    rust_full_model_required = _simopt_bool(
        simopt,
        'evas_rust_full_model_required',
        False,
    ) or os.environ.get("EVAS_RUST_FULL_MODEL_REQUIRED", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    event_trace_audit = _simopt_bool(
        simopt,
        'evas_event_trace_audit',
        False,
    ) or os.environ.get("EVAS_EVENT_TRACE_AUDIT", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    rust_required = _simopt_bool(
        simopt,
        'evas_rust_required',
        False,
    ) or os.environ.get("EVAS_RUST_REQUIRED", "").strip().lower() in {
        "1", "true", "yes", "on", "enabled"
    }
    if evas_rust_engine:
        rust_full_model_fastpath = True
        rust_full_model_required = True
        rust_required = True
    indexed_arrays_effective = (
        indexed_arrays
        or rust_static_eval
        or rust_transition_shadow
    )
    indexed_plan = None
    if indexed_parity:
        indexed_plan = build_indexed_run_plan(
            sim,
            extra_nodes=sorted(all_nodes | record_nodes),
        )

    log.write("")
    log.write("*****************************************************")
    log.write(f"Transient Analysis `{netlist.tran.name}': "
              f"time = (0 s -> {_eng_format(tstop, 's')})")
    log.write("*****************************************************")
    log.write("Important parameter values:")
    log.write("    start = 0 s")
    log.write(f"    stop  = {_eng_format(tstop, 's')}")
    log.write(f"    step  = {_eng_format(tstep, 's')}")
    log.write(f"    reltol = {reltol:g}")
    log.write(f"    vabstol = {vabstol:g}")
    log.write(f"    iabstol = {iabstol:g}")
    log.write(f"    refine_factor = {refine_factor}")
    log.write(f"    refine_steps  = {refine_steps}")
    if applied_profile:
        log.write(f"    evas_profile = {applied_profile}")
    if skip_source_error_control:
        log.write("    evas_skip_source_error_control = true")
    if profile_sections:
        log.write("    evas_profile_sections = true")
    if profile_model_eval:
        log.write("    evas_profile_model_eval = true")
    if profile_model_io:
        log.write("    evas_profile_model_io = true")
    if indexed_parity:
        log.write("    evas_indexed_parity = true")
        log.write(f"    indexed_node_count = {indexed_plan.node_count}")
    if indexed_snapshot_profile:
        log.write("    evas_indexed_snapshot_profile = true")
    if indexed_arrays_effective:
        log.write("    evas_indexed_arrays = true")
    if indexed_state_storage:
        log.write("    evas_indexed_state_storage = true")
    if state_local_fastpath:
        log.write("    evas_state_local_fastpath = true")
    if static_branch_fastpath:
        log.write("    evas_static_branch_fastpath = true")
    if not static_lifecycle_fastpath:
        log.write("    evas_static_lifecycle_fastpath = false")
    if transition_unchanged_fastpath:
        log.write("    evas_transition_unchanged_fastpath = true")
    if rust_static_eval:
        log.write("    evas_rust_static_eval = true")
    if rust_static_fast_sync:
        log.write("    evas_rust_static_fast_sync = true")
    if rust_transition_shadow:
        log.write("    evas_rust_transition_shadow = true")
    if rust_event_due_shadow:
        log.write("    evas_rust_event_due_shadow = true")
    if rust_event_write_shadow:
        log.write("    evas_rust_event_write_shadow = true")
    if rust_event_write_production:
        log.write("    evas_rust_event_write_production = true")
    if rust_timer_event:
        log.write("    evas_rust_timer_event = true")
    if rust_full_model_fastpath:
        log.write("    evas_rust_full_model_fastpath = true")
    if rust_full_model_required:
        log.write("    evas_rust_full_model_required = true")
    if evas_rust_engine:
        log.write(f"    evas_engine = {RUST_EVAS_ENGINE}")
    if event_trace_audit:
        log.write("    evas_event_trace_audit = true")
    if rust_required:
        log.write("    evas_rust_required = true")
    # Engine-level coefficient kappa: slack = kappa * |V_cross| / |slope|.
    # Folding 0.5 * reltol here keeps the FFI channel a single scalar while
    # the user-facing factor stays a multiple of the measured Spectre law.
    cross_acceptance_slack_factor = cross_acceptance_user_factor * 0.5 * reltol
    if cross_acceptance_user_factor:
        log.write(
            f"    evas_cross_acceptance_slack_factor = {cross_acceptance_user_factor:g}"
            f" (kappa = {cross_acceptance_slack_factor:g})"
        )
    log.write("")

    t_sim_start = time.time()
    result = sim.run(tstop, tstep=tstep,
                     refine_factor=refine_factor,
                     refine_steps=refine_steps,
                     reltol=reltol,
                     vabstol=vabstol,
                     record_step=tstep,
                     skip_source_error_control=skip_source_error_control,
                     profile_sections=profile_sections,
                     profile_model_eval=profile_model_eval,
                     profile_model_io=profile_model_io,
                     indexed_snapshot_profile=indexed_snapshot_profile,
                     indexed_arrays=indexed_arrays_effective,
                     indexed_state_storage=indexed_state_storage,
                     static_branch_fastpath=static_branch_fastpath,
                     static_lifecycle_fastpath=static_lifecycle_fastpath,
                     transition_unchanged_fastpath=transition_unchanged_fastpath,
                     rust_static_eval=rust_static_eval,
                     rust_static_fast_sync=rust_static_fast_sync,
                     rust_transition_shadow=rust_transition_shadow,
                     rust_event_due_shadow=rust_event_due_shadow,
                     rust_timer_event=rust_timer_event,
                     rust_event_write_shadow=rust_event_write_shadow,
                     rust_event_write_production=rust_event_write_production,
                     rust_full_model_fastpath=rust_full_model_fastpath,
                     rust_full_model_required=rust_full_model_required,
                     event_trace_audit=event_trace_audit,
                     cross_acceptance_slack_factor=cross_acceptance_slack_factor,
                     rust_required=rust_required)

    for pct in range(10, 101, 10):
        t_at = tstop * pct / 100.0
        log.write(f"    tran: time = {_eng_format(t_at, 's'):12s} ({pct:3d} %)")

    sim_elapsed_ms = (time.time() - t_sim_start) * 1000
    n_steps = len(result.time) - 1
    log.write(f"Number of accepted tran steps = {n_steps}")
    if getattr(sim, "_perf_stats", None):
        log.write("Performance counters:")
        for key, value in sorted(sim._perf_stats.items()):
            log.write(f"    {key} = {value}")
    model_perf_lines = []
    for idx, model in enumerate(sim.models):
        perf = getattr(model, "_perf_stats", None)
        if not perf:
            continue
        model_name = getattr(model, "__class__", type(model)).__name__
        model_perf_lines.append(f"    model[{idx}] {model_name}:")
        for key, value in sorted(perf.items()):
            model_perf_lines.append(f"        {key} = {value}")
    if model_perf_lines:
        log.write("Model event counters:")
        for line in model_perf_lines:
            log.write(line)
    if getattr(sim, "_profile_times", None):
        log.write("Section timing counters:")
        for key, value in sorted(sim._profile_times.items()):
            log.write(f"    {key} = {value:.6f} s")
    if getattr(sim, "_model_profile_stats", None):
        log.write("Model timing counters:")
        for model_key, stats in sorted(sim._model_profile_stats.items()):
            log.write(f"    {model_key}:")
            for key, value in sorted(stats.items()):
                if key.endswith("_s"):
                    log.write(f"        {key} = {value:.6f} s")
                else:
                    log.write(f"        {key} = {int(value)}")
    if getattr(sim, "_model_io_profile_stats", None):
        log.write("Model IO counters:")
        for key, value in sorted(sim._model_io_profile_stats.items()):
            log.write(f"    {key} = {value}")
    if getattr(sim, "_indexed_snapshot_stats", None):
        log.write("Indexed snapshot profile:")
        for key, value in sorted(sim._indexed_snapshot_stats.items()):
            log.write(f"    {key} = {value}")
    if getattr(sim, "_indexed_array_stats", None):
        log.write("Indexed array profile:")
        for key, value in sorted(sim._indexed_array_stats.items()):
            log.write(f"    {key} = {value}")
    if getattr(sim, "_indexed_model_io_stats", None):
        log.write("Indexed model IO plan:")
        for key, value in sorted(sim._indexed_model_io_stats.items()):
            log.write(f"    {key} = {value}")
    if getattr(sim, "_indexed_voltage_probe_stats", None):
        log.write("Indexed voltage read probe:")
        for key, value in sorted(sim._indexed_voltage_probe_stats.items()):
            log.write(f"    {key} = {value}")
    if getattr(sim, "_indexed_voltage_read_stats", None):
        log.write("Indexed voltage array reads:")
        for key, value in sorted(sim._indexed_voltage_read_stats.items()):
            log.write(f"    {key} = {value}")

    # Signal range summary
    log.write("")
    log.write("Maximum value achieved for any signal of each quantity:")
    max_v = 0.0
    max_v_name = ''
    for name, data in result.signals.items():
        peak = float(np.max(np.abs(data)))
        if peak > max_v:
            max_v = peak
            max_v_name = name
    if max_v_name:
        log.write(f"    V: V({max_v_name}) = {_eng_format(max_v, 'V')}")

    log.write("")
    log.write(f"Tran analysis time: CPU = {sim_elapsed_ms:.1f} ms, "
              f"elapsed = {sim_elapsed_ms:.1f} ms.")

    # 6b. Derive combined bus signals (e.g. dout_3..dout_0 -> dout_code)
    t_derive_start = time.time()
    derived = _derive_bus_signals(result)
    derive_elapsed = time.time() - t_derive_start
    result.signals.update(derived)
    if required_trace_signals:
        save_with_derived = _trace_output_signals_for_request(
            required_trace_signals,
            set(result.signals.keys()),
        )
    else:
        save_with_derived = list(netlist.save_signals) + list(derived.keys())

    if indexed_parity:
        report = check_indexed_trace_round_trip(
            result,
            node_index=indexed_plan.node_index if indexed_plan else None,
            signal_names=save_with_derived if save_with_derived else sorted(result.signals.keys()),
        )
        log.write("Indexed parity check:")
        log.write(f"    {report.summary()}")
        if report.length_mismatches:
            log.write(
                "    length_mismatches = "
                f"{', '.join(report.length_mismatches)}"
            )
        if report.missing_signals:
            log.write(
                "    missing_requested_signals = "
                f"{', '.join(report.missing_signals)}"
            )
        if not report.passed:
            log.write("ERROR: Indexed parity check failed")
            errors += 1

    # 7. Write CSV
    csv_path = out_dir / 'tran.csv'
    t_csv_start = time.time()
    _write_csv(csv_path, result, save_with_derived, netlist.save_formats)
    csv_elapsed = time.time() - t_csv_start

    signal_names = save_with_derived if save_with_derived else sorted(result.signals.keys())
    valid_signal_names = [name for name in signal_names if name in result.signals]
    log.write("Runner timing counters:")
    log.write(f"    derive_bus_signals_s = {derive_elapsed:.6f} s")
    log.write(f"    csv_write_s = {csv_elapsed:.6f} s")
    if required_trace_signals:
        log.write("Trace counters:")
        log.write(f"    required_trace_csv_signal_count = {len(valid_signal_names)}")
    log.write("")
    log.write(f"Writing CSV: {csv_path} "
              f"(signals: {', '.join(valid_signal_names)})")

    # 8. Collect $strobe / $display output (sorted by simulation time)
    strobe_entries = []
    for model in sim.models:
        strobe_entries.extend(model._strobe_log)
    strobe_entries.sort(key=lambda x: x[0])
    strobe_lines = [msg for _, msg in strobe_entries]

    if strobe_lines:
        s_path = Path(strobe_log_path) if strobe_log_path else out_dir / 'strobe.txt'
        s_path.write_text('\n'.join(strobe_lines) + '\n', encoding='utf-8')
        log.write(f"Writing strobe log: {s_path} ({len(strobe_lines)} lines)")
        print('\n'.join(strobe_lines))

    # Final summary
    total_elapsed = time.time() - t_total_start
    log.write("")
    log.write(f"evas completes with {errors} errors, {warnings} warnings.")
    log.write(f"Total time: CPU = {total_elapsed:.1f} s, "
              f"elapsed = {total_elapsed:.1f} s.")

    if log_file:
        log_file.close()

    return errors == 0
