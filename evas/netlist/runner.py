"""
runner.py -- EVAS runner: parse .scs, compile VA, simulate, produce log + CSV.
"""

import csv
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

from .spectre_parser import (
    SpectreNetlist,
    SpectreSource,
    _parse_suffix_number,
    has_transistors,
    parse_spectre,
)

VERSION = '0.1.0'

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


# ---------------------------------------------------------------------------
# VA model compilation
# ---------------------------------------------------------------------------

def _compile_va(va_path: str, source_dir: str = None):
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
    cls = compile_module(module, default_trans)
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


def _assignment_target_name(assign) -> Optional[str]:
    target = getattr(assign, "target", None)
    if isinstance(target, va_ast.Identifier):
        return target.name
    if isinstance(target, va_ast.ArrayAccess):
        return target.name
    return None


def _validate_transition_statement(stmt, conditional_depth: int = 0,
                                   genvar_names: Optional[set] = None) -> None:
    """Reject transition() where Spectre's analog-operator rules reject it."""
    if genvar_names is None:
        genvar_names = set()
    if stmt is None:
        return
    if isinstance(stmt, va_ast.Block):
        for child in stmt.statements:
            _validate_transition_statement(child, conditional_depth, genvar_names)
        return
    if isinstance(stmt, va_ast.Contribution):
        if conditional_depth > 0 and _expr_has_call(stmt.expr, "transition"):
            raise ValueError(
                "Spectre-incompatible Verilog-A: transition() contribution "
                "is inside a conditional/event/loop/case statement"
            )
        return
    if isinstance(stmt, va_ast.Assignment):
        if conditional_depth > 0 and _expr_has_call(stmt.value, "transition"):
            raise ValueError(
                "Spectre-incompatible Verilog-A: transition() expression "
                "is inside a conditional/event/loop/case statement"
            )
        return
    if isinstance(stmt, va_ast.SystemTask):
        return
    if isinstance(stmt, va_ast.EventStatement):
        _validate_transition_statement(stmt.body, conditional_depth, genvar_names)
        return
    if isinstance(stmt, va_ast.IfStatement):
        _validate_transition_statement(stmt.then_body, conditional_depth, genvar_names)
        _validate_transition_statement(stmt.else_body, conditional_depth, genvar_names)
        return
    if isinstance(stmt, va_ast.ForStatement):
        loop_var = _assignment_target_name(stmt.init)
        loop_depth = conditional_depth if loop_var in genvar_names else conditional_depth + 1
        _validate_transition_statement(stmt.body, loop_depth, genvar_names)
        return
    if isinstance(stmt, va_ast.WhileStatement):
        _validate_transition_statement(stmt.body, conditional_depth + 1, genvar_names)
        return
    if isinstance(stmt, va_ast.CaseStatement):
        for item in stmt.items:
            _validate_transition_statement(item.body, conditional_depth + 1, genvar_names)


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

        if period <= 0:
            warn.append(f"{src.name}: pulse period not set "
                        f"— treated as DC {v1} V")
            sim.add_source(node, dc(v1))
            return warn

        delay = float(params.get('delay', 0.0))
        rise = float(params.get('rise', 1e-12))
        fall = float(params.get('fall', 1e-12))
        width = params.get('width', None)
        duty = float(width) / period if width is not None and period > 0 else 0.5

        sim.add_source(node, pulse(
            v_lo=v0, v_hi=v1, period=period, duty=duty,
            rise=rise, fall=fall, delay=delay,
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
        offset = float(params.get('sinedc', params.get('offset', params.get('dc', 0.0))))
        ampl = float(params.get('ampl', params.get('mag', params.get('amplitude', 0.0))))
        freq = float(params.get('freq', 0.0))

        if freq <= 0:
            warn.append(f"{src.name}: sine freq not set "
                        f"— treated as DC {offset} V")
            sim.add_source(node, dc(offset))
            return warn

        if ampl == 0:
            warn.append(f"{src.name}: sine amplitude=0 — treated as DC {offset} V")
            sim.add_source(node, dc(offset))
            return warn

        sim.add_source(node, sine(offset=offset, amplitude=ampl, freq=freq))

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


def _write_csv(csv_path: Path, result: SimResult, save_signals: List[str],
               save_formats: Dict[str, str] = None) -> None:
    """Write simulation results to CSV file."""
    if save_formats is None:
        save_formats = {}
    signal_names = save_signals if save_signals else sorted(result.signals.keys())
    valid_signals = [s for s in signal_names if s in result.signals]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time'] + valid_signals)
        for i in range(len(result.time)):
            row = [f"{result.time[i]:.12e}"]
            for sig in valid_signals:
                v = result.signals[sig][i]
                if sig.endswith('_code'):
                    fmt = save_formats.get(sig, 'd')
                else:
                    fmt = save_formats.get(sig, '6e')
                row.append(_fmt_value(v, fmt))
            writer.writerow(row)


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
            cls, module = _compile_va(str(va_path))
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
        node_map = _build_node_map(inst, module)
        try:
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
    record_nodes = set(netlist.save_signals) if netlist.save_signals else all_nodes
    if record_nodes:
        sim.record(*sorted(record_nodes))

    # 6. Run simulation
    if netlist.tran is None:
        log.write("ERROR: No transient analysis found")
        if log_file:
            log_file.close()
        return False

    tstop = netlist.tran.stop
    tstep = netlist.tran.step
    simopt = netlist.simulator_options or {}
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

    evas_profile = str(simopt.get('evas_profile', '')).lower()
    refine_factor, refine_steps, reltol, applied_profile = _apply_evas_profile(
        evas_profile, refine_factor, refine_steps, reltol
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
    log.write("")

    t_sim_start = time.time()
    result = sim.run(tstop, tstep=tstep,
                     refine_factor=refine_factor,
                     refine_steps=refine_steps,
                     reltol=reltol,
                     vabstol=vabstol)

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
    derived = _derive_bus_signals(result)
    result.signals.update(derived)

    # 7. Write CSV
    csv_path = out_dir / 'tran.csv'
    save_with_derived = list(netlist.save_signals) + list(derived.keys())
    _write_csv(csv_path, result, save_with_derived, netlist.save_formats)

    signal_names = save_with_derived if save_with_derived else sorted(result.signals.keys())
    log.write("")
    log.write(f"Writing CSV: {csv_path} "
              f"(signals: {', '.join(signal_names)})")

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
