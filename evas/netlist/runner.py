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
    if default_trans is None:
        default_trans = 1e-12
    cls = compile_module(module, default_trans)
    return cls, module


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
        times = vals[0::2]
        values = vals[1::2]
        sim.add_source(node, pwl(times, values))

    elif stype in ('sin', 'sine'):
        offset = float(params.get('sinedc', params.get('dc', 0.0)))
        ampl = float(params.get('ampl', 0.0))
        freq = float(params.get('freq', 0.0))

        if freq <= 0:
            warn.append(f"{src.name}: sine freq not set "
                        f"— treated as DC {offset} V")
            sim.add_source(node, dc(offset))
            return warn

        if ampl == 0:
            warn.append(f"{src.name}: sine ampl=0 — treated as DC {offset} V")
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
            row = [f"{result.time[i]:.6e}"]
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

        cls, module = _compile_va(str(va_path))
        models_by_name[module.name] = (cls, module)
        log.write(f"Compiled Verilog-A module: {module.name}")
        for w in module.warnings:
            log.write(f"WARNING ({module.name}): {w}")
            warnings += 1

    if errors > 0:
        log.write(f"\nevas completes with {errors} errors, {warnings} warnings.")
        if log_file:
            log_file.close()
        return False

    # 3. Build simulator
    sim = Simulator()

    for src in netlist.sources:
        src_warnings = _add_spectre_source(sim, src, netlist.ground)
        for w in src_warnings:
            log.write(f"WARNING: {w}")
            warnings += 1

    # 4. Instantiate models
    instance_counts: Dict[str, int] = {}
    for inst in netlist.instances:
        if inst.model_name not in models_by_name:
            log.write(f"ERROR: Model {inst.model_name} not found "
                      f"(available: {list(models_by_name.keys())})")
            errors += 1
            continue

        cls, module = models_by_name[inst.model_name]
        model = cls()
        model.node_map = _build_node_map(inst, module)
        # Case-insensitive param update: netlist keys are lowercased, model keys
        # preserve the original VA case. Match by lowercase to update correctly.
        lower_to_model_key = {k.lower(): k for k in model.params}
        for k, v in inst.params.items():
            model_key = lower_to_model_key.get(k.lower(), k)
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

    log.write("")
    log.write("*****************************************************")
    log.write(f"Transient Analysis `{netlist.tran.name}': "
              f"time = (0 s -> {_eng_format(tstop, 's')})")
    log.write("*****************************************************")
    log.write("Important parameter values:")
    log.write("    start = 0 s")
    log.write(f"    stop  = {_eng_format(tstop, 's')}")
    log.write(f"    step  = {_eng_format(tstep, 's')}")
    log.write("")

    t_sim_start = time.time()
    result = sim.run(tstop, tstep=tstep,
                     refine_factor=netlist.tran.refine_factor,
                     refine_steps=netlist.tran.refine_steps)

    for pct in range(10, 101, 10):
        t_at = tstop * pct / 100.0
        log.write(f"    tran: time = {_eng_format(t_at, 's'):12s} ({pct:3d} %)")

    sim_elapsed_ms = (time.time() - t_sim_start) * 1000
    n_steps = len(result.time) - 1
    log.write(f"Number of accepted tran steps = {n_steps}")

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
