"""
runner.py — EVAS runner: parse .scs → compile VA → simulate → log + CSV + plot.

Produces a streamlined Spectre-like log, CSV waveform data, and PNG plot.
"""
import sys
import os
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, TextIO
from datetime import datetime

from .spectre_parser import (
    SpectreNetlist, SpectreSource, SpectreInstance, parse_spectre,
    evaluate_expr, _parse_suffix_number, has_transistors,
)

from evas.simulator.engine import Simulator, pulse, dc, sine, pwl, SimResult
from evas.simulator.backend import CompiledModel, compile_module
from evas.compiler.preprocessor import preprocess
from evas.compiler.parser import parse as parse_va

VERSION = '0.1.0'


# ---------------------------------------------------------------------------
# VA model compilation (returns class + port info)
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
    netlist_nodes = list(instance.nodes)
    ni = 0

    for port_name in module.ports:
        pd = decl_by_name.get(port_name)
        if pd and pd.is_array:
            hi = pd.array_hi if pd.array_hi is not None else 0
            lo = pd.array_lo if pd.array_lo is not None else 0
            if hi >= lo:
                indices = range(hi, lo - 1, -1)
            else:
                indices = range(hi, lo + 1)
            for idx in indices:
                port_key = f'{pd.name}[{idx}]'
                if ni < len(netlist_nodes):
                    node_map[port_key] = netlist_nodes[ni]
                    ni += 1
        else:
            if ni < len(netlist_nodes):
                node_map[port_name] = netlist_nodes[ni]
                ni += 1

    return node_map


# ---------------------------------------------------------------------------
# Source conversion: Spectre params → simulator waveform
# ---------------------------------------------------------------------------

def _add_spectre_source(sim: Simulator, src: SpectreSource, ground: str):
    """Convert a SpectreSource to a simulator waveform and add it."""
    node = src.node_pos if src.node_neg == ground else src.node_pos
    params = src.params
    stype = src.source_type

    if stype == 'dc' or stype == '':
        voltage = float(params.get('dc', 0.0))
        sim.add_source(node, dc(voltage))

    elif stype == 'pulse':
        v0 = float(params.get('val0', 0.0))
        v1 = float(params.get('val1', 1.0))
        period = float(params.get('period', 1.0))
        delay = float(params.get('delay', 0.0))
        rise = float(params.get('rise', 1e-12))
        fall = float(params.get('fall', 1e-12))
        width = params.get('width', None)
        if width is not None:
            duty = float(width) / period if period > 0 else 0.5
        else:
            duty = 0.5

        sim.add_source(node, pulse(
            v_lo=v0, v_hi=v1, period=period, duty=duty,
            rise=rise, fall=fall, delay=delay,
        ))

    elif stype == 'pwl':
        wave = params.get('wave', '')
        if isinstance(wave, str) and wave:
            tokens = wave.split()
            vals = [float(t) for t in tokens]
        else:
            vals = []
        times = [vals[i] for i in range(0, len(vals), 2)]
        values = [vals[i] for i in range(1, len(vals), 2)]
        sim.add_source(node, pwl(times, values))

    elif stype == 'sin' or stype == 'sine':
        offset = float(params.get('sinedc', params.get('dc', 0.0)))
        ampl = float(params.get('ampl', 1.0))
        freq = float(params.get('freq', 1e6))
        sim.add_source(node, sine(offset=offset, amplitude=ampl, freq=freq))


# ---------------------------------------------------------------------------
# Engineering number formatting
# ---------------------------------------------------------------------------

def _eng_format(val: float, unit: str = '') -> str:
    """Format a number in engineering notation with unit."""
    if val == 0:
        return f"0 {unit}".strip()

    abs_val = abs(val)
    prefixes = [
        (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
        (1, ''), (1e-3, 'm'), (1e-6, 'u'), (1e-9, 'n'),
        (1e-12, 'p'), (1e-15, 'f'),
    ]

    for scale, prefix in prefixes:
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

    def write(self, msg: str = ''):
        if not self.quiet:
            print(msg)
        if self.log_file:
            self.log_file.write(msg + '\n')

    def flush(self):
        if self.log_file:
            self.log_file.flush()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_spectre(scs_file: str, log_path: Optional[str] = None,
                output_dir: str = './output') -> bool:
    """Run a Spectre .scs netlist. Returns True on success."""
    scs_path = Path(scs_file).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if log_path:
        log_file = open(log_path, 'w', encoding='utf-8')

    log = _Logger(log_file, quiet=(log_path is not None))
    errors = 0
    warnings = 0

    t_total_start = time.time()

    # Banner
    now = datetime.now()
    timestamp = now.strftime("%I:%M:%S %p, %a %b %d, %Y").lstrip('0')

    log.write("EVAS — Event-driven Verilog-A Simulator")
    log.write(f"Version {VERSION} — {now.strftime('%b %Y')}")
    log.write("")
    log.write(f"Simulating `{scs_path.name}' at {timestamp}.")
    log.write("Command line:")
    cmd = ' '.join(sys.argv)
    log.write(f"    {cmd}")
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

    # Check for transistors — EVAS doesn't support them
    if has_transistors(netlist):
        log.write("ERROR: Netlist contains transistor-level devices.")
        log.write("       EVAS only supports behavioral Verilog-A models.")
        if log_file:
            log_file.close()
        return False

    # 2. Compile VA models
    models_by_name = {}
    for inc in netlist.ahdl_includes:
        va_path = (Path(netlist.source_dir) / inc.path).resolve()
        if not va_path.exists():
            log.write(f"ERROR: Cannot find VA file: {inc.path}")
            errors += 1
            continue

        cls, module = _compile_va(str(va_path))
        models_by_name[module.name] = (cls, module)
        log.write(f"Compiled Verilog-A module: {module.name}")

    if errors > 0:
        log.write(f"\nevas completes with {errors} errors, {warnings} warnings.")
        if log_file:
            log_file.close()
        return False

    # 3. Build simulator
    sim = Simulator()

    # Add voltage sources
    source_count = 0
    for src in netlist.sources:
        _add_spectre_source(sim, src, netlist.ground)
        source_count += 1

    # 4. Instantiate models
    instance_counts = {}
    for inst in netlist.instances:
        if inst.model_name not in models_by_name:
            log.write(f"ERROR: Model {inst.model_name} not found "
                      f"(available: {list(models_by_name.keys())})")
            errors += 1
            continue

        cls, module = models_by_name[inst.model_name]
        model = cls()
        model.node_map = _build_node_map(inst, module)

        for k, v in inst.params.items():
            model.params[k] = v

        sim.add_model(model)
        instance_counts[inst.model_name] = instance_counts.get(inst.model_name, 0) + 1

    # Circuit inventory
    log.write("")
    log.write("Circuit inventory:")

    all_nodes = set()
    for src in netlist.sources:
        all_nodes.add(src.node_pos)
        all_nodes.add(src.node_neg)
    for inst in netlist.instances:
        for n in inst.nodes:
            all_nodes.add(n)
    all_nodes.discard(netlist.ground)
    node_count = len(all_nodes)

    log.write(f"{'nodes':>20s} {node_count}")
    for mname, cnt in instance_counts.items():
        log.write(f"{mname:>20s} {cnt}")
    log.write(f"{'vsource':>20s} {source_count}")

    # 5. Record signals
    record_nodes = set()
    for sig in netlist.save_signals:
        record_nodes.add(sig)
    if not record_nodes:
        record_nodes = all_nodes

    if record_nodes:
        sim.record(*sorted(record_nodes))

    # 6. Run simulation
    if netlist.tran is None:
        log.write("ERROR: No transient analysis found")
        errors += 1
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
    log.write(f"    start = 0 s")
    log.write(f"    stop  = {_eng_format(tstop, 's')}")
    log.write(f"    step  = {_eng_format(tstep, 's')}")
    log.write("")

    t_sim_start = time.time()

    result = sim.run(tstop, tstep=tstep)

    # Progress lines
    for pct in range(10, 101, 10):
        t_at = tstop * pct / 100.0
        log.write(f"    tran: time = {_eng_format(t_at, 's'):12s} ({pct:3d} %)")

    t_sim_end = time.time()
    sim_cpu = (t_sim_end - t_sim_start) * 1000

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
    log.write(f"Tran analysis time: CPU = {sim_cpu:.1f} ms, "
              f"elapsed = {sim_cpu:.1f} ms.")

    # 7. Write CSV
    csv_path = out_dir / 'tran.csv'
    _write_csv(csv_path, result, netlist.save_signals)

    signal_names = netlist.save_signals if netlist.save_signals else sorted(result.signals.keys())
    log.write("")
    log.write(f"Writing CSV: {csv_path} "
              f"(signals: {', '.join(signal_names)})")

    # 8. Generate plot
    plot_path = out_dir / 'tran.png'
    _generate_plot(result, signal_names, scs_path.stem, plot_path)
    log.write(f"Writing plot: {plot_path}")

    # Final summary
    t_total_end = time.time()
    total_cpu = t_total_end - t_total_start

    log.write("")
    log.write(f"evas completes with {errors} errors, {warnings} warnings.")
    log.write(f"Total time: CPU = {total_cpu:.1f} s, "
              f"elapsed = {total_cpu:.1f} s.")

    if log_file:
        log_file.close()

    return errors == 0


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _generate_plot(result: SimResult, signal_names: List[str],
                   title: str, plot_path: Path):
    """Generate a multi-panel waveform plot as PNG."""
    valid_signals = [s for s in signal_names if s in result.signals]
    if not valid_signals:
        valid_signals = [n for n in result.signals if n != '__time__']

    n_plots = len(valid_signals) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    tstop = result.time[-1]
    if tstop < 1e-6:
        t_scale, t_unit = 1e9, 'ns'
    elif tstop < 1e-3:
        t_scale, t_unit = 1e6, 'us'
    else:
        t_scale, t_unit = 1e3, 'ms'

    t_plot = result.time * t_scale

    for i, node in enumerate(valid_signals):
        axes[i].plot(t_plot, result.signals[node], linewidth=0.8)
        axes[i].set_ylabel(f'{node} (V)')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_title(title)

    if result.step_sizes is not None and len(result.step_sizes) > 1:
        ax = axes[-1]
        dt = result.step_sizes * t_scale
        ax.semilogy(t_plot[1:], dt[1:], 'k-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f'Step size ({t_unit})')
        ax.set_xlabel(f'Time ({t_unit})')
        ax.grid(True, alpha=0.3)
        ax.set_title('Adaptive Step Size')

    fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(csv_path: Path, result: SimResult, save_signals: List[str]):
    """Write simulation results to CSV file."""
    signal_names = save_signals if save_signals else sorted(result.signals.keys())
    valid_signals = [s for s in signal_names if s in result.signals]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time'] + valid_signals)
        for i in range(len(result.time)):
            row = [f"{result.time[i]:.6e}"]
            for sig in valid_signals:
                row.append(f"{result.signals[sig][i]:.6e}")
            writer.writerow(row)
