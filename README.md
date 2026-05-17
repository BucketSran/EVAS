# EVAS — Event-driven Verilog-A Simulator

[![PyPI](https://img.shields.io/pypi/v/evas-sim.svg)](https://pypi.org/project/evas-sim/)
[![CI](https://github.com/Arcadia-1/EVAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Arcadia-1/EVAS/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-evas.tokenzhang.com-blue)](https://evas.tokenzhang.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, pure-Python behavioral simulator for digital/mixed-signal Verilog-A models.
Event-driven. No C compiler, no ngspice, no KCL/KVL solver.


---

> **If you're a human** — see the full docs at 

📖 **Docs:** [evas.tokenzhang.com](https://evas.tokenzhang.com)


> **If you're an AI agent** — load `skills/evas-sim/SKILL.md` (or copy the `skills/evas-sim/`
> folder into your skills directory). It contains the full compatibility table, CLI reference,
> output format, and common failure modes — everything you need to write and debug EVAS
> simulations without guessing.

---

## What EVAS does

EVAS simulates **voltage-mode, event-driven** Verilog-A behavioral models. You provide:

1. **A `.va` file** — your behavioral model (comparator, DAC, SAR logic, DWA controller, …)
2. **A `.scs` testbench netlist** — voltage sources, `ahdl_include`, and a `tran` statement
3. **Run `evas simulate`** — get `tran.csv` waveforms and optional plots

The bundled examples are a compact smoke-test set. For your own design, copy the
closest example directory, swap in your `.va`, adjust the stimulus sources and
`save` list, and run. The larger verification example library belongs with the
agent workflow in `veriloga-skills/evas-sim`.

## Installation

```bash
pip install evas-sim
evas list        # verify install — prints bundled example groups
```

If `evas` is not on PATH, use `python -m evas`.

## Simulating your own design

```bash
evas simulate path/to/tb.scs -o output/mydesign
```

Output in `-o` dir: `tran.csv` (waveforms), `strobe.txt` (log messages), `.png` plots.

**Minimal testbench template** (`.scs`):

```spectre
simulator lang=spectre
global 0

ahdl_include "my_module.va"

Vvdd (vdd 0) vsource type=dc dc=1.8
Vclk (clk 0) vsource type=pulse val0=0 val1=1.8 period=10n rise=0.1n fall=0.1n width=4.9n

IDUT (clk vdd out) my_module vdd=1.8

tran tran stop=200n maxstep=0.1n
save clk:2e out:6f
```

### Testbench structure reference

The bundled examples contain five small groups and their testbenches. They are
the in-package reference for common wiring patterns:

| Pattern needed | Look at |
|---------------|---------|
| Clocked digital logic | `clk_div`, `digital_basics` |
| Comparator with feedback | `comparator/cmp_offset_search` |
| ADC + sample-hold | `adc_dac_ideal_4b` |
| Noise / random stimulus | `noise_gen` |
| Multi-cycle edge timing | `comparator/cmp_delay`, `edge_interval_timer` |

## Supported Verilog-A

| Feature | Status |
|---------|--------|
| `V(node) <+`, `V(a,b)` differential | ✅ |
| `@(cross(...))`, `@(above(...))`, `@(initial_step)` | ✅ |
| `cross(expr, dir, time_tol, expr_tol)` event tolerances | ✅ (behavioral approximation) |
| `@(timer(period))`, `@(final_step)` | ✅ |
| `transition()` with delay / rise / fall | ✅ |
| `slew(x, maxrise, maxfall)` transient limiter | ✅ (behavioral approximation) |
| `for`, `if/else`, `case/endcase`, `begin/end` | ✅ |
| arrays, parameters (real / integer / string) | ✅ |
| `` `include ``, `` `define ``, `` `default_transition `` | ✅ |
| SI suffixes, math: `sin` `cos` `exp` `ln` `log` `pow` `floor` `ceil` … | ✅ |
| `$temperature`, `$vt`, `$abstime` | ✅ |
| `$bound_step()` | ✅ |
| `$fopen()`, `$fclose()`, `$fstrobe()`, `$fwrite()`, `$fdisplay()` | ✅ |
| `$display`, `$strobe`, `$random`, `$dist_uniform()`, `$rdist_normal()` | ✅ |
| `last_crossing(expr, dir, time_tol, expr_tol)` | ✅ (most-recent event-time approximation) |
| `I() <+`, `ddt()`, `idt()`, `q() <+` | not supported by design |
| AC/DC analysis, transistors | not supported by design |
| Spectre `subckt` hierarchy | not yet implemented |

### Accuracy Profiles

You can set `simulatorOptions options evas_profile=<mode>` in `.scs`:

- `fast`: lower refinement (`refine_factor=8`, `refine_steps=4`) for faster runtime
- `balanced`: default EVAS behavior (`16`, `8`)
- `precision`: higher refinement (`32`, `16`) for tighter event/cross timing

`errpreset=conservative/liberal` is still respected; `evas_profile` applies an explicit EVAS-side override when set.
Practical correspondence: `fast ≈ liberal`, `balanced ≈ moderate`, `precision ≈ conservative` (guidance only, not solver-equivalence claim).

## CSV output format

The `save` statement accepts per-signal format hints:

```
save vin:10e vout:6f clk:2e dout:d
```

| Suffix | Example |
|--------|---------|
| `:6e` (default) | `4.500000e-01` |
| `:Nf` | fixed-point, N decimal places |
| `:d` | integer (for digital buses) |

## Bundled examples (reference only)

Five groups ship with the PyPI package for install verification, CLI sanity
checks, and small starting templates. The full workflow-oriented example set is
maintained outside this simulator package in `veriloga-skills/evas-sim`.

| Group | Verilog-A modules | Notes |
|-------|------------------|-------|
| `clk_div` | `clk_div` | |
| `digital_basics` | `and_gate`, `or_gate`, `not_gate`, `dff_rst`, `inverter` | |
| `noise_gen` | `noise_gen` | |
| `adc_dac_ideal_4b` | `adc_ideal_4b`, `dac_ideal_4b`, `sh_ideal` | 3 stimuli: ramp / sine / 1000-pt sine |
| `comparator` | `cmp_ideal`, `cmp_strongarm`, `cmp_offset_search`, `cmp_delay`, `edge_interval_timer` | 4 sub-examples |

## Contributing

```bash
git clone https://github.com/Arcadia-1/EVAS.git
cd EVAS
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
