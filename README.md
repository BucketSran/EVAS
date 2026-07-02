# EVAS — Event-driven Verilog-A Simulator

[![PyPI](https://img.shields.io/pypi/v/evas-sim.svg)](https://pypi.org/project/evas-sim/)
[![CI](https://github.com/Arcadia-1/EVAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Arcadia-1/EVAS/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-evas.tokenzhang.com-blue)](https://evas.tokenzhang.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight behavioral simulator for digital/mixed-signal Verilog-A models.
EVAS ships with a Python compatibility engine by default and an optional evas-rust backend for supported event-driven designs. No ngspice, no KCL/KVL solver.


---

Docs: [evas.tokenzhang.com](https://evas.tokenzhang.com)

---

## What EVAS does

EVAS simulates **voltage-mode, event-driven** Verilog-A behavioral models. You provide:

1. **A `.va` file** — your behavioral model (comparator, DAC, SAR logic, DWA controller, …)
2. **A `.scs` testbench netlist** — voltage sources, `ahdl_include`, and a `tran` statement
3. **Run `evas simulate`** — get `tran.csv` waveforms and optional plots

The main simulator is transient-first. Compiled models also expose lightweight
behavioral helpers for `analysis("ac")`, `ac_stim()`, and Verilog-A noise source
functions so source-style models can run AC/noise sweeps from Python without
claiming SPICE-style linearized circuit analysis.

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

The packaged default uses the Python engine so examples and user netlists run
from PyPI or a fresh source checkout. Compatible Linux wheels also include the
evas-rust shared library. To request evas-rust explicitly, pass
`--engine evas-rust`, set `EVAS_ENGINE=evas-rust`, or add
`simulatorOptions options evas_engine=evas-rust` to the testbench. If your
platform installed the pure Python wheel, build the Rust backend from source
first. The legacy `evas2` and `rust2` selectors remain accepted as compatibility
aliases.

## Simulating your own design

```bash
evas simulate path/to/tb.scs -o output/mydesign
evas simulate path/to/tb.scs -o output/mydesign --engine evas-rust
```

Output in `-o` dir: `tran.csv` (waveforms), `strobe.txt` (log messages), `.png` plots.

Before a full simulation, you can run a Spectre/AHDL-style static lint pass:

```bash
evas lint path/to/model.va
evas lint path/to/tb.scs --format json
```

`evas lint` follows `ahdl_include` statements in `.scs` files and reports two
classes of issues: `compat-error` for EVAS/Spectre subset problems that should
block a candidate, and warning diagnostics for AHDL-style modeling risks such as
discrete signals directly driving analog contributions or suspicious transition
usage. Lint warnings do not change simulation pass/fail status.
Compatibility diagnostics include Cadence/Spectre-aligned cases such as
conditionally executed analog operators (`transition`, `slew`, `idt`) and
discipline vector ranges that depend on runtime variables instead of numeric or
parameter constant expressions.

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

### Support tiers

EVAS is strongest in the **behavioral-event / waveform-oriented** tier: voltage
reads and drives, event-controlled state, timers, transitions, table/random/file
helpers, and small mixed-signal logic/wreal subsets. This tier is implemented by
a lightweight event/waveform engine and does not require a full conservative
analog solver.

EVAS also accepts a limited **behavioral-continuous-time** tier. Operators such
as `ddt()`, `idt()`, `laplace_*`, `zi_*`, and `limexp()` compile and run with
explicit behavioral approximations for transient compatibility. They should not
be read as Spectre-equivalent continuous-time transfer-function solving.

EVAS does not implement the **conservative-current / KCL-MNA** tier. Device-style
models that rely on `I(p,n) <+ ...`, branch charge/current contributions,
nonlinear device equations, or transistor-level AC/DC matrix solving remain
outside the current simulator design.

| Feature | Status |
|---------|--------|
| `V(node) <+`, `V(a,b)` differential | ✅ |
| `@(cross(...))`, `@(above(...))`, `@(initial_step)` | ✅ |
| `cross(expr, dir, time_tol, expr_tol)` event tolerances | ✅ (behavioral approximation) |
| `@(timer(period))`, `@(final_step)` | ✅ |
| `transition()` with delay / rise / fall | ✅ |
| `slew(x, maxrise, maxfall)` transient limiter | ✅ (behavioral approximation) |
| `for`, `repeat`, `while`, `do while`, `if/else`, `case/endcase`, `begin/end` | ✅ |
| arrays, including integer/real 1-D and 2-D state arrays | ✅ |
| `branch (p,n) br; V(br)` named branch voltage probes | ✅ |
| `$analog_node_alias()` and string OOMR voltage probes such as `V(sigpath)` | ✅ |
| `$table_model()` 1-D file tables and simple 2-D array-backed surfaces | ✅ |
| parameters and variables (real / integer / string) | ✅ |
| user-defined functions/tasks, including bounded recursive functions | ✅ |
| `module`, `connectmodule`, simple behavioral hierarchy | ✅ |
| `` `include ``, `` `define ``, `` `default_transition `` | ✅ |
| SI suffixes, math: `sin` `cos` `exp` `ln` `log` `pow` `floor` `ceil` … | ✅ |
| `$temperature`, `$vt`, `$abstime` | ✅ |
| `$bound_step()` | ✅ |
| `$fopen()`, `$fclose()`, `$fscanf()`, `$fstrobe()`, `$fwrite()`, `$fdisplay()` | ✅ |
| `$display`, `$strobe`, `$sformat()`, `$swrite()` | ✅ |
| `$random`, `$dist_*()`, `$rdist_*()` behavioral random helpers | ✅ |
| `last_crossing(expr, dir, time_tol, expr_tol)` | ✅ (most-recent event-time approximation) |
| `analysis("ac")`, `ac_stim()` | ✅ (behavioral Python sweep helper) |
| `white_noise()`, `flicker_noise()`, `noise_table()` | ✅ (behavioral PSD / integrated-noise helper) |
| `ddt()`, `idt()`, `laplace_*()`, `zi_*()`, `limexp()` | ✅ (behavioral transient approximation) |
| `generate` / `genvar`, `specify` / `specparam`, `connectrules` | not supported by design |
| custom `nature` / `discipline` semantics beyond the bundled VAMS stubs | not supported by design |
| analog primitive instances such as `resistor` / `isource` | not supported by design |
| `I() <+`, `q() <+`, branch charge/current contributions | not supported by design |
| SPICE-style AC/DC matrix solving, transistors | not supported by design |
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
