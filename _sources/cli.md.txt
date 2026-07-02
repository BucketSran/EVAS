# CLI Reference

EVAS provides four subcommands:

## `evas list`

Print all bundled example names.

```bash
evas list
```

## `evas run <name>`

Copy a bundled example to the current directory and simulate it.

```bash
evas run clk_div
evas run digital_basics
evas run noise_gen
evas run clk_div --engine evas-rust
```

Multi-testbench examples (e.g. `adc_dac_ideal_4b`, `digital_basics`) use
`tb_<name>.scs` by default. Use `--tb` to select a different testbench:

```bash
evas run adc_dac_ideal_4b --tb tb_adc_dac_ideal_4b_sine.scs
evas run digital_basics --tb tb_not_gate.scs
```

Output goes to `./output/<name>/`. Analysis plots (if an `analyze_<name>.py`
script is present) are saved there as well.

Analysis scripts receive the output directory directly from `evas run`.

The default engine is `python`. Use `--engine evas-rust` only when the Rust
backend is bundled in the installed wheel or has been built from source, and
the selected design is covered by evas-rust.
The legacy `evas2` and `rust2` selectors remain accepted as compatibility aliases.

## `evas simulate <file.scs>`

Simulate an arbitrary Spectre netlist file directly.

```bash
evas simulate path/to/tb_mydesign.scs -o output/mydesign -log sim.log
evas simulate path/to/tb_mydesign.scs --engine evas-rust
evas simulate path/to/tb_mydesign.scs --ahdllint
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o / --output` | `./output` | Output directory |
| `-log` | *(none)* | Path for a log file |
| `--engine` | `python` | Engine override: `python`, `evas-rust`, `evas2`, or `rust2` |
| `--ahdllint` | off | Run EVAS AHDL-style lint as a non-blocking simulation preflight |
| `--ahdllint-min-transition` | `1e-12` | Minimum transition rise/fall time used by `--ahdllint` |

Exit code is `0` on success, `1` on simulation error.
The lint preflight reports diagnostics in the simulator log and increments the
warning count, but it does not by itself change simulation pass/fail status.
Netlists may also enable it with `simulatorOptions options ahdllint=true` or
`evas_ahdllint=true`.

## `evas lint <file.va|file.scs>`

Run EVAS/Spectre-style static checks without simulating.

```bash
evas lint path/to/model.va
evas lint path/to/tb_mydesign.scs --format json
```

For `.scs` inputs, EVAS parses the netlist and follows `ahdl_include` entries.
Diagnostics use `compat-error` for EVAS/Spectre subset issues and warning
severities for AHDL-style modeling risks. The command exits with `1` only when
at least one `compat-error` is reported.

Current warning diagnostics cover a Cadence AHDL-inspired static subset:
transition timing and simple continuous-input dataflow, conditional potential
contributions, case defaults, exact branch equality tests, floor/ceil
contribution discontinuities, `gnd` node portability, discrete function
arguments, implicit integer casts, and simulator-stop tasks inside loops.
The implementation keeps diagnostic metadata in a rule registry, including the
EVAS code, severity, rule name, lint phase, source category, and related
Cadence/Spectre identifiers when known.
Diagnostics emitted from parsed Verilog-A nodes include line and column
coordinates when available, which makes CLI output and JSON reports usable as
repair-loop anchors.

Compatibility diagnostics are intended to mirror concrete Cadence/Spectre
front-end failures when possible. Current examples include conditionally
executed analog operators such as `transition`, `slew`, and `idt`, plus
discipline vector ranges that use runtime variables instead of numeric or
parameter constant expressions.
The repository keeps small public oracle fixtures in
`tests/fixtures/lint_oracle_cases` for lint regression tests. These fixtures
store distilled expected EVAS diagnostic codes only, not raw Cadence/Spectre
reports or generated certification output.

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `text` | Diagnostic output: `text` or `json` |
| `--min-transition` | `1e-12` | Minimum transition rise/fall time used by lint warnings |
