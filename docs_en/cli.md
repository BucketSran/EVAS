# CLI Reference

EVAS provides three subcommands:

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
evas run clk_div --engine evas2
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

The default engine is `python`. Use `--engine evas2` only when the Rust backend
has been built and the selected design is covered by EVAS2.

## `evas simulate <file.scs>`

Simulate an arbitrary Spectre netlist file directly.

```bash
evas simulate path/to/tb_mydesign.scs -o output/mydesign -log sim.log
evas simulate path/to/tb_mydesign.scs --engine evas2
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o / --output` | `./output` | Output directory |
| `-log` | *(none)* | Path for a log file |
| `--engine` | `python` | Engine override: `python`, `evas2`, or `rust2` |

Exit code is `0` on success, `1` on simulation error.
