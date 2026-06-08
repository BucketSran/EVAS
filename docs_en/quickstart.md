# Quickstart

## Run a Bundled Example

The fastest way to try EVAS is to run one of the included examples:

```bash
evas run clk_div
evas run clk_div --engine evas2   # requires a built Rust backend
```

This will:
1. Copy the `clk_div` Verilog-A model and Spectre testbench into `./evas-run/clk_div/`
2. Simulate the netlist; outputs go to `./output/clk_div/`
3. Run the analysis script, saving `analyze_clk_div.png`

## Simulate Your Own Netlist

```bash
evas simulate path/to/tb.scs -o output/mydesign
evas simulate path/to/tb.scs -o output/mydesign --engine evas2
```

Each simulation run produces:

| File | Contents |
|------|----------|
| `tran.csv` | Time-domain waveform data |
| `tran.png` | Auto-generated multi-panel plot |
| `strobe.txt` | All `$strobe` log lines in time order |

## CSV Output Format

Signals default to 6-digit scientific notation (`:6e`).
The `save` statement accepts optional `:fmt` suffixes per signal:

```
save vin:10e vout:6e clk:2e dout_code:d
```

| Suffix | Format | Example |
|--------|--------|---------|
| `:6e` | `:.6e` (default) | `4.500000e-01` |
| `:10e` | `:.10e` | `4.5000000000e-01` |
| `:2e` | `:.2e` | `4.50e-01` |
| `:4f` | `:.4f` | `0.4500` |
| `:d` | integer | `7` |

## Supported Verilog-A Features

- `@(cross(...))`, `@(above(...))` zero-crossing events
- `@(initial_step)` initialization
- `transition()` operator with delay and rise/fall times
- `V(node) <+` voltage contributions
- Arithmetic, logical, bitwise, shift, ternary operators
- `for` loops, `if/else`, `begin/end` blocks
- Integer/real variables, arrays, parameters with ranges
- `` `include ``, `` `define ``, `` `default_transition `` directives
- SI suffixes; math functions: `ln`, `log`, `exp`, `sqrt`, `pow`, `abs`, `sin`, `cos`, `floor`, `ceil`, `min`, `max`
