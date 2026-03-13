# EVAS — Event-driven Verilog-A Simulator

EVAS is a lightweight, pure-Python behavioral simulator for digital/control-class Verilog-A models. It provides fast event-driven simulation with zero external dependencies beyond NumPy and Matplotlib.

## Target Use Cases

- Rapid behavioral verification of clocked digital blocks (comparators, ADCs, DACs, dividers, LFSRs, DFFs)
- Event-driven models using `@cross`, `@above`, `transition()`, `@initial_step`
- Voltage-mode contributions (`V() <+`)

## Installation

```bash
pip install numpy matplotlib
```

No other dependencies required. No C compiler, no ngspice.

## Usage

```bash
python evas.py <input.scs> [-o output_dir] [-log logfile]
```

### Examples

```bash
cd EVAS

# Clocked comparator
python evas.py examples/testbench/tb_L2_comparator.scs -o output/cmp

# D flip-flop with reset
python evas.py examples/testbench/tb_dff_rst.scs -o output/dff

# 18-bit SAR ADC
python evas.py examples/testbench/tb_ADC_18B.scs -o output/adc

# Divide-by-2 frequency divider
python evas.py examples/testbench/tb_L2_Divider_2.scs -o output/div

# Linear feedback shift register
python evas.py examples/testbench/tb_LB_LFSR.scs -o output/lfsr

# 11-bit weighted DAC
python evas.py examples/testbench/tb_VA_DAC_11B.scs -o output/dac
```

Each run produces:
- `tran.csv` — time-domain waveform data
- `tran.png` — multi-panel waveform plot
- Console log with Spectre-style formatting

## Supported Verilog-A Features

- Module declarations with parameters and port arrays
- `@(cross(...))`, `@(above(...))` zero-crossing events
- `@(initial_step)` initialization
- Combined events: `@(initial_step or cross(...))`
- `transition()` operator with delay, rise/fall times
- `V(node)`, `V(a, b)` voltage access
- `V(node) <+` voltage contributions
- Arithmetic, logical, bitwise, shift, ternary operators
- `for` loops, `if/else`, `begin/end` blocks
- Integer and real variables, arrays
- Parameters with ranges
- `\`include`, `\`define`, `\`default_transition` preprocessor directives
- SI suffixes (n, u, p, m, k, M, G, etc.)
- Math functions: `ln`, `log`, `exp`, `sqrt`, `pow`, `abs`, `sin`, `cos`, `floor`, `ceil`, `min`, `max`
- String parameters with `.substr()` method calls

## Spectre Netlist Support

- `vsource` with `dc`, `pulse`, `pwl`, `sin` types
- `ahdl_include` for VA model files
- `parameters` with expression evaluation
- `tran` analysis with `stop` and `maxstep`
- `save` signal selection
- Line continuation (`\\`), comments (`//`)

## Limitations

- No `I() <+` current contributions
- No `ddt()`, `idt()` calculus operators
- No MNA matrix solve (no KCL/KVL enforcement)
- No transistor-level simulation (no MOSFET models)
- No AC or DC analysis (transient only)
- No subcircuit hierarchy

## Project Structure

```
EVAS/
├── evas.py                  # CLI entry point
├── evas/
│   ├── compiler/            # Verilog-A front-end (lexer, parser, AST)
│   ├── simulator/           # Event-driven simulation engine + backend
│   ├── netlist/             # Spectre .scs parser + orchestration runner
│   └── vams/                # VAMS include files (constants, disciplines)
├── examples/
│   ├── veriloga/            # 6 example VA models
│   └── testbench/           # 6 matching .scs testbenches
└── requirements.txt
```

## License

See repository license.
