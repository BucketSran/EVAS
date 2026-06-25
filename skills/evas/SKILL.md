---
name: evas
description: |
  Use when working with EVAS/evas-sim: installing or running the EVAS
  Verilog-A behavioral simulator, checking whether a voltage-domain
  Verilog-A/Spectre testbench can run in EVAS, selecting the Python versus
  evas-rust engine, building the optional Rust backend, or interpreting EVAS
  outputs such as tran.csv and strobe.txt.
---

# EVAS

EVAS is a voltage-mode, event-driven Verilog-A behavioral simulator. It is not a SPICE analog solver: no KCL/KVL, transistor operating points, AC/DC analyses, current contributions, `ddt()`, `idt()`, or charge equations.

## First Checks

Before running a user design, read the `.va` and `.scs` files enough to identify whether EVAS is an appropriate tool.

Supported patterns include:
- `V(node) <+` and `V(a,b)` voltage-domain contributions
- `@(initial_step)`, `@(final_step)`, `@(cross(...))`, `@(above(...))`, and `@(timer(...))`
- `transition()`, `$abstime`, `$temperature`, `$vt`, `$bound_step()`
- behavioral control flow: `if`, `case`, loops, arrays, parameters, macros, and common math functions
- `$display`, `$strobe`, `$fopen`, `$fclose`, `$fstrobe`, `$fwrite`, `$fdisplay`, `$random`, `$dist_uniform`, `$rdist_normal`

Stop or reroute to a SPICE/Spectre/Xyce-style flow when the design relies on:
- `I(...) <+`, `q(...) <+`, `ddt()`, `idt()`, AC/DC/noise/PSS analyses, or transistor/device models
- Spectre `subckt` hierarchy not flattened into Verilog-A modules
- PDK model includes as meaningful circuit devices; EVAS may ignore such lines but cannot solve them

EVAS acceptance is not the same as Spectre portability. If Cadence/Spectre legality matters and Spectre was not run, say explicitly that the design is not Spectre-compiled.

## Install And Engine Selection

Default EVAS uses the Python compatibility engine and should work from PyPI or a source checkout without native compilation:

```bash
pip install evas-sim
evas list
evas simulate path/to/tb.scs -o WORK/evas-run
```

Use `python -m evas` if `evas` is not on `PATH`.

`evas-rust` is optional and must be built before explicitly selecting it. Build from the EVAS source repo:

```bash
cargo build --manifest-path evas/rust_core/Cargo.toml --release
evas simulate path/to/tb.scs -o WORK/evas-run --engine evas-rust
```

Engine selectors:
- CLI: `--engine python` or `--engine evas-rust`
- Environment: `EVAS_ENGINE=python` or `EVAS_ENGINE=evas-rust`
- Testbench: `simulatorOptions options evas_engine=evas-rust`
- Compatibility aliases: `evas2` and `rust2`

If the user explicitly wants `evas-rust` and `cargo` is missing, install a minimal Rust toolchain only with user approval or when the user has already authorized global Rust installation. Keep build artifacts local by setting `CARGO_TARGET_DIR` to the active project's `WORK/` directory when the build is only needed for that project.

## Working Directory Discipline

Do not write simulation output, fetched references, temporary scripts, or Rust target directories into global or unrelated project paths. For ad hoc work, create/use `WORK/` under the active repository and ensure it is gitignored.

Recommended pattern when running from another repo against a local EVAS checkout:

```bash
mkdir -p WORK/evas-output WORK/evas-rust-target WORK/tmp
export CARGO_TARGET_DIR="$PWD/WORK/evas-rust-target"
export TMPDIR="$PWD/WORK/tmp"
cargo build --release --manifest-path /path/to/EVAS/evas/rust_core/Cargo.toml
evas simulate path/to/tb.scs -o WORK/evas-output --engine evas-rust
```

## Testbench Rules

Prefer a two-file structure:
- `dut.va`: Verilog-A module(s), no hardcoded test stimulus or analyses
- `tb_<name>.scs`: voltage sources, DUT instance, `tran`, `save`, and `ahdl_include`

Use bare include filenames when the `.va` file sits next to the testbench:

```spectre
ahdl_include "my_module.va"
```

Keep saved signals explicit and narrow. EVAS writes primary results to the `-o` directory; Spectre PSF output options such as `write=`, `savetime=`, `savefile=`, and `flushtime=` are not EVAS output controls.

## Running Examples And Tests

Useful commands in the EVAS source repo:

```bash
pip install -e ".[dev]"
evas list
evas run digital_basics --tb tb_not_gate.scs
evas run clk_div
pytest tests/test_examples.py -q
```

Rust-specific tests usually build the Rust core first or skip when `cargo` is unavailable. If a test or user request requires Rust and the backend is absent, build it instead of reporting failure.

## Output Interpretation

Treat `tran.csv` as the source of truth. It contains `time` in seconds and saved signal columns in volts or integer code values. `strobe.txt` is optional and appears only when the model emits text output. Plots are optional convenience artifacts.

When judging behavior, use the task-specific validation script or a narrow checker over `tran.csv`. For waveform parity, compare common signals over a common time window; do not claim pointwise CSV equality when simulators output different timesteps. State the actual metric used, such as logic mismatch ratio, RMSE, max absolute error, or checker pass/fail.
