# Installation

## Requirements

- Python 3.9 or later
- NumPy and Matplotlib (installed automatically)

## From PyPI

```bash
pip install evas-sim
```

## From Source

```bash
git clone https://github.com/Arcadia-1/EVAS.git
cd EVAS
pip install -e ".[dev]"
```

## Verify

```bash
evas list
```

You should see the 5 bundled example groups printed.

## Engine Selection

The packaged default is the Python compatibility engine. It works from PyPI or
a fresh source checkout without compiling native code. Compatible Linux wheels
also include the evas-rust shared library for explicit Rust-engine runs.

evas-rust is the optional Rust-backed execution path for supported event-driven
designs. If your platform installed the pure Python wheel, build the Rust core
from source before selecting evas-rust:

```bash
cargo build --manifest-path evas/rust_core/Cargo.toml --release
evas simulate path/to/tb.scs --engine evas-rust
```

You can also select the engine with `EVAS_ENGINE=evas-rust` or
`simulatorOptions options evas_engine=evas-rust`. The legacy `evas2` and
`rust2` selectors remain accepted as compatibility aliases.
