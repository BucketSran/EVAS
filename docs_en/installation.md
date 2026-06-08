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
a fresh source checkout without compiling native code.

evas-rust is the optional Rust-backed execution path for supported event-driven
designs. To use it from source, build the Rust core first:

```bash
cargo build --manifest-path evas/rust_core/Cargo.toml --release
evas simulate path/to/tb.scs --engine evas-rust
```

You can also select the engine with `EVAS_ENGINE=evas-rust` or
`simulatorOptions options evas_engine=evas-rust`. The legacy `evas2` and
`rust2` selectors remain accepted as compatibility aliases.
