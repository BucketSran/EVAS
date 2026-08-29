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
evas --version
evas --version --format json
```

You should see the 5 bundled example groups printed.

## Engine Selection

The production engine is EVAS2/Rust. Compatible Linux wheels include the
evas-rust shared library, and source installs build it with cargo.

If the Rust core is missing, unloadable, or ABI-incompatible, EVAS exits with a
specific error and does not fall back to Python. Build the core from source:

```bash
cargo build --manifest-path evas/rust_core/Cargo.toml --release
evas simulate path/to/tb.scs
```

The legacy `evas2` and `rust2` selectors remain accepted as time-bounded input
aliases for `evas-rust`; logs and metadata always use the canonical identity.

For benchmark provenance, capture `evas --version --format json` in image
metadata alongside the image digest. Simulation output directories also contain
`evas_identity.json` with the same package, Rust core, ABI, revision, and core
loadability fields.
