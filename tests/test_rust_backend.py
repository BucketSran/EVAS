"""Tests for the optional EVAS Rust backend ctypes bridge."""

from __future__ import annotations

import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from evas.simulator.rust_backend import (
    StaticAffineOp,
    default_rust_core_library_path,
    load_rust_backend,
)


RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_CORE,
        check=True,
    )


def test_rust_backend_static_affine_batch_updates_array_buffer():
    _build_rust_core()
    backend = load_rust_backend(default_rust_core_library_path())
    values = array("d", [0.5, 0.0, 0.0])
    batch = backend.make_static_affine_batch(
        [
            StaticAffineOp(read_node_id=0, write_node_id=1, gain=2.0, bias=0.25),
            StaticAffineOp(read_node_id=1, write_node_id=2, gain=-1.0, bias=1.0),
        ]
    )

    backend.evaluate_static_affine(batch, values)

    assert values.tolist() == pytest.approx([0.5, 1.25, -0.25])


def test_rust_backend_default_library_name_matches_platform():
    path = default_rust_core_library_path()

    if sys.platform == "darwin":
        assert path.name == "libevas_rust_core.dylib"
    elif sys.platform.startswith("linux"):
        assert path.name == "libevas_rust_core.so"
    else:
        assert path.name == "evas_rust_core.dll"
