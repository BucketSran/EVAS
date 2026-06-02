"""Optional ctypes bridge for the EVAS Rust kernel prototype."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableSequence, Optional, Tuple, Union


class RustBackendError(RuntimeError):
    """Raised when an opt-in Rust backend call fails."""


class RustStaticAffineOp(ctypes.Structure):
    _fields_ = [
        ("read_node_id", ctypes.c_size_t),
        ("write_node_id", ctypes.c_size_t),
        ("gain", ctypes.c_double),
        ("bias", ctypes.c_double),
    ]


@dataclass(frozen=True)
class StaticAffineOp:
    """Python-side static affine operation consumed by the Rust ABI."""

    read_node_id: int
    write_node_id: int
    gain: float
    bias: float


class RustStaticAffineBatch:
    """ctypes-backed operation batch with stable storage for FFI calls."""

    def __init__(self, ops: Iterable[StaticAffineOp]):
        self.ops: Tuple[StaticAffineOp, ...] = tuple(ops)
        array_type = RustStaticAffineOp * len(self.ops)
        self._c_ops = array_type(
            *(
                RustStaticAffineOp(
                    int(op.read_node_id),
                    int(op.write_node_id),
                    float(op.gain),
                    float(op.bias),
                )
                for op in self.ops
            )
        )

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def ptr(self):
        return self._c_ops


class RustBackend:
    """Loaded Rust dynamic library wrapper."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self._lib = ctypes.CDLL(str(self.library_path))
        fn = self._lib.evas_rust_evaluate_static_affine
        fn.argtypes = [
            ctypes.POINTER(RustStaticAffineOp),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_int
        self._evaluate_static_affine = fn

    def make_static_affine_batch(
        self,
        ops: Iterable[StaticAffineOp],
    ) -> RustStaticAffineBatch:
        return RustStaticAffineBatch(ops)

    def evaluate_static_affine(
        self,
        batch: RustStaticAffineBatch,
        values: MutableSequence[float],
    ) -> None:
        if len(batch) == 0:
            return
        value_count = len(values)
        if value_count == 0:
            raise RustBackendError("cannot evaluate Rust static affine batch on empty values")

        try:
            value_buffer = (ctypes.c_double * value_count).from_buffer(values)
            copied = False
        except TypeError:
            # Tests and defensive callers may pass plain lists.  Production
            # opt-in code converts the indexed voltage array to array('d') so
            # this copy path is not used in the hot loop.
            value_buffer = (ctypes.c_double * value_count)(
                *(float(value) for value in values)
            )
            copied = True

        rc = self._evaluate_static_affine(batch.ptr, len(batch), value_buffer, value_count)
        if rc != 0:
            raise RustBackendError(f"Rust static affine evaluation failed with code {rc}")

        if copied:
            for idx, value in enumerate(value_buffer):
                values[idx] = float(value)


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libevas_rust_core.dylib"
    if os.name == "nt":
        return "evas_rust_core.dll"
    return "libevas_rust_core.so"


def default_rust_core_library_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rust_core"
        / "target"
        / "release"
        / _library_filename()
    )


def load_rust_backend(path: Optional[Union[os.PathLike[str], str]] = None) -> RustBackend:
    candidate = Path(path) if path is not None else Path(
        os.environ.get("EVAS_RUST_CORE_LIB", default_rust_core_library_path())
    )
    if not candidate.exists():
        raise RustBackendError(f"Rust backend library not found: {candidate}")
    return RustBackend(candidate)


def load_optional_rust_backend(
    path: Optional[Union[os.PathLike[str], str]] = None,
) -> Optional[RustBackend]:
    try:
        return load_rust_backend(path)
    except (OSError, RustBackendError):
        return None
