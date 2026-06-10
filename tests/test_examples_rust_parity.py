"""Bundled example parity checks for the evas-rust backend."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from evas.netlist.runner import evas_simulate

pytestmark = pytest.mark.rust_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CORE = REPO_ROOT / "evas" / "rust_core"


@dataclass(frozen=True)
class ExampleParityCase:
    scs: str
    waveform_atol: float


SUPPORTED_RUST_EXAMPLES = [
    ExampleParityCase("adc_dac_ideal_4b/tb_adc_dac_ideal_4b_ramp.scs", 5e-2),
    ExampleParityCase("adc_dac_ideal_4b/tb_adc_dac_ideal_4b_sine.scs", 1e-1),
    ExampleParityCase("adc_dac_ideal_4b/tb_adc_dac_ideal_4b_sine1000.scs", 4e-2),
    ExampleParityCase("clk_div/tb_clk_div.scs", 1e-9),
    ExampleParityCase("clk_div/tb_clk_div_div2.scs", 1e-9),
    ExampleParityCase("clk_div/tb_clk_div_div8.scs", 1e-9),
    ExampleParityCase("comparator/tb_cmp_delay.scs", 1e-3),
    ExampleParityCase("comparator/tb_cmp_ideal.scs", 1e-3),
    ExampleParityCase("comparator/tb_cmp_offset_search.scs", 5e-3),
    ExampleParityCase("comparator/tb_cmp_strongarm.scs", 1e-3),
    ExampleParityCase("digital_basics/tb_and_gate.scs", 1e-3),
    ExampleParityCase("digital_basics/tb_dff_rst.scs", 1e-3),
    ExampleParityCase("digital_basics/tb_inverter_chain.scs", 9e-1),
    ExampleParityCase("digital_basics/tb_not_gate.scs", 1e-3),
    ExampleParityCase("digital_basics/tb_or_gate.scs", 1e-3),
]


@pytest.fixture(scope="session")
def built_rust_core():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)
    return RUST_CORE


@contextmanager
def _evas_engine(name: str):
    old_engine = os.environ.get("EVAS_ENGINE")
    os.environ["EVAS_ENGINE"] = name
    try:
        yield
    finally:
        if old_engine is None:
            os.environ.pop("EVAS_ENGINE", None)
        else:
            os.environ["EVAS_ENGINE"] = old_engine


def _run_example(scs: Path, engine: str, out_dir: Path, log_path: Path) -> None:
    with _evas_engine(engine):
        with redirect_stdout(StringIO()):
            assert evas_simulate(str(scs), output_dir=str(out_dir), log_path=str(log_path))
    assert (out_dir / "tran.csv").exists()


def _read_tran_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return header, np.asarray(rows, dtype=float)


def _assert_waveforms_match(
    python_csv: Path,
    rust_csv: Path,
    *,
    waveform_atol: float,
) -> None:
    python_header, python_data = _read_tran_csv(python_csv)
    rust_header, rust_data = _read_tran_csv(rust_csv)

    assert rust_header == python_header
    assert python_data.ndim == 2 and rust_data.ndim == 2
    assert python_data.shape[1] == rust_data.shape[1]
    assert python_data.shape[0] > 0 and rust_data.shape[0] > 0
    assert python_data[0, 0] == pytest.approx(rust_data[0, 0], abs=1e-15)
    assert python_data[-1, 0] == pytest.approx(rust_data[-1, 0], rel=1e-6, abs=1e-12)

    t_start = max(float(python_data[0, 0]), float(rust_data[0, 0]))
    t_stop = min(float(python_data[-1, 0]), float(rust_data[-1, 0]))
    sample_times = np.linspace(t_start, t_stop, 1001)

    for column in range(1, len(python_header)):
        python_signal = np.interp(sample_times, python_data[:, 0], python_data[:, column])
        rust_signal = np.interp(sample_times, rust_data[:, 0], rust_data[:, column])
        assert np.allclose(
            rust_signal,
            python_signal,
            rtol=1e-6,
            atol=waveform_atol,
        ), python_header[column]


def _assert_logs_match_semantics(python_log: Path, rust_log: Path) -> None:
    python_text = python_log.read_text(encoding="utf-8")
    rust_text = rust_log.read_text(encoding="utf-8")

    assert "Transient Analysis" in python_text
    assert "Transient Analysis" in rust_text
    assert "Writing CSV:" in python_text
    assert "Writing CSV:" in rust_text
    assert "evas_engine = evas-rust" not in python_text
    assert "evas_engine = evas-rust" in rust_text
    assert "evas_rust_full_model_fastpath = true" in rust_text
    assert "evas_rust_full_model_required = true" in rust_text
    assert "evas_rust_required = true" in rust_text
    assert "rust_full_model_required_failures = 0" in rust_text


def _assert_strobe_logs_match(python_out: Path, rust_out: Path) -> None:
    python_strobe = python_out / "strobe.txt"
    rust_strobe = rust_out / "strobe.txt"
    assert rust_strobe.exists() == python_strobe.exists()
    if python_strobe.exists():
        assert rust_strobe.read_text(encoding="utf-8") == python_strobe.read_text(
            encoding="utf-8"
        )


@pytest.mark.parametrize(
    "case",
    SUPPORTED_RUST_EXAMPLES,
    ids=lambda case: case.scs,
)
def test_bundled_example_python_vs_evas_rust_parity(
    built_rust_core,
    tmp_path,
    case: ExampleParityCase,
):
    scs = REPO_ROOT / "evas" / "examples" / case.scs
    python_out = tmp_path / "python"
    rust_out = tmp_path / "evas-rust"
    python_log = tmp_path / "python.log"
    rust_log = tmp_path / "evas-rust.log"

    _run_example(scs, "python", python_out, python_log)
    _run_example(scs, "evas-rust", rust_out, rust_log)

    _assert_waveforms_match(
        python_out / "tran.csv",
        rust_out / "tran.csv",
        waveform_atol=case.waveform_atol,
    )
    _assert_logs_match_semantics(python_log, rust_log)
    _assert_strobe_logs_match(python_out, rust_out)


def test_noise_gen_python_vs_evas_rust_noise_parity(
    built_rust_core,
    tmp_path,
):
    scs = REPO_ROOT / "evas" / "examples" / "noise_gen" / "tb_noise_gen.scs"
    python_out = tmp_path / "noise_gen_python"
    rust_out = tmp_path / "noise_gen_evas_rust"
    _run_example(scs, "python", python_out, tmp_path / "noise_gen_python.log")
    _run_example(scs, "evas-rust", rust_out, tmp_path / "noise_gen_evas_rust.log")

    python_header, python_data = _read_tran_csv(python_out / "tran.csv")
    rust_header, rust_data = _read_tran_csv(rust_out / "tran.csv")
    assert rust_header == python_header
    assert rust_data.shape == python_data.shape
    assert np.allclose(rust_data, python_data, rtol=0.0, atol=0.0)

    vin = rust_data[:, rust_header.index("vin_i")]
    vout = rust_data[:, rust_header.index("vout_o")]
    noise = vout - vin
    assert rust_data.shape[0] > 100
    assert abs(float(np.mean(noise))) < 5e-2
    assert 5e-2 < float(np.std(noise)) < 2e-1
    assert float(np.max(np.abs(noise))) > 1e-1
