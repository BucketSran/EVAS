"""pytest test suite for EVAS examples.

Functional tests: key signal properties validated against expected behaviour.
Each test simulates the example and runs the corresponding validate_*.py checks.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from evas.netlist.runner import evas_simulate

EXAMPLES = Path(__file__).parent.parent / "evas" / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate(tb_rel: str, tmp_path: Path):
    """Run simulation and return the resulting tran.csv as a numpy structured array."""
    tb = EXAMPLES / tb_rel
    assert tb.exists(), f"Testbench not found: {tb}"
    ok = evas_simulate(str(tb), output_dir=str(tmp_path))
    assert ok, f"evas_simulate returned False for {tb.name}"
    csv = tmp_path / "tran.csv"
    assert csv.exists(), "tran.csv was not created"
    return np.genfromtxt(csv, delimiter=',', names=True, dtype=None, encoding='utf-8')


def _load_validate(tb_dir_name: str, validate_name: str = None):
    """Import the validate_*.py module from examples/.

    If *validate_name* is given (e.g. ``validate_cmp_delay.py``), load that
    specific file.  Otherwise fall back to the first ``validate_*.py`` found.
    """
    if validate_name:
        path = EXAMPLES / tb_dir_name / validate_name
        if path.exists():
            spec = importlib.util.spec_from_file_location("_validate", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        return None
    candidates = list((EXAMPLES / tb_dir_name).glob("validate_*.py"))
    if candidates:
        spec = importlib.util.spec_from_file_location("_validate", candidates[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def _run_validate(tmp_path: Path, tb_rel: str, fn_name: str, tb_dir: str,
                  validate_name: str = None):
    """Simulate and call a named validate function from examples/."""
    _simulate(tb_rel, tmp_path)
    mod = _load_validate(tb_dir, validate_name)
    assert mod is not None, f"No validate module found in examples/{tb_dir}"
    fn = getattr(mod, fn_name)
    failures = fn(out_dir=tmp_path)
    assert failures == 0, f"{fn_name} reported {failures} failure(s)"


# ---------------------------------------------------------------------------
# Functional: adc_dac_ideal_4b — 4-bit ADC→DAC round-trip
# ---------------------------------------------------------------------------

def test_adc_dac_ideal_4b(tmp_path):
    _run_validate(tmp_path, "adc_dac_ideal_4b/tb_adc_dac_ideal_4b_sine.scs",
                  "validate_csv", "adc_dac_ideal_4b")


# ---------------------------------------------------------------------------
# Functional: digital_basics — truth-table and clocked-sequence checks
# ---------------------------------------------------------------------------

def test_and_gate(tmp_path):
    _run_validate(tmp_path, "digital_basics/tb_and_gate.scs", "validate_and", "digital_basics")


def test_or_gate(tmp_path):
    _run_validate(tmp_path, "digital_basics/tb_or_gate.scs", "validate_or", "digital_basics")


def test_not_gate(tmp_path):
    _run_validate(tmp_path, "digital_basics/tb_not_gate.scs", "validate_not", "digital_basics")


def test_dff_rst(tmp_path):
    _run_validate(tmp_path, "digital_basics/tb_dff_rst.scs", "validate_dff", "digital_basics")


# ---------------------------------------------------------------------------
# Functional: clock / timing
# ---------------------------------------------------------------------------

def test_clk_div(tmp_path):
    _run_validate(tmp_path, "clk_div/tb_clk_div.scs", "validate_csv", "clk_div")


def test_clk_burst_gen(tmp_path):
    _run_validate(tmp_path, "clk_burst_gen/tb_clk_burst_gen.scs", "validate_csv", "clk_burst_gen")


# ---------------------------------------------------------------------------
# Functional: comparators
# ---------------------------------------------------------------------------

def test_cmp_ideal(tmp_path):
    _run_validate(tmp_path, "comparator/tb_cmp_ideal.scs",
                  "validate_csv", "comparator", "validate_cmp_ideal.py")


def test_cmp_strongarm(tmp_path):
    _run_validate(tmp_path, "comparator/tb_cmp_strongarm.scs",
                  "validate_csv", "comparator", "validate_cmp_strongarm.py")


@pytest.mark.xfail(reason="cmp_offset_search double-triggers cross event, converges to wrong offset")
def test_cmp_offset_search(tmp_path):
    _run_validate(tmp_path, "comparator/tb_cmp_offset_search.scs",
                  "validate_csv", "comparator", "validate_cmp_offset_search.py")


def test_cmp_delay(tmp_path):
    _run_validate(tmp_path, "comparator/tb_cmp_delay.scs",
                  "validate_csv", "comparator", "validate_cmp_delay.py")


# ---------------------------------------------------------------------------
# Functional: DACs
# ---------------------------------------------------------------------------

def test_dac_binary_clk_4b(tmp_path):
    _run_validate(tmp_path, "dac_binary_clk_4b/tb_dac_binary_clk_4b.scs",
                  "validate_csv", "dac_binary_clk_4b")


def test_dac_therm_16b(tmp_path):
    _run_validate(tmp_path, "dac_therm_16b/tb_dac_therm_16b.scs",
                  "validate_csv", "dac_therm_16b")


# ---------------------------------------------------------------------------
# Functional: encoding / bus drivers
# ---------------------------------------------------------------------------

def test_d2b_4b(tmp_path):
    _run_validate(tmp_path, "d2b_4b/tb_d2b_4b.scs", "validate_csv", "d2b_4b")


# ---------------------------------------------------------------------------
# Functional: algorithmic / sequenced
# ---------------------------------------------------------------------------

def test_ramp_gen(tmp_path):
    _run_validate(tmp_path, "ramp_gen/tb_ramp_gen.scs", "validate_csv", "ramp_gen")


def test_dwa_ptr_gen(tmp_path):
    _run_validate(tmp_path, "dwa_ptr_gen/tb_dwa_ptr_gen.scs", "validate_csv", "dwa_ptr_gen")


def test_lfsr(tmp_path):
    _run_validate(tmp_path, "lfsr/tb_lfsr.scs", "validate_csv", "lfsr")


def test_sar_adc_dac_weighted_8b(tmp_path):
    _run_validate(tmp_path, "sar_adc_dac_weighted_8b/tb_sar_adc_dac_weighted_8b.scs",
                  "validate_csv", "sar_adc_dac_weighted_8b")


# ---------------------------------------------------------------------------
# Functional: noise / analog utility
# ---------------------------------------------------------------------------

def test_noise_gen(tmp_path):
    _run_validate(tmp_path, "noise_gen/tb_noise_gen.scs", "validate_csv", "noise_gen")
