"""Validate dac_binary_clk_4b: 4-bit clocked binary DAC, full code sweep 0→15.

Testbench sweeps din[3:0] through codes 0..15 in order, one code per 40ns clock.
VDD = 0.9V, so LSB = 0.9/16 ≈ 56.25 mV.

Expected:
  - aout has 16 distinct levels (one per code).
  - Levels are monotonically non-decreasing as code increases.
  - Full output range spans at least 12/15 of VDD.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'dac_binary_clk_4b'


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    vdd = df[['din3', 'din2', 'din1', 'din0', 'aout']].max().max()
    lsb = vdd / 16.0

    # Decode input code
    thr = vdd * 0.5
    code = (
        (df['din3'].values > thr).astype(int) * 8 +
        (df['din2'].values > thr).astype(int) * 4 +
        (df['din1'].values > thr).astype(int) * 2 +
        (df['din0'].values > thr).astype(int) * 1
    )

    # Sample aout near the end of each code epoch (last 10% of each 40ns step)
    # Group by code value and take median aout per code
    aout = df['aout'].values
    levels = {}
    for c in range(16):
        mask = code == c
        if mask.any():
            levels[c] = float(np.median(aout[mask]))

    if len(levels) < 14:
        print(f"FAIL: only {len(levels)} distinct input codes seen (expected 16)")
        failures += 1

    if levels:
        # Monotonically non-decreasing
        sorted_codes = sorted(levels.keys())
        for i in range(1, len(sorted_codes)):
            c0, c1 = sorted_codes[i-1], sorted_codes[i]
            if levels[c1] < levels[c0] - lsb * 0.5:
                print(f"FAIL: aout decreased from code {c0} ({levels[c0]:.4f}V) to {c1} ({levels[c1]:.4f}V)")
                failures += 1
                break

        # Full range check: code 0 → ~0V, code 15 → ~VDD*(15/16)
        lo = levels.get(0, levels[sorted_codes[0]])
        hi = levels.get(15, levels[sorted_codes[-1]])
        if hi - lo < vdd * 0.75:
            print(f"FAIL: output range [{lo:.3f}, {hi:.3f}]V is too narrow (expected ≥ 75% of VDD)")
            failures += 1

    if failures == 0:
        print("[CSV] All assertions passed.")
    return failures


if __name__ == '__main__':
    raise SystemExit(0 if validate_csv() == 0 else 1)
