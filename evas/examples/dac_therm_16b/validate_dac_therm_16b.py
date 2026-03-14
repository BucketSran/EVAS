"""Validate dac_therm_16b: 16-bit thermometer DAC (vstep=1.0V).

Checkpoints (ones count -> vout):
  t=100ns:  0 ones  -> vout = 0.0V
  t=300ns:  4 ones  -> vout = 4.0V
  t=500ns:  8 ones  -> vout = 8.0V
  t=700ns:  12 ones -> vout = 12.0V
  t=1000ns: 16 ones -> vout = 16.0V
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'dac_therm_16b'

_CHECKPOINTS = [
    (100.0,  0,   0.0),
    (300.0,  4,   4.0),
    (500.0,  8,   8.0),
    (700.0,  12,  12.0),
    (1000.0, 16,  16.0),
]
_TOL = 0.1


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    t_ns = df['time'].values * 1e9

    for t_check, exp_ones, exp_vout in _CHECKPOINTS:
        idx = int(np.argmin(np.abs(t_ns - t_check)))
        got_vout = float(df['vout'].iloc[idx])
        if abs(got_vout - exp_vout) > _TOL:
            print(f"FAIL: at t={t_check}ns (ones={exp_ones}): vout={got_vout:.3f}V, expected {exp_vout:.3f}V")
            failures += 1

    # vout should be monotonically non-decreasing after reset
    active = df['vout'].values[t_ns > 10.0]
    diffs = np.diff(active)
    if np.any(diffs < -0.1):
        print("FAIL: vout decreased unexpectedly")
        failures += 1

    if failures == 0:
        print("[CSV] All assertions passed.")
    return failures


def validate_txt(out_dir: Path = OUT) -> int:
    txt_path = out_dir / 'strobe.txt'
    if not txt_path.exists():
        return 0
    # dac_therm_16b does not emit $strobe lines
    return 0


if __name__ == '__main__':
    f1 = validate_csv()
    f2 = validate_txt()
    total = f1 + f2
    print(f"Validation: {total} failure(s) [{f1} CSV, {f2} TXT]")
    raise SystemExit(0 if total == 0 else 1)
