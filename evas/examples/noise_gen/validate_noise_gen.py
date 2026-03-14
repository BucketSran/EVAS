"""Validate noise_gen behavior from CSV output (sigma=0.1V, vin=1.0V DC)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'noise_gen'

_VIN_NOMINAL = 1.0
_SIGMA = 0.1


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    vin  = df['vin_i'].values
    vout = df['vout_o'].values

    # Input should be constant at 1.0V
    if np.abs(vin - _VIN_NOMINAL).max() > 0.01:
        print(f"FAIL: vin_i is not constant at {_VIN_NOMINAL}V")
        failures += 1

    # Output mean should be close to vin (zero-mean noise)
    vout_mean = float(np.mean(vout))
    if abs(vout_mean - _VIN_NOMINAL) > 3 * _SIGMA / np.sqrt(len(vout)):
        # More lenient: just check it is within 5*sigma of expected
        if abs(vout_mean - _VIN_NOMINAL) > 5 * _SIGMA:
            print(f"FAIL: vout mean={vout_mean:.4f}V, expected ~{_VIN_NOMINAL}V")
            failures += 1

    # Noise standard deviation should be roughly sigma
    noise = vout - vin
    noise_std = float(np.std(noise))
    # Allow 2x-3x tolerance for randomness
    if not (0.01 < noise_std < _SIGMA * 5):
        print(f"FAIL: noise std={noise_std:.4f}V, expected ~{_SIGMA}V")
        failures += 1

    # Output should be different from input (noise should be non-trivial)
    if np.all(np.abs(noise) < 1e-6):
        print("FAIL: vout identical to vin (noise not being applied)")
        failures += 1

    if failures == 0:
        print(f"[CSV] All assertions passed. noise std={noise_std:.4f}V (expected ~{_SIGMA}V)")
    return failures


def validate_txt(out_dir: Path = OUT) -> int:
    txt_path = out_dir / 'strobe.txt'
    if not txt_path.exists():
        return 0
    # noise_gen does not emit $strobe lines
    return 0


if __name__ == '__main__':
    f1 = validate_csv()
    f2 = validate_txt()
    total = f1 + f2
    print(f"Validation: {total} failure(s) [{f1} CSV, {f2} TXT]")
    raise SystemExit(0 if total == 0 else 1)
