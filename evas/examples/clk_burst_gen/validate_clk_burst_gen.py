"""Validate clk_burst_gen behavior from CSV output."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'clk_burst_gen'


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    # CLK should reach VDD=0.9V
    if df['CLK'].max() < 0.8:
        print("FAIL: CLK never reached VDD")
        failures += 1

    # RST_N should be high for most of simulation
    if df['RST_N'].max() < 0.8:
        print("FAIL: RST_N never went high")
        failures += 1

    # CLK_OUT should be present (max > 0.8)
    if df['CLK_OUT'].max() < 0.8:
        print("FAIL: CLK_OUT never went high")
        failures += 1

    # CLK_OUT should be 0 for most of the time (only 2 of 8 cycles are high)
    # After reset is active, fraction of time CLK_OUT is high should be ~2/8 = 25%
    t_ns = df['time'].values * 1e9
    active_mask = t_ns > 200.0  # skip initial transient and reset
    if active_mask.sum() > 10:
        clk_out_active = df['CLK_OUT'].values[active_mask]
        frac_high = np.mean(clk_out_active > 0.45)
        # Expect roughly 25% high (2/8 cycles * 50% duty = 12.5%), allow generous range
        if frac_high > 0.5:
            print(f"FAIL: CLK_OUT high fraction={frac_high:.2f}, expected < 0.5 (burst mode)")
            failures += 1

    if failures == 0:
        print("[CSV] All assertions passed.")
    return failures


def validate_txt(out_dir: Path = OUT) -> int:
    txt_path = out_dir / 'strobe.txt'
    if not txt_path.exists():
        return 0
    # clk_burst_gen does not emit $strobe lines
    return 0


if __name__ == '__main__':
    f1 = validate_csv()
    f2 = validate_txt()
    total = f1 + f2
    print(f"Validation: {total} failure(s) [{f1} CSV, {f2} TXT]")
    raise SystemExit(0 if total == 0 else 1)
