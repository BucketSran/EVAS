"""Validate edge_interval_timer behavior from CSV and strobe output.

CLK_1 period=200ns, delay=5ns -> rising edges at 5, 205, 405, ... ns
CLK_2 period=200ns, delay=30ns -> rising edges at 30, 230, 430, ... ns
Expected interval = 30 - 5 = 25ns = 25000 ps
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'edge_interval_timer'

_EXPECTED_DELAY_PS = 25000.0  # 25ns in ps
_TOL_PS = 2000.0  # 2ns tolerance


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    # OUT_PS should stabilize around expected delay
    t_ns = df['time'].values * 1e9
    # After first measurement (~30ns), OUT_PS should be ~25000ps
    late_mask = t_ns > 200.0
    if late_mask.sum() > 0:
        out_ps_late = df['OUT_PS'].values[late_mask]
        # Filter out zero samples (before first measurement)
        nonzero = out_ps_late[out_ps_late > 100]
        if len(nonzero) == 0:
            print("FAIL: OUT_PS never had a nonzero measurement")
            failures += 1
        else:
            mean_ps = float(np.mean(nonzero))
            if abs(mean_ps - _EXPECTED_DELAY_PS) > _TOL_PS:
                print(f"FAIL: OUT_PS mean={mean_ps:.0f}ps, expected ~{_EXPECTED_DELAY_PS:.0f}ps (tol={_TOL_PS:.0f}ps)")
                failures += 1

    # Clocks should reach VDD
    if df['CLK_1'].max() < 0.8:
        print("FAIL: CLK_1 never reached high")
        failures += 1
    if df['CLK_2'].max() < 0.8:
        print("FAIL: CLK_2 never reached high")
        failures += 1

    if failures == 0:
        print("[CSV] All assertions passed.")
    return failures


def validate_txt(out_dir: Path = OUT) -> int:
    txt_path = out_dir / 'strobe.txt'
    if not txt_path.exists():
        return 0
    lines = txt_path.read_text().splitlines()
    failures = 0

    pattern = re.compile(
        r'\[edge_interval_timer\].*Time=([0-9.eE+\-]+)\s*ns.*Delay=([0-9.eE+\-]+)\s*ps'
    )
    measurements = []
    for line in lines:
        m = pattern.search(line)
        if m:
            t_ns   = float(m.group(1))
            delay  = float(m.group(2))
            measurements.append((t_ns, delay))

    if len(measurements) == 0:
        print("WARN: no [edge_interval_timer] strobe lines found")
        return 0

    # Each measurement should be ~25000ps
    for t_ns, delay in measurements:
        if abs(delay - _EXPECTED_DELAY_PS) > _TOL_PS:
            print(f"FAIL: at t={t_ns:.1f}ns delay={delay:.0f}ps, expected ~{_EXPECTED_DELAY_PS:.0f}ps")
            failures += 1

    print(f"[TXT] {len(measurements)} measurements checked.")
    return failures


if __name__ == '__main__':
    f1 = validate_csv()
    f2 = validate_txt()
    total = f1 + f2
    print(f"Validation: {total} failure(s) [{f1} CSV, {f2} TXT]")
    raise SystemExit(0 if total == 0 else 1)
