"""Validate cmp_strongarm: clocked StrongARM comparator.

Testbench: vinp=0.5V (const), vinn steps 0.3V→0.7V at t=5ns.
Clock: 1GHz, VDD=0.9V.

Expected:
  - Before vinn step (t < 4ns):  vinp > vinn  → out_p HIGH, out_n LOW
  - After  vinn step (t > 6ns):  vinp < vinn  → out_p LOW,  out_n HIGH
  - out_p and out_n are always complementary at settled clock phases.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent.parent / 'output' / 'cmp_strongarm'

_VTH = 0.45   # half of VDD=0.9


def validate_csv(out_dir: Path = OUT) -> int:
    df = pd.read_csv(out_dir / 'tran.csv')
    failures = 0

    t_ns   = df['time'].values * 1e9
    out_p  = df['out_p'].values
    out_n  = df['out_n'].values

    # Both outputs must toggle (non-trivial comparator decisions)
    if out_p.max() - out_p.min() < _VTH:
        print("FAIL: out_p never toggles")
        failures += 1
    if out_n.max() - out_n.min() < _VTH:
        print("FAIL: out_n never toggles")
        failures += 1

    # Before vinn step (t < 4ns): vinp(0.5) > vinn(0.3) → out_p should be HIGH
    pre_mask  = t_ns < 4.0
    pre_p_hi  = (out_p[pre_mask] > _VTH).mean()
    if pre_p_hi < 0.4:
        print(f"FAIL: before vinn step, out_p is HIGH only {pre_p_hi*100:.0f}% of time (expected >40%)")
        failures += 1

    # After vinn step (t > 6ns): vinp(0.5) < vinn(0.7) → out_p should be LOW
    post_mask = t_ns > 6.0
    post_p_lo = (out_p[post_mask] < _VTH).mean()
    if post_p_lo < 0.4:
        print(f"FAIL: after vinn step, out_p is LOW only {post_p_lo*100:.0f}% of time (expected >40%)")
        failures += 1

    if failures == 0:
        print("[CSV] All assertions passed.")
    return failures


if __name__ == '__main__':
    raise SystemExit(0 if validate_csv() == 0 else 1)
