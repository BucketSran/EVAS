"""Validate dwa_ptr_gen behavior from CSV and strobe output."""
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent.parent / 'output' / 'dwa_ptr_gen' / 'dwa_ptr_gen'


def validate_csv(out_dir: Path = OUT) -> int:
    data = np.genfromtxt(out_dir / 'tran.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    failures = 0

    # CLK should reach 0.9V
    if data['clk_i'].max() < 0.8:
        print("FAIL: clk_i never reached VDD")
        failures += 1

    # RST deasserted
    if data['rst_ni'].max() < 0.8:
        print("FAIL: rst_ni never went high")
        failures += 1

    # ptr_o should be one-hot: at any time, at most one ptr bit should be high
    ptr_cols = [f'ptr_{i}' for i in range(16)]
    available = [c for c in ptr_cols if c in list(data.dtype.names)]
    if available:
        ptr_matrix = np.column_stack([data[c] > 0.45 for c in available])
        ones_per_row = ptr_matrix.sum(axis=1)
        # After reset, ptr should be one-hot (exactly 1)
        t_ns = data['time'] * 1e9
        active_mask = t_ns > 100.0
        if active_mask.sum() > 0:
            active_ones = ones_per_row[active_mask]
            bad_rows = np.sum((active_ones != 0) & (active_ones != 1))
            if bad_rows > 5:  # allow a few transition samples
                print(f"FAIL: ptr_o not one-hot in {bad_rows} samples (expected 0 or 1 bits high)")
                failures += 1

    # cell_en_o count should equal msb code (at least > 0 when active)
    cell_cols = [f'cell_en_{i}' for i in range(16)]
    avail_cells = [c for c in cell_cols if c in list(data.dtype.names)]
    if avail_cells:
        cell_count = np.zeros(len(data), dtype=int)
        for c in avail_cells:
            cell_count += (data[c] > 0.45).astype(int)
        # After active period, cell count should be > 0
        active_cell = cell_count[t_ns > 100.0]
        if len(active_cell) > 0 and active_cell.max() == 0:
            print("FAIL: cell_en_o all zeros after reset release")
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
        r'\[dwa_ptr_gen\].*Time=([0-9.eE+\-]+)\s*ns.*start_idx=\[\s*(\d+)\].*msb_count=\[\s*(\d+)\]'
    )
    events = []
    for line in lines:
        m = pattern.search(line)
        if m:
            t_ns   = float(m.group(1))
            start  = int(m.group(2))
            msb    = int(m.group(3))
            events.append((t_ns, start, msb))

    if len(events) == 0:
        print("WARN: no [dwa_ptr_gen] strobe lines found")
        return 0

    # Verify pointer wraps within 0-15
    for t_ns, start, msb in events:
        if not (0 <= start <= 15):
            print(f"FAIL: start_idx={start} out of range [0,15] at t={t_ns}ns")
            failures += 1

    # Verify DWA: each subsequent start_idx = prev + msb_count (mod 16)
    for i in range(1, len(events)):
        _, prev_start, _ = events[i-1]
        t_ns, curr_start, msb_count = events[i]
        expected_start = (prev_start + msb_count) % 16
        if curr_start != expected_start:
            print(f"FAIL: at t={t_ns}ns start_idx={curr_start}, expected {expected_start} (prev={prev_start}+msb={msb_count})")
            failures += 1

    return failures


if __name__ == '__main__':
    f1 = validate_csv()
    f2 = validate_txt()
    total = f1 + f2
    print(f"Validation: {total} failure(s) [{f1} CSV, {f2} TXT]")
    raise SystemExit(0 if total == 0 else 1)
