"""Analyze 4-stage inverter chain: td=100ps, tr=50ps per stage.

Runs the simulation and saves a waterfall plot showing delay propagation.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import time

import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
BASE = HERE.parent.parent / 'output' / 'digital_basics'
OUT  = BASE / 'inverter_chain'

# ── 1. Simulate ───────────────────────────────────────────────────────────────
t0 = time.perf_counter()
evas_simulate(str(HERE / 'tb_inverter_chain.scs'), output_dir=str(OUT))
sim_s = time.perf_counter() - t0

# ── 2. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9   # ns

signals = [
    ('in',  'IN',   'tab:blue'),
    ('n1',  'OUT1', 'tab:orange'),
    ('n2',  'OUT2', 'tab:green'),
    ('n3',  'OUT3', 'tab:red'),
    ('out', 'OUT4', 'tab:purple'),
]

# ── 3. Waterfall plot — IN at top, OUT4 at bottom ────────────────────────────
OFFSET = 1.1   # V between rows
N = len(signals)

fig, ax = plt.subplots(figsize=(12, 6))

for i, (col, label, color) in enumerate(signals):
    offset = (N - 1 - i) * OFFSET          # top → bottom
    ax.plot(t, df[col] + offset, color=color, linewidth=1, label=label)
    ax.axhline(offset,       color=color, linewidth=1, linestyle='--', alpha=0.4)
    ax.axhline(offset + 0.8, color=color, linewidth=1, linestyle='--', alpha=0.4)

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Voltage (V)  +  stage offset')
ax.set_title(f'4-stage inverter chain  —  td=100 ps, tr=50 ps per stage  [{sim_s:.3f} s]')

# Y ticks: label the 0 V and 0.8 V reference for each row
yticks  = []
ylabels = []
for i, (_, label, _) in enumerate(signals):
    yticks.append((N - 1 - i) * OFFSET)
    ylabels.append(label)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_ylim(-0.3, len(signals) * OFFSET + 0.2)

ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

path = BASE / 'analyze_inverter_chain.png'
fig.savefig(str(path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")
