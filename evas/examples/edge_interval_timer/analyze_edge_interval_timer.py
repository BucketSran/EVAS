"""Analyze edge_interval_timer: measures CLK_1 to CLK_2 rising-edge delay."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'edge_interval_timer'

# 1. Simulate
evas_simulate(str(HERE / 'tb_edge_interval_timer.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

# 3. Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].plot(t, df['CLK_1'], linewidth=1.0, drawstyle='steps-post', label='CLK_1')
axes[0].plot(t, df['CLK_2'], linewidth=1.0, drawstyle='steps-post', label='CLK_2', alpha=0.7)
vdd = df[['CLK_1', 'CLK_2']].max().max()
axes[0].set_ylabel('Clocks (V)')
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].legend(fontsize=8)
axes[0].set_title('edge_interval_timer (CLK_2 delayed ~25ns from CLK_1)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['OUT_PS'], linewidth=1.0, color='green', drawstyle='steps-post')
axes[1].set_ylabel('OUT_PS (ps)')
axes[1].axhline(25000, color='red', linestyle='--', linewidth=1.0, label='expected ~25000 ps')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Zoom into first 1000ns for clarity
axes[2].plot(t, df['CLK_1'], linewidth=1.0, drawstyle='steps-post', label='CLK_1')
axes[2].plot(t, df['CLK_2'], linewidth=1.0, drawstyle='steps-post', label='CLK_2', alpha=0.7)
axes[2].set_xlim(0, 800)
axes[2].set_ylabel('Clocks zoom (V)')
axes[2].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_edge_interval_timer.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_edge_interval_timer.png'}")
