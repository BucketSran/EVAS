"""Analyze lfsr: Linear Feedback Shift Register output."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'lfsr'

evas_simulate(str(HERE / 'tb_lfsr.scs'), output_dir=str(OUT))

df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 1.5, 3]})
fig.suptitle('LFSR')

axes[0].plot(t, df['rstb'], linewidth=1.0)
axes[0].set_ylabel('rstb (V)')
vdd = df[['rstb', 'clk', 'dpn']].max().max()
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['clk'], linewidth=1.0)
axes[1].set_ylabel('clk (V)')
axes[1].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['dpn'], linewidth=1.0, color='tab:green')
axes[2].set_ylabel('dpn (V)')
axes[2].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_lfsr.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_lfsr.png'}")
