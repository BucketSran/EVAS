"""Analyze DAC_4B: 4-bit binary-weighted DAC full code sweep (0 → 15)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'DAC_4B'

# 1. Simulate
evas_simulate(str(HERE / 'tb_DAC_4B.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

# Reconstruct input code from individual bits
vdd = 0.9
code = (df['din3'].values * 8
      + df['din2'].values * 4
      + df['din1'].values * 2
      + df['din0'].values * 1)

# 3. Plot: rdy | input code | aout (staircase)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.5, 2.5]})

axes[0].plot(t, df['rdy'], linewidth=1.0)
axes[0].set_ylabel('rdy (V)')
axes[0].set_title('DAC_4B — Full Code Sweep (0 → 15)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, code, linewidth=1.0, drawstyle='steps-post')
axes[1].set_ylabel('input code')
axes[1].set_ylim(-0.5, 16.5)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['aout'], linewidth=1.0, color='tab:orange')
axes[2].set_ylabel('aout (V)')
axes[2].set_ylim(-0.05, vdd + 0.05)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_DAC_4B.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_DAC_4B.png'}")
