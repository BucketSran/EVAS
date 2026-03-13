"""Analyze DAC_11B: 11-bit weighted DAC input-to-output transfer."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'DAC_11B'

# 1. Simulate
evas_simulate(str(HERE / 'tb_DAC_11B.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

# 3. Plot: rdy | din_12 / din_6 / din_1 (digital) | aout (analog)
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.0, 2.5]})

axes[0].plot(t, df['rdy'], linewidth=1.0)
axes[0].set_ylabel('rdy (V)')
axes[0].set_title('DAC_11B')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['din_12'], linewidth=1.0, label='din_12')
axes[1].plot(t, df['din_6'],  linewidth=1.0, label='din_6')
axes[1].plot(t, df['din_1'],  linewidth=1.0, label='din_1')
axes[1].set_ylabel('din (V)')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['aout'], linewidth=1.0, color='tab:orange')
axes[2].set_ylabel('aout (V)')
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_DAC_11B.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_DAC_11B.png'}")
