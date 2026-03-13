"""Analyze L2_comparator: Clocked differential comparator."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'L2_comparator'

# 1. Simulate
evas_simulate(str(HERE / 'tb_L2_comparator.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

# 3. Plot: clk | vinp & vinn (overlay) | out_p & out_n (overlay)
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.5, 1.5]})

axes[0].plot(t, df['clk'], linewidth=1.0)
axes[0].set_ylabel('clk (V)')
axes[0].set_title('L2_comparator')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['vinp'], linewidth=1.0, label='vinp')
axes[1].plot(t, df['vinn'], linewidth=1.0, label='vinn')
axes[1].set_ylabel('V (V)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['out_p'], linewidth=1.0, label='out_p')
axes[2].plot(t, df['out_n'], linewidth=1.0, label='out_n', linestyle='--')
axes[2].set_ylabel('output (V)')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_L2_comparator.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_L2_comparator.png'}")
