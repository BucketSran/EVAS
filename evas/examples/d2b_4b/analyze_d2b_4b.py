"""Analyze d2b_4b: unified static code driver (trim_code=9).

Six subplots: bin, bin_n, onehot, onehot_n, therm, therm_n.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'd2b_4b'

# 1. Simulate
evas_simulate(str(HERE / 'tb_d2b_4b.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

bin_cols      = [f'bin_o_{i}'      for i in range(4)]
bin_n_cols    = [f'bin_n_o_{i}'    for i in range(4)]
onehot_cols   = [f'onehot_o_{i}'   for i in range(16)]
onehot_n_cols = [f'onehot_n_o_{i}' for i in range(16)]
therm_cols    = [f'therm_o_{i}'    for i in range(15)]
therm_n_cols  = [f'therm_n_o_{i}'  for i in range(15)]

# 3. Plot — 6 subplots
fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
fig.suptitle('d2b_4b (trim_code=9)', fontsize=12)

# Helper: stacked bit plot
def plot_stacked(ax, cols, title):
    for i, col in enumerate(cols):
        if col in df.columns:
            ax.plot(t, df[col] + i * 1.1, linewidth=1.0,
                    drawstyle='steps-post',
                    label=col if i in [0, len(cols) - 1] else '')
    ax.set_ylabel('bits (stacked)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

plot_stacked(axes[0], bin_cols,      'bin_o[3:0] — binary active-high')
plot_stacked(axes[1], bin_n_cols,    'bin_n_o[3:0] — binary active-low')
plot_stacked(axes[2], onehot_cols,   'onehot_o[15:0] — one-hot active-high')
plot_stacked(axes[3], onehot_n_cols, 'onehot_n_o[15:0] — one-cold active-low')
plot_stacked(axes[4], therm_cols,    'therm_o[14:0] — thermometer active-high')
plot_stacked(axes[5], therm_n_cols,  'therm_n_o[14:0] — thermometer active-low')

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_d2b_4b.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_d2b_4b.png'}")
