"""Analyze cmp_strongarm: clocked strong-arm comparator."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'cmp_strongarm'

evas_simulate(str(HERE / 'tb_cmp_strongarm.scs'), output_dir=str(OUT))

df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.5, 2.5]})
fig.suptitle('cmp_strongarm — Clocked Comparator')

axes[0].plot(t, df['clk'], linewidth=1.0)
axes[0].set_ylabel('clk (V)')
vdd = df['clk'].max()
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['vinp'], linewidth=1.0, label='vinp')
axes[1].plot(t, df['vinn'], linewidth=1.0, label='vinn')
axes[1].set_ylabel('input (V)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['out_p'], linewidth=1.0, label='out_p')
axes[2].plot(t, df['out_n'], linewidth=1.0, label='out_n')
axes[2].set_ylabel('output (V)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_cmp_strongarm.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_cmp_strongarm.png'}")
