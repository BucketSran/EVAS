"""Analyze cmp_offset_search: binary search convergence for comparator offset."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'cmp_offset_search'

# 1. Simulate
evas_simulate(str(HERE / 'tb_cmp_offset_search.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

# 3. Plot
signals_top = ['CLK', 'dcmpp']
signals_mid = ['vinp_node', 'vinn_node']

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for i, sig in enumerate(signals_top):
    axes[0].plot(t, df[sig], linewidth=1.0, drawstyle='steps-post', label=sig)
vdd = df[signals_top].max().max()
axes[0].set_ylabel('Digital (V)')
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].legend(loc='upper right', fontsize=8)
axes[0].set_title('cmp_offset_search (voffset=10mV, binary search convergence)')
axes[0].grid(True, alpha=0.3)

for sig in signals_mid:
    axes[1].plot(t, df[sig] * 1e3, linewidth=1.0, label=sig)
axes[1].set_ylabel('VINP/VINN (mV)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

# Differential input voltage
vdiff = (df['vinp_node'] - df['vinn_node']) * 1e3
axes[2].plot(t, vdiff, linewidth=1.0, color='purple')
axes[2].axhline(10.0, color='red', linestyle='--', linewidth=1.0, label='target offset=10mV')
axes[2].set_ylabel('VINP-VINN (mV)')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_cmp_offset_search.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_cmp_offset_search.png'}")
