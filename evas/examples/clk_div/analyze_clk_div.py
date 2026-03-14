"""Analyze clk_div: clock divider by ratio=4."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'clk_div'

# 1. Simulate
evas_simulate(str(HERE / 'tb_clk_div.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

# 3. Plot
signals = ['clk_in', 'clk_out']
fig, axes = plt.subplots(len(signals), 1, figsize=(12, 5), sharex=True)

for i, sig in enumerate(signals):
    axes[i].plot(t, df[sig], linewidth=1.0, drawstyle='steps-post')
    axes[i].set_ylabel(sig)
    axes[i].set_ylim(-df[sig].max() * 0.1, df[sig].max() * 1.2)
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_title('clk_div (ratio=4: output period = 4x input period)')

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_clk_div.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_clk_div.png'}")
