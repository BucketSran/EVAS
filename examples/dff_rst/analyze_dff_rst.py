"""Analyze dff_rst: D Flip-Flop with synchronous reset."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'dff_rst'

# 1. Simulate
evas_simulate(str(HERE / 'tb_dff_rst.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e6  # → us

# 3. Plot: 5 digital waveforms stacked
signals = ['clk', 'data', 'reset', 'q', 'qbar']
fig, axes = plt.subplots(len(signals), 1, figsize=(12, 7), sharex=True)

for i, sig in enumerate(signals):
    axes[i].plot(t, df[sig], linewidth=1.0)
    axes[i].set_ylabel(sig)
    axes[i].set_ylim(-0.5, df[sig].max() * 1.2 + 0.5)
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_title('dff_rst')

axes[-1].set_xlabel('Time (us)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_dff_rst.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_dff_rst.png'}")
