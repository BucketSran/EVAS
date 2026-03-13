"""Analyze LB_LFSR: Linear Feedback Shift Register output pattern."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'LB_LFSR'

# 1. Simulate
evas_simulate(str(HERE / 'tb_LB_LFSR.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

# 3. Plot: rstb | clk | dpn (LFSR output)
signals = ['rstb', 'clk', 'dpn']
fig, axes = plt.subplots(len(signals), 1, figsize=(12, 5), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 1.5, 1.5]})

for i, sig in enumerate(signals):
    axes[i].plot(t, df[sig], linewidth=1.0)
    axes[i].set_ylabel(f'{sig} (V)')
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_title('LB_LFSR')

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_LB_LFSR.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_LB_LFSR.png'}")
