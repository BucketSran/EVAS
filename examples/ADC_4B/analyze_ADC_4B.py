"""Analyze ADC_4B: 4-bit SAR ADC full-sweep (GND → VDD)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'ADC_4B'

# 1. Simulate
evas_simulate(str(HERE / 'tb_ADC_4B.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

# 3. Plot: clks | vin | dout_code
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.5, 2.5]})

axes[0].plot(t, df['clks'], linewidth=1.0)
axes[0].set_ylabel('clks (V)')
axes[0].set_title('ADC_4B')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['vin'], linewidth=1.0)
axes[1].set_ylabel('vin (V)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['dout_code'], linewidth=1.0)
axes[2].set_ylabel('dout_code')
axes[2].set_ylim(-0.5, int(df['dout_code'].max()) + 1.5)
axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_ADC_4B.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_ADC_4B.png'}")
