"""Analyze dac_therm_16b: 16-bit thermometer DAC (vstep=1.0V)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'dac_therm_16b'

# 1. Simulate
evas_simulate(str(HERE / 'tb_dac_therm_16b.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

# Count thermometer ones
din_cols = [f'd{i}' for i in range(16)]
ones_count = np.zeros(len(df), dtype=int)
for col in din_cols:
    if col in df.columns:
        ones_count += (df[col].values > 0.45).astype(int)

# 3. Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, df['rst_n'], linewidth=1.0, drawstyle='steps-post', color='orange')
axes[0].set_ylabel('rst_n')
vdd = df['rst_n'].max()
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].set_title('dac_therm_16b (thermometer DAC, vstep=1.0V)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, ones_count, linewidth=1.0, drawstyle='steps-post', color='steelblue')
axes[1].set_ylabel('thermometer ones count')
axes[1].set_ylim(-1, 18)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['vout'], linewidth=1.0, color='green')
axes[2].set_ylabel('vout (V)')
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_dac_therm_16b.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_dac_therm_16b.png'}")
