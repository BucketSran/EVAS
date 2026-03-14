"""Analyze ramp_gen: 12-bit up-ramp from 0 to 15 (step=1, N_CYCLE_START=2)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'ramp_gen'

# 1. Simulate
evas_simulate(str(HERE / 'tb_ramp_gen.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

# Decode 12-bit code
code_cols = [f'code_{i}' for i in range(12)]
ramp_code = np.zeros(len(df), dtype=int)
for i, col in enumerate(code_cols):
    if col in df.columns:
        ramp_code += ((df[col].values > 0.45).astype(int) << i)

# 3. Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, df['clk_dtc'], linewidth=1.0, drawstyle='steps-post', label='clk_dtc')
axes[0].plot(t, df['rst_n'],   linewidth=1.0, drawstyle='steps-post', label='rst_n', alpha=0.7)
vdd = df[['clk_dtc', 'rst_n']].max().max()
axes[0].set_ylabel('Control (V)')
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].legend(fontsize=8)
axes[0].set_title('ramp_gen (DIRECTION=1, MIN=0, MAX=127, STEP=1, N_CYCLE_START=2)')
axes[0].grid(True, alpha=0.3)

# Show individual bits
for i in [0, 1, 2, 3]:
    col = f'code_{i}'
    if col in df.columns:
        axes[1].plot(t, df[col] + i * 1.1, linewidth=1.0, drawstyle='steps-post', label=col)
axes[1].set_ylabel('LSB bits (stacked)')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, ramp_code, linewidth=1.0, drawstyle='steps-post', color='green')
axes[2].set_ylabel('ramp code (integer)')
axes[2].set_ylim(-1, 132)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_ramp_gen.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_ramp_gen.png'}")
