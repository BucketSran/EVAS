"""Analyze noise_gen: Gaussian noise added to a DC input (sigma=0.1V, vin=1.0V)."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'noise_gen'

# 1. Simulate
evas_simulate(str(HERE / 'tb_noise_gen.scs'), output_dir=str(OUT))

# 2. Load results
df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # -> ns

noise = df['vout_o'].values - df['vin_i'].values

# 3. Plot - overlay vin and vout to show noise
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Top: overlay vin and vout
axes[0].plot(t, df['vin_i'], linewidth=1.0, color='steelblue', label='vin_i (DC=1.0V)', zorder=3)
axes[0].plot(t, df['vout_o'], linewidth=1.0, color='tomato', alpha=0.8, label='vout_o (noisy)')
axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('noise_gen (sigma=0.1V, vin=1.0V DC): overlay of input and noisy output')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
# Zoom y-axis to show noise
v_mean = float(df['vin_i'].mean())
axes[0].set_ylim(v_mean - 0.5, v_mean + 0.5)

# Bottom: noise signal
axes[1].plot(t, noise, linewidth=1.0, color='purple', alpha=0.8)
axes[1].axhline(0, color='black', linewidth=1.0, linestyle='--')
axes[1].axhline( 0.1, color='red', linewidth=1.0, linestyle=':', label='+1σ')
axes[1].axhline(-0.1, color='red', linewidth=1.0, linestyle=':', label='-1σ')
axes[1].set_ylabel('Noise (V)')
axes[1].set_xlabel('Time (ns)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUT / 'analyze_noise_gen.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_noise_gen.png'}")
