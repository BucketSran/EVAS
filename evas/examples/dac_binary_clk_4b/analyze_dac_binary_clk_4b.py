"""Analyze dac_binary_clk_4b: 4-bit clocked binary DAC full code sweep (0 → 15)."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from evas.netlist.runner import evas_simulate

HERE = Path(__file__).parent
OUT  = HERE.parent.parent / 'output' / 'dac_binary_clk_4b'

evas_simulate(str(HERE / 'tb_dac_binary_clk_4b.scs'), output_dir=str(OUT))

df = pd.read_csv(OUT / 'tran.csv')
t  = df['time'].values * 1e9  # → ns

vdd = 0.9
code = (df['din3'].values * 8 + df['din2'].values * 4
      + df['din1'].values * 2 + df['din0'].values * 1)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 2.5, 2.5]})
fig.suptitle('dac_binary_clk_4b — Full Code Sweep (0 → 15)')

axes[0].plot(t, df['rdy'], linewidth=1.0)
axes[0].set_ylabel('clk (V)')
axes[0].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, code, linewidth=1.0, drawstyle='steps-post')
axes[1].set_ylabel('input code')
axes[1].set_ylim(-0.5, 16.5)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, df['aout'], linewidth=1.0, color='tab:orange')
axes[2].set_ylabel('aout (V)')
axes[2].set_ylim(-vdd * 0.1, vdd * 1.2)
axes[2].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ns)')
fig.tight_layout()
fig.savefig(str(OUT / 'analyze_dac_binary_clk_4b.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {OUT / 'analyze_dac_binary_clk_4b.png'}")

# Bits figure
bits = ['din3', 'din2', 'din1', 'din0']
fig2, axes2 = plt.subplots(len(bits), 1, figsize=(12, 6), sharex=True)
fig2.suptitle('dac_binary_clk_4b — Input Bits')
for ax, bit in zip(axes2, bits):
    ax.plot(t, df[bit], linewidth=1.0)
    ax.set_ylabel(bit)
    ax.set_ylim(-vdd * 0.1, vdd * 1.2)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
axes2[-1].set_xlabel('Time (ns)')
fig2.tight_layout()
fig2.savefig(str(OUT / 'analyze_dac_binary_clk_4b_bits.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Plot saved: {OUT / 'analyze_dac_binary_clk_4b_bits.png'}")
