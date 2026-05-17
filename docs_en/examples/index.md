# Examples Catalog

EVAS ships with **5 compact example groups**, each containing one or more Verilog-A model files (`.va`),
Spectre-format testbench netlists (`.scs`), and Python analysis / visualisation scripts.

Use `evas run <name>` to run any bundled example. The larger workflow-oriented
example library is maintained outside the simulator package in
`veriloga-skills/evas-sim`.

## Available Examples

| Group | Variants / sub-examples |
|-------|------------------------|
| `clk_div` | Clock divider (ratio = 4) |
| `digital_basics` | AND, OR, NOT gates; D flip-flop with reset; inverter chain |
| `noise_gen` | Gaussian noise generator |
| `adc_dac_ideal_4b` | 4-bit ideal ADC + DAC with sample-hold: a) ramp  b) single-tone sine  c) 1000-point sine |
| `comparator` | a) Ideal comparator  b) StrongARM clocked comparator  c) Binary-search offset calibration  d) Propagation delay measurement |

## Running a specific sub-example

Use `--tb` to select a testbench when an example has multiple:

```bash
# adc_dac_ideal_4b stimuli
evas run adc_dac_ideal_4b --tb tb_adc_dac_ideal_4b_ramp.scs
evas run adc_dac_ideal_4b --tb tb_adc_dac_ideal_4b_sine.scs

# comparator sub-examples
evas run comparator --tb tb_cmp_ideal.scs
evas run comparator --tb tb_cmp_strongarm.scs
evas run comparator --tb tb_cmp_offset_search.scs
evas run comparator --tb tb_cmp_delay.scs

# digital_basics gates
evas run digital_basics --tb tb_and_gate.scs
evas run digital_basics --tb tb_not_gate.scs
```
