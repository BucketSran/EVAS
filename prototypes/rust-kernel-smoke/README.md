# Rust Kernel Smoke Prototype

This directory contains a small EVAS-oriented microbenchmark. It is not a
Verilog-A simulator. Its only purpose is to test whether the current EVAS hot
path shape, roughly `dict[str, float]` plus per-step snapshots and helper calls,
has enough overhead to justify a native indexed-array backend.

The benchmark compares:

- `python_dict_baseline.py`: string node names, dictionaries, per-step `dict`
  snapshots, and tuple model maps.
- `src/main.rs`: integer node indices, `Vec<f64>`, and double-buffer snapshots.
- `project_rust_backend.py`: a small projection script that applies targeted
  Rust-kernel assumptions to the real r14 EVAS/Spectre AX subprocess timings.

Run:

```bash
cargo test --release
cargo run --release -- --kernel measurement-indexed --steps 200000 --models 64 --record-stride 16
cargo run --release -- --kernel pfd-fixed-step --steps 60062 --models 64 --record-stride 16
cargo run --release -- --kernel pfd-event-queue --steps 60062 --models 64 --record-stride 16
python3 python_dict_baseline.py --steps 200000 --models 64 --record-stride 16
python3 project_rust_backend.py
```

## Initial Result

Run on 2026-06-02 with `rustc 1.95.0` / `cargo 1.95.0`:

| Engine | Kernel | Steps | Models | Elapsed |
| --- | --- | ---: | ---: | ---: |
| Python variant | measurement dict | 200000 | 64 | 14.872653 s |
| Python variant | measurement indexed | 200000 | 64 | 14.003133 s |
| Rust | measurement indexed | 200000 | 64 | 0.149962 s |
| Python variant | PFD fixed-step | 60062 | 64 | 1.217709 s |
| Python variant | PFD event-queue | 4719 processed | 64 | 0.114909 s |
| Rust | PFD fixed-step | 60062 | 64 | 0.005754 s |
| Rust | PFD event-queue | 4719 processed | 64 | 0.000485 s |

Repeated benchmark result, `3` runs per variant, median reported:

| Engine | Kernel | Median | Min | Max | Processed steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Python variant | measurement dict | 16.590893 s | 11.949536 s | 19.012042 s | 200000 |
| Python variant | measurement indexed | 9.852070 s | 9.736721 s | 10.091303 s | 200000 |
| Rust | measurement indexed | 0.135588 s | 0.135215 s | 0.141951 s | 200000 |
| Python variant | PFD fixed-step | 0.732351 s | 0.700230 s | 0.747094 s | 60062 |
| Python variant | PFD event-queue | 0.060759 s | 0.060687 s | 0.063158 s | 4719 |
| Rust | PFD fixed-step | 0.004671 s | 0.004669 s | 0.004823 s | 60062 |
| Rust | PFD event-queue | 0.000474 s | 0.000472 s | 0.000496 s | 4719 |

Median speedups:

| Comparison | Speedup |
| --- | ---: |
| Python indexed / Python dict | 1.684x |
| Rust indexed / Python dict | 122.363x |
| Rust indexed / Python indexed | 72.662x |
| Python PFD event queue / Python PFD fixed-step | 12.053x |
| Rust PFD event queue / Rust PFD fixed-step | 9.854x |

Raw repeated-run output is stored in `benchmark_results_20260602.json`.

Both engines produced the same event count and checksum:

```text
events=197 checksum=8096.309486241 err_acc=1170.783750964
```

The measurement toy has an important split:

- Pure Python `dict -> indexed list` improves the median from `16.590893 s` to
  `9.852070 s` (`1.684x`). This is useful, but it is not enough to explain a
  large simulator speedup by itself.
- Rust indexed-array execution is `0.135588 s` median for the same workload, so
  the per-step hot loop should move to a native backend if measurement-heavy
  rows remain important.

The PFD toy shows a separate direction: an event queue can cut processed steps
from `60062` to `4719` while preserving the same event count in this toy
problem. In the repeated benchmark, Python fixed-step to event-queue improves
from `0.732351 s` to `0.060759 s` median (`12.053x`). The real EVAS PFD rows
still need Spectre-equivalence validation.

## Planned Python Kernel Changes

The Python implementation should change in stages instead of jumping directly
to a full Rust rewrite:

1. Add an indexed node table at netlist lowering time:
   `node_name -> node_id`, `save_signal -> node_id`, `model port -> node_id`.
2. Add an indexed transient backend behind the existing simulator API:
   `prev: list[float]`, `curr: list[float]`, `state: list[float]`.
3. Keep Python checkers and CSV/reporting unchanged at the boundary.
4. Implement event-queue scheduling for timer/breakpoint-heavy rows in Python
   first, because the PFD toy shows this can pay off before Rust integration.
5. Move the indexed transient backend into Rust once the IR shape is stable.

Expected effect:

- Python-only indexed nodes: small per-step improvement; useful mainly as a
  migration bridge.
- Python event queue: potentially large for PFD/PLL rows if parity is preserved.
- Rust indexed backend: required for measurement-heavy rows where Python loop
  overhead dominates.

## r14 Projection

The projection script changes only the already-known hot rows:

- measurement-heavy rows target a lower per-step cost.
- `vbr1_l1_pfd_small_phase_error_response` targets event-queue runtime.

| Scenario | Assumption | Projected EVAS subprocess | AX/EVAS speedup |
| --- | --- | ---: | ---: |
| Current r14 | no Rust-kernel change | 100.368435 s | 2.073x |
| Conservative | measurement `60 us/step`, PFD `1.20 s/row` | 80.039626 s | 2.600x |
| Balanced | measurement `50 us/step`, PFD `0.70 s/row` | 73.615546 s | 2.826x |
| Aggressive | measurement `35 us/step`, PFD `0.40 s/row` | 65.779426 s | 3.163x |

This is a planning estimate, not a claim. The next required step is a real-row
Rust replay prototype for `vbr1_l1_gain_estimator` and a Spectre-parity checked
event-queue prototype for `vbr1_l1_pfd_small_phase_error_response`.

Interpretation:

- This benchmark only measures the data-structure/native-loop direction.
- It does not test Verilog-A semantics, event correctness, or Spectre parity.
- If Rust is much faster here, the next useful prototype is an EVAS IR runner
  for two real rows: `vbr1_l1_gain_estimator` and
  `vbr1_l1_pfd_small_phase_error_response`.
