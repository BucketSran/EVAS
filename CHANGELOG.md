# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## [0.4.6] — 2026-06-30

### Fixed — Verilog-A Compatibility
- Fixed affine lowering for valid voltage-domain expressions such as
  `V(out) <+ V(in) - param`, including symbolic parameter subtraction in the
  Rust static-affine path.
- Fixed compilation of real parameter gains applied to boolean-sum state
  variables, such as `gain * ones`, by avoiding tuple-to-float coercion in the
  simple state-output optimizer.
- Added source-level parsing and static elaboration for bitwise operators,
  shifts, unary bit-not, and right-associative power expressions in parameter
  defaults and variable initializers.

### Fixed — Rust Backend Coverage
- Let Rust event bodies execute parameter-bound `for` loops by lowering them to
  guarded body-IR loops when static unrolling cannot prove bounds.
- Added adaptive-step shrink floors that prevent runaway default error-control
  microsteps while preserving explicit `min_step`, source breakpoints, model
  breakpoints, bound-step contracts, and transition timing.

### Fixed — Stochastic and Event Timing
- Restored stochastic `transition()` ramp semantics while keeping Python and
  `evas-rust` random draws schedule-independent through per-seed draw indices.
- Updated cross-acceptance slack handling to use the measured Spectre lateness
  law as an explicit opt-in mode.

## [0.4.5] — 2026-06-25

### Fixed — Spectre Compatibility
- Reject Verilog-A identifiers that reuse Spectre-reserved built-in, simulator
  library, operator, and event function names while keeping legal function-call
  usage intact.
- Emit Spectre-style `VACOMP-2174` diagnostics for reserved identifiers,
  including the marked source line and matching reserved-name guidance.
- Suppress the incorrect `ahdl_include` fallback warning for normal bare
  includes resolved relative to the `.scs` file directory.

### Fixed — Transition Timing
- Folded in transition and cross-acceptance fixes that align stochastic and
  event-driven transition scheduling with the measured Spectre behavior.

## [0.4.4] — 2026-06-08

### Fixed — Engine Selection
- Restored the packaged default to the Python compatibility engine so `evas run`
  and `evas simulate` work from PyPI or a fresh source checkout without a
  pre-built Rust shared library.
- Added CLI `--engine` overrides for explicit EVAS2/Rust runs and updated docs
  to describe Rust backend build requirements.
- Synchronized runtime/docs version reporting with the package version.

### Changed — Examples
- Reduced the EVAS bundled example set to five smoke-test groups: `digital_basics`, `clk_div`, `comparator`, `adc_dac_ideal_4b`, and `noise_gen`.
- Removed the larger workflow-oriented example groups from this simulator package; those assets are intended to live with `veriloga-skills/evas-sim`.

## [0.3.0] — 2026-03-16

### Added — Language Features
- `case/endcase` statement: lexer keywords, AST node `CaseStatement`, parser, backend codegen (compiles to chained `if/elif/else`)
- `@(timer(period))` event: periodic firing every `period` seconds; engine breakpoints track next fire time
- `@(final_step)` event: fires after the main simulation loop ends, before result arrays are built
- `$temperature` expression: returns ambient temperature in Kelvin (default 27 °C → 300.15 K)
- `$vt` expression: thermal voltage kT/q (≈25.85 mV at 300.15 K)
- `$bound_step(dt)` system task: sets a per-model maximum timestep; engine respects it each iteration
- `$fopen(filename, mode)` / `$fclose(fd)` / `$fstrobe(fd, ...)` / `$fwrite(fd, ...)` / `$fdisplay(fd, ...)`: file I/O via Python `open()` with auto-close at simulation end

### Fixed
- **Cross detector double-trigger bug**: the `_tol = 1e-12` tolerance in `CrossDetector.check()` could leave `prev_val` slightly positive after a crossing, causing the detector to re-fire on the very next evaluation. Fixed by clamping `prev_val` to the post-crossing side. Same fix applied to `AboveDetector`.
- `test_cmp_offset_search` now passes without `xfail` — binary search converges to ≈10 mV (was stuck at 100 mV due to double-trigger)

### Changed — Examples
- `dwa_ptr_gen_msb` renamed to `dwa_ptr_gen_no_overlap` across all files (`.va`, `.scs`, `analyze_*.py`, `validate_*.py`, tests)
- Both DWA testbenches migrated from 4-bit PWL buses to a single analog voltage source + new `v2b_4b` ideal ADC block
- DWA clock: 10 MHz → **100 MHz** (T = 10 ns); stop time 1700 ns → 175 ns
- Input code sequence: 6 cycling values → 16 distinct values `[3,7,2,5,1,8,4,6,3,5,2,7,1,4,8,2]` with correct step-function PWL (hold points included)
- `analyze_cmp_strongarm`: input panel Y-axis now auto-scales to data range ± 20 % margin
- `analyze_cmp_offset_search`: VINP/VINN panel Y-axis now data-driven (was hardcoded 350–450 mV)
- `analyze_dwa_ptr_gen*`: plots saved directly to `output/dwa_ptr_gen/` (not subdirectories); no-overlap variant shows ptr diamond at last selected cell (shifted −1)

### Added — New Module
- `v2b_4b.va`: ideal 4-bit voltage-to-binary converter — samples `V(vin)` on CLK rising edge, maps 0–15 V to digital code `[0..15]`, drives four output bits via `transition()`

## [0.2.0] — 2026-03-16

### Added
- `@(timer(period))`, `@(final_step)`, `$temperature`, `$vt` — initial implementation (superseded by 0.3.0 with bug fixes)

## [0.1.2] — 2026-03-15

### Fixed
- `$strobe` / `$display` output now sorted strictly by simulation time across all module instances (previously grouped by instance instantiation order)

### Changed
- `comparator` example: delay panel now uses scatter plot (one point per CLK cycle) parsed from strobe log, replacing the misleading step-held waveform

## [0.1.1] — 2026-03-14

### Added
- Sphinx documentation with Apple Developer Documentation aesthetic (SF Pro fonts, `#0071e3` blue)
- busuanzi visitor counter
- Streamlined project folder structure

### Added
- Initial public release on PyPI
- Event-driven Verilog-A simulator engine
- Spectre netlist parser
- `evas simulate` CLI command
- `evas run <name>` to run any of 17 bundled examples
- `evas list` to enumerate available examples
- 17 example circuits: clk_div, lfsr, cmp_strongarm, SAR ADC, and more
- Pure-Python implementation — no C compiler, no ngspice required
