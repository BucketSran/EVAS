# Support Tiers

EVAS is an event-driven voltage-domain simulator. Benchmark reports should use
the support tiers below instead of treating every Verilog-A/AMS row as one flat
support target.

Unsupported-feature diagnostics include a stable `[support-tier: <name>]`
suffix in text output and a `support_tier` field in JSON lint output.

| Tier | Boundary | Benchmark interpretation |
|------|----------|--------------------------|
| `behavioral-event` | Voltage-domain behavioral transient models: `V(...)` reads/drives, `cross`/`above`/`timer`/`initial_step`/`final_step`, `transition`, state machines, control logic, table/random/file helpers. This is the current EVAS core strength. | A valid failure here is a supported EVAS bug unless a narrower diagnostic says otherwise. |
| `behavioral-continuous-time` | Legal voltage-domain `ddt`, `idt`, `idtmod`, `laplace_*`, `zi_*`, and `limexp` style models. EVAS has behavioral transient approximations for selected forms, but this is not a Spectre-equivalent continuous-time solver claim. | Treat unsupported or inaccurate rows as planned/limited continuous-time work, not as AMS/KCL coverage. |
| `ams-digital` | `wreal`, `logic`, `always`, continuous `assign`, packed logic vectors, `specify`/`specparam`, `connectmodule`, and `connectrules`. EVAS supports only small behavioral bridge subsets where documented; a full AMS/digital event kernel is outside the certified core. | Full AMS/digital failures are outside current benchmark certification unless the row is explicitly documented as a supported bridge subset. |
| `conservative-current-kcl` | `I(...) <+ ...`, branch currents, current probes, indirect branch equations, charge/current branch contributions, and KCL/MNA topology solving. | Architectural roadmap item, not a parser bug and not part of current certified EVAS support. |

Current EVAS benchmark PASS claims apply to `behavioral-event` designs, plus
explicitly documented behavioral helper subsets. Rows that require full
`ams-digital` or `conservative-current-kcl` semantics should be reported
separately from supported EVAS bugs.

## Diagnostic Taxonomy

`evas lint` and the optional `evas simulate --ahdllint` preflight expose the
same taxonomy. Unsupported current contributions and current probes report
`support_tier="conservative-current-kcl"`. Unsupported digital procedural syntax
reports `support_tier="ams-digital"` when it reaches the parser. Unknown
unregistered function/operator calls report `support_tier="outside-current-scope"`
so benchmark tooling can separate truly unclassified constructs from the four
documented support tiers.

Strict standalone Spectre compatibility is available through
`evas lint --spectre-strict`, `evas simulate --spectre-strict`, or
`simulatorOptions options spectre_strict=true`. This mode is intentionally
narrower than the default EVAS extension mode: it reports
`EVAS-COMP-ESPECTRESTRICT` for EVAS-supported syntax that the current standalone
Spectre target rejects, such as AMS bridge constructs (`logic`, `wreal`,
continuous `assign`, `always`), task/do-while extensions, runtime
electrical-node indexing, selected version-gated random distributions,
seeded `$rdist_*` distributions whose Spectre PRNG sequence parity is not
certified, integer bit/part-select concatenation gaps, `generate`, `specify`,
`connectmodule`, and `connectrules`.

## Continuous-Time Operators

EVAS supports `ddt()`, `idt()`, `idtmod()`, `laplace_nd()`, `laplace_np()`,
`laplace_zd()`, `laplace_zp()`, `zi_nd()`, `zi_np()`, `zi_zd()`, `zi_zp()`,
and `limexp()` as voltage-domain behavioral transient approximations. The
implementation uses finite-difference, trapezoidal integration, and simple
state/filter updates inside the behavioral evaluator. This is intentionally
separate from current-domain contribution stamping or full Spectre
transfer-function solving.

Spectre-restricted placements, such as conditional `idt()` or event-local
analog contributions, are reported through compatibility diagnostics.
Unsupported operators such as `absdelay()` are classified as
`behavioral-continuous-time` rather than as generic parser gaps.

## Noise and Stochastic Semantics

In ordinary transient analysis, `white_noise()`, `flicker_noise()`, and
`noise_table()` do not create arbitrary random transient waveforms. They return
a zero transient contribution and expose PSD data through the behavioral
`evaluate_noise()`, `noise_spectrum()`, and `integrated_noise()` helpers.

Explicit `$random`, `$dist_*()`, and `$rdist_*()` calls are behavioral random
draws. EVAS makes fixed-seed draws reproducible; the same seed and draw order
produce the same sequence across runs. Supported distributions include uniform,
normal, exponential, poisson, chi-square, t, and erlang variants. Unsupported
distribution names, such as an unimplemented `$rdist_gamma()`, fail with
`EVAS-COMP-EUNSUPPORTED` and `support_tier="behavioral-event"`.
Strict Spectre mode rejects seeded `$rdist_exponential`,
`$rdist_poisson`, `$rdist_normal`, and `$rdist_erlang` until their exact
Spectre PRNG sequences are certified; default EVAS extension mode keeps the
deterministic EVAS streams for reproducible behavioral simulation.

## Subprograms

EVAS supports Spectre-style function/task declarations such as `input x; real
x;`, ANSI-style task/function arguments, local variables, multi-argument calls,
bounded recursive functions, and task/function calls inside event bodies. These
are covered by conformance tests for simple functions, recursive functions,
task-local variables, and state updates through task calls.

## Vectors, Generate, and Electrical Indexing

Pure expression vector semantics are separate from electrical topology. EVAS
supports bit select, part select, concatenation, replication, reductions,
packed logic vectors, and integer/real state arrays.

Static `generate`/`genvar` elaboration is supported for the limited
behavioral/continuous-assign subset used by supported bridge models. This is
not the same as a full AMS generate/connect-rule implementation.

Runtime voltage-domain electrical indexing such as `V(bus[i])` and
`V(bus[i]) <+ ...` is supported by dynamic node resolution and cache-backed
node lookup. Runtime current-domain indexing belongs to the unsupported
`conservative-current-kcl` tier.

Strict Spectre mode accepts constant or statically elaborated electrical indexes
but rejects runtime electrical-node indexing so benchmark rows can be separated
from EVAS extension-mode capabilities.
It also rejects integer scalar bit/part-select and integer-select
concatenation forms whose Spectre output diverges from EVAS extension-mode
evaluation.

## AMS-Digital Boundary

EVAS includes a small `ams-digital` bridge subset: `logic` and `wreal` ports,
simple continuous `assign`, edge-sensitive `always @(posedge/negedge ...)`,
packed logic vector operations, and simple `specify`/`specparam` path delays on
behavioral assignments. Full four-state logic, connectmodule/connectrules
resolution, and a complete AMS event kernel are outside current benchmark
certification.

## KCL/MNA Boundary

EVAS does not certify conservative current solving. `I(...) <+ ...`, `I(...)`
current probes, indirect branch equations, charge/current contributions, and
SPICE-style AC/DC matrix solving are classified as `conservative-current-kcl`.
`evas lint` and `evas simulate --ahdllint` report this tier explicitly so these
rows are tracked as architectural roadmap work rather than parser bugs.
