# Iteration 3 official synthetic failure audit

Date: 2026-07-18 (Europe/Berlin)  
Disposition: **CONSUMED / INCOMPLETE / FAILED-EXECUTION**  
Real-data stage: **WITHHELD**

The exact official command was run once:

```bash
.venv/bin/python benchmarks/inverse_projection_fiber_iter3.py \
  --mode official --confirm-official-roots
```

The exclusive ATTEMPT was durably written, root 0 completed, root 1 wrote only its initial PLY,
and root 2 was never started. No top-level synthetic RESULT exists. The same official roots must
not be resumed, reconstructed, or rerun. The real ATTEMPT, RESULT, output directory, bundle fit,
validation fields, and C1004 report data remain untouched.

Terminal-only context reported
`RuntimeError: a supported projection left the valid camera domain during M-step` at
`src/rtgs/lift/fiber_correspondence.py:939`. Because stderr was not saved and arm-level progress is
written only after a root completes, the exact root-1 arm, outer step, geometry substep, view, and
track are not artifact-grade facts. The durable statement is only that root 1 stopped after
initialization and before completed arm artifacts.

## Independently recomputed partial evidence

The read-only audit recomputed association, declared-capacity dust, geometry validity, source
invariants, held-out assignment, and UOT mass checks directly from root 0's sealed NPZ. Every
checked value matched `ROOT_RESULT.json`.

| Arm | Purity | Completeness | Track outlier recall | Observation outlier recall | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardmin A | 0.4248 | 0.75 | 0.0000 | n/a | 0.5000 |
| row B | 0.4665 | 0.50 | 0.6285 | n/a | 0.5125 |
| UOT-uniform C | 0.5259 | 0.25 | 0.2068 | 0.0349 | 0.4688 |
| UOT-area D | 0.5468 | 0.25 | 0.2730 | 0.0560 | 0.4750 |
| oracle O | 1.0000 | 1.00 | 1.0000 | 1.0000 | 0.6125 |
| shuffled S | 0.4342 | 0.125 | 0.2561 | 0.0384 | 0.4688 |

D's source center/covariance equalities held to `7.94e-15 px` and `7.29e-16` relative error;
all geometry was finite/SPD and all UOT mass diagnostics passed. This is therefore an association
and model-construction failure, not a broken source projection or collapsed transport calculation.

The frozen real release requires, in **every** root, purity and completeness at least `0.90`, both
outlier recalls at least `0.80`, both inlier dust false-positive rates at most `0.20`, and valid
mass diagnostics. Root 0 rejects D and C on purity, completeness, and both recall routes. No result
from roots 1–2 could restore either arm's all-root acceptance. The formal three-root result remains
incomplete, but real-data release is conclusively impossible.

Root-local purity differences were D-A `+0.1220`, D-B `+0.0803`, D-C `+0.0209`, and D-S
`+0.1125`. These are partial diagnostics, not the preregistered three-root mean effects. Capacity
attribution, aggregate negative control, and cross-root stability remain unresolved.

## What this says about the method

There is a real but insufficient soft-transport signal: D improves root-local purity over hardmin,
row-softmax, and shuffled evidence. It does not preserve enough correct capacity in every view,
and it routes most outlier capacity into real matches. Area weighting changes purity only slightly
relative to uniform UOT and does not solve unmatchedness.

More importantly, oracle labels do not recover the latent 3D geometry: oracle center p90 is
`1.0593`, and held-out parent assignment is only `0.6125`. The synthetic input explains why the
original hard-ray statement needs a granularity qualification. Of 80 inlier split children,
83.75% have nonzero displacement from their view/parent moment center; the median is `0.522 px`,
p90 `0.843 px`, and maximum `1.114 px`. A child produced by an arbitrary 2D moment split is not
itself the projection of the single latent parent Gaussian. Pinning every such fragment to its own
exact ray preserves the fragment perfectly but makes those raw fragments mutually inconsistent as
copies of one parent.

The corrected conclusion is: **keep the exact fiber for a stable 2D track or moment-merged source
aggregate, not automatically for every independently fitted 2D mixture component.** The fiber
math remains sound; the missing object is track topology at the same time as fitting. Some oracle
error can also come from the optimizer, so the partial root does not allocate all geometry error to
granularity alone.

## Problems and proposed repairs

1. **Unsafe frozen-plan update.** With two optimizer steps per E-step, the first tentative update
   can move a plan-supported projection outside a secondary camera's valid domain; the next M-step
   then fails closed. Make updates transactional: save parameters and optimizer state, apply the
   step, validate every supported projection, and roll back/backtrack on invalidity. A future
   protocol can instead recompute assignment after every geometry step and add a frustum barrier or
   support-conditioned depth interval.
2. **Insufficient failure provenance.** Future one-shot runners should write an exclusive failure
   receipt containing root, arm, temperature index, geometry substep, failing views/tracks,
   exception, and source/protocol hashes. Per-arm checkpoints must be written before the next arm.
3. **Wrong primitive granularity.** Add topology to the latent state. Merge source-side fragments
   by weighted moment matching and attach the exact fiber to that aggregate; alternatively use
   birth/death/merge competition so one surviving track owns a representative or evolving source
   aggregate. Test this with an oracle-topology ceiling before learned matching.
4. **Poor unmatched model.** Dust cost `4.0` leaves 72.7% of outlier-track capacity and 94.4% of
   outlier-observation capacity outside dust in D. Use an explicit calibrated outlier likelihood or
   null distribution and sparse epipolar/visibility candidate masks. Keep mass capacity, posterior
   association, existence, and rendering opacity separate.
5. **Evidence-loop stop.** This was the third and final iteration. There is no Iteration 4 repair
   or corrected official retry. Any future experiment needs a newly authorized question, fresh
   roots, and a new preregistration; the current artifacts stay immutable.

## Claim table

| Claim | Kind/scope | Disposition | Evidence |
| --- | --- | --- | --- |
| Exact fiber preserves its source projection | measured, synthetic root 0 | confirmed | max errors about `1e-14` / `1e-15` |
| D passes the absolute association mechanism | preregistered, all-root | refuted | root 0 D `0.5468` purity, `0.25` completeness |
| C can serve as real-data fallback | preregistered, all-root | refuted | root 0 C `0.5259` / `0.25`, both recalls below floor |
| Area capacity is beneficial | causal, three-root | unresolved | only one root; D-C purity `+0.0209` |
| Exact fibers plus true labels recover geometry | measured, root 0 oracle | refuted for this construction | center p90 `1.0593`, held-out `0.6125` |
| Calibrated bundle interaction works | real exploratory | withheld | synthetic release impossible; real namespace untouched |
| Runtime or GPU behavior | performance | unverified | CPU-only, contended-host wall time descriptive |

## Evidence and checks

- ATTEMPT SHA-256:
  `3a2b3cdb33e4e68574c1adeb3e44a3a59dfd1f6a22bbd5c980ebf2eafaf07bd1`
- Root-0 result SHA-256:
  `3119fcf8c01b72ade316e3f9af00d7c8e31d10e34f47a10bf5a79cce0450b56b`
- Root-0 NPZ SHA-256:
  `a86c7e7c1624edcf55a26a3f4ed1988122cb5d59a4fb252bb14cae9b4b1ff2bd`
- Root-1 initial PLY SHA-256:
  `0a79e2e206f8885c4e0645f8f895436b2b94a212388ad333657cf8693272e336`
- Read-only audit script SHA-256:
  `10d8f05ade7f3876ba4da18636930afdb912b3ac33bc41219daa5a3c42bb7db6`

The audit script was linted/formatted and recomputed all checked root-0 values without importing
the generator or constructing an official root. An independent reviewer reproduced C/D directly
from the NPZ, confirmed the fail-closed transaction, and accessed neither official roots nor real
data. CUDA/GPU, real-data, appearance, C1004, and viewer work were deliberately skipped because the
synthetic release gate failed.

The final focused fiber/transport/harness suite passed 84/84 tests. A first unisolated
`./scripts/verify.sh` exposed 21 visible-CUDA environment failures: deterministic-algorithm state
reached later CuBLAS operations without `CUBLAS_WORKSPACE_CONFIG`, and the local gsplat extension
required `CXXABI_1.3.15` missing from the host `libstdc++`. The repository's authoritative
CPU-isolated command then passed completely:

```bash
CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh
```

That run passed Ruff, the complete non-slow CPU test path, and `docs_sync`. The failed visible-CUDA
run is not hidden and provides no GPU correctness or performance evidence.
