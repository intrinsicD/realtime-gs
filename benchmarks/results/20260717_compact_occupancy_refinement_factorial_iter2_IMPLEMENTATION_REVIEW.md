# Compact occupancy-point refinement factorial iter2 — implementation review — 2026-07-17

Verdict: PASS

Reviewed source aggregate SHA-256: fdcd116e5d6f9b9d58f4963a6790ac79135c2f13813852e8cf4eefd1e92ca4d4

## Scope and disposition

This is an independent adversarial **pre-seal implementation review**, not an experiment result.
It covers the endpoint-safe iter2 sampler repair, the fresh factorial harness, compact trainer,
focused tests, exact compact inputs, once-only lifecycle, and failure behavior. It authorizes only
publication of the iter2 seal while the reviewed aggregate, inputs, runtime, preregistration, and
empty namespace still match. It does not authorize a scientific claim, density follow-up,
default change, scaling claim, source-RGB-equivalence claim, or viewer result.

Three blockers were found before this final snapshot and repaired before the aggregate above was
computed:

1. bank shape, sample-finiteness, and color-finiteness failures initially fell through to generic
   protocol errors rather than retaining the preregistered structured bank context;
2. the new forced endpoint test initially used seed `991001`, outside the frozen iter2 test-only
   seed list; and
3. the runtime record initially named and hashed the intended preload library without proving it
   was the effective `LD_PRELOAD`.

The reviewed source now gives every pre-archive sample/shape/finiteness and post-query color
failure a finite `BankInvariantError` receipt; uses frozen test seed `991601`; and requires and
records the exact effective preload. Focused negative tests cover wrong shape, non-finite sample
density, non-finite color, bounded terminal retention, and unavailable scientific decision.
There is no remaining pre-seal blocker in the reviewed snapshot.

## Endpoint repair and scientific-contract check

The failed attempt's executed-source tar was compared directly with the current sampler. The only
sampler-source change is the addition of `_half_open_uniform_xy` and replacement of the two
per-axis affine assignments by a call to that helper. The helper performs no random operation. It
maps the already-drawn unit tensor through the same field-dtype affine transform and clamps only
to the lower endpoint and `nextafter(upper, lower)`. Selection, unit-coordinate, multinomial,
normal, and acceptance draws retain their prior order and counts. There is no resampling,
dropping, duplication, active-count normalization, or null reinterpretation. Uniform target
density remains `1/(w*h)`, mixed active density remains the same `q`, and the fixed-attempt
denominator remains unchanged.

Forced tests cover float32 and float64, zero-origin and translated native-scale windows, and both
axes at the largest representable value below one. The mixed `eta=0.25` test proves active/direct
uniform attempts remain strictly half-open with finite proposal density and importance bounded by
four. A separate million-draw test-only comparison found the repaired and consumed affine maps
bit-exact on every non-endpoint draw observed; constructed endpoint cases alone are clamped.

The original and iter2 factorial constants are unchanged for arms, checkpoints, attempts,
uniform fraction, extent, timeout, optimizer/trainer call, metric equations, and decision gates.
An AST comparison found the complete `CompactTrainConfig(...)` call in the consumed harness and
the iter2 `_frozen_config` call exactly equal. Iter2 changes the seed domains and protocol repair
surface, not the scientific contrast or threshold.

## Isolation, lifecycle, and evidence bindings

| Boundary | Independent check | Disposition |
| --- | --- | --- |
| Consumed attempt | Failure audit SHA-256 is `67bf419e696273a7b47d729b7e0c07f5afb468e297568bfc694e6ddec5c0ccc7`; all seven failed-attempt provenance artifacts match the frozen hashes. | Preserved; no first-attempt bank or outcome is reusable. |
| Iter2 preregistration | SHA-256 is `da4ef58a620c687e6eccfae959113c7e1bf7f25242f2d2f4a05b885c26047278`; it predates this review and freezes the unchanged science plus repair. | Bound. |
| Seeds | Official train `76601..76603`, official evaluation `76701..76703`, focused train `991601..991603`, and focused evaluation `991701..991702` are pairwise disjoint from the consumed `764xx/765xx` and excluded `76201` domains. Focused commands set the seed-firewall environment. | Fresh and separated. |
| Official namespace | Before review publication only the preregistration existed; afterward only the preregistration and this review existed. Seal, attempt, result, run directory, banks, and workers were absent throughout. | Clean for one seal; no official iter2 seed was evaluated. |
| Bank creation | A new iter2 run directory is created exclusively after the attempt token. Every new evaluation archive is generated under the new path; all three and the manifest must finish before the first of twelve bounded fresh workers. No old partial-bank path is referenced. | No partial-bank reuse or worker-before-bank path. |
| Bank integrity/failure | Every archived tensor has a descriptor/hash and the NPZ has file and semantic hashes. Structured failures retain seed/domain/view/kind/generator, first row/coordinate, fitted window, predicate counts, shape mismatches, and tensor hashes. Terminal failures set both decision fields unavailable and `promotion_authorized=false`. | Fail-closed. |
| Source closure | 23 reviewed files are in the aggregate above. Live repository-local imports have no unbound source and all loaded `rtgs` module origins resolve to the bound workspace files. The seal rechecks source/input/runtime/config before and after focused verification and immediately before exclusive publication. | Exact dirty-tree source is bindable. |
| Inputs | Teacher aggregate `56a02fbdf3f4f2d61d9358f486c90f6c963449c0642533859395b0c6e2f21db7`; proxy aggregate `73e070fdfab42147501f94561a47681f79d26b7ff98450e31d4bf0a8d6084176`; initialization PLY `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`. Strict alignment covers ordered cameras, dtype, canvas/window, blend/support semantics, `m_init`, variable `m_opt,i`, geometry, filter variance, and scalar range before exact amplitude multiplication. | Bound and aligned. |
| Runtime | Effective preload is exactly `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33`, SHA-256 `1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11`. Runtime check observed PyTorch `2.9.0+cu128`, CUDA `12.8`, NVIDIA GeForce RTX 3050 capability 8.6, UUID `4cee065a-aab7-97b8-40be-55d516d2a53c`, and driver `590.48.01`. | Exact effective runtime is seal-checkable. |

## Trainer, metrics, accounting, and RGB boundary

- All four arms use the same 835-Gaussian initialization, 140 steps, 128 attempts per step,
  point/gaussian chunks 256, outer microbatch 128, query chunk 640, degree-zero hard SH, hard EWA
  support, explicit extent, six separately clocked Adam groups, and disabled built-in checkpoint
  risk evaluation. Callback snapshots at `0,35,70,140` must match trainer semantic hashes and
  retain null built-in evaluations.
- A/C and B/D must match all 140 sampling fields and differ in target-density and importance
  hashes at all 140 steps. All four arms must have exact step-zero snapshot and fixed-bank metric
  equality. Any mismatch makes the scientific decision unavailable rather than negative.
- Evaluation uses immutable per-seed banks, float64 loss accumulation, equal-view `J_U` and
  fixed-attempt active-submeasure `J_Q`, the frozen log-AUC abscissae, and exactly the
  preregistered D/B gates. Secondary contrasts cannot authorize promotion.
- Receipts retain `m_init,i^2D`, variable `m_opt,i^2D`, their sum, `N_init^3D`, every worker's
  fixed `N_opt^3D`, bank/training attempt counts, teacher and proposal tile-overlap preflights,
  proposal normalizers, index diagnostics, parameter motion, memory, and explicit
  `scaling_claim_authorized=false`.
- Parent and every worker enter an import-and-filesystem RGB denial boundary before loading the
  strict compact bundles. Negative controls exercise `builtins.open`, `io.open`/path access, and
  forbidden image/calibrated imports. Proposal colors are never queried. Source RGB is permitted
  only after an immutable passing result for diagnostic viewer/reference use.
- Visualization is explicitly `DEFERRED_POST_RESULT`. No pre-result viewer, source reference,
  gsplat rendering, or contact sheet participates in fitting, selection, metrics, or decision.

## Checks actually executed

- Read `CLAUDE.md`, the experiment skill, and the results-audit skill in full; read the failure
  audit, iter2 preregistration, harness, sampler, compact trainer, and all three focused test files.
- Ran the exact focused pytest set under the frozen preload and
  `RTGS_FACTORIAL_FOCUSED_TEST=1`: **81 passed**.
- Ran Ruff check and format-check over the eight frozen harness/sampler/trainer/test/init files:
  both passed.
- Ran the exact real compact teacher/proxy bank path with test-only evaluation seed `991702` in a
  temporary directory under the RGB guard. All seven native fitted windows had 4096/4096 active,
  direct, inside, strictly half-open uniform draws. The archive hash round-trip passed; semantic
  SHA-256 was `107fad8f5f42a05c4c061a6ba591b8e5e7592405b38e4e42ec6eb63d3bb497dd`;
  proposal active fraction minimum was `0.99365234375`, max/min was
  `1.0034398034398035`, and source-RGB/import attempts were zero. The temporary archive was
  deleted.
- Queried only runtime identity on the bound CUDA device; no training, optimization, timing
  benchmark, or quality metric was run.
- Rechecked the empty iter2 namespace and the reviewed aggregate after the repairs.

No seal, attempt, result, run directory, official bank, official worker, official training seed,
or official evaluation seed was created or invoked during this review. GPU timing and scalability
remain non-decisional. A passing official result would still require an independent post-result
audit before any follow-up authorization or quantitative claim, followed by the preregistered
native-resolution gsplat/contact-sheet/viewer handoff.
