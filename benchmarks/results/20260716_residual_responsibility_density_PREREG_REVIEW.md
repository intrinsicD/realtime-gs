# Residual-responsibility density preregistration executability review

Verdict: PASS

Unresolved findings: none

## Independent re-review — 2026-07-16T03:10:04+02:00

I freshly reviewed the complete append-only amendment, the preserved initial FAIL review, and the
current CPU renderer, `RenderOutput`, classic `DensityController`, parameter/Adam surgery,
`Trainer`, and metric seams. The amended preregistration has the requested SHA-256
`f65b4afecc09532dd2113c353043afebc8607e11cbdc714cee326ceec8e3e368`.

This remains an outcome-free executability review. I did not implement a seam, run a toy or
official pilot, prepare seeds `6,7,8`, probe their schedules or generators, fit, lift, render,
score, select, train, seal, create a marker, inspect an outcome, or edit the preregistration,
source, tests, documentation, or ARA. No experiment harness, implementation review, seal, marker,
or official result exists at re-review time.

Current source bindings are unchanged from the initial review where applicable:

- `src/rtgs/render/base.py`:
  `1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9`
- `src/rtgs/render/torch_ref.py`:
  `61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e`
- `src/rtgs/render/__init__.py`:
  `5a19ccf30c4a571ffac7287a7166e42d6e30ece721109448eb05b0256b5ab876`
- `src/rtgs/optim/density.py`:
  `d56d650eaf0cb758b53111a158f3721b8be69b31292a1785e3f33430e686d375`
- `src/rtgs/optim/trainer.py`:
  `3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f`
- `src/rtgs/optim/__init__.py`:
  `1196f76c9386d808b88a0940f562b29a85b3598e182ba7e997ebf2f769e4d53a`
- `src/rtgs/core/metrics.py`:
  `d489c07c65ac4c74f0f927d41c62b887724cf3216f2ef28a116ff169d08272d4`

### Resolution of the seven initial blockers

1. **Density scheduling and infinity serialization: resolved.** The arm now contains a literal
   `DensityConfig(start_iter=40, stop_iter=120, every=40, ...)`, including all established fields,
   `max_gaussians=N0+96`, no pruning/reset, and the fixed split factor. Under the current inclusive
   controller predicate
   `start_iter <= iteration <= stop_iter and iteration % every == 0`, the only surgery steps are
   40, 80, and 120; step 160 is excluded. Runtime positive infinity is explicitly normalized to
   the standard-JSON string `"positive_infinity"`, and non-standard JSON `Infinity` is forbidden.

2. **160-step Phase-A horizon and exact hook boundary: resolved.** Phase A retains the complete
   `TrainConfig(iterations=160, ...)`, so the means multiplier remains
   `0.01**(1/160)` and the SH interval remains 40. The new opt-in stop is uniquely placed after
   step-40 backward/statistics, all six Adam updates in insertion order, established history, and
   means-LR decay, but before density surgery, native evaluation, or callback. That is exactly the
   current seam between Trainer's LR update and `DensityController.step`. Phase A returns a
   detached, non-resumable snapshot; every Phase-B arm instead uses one persistent 160-update
   Trainer invocation and must match the audited prefix at its first pre-surgery hook.

3. **Authoritative score arithmetic: resolved.** `gradient_topB` is now selected only from the
   current controller's native-float32 path: native float32 screen-gradient norm, float32
   `index_add_` accumulator and count, and float32 division. Coarse-visible rows increment the
   count exactly as current `DensityController.accumulate` does. The separately reported float64
   gradient reconstruction is audit-only and cannot select or tie-break. Residual/support
   operands are independently cast to float64 before multiplication, concatenated in increasing
   flattened pixel intervals, and reduced by the literal formulas
   `(e64[:,None]*w64).sum(dim=0,dtype=torch.float64)` and
   `w64.sum(dim=0,dtype=torch.float64)`. The same operands/order govern the assigned-residual gate;
   max and selection ties are also fixed.

4. **Diagnostic reconstruction schema, including empty visibility: resolved.** The additive
   default is exactly `None`. Diagnostics-on now binds detached contribution chunks, literal final
   background weights, sorted global indices, the exact activated visible colors and camera-space
   depths consumed by compositing, and half-open row-major pixel intervals. The `V=0` contract is
   explicit: normal row chunks, `(P,0)` weights, background one, empty index/color/depth tensors,
   and the unchanged normal background render. Immediate contraction/clearing is required. Native
   reconstruction, a slow explicit float64 compositor, and exact off/on loss/gradient parity are
   mandatory Phase-A invariants; diagnostic tensors cannot enter renderer output arithmetic or
   the loss.

5. **Persistent identities and split-noise assignment: resolved.** Survivors retain physical row
   order and IDs. Selected parents are materialized in ascending persistent-ID order, append blocks
   are clone, split-child-0, then split-child-1, and monotonically increasing birth IDs follow that
   exact physical order. Each wave resets its isolated CPU split generator to the frozen seed,
   draws child-0 then child-1 `(16,3)` blocks, and assigns row `k` to the same ordered parent.
   Shuffle, view, and split generators are disjoint. The row/ID transformation mirrors parameter
   surgery; survivor Adam moments remain bitwise intact, newborn moments are zero, and scalar
   optimizer state and param-group fields remain unchanged.

6. **Held-out arithmetic and pre-/post-surgery step 40 routing: resolved.** Primary checkpoint
   RGB error casts the clamped native-float32 prediction and target to float64 before subtraction
   and pools named SSE/counts. Crop SSIM remains the exact repository 11x11 call on the common
   truth-support masked crop. Depth, IoU, coverage, paired seed vectors, `statistics.fmean`
   reductions, signed directions, relative-versus-absolute regressions, and inclusive guards are
   all literal. The common pre-surgery step-40 metric can be created only after Phase-B
   authorization, after all three selections are frozen, from one detached first-arm snapshot; it
   returns nothing to policy, is hash-bound, and is reused by later arms. Each arm's AUC still uses
   its own post-surgery step-40 value. Phase A never constructs held-out truth or renders a
   held-out metric.

7. **Official routing and Phase-A-to-B authorization: resolved.** The fixed harness, seal, two
   marker paths, result namespaces, literal verification sequence, and only three official CLI
   forms are now specified. Resolved-path and basename validation rejects copied seals,
   latest/glob lookup, alternate modes, and ambiguous outputs. The independent Phase-A audit has a
   sole derived path, strict standard-JSON artifact type, exact PASS/FAIL field, empty-findings
   requirement, provenance, complete bindings, and recomputed evidence. Before creating a Phase-B
   marker, the harness must derive the one authorized Phase-A result from the fixed marker,
   rehash every file and canonical payload, validate all bindings, and independently recompute
   every gate. Any mismatch refuses before marker creation; later drift consumes the marked
   attempt.

### Fresh source-to-protocol executability checks

- Current Trainer order is zero-grad, sampled view, one training render, loss, backward, classic
  accumulation, the six Adam optimizers in parameter insertion order, history, means-LR decay,
  classic surgery, then native evaluation/callback. The amended research hook and observer order
  are compatible with that sequence and do not require a restart.
- `active_degree=min(target_sh_degree,it//40)` gives degree 0 through completed step 40, degree 1
  for steps 41-80, degree 2 for 81-120, and degree 3 for 121-160. Surgery therefore occurs after
  the last update of each completed score window and before the normal post-surgery evaluation.
- Current classic accumulation uses coarse `visible` rows, native screen-center gradients,
  `max(width,height)/2`, float32 `index_add_`, and a clamped positive count denominator, exactly
  matching the authoritative comparator. The amendment does not silently substitute its float64
  audit reduction for selection.
- Current `_edit_params` preserves surviving Adam rows and scalar state and appends zero moment
  rows. Current clone, split offset/rotation, `/1.6` scale, revised-opacity, and survivor ordering
  arithmetic can be retained behind the fixed-selection seam. The amendment uniquely adds the
  ordered selected-parent materialization and external ID map that the default boolean-mask path
  lacks.
- Four strata with eight selected parents each imply exactly 16 small clones and 16 large splits:
  net `+32` per wave and `N0+96` after wave three. The explicit insufficient-stratum failure makes
  that trajectory fail-closed rather than permitting quota borrowing or an outcome-dependent
  repair.
- A non-`None` fixed-view seam replaces only Trainer's per-step `randint`; the default/`None` path
  retains the current shared-generator behavior. The experimental fixed policy uses isolated,
  reset split generators, so its parent choices cannot alter the training-view stream.
- The renderer's current hard kernel, alpha clamp, epsilon-bearing exclusive transmittance, sorted
  visible indices, accumulated alpha/depth, and literal last background factor match the frozen
  diagnostic equations. The specified additive field and `V=0` behavior are implementable without
  changing default `RenderOutput` or renderer arithmetic.
- Physical removal of held-out views before fit/lift/optimization, Phase-B-only truth
  construction, a detached no-return observer, pre-observer selection hashes, and post-surgery
  callbacks provide a complete isolation boundary. A no-op at Trainer's ordinary step-20 callback
  is sufficient to preserve the frozen held-out checkpoint list beginning at step 40.
- Phase A cannot authorize Phase B through Markdown, a user-supplied passing-looking file, or a
  glob. The audit JSON is necessary but not sufficient: Phase B must redo the raw-evidence gates
  itself before its once-only marker can exist.

### Scientific-drift check

The amendment is limited to the seven execution repairs requested by the initial review. It does
not alter seeds, scene, train/held-out split, fit, Carve initialization, the three arms, native
residual or gradient score, max-over-window rule, four strata, eight-per-stratum quota, three
waves, count trajectory, optimizer or learning rates, training loss, checkpoints, primary/safety
thresholds, comparator logic, interpretation, claim boundary, or stopping rules. It introduces no
pilot-derived choice and exposes no outcome. The chronology assertion is consistent with the
absence of an implementation harness, seal, marker, or result.

This PASS clears the amended preregistration for implementation review only. It does not certify a
future implementation or authorize sealing. The implemented harness, default-preserving seams,
focused tests, raw-evidence sufficiency, and exact machine gate still require the separately named
independent implementation review with `Verdict: PASS` and `Unresolved findings: none`.

---

## Preserved initial review chronology

Initial verdict at review SHA-256
`666e61b4dc8a41aabd68fa96710f30832236513e15b3eaa55d348d95ed04812d`: **FAIL**

## Scope and chronology

This is an independent, outcome-free review of the complete preregistration and the current CPU
renderer, `RenderOutput`, classic `DensityController`, optimizer surgery, and `Trainer` seams. I
did not implement a seam, run a diagnostic or pilot, prepare seeds `6,7,8`, generate an official
view or surgery stream, fit, lift, train, seal, create a marker, inspect an outcome, or edit any
source, test, documentation, preregistration, or ARA file.

Reviewed snapshot:

- `benchmarks/results/20260716_residual_responsibility_density_PREREG.md`:
  `626a35cf935fba198833567af10fa485c680837a48e67101ed206132647ec60f`
- `src/rtgs/render/base.py`:
  `1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9`
- `src/rtgs/render/torch_ref.py`:
  `61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e`
- `src/rtgs/render/__init__.py`:
  `5a19ccf30c4a571ffac7287a7166e42d6e30ece721109448eb05b0256b5ab876`
- `src/rtgs/optim/density.py`:
  `d56d650eaf0cb758b53111a158f3721b8be69b31292a1785e3f33430e686d375`
- `src/rtgs/optim/trainer.py`:
  `3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f`
- `src/rtgs/optim/__init__.py`:
  `1196f76c9386d808b88a0940f562b29a85b3598e182ba7e997ebf2f769e4d53a`

No experiment harness, seal, Phase-A marker, Phase-B marker, or matching official result existed
at review time.

## Unresolved blocking findings

### 1. The literal `TrainConfig` schedules density at 80/120/160, not 40/80/120

The frozen `TrainConfig` omits `density=DensityConfig(...)`. The current defaults are
`start_iter=60`, `stop_iter=10_000_000`, and `every=40`; the current inclusive scheduling test is
`start_iter <= iteration <= stop_iter and iteration % every == 0`. The resulting classic waves
are therefore 80, 120, and 160. This directly conflicts with the frozen score windows, surgeries,
post-surgery step-40 checkpoint, and topology trajectory. The later policy paragraph sets only
four density fields and does not repair the schedule. It also freezes `inf` without defining a
standard-JSON representation.

Exact outcome-neutral amendment text:

> Replace the arm configuration's implicit density default with
> `density=DensityConfig(start_iter=40, stop_iter=120, every=40,
> grad_threshold=2e-4, absgrad=False, split_scale_frac=0.01, split_factor=1.6,
> prune_opacity=-1.0, prune_scale_frac=float("inf"), max_gaussians=N0+96,
> opacity_reset_every=0, opacity_reset_value=0.011, revised_opacity=True,
> mcmc_noise_lr=500000.0)`. The opt-in experiment policy bypasses gradient-threshold,
> significance-budget, and prune selection and supplies the two exact selected-parent masks; it
> does not bypass the 40/80/120 schedule or the established field arithmetic. No density action
> occurs at step 160. Runtime positive infinity is serialized in standard JSON as the literal
> string `"positive_infinity"` in a normalized configuration record; non-standard `Infinity` is
> forbidden.

### 2. Phase A has no executable 160-horizon prefix/stop seam or exact pre-surgery boundary

Phase A must stop after 40 updates while Phase B must reproduce the prefix of a 160-update run.
Calling the current Trainer with `iterations=40` is not equivalent: means learning-rate decay is
computed from `iterations`, so it would use `0.01**(1/40)` rather than `0.01**(1/160)`. The current
Trainer also has no stop-before-density seam. Its actual order is backward, classic score
accumulation, Adam, history, means-LR decay, density surgery, and only then evaluation/callback.
The preregistration does not identify the precise Phase-A snapshot boundary, the hook that owns
residual/support accumulation, or how Phase B reaches that boundary without restarting an arm.

Exact outcome-neutral amendment text:

> Add an opt-in `stop_before_density_after_step: int | None = None` Trainer seam and an opt-in
> fixed-density-policy seam, both with omitted/`None` paths proven bit-exact to the current Trainer.
> Phase A retains the complete 160-step `TrainConfig`, its `0.01**(1/160)` means decay, and its
> 40-step SH interval, but stops at the step-40 pre-surgery hook. At every step the exact order is:
> zero gradients; consume the fixed view; render once; form residual evidence from that render;
> form the scalar loss; backward; accumulate `G`, `R`, `S`, visibility, and argmax evidence from
> the pre-Adam row identities; run the six Adam optimizers in current insertion order; append
> history; decay the means LR; then, only at 40/80/120, read post-Adam scale/opacity, construct
> strata and selections, and perform surgery. The Phase-A snapshot is after means-LR decay and
> before density surgery or native evaluation. Its optimizer binding includes parameter order,
> each optimizer's scalar `step`, `exp_avg`, `exp_avg_sq`, param-group LR/name/eps, and hashes.
> Each Phase-B arm uses one persistent 160-update Trainer invocation and may not restart around a
> surgery. Immediately at its step-40 pre-surgery hook it must match the audited Phase-A snapshot
> before the hook is permitted to mutate topology.

### 3. The authoritative gradient comparator and float64 score arithmetic conflict

The preregistration calls `G_i` the current controller's literal score but also says all decision
reductions are float64. The current literal controller computes the norm, `index_add_` accumulator,
count, and division in native float32. A float64 accumulation of detached per-step norms can rank
near-tied parents differently and is not the literal current comparator. For `R` and `S`, the text
also does not state whether operands are multiplied before or after casting, nor whether chunk
sums or one concatenated row-major reduction are authoritative. These choices affect selections
and Phase-A gates.

Exact outcome-neutral amendment text:

> The authoritative `gradient_topB` value is the current native-float32 classic-controller value:
> compute each native float32 `g_ti`, accumulate with the current float32 `index_add_`, increment
> the current coarse-visible count, and divide in float32 exactly as `DensityController` does.
> Serialize an independent float64 recomputation only as audit evidence; it cannot select or break
> a tie. For residual/support decisions, concatenate detached diagnostic chunks in increasing
> flattened pixel interval, cast `e` and `w` separately to float64, and compute literally
> `r=(e64[:,None]*w64).sum(dim=0,dtype=torch.float64)` and
> `s=w64.sum(dim=0,dtype=torch.float64)`. Accumulate the pooled assigned-residual numerator and
> denominator from those same float64 operands. `R` and `S` are maxima of these float64 per-step
> values; exact max ties retain the earliest step, then the lower physical training-view index for
> diagnostic argmax fields. Selection ties always use ascending persistent ID. No rounded or
> independently regrouped reduction is authoritative.

### 4. Diagnostic evidence is undefined for empty visibility and does not bind reconstruction inputs

The proposed diagnostics list `w`, background weight, sorted indices, and pixel bounds, while the
mandatory reconstruction additionally needs the exact activated visible colors and camera-space
depths consumed by the render. The text does not say whether these are captured or recomputed.
The current renderer returns early when no primitive is coarse-visible, so an implementation can
legitimately return `None`, empty chunks, or full background chunks; only one of those permits the
required every-render reconstruction and hashing. Chunk interval convention and the literal
background formula also need a single schema.

Exact outcome-neutral amendment text:

> `RenderOutput.compositing_diagnostics` is an additive optional field with default `None`.
> Diagnostics-on returns detached finite `w_chunks`, `background_weight_chunks`, sorted
> `gaussian_indices`, exact activated `visible_colors`, exact camera-space `visible_depths`, and
> half-open flattened row-major `pixel_intervals`. For a row chunk `[r0,r1)`, its interval is
> `[r0*W,r1*W)`, `w` has shape `((r1-r0)*W,V)`, and background weight has shape
> `((r1-r0)*W,)`. For `V>0`, background weight is literally
> `exclusive_transmittance[:,-1] * (1-alpha[:,-1])`; it is not `1-sum(w)`. For `V=0`, return the
> normal row chunks with `w.shape=(P,0)`, background weights exactly one, and empty index/color/depth
> tensors. Diagnostics tensors are detached at capture, are never used to form renderer outputs or
> loss, and all chunk lists are cleared and the output field set back to `None` immediately after
> contraction. Existing output values, fields, graph topology, gradients, and `means2d.grad` must
> remain bit-exact when collection is off and in the required off/on parity checks.

### 5. Persistent-ID birth order and split-noise-to-parent assignment are ambiguous

The current surgery filters survivors in current row order and appends clone rows, split child
block 1, then split child block 2. Its boolean indexing also associates random tensor row `k` with
selected physical row `k`. The preregistration separately orders birth IDs by operation block,
canonical parent ID, and child ordinal, but it never states whether canonical ID order or current
physical row order controls appended tensors and the two `(16,3)` normal arrays. Those orders can
diverge after the first surgery. Different legitimate implementations would produce different
children, IDs, later strata, and optimizer-state hashes from the same frozen stream.

Exact outcome-neutral amendment text:

> Survivors retain current physical row order and persistent IDs. Within each operation, selected
> parents are materialized in ascending persistent-ID order. Append blocks in this exact order:
> clone children, split children with child ordinal 0, split children with child ordinal 1. Assign
> monotonically increasing birth IDs in that physical append order; the ordering key is
> `(wave, block_code, parent_persistent_id)`, with block codes clone=0, split-child-0=1, and
> split-child-1=2. Reset the split generator immediately before the wave, draw child-0
> `torch.randn((16,3),...)` first and child-1 second, and associate row `k` in each draw with the
> `k`th ascending-persistent-ID split parent. The row-to-ID map is edited by the identical
> survivor/block concatenation. For every parameter tensor, survivor `exp_avg`/`exp_avg_sq` rows
> are bitwise preserved, all appended moment rows are exact zero, and the optimizer scalar step
> and param-group fields remain current. A default/`None` selected-mask policy must call the
> existing threshold controller path without constructing IDs or changing RNG consumption.

### 6. The held-out safety gates and the common pre-surgery step-40 metric are not uniquely computable

The primary pooled foreground PSNR/AUC is executable. The safety gates are not: foreground crop
bounds/masking, crop-SSIM aggregation, depth aggregation, alpha-IoU aggregation, signed regression,
and across-seed reduction order are unspecified. “Mean normalized depth-RMSE relative regression”
can mean a ratio of means or a mean of paired ratios. “Mean alpha-IoU regression” can be absolute
or relative. These distinctions can flip inclusive gates. In addition, the current Trainer's
ordinary step-40 callback occurs after surgery, whereas the result must also contain one common
pre-surgery step-40 held-out metric; Phase A is forbidden from rendering held-out metrics.

Exact outcome-neutral amendment text:

> All held-out metric operands are detached float32 render fields cast to float64 before
> subtraction or multiplication, and raw per-view numerators/counts are serialized. The final
> metric is step 160. For each view, define crop bounds from the fixed truth support using
> `rtgs.core.metrics.masked_crop(..., margin_fraction=0.05)`; apply the same truth-derived bounds
> and binary support mask to clamped prediction and target, then compute repository 11x11 SSIM.
> A seed's crop SSIM is the arithmetic mean of its three per-view values. A seed's normalized
> depth RMSE is `sqrt(sum_v depth_SSE_v / sum_v depth_count_v) / extent`. A seed's alpha IoU is
> `sum_v intersection_v / sum_v union_v`. For each comparator form literal seed-ordered paired
> vectors `error-comparator` for PSNR and SSIM, and `comparator-error` for alpha IoU; every reported
> mean is `statistics.fmean` of that paired vector. Depth relative regression is exactly
> `(fmean(error_depth)-fmean(comparator_depth))/fmean(comparator_depth)` with a finite positive
> comparator denominator. The inclusive thresholds apply to those values separately for both
> comparators. At the first Phase-B arm's step-40 pre-surgery hook, render the detached common
> snapshot once and bind its parameter hash; later arms must match that hash and reuse the bound
> metric without another pre-surgery held-out render. Freeze and hash all three selections before
> that evaluator receives the detached snapshot; it returns no value to the policy. Each arm's
> step-40 post-surgery metric is rendered only after surgery. No held-out quantity is available to
> the density policy or Phase A.

### 7. Seal/CLI routing and Phase-A-audit authorization lack a sole machine-readable path

The namespace gives file patterns but no exact subcommands, argument schema, fixed implementation
review, full-verification command sequence, independent-audit artifact type/path/verdict schema, or
rule for selecting the one Phase-A result when Phase B starts. A Markdown audit alone cannot be
recomputed as a strict machine authorization, and “bind the independent audit” does not say which
fields must match. A `latest`/glob choice or manually supplied passing-looking file would satisfy
the current prose. This is blocking before either marker can be safely routed.

Exact outcome-neutral amendment text:

> Before sealing, an independent implementation review at
> `benchmarks/results/20260716_residual_responsibility_density_IMPLEMENTATION_REVIEW.md` must contain
> an explicit `Verdict: PASS` and `Unresolved findings: none`; it is part of the sealed manifest.
> Seal creation hashes the complete manifest before and after, rejects drift, and runs in order:
> `.venv/bin/python -m ruff check .`; `.venv/bin/python -m ruff format --check .`;
> `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q -m
> "not slow"`; `.venv/bin/python scripts/docs_sync.py`; and `git diff --check`. The only official
> entry points are `.venv/bin/python benchmarks/residual_responsibility_density_ablation.py seal
> --output <fixed-seal>`, `... phase-a --seal <fixed-seal> --output <validated-phase-a-path>`, and
> `... phase-b --seal <fixed-seal> --phase-a-result <exact-path> --phase-a-audit <exact-path>
> --output <validated-phase-b-path>`. Equivalent seals copied to another path and implicit
> latest/glob discovery are rejected. Each marker payload binds its exact requested output path and
> is rehashed before decisions and serialization.
>
> The independent Phase-A audit writes standard JSON beside the Phase-A result as
> `<phase-a-result-stem>_AUDIT.json`, artifact type
> `residual_responsibility_density_phase_a_results_audit`, with `verdict` exactly `PASS` or `FAIL`,
> hashes of preregistration, fixed seal, Phase-A marker and payload, Phase-A result and payload,
> auditor identity/provenance, recomputed per-seed invariants/gates, and `unresolved_findings`.
> Phase B accepts only `PASS` with an empty unresolved list. Before creating the Phase-B marker it
> obtains the sole Phase-A result path from the fixed Phase-A marker payload, requires the CLI path
> to equal it, rehashes all four bound artifacts, recomputes every Phase-A gate from raw result
> evidence, requires equality with both result and audit decisions, and records those exact hashes
> in the Phase-B marker. Any authorization mismatch refuses before creating a Phase-B marker and
> cannot be bypassed; any failure after marker creation consumes the attempt.

## Executable aspects that do not clear the findings

- The stated contribution formula matches the current CPU renderer's hard kernel, clamped alpha,
  exclusive transmittance, sorted global `visible` indices, color/alpha/depth accumulation, and
  literal final background factor. It correctly excludes opacity-only, normalized, binary, and
  re-rendered approximations.
- Backward-time gradient accumulation followed by Adam and then current classic surgery is
  compatible with the proposed score timing once the exact pre-surgery hook is frozen.
- Current `_edit_params` can preserve survivor Adam moments and append zero newborn moments, and
  current split scale/offset/revised-opacity arithmetic matches the prose. A selected-index seam
  and external persistent-ID map are feasible without changing the default path.
- The four fixed strata imply exactly 16 clones and 16 splits when every stratum has at least eight
  eligible parents. Treating insufficiency as a permanent Phase-A failure is a valid no-pilot,
  fail-closed feasibility design; no quota borrowing is needed.
- The view, shuffle, and split seed namespaces are disjoint and can be implemented with isolated
  CPU generators. The remaining RNG gap is only the unresolved normal-row-to-parent ordering.
- Physical removal of held-out views before fit/lift/training, plus a detached evaluator, provides
  a workable held-out isolation boundary. The seven-point trapezoidal AUC denominator of 120 is
  consistent with checkpoints 40 through 160.
- Seeds, CPU-only backend, thread/environment constraints, no-prune fixed growth, claim boundary,
  and stopping rules are explicit. No GPU, real-scene, default, unrestricted-density, or SOTA claim
  is authorized.

The preregistration needs an append-only, outcome-neutral executability amendment and a fresh
independent review before implementation or sealing. None of these findings requires a pilot or a
scientific threshold, seed, score, quota, or arm change.
