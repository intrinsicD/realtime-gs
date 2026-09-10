# Prospective Design Review

- Task ID: `20260909_field_target_gradient_stage_frame00008`
- Protocol SHA-256: `2d150044ba267a15a4df5d69780e8266a72251300db6f3a8ae5854fcc4118fb9`
- Reviewer: `Codex-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`
- Scope: `design only; no init-run or protected execution authorization`
- Reviewed task file SHA-256: `2912fa35687116505153c89e4d48c17a62c7ef4d5cb130fe643f916d36ecb1f6`
- Review date: `2026-09-09`

## Decision and boundary

Approve implementation of the corrected prospective design. The task-specific driver and
canonical run directory are absent at this review. This approval is methodological only:
the completed implementation, tests and exact live source binding require a separate final
protocol review before `init-run`. This record must not be substituted for that final review.

The design describes training-target residuals and first-order gradients of a fixed L1/SSIM
objective at nine predeclared saved states. Within each state, both targets have identical
parameters, cameras, topology and renderer settings. This supports a description of how the
target changes that local objective gradient. It does not establish the cause of the previous
halos, an optimizer update, a training trajectory, reconstruction improvement, physical density,
or high-quality field-only recovery.

The reviewer previously audited the completed source experiment. `Outcome Access: none` here
means no outcomes of this new residual/gradient diagnostic were generated or inspected. Prior
capture and source-experiment exposure is expressly part of the development claim boundary.

## Independently inspected evidence

- Read the complete draft and corrected protocol, experiment lifecycle, relevant experiment
  and results-audit instructions, and the existing Gaussians3D, GsplatRasterizer, Trainer and
  SSIM source APIs. No new result-bearing driver was present or executed.
- Independently recomputed the corrected protocol digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260909_field_target_gradient_stage_frame00008.json`.
- Rehashed all 76 unique listed cache/state files and checked their byte counts: 244,219,355
  bytes total, with every SHA-256 matching. Read the camera metadata only and verified that
  its 22 view IDs exactly equal the fixed training split. Did not decode target/model arrays,
  compute residuals or gradients, render capture views, inspect heldout sources or execute
  a protected phase.
- Confirmed the new driver and new canonical run directory still do not exist after the
  amendments. Source implementation and final source-envelope equality are therefore pending.

## Methodological checks

The input boundary is coherent: 44 cached RGB/field training targets, 22 cached masks, nine
saved models and one camera metadata file form the exact 76-file allowlist. The inherited
raw seal is administrative byte provenance; its validation may hash heldout bytes but does
not decode or score them. The result-bearing worker must reject all raw-dataset opens and
all other source-run reads. Cached masks define diagnostic strata only and cannot mask the
loss or its gradients. Entry/exit hashes and guarded read records must be retained.

The three mask regions are disjoint and exhaustive under the specified square-radius-three
morphology and outside-image background. Per-channel bias divides by region pixel count;
RGB scalar bias/MAE/MSE divide by three times that count; full-canvas contribution and error
share have distinct denominators. Empty regions and zero-total-error shares have explicit null
semantics. Reflect-padded box filtering is applied to complete images before region scoring.
Lowpass/highpass summaries are correctly described as spatial-scale summaries rather than
an orthogonal decomposition of error energy.

The fixed loss matches the prior Trainer's unmasked objective: 0.8 times full-canvas RGB L1
plus 0.2 times one minus the existing tiled 11x11 sigma-1.5 SSIM. The renderer returns unclamped
RGB; no output clamp, mask, regularizer or extra loss may be introduced. Separate component
gradients are unweighted, and their weighted sum is the total gradient.

Existing APIs support leaves for means, saved raw quaternions, log scales, effective opacity,
SH0 and SHN. The current Trainer quaternion entry policy retains the saved values; renderer
normalization is part of the differentiated model. Initial opacity follows the documented
CUDA clamp/logit/sigmoid entry convention. Final opacity is rendered verbatim and must not
undergo the default initialization clamp or a logit roundtrip. Multiplying the effective-opacity
gradient by alpha times one minus alpha gives the stated local sigmoid-coordinate gradient;
endpoint zero/one counts and zero mapped derivative must be recorded. This reconstructs a
local coordinate derivative, not original optimizer logits or Adam state.

Initial SH is padded to 16 bases and evaluated at degree zero, making the inactive SHN
gradient zero. Zero or near-zero reference norms remain visible through absolute norms and
undefined counts; undefined cosine/relative ratios must not be replaced by perfect agreement
or zero disagreement. Statistics use float64 reductions of float32 CUDA gradients. Only
within-state vectors share row identity; different final topologies cannot be subtracted or
concatenated as if corresponding. Norms from groups with different units are not comparable.

Per-view statistics and statistics of the full-view mean gradient are different quantities and
are retained separately. The nested state/seed/view headline is descriptive, with every row
available; three saved-state seeds do not create independent captures. The equal-target
independent render/backward repeat at C0004 for each state is a numerical sanity check at those
nine sites, not a global bound on numerical variation at every other view. Claims that a local
difference exceeds repeat variation must stay within what the recorded controls establish.

The fixed state/view order, no-update constraint, fail-closed numerical control and finite-value
checks, CUDA-only policy, state and total time limits, and no automatic retry after outcome are
appropriate. Shared-GPU resource receipts are descriptive. The selected saved-model previews
and ordinal diagnostic history are correctly distinguished from new fitting results.

## Required final implementation review

Before execution, verify the following against exact bound source and synthetic tests:

1. Each independent target/control evaluation uses clean leaves and a fresh render graph;
   gradient extraction does not accumulate stale `.grad` values, alter tensors or run optimizer
   or topology updates. Unused degree-zero SHN gradients become explicit zero tensors.
2. Direct autograd of the full loss agrees with the weighted component combination under a
   prospective synthetic tolerance. Local opacity chain mapping and endpoint handling do not
   change final saved effective values. Raw saved quaternion and row identities are preserved.
3. The identity control exercises both actual target paths with independently rendered graphs,
   preserves every component/group comparison, and fails at the stated tolerances without
   relaxing them after new capture outcomes.
4. Per-view and aggregate gradient arrays, losses, statistic denominators, null counts and
   target/model hashes permit independent reduction checks. Mean-of-norm and norm-of-mean
   labels remain distinct. Region partition, empty regions and zero-error shares have direct
   synthetic tests.
5. Input guard negative controls deny raw dataset files, heldout material and unlisted source-run
   files. Imports, saved output writes and administrative raw-seal hashing do not accidentally
   broaden the result worker's read boundary. Enforce declared timeouts and preserve failure
   receipts without starting another diagnostic attempt automatically.
6. Exact source/environment/configuration, all source bytes, protocol and final review are
   preserved before the worker. Shared report schemas, actual stage clocks, historical-model
   labels, mandatory visuals and browser/manifest handoff are supported by the implementation.

These are implementation acceptance checks under the approved design, not permission to run
capture diagnostics before the separate final protocol review.

## Prospective corrections and delivery

The original unexecuted design digest
`cab00c48fa1b949d8df052a06d7b31cd312549f256e4294ac07a8f602ca2e396`
had an ambiguous blanket region denominator and requested report port 8766, unsupported by
the existing v2 canonical command checker. The Driver amended the denominator and null rules
and selected canonical port 8765 before any implementation or new outcomes. Viewer port 8880
is separate. Replacing a verified earlier task-owned report server is allowed at delivery;
unrelated services must remain intact. No checker or production-source change is necessary
for this operational correction.

No private data or source context was sent to Claude. No task record, protocol or prior result
was modified by this reviewer. The final source review and independent results audit remain
separate required stages.
