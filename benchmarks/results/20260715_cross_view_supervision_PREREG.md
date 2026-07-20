# Leave-one-source-view-out photometric supervision preregistration

Written on 2026-07-15 before implementing source-aware photometric supervision or running any
leave-one-source-view-out benchmark.

## Question and literature boundary

Does excluding a target view's own fitted 2D-Gaussian descendants from that target's photometric
loss remove a source-reconstruction shortcut and expose a materially stronger cross-view depth
signal? Each primitive begins on its source-camera ray and its scale grows with depth to preserve
approximately the same source footprint, so its own image can be explained with little depth
information.

The 2026-07-15 Scholar Inbox digest grounds the diagnosis, not the implementation or thresholds.
MAC-Splat reports that photometric-only sparse-view supervision leaves depth/correspondence
ambiguities and instead uses matched world-frame attribute consistency. ExtraGS and FlowPainter
support confidence only when it comes from observation uncertainty or localized guidance, so they
do not reopen the closed confidence-anchor branch. Incremental Gaussian Triangulation motivates a
later local-plane alternative. LOSO is a repository-specific diagnostic, not a reproduction of any
paper, and no paper supplies the gates below.

## Frozen families and arms

Both families reuse the same fitted 2D tensors and final evaluation always renders the full output.
Only the primitive subset rendered inside each training loss differs.

1. `gradient/all`: current `GradientLifter` objective with every retained primitive.
2. `gradient/leave_one_source_out`: for target local view `v`, render only primitives whose source
   view is not `v`.
3. `gradient/matched_nonself_dropout`: retain all source-`v` primitives but exclude exactly the
   same number of frozen non-`v` primitives as arm 2, selected once per target with a dedicated RNG.
4. `hybrid/all`: current depth-seeded objective with every retained primitive.
5. `hybrid/leave_one_source_out`: the same source exclusion as arm 2.
6. `hybrid/matched_nonself_dropout`: the same count-matched control as arm 3.

`photometric_supervision_mode="all"` must remain the default. Because opacity is frozen to the same
value for every primitive, exact count matching also matches excluded opacity mass. The dropout RNG
is seeded independently from depth jitter and target-view sampling and its masks remain fixed for
the entire optimization.

## Frozen data and execution

- Seeds 0, 1, 2; CPU reference rasterizer; four PyTorch threads.
- Forty-Gaussian synthetic scenes; twelve 48x48 cameras.
- Training views `[0,1,2,4,5,6,8,9,10]`; strictly held-out views `[3,7,11]`.
- Fit 150 2D Gaussians per training image for 120 iterations once per seed; reuse exact tensors
  across all six arms.
- Ninety lift iterations, learning rate 0.1, and minimum fitted weight 0.05.
- Disable rotation optimization, scale optimization, merging, density control, and downstream
  refinement.
- Gradient family: no depth prior, legacy anchor semantics, depth jitter 0.15, and the historical
  GradientLifter depth-prior lambda 0.001.
- Hybrid family: metric training GT depth with deterministic 8x8 corruption blocks selected by
  `(x // 8 + y // 8 + local_source_index) % 3 == 0`; multiply corrupted depth by 1.20 for even
  local source indices and 0.80 for odd indices. The backend returns no confidence. Use legacy
  anchor semantics, depth jitter 0.02, and depth-prior lambda 0.01.
- One official three-seed run. No seed replacement, schedule/lambda/iteration change, mask redraw,
  tuning, rerun, or post-hoc refinement is permitted for the decision.

## Pre-metric validity assertions

The official artifact is invalid rather than replaceable if any assertion fails:

1. within each family, all arms have bit-identical step-0 means, quaternions, scales, opacity, SH,
   primitive ordering, source-view labels, and source-group boundaries;
2. explicit `all` and the default pre-change behavior are bit-identical in tests;
3. complete target-view schedules are identical across arms, contain training views only, and visit
   every training view at least once;
4. LOSO excludes exactly the target source group and no other primitive on every target view;
5. matched dropout excludes exactly the LOSO count, excludes zero target-source primitives, matches
   excluded opacity sum, and uses one frozen mask per target from a separate generator;
6. a two-step zero-learning-rate check has bit-identical full outputs and identical target schedules
   across modes, proving the intervention does not disturb initialization or the optimizer RNG;
7. final primitive count/layout is shared, final evaluation uses all primitives, and every output,
   loss, and recorded diagnostic is finite; and
8. `HybridLifter` forwards the supervision mode exactly.

Fitted-tensor hashes, corrupted-prior hashes, source labels/boundaries, target schedule hashes, and
dropout-mask hashes must be recorded per seed.

## Primary metrics and decision rule

For both families, primary geometry metrics are:

- strict held-out expected-depth RMSE divided by scene extent; and
- source-ray absolute-relative depth-error p90 over every retained ray with valid GT depth.

The Hybrid family additionally uses source-ray absolute-relative p90 on the predetermined corrupted
blocks. Ground truth is used only to construct the controlled Hybrid corruption and report geometry.

For a lower-is-better metric `m`, define
`gain(arm) = (m(all) - m(arm)) / m(all)`.

A family has a material LOSO effect only if LOSO:

1. reduces mean held-out depth RMSE by at least 2%;
2. reduces mean all-source p90 by at least 10%;
3. wins at least two of three paired seeds on both metrics;
4. has mean held-out PSNR no more than 0.10 dB worse; and
5. loses no more than 0.02 absolute foreground coverage or alpha IoU.

Hybrid must additionally reduce corrupted-source p90 by at least 15% with at least two of three
paired seed wins.

The material effect is attributable specifically to own-source exclusion only if every material
gate passes, LOSO beats matched dropout in at least two of three seeds on every primary geometry
metric, and matched dropout preserves at most half the LOSO gain for each metric:
`gain(matched_nonself_dropout) <= 0.5 * gain(leave_one_source_out)`.

Held-out SSIM, nearest-GT-center median/p90, source median error, full-training PSNR, loss
checkpoints, primitive count, lift time, and the matched-control mask composition are secondary
diagnostics and cannot rescue a failed primary rule.

## Stopping and interpretation rule

- If both families pass attribution, authorize calibrated real-data replication; do not change the
  default yet.
- If exactly one family passes, scope the follow-up to that family only.
- If LOSO passes the all-arm utility gates but fails the matched control, classify the effect as
  generic primitive-dropout/density behavior rather than source-specific identifiability.
- If neither family passes, stop LOSO/dropout/schedule sweeps on this setup and pivot to one direct,
  robust world-frame position-consistency term between fixed train-view matches while depths remain
  ray-bounded. Shape/appearance consistency and local plane/normal constraints remain later steps.

No outcome reopens confidence-anchor sweeps, and this synthetic experiment cannot change production
defaults by itself.

## Pre-run audit scope clarification

Added after implementation/smoke validation but before the one official run; no arm, threshold,
metric, or stopping rule changed. The matched control removes a dispersed non-self subset. It
matches submitted primitive count and frozen scalar opacity sum, but not coherent source-group
topology, target-view visibility, projected alpha, color, or spatial coverage. Therefore a positive
`own_source_attribution_pass` means LOSO beats this exact count/opacity-matched dispersed control;
it does not rule out every coherent whole-view dropout mechanism. The pre-run audit additionally
required one global non-self assignment: every primitive is excluded from exactly one matched
target, eliminating per-primitive supervision-exposure imbalance without changing any arm or gate.
A Hybrid-only pass is also scoped to deterministic corrupted metric priors because clean-prior
safety is not part of this run. Neither caveat weakens an all-versus-LOSO utility failure.
