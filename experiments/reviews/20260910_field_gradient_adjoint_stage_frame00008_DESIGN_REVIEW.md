# Prospective Design Review

- Task ID: `20260910_field_gradient_adjoint_stage_frame00008`
- Protocol SHA-256: `e7b52a8d71a6ac264fe0311be424ceaa2e677441d4a5aeac5c52cd558e6ce52a`
- Reviewer: `Codex-continuation-reviewer`
- Verdict: `approved`
- Outcome Access: `none`
- Scope: `design only; no init-run or protected execution authorization`
- Reviewed task file SHA-256: `ea9662526eeacb9c44e403691d9693fba5873f59bd82f2ea2528f8fcc1334dee`
- Review time: `2026-09-09T22:33:05Z` (2026-09-10 Europe/Berlin)

## Decision and exposure boundary

Approve implementation of this bounded numerical-method follow-up. Exact source,
synthetic tests, final protocol binding and a distinct final prospective review remain
required before execution. This document cannot substitute for that review.

Outcome Access means no outcomes of this new diagnostic were generated or inspected.
The reviewer was given the failed predecessor's numerical-control summary and reviewed
the already exposed RTGS-021 results in an earlier task. The design expressly records
predecessor exposure. Its failed run, thresholds, source and partial outputs remain
unchanged; this approval neither rescues nor authorizes completing that run.

## Mathematical and control findings

For the fixed render Jacobian J and image-loss adjoints h, the chain rule gives
`g_field - g_photo = J^T(h_field - h_photo)` in exact arithmetic. Direct propagation
avoids estimating the difference by subtracting two independently accumulated large
parameter gradients. It does not make floating-point VJPs exact. The derived field
gradient and local opacity chain are correctly distinguished from historical raw
optimizer coordinates or update replay.

The revised photo-only numerical stage now includes three distinct comparisons:
fresh whole-chain repeats, adjoint repeats on one fixed detached image, and nonzero
VJP repeats with one fixed adjoint and retained render graph. This avoids attributing
downstream variation to VJP accumulation merely because upstream differences passed a
tolerance. Loss/render and adjoint gates remain fixed and fail closed before field
measurements. Ordinary parameter-repeat variation is descriptive and cannot be called
a pass of the predecessor's gate.

The `J^T0` check is explicitly a null/routing control. Independent synthetic
distinct-target differentiation is required to test nonzero algebra; a zero-input
pass alone would be insufficient. Two nonzero photo and delta VJPs at every planned
state/view provide local observed-repeat context. The frozen ten-times rule is a
descriptive precision flag, not a confidence interval, global error bound or material
effect threshold. Shared systematic error and unobserved repeat variation remain
possible. Weak effects and angular comparisons must retain their unresolved status.

All comparisons remain within identical state/view parameter coordinates. The fixed
loss, masks used only for residual strata, cached training inputs, nine saved states,
22 training cameras and explicit heldout prohibition preserve the diagnostic scope.
No fitting, topology change, causal attribution, high-quality reconstruction,
physical-density or strict field-only capability claim follows from this design.

## Read-only checks

- Read the complete task, experiment lifecycle and applicable agent/experiment/review
  instructions; inspected the existing target routes, gradient mapping, tiled SSIM
  and gsplat wrapper source without executing them on capture data.
- Recomputed both the initial and revised design digests with
  `scripts/experiment_contract.py review-digest`.
- Checked 76 unique listed cache/state paths and absence of heldout target paths.
  Did not decode their arrays or calculate new residuals, gradients or renders.
- Confirmed the new driver and canonical run were absent during design inspection.

## Required final implementation checks

Verify clean graph/state handling and no stale gradient accumulation; isolated repeats
must use the identical stored image, graph and adjoint specified by the protocol.
Check independent CPU/CUDA distinct-target algebra, weighted full-loss parity and
opacity mapping before new capture outcomes. Retain raw repeat arrays sufficient to
recompute component and total means, norms, flags and aggregate statistics; keep
unresolved counts visible beside descriptive headline cosines. No cross-topology
vector subtraction or cross-group raw-norm comparison is valid.

Verify input guards, finite checks, fixed failure gates, time limits, source/cache
entry and exit hashes, and existing-model presentation labels. Any storage cap,
preflight requirement or other implementation-stage protocol amendment needs the
final exact digest and review before `init-run`. Preserve failures without automatic
retry or post-outcome tolerance changes.

No source, old protocol, active task, private data, commit or remote state was modified
by this reviewer. Only this design-review record was written.
