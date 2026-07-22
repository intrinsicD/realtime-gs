# Task: Confidence-gated dense image-free initialization (chained experiment plan)

Status: E1/I1/E2 executed and independently audited — negative E2; no default change; I2/E3 closed
Primary path: `CompactCarveConfig.select_all_eligible` → `merge_by_voxel(return_group=True)` →
correspondence-confidence gate → fixed 3D fitting + density control
Depends on: the CSR-accelerated placement, `rtgs.lift.compact_init_eval`,
`benchmarks/compact_init_eval.py`, and `rtgs.lift.compact_refine` (all landed 2026-07-20).

## Outcome (2026-07-20)

- **E1 failed its count-controlled gate.** Dense+merge improved all seven calibrated compact
  training views and gained **+1.9714 dB** mean foreground PSNR over balanced top-K, but used
  **13.48×** as many Gaussians (2,319 versus 172), above the frozen 2× limit.
- **I1 reproduced its frozen gate.** The deterministic confidence classifier retained exactly
  **60/2,319** clusters. Its same-view init-only screen was +0.4505 dB over top-K, but was
  exploratory because those views parameterized the classifier.
- **E2 rejected easy-only under the exact frozen downstream schedule.** After 300 matched
  gsplat-Default steps, late-release C1004 foreground PSNR was **14.9079 dB** for dense-all,
  **12.7332 dB** for easy-only, and 11.2280 dB for top-K. Easy-only was 2.1747 dB behind dense-all,
  far outside the 0.0071 dB control-repeat envelope, despite ending smaller and slightly faster.
- Balanced top-K remains the default. Easy-only was still growing at step 300, so the negative
  result applies to this schedule rather than proving incapability. The deficit was not spatially
  localized to hard-dropped regions, so the plan's condition for opening I2/E3 was not met.

Canonical records:
[`E1 result`](../benchmarks/results/20260720_dense_confidence_gated_init_e1_RESULT.md),
[`E1 audit`](../benchmarks/results/20260720_dense_confidence_gated_init_e1_AUDIT.md),
[`I1 result`](../benchmarks/results/20260720_dense_confidence_gated_init_i1_RESULT.md),
[`I1 audit`](../benchmarks/results/20260720_dense_confidence_gated_init_i1_AUDIT.md),
[`E2 preregistration`](../benchmarks/results/20260720_dense_confidence_gated_init_e2_PREREG.md),
[`E2 result`](../benchmarks/results/20260720_dense_confidence_gated_init_e2_RESULT.md), and
[`E2 audit`](../benchmarks/results/20260720_dense_confidence_gated_init_e2_AUDIT.md).

### Longer all-view development context (2026-07-21)

A separately preregistered full 26-view suite retained native counts and let the same ordinary
adaptive-density schedule run to the 70k plateau assessment. Dense+merge initialized 2,088
Gaussians and led initialization foreground PSNR at 20.7546 dB; it also led terminal fitted-view
foreground PSNR at 38.2480 dB with 49,177 Gaussians. Easy-only kept only 7 clusters, grew to
35,644, and finished last at 36.9587 dB. This confirms the earlier direction under a much longer
all-fitted-view schedule, but does **not** supersede E2's held-out matched-cap decision: native
counts and final capacities differ, every camera was fit, and dense's selected objective was 4.40%
worse than beam fusion's. The audited suite therefore found no materially superior converged
initializer and made no default change. Records:
[`suite result`](../benchmarks/results/20260721_all_initializers_frame00008_RESULT.md) and
[`suite audit`](../benchmarks/results/20260721_all_initializers_frame00008_AUDIT.md).

## Why this exists

Two 2026-07-20 pre-execution findings motivated this chain:

1. **Dense+merge is a cheap, promising denser init.** Retaining one carve lift per supported 2D
   Gaussian across all views and deduplicating with the voxel-hash moment merge led the balanced
   top-K by +1.04 dB init-only mean foreground PSNR on the synthetic scene. This was a
   mechanism check only; the calibrated-frame number was unmeasured at preregistration.
2. **Correspondence-free consensus does not pin geometry.** The local 4-dof refine
   (`rtgs.lift.compact_refine`) reliably maximizes its multi-view consensus objective but can move
   geometry *away* from the surface (it rewards coverage, drifting to the volumetric density core).
   Pinning depth needs explicit cross-view correspondence, not consensus.

The strategy this plan tests: instead of trusting every dense lift or a fragile consensus refine,
**keep only the confident ("easy") correspondence clusters as a sparse-but-accurate seed, drop the
ambiguous ("hard") ones, and let post-merge Adam + density control (split/clone/prune + MCMC
teleport) reconstruct the rest.** Explicit correspondence is added back only for the hard set, and
only if the evidence shows densification cannot cover it.

Every stage is preregistered here, uses a calibrated `dataset/` frame (synthetic is mechanism-only),
records init-only *and* downstream metrics on a frozen train/validation/held-out split, and passes
the results-audit skill before any claim enters README/docs or changes a default. No stage changes
the balanced top-K default until its gate is met.

## The easy/hard correspondence signal (used by I1/E2)

A merge cluster (the group returned by `merge_by_voxel(..., return_group=True)`) is a putative
correspondence: the per-view lifts that fused to one voxel. Its confidence is computed from data
already produced by placement + merge — no new geometry:

- **View multiplicity** `m_c` — distinct source views in the cluster
  (`lineage.source_view_indices[group == c].unique().numel()`). The primary easy/hard separator:
  high `m_c` = many cameras independently triangulated here; `m_c == 1` = a monocular guess.
- **Spatial cohesion** — spread of the pre-merge means within the cluster relative to the voxel.
- **Depth sharpness** — `CompactCandidateAudit.candidate_score_margins` and
  `candidate_half_max_widths` (narrow, high-margin peak = confident depth; flat/wide = ambiguous),
  plus `candidate_best_n_covered` ≥ `min_views`.
- **Color agreement** — within-cluster consensus-color variance (the merge `color_bin_size` already
  gates gross front/back splits; this quantifies the residual).
- **Reprojection residual** (optional, stronger) — reproject the merged mean into each contributing
  view and compare to that view's source `xy`; small residual = geometrically consistent
  (`rtgs.lift.gaussian_correspondence.triangulate_centers_dlt`).

"Easy" = high multiplicity, tight, sharp, color-consistent, low reprojection residual. The exact
thresholds are preregistered in I1 and frozen before E2.

## Chain: experiment → evidence → implementation/follow-up → experiment …

### E1 — Dense+merge vs balanced top-K, init-only, on a calibrated frame — complete

- **Hypothesis**: on a calibrated `dataset/` frame, dense+merge yields higher init-only compact-view
  mean foreground PSNR than the balanced top-K at a controlled Gaussian count, and its merge
  clusters show a nontrivial easy/hard multiplicity distribution.
- **Protocol**: build `ReconstructionInputs` for one frame (e.g. `dataset/karate/frame_00005`);
  `python benchmarks/compact_init_eval.py --bundle <dir> --out <dir>`; report per-view and mean
  full/foreground PSNR + SSIM, Gaussian counts, and the cluster view-multiplicity histogram; save
  `init_topk.ply` / `init_dense_merged.ply` and the `rtgs view` command.
- **Preregistered decision rule**: dense+merge is a "better init" if mean foreground PSNR gain
  ≥ 0.5 dB with no view regressing > 0.25 dB, at a Gaussian count within 2× of the top-K control.
  Record the result regardless of sign.
- **Artifacts**: `init_eval.json`, PLYs, and a results-audit disposition.
- **Unlocks**: the cluster multiplicity distribution parameterizes I1 either way; a negative E1 still
  informs the gate (e.g. dense init is noisy → gating is more important, not less).

### I1 — Correspondence-confidence gate (implementation) — complete

- Build a per-cluster confidence record from the signals above and a preregistered classifier
  (start with `m_c ≥ 2` AND spread ≤ τ·voxel AND `candidate_half_max_widths` ≤ w AND
  reprojection residual ≤ ρ). Emit an "easy-only" gated initialization plus a diagnostics table
  (kept/dropped counts, per-signal distributions).
- CPU-first, deterministic, tested on fixtures where the easy/hard split is known by construction
  (e.g. a target seen by 3 views is easy; a single-view-only decoy is hard). Off by default; opt-in
  config on the dense path. Add a `--gate` mode to `benchmarks/compact_init_eval.py`.

### E2 — Easy-only seed + density control vs dense-all vs top-K, downstream — complete

- **Hypothesis**: an easy-only accurate seed + Adam + density control (gsplat DefaultStrategy
  clone/split/prune and MCMC teleport/relocation via `rtgs.optim.strategies`, or the CPU classic
  `rtgs.optim.density` for the reference) reaches equal-or-better final quality than the dense-all
  and top-K inits, with fewer/cleaner primitives and less optimizer fighting — i.e. densification
  reconstructs the dropped-hard regions.
- **Protocol**: from each of {top-K, dense-all+merge, easy-only-gated} run the *same* fixed fitting
  schedule to a matched primitive budget on the frozen split; report final compact-view and
  held-out PSNR/SSIM/alpha-IoU, the primitive-count trajectory, and time-to-quality. GPU path uses
  gsplat; keep a CPU-reference smoke.
- **Preregistered decision rule**: easy-only wins if final held-out foreground PSNR is within
  −0.1 dB of the best competing init at ≤ its primitive count and time, or better on either at
  equal quality. Compare against a control/control repeat envelope; use the tighter bound.
- **Artifacts**: benchmark JSON, viewer PLYs (init + final for each arm), audit.
- **Unlocks**: if densification does *not* cover the dropped-hard regions (held-out regressions
  localize to hard-dropped areas), open I2/E3; otherwise easy-only becomes the candidate default and
  the correspondence work is deprioritized.

### I2 / E3 — Explicit correspondence for the hard set — closed by E2 unlock rule

- **I2**: wire `rtgs.lift.fiber_correspondence` into `compact_refine` so hard-set depth is
  constrained by cross-view matches (not consensus), recovering the correspondences the consensus
  refine could not. Respect the module's documented failure modes; keep it CPU-first and audited.
- **E3**: compare {easy-only + densify} vs {easy + hard-via-correspondence + densify} on the same
  frozen split. **Hypothesis**: adding matched hard correspondences improves held-out quality in the
  regions densification missed, without regressing the easy set. Preregister the region-localized
  decision rule before running. Never combine an I2 change and a densification change in the same
  confirmatory run.

## Cross-cutting rules

- Calibrated `dataset/` frame for every results-bearing stage; synthetic is mechanism-only.
- Report init-only metrics beside downstream metrics; never attribute densification recovery to the
  initializer.
- Deterministic for fixed inputs/seed/dtype/thread config; preregister decision rules before running.
- Run the results-audit skill after each stage; add a dated `docs/EXPERIMENTS.md` entry; update the
  generated benchmark table via `benchmarks/run.py --update-docs`; preserve JSON, PLYs, and the exact
  `rtgs view` command.
- No default change (top-K → dense/easy-only) until the corresponding gate is met and audited.

## Definition of done (per stage)

E1, I1, and E2 are complete, audited, and logged. E2 did not meet its preregistered win rule,
and its aggregate held-out deficit was not localized to hard-dropped regions; therefore I2/E3 were
not opened. A longer budget-filling schedule or spatial-localization diagnostic would be a new,
separately preregistered experiment rather than a reinterpretation of this chain.
