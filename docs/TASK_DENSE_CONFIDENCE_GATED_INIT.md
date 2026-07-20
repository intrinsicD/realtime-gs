# Task: Confidence-gated dense image-free initialization (chained experiment plan)

Status: preregistered plan — not yet executed on calibrated data
Primary path: `CompactCarveConfig.select_all_eligible` → `merge_by_voxel(return_group=True)` →
correspondence-confidence gate → fixed 3D fitting + density control
Depends on: the CSR-accelerated placement, `rtgs.lift.compact_init_eval`,
`benchmarks/compact_init_eval.py`, and `rtgs.lift.compact_refine` (all landed 2026-07-20).

## Why this exists

Two 2026-07-20 findings motivate this chain:

1. **Dense+merge is a cheap, promising denser init.** Retaining one carve lift per supported 2D
   Gaussian across all views and deduplicating with the voxel-hash moment merge led the balanced
   top-K by +1.04 dB init-only mean foreground PSNR on the synthetic scene. This is a mechanism
   check only; the calibrated-frame number is unmeasured.
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

### E1 — Dense+merge vs balanced top-K, init-only, on a calibrated frame

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

### I1 — Correspondence-confidence gate (implementation)

- Build a per-cluster confidence record from the signals above and a preregistered classifier
  (start with `m_c ≥ 2` AND spread ≤ τ·voxel AND `candidate_half_max_widths` ≤ w AND
  reprojection residual ≤ ρ). Emit an "easy-only" gated initialization plus a diagnostics table
  (kept/dropped counts, per-signal distributions).
- CPU-first, deterministic, tested on fixtures where the easy/hard split is known by construction
  (e.g. a target seen by 3 views is easy; a single-view-only decoy is hard). Off by default; opt-in
  config on the dense path. Add a `--gate` mode to `benchmarks/compact_init_eval.py`.

### E2 — Easy-only seed + density control vs dense-all vs top-K, downstream

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

### I2 / E3 — Explicit correspondence for the hard set (conditional on E2)

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

E1 done when the calibrated dense-vs-top-K init-only comparison is measured, audited, and logged.
I1 done when the gate is implemented, tested, and its diagnostics reproduce the E1 cluster
distribution. E2 done when the three-arm downstream comparison meets its preregistered rule and is
audited. I2/E3 are opened only by an E2 outcome that localizes held-out regressions to hard-dropped
regions.
