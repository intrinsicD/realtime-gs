# Prospective Protocol Review

- Task ID: `20260926_field_only_distillation_stage_frame00008`
- Protocol SHA-256: `ef904cd749a80de5f25b9c6706cc705617aebfe56bd0b16912695626a99e44af`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

The protocol asks whether a random-initialized 3DGS trained only from decoded compact 2D
Gaussian fields of 22 training views, their calibrated cameras and the packed training alpha,
with colour supervised inside the mask and silhouette/outside-alpha terms against floaters,
matches photograph supervision inside four held-out masks without more floaters (H1), and whether
the teacher fitted without mask containment beats the mask-contained teacher (H2). H3 (random
versus packed-alpha visual-hull initializer) and H4 (silhouette objective versus premultiplied
black) are descriptive with paired deltas and no pass/fail.

Evidence maturity this protocol may establish: a development screen on one previously
outcome-exposed frame (00008), one frozen train/held-out split, three paired seeds and one fixed
8000-step gsplat budget at downscale 8. A pass may keep the RTGS-025 main path and select the
uncontained teacher family for the next registered task. It cannot establish a default, SOTA,
generalization, physical-geometry, speed or VRAM claim. Because the packed alpha is mask-derived
and drives supervision and the hull, the field arms are field-plus-silhouette, not strictly
Gaussian-only. H2 compares teacher families (count, topology schedule and containment differ
together), not containment alone. A verdict here says only whether the frozen design is fit to
execute; it is not evidence that the method works.

## Checks

Read: `CLAUDE.md`, `experiments/README.md`, `docs/AGENT_WORKFLOW.md`,
`.agents/state/current-task.md`, the `rtgs-experiment` and `rtgs-review` skills, the reviews
README and template, the prior task `20260908_field_teacher_information_stage_frame00008.json`
with its protocol review (not its RESULT/AUDIT records), the task JSON, the driver, the report
module, `rtgs.data.field_targets`, both test files, `rtgs.optim.trainer` (masked objective,
history, seeding), `rtgs.data.compact_views`, `rtgs.core.observation2d` and
`observation2d_cuda`, `rtgs.core.camera`, `rtgs.data.calibrated`, `rtgs.visualize`,
`rtgs.render.gsplat_backend`, `rtgs.cli` (view flags), `scripts/experiment_contract.py`
(protocol digest, review-artifact validation, data seal, source binding, `init-run`) and the
experiment templates.

Commands executed (all read-only or CPU unit tests; none touched images, masks, `.rtgsv`,
`.npz`, `.ply`, `.scratch/` or any run outcome):

```text
git status --short; git rev-parse HEAD; git log --oneline
git diff afa9b35 298a9b5 -- experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_field_only_distillation.py tests/test_field_targets.py
sha256sum <task-owned sources and data seal>   (table below)
```

Results: `review-digest` reproduces `ef904cd749a80de5f25b9c6706cc705617aebfe56bd0b16912695626a99e44af`
exactly; `validate` and `validate-data` print OK; pytest reports 14 passed and 1 skipped (the
CUDA decode-parity test self-skips in the review environment). The worktree is clean at
`298a9b5`. No `runs/20260926_field_only_distillation_stage_frame00008/` root and no
`benchmarks/results/20260926_*` evidence exist.

- Digest scope. `protocol_sha256` drops only `status` and `protocol_review`, so recording this
  verdict does not change the digest. Every change required below alters bound protocol text or
  bound source and therefore needs a new digest and a new review.
- Data roles and leakage. The seal binds the calibration, all 26 photographs, all 26 lossless
  masks and, for each of the three field families, all 26 views plus both manifests; the
  coordinator only hashes those bytes at entry and exit. `prepare` and `initialize` load training
  views only (`view_ids=[view]`, `test_every=0`), and the audit hook denies any open under the
  frame whose stem is a held-out id. `fit` denies every open under the frame and every
  target-cache open outside the worker's own family plus `training_masks` and `metadata.json`,
  so field workers cannot read photographs or the photograph cache and photograph workers cannot
  read fields. `evaluate` refuses to start until all 21 cell receipts read `completed`; only then
  are held-out photographs, masks and (for the field-consistency diagnostic) held-out compact
  views opened, and the guard records every dataset open. Fitting scenes carry no test indices,
  bounds come from training cameras and packed alpha only, and `prepare` aborts unless the
  packed alpha equals the thresholded lossless mask for every training view. Photographs reach the
  field arms nowhere except through that mask-equal packed alpha.
- Masked objective. With `use_masks=True` and `random_background=True` the Trainer composites
  `target*mask + background*(1-mask)` with a fresh random background each step, weights L1 by
  `0.1 + 0.9*mask`, adds `outside_alpha_lambda*mean(alpha*(1-mask))` and
  `mask_alpha_lambda*mean|alpha-mask|`, and takes SSIM on the mask crop. Where the packed-alpha
  fraction is exactly zero the field colour is ignored and only alpha is penalized, so floaters
  are controlled without trusting outside-mask field colour. At partial-coverage pixels
  (fractions 0.25/0.5/0.75 from the four-site box average) outside-mask sites still enter the
  target, weighted by the fraction; see B2.
- Scoring. `mask_scores` computes foreground PSNR only over the binarized held-out mask, LPIPS on
  render and photograph both multiplied by the mask and cropped to the 8-px-padded mask box,
  outside-alpha mass and floater fraction outside the 3-px dilated mask, and interior alpha inside
  the 3-px eroded mask, all from a black-background gsplat render of the saved final model. The
  dilate/erode operators are correct square structuring elements with conservative borders; the
  unit tests cover unscored outside colour, a floater row and an in-band halo. The report
  recomputes per-view means, requires exact held-out view coverage in order, and rejects
  nonfinite values.
- Coordinate conventions. Camera projection, field query, `grid_sample(align_corners=False)` and
  `decode_alpha_grid` all use edge-origin coordinates with pixel centres at +0.5; `downscale_pinhole`
  divides intrinsics exactly; the quarter-offset sites are identical for fields and photographs;
  the compact camera record must equal the calibrated full-resolution camera exactly; the alpha
  lookup is a floor lookup on the same sites. The held-out evaluation mask uses bilinear
  sampling of the binary mask at the same sites, a slightly different operator from the
  nearest-lookup training alpha, but it is held-out-only and identical across arms.
- Initializers. Random: CPU generator seeded by the paired seed, float64 uniform-ball draws of
  radius extent/2, mean nearest-neighbour isotropic scale, opacity 0.1, grey; byte-identical
  across random cells and digest-checked in every fitting worker. Hull: 128^3 lattice over the
  bound cube, soft alpha >= 0.5, positive depth and in-image in every training view, face-touch
  abort, minimum 1000 shell voxels, colour averaged over all training projections of the arm's own
  targets. The disclosed count confound (about 7740 shell voxels versus 20000 random points) and
  the undisclosed scale difference (mean nearest-neighbour distance versus one voxel) both leave
  H3 descriptive, as the protocol states.
- Premultiplied control. `nb_rand_pm` multiplies the decoded colour by the same soft alpha,
  passes no masks, and its resolved config differs from the masked config in exactly `use_masks`
  and `random_background` (unit-tested). H4 is descriptive.
- Gates. `gates()` is inclusive and per paired seed over the frozen seeds; H1 matches the text;
  G0 applies the numeric floor and defers visual adequacy to the independent audit, with failure
  making H1/H2 inconclusive; the H2 pass condition matches the text; the H2 reject condition is
  implemented as the mirrored pair (see B3). `check_protocol_tables` requires the driver's
  condition table to equal the frozen comparators and every condition/seed cell exactly once.
- Stopping and failure. Fixed 8000 steps, `checkpoint_policy: final`, no plateau or held-out
  selection; `executed_iterations` must equal 8000; nonfinite renders, parameters, losses and
  metrics abort; each fitting worker has a 3600 s timeout; the matrix stops on the first
  failure and writes `execution_failure.json` plus an explicit failure report. A timeout kill
  leaves no cell receipt (R8).
- Resolved configs and RNG. All six frozen configs round-trip through `TrainConfig` and
  `DensityConfig` at the bound source (unit-tested). Python, NumPy, global Torch and all CUDA
  generators are seeded after the disposable warmup; the Trainer's private camera/background
  generator is seeded from the same paired seed. CUDA bitwise determinism is not claimed.
- Source binding. The 118-file count equals the four `src/rtgs` patterns (107 `.py`, 4
  `.cu/.cpp/.h`) plus the seven named files. The sandbox denied the pipeline needed to recompute
  the aggregate, so `b924ea17f43cd8d8a9041ccc5b76a4a1d8ff8d4e2d535ea25ba4af937e24b77d` was not
  independently reproduced. The registry `validate` command calls `validate_task(...,
  check_live_source=False)`, so the Driver's green `validate` does not prove live equality either;
  `init-run` and `check-run` enforce it and fail closed.
- Resource accounting. Fresh process per cell, warmup excluded from the cell wall clock, peak
  allocated/reserved and `ru_maxrss` per cell, descriptive only with no speed inference. The
  Trainer resets the CUDA peak again after the driver's reset (R4).
- Pre-review smoke. `git diff afa9b35 298a9b5` on the task shows hypotheses, thresholds, seeds,
  split, comparators, metrics, resolved training configs and execution order byte-identical before
  and after the smoke; the only changes are the disclosure sentence, cleared blockers, the explicit
  parity seed, machine-readable initialization parameters already present in the prose, and the
  source binding. The exposed quantities were training-view teacher fidelity (report-only) and
  the hull shell count (input-structural); no held-out view was touched. The smoke is acceptable
  with the disclosure already in `claim_boundary`. It did not exercise the coordinator start-up
  path (see B1).

Reviewed task-owned identities:

| File | SHA-256 |
|---|---|
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008.py` | `a3326987bd22487369132856605d25fa2c82b5167b5010141bdeff79451a6bda` |
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008_report.py` | `2726d761c9aea31f40b1f208680a50de6f6db7b17c3a9104d8d79d55f882daf0` |
| `tests/test_field_only_distillation.py` | `1e08d4a414e3d108d15c8dc2c4459d8bdaacd66feacbec60510ae93eee140df6` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` |
| `experiments/data/20260926_field_only_distillation_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` |

## Findings

Rejected for three bounded reasons. The comparison design, leakage boundaries, mask-restricted
scoring, initializer sharing, premultiplied control, per-seed inclusive gates and fixed-budget
stopping are correct as implemented; nothing below changes the scientific design.

Required changes:

- B1. Coordinator start-up is not fail-safe and will very likely crash on the first `run`.
  `snapshot_source` calls `importlib.metadata.version("realtime-gs")` unguarded. The project
  distribution is `rtgs` (`pyproject.toml` `[project] name = "rtgs"`; `.venv` carries
  `rtgs-0.1.0.dist-info` and `__editable__.rtgs-0.1.0.pth`, and no `realtime_gs` distribution),
  and every previously executed driver that records this name (20260806, 20260907, 20260909
  with `rtgs`) wraps the lookup in `except PackageNotFoundError`. The call sits before the
  coordinator's `try`, after `source_snapshot/` has been written, so the failure leaves no
  `execution_failure.json`; a second `run` returns early from `snapshot_source` and never writes
  `environment.json`, which `metrics.json` later declares as an artifact, so a full 21-cell run
  would end in a bundle that cannot pass `check-run`. Fix, bounded to the driver: (a) query the
  real distribution name or guard the lookup as the earlier drivers do; (b) write
  `environment.json` independently of the snapshot early-return; (c) move `snapshot_source` inside
  the `try` so any start-up failure is receipted; (d) preflight, before the first protected
  worker, `torch.cuda.is_available()`, `import gsplat` and `lpips.LPIPS(net="alex",
  pretrained=True)` with one disposable forward, so a missing evaluation dependency cannot burn the
  canonical run root after hours of fitting. The base interpreter's site-packages could not be
  listed in the review environment; even if a `realtime-gs` distribution exists there, (b)-(d)
  still apply.
- B2. `preprocessing.target_operator` states that field colours outside the training mask are
  "never supervised". That is exact only where the packed-alpha fraction is zero. At
  partial-coverage pixels the four-site box average contains outside-mask sites, and the Trainer
  supervises `target*mask + background*(1-mask)` with weight `0.1 + 0.9*mask`, so outside-mask
  field colour enters with weight proportional to the fraction. Replace the sentence with a precise
  statement of both regimes, and add that this boundary band is where the teacher families differ
  (contained teachers are black at outside sites, uncontained teachers extrapolate foreground,
  photographs contain room background); that difference is in-scope for H2 but must be named for
  the audit. No code change is needed.
- B3. `decision_policy.h2_teacher_containment` says "reverse inequality in every seed rejects".
  `gates()` implements the mirrored pair: reject iff for every paired seed `nb_rand_ms`
  foreground PSNR <= `mc_rand_ms` - 0.2 dB and `nb_rand_ms` outside_alpha_mass >= `mc_rand_ms`
  - 0.005; otherwise inconclusive. Write that condition into the protocol text so the frozen
  policy and the bound code agree exactly.

Optional improvements:

- R1. Audit hook robustness: compare `Path(...).resolve()` rather than `.absolute()` against the
  resolved frame directory so a symlinked dataset component cannot bypass the deny rules (the
  current dataset tree has no symlinks), and assert inside `publish()` that the run root equals
  `runs/<task_id>` so a future smoke that imports the report module cannot write
  `benchmarks/results/<task_id>_RESULT.*`.
- R2. In `prepare`, abort if any training pixel with soft alpha > 0 has decoded `coverage` < 1 for
  any family (the value is already returned by `decode_compact_view`) and record per-family
  coverage in `preparation.json`, so a fit window that clips the silhouette cannot silently
  handicap one teacher family.
- R3. Cross-check each compact view's `source.rgb` and `source.mask` SHA-256 metadata against the
  sealed photograph and mask digests in `prepare`, binding each teacher to the exact photograph it
  was fitted to.
- R4. `reset_cuda_peak_stats: true` makes the Trainer reset the CUDA peak after the driver's own
  reset, so each cell receipt's peak excludes pre-training allocations; either set it false as
  the 20260908 protocol did or say so in `resource_protocol.scope`. Descriptive only.
- R5. Evaluate gate inequalities in the written form (`a >= b - margin`) rather than on deltas, or
  state that ties are resolved on the delta; the difference is at most one ulp.
- R6. `preprocessing.training_masks` says the packed-alpha/source-mask equality is "checked once
  for the photograph arms"; the code checks every training view of the `no_boundary` family once
  in `prepare` on behalf of all arms. Align the sentence.
- R7. `gi_rand_ms` and `ph_hull_ms` enter no gate; if GPU time is constrained they could be
  dropped (6 of 21 cells) without touching H1-H4.
- R8. A fitting-worker timeout kill leaves no `receipt.json` for that cell; have the coordinator
  write a `timed_out` receipt so the failure report is complete.

After B1-B3: recompute `frozen_configuration.source_binding` (the driver changes), rerun
`review-digest`, and request a review of the new digest. Record this verdict in
`protocol_review` with status `blocked`; when a revised protocol is approved, retain this record
as `..._PROTOCOL_REVIEW_V1_REJECTED.md` following the existing convention.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`,
`initialize`, `fit` or `evaluate` commands, performed no GPU work, and did not enter
`.scratch/` or any smoke directory. No image, mask, `.rtgsv`, `.npz`, `.ply`, run or benchmark
outcome content was opened or printed; `validate-data` hashed sealed bytes opaquely. The prior
task's RESULT and AUDIT records were not read. No file was modified, no run root was created, no
checkpoint was selected, and the owner's task record was not written. This review has no outcome
access.
