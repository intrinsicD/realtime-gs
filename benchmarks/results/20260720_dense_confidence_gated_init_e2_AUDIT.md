# Scientist audit: E2 easy-only seed plus density control

Date: 2026-07-20
Audited result:
[`20260720_dense_confidence_gated_init_e2_RESULT.md`](20260720_dense_confidence_gated_init_e2_RESULT.md)
Disposition: **accept negative E2 result; reject broader incapability claim**

## Claim inventory

| Claim | Evidence | Audit disposition |
|---|---|---|
| Easy-only loses the frozen E2 gate | Late-release C1004 metrics + control repeat | Accepted |
| Dense-all is best on C1004 under this 300-step schedule | Replayed saved PLY metrics | Accepted |
| Easy-only uses fewer primitives and slightly less native time | Histories/counts | Accepted |
| Density control can never recover the hard set | One short schedule; easy still rising | Rejected |
| The deficit is caused by hard-dropped regions | No held-out spatial localization | Rejected/not measured |
| I2/E3 is unlocked | Required localization absent | Rejected; remains closed |

## Chronology and split integrity

- The E2 schedule, split, cap, seeds, and decision arithmetic were frozen
  before any real optimizer arm started.
- Read-only preflight found the strict compact bundle had no explicit bounds
  hint. Before training, the preregistration and harness were corrected to use
  the exact E1 seven-camera fallback. No outcome had been opened.
- Source manifest hashes every selected RGB/mask and calibration file.
- Training/validation tensors contained only the seven named optimization
  views plus C1002. C1004 had only its path/byte hash bound before the
  late-release boundary.
- `PREHELDOUT.json` records all four completed final/history hashes, contains
  `heldout_materialized=false`, and contains no `C1004` string.
- All three main arms have an identical 300-entry sampled-view sequence
  (SHA-256
  `b05027c6e1a9291813736b7c7d65579d1c31331f664cf502f678faaaaab398c4`);
  every sampled index is in `0..6`.

## Independent verification

- Raw result SHA-256:
  `1990a5e9510e83da5a94f5d8684700149e6bba6e77bba9eee0960fef5bf91e32`.
- Pre-heldout receipt SHA-256:
  `f69e179247a56b9ad72bf756359d26bfedb4891c5c72545a9bc48d7fab79232e`.
- Every history contains 300 finite losses and nine density events.
- Observed checkpoint maxima were 205 top-K, 2,319 dense-all, and 1,229
  easy-only; no arm exceeded the frozen 2,319 cap.
- Independent decision recomputation gives control envelope/tolerance
  `0.0070991516 dB` and easy-only minus dense-all
  `-2.1746625900 dB`.
- Replaying all four saved final PLYs on C1004 reproduces foreground PSNR
  exactly; crop SSIM differs by at most one float32 last-place rounding.
- The CPU classic-density smoke passed and a separate synthetic CUDA smoke
  exercised two DefaultStrategy growth events (4→8→16).
- Calibrated viewer smoke returned HTTP `200`.

## Confounds and narrow interpretation

1. The experiment matched a hard primitive cap, not final cardinality.
   Dense-all had the full 2,319 parameters from entry; easy-only reached only
   1,229 by step 300.
2. Easy-only's validation quality and count were rising sharply at the final
   checkpoint. Optimization horizon and allocation rate remain live
   alternatives to a correspondence-causality explanation.
3. This is one frame, one main seed, one control repeat, one strategy, and one
   1/8-resolution schedule.
4. Native time includes fixed validation checkpoints but excludes progress
   callback work. The 0.107 s easy-vs-dense difference is not a robust speed
   claim without timing repeats.
5. Exact compact-teacher evaluation is at the original fit-window resolution,
   while RGB optimization/validation/held-out metrics are at 1/8 resolution.
6. Dense/easy initial viewer copies are semantic PLY round-trips, not
   byte-identical copies of their source PLYs; the source hashes are separately
   bound in `source_manifest.json`.
7. The absolute source frame is external to the repository. Its exact inputs
   are hash-bound, but another machine needs those bytes to reproduce E2.

## Decision

Accept the negative preregistered E2 outcome. Keep balanced top-K as the
default. Do not open I2/E3 or claim a hard-correspondence failure without a
new, separately preregistered spatial-localization test. A longer
budget-filling control is scientifically motivated but would be a new
experiment, not a reinterpretation of E2.

Repository-wide verification and final docs-sync must pass before merge.
