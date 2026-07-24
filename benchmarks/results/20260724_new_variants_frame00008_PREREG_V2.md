# New opt-in variants on Janelle `frame_00008` — frozen protocol v2

Date frozen: 2026-07-24 (Europe/Berlin)

This document amends and otherwise incorporates the complete frozen protocol
[`20260724_new_variants_frame00008_PREREG.md`](20260724_new_variants_frame00008_PREREG.md),
SHA-256 `eb8e053d823462485f0c1e7b11f269aae0b0efbe25f8136c3a22e37824b14a22`.
Every arm, split, input, hyperparameter, metric threshold, interpretation gate, and scope
restriction in v1 remains frozen except for the explicit corrections below.

## Why v2 exists

The v1 command created:

- `runs/new_variants_frame00008_20260724/plan.json`, SHA-256
  `4184a61f0d8ef1a2a96568bfd79243a21ba892b593afb639403e3c6d5bd44e07`;
- `runs/new_variants_frame00008_20260724/failure.json`, SHA-256
  `95d5dad253eba9dd46e70ecc9d7a2e64c4c2f5fc2a1508a87dd6a09150cb7642`.

It stopped during reporting immediately after fitting the first baseline camera (`C0001`) because
the harness requested `image_metrics()["ssim_fg"]`. The metrics API exposes `ssim_crop`, not
`ssim_fg`. No fit, metric, preview, lift, refinement, or treatment artifact was saved; no numeric
Janelle outcome was printed or inspected. The failed directory is preserved and is not resumed or
overwritten.

## Frozen corrections

1. The stage-1 metric is named and read as `ssim_crop`, matching
   `rtgs.core.metrics.image_metrics`. The v1 phrase “foreground SSIM” is replaced by “foreground
   crop SSIM.” No loss, fit, render, arm, threshold, or model state changes.
2. The official output moves to the fresh directory
   `runs/new_variants_frame00008_20260724_v2`.
3. The corrected harness is `benchmarks/new_variants_frame00008.py`, SHA-256
   `f116a6125fe2d22e718a74157767e95ce597eaeb6aba454f7c7975b3b57ba002`.

The correction is reporting-only and was frozen before any treatment arm executed. The v2 run
must still reconstruct the baseline from scratch; it may not load state from the failed attempt.

## Gates

The same 50-test focused CUDA/mechanism command, Ruff check, Ruff format check, Python compilation,
and `git diff --check` pass after the correction.

## Official v2 command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/new_variants_frame00008.py \
  --protocol benchmarks/results/20260724_new_variants_frame00008_PREREG_V2.md \
  --out runs/new_variants_frame00008_20260724_v2
```

The viewer manifest path and v1 viewer command remain unchanged; the generated manifest must point
to the v2 output directory.
