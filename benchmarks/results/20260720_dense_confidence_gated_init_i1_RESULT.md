# I1 result: correspondence-confidence gate

Date: 2026-07-20
Disposition: implementation/count reproduction passed; quality remains exploratory

## Frozen input and classifier

- Bundle: `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`
- Manifest SHA-256:
  `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`
- Seed: `0`; merge voxel: `0.06`; dense lifted rows: `3,488`; merged clusters: `2,319`
- Classifier: view multiplicity `>=2`, RMS spread `<=0.50` voxel, maximum half-max
  width `<=0.20`, minimum covered views `>=2`, maximum reprojection residual
  `<=16 px`.
- Preregistration:
  [`20260720_dense_confidence_gated_init_i1_PREREG.md`](20260720_dense_confidence_gated_init_i1_PREREG.md)

## Implementation result

The CPU-first gate retained exactly the preregistered 60 clusters and dropped
2,259. It emits one typed record per merged cluster, an exact keep mask,
failure reasons, failure counts, and signal quantiles. The constructed fixture
keeps a co-located three-view target and rejects a single-view decoy. The
benchmark remains unchanged unless `--gate` is passed.

Failure counts overlap because one cluster can fail several conditions:

| Condition | Clusters failing |
|---|---:|
| Source-view multiplicity | 1,819 |
| Maximum reprojection residual | 633 |
| Maximum half-max width | 603 |
| RMS spatial spread | 75 |

## Exploratory init-only screen

These seven views selected and parameterized the classifier. They are not
held-out evidence and cannot be used to retune the frozen thresholds.

| Initialization | Gaussians | Mean foreground PSNR | Mean SSIM |
|---|---:|---:|---:|
| Balanced top-K | 172 | 18.1699 dB | 0.760626 |
| Dense-all + merge | 2,319 | 20.1376 dB | 0.632195 |
| Easy-only gate | 60 | 18.6204 dB | 0.762342 |

Easy-only minus top-K is `+0.4505 dB` mean foreground PSNR and
`+0.001716` mean SSIM at `0.349x` the primitive count. Every training view is
positive; the per-view foreground PSNR deltas are:

`+0.3376, +0.3677, +0.4241, +0.6178, +0.4162, +0.4357, +0.5544 dB`.

This supports running the already-preregistered three-arm E2 experiment; it
does not show downstream recovery, held-out generalization, or a default win.

## Runtime and progress

The calibrated GPU replay, including top-K, dense-all, and easy-only
evaluation, completed in `64.95 s` wall time with `1,942,228 KiB` peak RSS.
Stage timings were:

| Stage | Seconds |
|---|---:|
| top-K placement | 0.788 |
| top-K evaluation | 20.765 |
| dense placement | 0.721 |
| dense merge | 0.010 |
| confidence gate | 0.587 |
| dense evaluation | 20.170 |
| easy-only evaluation | 20.473 |

Progress is callback-based and silent by default. The benchmark prints
throttled placement batches, calibrated-view completions, visible-Gaussian
counts, per-view time, and total elapsed time. The Torch reference can
additionally report row-chunk progress without timing or allocation overhead
when no callback is installed.

## Artifacts and reproduction

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/usr/bin/time -v .venv/bin/python benchmarks/compact_init_eval.py \
  --bundle runs/compact_masked_bundle_640_20260717/reconstruction_inputs \
  --out runs/dense_confidence_gated_init_i1_20260720 \
  --seed 0 --rasterizer gsplat --device cuda --gate
```

SHA-256:

- `init_eval.json`:
  `9980d91536a622808acd33076c7325707385d160d5c375363cafbc24d60986c4`
- `init_topk.ply`:
  `d83ee1e764ee6bc0d1cf7696e848df91b0a92d33ad5c9932c9e1138e8564e9fb`
- `init_dense_merged.ply`:
  `56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7`
- `init_easy_gated.ply`:
  `1d3205755d67e6e3badd48a9d41a1329a38898e6e6178150cac25aadc57b6a9f`

Viewer:

```bash
.venv/bin/rtgs view \
  --gaussians runs/dense_confidence_gated_init_i1_20260720/init_easy_gated.ply \
  --initial runs/dense_confidence_gated_init_i1_20260720/init_topk.ply \
  --host 127.0.0.1 --port 8774 --no-open
```

The viewer returned HTTP `200` with `2,888,259` bytes.
