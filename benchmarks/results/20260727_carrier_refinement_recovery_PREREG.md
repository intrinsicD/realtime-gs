# Full-resolution carrier experiment — fixed-topology recovery protocol

**Frozen:** 2026-07-27 after the parent v4 result and scientist-pass finding
**Evidence class:** explicitly post-outcome exploratory remediation; never confirmatory
**Parent summary SHA-256:**
`cfea7b1d77b3cfa02890750cded16e9b9d09304db7b93edc39cd222220b8343b`

## Finding being tested

The parent experiment's standard-density arms schedule a density event at step 160 and evaluate
immediately. The saved density receipts show non-empty final births for:

- `beam-standard`: 622 newborns;
- `carrier-schedule`: 1,877;
- `carrier-schedule-clone`: 1,364;
- `no-covariance`: 660;
- `no-opacity`: 1,415;
- `split-immediately`: 1,147;
- `random-rgb`: 1,237;
- `random-jpeg-q50`: 1,256;
- `means-only`: 1,166.

Those newborns have inherited parameters and zero recovery optimization, so the parent quality
endpoints cannot be treated as mature final models. `no-warmup` executes an empty step-160 event;
`clone-only` and `particle` last change topology at step 120 and already receive 40 subsequent
updates. They are nevertheless included in the uniform follow-up so every optimized arm receives
the same total update count.

## Frozen recovery

- Load, by exact SHA-256, each parent arm's saved `gaussians.ply`.
- Exclude `beam-only`, which performed no dense optimization and has no final density event.
- Apply exactly 40 full-resolution masked updates to all other 12 arms.
- Native resolution `5328×4608`; same train cameras `C0001`, `C0014`, `C0028`; same validation
  cameras `C0031`, `C1000`, `C1002`; report-only `C1001`, `C1004` remain unopened.
- Density is disabled: no clone, split, prune, or opacity reset.
- CUDA gsplat, packed mode, full SH, masks on, black background, ordinary L1 + D-SSIM loss.
- Segment coordinates: iteration offset 160, schedule length 200, seed `27187`.
- All parameter groups remain trainable. Adam is explicitly restarted because the parent bundle
  did not serialize optimizer state. This is a recovery/polish segment, not an exact continuation.
- `random-jpeg-q50` continues to train on the same second-generation JPEG-q50 images. Every other
  arm uses original full-resolution RGB.
- Primitive count must remain bit-exact through the segment.

## Metrics and interpretation

Recompute the parent saved PLY endpoint and the recovered endpoint using the parent's exact metric
definitions:

- median validation foreground PSNR;
- crop PSNR/SSIM;
- masked-crop AlexNet LPIPS at maximum side 512;
- alpha IoU/inside/outside;
- time and peak VRAM.

The parent-reload PSNR difference is an artifact-roundtrip check. Recovery deltas and rank changes
are descriptive. The follow-up may establish a usable mature endpoint for this development scene,
but cannot retroactively make the parent gate confirmatory or justify a default.

## Bound implementation

- `benchmarks/carrier_refinement_recovery.py` SHA-256:
  `8d9c653f62c78b432825ed10ac753d2b51516eb0e57042d7d29d90c2662041cf`
- parent harness SHA-256:
  `09a84a30f1783018c5963a701803ec5eae03efe07aee951870c69c89fbdce98f`
- parent summary SHA-256:
  `cfea7b1d77b3cfa02890750cded16e9b9d09304db7b93edc39cd222220b8343b`

## Command

```bash
PYTHONUNBUFFERED=1 .venv-cuda/bin/python benchmarks/carrier_refinement_recovery.py \
  --protocol benchmarks/results/20260727_carrier_refinement_recovery_PREREG.md \
  --parent runs/carrier_refinement_fullres_frame00008_20260727 \
  --out runs/carrier_refinement_fullres_frame00008_20260727_recovery
```
