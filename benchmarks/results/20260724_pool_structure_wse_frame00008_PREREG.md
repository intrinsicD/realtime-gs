# Pooled structure-tensor WSE ablation on Janelle `frame_00008` — frozen protocol

Date frozen: 2026-07-24 (Europe/Berlin)

Status at freeze: no calibrated outcome from the new no-WSE arm has been rendered, measured, or
inspected. Synthetic/unit mechanism checks only have run.

## Question

Conditional on the fixed-capacity pool/free-list stage-1 policy, does anisotropic Weighted Sample
Elimination (WSE) add value to structure-tensor initialization, or is density sampling plus the
same tensor-oriented covariance sufficient?

This is a direct WSE ablation under pooling, not a test of the WSE×pool interaction: every arm is
pooled, so no claim about whether WSE behaves differently without pooling is authorized.

## Matched no-WSE control

`StructureInitConfig.sampling_mode` is an opt-in research seam:

- `wse` is the unchanged default: draw `ceil(4N)` points from the structure-energy density, then
  apply anisotropic crowding elimination to retain exactly `N`;
- `density` draws the identical `ceil(4N)` candidate stream from the identical RNG state but keeps
  its first `N` points.

The two structure arms therefore keep the structure tensor, density PMF, candidate stream,
orientation, coherence, covariance equation, sampled color, count, pool allocation/triage,
optimizer, loss, and schedule fixed. Only the final candidate subset differs. The density control
is not proposed as a default.

Mechanism tests frozen before the calibrated run:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q \
  tests/test_structure_init.py \
  tests/test_image2gs_init_strategy.py \
  tests/test_image2gs_pool.py
................................... [100%]
```

## Arms

All arms fit 640 live Gaussians in a 1,280-row pool for 300 native-CUDA iterations.

1. `pool-gradient`: the previously positive pooled gradient-magnitude initializer; anchor.
2. `pool-structure-density`: structure tensor and oriented covariance, matched density-prefix
   placement without WSE.
3. `pool-structure-wse`: structure tensor and oriented covariance with default anisotropic WSE.

Shared stage-1 settings:

```text
n_gaussians=640, max_gaussians=None, iterations=300
backend=native, native_renderer=cuda, lr=0.01
appearance_parameterization=weight_color_9p
pool=True, pool_capacity=1280
pool_triage_every=50, pool_prune_count=32, pool_spawn_count=32
convergence_patience=0, row_chunk=64
```

Seeds are `0..6`, paired by training camera.

## Calibrated data and split

- Raw capture:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
- Calibration:
  `dataset/2025_03_07_stage_with_fabric/calibration_dome.json`.
- Loader: downscale 16, maximum eight evenly selected images, undistortion and masks enabled,
  `test_every=8`.
- Frozen order:
  `C0001,C0008,C0014,C0021,C0026,C0031,C0039,C1004`.
- Train-only:
  `C0001,C0008,C0014,C0021,C0026,C0031,C0039`.
- Reporting-only held-out camera: `C1004`.

`C1004` may not be fitted, lifted, sampled during refinement, used for checkpoint selection, or
used to change this protocol. It is evaluated only at the frozen initialization and final
endpoints.

## Common lift and refinement

Each arm uses the same train-only carve lift:

```text
grid_res=32, bounds_scale=0.5, min_views=2, hull_fraction=0.85
color_std_sigma=0.20, color_match_sigma=0.35
coverage_thresh=0.40, samples_per_ray=48
min_score=0.05, min_weight=0.05, merge=True, merge_voxel_scale=1.0
init_opacity=0.10, sh_degree=0
```

Each arm then receives 2,000 CUDA/gsplat steps with unpacked antialiased RGB+D rendering, random
backgrounds, masks, `outside_alpha_lambda=0.01`, `mask_alpha_lambda=0.05`, target SH degree 3,
and the common gsplat DefaultStrategy/density settings from the 2026-07-24 single-factor run.
Density surgery may occur only through step 1,000; at least 1,000 recovery steps must follow the
last surgery. Final endpoints are used; no checkpoint selection occurs.

## Metrics and frozen gates

Stage-1 metrics are equal-camera means over the seven training fits: foreground/crop PSNR,
crop SSIM, masked-full PSNR, coverage IoU at 0.1, inside/outside mean coverage, foreground support
recall, and count. Pairwise wins are computed on per-camera foreground PSNR.

The primary WSE-under-pool stage-1 claim passes only if `pool-structure-wse` versus
`pool-structure-density` has all of:

1. mean foreground PSNR delta at least `+0.10 dB`;
2. foreground PSNR wins on at least 5/7 cameras;
3. mean outside-coverage ratio at most `1.10`.

Each structure arm is also tested independently against `pool-gradient` with the same three
clauses. These gates are separate; a secondary metric cannot rescue a failed primary gate.

Downstream metrics are equal-camera train means plus the single held-out `C1004` endpoint:
foreground/crop/full PSNR, SSIM, alpha IoU, inside/outside alpha, and primitive count.

The primary WSE-under-pool downstream claim passes only if WSE versus density has all of:

1. held-out foreground PSNR delta at least `+0.10 dB`;
2. held-out alpha-IoU delta at least `-0.01`;
3. train foreground PSNR delta at least `-0.25 dB`.

Each structure arm is independently tested against `pool-gradient` with the same clauses. A
combined method may be called a balanced positive treatment only if its own stage-1 and downstream
gates both pass. Because this is one scene, seed, and held-out camera, passing authorizes only a
development observation and a replication experiment—not a default change.

## Artifacts and visual handoff

Official output:

```text
runs/pool_structure_wse_frame00008_20260724
```

The run must preserve the plan, all 21 stage-1 NPZ/history pairs, initial/final NPZ+PLY models,
training histories, exact metric records, a `C0014` stage-1 sheet, train and held-out
reconstruction sheets, per-arm calibrated-path and novel-view animations, and a three-method
initial/final viewer manifest.

Official command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG.md \
  --out runs/pool_structure_wse_frame00008_20260724
```

Planned CPU viewer:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8785 --no-open
```

## Source binding at freeze

- Git revision: `7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11`.
- Harness:
  `benchmarks/pool_structure_wse_frame00008.py`,
  SHA-256 `47b91360df248cfe7d72ea22f70db02ddf37d515e79ae778e06e2a8a5047d07a`.
- Shared artifact runner:
  `benchmarks/new_variants_frame00008.py`,
  SHA-256 `d0f429352a28bdb1584cc30ff9b92a7a70b94c168966a19e4785876ea7cc1e8c`.
- Fit integration:
  `src/rtgs/image2gs/fit.py`,
  SHA-256 `4bcf90764277fc35c5ad10023e57103378764d428dde637e28a6c6e1fa75bd55`.
- Structure initializer:
  `src/rtgs/image2gs/structure_init.py`,
  SHA-256 `dc94c46873cfb58729b85120f040fed966b4386c14aa793c686d87cae9503bad`.
- Pool:
  `src/rtgs/image2gs/pool.py`,
  SHA-256 `cad33b6def38e1c43011ae61b07292cb370ab876dc20773cb122f5fc96835999`.
- Prior single-factor summary, context only:
  `runs/new_variants_frame00008_20260724_v3/summary.json`,
  SHA-256 `f302b6eaaae6eac8dd7e0894b371f8860df03d047ebb73e037e72ee90be166e9`.

The working tree contains the immediately preceding experiment handoff and this new opt-in seam.
The official plan must bind the full git status/diff plus every executed source and input hash.
Timings and peak allocations are non-decisional because the GPU is not reserved or repeated.
