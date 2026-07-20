# N78 Stage-1 fit visualization through gsplat

- Status: exploratory post-hoc visualization; no refit and no official-result mutation
- Source raw archive: `20260716T101608Z_cpu_stage1_fit_parameterization_RAW.npz`
- Source raw SHA-256: `028c93f350b30b61debebd5bf0706ff128f2c54faaee04614d1ee12191a3aeb7`
- Replay script: `benchmarks/visualize_stage1_fit_parameterization.py`
- Replay script SHA-256: `33464bd9e14d8f500a3e5295a9141fc0697fe85ad1b4b2189026e44d5e41f110`
- Machine-readable manifest: `20260716T101608Z_cpu_stage1_fit_parameterization_GSPLAT_REPLAY.json`
- Manifest SHA-256: `145df135bc8e5ff28552621ee242ad5c2218bc6529cb195875a650c349046f11`
- Compact overview SHA-256: `1107efe22441a8289e5ca455aab5a1ea6955383190d37ad10619a12ea9179029`
- All-view sheet SHA-256: `712c8bee320e334d615ed8ce8767ed6e4cdb69e45b2384e8b01a588014e507b1`

## What is shown

The two PNGs replay all 108 saved terminal fits at step 120 through the installed
GaussianImage `gsplat` 1.1.3 fork on an NVIDIA GeForce RTX 3050. The compact
sheet uses fixed `local_view == 0` for every seed; the exhaustive sheet shows all
54 paired source images. Both sheets use nearest-neighbor enlargement of the
original 48x48 images.

This is the closest semantics-preserving gsplat visualization available for this
Stage-1 result. It is **not** the repository's 3D `GsplatRasterizer`: the N78 fit
stores 2D additive weights/amplitudes but no depth or 3D alpha-compositing
semantics. Lifting it merely to invoke the 3D backend would introduce a new depth,
opacity, ordering, and compositing model.

The adapter instead uses the fork's 2D additive CUDA calls
`project_gaussians_2d_covariance` and `rasterize_gaussians_plus`. It maps the
saved Cholesky factor to covariance, shifts saved pixel-center coordinates by
-0.5 pixels for the CUDA sampling convention, and supplies saved built amplitude
with unit opacity. Projection uses `clip_coe=sqrt(12)`. The CUDA rasterizer still
drops contributions below 1/255, unlike the native renderer's exact `q < 12`
support, so byte-exact identity is neither expected nor claimed.

## Replay agreement

Across all 108 terminal images, gsplat-to-native agreement is:

| Comparison | Frame PSNR min / median / mean / max | Global mean abs. error | Max abs. error |
|---|---:|---:|---:|
| Display-clamped | 66.369 / 68.024 / 68.013 / 69.638 dB | 0.0001982 | 0.0044225 |
| Unclamped | 66.348 / 68.018 / 67.996 / 69.638 dB | 0.0001989 | 0.0044225 |

The small discrepancy is consistent with the known support-threshold difference.
This validates the replay for visualization, not backend parity in the repository's
modern 3D sense.

## Nine-view gsplat replay summary

| Block / seed | Current 9p mean PSNR | Candidate 8p mean PSNR | Candidate - current |
|---|---:|---:|---:|
| Appearance-only / 7727 | 20.533 dB | 18.848 dB | -1.685 dB |
| Appearance-only / 8837 | 20.219 dB | 18.282 dB | -1.937 dB |
| Appearance-only / 9941 | 19.883 dB | 18.120 dB | -1.763 dB |
| Joint Stage 1 / 10007 | 27.799 dB | 26.452 dB | -1.346 dB |
| Joint Stage 1 / 11003 | 28.403 dB | 26.676 dB | -1.727 dB |
| Joint Stage 1 / 12007 | 29.087 dB | 27.664 dB | -1.423 dB |

The visual replay preserves the official result's direction: the current 9p arm
is better for every seed. It authorizes no new default, 3D, throughput, memory,
or downstream claim.

## Command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python benchmarks/visualize_stage1_fit_parameterization.py
```
