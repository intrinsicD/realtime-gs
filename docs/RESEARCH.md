# State-of-the-art survey and reuse decisions

Compiled 2026-07-07 from three research sweeps (3DGS fitting/rendering; 2D gaussian image
representations; depth/lifting/carving). Each section ends with what this repo reuses.
License claims were verified against the repositories on the compile date — re-verify
before anything license-sensitive.

## 1. 3DGS fitting & rendering backends

- **gsplat** (Ye et al., JMLR 2025, [arXiv 2409.06765](https://arxiv.org/abs/2409.06765),
  [code](https://github.com/nerfstudio-project/gsplat), **Apache-2.0**, v1.5.x on PyPI).
  Independent CUDA reimplementation of 3DGS: `rasterization()` takes means/quats/scales/
  opacities/SH + viewmats/Ks and returns colors/alphas/meta (with `means2d`, `radii`,
  `depths` for densification). Flags we care about: `packed=True` (memory),
  `rasterize_mode="antialiased"` (Mip-Splatting AA), `absgrad=True` (AbsGS gradients),
  `render_mode="RGB+D"` (built-in depth). Ships densification as library code:
  `DefaultStrategy` (classic clone/split/prune/reset) and `MCMCStrategy` (3DGS-MCMC).
  Installs without a GPU; only kernel calls need CUDA. Native 2DGS entry point exists
  (`rasterization_2dgs`). ~10-15% faster and up to 4x less memory than the INRIA code at
  equal quality.
- **INRIA 3DGS** (Kerbl et al., SIGGRAPH 2023, [arXiv 2308.04079](https://arxiv.org/abs/2308.04079),
  [code](https://github.com/graphdeco-inria/gaussian-splatting)) — **non-commercial
  license that virally taints forks** (3DGS-MCMC, AbsGS, Speedy-Splat, RAIN-GS repos are
  all INRIA-derived). Use only as a numbers/behavior reference; never vendor code.
- **Densification SOTA**: 3DGS-MCMC (Kheradmand et al., NeurIPS 2024,
  [2404.09591](https://arxiv.org/abs/2404.09591)) replaces heuristics with relocation +
  noise under a fixed budget and is notably **less init-sensitive** — relevant since our
  init is not SfM; reusable via gsplat's Apache `MCMCStrategy`. AbsGS
  ([2404.10484](https://arxiv.org/abs/2404.10484)): absolute-value gradient accumulation
  fixes gradient collision (gsplat `absgrad=True`). Revising Densification (Bulò et al.,
  ECCV 2024, [2404.06109](https://arxiv.org/abs/2404.06109)): per-pixel-error criterion,
  corrected clone opacity (gsplat `revised_opacity`). Taming 3DGS
  ([2406.15643](https://arxiv.org/abs/2406.15643)): budgeted score-based densification +
  fused SSIM + per-splat backward (perf parts MIT, upstreamed into gsplat).
- **Training-speed SOTA**: the recurring ingredients are fused SSIM
  ([rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim), MIT), per-gaussian
  sparse Adam, tighter tile culling (Speedy-Splat's SnugBox/AccuTile — adopted by gsplat),
  resolution/frequency schedules (DashGaussian, [2503.18402](https://arxiv.org/abs/2503.18402),
  ~200 s/scene; FastGS [2511.04283](https://arxiv.org/abs/2511.04283), ~100 s), and strict
  primitive budgets. Faster-GS (CVPR 2026, [2602.09999](https://arxiv.org/abs/2602.09999))
  consolidates best practices. **The lever no one fully exploits is initialization — our
  target.**

## 2. Images as 2D gaussians (stage 1 foundations)

- **GaussianImage** (Zhang et al., ECCV 2024, [2403.08551](https://arxiv.org/abs/2403.08551),
  [code](https://github.com/Xinjie-Q/GaussianImage), **MIT**): 8 params/gaussian —
  position, **Cholesky factor (l11, l21, l22)** of the 2D covariance, weighted color.
  **Accumulated summation** blending (no sorting, no alpha compositing) is
  order-independent and beats alpha blending by **+0.8 dB** while being faster. Kodak
  768x512: 70k gaussians → 44.1 dB in ~107 s (V100, 50k Adan steps); 30k → 38.6 dB.
  Loss ablation: **plain L2 beats L1/L1+SSIM/L2+SSIM for PSNR**. Rendering ~2000 FPS.
- **Image-GS** (Y. Zhang et al., SIGGRAPH 2025, [2407.01866](https://arxiv.org/abs/2407.01866),
  [code](https://github.com/NYU-ICL/image-gs), **MIT**): content-adaptive allocation —
  positions sampled from **gradient magnitude mixed ~70/30 with uniform**, then
  **error-guided progressive addition** (start N/2, add N/8 every 500 steps at max-error
  pixels); reaches ~95% of final quality within ~400 steps. Loss L1 + 0.1·SSIM.
- **Feed-forward fitting**: Instant-GI (ICCV 2025, [2506.23479](https://arxiv.org/abs/2506.23479),
  MIT) predicts a full 2D gaussian set in one pass (~10x less wall-clock than optimizing);
  Fast 2DGS ([2512.12774](https://arxiv.org/abs/2512.12774)) similar. Future stage-1
  speedup path.
- **"StructSplat" clarification**: the only paper by that name
  ([2606.28321](https://arxiv.org/abs/2606.28321)) is *feed-forward 3D reconstruction from
  uncalibrated sparse views* — not a 2D image representation. Closest 2D works:
  Structure-Guided Allocation ([2512.24018](https://arxiv.org/abs/2512.24018), gradient-aligned
  orientation regularization) and SGI (CVPR 2026, [2603.07789](https://arxiv.org/abs/2603.07789)).
- **Do not vendor**: LIG (GPL-3.0), MiraGe (INRIA non-commercial license via GaMeS).
  MiraGe ([2410.01521](https://arxiv.org/abs/2410.01521)) is conceptually interesting:
  flat gaussians in 3D rendered by the 3DGS renderer for a single image.
- **Budget rule of thumb** (from GaussianImage/Image-GS numbers): ~15-30k gaussians per
  512x512 image for 36-40 dB; fitting seconds/image on GPU with error-driven init and
  early stopping.

## 3. Depth estimation & feed-forward geometry (variant B backends)

| Model | Output | License | Integration |
| --- | --- | --- | --- |
| **Depth Anything V2 Small** ([2406.09414](https://arxiv.org/abs/2406.09414)) | relative inverse depth | **Apache-2.0** (Small only! B/L are CC-BY-NC) | HF `transformers` pipeline — our default real backend |
| **MoGe-2** (Microsoft, [2507.02546](https://arxiv.org/abs/2507.02546)) | **metric point maps** + normals | **MIT** | pip from GitHub; ~60 ms/img; ships ROE scale/shift solvers — best upgrade target |
| **Metric3D v2** ([2404.15506](https://arxiv.org/abs/2404.15506)) | metric depth + normals | **BSD-2** | one-line torch.hub |
| Depth Pro (Apple, [2410.02073](https://arxiv.org/abs/2410.02073)) | metric + focal | research-only weights | reference/eval only |
| UniDepth v2 ([2403.18913](https://arxiv.org/abs/2403.18913)) | metric + intrinsics + uncertainty | CC-BY-NC | blocked for reuse |
| **Depth Anything 3 Small/Base** ([2511.10647](https://arxiv.org/abs/2511.10647)) | multi-view depth+ray | **Apache-2.0** (S/B) | multi-view alternative |
| **MapAnything** (Meta, [2509.13414](https://arxiv.org/abs/2509.13414)) | metric multi-view + cameras | code Apache; one **Apache model** | best permissive multi-view backend |
| VGGT (CVPR 2025 best paper, [2503.11651](https://arxiv.org/abs/2503.11651)) | cameras+depth+pointmaps, <1 s many views | commercial checkpoint gated | candidate fourth lifter |
| DUSt3R/MASt3R/Fast3R/CUT3R | pointmaps | all non-commercial | avoid in license-sensitive work |

**Depth-to-covariance in feed-forward gaussian methods**: pixelSplat (MIT,
[2312.12337](https://arxiv.org/abs/2312.12337)) sets `scale = bounded_factor * depth *
||K^-1 pixel||` — the **z/f pixel-footprint prior** (verified in its gaussian_adapter);
SplaTAM ([2312.02126](https://arxiv.org/abs/2312.02126)) uses isotropic radius z/f;
MVSplat (MIT), DepthSplat (**Apache-2.0**, [2410.13862](https://arxiv.org/abs/2410.13862),
DA-V2 + cost volume) and Flash3D predict scales by network around depth-placed centers.
No published closed form for the **along-ray** dimension — ours (below) fills that gap.

**Scale alignment of relative depth**: closed-form least-squares (s, b) against projected
SfM points (Chung et al. [2311.13398](https://arxiv.org/abs/2311.13398); the official 3DGS
`make_depth_scale.py` does exactly this; MoGe's ROE solver is a permissive
implementation). Correlation losses (FSGS/SparseGS) sidestep alignment but only as a
training loss. Implemented in `rtgs/depth/align.py`.

## 4. Initialization literature (closest related work)

- **EDGS** (CompVis, CVPR 2026, [2504.13204](https://arxiv.org/abs/2504.13204),
  [code](https://github.com/CompVis/EDGS)) — triangulates dense 2D correspondences (RoMa)
  into a one-shot dense init and **disables densification entirely**; reaches 3DGS LPIPS
  in 25% of training time. **Closest published work to our idea** — but its per-image
  unit is a point match; ours is a fitted 2D gaussian carrying covariance + color.
- **InstantSplat** (NVIDIA, [2403.20309](https://arxiv.org/abs/2403.20309)) — MASt3R
  pointmaps + short joint pose/gaussian optimization, no densification (non-commercial
  stack). The "foundation-model init + short joint refine" pattern is worth copying on a
  permissive stack.
- **Init sensitivity**: "Does 3DGS need SfM init?" ([2404.12547](https://arxiv.org/abs/2404.12547))
  — good random init in the right bounding volume closes much of the gap; RAIN-GS
  ([2403.09413](https://arxiv.org/abs/2403.09413)) — splats overfit in place instead of
  relocating (a failure our dense near-surface init sidesteps). Desiatov & Sattler
  ([2603.20714](https://arxiv.org/abs/2603.20714)): **current densifiers cannot fully
  exploit dense inits** — both a warning and our opportunity; their benchmark is reusable.
- **Feed-forward per-pixel gaussians** (Splatter Image, pixelSplat, MVSplat, GS-LRM,
  AnySplat, Flash3D, DepthSplat): networks predicting dense pixel-aligned gaussians.
  Conceptually "per-view gaussians → 3D", but none uses a *fitted, sparse* 2D
  representation.
- **Novelty check (2026-07-07)**: no published work fits 2D gaussian splats per image and
  lifts those primitives into a 3DGS initialization. Nearest neighbors: EDGS, MiraGe,
  Splatter Image. Re-verify at publication time.

## 5. Visual hull & space carving (variant C foundations)

- Classics: Laurentini's visual hull (TPAMI 1994); **Seitz-Dyer voxel coloring**
  (CVPR 1997) — keep voxels whose projected colors are consistent across views;
  **Kutulakos-Seitz space carving** (IJCV 2000) — the photo hull as the maximal
  photo-consistent shape.
- **torchhull** ([code](https://github.com/vc-bonn/torchhull), **MIT**, pip) — CUDA visual
  hull from masks via sparse voxel octrees; drop-in when masks exist.
- **GaussianObject** (SIGGRAPH Asia 2024, [2402.10259](https://arxiv.org/abs/2402.10259))
  — initializes 3DGS from a visual hull of ~4 masked views; proof hull-init beats sparse
  SfM in the few-view regime. Nobody has shipped **photo-consistency carving as a 3DGS
  initializer for unmasked scenes** — open space our `carve` variant occupies (using 2D
  gaussian coverage as a learned silhouette substitute).

## 6. Gaussian merging (mixture reduction)

Moment-preserving merge of components (w_i, mu_i, Sigma_i):
`w = sum w_i; mu = sum w_i mu_i / w; Sigma = sum w_i (Sigma_i + (mu_i-mu)(mu_i-mu)^T) / w`.
**Runnalls 2007** gives a cheap KL-bound merge cost for greedy pairwise reduction.
**Hierarchical 3DGS** (Kerbl et al., SIGGRAPH 2024) is the production precedent for
merging splats: weights ∝ opacity × splat size, opacity renormalized after merging.
LightGaussian ([2311.17245](https://arxiv.org/abs/2311.17245)) significance scores and
Reduced-3DGS redundancy tests are useful merge-candidate selectors. Implemented in
`rtgs/lift/merge.py` (voxel-hash grouping + moment matching, opacity 1-prod(1-a)).

## 7. Evaluation protocol (for GPU-scale runs, roadmap M2)

- Datasets: Mip-NeRF 360 (7 public scenes, every-8th-image test split, outdoor at 1/4
  resolution, indoor 1/2), Tanks&Temples (truck, train), Deep Blending (drjohnson,
  playroom), NeRF-Synthetic for sanity.
- Metrics: PSNR, SSIM, **LPIPS-VGG** (state the variant!), plus wall-clock, peak VRAM,
  final gaussian count, FPS. Checkpoints at 7k/30k iterations; for a speed paper the
  headline is **time-to-quality curves** (e.g., time to reach 3DGS-30k LPIPS).
- Reference wall-clocks (consumer GPU, per 360 scene): INRIA 3DGS ~25-40 min to 30k;
  gsplat ~19 min; accelerated ~10-15 min; DashGaussian ~200 s; FastGS ~100 s; EDGS ~25%
  of baseline. Consider [nerfbaselines](https://github.com/nerfbaselines/nerfbaselines)
  for reproducible comparisons.
- Baselines to beat: SfM-init 3DGS (gsplat Default/MCMC), EDGS, InstantSplat,
  DashGaussian/FastGS.

## 8. Decisions adopted in this repo

1. **Rasterizer**: gsplat (Apache-2.0) as the GPU backend behind `rtgs.render.base`;
   pure-PyTorch reference renderer defines semantics and keeps CPU CI honest. No INRIA
   code anywhere.
2. **Stage 1**: GaussianImage Cholesky parametrization + accumulated summation, with the
   amplitude factored as `weight * color`. Accumulated amplitude is **not identifiable as
   alpha opacity**, so lifting uses an independent low opacity prior and fuses repeated
   observations without union-inflating it. L2 loss; gradient-magnitude init at 70/30 mix
   (Image-GS). Error-driven progressive addition is roadmap M3.
3. **Missing-dimension covariance** (variant B): lateral `Sigma_lat = (z/f)^2 * Sigma_2D`
   (pixelSplat's footprint prior generalized to anisotropic gaussians); along-ray
   `sigma_ray^2 = grad(D)^T Sigma_2D grad(D) + (z/f)^2 s_min^2` (depth slant + footprint
   floor), clamped to [0.1 x lateral min, 3 x lateral max]. Implemented in
   `rtgs/lift/base.py`.
4. **Depth backends**: GT/mock for tests; Depth Anything V2 **Small** (the Apache one)
   as the first real backend, always through scale/shift alignment; MoGe-2 (MIT, metric
   point maps) is the planned upgrade; Metric3D v2 optional.
5. **Variant C**: coverage-based hull test (2D gaussian weight maps as soft silhouettes)
   + Seitz-Dyer color-consistency scoring on a dense grid; ray-tunnel argmax placement;
   Runnalls/Hierarchical-3DGS moment-matched merging.
6. **Refinement**: classic 3DGS recipe with screen-space-gradient density control on the
   reference stack; on GPU, gsplat strategies (MCMC first — least init-sensitive) are the
   intended replacements. Given EDGS and Desiatov-Sattler, evaluate with densification
   *disabled or shortened* — that is where the speed win lives (roadmap M3).
7. **Evaluation**: synthetic ground-truthed scenes for CI-scale claims; §7 protocol for
   real claims (roadmap M2).
