# Architecture

## Dataflow

```
                    ┌────────────────────────────────────────────────┐
 images  ─────────► │ stage 1  rtgs.image2gs                         │
 (per view)         │   fit.py: configurable compact start          │
                    │   structsplat_backend.py: residual growth     │
                    │   renderer2d.py: differentiable accumulated    │
                    │   splatting (no sorting, GaussianImage-style)  │
                    └───────────────┬────────────────────────────────┘
                                    │ Gaussians2D per view (xy, cholesky cov, color, weight)
                    ┌───────────────▼────────────────────────────────┐
 cameras ─────────► │ stage 2  rtgs.lift   (five variants)           │
 (COLMAP, JSON,     │   gradient.py: bounded ray depth+rot+scale     │
  or synthetic)     │     anchors + fixed-pair position consistency  │
                    │   matching.py: train-only fixed-pair backends  │
                    │   surface.py: oriented backends + targets/loss │
                    │   depth.py: aligned depth + selectable         │
                    │     surface / footprint / isotropic covariance │
                    │   hybrid.py: depth seed + bounded-ray descent   │
                    │   carve.py: voxel color-consistency carving,   │
                    │     ray-tunnel placement, moment-match merging │
                    │   field_lifter.py: image-free field placement, │
                    │     fiber refit + transactional topology moves │
                    └───────────────┬────────────────────────────────┘
                                    │ Gaussians3D (means, quats, scales, opacity, SH)
                    ┌───────────────▼────────────────────────────────┐
                    │ stage 3  rtgs.optim                            │
                    │   trainer.py: mask-aware L1+D-SSIM, full SH    │
                    │   density.py: CPU classic clone/split/prune    │
                    │   strategies.py: gsplat Default or MCMC        │
                    │     (relocation/noise), hard budget            │
                    └───────────────┬────────────────────────────────┘
                                    ▼
                          refined Gaussians3D (.ply / .npz)
```

The established end-to-end path above still passes converted `Gaussians2D` and `SceneData`.
The registered research variant `field` has a separate native image-free entry:
`SceneFits` → `run_field_pipeline` / `FieldLifter.fit`. Stage 1 can freeze each live StructSplat
result as `GaussianObservationField`; `SceneFits` binds ordered fields to cameras, preserves
optional `PackedAlpha`, and requires explicit complete disjoint train/held-out indices. Strict bundle
loading enforces exact keys, bounded archives, ordinary contained files, and a restricted
identifier grammar. A generic caller can still retain the original `SceneData`, so the schema
alone is not process-level erasure; the calibrated research harness additionally runs compact
initialization, training, and sampled teacher evaluation in fresh workers that deny dataset,
PIL, calibrated-loader, and `SceneData` access.

Repository datasets may cross that boundary permanently through one capped `.rtgsv` file per
camera. `CompactView` stores the exact teacher NPZ byte-for-byte, the calibrated camera, source
and calibration digests, and optional lossless crop-local bit-packed alpha in a strict ZIP whose
complete size is at most 168,000 decimal bytes. `CompactDataset` verifies the ordered frame
manifest and converts directly to `ReconstructionInputs`; alpha remains separately available.
`scripts/convert_datasets_to_gaussians2d.py` performs the CUDA fitting resumably and removes
source `rgb/`/`mask/` directories only after a global strict-reload, source-rehash, camera,
configuration, provider, byte-cap, and inventory gate succeeds.

The standalone CPU `CompactCarveInitializer` consumes this bundle for sampled ray-tunnel
initialization. Its optional prebuilt observation indexes are reusable by `CompactTrainer` and
are rejected unless their tile size, total entries, and maximum candidates satisfy the configured
caps. `TorchPointRasterizer` evaluates explicit image coordinates while preserving the dense
reference renderer's camera-wide visible set, global depth order, and alpha compositor. Separate
continuous-area and discrete-pixel proposals use fixed-attempt importance correction. The audited
four-arm synthetic experiment found no global sampling win: the Gaussian discrete-pixel proposal
was `NEUTRAL_OR_NEGATIVE`, and the continuous-area proposal was only `NONINFERIOR`. The compact
trainer preflights the original inputs before transferring the teacher/camera working set or
initialization to the configured device, then constructs a teacher/camera-only device-tensor working
set that omits optional points, visibility, and bounds tensors; non-CPU tile-overlap estimates use
bounded device-native chunks. All teachers still remain resident and
the Torch backward retains approximately one outer microbatch times the visible Gaussian count.
The sampled `CompactTrainer` path remains fixed-topology and research-only; it is distinct from
the variable-topology `FieldLifter` and is not wired into the production CLI/pipeline. Neither
path establishes a reconstruction-quality, performance, density-control, or default claim.

## Subpackages

| Package | Responsibility |
| --- | --- |
| `rtgs/core` | Shared math and containers: `gaussians2d` (xy, Cholesky cov, color, weight), `gaussians3d` (means, quats, log-scales, opacity, SH; PLY/NPZ IO), `observation2d` (lossless frozen StructSplat field semantics, CPU-reference-equation-matched dense/indexed point queries, separate continuous-area and discrete-pixel rejection-thinned proposals, fixed-attempt importance correction, and integrity-checked NPZ IO), `camera` (COLMAP-convention pinhole, project/unproject/rays), `sh` (real spherical harmonics deg ≤ 3, explicit shifted-color preactivation, standard hard nonnegative floor, and opt-in SMU-1/negative-gradient research controls), `metrics` (full, foreground, and foreground-crop PSNR/SSIM; separable differentiable SSIM). `observation2d_cuda` adds an experimental GPU implementation of the same indexed query protocol (`cuda/` JIT sources): it wraps the CPU-built CSR index, evaluates each point's row sequentially without atomics (deterministic across runs), is inference-only, serves CPU or CUDA points, and self-skips its parity tests without a GPU; the CPU index remains the oracle and the default. |
| `rtgs/image2gs` | Stage 1. `renderer2d` performs sparse accumulated (sum) blending and exposes scalar-density coverage; `fit` optimizes foreground-cropped masked images with gradient-magnitude initialization. RGB rendering identifies only the product `weight*color`; the independently audited 2026-07-16 contract test found that current downstream coverage/retention/color use is materially gauge-dependent, but authorized no replacement or default change. The native fitter keeps `weight_color_9p` as its bit-exact default and exposes an opt-in `unit_weight_bounded_8p` research seam, a benchmark-only geometry freeze, shared-initialization fitting, and detached diagnostic snapshots. Its independently audited once-only CPU-synthetic comparison found material local null-direction motion in the current Adam path, but the bounded 8p arm lost every appearance-only and joint seed and failed joint non-inferiority; retain the current default and treat the seam as research-only. The optional, lazy `structsplat_backend` starts at a configurable count (640 by default), uses residual/tensor growth until convergence or a separate configurable maximum, and can freeze the live normalized field without clamping before conversion; captured field semantics currently have tested CPU-reference pixel-grid parity only. `adapters` converts native, StructSplat RS, and GaussianImage-style NPZ fields into the common Cholesky initialization representation; those converted files are not exact normalized-renderer teachers. A benchmark-only GaussianImage_plus direct-covariance adapter/reference matches one exact native CUDA binary and one finite-SPD checkpoint subset, but filtering changed 570/19,200 pixels and no production observation-provider integration or quality claim is authorized. Stage-1 execution adds two opt-in acceleration seams, both defaulting off with the serial torch path as the correctness anchor: `FitConfig.batch_views` fits all equally sized, unmasked views jointly (`batched.fit_views_batched`: identical per-view initialization seeds, summed per-view-mean loss, so per-view gradients match the serial path up to float summation order), and `FitConfig.native_renderer` selects the reference `torch` renderer or the experimental JIT-compiled CUDA extension (`cuda_backend` plus `cuda/` sources) behind `renderer2d`'s shared single/batched entry points. The CUDA path follows StructSplat's exact-renderer methodology (per-gaussian blocks, analytic backward, atomic accumulation — not bit-exact across runs) adapted to this repository's additive compositor and half-pixel centers; its GPU parity tests in `tests/test_renderer2d_cuda.py` self-skip on CPU and must pass on a GPU box before any default changes. |
| `rtgs/lift` | Stage 2. `base` implements projection-consistent covariance lifting, validity-aware depth gradients, depth-surface covariance, and separate legacy `Lifter` / compact `CompactInitializer` protocols. `depth` exposes `surface` (default), per-Gaussian `footprint`, and global `isotropic` covariance controls behind the same lifter interface. `gradient` bounds every optimized depth to its ray/object-volume intersection and exposes `legacy`, `normalized`, `valid_uniform`, `confidence`, `confidence_shuffled`, and `thresholded` anchor modes; legacy remains the default and the exact-uniform/shuffled modes are attribution controls. Its training render also exposes inclusive `all` (default), `leave_one_source_out`, and globally balanced `matched_nonself_dropout` supervision. Research callers may pass a detached, validated cross-source `(E,2)` pair tensor to `lift_with_position_pairs`; an extent-normalized Huber-after-L1 world-position term is disabled by default. `matching` defines a separate CPU-first `PositionMatcher` protocol and a deterministic raw-patch/epipolar research backend that accepts only training RGB/cameras plus detached retained centers; its failed synthetic precision audit prevents production use. `surface` defines a view-keyed `OrientedPointBackend` protocol, immutable provenance/prediction/canonical-map records, deterministic registered-depth normal estimation, and validation into safe world-space maps. It also validates retained-indexed plane/alignment targets and implements extent-normalized point-to-plane and sign-invariant selected-axis losses; both coefficients default to zero. The TUM registered-RGB-D backend remains harness-local because its confirmatory cross-view tail audit failed. `hybrid` seeds bounded rays with aligned depth and forwards the same supervision, fixed-pair, and oriented-target controls before confidence/color-aware fusion. Legacy `carve` consumes source images/masks and a dense voxel volume. The CPU-only `compact_carve` instead samples a fixed candidate-ray pool, scores every depth with coverage-weighted queries to every compact teacher, and returns the requested $N_{\mathrm{init}}^{3D}$ when enough candidates pass. `field_lifter` composes compact placement with exact source fibers, `field_refit`, `field_visibility`, `field_observability`, and deterministic `field_topology`; `field_measurement` supplies oracle controls and `field_validation` separately samples immutable normalized/additive teachers. `field_loss` is exact only for additive whole-plane peak-mixture density and RGB numerator, not normalized finite-support/fade/affine StructSplat RGB. The field topology transaction accepts moves on the additive density proxy plus parsimony with fixed visibility/gains; it does not establish move utility. `merge` performs legacy weighted moment matching. `splat_sfm` is the calibrated, RGB-free SfM analog on 2D Gaussian primitives: epipolar mutual/ratio matching, per-view-unique union-find tracks, DLT center triangulation, and linear least-squares covariance triangulation with SPD projection (screen arm, not a default). `beam_fusion` is the tomographic density alternative: analytic beam back-projection and covariance-intersection Gaussian products localize components at ray intersections with contributor lineage (screen arm, not a default). Registry: `rtgs.lift.get_lifter(name)` includes the structural `FieldLifter` research adapter alongside legacy variants. |
| `rtgs/depth` | Depth estimation behind the `DepthBackend` protocol: `mock`; permissive-allowlisted Depth Anything V2 Small through `transformers` (lazy import); robust scale/shift alignment to per-view COLMAP tracks; and object-bounds alignment when calibrated captures have no sparse points. |
| `rtgs/render` | Dense rasterization behind the `Rasterizer` protocol: `torch_ref` (pure-PyTorch EWA splatting + depth-sorted alpha compositing; the correctness anchor, CPU-capable, fully differentiable) and `gsplat_backend` (CUDA, lazy import). Sparse evaluation has a separate `PointRasterizer` protocol and CPU `TorchPointRasterizer`; it accepts arbitrary finite `(u,v)` coordinates, forms one camera-wide visible set and global depth order, and streams explicit point and Gaussian chunks without proposal/lineage filtering. Dense parity is established on the complete frozen synthetic pixel grids and 4,096 sampled calibrated pixel centers. A later compact-training prerequisite established materially nonzero off-grid coordinate, mean, and log-scale gradients with float64 central-difference agreement on a safely interior anisotropic fixture; dense off-grid equivalence and CUDA point-render parity remain untested. The standard hard post-SH color floor and hard `exp(-q/2) * 1[q<12]` EWA support remain the defaults. SMU-1, an outward `12<=q<16` C1 kernel tail, and their hard-forward gradient controls are opt-in research modes. Only `torch_ref` exposes the retained tensors used by the SH-color and kernel-support gradient audits; gsplat rejects kernel-tail modes and either diagnostic explicitly. `visibility_margin_sigma` is a finite-positive Torch-only coarse-cull control whose default `3.0` expression remains bit-exact; gsplat rejects non-default values. For margins above 3, `torch_ref` preserves the established current-set depth order, sorts newly admitted primitives separately, then stable-sorts the concatenation so an expanded set cannot reorder an exact-depth tie among current primitives. The point renderer currently accepts only the default margin. The support-safe `sqrt(12)` setting remains research-only: its 2026-07-15 CPU synthetic incidence gate failed, so no candidate arm ran and the 3-sigma default remains unchanged. The gsplat backend also exposes packed, AbsGS-gradient, and antialiased modes plus raw strategy metadata. `get_rasterizer("auto", device=...)` selects gsplat only for CUDA data. No sparse-render experiment authorizes production quality, memory, speed, density, CUDA/gsplat parity, or a default. |
| `rtgs/optim` | Stage 3. `trainer` uses canonical per-field Adam optimizers, mask/alpha supervision with random backgrounds, separate DC/rest SH learning rates, and a schedule that activates every requested band even in short runs. `TrainConfig` carries opt-in SH-color activation, kernel-support mode, finite-render validation, Torch-only gradient-summary controls, the default-preserving visibility-margin option, and research-only `unit_retraction` / `tangent_displacement_retraction` quaternion update policies; `current` remains the exact quaternion default. Both quaternion Phase-A attempts were invalid before any optimizer/materiality result, so the seam authorizes no policy or default conclusion. The separate `compact_trainer` accepts only `ReconstructionInputs`, a fixed 3D initialization, compact query backends, and a `PointRasterizer`; it preflights before working-set/init transfer, builds a teacher/camera-only device-tensor working set, and uses six aligned Adam groups, isolated per-step RNG streams, fixed-attempt point losses, immutable teachers, and fixed topology. Its audited CPU experiment rejected a general Gaussian-proposal convergence win, and its bounded full-resolution interaction is diagnostic only. The standard hard color/kernel and 3-sigma visibility semantics remain the defaults and diagnostic overhead is disabled. Kernel-support diagnostics reduce retained per-chunk `q`/kernel gradients immediately after each backward pass. `density` is the CPU-compatible classic controller. Lazy `strategies` adapters drive gsplat Default (clone/split/prune/reset, AbsGS, revised opacity) or MCMC (low-opacity relocation/teleportation, growth, position noise) and preserve optimizer state under a hard primitive budget. Evaluation reports held-out image and alpha-IoU/leakage diagnostics. |
| `rtgs/data` | `synthetic` builds ground-truthed tests; `colmap` parses text/binary reconstructions and observation tracks; `calibrated` loads the object-capture JSON format, applies OpenCV distortion correction to RGB/masks, preserves view ids, estimates object bounds, and creates an every-eighth train/test split. `compact_views` strictly loads complete capped `.rtgsv` camera/teacher/optional-alpha files and ordered frame manifests without Pillow, StructSplat, or CUDA. `field_inputs.SceneFits` preserves those teachers, cameras, optional `PackedAlpha`, depth/geometry hints, and an explicit complete disjoint train/held-out partition for native field lifting. `reconstruction_inputs` remains the fixed-topology post-Stage-1 typed/serialized seam and intentionally does not carry packed alpha. Strict loading requires exact nested key sets, identifiers matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`, bounded ZIP metadata before array loading, ordinary files, no symlinks, and resolved containment. The caller remains responsible for dropping any separately retained `SceneData`; `from_scene` also does not yet prove that optional points or `bounds_hint` were derived only from selected training views. |
| `rtgs/pipeline` | `pipeline.py` orchestrates the RGB-backed stages 1–3 with timing and strict train-only initialization; held-out RGB is used only for reporting. `compare_lifters` shares train-view fits across variants. The separate `run_field_pipeline(SceneFits, FieldLiftConfig)` entry runs the image-free field lift and leaves held-out compact teachers for semantic reporting only. |
| `rtgs/visualize` | Writes sampled calibrated-camera reference/init/final/error comparisons, a contact sheet, a calibrated-camera animation, an interpolated object orbit, and an elevation-varying novel path (bounded to 48 frames each). |
| `rtgs/viewer` | Optional, lazily imported Viser WebGL viewer for orbit navigation, initialization/final comparison, splat controls, calibrated train/test cameras, and exact snapshots through the pluggable rasterizer. |
| `rtgs/cli` | `cli.py`, argparse-based. |

Registered lifters: `gradient`, `depth`, `hybrid`, `carve`, and research variant `field` (plus
the `sfm` baseline that mimics classic SfM-point initialization for comparison, and `random` as
the lower-bound baseline).

`inverse_projection_fiber` is now reused by the registered field research path. The separate
`fiber_correspondence` and `source_anchored_sh` modules still expose detached soft/UOT plans and
exact source-direction SH preactivation from the closed correspondence study; those modules are
not production paths. The exact constraint remains a seam for source-anchored field components,
not evidence that every independently fitted 2D mixture fragment is one physical 3D primitive.

## Agent workflow surfaces

Repository task recipes live under `.claude/skills/`. The repo-specific
`realtime-gs-results-audit` scientist pass is also exposed through the matching
`.agents/skills/` discovery symlink; `CLAUDE.md` is the routing authority.

## CLI

| Command | Purpose |
| --- | --- |
| `rtgs fit-images ...` | Stage 1 only: fit 2D gaussians, optionally growing StructSplat from `--initial-gaussians` to `--max-gaussians`; save initialization `.npz` files and, with `--save-observation-teachers`, lossless `.teacher.npz` files for captured field semantics (tested against CPU-reference fixture pixel grids). |
| `rtgs lift ...` | Stage 2 only: lift fitted 2D gaussians into a 3D gaussian set. |
| `rtgs lift-field --dataset ... --heldout-stride ... --field-args ... --out ...` | Image-free Stage 2 research path: strictly load a compact dataset on CPU, create an explicit deterministic train/held-out partition, ignore pre-split points/bounds unless explicitly attested train-only, and run `FieldLifter` without reference images. It saves the requested standard PLY/NPZ; `Path(--out).with_suffix(".field.npz")` stores masses, render opacity, fiber/source state, fitting/all-view correspondence visibility, gains, split indices, and correspondences; strict `Path(--out).with_suffix(".diagnostics.json")` stores isolated train/held-out semantic validation and diagnostics. |
| `rtgs refine ...` | Stage 3 only: run 3DGS optimization from an initialization; select `classic`, `gsplat-default`, or `gsplat-mcmc` density control and save metrics/history/previews. |
| `rtgs run ...` | End-to-end on synthetic, COLMAP, or calibrated-frame data; `--fits` skips stage 1 using native/StructSplat/GaussianImage NPZ files; `--batch-views` fuses stage-1 fitting across training views and `--native-renderer` selects the torch/CUDA stage-1 renderer. `--out` also saves initialization/final PLY and visual previews. |
| `rtgs render ...` | Render a saved gaussian set from a camera path / dataset cameras. |
| `rtgs view ...` | Interactively inspect saved PLY/NPZ gaussians in a browser; optionally load a scene for reference images, train/test camera frusta, and exact torch/gsplat snapshots. |
| `rtgs bench ...` | Delegates to `benchmarks/run.py` (variant comparison + micro-benchmarks). |

## Backend abstractions (hard rule: pluggable, CPU-first)

- **Rasterizer** (`rtgs.render.base.Rasterizer`): `render(gaussians3d, camera, bg) -> RenderOutput(color, alpha, depth, means2d, strategy_info)`. `strategy_info` is optional backend-native metadata used only by density strategies; it is `None` on the CPU reference path. `torch_ref` is authoritative for image semantics; `gsplat_backend` must match it (parity test, `@pytest.mark.cuda`). Auto-selection respects the data device, including an explicit CPU request on a CUDA host. The trainer and ray lifters only speak to this interface.
- **DepthBackend** (`rtgs.depth.base.DepthBackend`): `predict(image) -> DepthPrediction(depth, kind)` where kind ∈ {`metric`, `relative`, `affine`, `inverse`}. Non-metric predictions are aligned (`rtgs.depth.align`) before lifting.
- **OrientedPointBackend** (`rtgs.lift.surface.OrientedPointBackend`): view-keyed predictions declare
  geometry kind, normal frame, validity/confidence, and immutable provenance. Canonicalization
  validates the contract and returns detached world-space points/normals. No implementation is a
  production default; the real TUM registered-depth reference is isolated in its research harness.
- **Compact observation query** (`GaussianObservationField` / `GaussianObservationIndex`): the
  dependency-free field is the CPU equation anchor and the sparse CPU tile index implements the
  same point-query surface. The index stores three flattened CSR arrays (`tile_keys`,
  `tile_offsets`, `component_ids`) instead of one tensor per tile, and answers queries by streaming
  a bounded `(point, component)` pair sequence — in canonical point-major, ascending-component
  order — through exact paired field evaluation, replacing the eager per-tile Python query loop.
  `GaussianPointProposal` has O(`N_opt,2D`) base component state; the reference query is
  O(samples x components), while the CSR index stores O(component x overlapped tiles) entries and
  evaluates only local candidates. The pre-CSR grouped index is retained privately
  (`_GroupedObservationIndexReference`) as the frozen parity/benchmark oracle. It uses
  fixed-attempt null thinning plus a uniform floor. Its declared risk is continuous fitted-window
  area, not the legacy discrete-pixel loss; equivalence is an experiment, not an assumption.
  Current parity evidence covers complete CPU pixel grids, not CUDA or arbitrary continuous
  coordinates.
- **CompactInitializer** (`rtgs.lift.base.CompactInitializer`): consumes
  `ReconstructionInputs` rather than `SceneData`. The first implementation is the standalone
  `CompactCarveInitializer`; it is deliberately not registered in the legacy lifter registry or
  wired into the CLI. A bounded research harness now composes it with `CompactTrainer`, but the
  terminal calibrated lifecycle failed at its later exact-viewer ABI gate and therefore did not
  establish an end-to-end integration PASS.
  Query point batches and the transient streamed point–component pair chunk are explicitly capped
  (`CompactCarveConfig.max_query_pairs`, plumbed into each index); the CSR index still stores all
  component–tile overlaps in contiguous arrays, aggregate bundle/component/index/3D-cardinality
  budgets are incomplete, and custom query backends must honor the chunk contract. Placement emits
  a silent-by-default typed `CompactPlacementProgress` record and persists final pair/chunk/payload
  counters in the initialization diagnostics. An opt-in `select_all_eligible` mode retains every
  globally supported candidate (one lift per proposed 2D Gaussian across all views) instead of the
  balanced top-K; `rtgs.lift.merge.merge_by_voxel(..., return_group=True)` then deduplicates the
  dense set and returns the cluster map, whose composition with per-Gaussian lineage is the
  cross-view correspondence byproduct. `compact_confidence_gate` deterministically filters those
  clusters using frozen multiplicity, cohesion, depth-sharpness, covered-view, and reprojection
  thresholds; it is opt-in and emits complete typed per-cluster diagnostics.
  `rtgs.lift.compact_init_eval` scores an initialization *before* optimization against exact compact
  teachers (full/foreground PSNR + SSIM), renders only each teacher's fit window, accepts pluggable
  rasterizers, exposes silent-by-default view/row progress, and can cache immutable teacher/support
  targets across candidates. `benchmarks/compact_init_eval.py` saves metrics and viewer PLYs for
  top-K, dense+merge, and optional easy-only arms. The audited calibrated chain found that dense
  improved init quality but failed its count gate, while easy-only failed the frozen downstream
  held-out gate; balanced top-K therefore remains the default. `rtgs.lift.compact_refine` remains an
  off-by-default correspondence-free prototype: it can optimize consensus while drifting toward
  the density core, so it does not authorize a geometry claim.
- **Structure-from-splats** (`rtgs.lift.splat_sfm`): the calibrated SfM analog operating on 2D
  Gaussian primitives instead of keypoints, entirely RGB-free. Pairwise matching solves closed-form
  ray–ray closest points per candidate, gating on epipolar residual (normalized by the candidate's
  own pixel sigma), color distance, and metric size consistency `sigma_px * z / f`, then keeps
  mutual, ratio-tested matches; union-find builds tracks under a strict one-splat-per-view
  invariant; centers triangulate through the batched calibrated DLT with cheirality, reprojection,
  and triangulation-angle gates; and covariances solve the stacked linear system
  `vech(Sigma2D_v) = A_v vech(Sigma3D)` (three equations per view, six unknowns) by least squares
  with bounded-eigenvalue SPD projection. Every output Gaussian carries its full track lineage and
  unmatched splats are reported for downstream densification. On the EWA-projected fixture the
  inversion is near-exact (centers ~1e-7, covariances ~1e-6 relative); segmentation-mismatched
  real fields and downstream utility are unmeasured, so it is a screen arm
  (`benchmarks/splat_sfm_screen.py`), not a default.
- **Tomographic beam fusion** (`rtgs.lift.beam_fusion`): the density-family alternative to both
  consensus scoring and discrete matching. Every 2D splat back-projects to an analytic 3D beam
  (tangent-plane covariance at its implied depth plus a long along-ray variance); closed-form
  ray–ray closest points seed gated pairs; fusion uses **covariance intersection**
  (`Lambda = mean_k Lambda_k`) — the naive Gaussian product is rejected because correlated views
  double-count shared axes (measured ~1/K overconfident) while CI is exact on fully-shared
  directions and conservative, never overconfident, elsewhere; remaining views fold in greedily by
  projected-pixel gating, and reduction is signature dedupe plus per-voxel weight NMS.
  Association emerges from density overlap and contributor lineage is returned. Centers are exact
  on the EWA fixture; covariances are approximate-by-construction (compose with the splat-SfM
  linear covariance triangulation when exactness matters). Screen arm only; no default change.

No module imports CUDA-only or heavyweight optional dependencies at import time; they are
imported inside functions and failures produce actionable error messages.

Viser's WebGL preview consumes explicit centers, covariances, RGB, and opacity; because its wire
format has no SH fields, the viewer evaluates all active SH bands on CPU and refreshes RGB as the
browser camera moves. Exact viewer snapshots still go through `Rasterizer`, so gsplat/CUDA
snapshots retain authoritative sorting/rasterization and the same backend parity contract as
training and `rtgs render`.

## Conventions

- Camera extrinsics are **world-to-camera** (COLMAP): `x_cam = R @ x_world + t`; `Camera.position` is the camera center in world space. +z is the viewing direction (OpenCV).
- Images are float32 tensors in `[0,1]`, shape `(H, W, 3)`; pixel `(u, v)` = (column, row); the pixel center of the top-left pixel is `(0.5, 0.5)`.
- 2D covariances are parametrized by Cholesky factors `(l11, l21, l22)` with positive diagonal (GaussianImage). The product `weight*color` is accumulated RGB; `weight` alone is a repository-added scalar factor, not identifiable alpha or opacity. Lifted observations start with independent conservative opacity. 3D covariances use unit quaternion + log-scales (3DGS).
- Exact StructSplat teachers retain RS scales/rotation, unclamped colors, activated amplitudes,
  optional affine colors and covariance filtering, normalized-blend epsilon, compact support/fade,
  full canvas plus fitted viewport, and independent `N_init,2D`/`N_opt,2D` counts. Converted
  `Gaussians2D` NPZs omit these semantics and remain initialization-only.
- Colors in `Gaussians3D` are SH coefficients `(N, K, 3)`, `K = (deg+1)^2`; degree-0 stores `(rgb - 0.5)/C0` (3DGS convention).
- Production pipeline tensors are `torch.float32`; validation/canonicalization callers may request
  float64 explicitly. Tests seed all RNGs.
