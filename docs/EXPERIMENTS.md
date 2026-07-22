# Experiment log

Dated, append-only log of research experiments — positive **and negative** results.
Template:

```markdown
## YYYY-MM-DD — short title
- **Question**: what hypothesis was tested?
- **Setup**: exact command(s)/config, git rev, seed(s), scene(s)
- **Result**: the numbers (paste the relevant benchmark/test output)
- **Conclusion**: what we now believe, and with how much confidence
- **Follow-ups**: next experiments this suggests (mirror into ROADMAP if substantial)
```

Rules: an experiment that changes a default hyperparameter must be linked from the code
comment at the changed default. Threshold changes in tests must cite an entry here.

---

## 2026-07-22 — Aggregate index budgets + checkpointed pair-chunk queries (mechanism)

- **Question**: The ROADMAP compact-scaling bullet requires aggregate device-byte/index
  budgets and a backward-activation-memory bound before any production-scale claim. Can both
  land as opt-in controls without changing default behavior or numerics?
- **Setup**: (1) `CompactCarveConfig.max_index_entries_total` / `max_index_bytes_total`
  (default `None` = prior behavior): the summed per-view CSR cost is preflighted via
  `GaussianObservationIndex.estimate_entries` before any allocation, and built or
  caller-supplied backends are re-checked against both totals in `build_query_backends`,
  `score_world_points`, and `CompactCarveInitializer.initialize`. (2)
  `checkpoint_pair_chunks` keyword on the CSR index's `query`/`query_weight_sum`:
  gradient-carrying queries evaluate each pair chunk under non-reentrant activation
  checkpointing, retaining only chunk inputs and recomputing one chunk at a time during
  backward; inert without gradients and off by default.
- **Result**: CPU tests (`tests/test_compact_budgets_and_checkpoint.py`): budget preflight
  rejects one entry over the exact estimated total before allocation and passes at the exact
  total; byte budget rejects on built backends; supplied-backend enforcement verified through
  `score_world_points`. Checkpointed queries are bit-identical forward and gradient-equal to
  1e-7 against the baseline, and a `saved_tensors_hooks` byte probe measured saved activation
  bytes at **0.48x baseline** (568,416 → 272,232 bytes; 3,612 pairs streamed in 512-pair
  chunks). Toy-scale measurement; no production memory claim.
- **Conclusion**: Both mechanisms exist, are exercised by CPU tests, and leave every default
  untouched. With CSR (2026-07-20), the CUDA query backend (2026-07-21, GPU-unverified), and
  these two controls, the compact-scaling bullet's remaining work is the production-scale
  confirmatory run, not mechanism.
- **Follow-ups**: Choose production budget values from the 26-view Janelle preflight numbers;
  wire `checkpoint_pair_chunks` into the compact trainer behind its own preregistered config
  once a training-memory question is actually opened.

## 2026-07-21 — Indexed CUDA compact-teacher query backend (mechanism, GPU-unverified)

- **Question**: Can the compact-placement observation queries — the workload behind the one
  observed ~4 h CPU placement run (26 views, 130k components, 108.16M point-view
  projections) — run on the GPU behind the existing `ObservationQueryBackend` seam without
  touching the placement pipeline?
- **Setup**: New `rtgs.core.observation2d_cuda.GaussianObservationIndexCuda` (JIT extension in
  `rtgs/core/cuda/`) implementing the same `query`/`query_weight_sum` protocol: it wraps the
  CPU-built CSR index verbatim (identical caps, tile membership, and canonical
  ascending-component order), binary-searches each point's tile, and accumulates its row
  sequentially in registers — no atomics, so repeat queries are bit-identical; only FMA
  contraction can differ from the CPU reference. Inference-only by design (gradient queries
  stay on the CPU index). Entry points: `build_query_backends(observations, config,
  device="cuda")` in `rtgs.lift.compact_carve`, feeding the existing `backends` parameter of
  `score_world_points` / `CompactCarveInitializer.initialize`; a conditional CUDA arm in the
  tracked `compact_placement_csr_cpu` micro-benchmark reports `cuda_seconds` /
  `cuda_speedup_vs_csr` / parity errors when a GPU is present.
- **Result**: CPU-side only on this container: guards, helper-construction parity, and the
  full existing CSR/grouped suite pass; the seven GPU parity/determinism/counter tests
  self-skip. No GPU execution, timing, or parity number exists yet.
- **Conclusion**: Mechanism in place behind the anticipated seam ("a future CUDA backend can
  implement this same interface", observation2d.py). **No correctness or performance claim**
  until the `cuda`-marked tests and the benchmark arm run on a GPU box; the CPU CSR index
  remains the default and the oracle.
- **Follow-ups**: On a GPU box: `pytest -q tests/test_observation2d_cuda.py` (builds the
  extension, runs the parity matrix), then `benchmarks/run.py --quick --update-docs` for the
  tracked CUDA-arm numbers; then a placement-scale rerun of the compact pipeline with
  `build_query_backends(..., device="cuda")` on the Janelle frame to retire or confirm the
  ~4 h CPU placement cost, with the idle-GPU/repeats protocol before any speedup claim.

## 2026-07-21 — Fused batch_views stage-1 fitting + CUDA extension skeleton (mechanism)

- **Question**: Can stage-1 fitting run as one fused multi-view optimization (the prerequisite
  for the ROADMAP M4 batched CUDA kernel) without changing per-view results?
- **Setup**: New opt-in `FitConfig.batch_views` path (`rtgs.image2gs.batched`) and batched
  renderer entry (`render_gaussians_2d_batched`) sharing the serial scatter core; new JIT CUDA
  extension for the native additive compositor (`rtgs.image2gs.cuda_backend` +
  `rtgs/image2gs/cuda/`, modeled on StructSplat's exact CUDA renderer) behind
  `FitConfig.native_renderer` (default `torch`). Commands: `./scripts/verify.sh`;
  `.venv/bin/python benchmarks/run.py --quick --update-docs` on a CPU-only cloud container
  (shared host, single trial, torch 2.13.0+cpu). The run executed the dirty working tree that
  is committed together with this entry; the JSON meta records base rev `892f448`.
- **Result**: Parity — batched step-0 PSNR equals serial (asserted < 1e-3 dB in
  `tests/test_image2gs_batched.py`), final per-view PSNR within a 1 dB proximity floor, and the
  tracked quick run's 12-view mean final PSNR agrees to seven digits (28.2822013 batched vs
  28.2822014 serial). Timing is an indication only (single trial, shared container, no
  repeats): batched 7.58 s vs 11.18 s serial (1.48x) at quick config, 2.57x at smoke config.
  JSON: `benchmarks/results/20260721T191424Z_cpu.json`.
- **Conclusion**: The fused path reproduces serial fits on CPU (high confidence at mechanism
  level; float summation order is the only divergence source). CPU timing is not a performance
  claim. The CUDA extension is written and guarded but **unverified on GPU hardware** — its
  parity tests (`tests/test_renderer2d_cuda.py`) self-skip on CPU; no correctness or speed
  claim is made for it, and `torch` remains the default renderer.
- **Follow-ups**: On a GPU box: run the `cuda`-marked parity tests; rerun
  `benchmarks/run.py --update-docs` (full config, idle GPU, repeats) for serial-vs-batched and
  torch-vs-cuda stage-1 numbers; exercise a calibrated `dataset/` frame with `--batch-views`
  before any default change (see the ROADMAP M4 note).

---

## 2026-07-22 — Seven-initializer endpoint comparison viewer (integration only)

- **Question**: Can the exact initial and selected-final PLYs from the completed compact
  initializer suite be inspected from one unchanged orbit camera instead of seven independent
  viewer processes?
- **Setup**: Added the strict checked-in
  `benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json` manifest (SHA-256
  `edf774227a35479d600d939e14fc631c9fa1a1598625a2c5c95d70a915022448`) and launched
  `.venv-cuda/bin/rtgs view --comparison-manifest
  benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json
  --max-viewer-gaussians 50000 --device cpu --port 8782 --no-open` with
  `CUDA_VISIBLE_DEVICES=''`. The manifest names top-K, beam fusion, dense+merge, easy-only,
  splat-SfM, field, and random, with paths resolved relative to the manifest. This is a post-result
  handoff; it did not run during training or participate in checkpoint selection.
- **Result**: The strict loader prepared all 14 endpoints in method/initial/final order and exposed
  their actual counts in the selector; counts were 5,000/43,288, 5,000/44,222, 2,088/49,177,
  7/35,644, 943/39,987, 127/39,059, and 5,000/39,513 respectively. The 50,000 preview cap therefore
  retained every endpoint. Eleven focused CPU viewer tests passed. PID 1725254 owned the exact
  `127.0.0.1:8782` socket and returned HTTP 200; one post-start sample was 721,572 KiB RSS, and that
  PID was absent from the `nvidia-smi` compute-process query. The prior dense-only viewer was
  stopped after this replacement passed. The run-local receipt is
  `runs/all_initializers_frame00008_20260721/comparison_viewer_receipt.json`.
- **Conclusion**: The viewer now supports flicker-style endpoint comparison: orbit once, then
  switch the **Gaussian set** entry without resetting the client camera. It loads all models in
  host memory and sends the selected model to the WebGL client. This is not a simultaneous
  side-by-side render, a visual-quality conclusion, a performance benchmark, or a zero-overhead
  claim. The compact bundle has no source RGB, so calibrated reference-image snapshots are outside
  this handoff; a same-machine browser may still use the display GPU.
- **Follow-ups**: Record qualitative observations separately from the frozen quantitative result.
  If simultaneous panels become necessary, add a synchronized two-canvas UI and benchmark its
  browser-side memory before recommending it during training.

---

## 2026-07-21 — Full compact-compatible initializer convergence suite

- **Question**: On the same full 26-view/130,000-component `frame_00008` compact bundle and
  ordinary adaptive-density schedule, does any repository initializer that can legitimately
  consume compact-only evidence converge to materially better fitted quality than every other
  arm?
- **Setup**: Prospective descriptive six-arm execution at revision
  `d74c9a623cba8af4694e0112753927407c7fdab5`, seed 0, PyTorch 2.12.0+cu132, CUDA 13.2, gsplat
  1.5.3, RTX 4090. The arms were top-K, dense+0.06 merge, frozen easy-only gate, all-pairs
  splat-SfM, complete 128-track field lift, and a 5,000-point bounds-only random control; the prior
  full beam fit entered as a disclosed historical anchor. Native counts were retained. Every arm
  used the same 30k gsplat DefaultStrategy parent (density 500–15k, every 100, cap 100k), then
  non-exact fixed-topology 10k segments to the first joint plateau or 70k. All 26 compact teachers
  were fit and source RGB was not opened. Protocol/result/audit:
  `benchmarks/results/20260721_all_initializers_frame00008_{PREREG,RESULT,AUDIT}.{md,json}`
  (where present).
- **Result**: Initial 3D counts were top-K 5,000, beam 5,000, dense 2,088, easy-only 7,
  splat-SfM 943, field 127 (128 before one accepted topology move), and random 5,000. All arms
  reached the joint plateau at the 70k assessment; selected steps were 69k for beam, easy-only,
  and splat-SfM and 70k otherwise. Final foreground-PSNR order was dense **38.248049**, beam
  37.887375, splat-SfM 37.706291, random 37.425717, top-K 37.299174, field 37.240826, easy-only
  36.958743 dB. Dense led beam by **0.360674 dB**, but its selected objective 0.002554868 was
  **4.4003% worse** than beam's best 0.002447185. Thus dense failed the required objective
  improvement and no arm passed both materiality gates. Density-stop/final counts ranged from
  35,644 to 49,177.
- **Conclusion**: `NO_MATERIALLY_SUPERIOR_CONVERGED_INITIALIZER`. Dense+merge and beam form the
  fitted-quality Pareto front; the frozen practical-equivalence intersection is empty because the
  two best metrics split. Balanced top-K remains the default. Dense is the initialization-quality
  leader on this scene (20.7546 dB before optimization), but the final rank cannot be causally
  attributed to placement after major topology growth. Random's fourth-place finish emphasizes
  the robustness—and confounding effect—of ordinary adaptive density. This is one seed, one scene,
  native-count, all-fitted-view evidence, not held-out geometry or generalization.
- **Applicability and audit**: Gradient, legacy carve, depth, hybrid, and classic SfM were not
  called losers; their required RGB/depth/sparse-point inputs do not exist in the compact-only
  bundle. The independent replay passed 10,012 checks, loaded 482 PLYs/4.63 GB finite with exact
  counts, and exactly reproduced all seven selected compact metric sets. Field quality remains
  valid, but its harness saved only aggregate topology counts (7 proposed/1 accepted), not the
  protocol-required individual move receipts; move-level topology utility is unaudited. Timings
  remain nonportable sequential diagnostics. All 199 focused initializer/recovery/viewer tests
  passed. The repository-wide non-slow gate retained 16 unrelated failures: nine frozen ABI/source
  binding checks and seven checks requiring a missing historical G2SR input artifact; no threshold
  or fail-closed binding was weakened.
- **Viewer handoff**: No viewer ran during measured execution. After audit, a CPU/Torch viewer for
  dense+merge initial versus selected final returned HTTP 200 at `http://127.0.0.1:8781`, used
  about 578 MiB RSS at its launch sample, and owned no `nvidia-smi` compute process. This supports
  visual inspection without a viewer-server CUDA allocation, not a zero-overhead claim; live
  watching uses `--watch-checkpoints` and the browser may still use a same-machine display GPU.
- **Follow-ups**: Do not tune these consumed all-view outcomes. A default-selection follow-up must
  be a fresh multi-scene/multi-seed protocol with train-only selection, genuinely held-out cameras,
  and explicit capacity/budget control. Add execution-time field move receipts before rerunning
  topology utility. Compare RGB/depth/SfM-required methods only in a separately named cohort whose
  inputs actually exist.

---

## 2026-07-21 — Full `frame_00008` bounded beam fusion and convergence

- **Question**: Can the tomographic beam idea consume the full 26-view, 130,000-splat compact
  bundle, initialize exactly 5,000 3D Gaussians, and reach the frozen compact-training plateau;
  and can a separate CPU viewer expose 1k-step progress without taking CUDA resources?
- **Setup**: Preregistered all-view development run at revision
  `d74c9a623cba8af4694e0112753927407c7fdab5`, seed 0, PyTorch 2.12.0+cu132, gsplat 1.5.3,
  RTX 4090. Beam fusion evaluated all 325 view pairs/all 8.125 billion 5k×5k ray pairs with
  minimum 3 views, transverse/fold gates 3σ, RGB gate 0.35, RGB σ=0.25, a 0.0223616-world-unit
  voxel, 20k seed budget, and final cap 5k. The count-matched top-K control used 32 depths,
  minimum 2 views, robust fraction 0.60, and score floor 0.01. Only beam received the new fit:
  DefaultStrategy density through 15k, 30k parent, then non-exact fixed-topology 10k segments to
  the first joint plateau or 70k. Protocol/result/audit:
  `benchmarks/results/20260721_beam_fusion_full_frame00008_{PREREG,RESULT,AUDIT}.{md,json}`
  (where present).
- **Result**: Both arms initialized exactly 5,000. Beam placement took 138.326 s, admitted
  345,109,938 gated seeds into 743,844 seed voxels, and returned components with 18–26
  contributing views. Its initialization was **11.5826 dB** mean foreground PSNR versus
  **11.8629 dB** for top-K (beam − top-K **−0.2803 dB**); crop SSIM was 0.72967 versus 0.77155
  and alpha IoU 0.00199 versus 0.26740. Density grew beam to 44,222 Gaussians by 15k. Both frozen
  convergence rules plateaued at the 70k assessment and selected 69k: fitted compact-target
  foreground PSNR **37.8874 dB**, crop SSIM **0.995821**, alpha IoU **0.976061**. Seventy PLY
  checkpoints plus init/final reloaded finite and stayed under the 100k cap.
- **Conclusion**: Full-bundle bounded beam fusion is computationally feasible and trainable, but
  it is **not a better initializer on this scene**. The later quality cannot be attributed to beam
  fusion because ordinary densification expanded the model 8.84× and there is no matched top-K
  downstream fit. All views were fit, so there is no held-out/generalization claim and no default
  change. The 91.665-second top-K diagnostic clears the linked CSR task's numeric time target once,
  but does not close its missing exact-parity, repeated-benchmark, CSR-memory, or tracked-table
  gates.
- **Viewer and audit**: The training watcher was CPU-only and owned no CUDA allocation; PLY-save
  callbacks totaled 0.896 s versus 2,163.649 s of optimizer time. CPU/RAM/I/O were nonzero and no
  controlled on/off run exists, so “zero impact” is rejected. The executed watcher required
  manually expanding the count slider beyond the initial 5k; a post-run UX fix now follows growth
  automatically and passed a separate HTTP-200 smoke. The independent scientist pass accepts the
  single-scene result but reports 16 unrelated full-suite failures from stale frozen ABI bindings
  and a missing historical G2SR artifact; 150 focused tests had 145 pass/5 skip, with no relevant
  failure.
- **Follow-ups**: Before reconsidering beam fusion, run a matched top-K downstream arm with the
  same target hashes/schedule and a held-out-view protocol, then test whether greater source-splat
  coverage or beam-association plus splat-SfM covariance triangulation improves the initial alpha
  failure. Measure viewer overhead with randomized on/off repeats if it becomes a performance
  claim.

---

## 2026-07-21 — Full compact StructSplat 2D reconstruction gallery

- **Question**: Do all 52 compact 5,000-component 2D teachers in
  `dataset/2025_03_07_stage_with_fabric` reconstruct their calibrated source foregrounds well
  enough for direct visual inspection, and can the result be handed off as one browser gallery?
- **Setup**: CPU-only diagnostic at revision `e9e98b07edc41c2c7a229ce2110539a1493a4591`
  with PyTorch 2.12.1+cpu, 16 Torch threads, and an AMD Ryzen 9 7950X. The exact command was
  `.venv/bin/python scripts/render_compact_structsplat_gallery.py --out
  runs/structsplat_teacher_gallery_20260721`; executed script SHA-256
  `884421b536e933bc3887ccbb106f618da7a4ceac9e0684b3e2d13cc966f83ec0`. Each strict `.rtgsv`
  archive was decoded and rendered over its native fit window by the normalized CPU reference in
  `external/structsplat` revision `e9206cdfa1a2ebd4d44301569f63d0fa10ba82fb`, while the matching
  JPEG from `external/dataset` received the converter's calibrated bilinear undistortion. Metrics
  use float tensors before display clamping/JPEG encoding and the stored mask limits the primary
  comparison to foreground. Compact frame-manifest SHA-256 values are
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847` and
  `c31f976e016b3f681ac7aed528bae660ae77f315f37cf2128024fdef5a413262`.
- **Result**: All **52/52** views rendered and all source/archive hashes and native output
  dimensions revalidated. Mean per-view clamped foreground PSNR was **35.5733 dB** for
  `frame_00008` (range 33.9638--38.5182) and **35.5626 dB** for `frame_00009` (range
  34.0489--38.1443), or **35.5679 dB** across all views. Across 256 deterministic pixels/view,
  the largest archive-query versus independent StructSplat-raster difference was
  `5.5522e-5`; a separate CSR full-image recomputation of the minimum/maximum-quality view from
  each frame changed PSNR by at most `8.99e-7` dB. The independent render calls took 135.69 s and
  the full decode/undistort/render/JPEG run took 205.69 s, but this unrepeated mixed-resolution
  pass is diagnostic timing, not a benchmark. The 91 MiB, 315-file artifact includes 52 native
  original crops, 52 native reconstructions, 156 gallery previews, per-view JSON, a manifest, and
  `index.html`.
- **Conclusion**: The compact teachers provide visually close 2D foreground reconstructions;
  residuals concentrate around high-frequency lace, hair, and thin fabric boundaries. This says
  nothing by itself about compact 3D placement or refined 3DGS quality, and it does not resolve the
  sparse-initialization concern in `TASK_COMPACT_PLACEMENT_CSR_ACCELERATION.md`. The current
  StructSplat source digest `e186bb4e...` differs from the producer digest `f468ff32...` stored in
  the archives, so this is an equation-checked render with the requested current checkout, **not**
  a bit-exact replay of the historical producer. No default or capability claim changes.
- **Viewer handoff**: `runs/structsplat_teacher_gallery_20260721/index.html`; from the repository
  root run `.venv/bin/python -m http.server 8766 --bind 127.0.0.1` and open
  `http://127.0.0.1:8766/runs/structsplat_teacher_gallery_20260721/index.html`. The index and a
  representative lazy-loaded image returned HTTP 200 and the original/reconstruction/error panels
  were visually inspected. This is a purpose-built 2D teacher viewer, so `rtgs view` (which accepts
  3D Gaussian PLY/NPZ files) is not applicable.
- **Independent audit**: `runs/structsplat_teacher_gallery_20260721/AUDIT.md` disposes the result as
  a valid diagnostic with narrowed source and timing scope. Focused StructSplat observation,
  compact-view, and compact-init-evaluation tests passed (26/26).
- **Follow-ups**: Use the gallery to separate Stage-1 appearance errors from later 3D placement
  failures, but require a controlled 3D experiment before attributing any final-view artifact to
  either stage. Recover the exact historical StructSplat source tree before claiming producer-level
  bit identity; use repeated warm runs if renderer performance becomes a decision question.

---
## 2026-07-21 — Tomographic Gaussian beam fusion: density-based initializer (mechanism)

- **Question**: Setting aside correspondence search entirely, can 3D initialization be posed as
  reconstruction-from-projections — back-project every 2D Gaussian as an analytic 3D beam and let
  closed-form Gaussian products localize density at ray intersections?
- **Setup**: New `rtgs.lift.beam_fusion` (CPU-first, deterministic, RGB-free). Each splat's 2D
  covariance lifts to the ray-orthogonal tangent plane at its implied depth (the fiber's exact
  construction) plus a long along-ray variance; pair seeding uses closed-form ray–ray closest
  points gated by transverse footprint distance and color; fusion uses **covariance intersection**
  (`Lambda = mean_k Lambda_k`) rather than the naive Gaussian product (`sum_k`), because the views
  are correlated observations of one splat and the product double-counts every shared axis;
  remaining views fold in greedily by projected-pixel gating; reduction is exact
  contributor-signature dedupe plus per-voxel weight NMS (selection, not moment matching).
  Association emerges from density overlap — no mutual/ratio matching, no union-find, no DLT.
  Forward oracle: `project_covariances_ewa(dilation=0)`; 19 CPU tests
  (`tests/test_beam_fusion.py`).
- **Result** (mechanism, idealized fixture): centers are exact (**max error ~2e-7** world units;
  the CI mean of beams through the true point is the true point), all ground-truth Gaussians
  return as full-view components with zero unmatched, identical-color twins produce no ghosts at
  `min_views=3`, and single-view decoys are excluded and counted. The covariance contract was
  **verified in both directions**: CI eigen-ratios vs truth were `[1.004, 1.053, 18.1]` on an
  isotropic blob — exact on directions all views observe, conservative (never overconfident) on
  the weakly-shared depth axis — while the rejected naive product measured `[0.20, 0.21, ...]`,
  i.e. ~1/K overconfident with K=5 views, confirming the design backtrack numerically. On the
  shared four-arm synthetic screen (opacity-matched): top-K 6.45 / splat-SfM 6.56 /
  **beam-fusion 6.74** dB init-only mean foreground PSNR at the same count, dense+merge 7.49 at
  2× count.
- **Conclusion**: The tomographic family works and is a genuinely different mechanism from both
  consensus scoring and discrete matching: exact centers, provably-never-overconfident (but
  deliberately conservative) covariances, correspondences as a byproduct of density overlap. Its
  covariance is approximate where splat-SfM's least-squares triangulation is exact; its
  association is soft-gated where splat-SfM's is discrete — complementary tools. High confidence
  in the mechanism; **no claim** about segmentation-mismatched real fields or downstream utility,
  and absolute synthetic numbers are relative-only.
- **Follow-ups**: Run the four-arm `benchmarks/splat_sfm_screen.py --bundle` on the calibrated
  seven-view bundle (yield/ghost/unmatched profiles for both new arms vs the E1 histogram); if
  exact covariances matter downstream, compose the two methods (beam-fusion association → linear
  covariance triangulation on its contributor sets); include both as arms in the next
  preregistered matched-budget downstream experiment.

---

## 2026-07-21 — Structure-from-splats: calibrated SfM analog on 2D Gaussians (mechanism)

- **Question**: With calibrated cameras, can classical SfM's structure half — epipolar matching,
  track building, triangulation — be re-derived for 2D Gaussian primitives (RGB-free, no
  keypoints) to produce a well-defined 3D Gaussian initialization, including full covariances?
- **Setup**: New `rtgs.lift.splat_sfm` (CPU-first, deterministic). Matching solves closed-form
  ray–ray closest points per candidate pair and gates on epipolar residual normalized by the
  candidate's pixel sigma, color distance, and metric size consistency `sigma_px * z / f`;
  mutual-best + SIFT-style ratio test; union-find tracks with a strict one-splat-per-view
  invariant; centers via the existing batched calibrated DLT
  (`triangulate_centers_dlt`) with cheirality/reprojection/angle gates; covariances via the
  stacked linear system `vech(Sigma2D_v) = A_v vech(Sigma3D)` (3 equations per view, 6 unknowns,
  least squares + bounded-eigenvalue SPD projection); colors amplitude-weighted; unmatched splats
  reported for densification. Forward oracle for tests:
  `project_covariances_ewa(dilation=0)` — ground-truth 3D Gaussians are projected into 5–6 views
  to build exact 2D fields, and the module must invert that construction blind
  (`tests/test_splat_sfm.py`, 18 CPU tests).
- **Result** (mechanism, idealized fixture): recovery is near-exact — all ground-truth Gaussians
  come back as full-length tracks with **max center error ~1e-7** world units, **covariance
  relative error ~6e-7**, reprojection ~2e-6 px, zero unmatched. **Identical-color twins** (color
  totally uninformative) are correctly disambiguated by epipolar geometry + multi-view
  consistency — the case where the correspondence-free consensus refine drifted. A single-view
  decoy is excluded and surfaced in `unmatched_per_view`. On the shared synthetic screen
  (`benchmarks/splat_sfm_screen.py --synthetic`, opacity-matched arms): top-K 6.45 dB /
  dense+merge 7.49 dB (2× count) / splat-SfM **6.56 dB** init-only mean foreground PSNR at the
  top-K count, with splat-SfM the only arm at ~0 px mean reprojection and the fastest placement
  (0.01 s vs 0.02 s). Init-only photometrics remain mass-dominated (the E2 lesson), so the
  geometric advantage does not show in this metric.
- **Conclusion**: The SfM analog is mathematically sound and implemented end-to-end: with known
  cameras, correspondence is the only hard problem, and epipolar + multi-view verification solves
  it exactly on consistent inputs — including full 3D covariance triangulation, which neither
  carve nor merge provides. High confidence in the mechanism; **no claim** yet about
  segmentation-mismatched real fields (2D fits are per-view segmentations, so partial match rates
  are expected by design) or downstream utility. Absolute synthetic numbers are relative-only.
- **Follow-ups**: Run `benchmarks/splat_sfm_screen.py --bundle` on the calibrated seven-view
  bundle (one command on the workstation) to measure real track yield, reprojection distribution,
  and unmatched fractions vs the E1 cluster histogram (78.44% monocular); then add splat-SfM as a
  fourth arm to the next matched-budget downstream experiment (count-matched, budget-filling
  MCMC growth, longer horizon) where its exact geometry should matter; consider union with
  dense-carve for unmatched regions.

---

## 2026-07-20 — Calibrated dense confidence-gated initialization chain (E1/I1/E2)

- **Question**: Can dense all-Gaussian compact placement improve calibrated initialization, can a
  frozen correspondence-confidence classifier compress it to an accurate easy-only seed, and can
  matched density control reconstruct the dropped hard set without losing held-out quality?
- **Setup**: E1/I1 used the strict seven-view, 640-components/view Janelle `frame_00008` compact
  bundle with seed 0 and voxel size 0.06. E1 compared balanced top-K with dense-all+merge against
  exact compact teachers. Before opening easy-only quality, I1 froze multiplicity/cohesion/depth/
  reprojection thresholds and expected 60 retained clusters. E2 then froze seven optimization
  views, C1002 validation, late-release C1004 held-out, `downscale=8`, a 300-step gsplat
  DefaultStrategy schedule, a 2,319 cap, seeds 20260720/20260721, and the control-repeat-calibrated
  decision. Canonical result/audit files are
  `benchmarks/results/20260720_dense_confidence_gated_init_{e1,i1,e2}_{RESULT,AUDIT}.md`; E2's
  preregistration SHA-256 is
  `9a7107a3314f17b514c64d7aa91d656e81535b75fc2f032d795a8547547d9f9e`.
- **Result**: E1 dense-all gained **+1.9714 dB** mean foreground PSNR with every view positive, but
  failed the count gate at **2,319/172 = 13.48×**. I1 retained exactly **60/2,319** clusters and its
  exploratory same-view screen led top-K by +0.4505 dB. In E2, C1004 foreground PSNR was
  **14.9079 dB dense-all**, **12.7332 easy-only**, and 11.2280 top-K; the top-K repeat was 11.2351.
  Easy-only therefore missed dense-all by **2.1747 dB**, outside the **0.0071 dB** repeat envelope,
  while ending at 1,229 versus 2,319 Gaussians and taking 2.191 versus 2.298 native seconds. It was
  still growing and improving at step 300. The raw E2 result SHA-256 is
  `1990a5e9510e83da5a94f5d8684700149e6bba6e77bba9eee0960fef5bf91e32`.
- **Performance diagnosis**: Host `perf` was blocked by `kernel.perf_event_paranoid=4`, so a
  one-view `cProfile` decomposition found 634.0 s in the full-frame Torch render versus 2.2 s in
  teacher construction. Cropping the camera to the scored fit window and using the pluggable
  gsplat backend reduced the calibrated E1 replay from 98m31s/17.6 GB RSS to 47.2s/1.81 GB, with
  exact PLY hashes and at most 0.003812 dB aggregate metric drift from the CPU anchor. A post-result
  immutable-target cache was exact against the frozen GPU metrics: 14.31 s one-time preparation,
  then 7.55/7.11 s repeated evaluations, 652,517,359 retained bytes, and 2,269,648 KiB peak RSS.
  These are single-machine diagnostics, not portable benchmark claims.
- **Conclusion**: Dense placement has real calibrated init signal, but neither dense-all nor the
  easy-only compression passes the frozen route to a default change. E2 is a valid negative for
  this short matched-cap schedule, not proof that densification can never recover. Because the
  held-out deficit was not spatially localized to hard-dropped regions, I2/E3 remains closed and
  balanced top-K remains the default.
- **Follow-ups**: If revisited, preregister a longer budget-filling control or spatial-localization
  diagnostic on fresh evidence. Do not attribute the deficit to missing hard correspondences or
  tune I1 thresholds on the consumed E2 result. Re-run clean repeated timing on an idle machine
  before making performance claims.

---

## 2026-07-20 — Dense all-Gaussian init, voxel merge, and a correspondence-free 4-dof refine

- **Question**: Does lifting *every* supported 2D Gaussian (not the sparse top-K), then
  deduplicating with a voxel-hash merge — optionally with a local 4-dof depth refine between lift
  and merge — improve the image-free initialization, and does a correspondence-free local refine
  help?
- **Setup**: CPU-only, deterministic. New opt-in seams: `CompactCarveConfig.select_all_eligible`
  (one carve lift per 2D Gaussian across all views), `merge_by_voxel(return_group=True)` (dedup +
  cross-view correspondence byproduct), init-only compact-view metrics rendering each 3D
  initialization against its exact 2D teacher (`rtgs.lift.compact_init_eval`), a runnable harness
  (`benchmarks/compact_init_eval.py --synthetic|--bundle [--refine]`), and a correspondence-free
  local 4-dof refine (`rtgs.lift.compact_refine`) that lifts a `CompactInitializationResult` into
  the exact `InverseProjectionFiber` and gradient-optimizes depth (optionally the covariance
  ray-scale) against a smooth multi-view consensus objective (soft coverage × color agreement,
  source view excluded). Evidence is CPU-fixture mechanism only; no calibrated bundle was run.
- **Result**: On the synthetic 5-camera scene, dense+merge leads the balanced top-K by **+1.04 dB**
  init-only mean foreground PSNR (7.49 vs 6.45) at a comparable Gaussian count, and the merge group
  map recovers the expected cross-view correspondence clusters. The local refine **does** maximize
  its consensus objective and is deterministic, but its effect on geometry is at best neutral and
  can be **negative**: on a deliberately coarse-sampled fixture the objective rose (3.9938→3.9956)
  while mean distance-to-truth *worsened* (0.122→0.140). Correspondence-free consensus rewards
  multi-view coverage, so it drifts toward the volumetric density core rather than the exact
  surface.
- **Conclusion**: (1) Retaining all 2D Gaussians + merge is a cheap, real denser-init gain on the
  fixture and is the first thing to measure on a calibrated frame. (2) The consensus objective is a
  poor *depth* refiner on its own — this empirically reproduces the documented finding that pinning
  fiber geometry needs explicit cross-view correspondence, not consensus. The refine is retained as
  an opt-in prototype and a seam the correspondence path can plug into; it is **off by default** and
  makes no quality claim. Absolute synthetic numbers are relative-only.
- **Follow-ups**: Execute the preregistered chain in
  `docs/TASK_DENSE_CONFIDENCE_GATED_INIT.md` — E1: `benchmarks/compact_init_eval.py --bundle` on a
  calibrated `dataset/` frame (init-only top-K vs dense+merge, audited); I1: a correspondence-
  confidence gate (view multiplicity + cohesion + depth sharpness) that keeps only easy clusters;
  E2: easy-only seed + density control (split/merge/prune + MCMC teleport) vs dense-all vs top-K on
  downstream held-out quality; I2/E3: wire `fiber_correspondence` for the hard set only if
  densification cannot cover the dropped regions.

---

## 2026-07-20 — Flattened exact CPU CSR observation index (placement Phase 1)

- **Question**: Can the exact, image-free compact-placement query be accelerated by replacing the
  per-tile Python query loop in `GaussianObservationIndex` with one flattened CSR index and a
  bounded vectorized point/component pair stream, **without** changing any selected ray, depth
  sample, score equation, or downstream geometry? (Task:
  `docs/TASK_COMPACT_PLACEMENT_CSR_ACCELERATION.md`, Phase 1.)
- **Setup**: CPU-only, seed 0. The index now retains three contiguous arrays (`tile_keys` int64
  `[T]`, `tile_offsets` int64 `[T+1]`, `component_ids` int32/int64 `[E]`) built directly in
  canonical component-ID order, and answers queries by `torch.searchsorted` + a bounded
  `(point, component)` pair stream evaluated through the exact
  `GaussianObservationField._paired_values` / `_paired_weights`, reduced with deterministic
  `index_add`. Streaming is capped by `CompactCarveConfig.max_query_pairs` (default 1,048,576),
  plumbed into every index. The pre-CSR grouped index is retained privately
  (`_GroupedObservationIndexReference`) as the frozen parity/benchmark oracle. Parity is tested
  against both that oracle and the all-component `GaussianObservationField.query` reference across
  tile sizes 1/2/16, normalized/additive blending, support fade, filter variance/AA dilation,
  affine color, odd-crop mean residuals, zero amplitudes, mixed radii, window/support/empty-tile
  edges, and coordinate gradients (`tests/test_observation_csr.py`,
  `tests/test_compact_carve.py`). Tracked micro-case `compact_placement_csr_cpu` added to
  `benchmarks/run.py`; result file `benchmarks/results/20260720T123859Z_cpu.json`
  (torch 2.13.0+cpu).
- **Result**: The CSR query is numerically indistinguishable from the reference within the float32
  contract — **max color error 2.4e-7**, **max weight-sum error 2.4e-7** vs the grouped oracle,
  and **bit-exact (0.0)** vs the sequential all-component `field.query` reference (both sum the
  identical candidate subset in ascending component order). Discrete decisions (`n_seen`,
  `n_covered`, winning depth index, selected candidate indices, source lineage) are **exactly
  equal** to the grouped reference on the deterministic placement fixtures, and are invariant to
  the legal pair budget. The tracked `--quick` micro-case (600 components, 2,048 query points,
  tile 16) recorded **grouped 0.1427 s → CSR 0.0082 s = 17.4× speedup**, retained payload 20,480
  bytes, `int32` component IDs, 4,094 entries over 256 non-empty tiles (max 27 candidates), peak
  pair chunk 32,766, `within_contract=1`. The speedup grows with field size (the per-query
  candidate list, not projection, dominated the old path), consistent with — but not a substitute
  for — the session-local 108–120× production-batch prototype.
- **Conclusion**: Phase 1's flattened CSR CPU path is implemented as the production default,
  preserves exact placement identity and downstream geometry, bounds transient memory, and makes
  progress visible (`CompactPlacementProgress`) and persisted (placement counters in the
  initialization diagnostics). The CPU micro-benchmark and unit parity confirm the mechanism and
  the direction of the speedup with high confidence. **What is not established here**: the full
  production confirmatory gate — all 26 views, 130,000 compact 2D Gaussians, 32 depth samples,
  4.16M sampled points, selecting 5,000 seeds, wall time `4:02:28 → ≤ 300 s` (target ≤ 180 s) — was
  **not** run in this CPU environment (the production bundle and a comparable baseline workstation
  are unavailable here); it remains open on the baseline machine, to be measured against the frozen
  reference audit. Per the task, this acceleration does **not** improve the visibly weak
  5,000-Gaussian initialization quality; that remains work for the correspondence-aware fiber
  initializer.
- **Follow-ups**: Run the frozen 26-view/130k production placement on the baseline workstation and
  confirm the ≤5-minute gate and exact-parity/quality guardrails; save initialization-only metrics
  and viewer-ready PLYs beside the downstream metrics; only then consider the optional pluggable
  CUDA scorer (Phase 2) and evidence-gated hierarchy (Phase 3, opened only on a post-CSR profile).

---

## 2026-07-20 — Full compact all-view reconstruction and placement diagnosis (development)

- **Question**: Are the 26 mask-aware, sub-168,000-byte StructSplat bundles sufficient to fit a
  high-resolution 3D Gaussian model without opening source RGB during fitting, what quality remains
  relative to compact 2D playback, and why is the initial placement slow?
- **Setup**: Single-scene development run on Janelle `frame_00008`; the GPU quality path used a
  local RTX 3050 (a post-run session observation, not a receipt-bound environment field), PyTorch
  2.9.0+cu128, gsplat 1.5.3, StructSplat 0.1.0, seed 0, native compact fit windows, all 26 views
  fitted, 5,000 components/view (130,000 total), lossless packed alpha in every bundle, and 32
  depth samples per component-center ray. The 26 bundles total 4,152,383 bytes and the largest is
  162,657 bytes. CPU
  placement selected 5,000 initial 3D Gaussians; gsplat refinement grew to 36,816. The 30k parent
  recovered non-exactly from step 4k, then four fixed-topology 10k segments restarted Adam/RNG at
  30k, 40k, 50k, and 60k. The last two segments reduced every learning rate by 0.25 each. Compact
  targets alone selected checkpoints and stopping; provenance-matched source RGB was opened only
  after the 70k model and plateau decision were frozen. Command-preservation limits, artifact
  hashes, aggregate metrics, and audit dispositions are in
  `benchmarks/results/20260720T002059Z_full_compact_all_view_development.json`.
- **Result**: The one CPU placement invocation took **14,547.616 s (4:02:27.6)** to score 4.16M
  depth samples across 26 views and retained 5,000 seeds. This is an observed local duration, not a
  portable performance benchmark: CPU model, threads, load, and repeats were not recorded. The
  compact-only selector chose step 70,000 (SHA-256 `078ecabe...76cbd`) and both frozen stopping
  rules reported plateau: last-six Theil–Sen slope **+0.002212 dB/1k**, recent median gain
  **+0.009821 dB**, median per-view objective improvement **0.148811%**, **2/26** views improving
  by more than 1%, and ten trailing non-material transitions. From 30k to 70k, compact objective
  fell **8.6368%** and fitted-view foreground PSNR rose **0.5892 dB**. Recomputed over all 26
  provenance-matched source views, compact 2D playback reached **35.5733 dB foreground / 40.5409
  dB crop / 0.970498 crop SSIM**; the final 3D model reached **33.4500 / 38.4176 / 0.966955** with
  **0.974411 mean alpha IoU**. Thus the 3D fit remains about **2.1233 dB** below its compact 2D
  teachers on fitted source views.
- **Conclusion**: The compact bundles preserve enough source-view information to produce a strong
  qualitative 3D reconstruction on this scene, but they do not establish held-out or novel-view
  quality because every T/V/H view was fitted. “Converged” here means only that the final settle
  segment met its pre-existing compact-training plateau rule; it is not mathematical/global
  convergence. The user's viewer inspection judged the true 5,000-Gaussian initialization visibly
  weaker than the final model, but no initialization-only metric was saved. Fitting never opened
  raw RGB, yet it still rendered dense native-resolution targets from the compact fields and used
  image-space optimization. The dirty-source, non-exact recovery/restart chain is artifact-bound
  but not replay-complete.
- **Follow-ups**: Implement and remeasure the exact CPU CSR query task in
  `docs/TASK_COMPACT_PLACEMENT_CSR_ACCELERATION.md`; treat its 108–120× micro-speedup and 2.5–3
  minute projection as hypotheses until a tracked idle-machine benchmark confirms them. Save
  initialization-only metrics next time, then replace the sparse top-K placement with the planned
  source-fiber/correspondence-aware lift so runtime acceleration is not mistaken for better
  initialization.

## 2026-07-18 — Mask-gated StructSplat view under a strict 168,000-byte cap

- **Question**: Can one calibrated masked dataset image be replaced by an exact native
  StructSplat 2D-Gaussian field below 168,000 bytes without spending Gaussian centers on the
  masked background, and does the field still preserve useful image quality?
- **Setup**: Real training view `C0014` from Janelle `frame_00008`, calibrated native-resolution
  RGB/mask undistortion and tight crop, 5,000 fixed `aniso_onedge` WSE Gaussians, masked density,
  foreground-only L1 plus mask-normalized SSIM, hard foreground center projection, 1,000 Adam
  steps, and StructSplat `cuda_tiled` playback on the local RTX 3050. The strict decimal cap was
  168,000 bytes. Command, source/provider hashes, protocol, and artifacts are in
  `benchmarks/results/20260718_structsplat_masked_168kb_example_RESULT.md`.
- **Result**: **PASS for one-view archive integrity and size; visual acceptance remains pending,
  and mask-free silhouette playback failed.** The exact 5,000-splat archive is **150,492 bytes**
  (17,508 spare), contains no RGB or mask member, and is 98.46x smaller than the 14.82 MB JPEG.
  All initial/final rounded centers stayed inside the foreground. Strict archive playback matched
  the recorded live native render exactly and reached **36.8788 dB foreground PSNR / 0.901959
  weighted SSIM**. However, raw playback
  against the masked crop was only **17.8756 dB / 0.729800 SSIM**: finite-support silhouette IoU
  was 0.6032 and 31.18% of outside-mask crop pixels exceeded one 8-bit code value.
- **Conclusion**: Mask-gated fitting is a viable compact foreground-observation format for this
  view and does not create background-centered tokens. It does **not** make a normalized RGB
  Gaussian field an exact replacement for alpha. A post-run diagnostic found the lossless binary
  crop mask compresses to 7,226 zlib bytes, so teacher plus mask would still fit this example's
  cap before a small wrapper; that packaging result is not yet implemented or generalized.
- **Follow-ups**: Let the user judge the saved raw/composited panel. Then freeze either a
  lifting-only contract or a per-view teacher-plus-lossless-alpha package, implement adaptive
  per-view count backoff, and run every view through strict byte/reload/render gates. Do not infer
  that all dataset views fit from this single example or that fixed 5,000 components guarantee
  the cap.
- **Independent audit**: **ACCEPT NARROW FEASIBILITY WITH CAVEATS**. The archive, size arithmetic,
  strict load, native rerender, metrics, and center invariant recompute. The evidence is
  artifact-verifiable rather than replay-complete, and the panel could not be visually inspected
  through the audit sandbox. See
  `benchmarks/results/20260718_structsplat_masked_168kb_example_AUDIT.md`.

## 2026-07-17 — Exact inverse-projection fibers with latent hard-min correspondence, Iteration 1

- **Question**: If every fitted 2D Gaussian is lifted onto its exact source inverse-projection
  fiber, can multi-view center-plus-conic gradient descent recover the hidden 3D Gaussians and
  thereby establish correspondences without supplied tracks?
- **Setup**: Eight synthetic degree-zero 3D Gaussians, six calibrated ring cameras, four fitting
  and two held-out views, 32 source hypotheses, exact source mean/covariance fibers, CPU float64
  Adam for 400 updates, and roots `17687011..17687013` paired with depth roots
  `17687111..17687113`. Controls were free geometry, oracle correspondence, and cyclic shuffled
  correspondence. The committed result is
  `benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json`, SHA-256
  `2601a45d...`; the exact 50-file executed source closure is archived at
  `benchmarks/results/20260717_inverse_projection_fiber_iter1e_EXECUTED_SOURCES.tar`.
- **Result**: **FAIL**. The source projection invariant held to at most `7.11e-15` px center and
  `7.85e-16` relative covariance error. Fiber-conic nevertheless reached only
  `0.625/0.833/0.573` train association, `0.609/0.844/0.594` held-out association,
  `0.594/0.813/0.500` correct tracks, and `0.639/0.294/0.558` world-center p90. The paired oracle
  reached 100% association/tracks and center p90 below `3.9e-8` in every root. Fiber's mean p90
  was almost the shuffled control (`0.4968` versus `0.4992`). The free-source attribution control
  drifted and is uninterpretable.
- **Independent audit**: **ACCEPT VALID NEGATIVE WITH CAVEATS**. Transaction hashes, publication
  order, current inodes, aggregates, and gates recompute. Exact per-hypothesis arrays/float64 final
  tensors were not saved, and the sentinel combiner omitted three passing families; the latter is
  repaired for future runs with a focused `2 passed` regression. See
  `benchmarks/results/20260717_inverse_projection_fiber_iter1e_AUDIT.md`.
- **Conclusion**: One exact source projection correctly constrains each candidate's inverse fiber,
  but does not choose its cross-view correspondence. Independent row-wise minima admit stable
  many-to-one and view-inconsistent assignments. More steps or a different optimizer are not
  supported as the next repair. No global correspondence, appearance, topology, real-data, or
  performance claim is established.
- **Viewer handoff**: The post-result CPU viewer for root 17687011's initial/final fiber-conic PLYs
  returned HTTP 200 at `127.0.0.1:8891`; the exact command is in the result note. It was
  qualitative and non-decision-bearing.
- **Follow-ups**: On fresh roots, freeze and test the outcome-informed residual `<0.1` prune gate,
  source-preserving duplicate contraction within `0.01`, balanced rematching of all original 2D
  observations, and fixed-track refitting. Unequal counts, dustbins/occlusion, appearance, and
  split decisions remain for the calibrated-data iteration.

## 2026-07-17 — Residual topology repair for exact inverse-projection fibers, Iteration 2

- **Question**: Can the frozen residual-prune, proximity-contract, balanced-rematch, and
  fixed-track-refit bundle repair Iteration 1's many-to-one hard-min correspondence failure without
  changing the exact source fiber?
- **Setup**: One committed, once-only CPU float64 synthetic run on fresh paired scene/depth/order
  roots `27688011..27688013`, `27688111..27688113`, and `27688211..27688213`. Each root used the
  same eight hidden 3D Gaussians, four fitting and two withheld views, 32 source hypotheses, 400
  hard-min updates, the frozen residual `<0.1` retain rule, source-preserving radius-`0.01`
  contraction, exact balanced rematching, and 200 recovery updates. Hard-min, proposed, cyclic
  shuffled-score, and oracle arms shared exact input tensors. The committed result is
  `benchmarks/results/20260717_inverse_projection_fiber_iter2_RESULT.json` (SHA-256
  `d153706a...`); the executed-source archive is SHA-256 `373545e0...`.
- **Result**: **FAIL**, with Gate 1 validity PASS and selection, correspondence/geometry, and
  negative-control Gates 2--4 FAIL. Proposed representative counts were **8/8/7**. Roots 0 and 1
  accepted exact eight-track fits with fitting and held-out association `1.0`, center p90
  `3.103e-7` and `2.955e-7`, and covariance medians `1.255e-6` and `1.381e-6`. Root 0 nevertheless
  missed the frozen survivor-precision floor (`0.904762 < 0.95`). Root 2 retained seven tracks,
  covered `0.875` of hidden modes, and had no candidate for hidden mode 2, so it was correctly
  rejected. Mean exact-track gain over hard-min was `0.135417`, below `0.20`. Proposed and shuffled
  both had mean hidden-mode coverage `0.958333`, so the required identity separation was zero.
- **Independent audit**: **ACCEPT VALID NEGATIVE**. The audit reloaded the sealed code and all raw
  NPZ evidence, independently reconstructed every cost, assignment, acceptance, and gate, found
  zero false checks and zero scalar-summary mismatches, and verified one committed transaction with
  all roots consumed. See
  `benchmarks/results/20260717_inverse_projection_fiber_iter2_AUDIT.md`; its machine-readable audit
  is SHA-256 `98440bd7...`.
- **Conclusion**: Fixed residual topology can contract already-correct duplicate candidates, but
  it cannot invent a mode that hard-min fitting failed to localize. Exact final geometry on two
  accepted roots does not rescue the topology failure, and the shuffled control prevents a
  residual-identity claim. Do not report the result JSON's aggregate `0.999999642` center reduction:
  the rejected root carries zero placeholder geometry and biases that field. No production default,
  real-data, unequal-count, appearance, occlusion, GPU, or performance claim follows.
- **Viewer handoff**: Initial, hard-min, proposed (when accepted), shuffled, and oracle PLYs are
  preserved under `runs/inverse_projection_fiber_iter2_official_20260717/`. This iteration's
  quantitative decisions came only from sealed array evidence; no GPU work or one-shot replay was
  performed during the independent audit.
- **Follow-ups**: Iteration 3 is preregistered in
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG.md` (SHA-256 `59f0de21...`).
  It moves capacity and unmatched mass into the fitting loop using Bhattacharyya row-softmax and
  augmented unbalanced transport, tests one/two/three-way view decompositions plus outliers, and
  then performs a bounded calibrated compact-bundle interaction. Track death, coherence, camera
  correction, and full-scale split/merge remain separate until the assignment mechanism survives.

## 2026-07-18 — Capacity-aware correspondence on exact inverse-projection fibers, Iteration 3

- **Question**: Can dustbin-aware full-covariance row attention or augmented unbalanced transport
  preserve hidden modes under unequal one/two/three-way 2D Gaussian decompositions, while every
  source component remains on its exact four-DOF inverse-projection fiber?
- **Setup**: The final three-iteration protocol froze three new scene/depth/order root tuples,
  five fitting and two held-out cameras, eight hidden parents, 90 root-0 source tracks, two
  outliers per fitting view, hardmin/row/UOT-uniform/UOT-area/oracle/shuffled arms, 20 detached
  E-steps with two Adam updates each, and declared-capacity completeness/dust gates. Four protocol
  documents, final source hashes, 84 focused CPU tests, and an independent prospective review
  preceded the exact once-only command. See
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_IMPLEMENTATION_REVIEW.md`.
- **Execution**: **CONSUMED / INCOMPLETE / FAILED-EXECUTION**. Root 0 completed; root 1 wrote only
  its initial PLY before terminal-only context reported that a plan-supported projection left the
  valid camera domain during an M-step; root 2 never started. The durable ATTEMPT remains at
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_ATTEMPT.json` (SHA-256
  `3a2b3cdb...`). No top-level RESULT was manufactured and the roots cannot be resumed or rerun.
- **Independently recomputed root-0 result**: D/UOT-area reached `0.5468` purity, `0.25`
  completeness, and track/observation outlier recall `0.2730/0.0560`; C/UOT-uniform reached
  `0.5259`, `0.25`, and `0.2068/0.0349`. Both transport-mass diagnostics passed, and source
  center/covariance equalities remained about `1e-14/1e-15`, but both arms fail the frozen
  per-root `0.90/0.90/0.80/0.80` acceptance floors. Later roots therefore could not restore real
  release. D's root-local purity deltas were `+0.1220` vs hardmin, `+0.0803` vs row, `+0.0209` vs
  uniform UOT, and `+0.1125` vs shuffled; cross-root means and capacity attribution are withheld.
- **Structural diagnostic**: Even oracle labels yielded center p90 `1.0593` and held-out parent
  assignment `0.6125`. Of 80 inlier split children, 83.75% differ from their parent moment center
  (p90 `0.843 px`). An arbitrary independently fitted 2D fragment is therefore not automatically
  the projection of the single latent parent Gaussian. The exact fiber belongs at a stable track
  or moment-merged source-aggregate level, not blindly on every decomposition fragment.
- **Conclusion**: The exact fiber implementation is numerically correct, and two-sided transport
  shows a limited root-local purity signal, but the proposed raw-fragment correspondence layer is
  insufficient. It fails absolute purity/completeness and unmatchedness; capacity weighting alone
  does not solve outliers. The calibrated bundle, appearance, C1004, viewer, GPU, and performance
  stages are **withheld**. The independent scientist pass and raw hashes are in
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_FAILURE_AUDIT.md` and `.json`.
- **Follow-ups**: This evidence loop is closed—there is no Iteration 4 retry. A newly authorized
  question should first test an oracle topology ceiling with dynamic moment-merged source anchors,
  transactional/backtracked M-steps, per-arm failure receipts, and a calibrated outlier/null model
  plus sparse epipolar/visibility candidates. Only after that ceiling passes should fresh roots
  test learned transport or unlock real data.

## 2026-07-17 — Compact proposal-target refinement factorial, iter3

- **Question**: With topology fixed at 835 3D Gaussians and source RGB forbidden, does optimizing
  the active compact Gaussian proposal-attempt measure improve occupancy-region fitting over
  uniform-area importance correction, and is any effect attributable to the camera schedule?
- **Setup**: One sealed four-arm factorial on the seven native-resolution compact teachers from
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`, the aligned center-occupancy
  proposal, and the common 835-Gaussian compact-Carve initialization. Arms crossed IID versus
  balanced-cycle scheduling with uniform-area versus proposal-attempt targets for 140 CUDA point
  updates, 128 attempts/update, checkpoints 0/35/70/140, train roots 76801--76803, and fresh
  4,096-attempt uniform/proposal banks rooted at 76901--76903. All fitting and evaluation were
  RGB-denied. The immutable result is
  `benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json`
  (SHA-256 `c0a278a8...`).
- **Result**: The authorizing D/B contrast changed only the target while retaining balanced-cycle
  exposure. Its final proposal-risk ratios were `0.7816781`, `0.7799140`, and `0.7705785`;
  geometric ratio `0.7773749` (22.26% lower), with `3/3` wins. The checkpoint log-AUC-derived
  geometric ratio was `0.8812884` (11.87% lower). Final uniform-risk geometric ratio was
  `0.9480007`, and every preregistered safety/population gate passed. Scheduling alone was not
  established: B/A and D/C AUC ratios were `0.9993201` and `1.0031376`.
- **Independent audit**: **PASS**, narrowly confirming `AUTHORIZE_DENSITY_FOLLOWUP`. All seven
  gates, 12 workers, three banks, paired streams, endpoint invariants, runtime/source/input
  bindings, and RGB denial were independently recomputed. See
  `benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_AUDIT.md` (SHA-256
  `44836994...`).
- **Conclusion**: Proposal-attempt targeting is supported for the next compact density-control
  mechanism test on this scene. Balanced-cycle scheduling is retained only as an execution regime,
  not a proven improvement or default. This result is same-camera compact-teacher MSE with fixed
  `m_opt_i^2D=640` and `N_init^3D=N_opt^3D=835`; it establishes no source-RGB equivalence,
  novel-view quality, geometry accuracy, production scaling, speed, ordinary-3DGS superiority, or
  default. One per-view uniform-risk ratio worsened by about 8.7% despite the aggregate safety
  pass.
- **Viewer handoff**: Exact gsplat 1.5.3 CUDA renders for all four arms and seven 5328x4608 cameras
  are in
  `runs/compact_occupancy_refinement_factorial_iter3_20260717/visualization_seed_76801/`; the
  contact sheet is `CONTACT_SHEET.png` (SHA-256 `add70e4e...`). The live viewer compares final D
  against final/initial B at `http://127.0.0.1:8879`; visualization is post-result and
  non-decision-bearing.
- **Follow-ups**: Run a separately preregistered RGB-free variable-count experiment with matched
  topology/count controls, persistent IDs, explicit optimizer surgery, and fresh banks before
  considering pruning, repeated waves, or CLI integration.

## 2026-07-17 — GaussianImage_plus direct-covariance provider parity

- **Question**: Can an isolated adapter recover a compact additive 2D Gaussian field from the
  frozen GaussianImage_plus checkpoint format and reproduce that repository's exact native CUDA
  renderer closely enough to justify a later provider-quality experiment?
- **Setup**: One sealed, once-only mechanism test bound the clean external checkout at commit
  `549cfaab...`, exact `csrc.so` hash `9b57b7e...`, RTX 3050/Torch 2.9 CUDA 12.8 worker, seven
  rendered semantic fixtures plus a 257-candidate rejection sentinel, and one 160x120 checkpoint.
  The raw checkpoint contained 639 components; the deterministic adapter retained the 626 finite
  SPD components without repair. No source RGB, image fitter, calibrated scene, or 3D stage was in
  scope.
- **Result**: All frozen renderer/adapter gates passed. The raw checkpoint diagnostic had maximum
  raw image error `3.4571e-6`; the 626-component provider field had maximum raw error `9.5367e-7`.
  Projected means, radii, hit counts, and complete candidate sets agreed, and the over-cap sentinel
  was rejected before native dispatch. Filtering was visibly non-neutral: 570/19,200 pixels changed
  by more than `1e-6`, with maximum clamped-channel change `0.3718417883`.
- **Independent audit**: **QUALIFIED PASS**. Lifecycle, source/external/checkpoint/worker hashes,
  every gate, and a separately implemented NumPy renderer recomputation passed. See
  `benchmarks/results/20260717_gaussianimage_plus_provider_parity_AUDIT.md` (SHA-256
  `484d2f27...`).
- **Conclusion**: The exact frozen binary and checkpoint adapter are qualified as a mechanism seam
  for a later experiment. This is not evidence that SPD filtering is harmless, that
  GaussianImage_plus fits full-resolution images well, that it scales better than StructSplat, or
  that it improves initialization/refinement/viewer quality. Provider promotion requires a new
  preregistered full-resolution Stage-1 quality and downstream 3D experiment. No default changed.

## 2026-07-17 — One-view full-resolution Stage-1 mask and residual-growth screen

- **Question**: On calibrated view C0001, does fitting the foreground mask/crop improve the compact
  2D teacher over the frozen full-frame 640-component/100-update teacher, and does one residual-
  growth wave from 640 to 1280 components improve further when given 200 updates?
- **Setup**: Exploratory, non-decision-bearing screen on the existing dirty tree at revision
  `2dddca4aff59702341af9faceefa76ad2505dd83`, seed 0, exact undistortion and mask crop, native
  5328x4608 coordinates, and StructSplat's `cuda_tiled` backend. `plan.json` binds source aggregate
  `43df93cc0175be617032711210bb943403c4b041d5fa968167bd261adbaaaaf0`, inputs, effective
  external configs, dirty StructSplat provider source, environment, and loaded extension. RGB was
  allowed only for Stage-1 fit and isolated evaluation; the lossless teacher archives contain no
  RGB tensor or source path. The command was:

  ```bash
  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
    .venv/bin/python benchmarks/compact_stage1_mask_screen.py \
    --out runs/compact_stage1_mask_screen_20260717
  ```

- **Result**: Under the frozen raw-foreground-PSNR criterion, masked/cropped 640/100 won. The
  original run scored 21.0036 dB versus the frozen baseline's 17.8827 dB (+3.1209 dB), cut
  foreground MSE by 51.26%, and reduced foreground holes from 5.7925% to 0.2984%. The one-wave
  640-to-1280/200 arm scored 20.5525 dB, 0.4511 dB below masked 640, with 0.7730% holes. Its split
  at update 99 collapsed logged masked-crop PSNR to 17.428 dB at update 100 and recovered only to
  25.190 dB at update 199. A fresh-process exact-plan replay preserved both directions: masked
  640 gained 3.0941 dB over baseline and growth lost 0.4887 dB relative to masked 640. The fitted
  archives were not byte-identical across runs, so current CUDA fitting is not bitwise
  deterministic under the recorded seed. Evidence, contact sheets, hashes, caveats, and replay
  details are in `runs/compact_stage1_mask_screen_20260717/`.
- **Independent audit**: **QUALIFIED**. The referee independently recomputed the input/tensor/
  archive bindings, common foreground denominator, metrics, winner, split event, and sampled
  archive/CUDA parity (worst maximum absolute error `2.77e-6`). It confirmed the two-run ordering
  but retired deterministic-replay, speed/VRAM, held-out, causal-mask-only, capacity-ceiling, 3D,
  and default claims. The realtime-gs source binding is strong but omits some transitive executed
  modules, so it is not fully replay-complete. See
  `runs/compact_stage1_mask_screen_20260717/AUDIT.md`.
- **Conclusion**: On this one view and in both runs, foreground-focused fitting materially improved
  the compact teacher at the same 640-component/100-update budget. The tested one-wave residual-
  growth/recovery schedule regressed; this does not show that 1280-component capacity is worse, a
  StructSplat ceiling, a general mask benefit, or any 3D/novel-view improvement. The frozen
  full-frame teacher's masked-crop score is objective-mismatched and is not used for the capacity
  claim. Full-canvas masked scores are background-zero dominated. No default changed. `rtgs view`
  is not applicable because the screen deliberately ends at 2D teachers rather than a 3D PLY.
- **Follow-ups**: Before any default or broad claim, replicate across views and seeds and use a
  matched masked fixed-1280 control or gentler multi-wave growth/recovery schedule. The immediate
  3D follow-up should consume the winning compact teacher only after that scope and the
  nondeterminism are acknowledged.

## 2026-07-17 — Seven-view masked 640/100 compact-teacher acquisition

- **Question**: Can the C0001 screen's masked/cropped 640-component, 100-update configuration be
  acquired independently on all seven calibrated training views as a strict lossless
  `ReconstructionInputs` bundle for the next exploratory RGB-denied lift?
- **Setup**: Existing dirty tree at revision
  `2dddca4aff59702341af9faceefa76ad2505dd83`; native 5328x4608 calibrated undistortion;
  per-view masks; ordered views `C0001,C0008,C0014,C0021,C0026,C0031,C0039`; seeds 0–6;
  StructSplat `cuda_tiled`; $N_{\mathrm{init},i}^{2D}=N_{\mathrm{opt},i}^{2D}=640$; 100
  updates. Each view ran in a fresh process. C1004 was excluded from fitting, metrics, and compact
  payload, although startup checked that its files existed. RGB and masks were restricted to the
  acquisition/QA workers; the parent assembled the bundle from compact archives and cameras. The
  source-bound command was:

  ```bash
  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
    .venv/bin/python benchmarks/compact_masked_bundle_acquisition.py \
    --out runs/compact_masked_bundle_640_20260717
  ```

- **Result**: All seven workers passed and atomically wrote the strict bundle at
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs` with 4,480 compact components,
  `geometry:null`, and no dense raster/source-path field. Same-training-image foreground PSNRs
  were `21.0008,22.6388,21.9038,18.9990,23.0282,18.5951,19.5257` dB (equal-view arithmetic mean
  `20.8131` dB). The original parent lifecycle nevertheless remains terminal **FAIL**: after
  bundle creation, an over-broad manifest-value search mistook `masked` in the required bundle
  name for a forbidden `mask` field. A separate no-refit/no-overwrite verifier strict-loaded the
  immutable outputs and wrote `recovery_result.json`:

  ```bash
  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
    .venv/bin/python benchmarks/compact_masked_bundle_recovery.py \
    --out runs/compact_masked_bundle_640_20260717
  ```

  Plan, manifest, and bundle-aggregate SHA-256s are `21a32010...`, `6ed60cf3...`, and
  `3920f3ae...`. The untouched acquisition harness exactly matches its plan-bound hash
  `6a11d589...`. A visual target/teacher sheet and full hashes are adjacent to `RESULT.md`.
- **Independent audit**: **QUALIFIED**. The referee verified all 53 plan-bound source hashes,
  inputs/tensors/cameras/crops, seven worker records, strict archive semantics, byte-identical
  acquisition/bundle teachers, CUDA rerenders, and fresh query/raster parity (worst maximum
  absolute error `6.855e-7`). It found one provenance defect: the reused effective-config helper
  records `effective_structsplat.init.seed=0` for all views, while the bound worker actually passes
  seeds 0–6; the external fit digest does not cover `InitConfig`. See
  `runs/compact_masked_bundle_640_20260717/AUDIT.md`.
- **Conclusion**: The immutable bundle is content-valid and usable as a frozen exploratory
  Stage-1 input, but recovery does not erase the original lifecycle failure and the effective-init
  record is wrong for six views. This acquisition has no paired full-frame arm and therefore does
  not establish a seven-view causal improvement. It provides no deterministic, held-out,
  novel-view, 3D, performance, capacity, default, or end-to-end claim. No default changed. Viewer
  handoff is intentionally incomplete/not applicable because the requested phase creates no 3D
  PLY.
- **Follow-ups**: Treat the exact archive hashes—not the incorrect descriptive init-seed field—as
  the frozen input identity for the already-started exploratory lift. Future acquisitions should
  construct and digest each effective `InitConfig` with the actual per-view seed before execution,
  and should not promote masking generally without a paired multi-view baseline.

## 2026-07-17 — Masked compact-lift occupancy screen and replay qualification

- **Question and setup**: On the seven frozen masked 640-component teachers, does explicit
  silhouette occupancy repair component-center ray selection relative to the teachers' normalized
  blend density at fixed $N_{\mathrm{init}}^{3D}=835$? The exploratory five-arm screen and exact
  inputs are under `runs/compact_masked_lift_screen_20260717/`; it is not preregistered or
  decision-bearing, and its dense-mask and mask-derived compact-proxy arms use the same training
  masks later used by their diagnostics.
- **Result and audit**: The independent verdict is **QUALIFIED**. The four masked arms replayed
  bit-for-bit, supporting only the narrow same-training-mask mechanism finding that explicit
  occupancy repaired foreground-oriented selection. The aggregate screen `PASS` is **retired**:
  its mandatory historical full-frame control changed PLY hash from `5208181a...` to
  `2762eaff...` in the fresh replay, so the official result is not a replay-confirmed or
  deterministic pass. See
  `runs/compact_masked_lift_screen_20260717/AUDIT.md` for the allowed/forbidden claims and the
  propagated Stage-1 qualification.
- **Post-audit determinism addendum**: Read-only probes subsequently localized the control mismatch
  to byte-distinct float32 centers returned by the default CPU `torch.linalg.lstsq` `gelsy`
  driver. Candidate and depth-bin identities agreed for all 835 rows; fixed-world score reductions
  and repeated eigendecompositions were stable. The post-result repair uses explicit rank-aware
  `gelsd`, explicit baseline-preserving `gels` only for full-rank systems, and the minimum-norm
  `gelsd` result for degenerate rigs. This source change does not rehabilitate the frozen result
  without a new replay. Evidence and claim limits are in
  `runs/compact_masked_lift_screen_20260717/DETERMINISM_DIAGNOSIS.md`.
- **Conclusion and follow-up**: Keep the aggregate pass retired and retain the occupancy result only
  as qualified single-scene mechanism evidence. Re-run the full source-bound screen before making
  a reproducibility claim. Separately test an index-space near-peak window: the current floating
  depth comparison amplified the rare center perturbation, but changing it may alter the frozen
  covariance baseline and was not included in the minimal determinism repair. No default changed.

## 2026-07-17 — Footprint occupancy-scalar ablation (negative result)

- **Question**: Does summarizing a 2D Gaussian's mask occupancy over its footprint with a mean,
  normalized log-sum-exp smooth maximum, or hard maximum improve the center-sampled scalar used by
  compact-Carve at fixed $N_{\mathrm{init}}^{3D}=835$?
- **Setup**: The sealed exploratory run used the seven frozen masked 640-component teachers,
  deterministic 32-sample antithetic scrambled-Sobol Gaussian footprints, selector-only views
  `C0001,C0014,C0026`, report-only views `C0008,C0021,C0031,C0039`, and the common frozen
  center/extent. Stage B ran center, mean, and the selector-chosen LSE beta 2 with identical
  component-center rays, configuration, colors, and output budget. The command and evidence are
  under `runs/compact_occupancy_scalar_ablation_20260717/`.
- **Result**: The selector chose beta 2 by a narrow tuning precision guard, but it did not transfer.
  On report views, center precision/recall/IoU were `0.981448/0.778644/0.767349`; beta 2 changed
  them to `0.868573/0.780087/0.697728`. At Stage B, center versus beta 2 produced 1,111 versus
  1,162 eligible candidates, foreground-in-at-least-six-view fractions `0.899401` versus
  `0.871856`, and foreground MSE `0.045387` versus `0.045622`. Mean was also worse than center on
  every pooled Stage-B quality diagnostic. All three independent replay PLYs were byte-identical
  to the official files; the center PLY also exactly replayed the prior center arm.
- **Independent audit**: **QUALIFIED PASS** for protocol validity and replayability, with the
  positive hypothesis rejected. A system-call-traced full replay found zero source-RGB/C1004
  opens, independently recomputed selector/AUC and Stage-B arithmetic, and passed 39 focused CPU
  tests. Upstream Stage-1 lifecycle/provenance qualifications propagate. See
  `runs/compact_occupancy_scalar_ablation_20260717/AUDIT.md`.
- **Conclusion and follow-up**: Retain the center scalar only as the current local baseline; do not
  promote a smooth/hard footprint maximum or claim center is globally optimal. The next refinement
  experiment uses direct continuous Gaussian-point sampling from an amplitude-times-center
  occupancy field, separates sampling density from exact compact color queries, and tests view
  balancing and target risk independently. No default changed.

## 2026-07-16 — RGB-free compact point refinement and full-resolution bounded interaction

- **Question**: Can fixed-topology 3D Gaussians learn directly from compact per-view 2D Gaussian
  teachers without RGB access, and do continuous-area or discrete-pixel teacher proposals improve
  convergence over their matched uniform controls at a fixed attempt budget?
- **Setup**: The official CPU-synthetic comparison used three frozen seeds, 120 optimizer steps,
  128 attempts per step, identical initial states, and four arms: pixel-uniform versus discrete-
  pixel mixture and area-uniform versus continuous-area mixture. The preregistration, seal, raw
  record, result, and independent audit are under
  `benchmarks/results/20260716_compact_point_training_*`. The exact lifecycle commands were:

  ```bash
  .venv/bin/python benchmarks/compact_point_training.py seal
  .venv/bin/python benchmarks/compact_point_training.py run
  .venv/bin/python benchmarks/compact_point_training.py calibrated
  ```

  After the audited synthetic result, the bounded calibrated interaction acquired seven full-
  resolution 5328x4608 views (C0001, C0008, C0014, C0021, C0026, C0031, C0039), fitted 640
  StructSplat Gaussians per view for 100 iterations, serialized the exact compact fields, ran
  compact-Carve, and performed 40 fixed-topology steps in fresh RGB-denied workers. RGB was allowed
  only during teacher acquisition and post-training evaluation. The complete calibrated lifecycle
  also required exact initial/final native gsplat snapshots and an HTTP viewer smoke.
- **Result**: The official synthetic decision was `NO_GLOBAL_SAMPLING_WIN`. Relative to uniform,
  discrete-pixel mixture had geometric-mean initial/final/AUC loss ratios
  `0.5537714436 / 1.0681355694 / 1.0245665262` and lost final/AUC direction in all three seeds.
  Continuous-area mixture ratios were `0.5079419542 / 0.9873547158 / 0.9910818462`; all three AUC
  directions favored the mixture, but its AUC ratio missed the preregistered `0.95` materiality
  threshold.

  The full-resolution phase fitted its seven teachers to mean full-image training PSNR 19.486 dB.
  Its seven compressed teacher archives contained 4,480 2D Gaussians and totaled 140,945 bytes;
  the separate manifest was 4,146 bytes. Compact-Carve formed 3,340 candidates, found 1,433
  eligible, and selected exactly $N_{\mathrm{init}}^{3D}=N_{\mathrm{opt}}^{3D}=835$. All five
  effective degree-zero parameter families moved over 40 RGB-denied steps; the empty higher-order
  SH Adam group clock advanced without parameter motion. Equal-view compact-teacher MSE fell from
  `0.2846208576` to `0.2267813044` (-20.32%). On held-out C1004, 4,096 RGB-evaluation samples
  changed from
  `0.3904081687` to `0.3402060777` (-12.86%, +0.598 dB); 256 foreground samples improved 8.40%
  (+0.381 dB).

  The calibrated attempt is nevertheless an immutable terminal **FAIL**. At its first exact
  snapshot, the already-started Miniconda process still held a `libstdc++` exposing only CXXABI
  1.3.13 while the gsplat extension required 1.3.15; setting `LD_PRELOAD` inside that process was too
  late. The first exact-render operation failed during gsplat import, so no authorized snapshot was
  saved and the HTTP smoke was never reached. Separately labelled post-failure exact 5328x4608
  gsplat snapshots were generated only as diagnostics; they do not change the frozen outcome and
  show a coarse, blurry 835-splat reconstruction. A later non-authorizing live-viewer diagnostic
  bound PID 3179378 to its `127.0.0.1:8876` socket inode and 200 response, plus the exact initial/
  final PLY hashes; its receipt is
  `runs/compact_point_training_20260716/postfailure_viewer_diagnostic.json`.
- **Independent audit**: The synthetic audit verdict is `PASS` for the literal frozen proposal
  comparison and its `NO_GLOBAL_SAMPLING_WIN` decision. The calibrated failure audit accepts only
  the acquisition, RGB-denied optimization, and held-out values as phase-local diagnostics and
  forbids an overall success or end-to-end capability claim. Evidence is
  `benchmarks/results/20260716_compact_point_training_AUDIT.md`,
  `runs/compact_point_training_20260716/calibrated_result.json`, and
  `runs/compact_point_training_20260716/CALIBRATED_FAILURE_AUDIT.md`.
- **Conclusion**: Direct compact-field refinement is mechanically viable in this bounded run, but
  teacher-density proposals did not beat uniform sampling under the frozen synthetic protocol, and
  835 fixed splats are visibly insufficient for this full-resolution scene. No default changed.
  The harness now routes exact-snapshot attempts through a fresh preload-inheriting spawn worker;
  new plans bind the requested/resolved library paths, SHA-256, and required CXXABI symbol/version,
  and the child proves the default-namespace library with `dlvsym`, `dladdr`, and `/proc/self/maps`.
  Focused tests and a separate preload-started diagnostic pass
  (`runs/compact_point_training_20260716/postfailure_abi_diagnostic.json`), but no real spawned
  gsplat/CUDA render has validated the repaired complete lifecycle. A new preregistered namespace
  is required before this can support a success claim.
- **Follow-ups**: Keep uniform sampling as the baseline. Test density control with explicit variable
  $N_{\mathrm{opt}}^{3D}$, comparing residual/responsibility-driven allocation against a matched
  uniform birth/death control. Before scale claims, add aggregate device-byte/index budgets,
  CSR/lazy indexes, indexed CUDA teacher queries, and bounded backward activation memory. Before
  replay, exercise the bound worker with a real gsplat/CUDA render inside the new once-only
  lifecycle.

## 2026-07-16 — Sparse point compositor and discrete-pixel proposal parity (CPU prerequisite)

- **Question**: Can compact 2D teachers supervise selected pixels without materializing dense 3D
  renders, while preserving the dense CPU renderer's camera-wide visibility, global depth order,
  alpha compositor, parameter gradients, and ordinary uniform discrete-pixel risk?
- **Setup**: A preregistered, sealed, CPU-only mechanism experiment on revision
  `2dddca4aff59702341af9faceefa76ad2505dd83`. Phase A used three preregistered synthetic seeds
  whose fixtures and outcomes remained unconstructed and unseen until the run, two backgrounds,
  SH degrees 0/2, all nine pairs of point/Gaussian chunk sizes, four supplemental
  activation/kernel modes, and a separately constructed float64 discrete-risk fixture. The
  exclusive official commands were:

  ```bash
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python benchmarks/point_rasterizer_parity.py seal
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python benchmarks/point_rasterizer_parity.py run
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python benchmarks/point_rasterizer_parity.py calibrated
  .venv/bin/rtgs view \
    --gaussians runs/dataset_viewer_fullres_20260716/gaussians_init.ply \
    --scene dataset/2025_03_07_stage_with_fabric/frame_00008 --downscale 16 \
    --device cpu --rasterizer torch \
    --snapshot-dir runs/point_rasterizer_parity_20260716/viewer_snapshots \
    --host 127.0.0.1 --port 8767 --no-open
  ```

  The `run` lifecycle is intentionally one-shot and fails closed if repeated. The calibrated
  interaction was authorized only after an independent Phase-A audit.
- **Result**: Phase A passed all 108 forward arms: worst absolute color/alpha/depth errors were
  `5.9604645e-08`, `1.1920929e-07`, and `2.3841858e-07` against the dense anchor. All 27 gradient
  arms passed for means, quaternions, log-scales, opacity, SH, and retained `means2d`; the worst
  absolute error was `1.8626451e-09`. A non-proposer near Gaussian changed color by
  `0.3537486792`, confirming global rather than lineage-filtered compositing. On the exact finite
  pixel fixture, the enumerated target and importance-corrected expectation both equal
  `55/96`; the 64-seed pooled Monte Carlo error was `0.0075645968`, below its frozen
  `0.0141208072` gate, and fixed-attempt microchunk discrepancy was `2.2204460e-16`.
  The no-RGB calibrated interaction read only the existing 835-vertex PLY and C0001 calibration,
  sampled 4,096 replacement draws (3,998 unique) from the 333x288 downscale-16 pixel domain, and
  passed with worst absolute color/alpha/depth errors `8.9406967e-08`, `1.7881393e-07`, and
  `4.7683716e-07`. Its recorded 1.390 s wall time combines both renderers and provenance checks
  and is not a speed measurement. The separate live viewer HTTP/UI smoke loaded its normal RGB
  references and its exact Torch/CPU action saved scene camera 0 (`C0000`) as
  `viewer_snapshots/final_camera_0000.png` (333x288 RGB, 835 splats); this was not another C0001
  parity check. Machine evidence
  is `benchmarks/results/20260716_point_rasterizer_parity_RESULT.json`,
  `runs/point_rasterizer_parity_20260716/calibrated_parity.json`, and the adjacent independent
  `_AUDIT.md`; the viewer receipt is `runs/point_rasterizer_parity_20260716/viewer.log`.
- **Independent audit**: Verdict `PASS` for the literal synthetic and calibrated CPU parity gates.
  The referee independently recomputed artifact/source/input/sample bindings, all reductions, and
  the discrete expectation. It narrowed the arbitrary-coordinate result: all tested coordinate
  gradients were finite but exactly zero, so this is not evidence for active off-grid coordinate
  differentiation. Calibrated parity covers only the frozen pixel sample, not all 95,904 pixels.
- **Conclusion**: The repository now has a correctness anchor for sparse selected-point rendering
  and an O(component)-state unbiased estimator of uniform discrete-pixel risk. Pair temporaries are
  bounded by both chunk controls and proposal state avoids a dense pixel table, but no end-to-end
  memory or runtime scaling was measured. This experiment does not establish optimization,
  convergence, quality, compact refinement, density control, CUDA/gsplat parity, or a new default.
- **Follow-ups**: Add a bundle-only fixed-topology trainer that queries each teacher independently,
  compare uniform, continuous-area, and discrete-pixel proposals under matched attempts, and only
  then test split/merge/prune growth toward $N_{\mathrm{opt}}^{3D}$. Add a nonzero off-grid
  derivative fixture before relying on continuously sampled coordinates.

## 2026-07-16 — RGB-free compact-Carve initialization (CPU mechanism only)

- **Question**: Can a standalone initializer consume only calibrated compact teachers, keep the
  3D initialization budget independent of all per-view 2D counts, and score source-proposed rays
  against every view without allocating source images or a dense voxel grid?
- **Setup**: CPU-only deterministic tests on the existing dirty research tree at revision
  `2dddca4aff59702341af9faceefa76ad2505dd83`. The mechanism fixture has two 32×32 calibrated
  cameras, four colored planar targets per teacher, seed 17, 32 depth samples per ray, and a fixed
  candidate multiplier. It tests the standalone `CompactInitializer` protocol and
  `CompactCarveInitializer`, not the production CLI/pipeline. Commands:

  ```bash
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
    tests/test_compact_carve.py
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
    tests/test_compact_carve.py tests/test_observation2d.py \
    tests/test_structsplat_observation.py tests/test_reconstruction_inputs.py \
    tests/test_lift.py
  ```

- **Result**: All 23 focused mechanism cases pass. On successful initialization the output has the
  requested $N_{\mathrm{init}}^{3D}$; changing only $N_{\mathrm{init},i}^{2D}$ metadata changes
  nothing, and an exact co-located identical-component amplitude split changes a teacher count from
  four to five without changing the tested scores or initialized geometry beyond tolerance. A
  second teacher changes source-ray scores, confirming coverage-weighted all-view scoring rather
  than parent-color supervision. Bundle reload and initialization succeed with PIL decoding patched
  to fail. Point batches, component chunks, and reference-backend point–component pairs are capped
  and instrumented; the static tile-overlap index is not. The source camera-depth spread is converted
  to Euclidean ray-axis sigma, and the regression test verifies both the conversion and covariance.
  Invalid discrete configs, swapped built-in teacher backends, insufficient view count, and
  insufficient eligible candidates fail closed.
- **Independent audit**: The results-audit scientist pass found and caused repair of two substantive
  gaps: uncapped local point–component temporaries and an off-axis depth/ray sigma unit mismatch. Its
  addendum accepts the corrected synthetic CPU mechanism scope. It explicitly withholds runtime,
  peak-memory, arbitrary fragmentation, strict held-out isolation, calibrated-data, CUDA, viewer,
  reconstruction-quality, refinement, and production-default claims.
- **Conclusion**: The compact bundle now has a correctness-first CPU initialization consumer.
  Source lineage remains hard for ray proposal and initial covariance, but never selects a teacher
  target or rendering subset. This is not yet the desired global differentiable 3D compositor, and
  the reference query bounds do not make the overlap index or whole initializer proven scalable.
- **Follow-ups**: (1) add a point-rasterizer protocol and prove selected-pixel forward/gradient
  parity with the dense CPU rasterizer; (2) prove discrete and continuous estimators separately on
  an exact tiny risk; (3) add a bundle-only fixed-topology compact trainer; (4) filter or provenance-
  bind sparse points/bounds to training views; then (5) preregister a freshly exported calibrated
  full-resolution run, freeze the checkpoint, evaluate RGB only afterward, and smoke-test the saved
  initial/final PLYs in the viewer.

## 2026-07-16 — Exact RGB-free StructSplat teacher contract (mechanism only)

- **Question**: Can the terminal compact 2D fields be preserved as
  CPU-reference-equation-matched compact supervision
  without retaining source RGB, while keeping the per-view initialization and optimized counts
  independent?
- **Setup**: CPU tests on repository revision `2dddca4aff59702341af9faceefa76ad2505dd83`
  in the existing dirty research tree and local dirty StructSplat source revision
  `5dc649397c40e69cf3e96bd27df2c5e2812d003d`. The optional dependency parity fixture covers
  normalized/additive rendering, off-canvas means, opacity, rotation, affine color, covariance
  filtering, AA dilation, support fade, and translated crop clipping. A one-iteration 16-component
  public `fit_image` library-entrypoint smoke also exercises live-field export, reload, and
  querying with image decoding patched to fail. Commands:

  ```bash
  PYTHONPATH=src python3 -m pytest -q \
    tests/test_observation2d.py tests/test_structsplat_observation.py \
    tests/test_reconstruction_inputs.py
  ```

- **Result**: All 22 focused tests pass locally with the optional StructSplat dependency installed.
  Complete CPU pixel-grid queries on the three-component 4×5 float64 fixture match the independent
  StructSplat CPU renderer at the test's `1e-12` tolerance. Exact amplitude splitting preserves field
  numerator/denominator/color and proposal density. The null-thinned proposal records
  `continuous_area` explicitly, and uses O(`N_opt,2D`) proposal-component state. The reference
  query is O(samples x components); the optional sparse index stores O(component x overlapped
  tiles) entries and agrees with the all-component reference. Neither path has a performance or
  memory benchmark. Integrity-checked archives and
  camera/teacher bundles round-trip; the post-Stage-1 schema declares no RGB, mask, or source-path
  member, and reload/query succeeds with image decoding disabled. Free-form identifiers may still
  contain path-like text. Live exports record the provider version, a digest of selected source
  files under the imported StructSplat package, and the effective external `FitConfig`; these fields
  are not replay-complete execution provenance and omit the rtgs source, input, seed/RNG state,
  environment, and compiled binary.
- **Conclusion**: The compact teacher and typed/serialized no-RGB seam are ready as a correctness substrate,
  not as a reconstruction result. Converted `Gaussians2D` files—including the seven existing
  full-resolution fits—remain initialization-only because they discard normalized-renderer
  semantics. Continuous sampling deliberately changes the risk measure relative to ordinary
  discrete-pixel fitting; no equivalence, speed, memory, convergence, quality, capacity, or default
  claim is authorized yet. This does not prove CUDA or arbitrary continuous-coordinate parity, and
  it does not erase RGB from a process whose caller retains the original `SceneData`. The tested
  library entrypoint does not cover the CLI, calibrated data, adaptive growth, crop/mask handling,
  CUDA/tiled rendering, or a live `N_init,2D != N_opt,2D` fit.
- **Follow-ups**: (1) consume `ReconstructionInputs` in an RGB-free initializer with independent
  `N_init,3D`; (2) add the sampled 3D-to-2D-field refinement seam and dynamic `N_opt,3D`; (3)
  preregister discrete-pixel versus continuous-area sampling and indexed versus reference parity;
  (4) refit/re-export a calibrated dataset view, run held-out evaluation only after freezing, and
  save viewer-ready initial/final PLYs before considering the branch complete.

## 2026-07-16 — Native-resolution calibrated-data viewer handoff (integration only)

- **Question**: Can the repository use the supplied 5328×4608 calibrated images without an image
  downscale and hand the resulting lifted initialization to the interactive viewer on this host?
- **Setup**: Revision `2dddca4aff59702341af9faceefa76ad2505dd83` in the existing dirty research
  tree. `frame_00008` was loaded with `--downscale 1 --max-images 8`; cameras
  `C0001,C0008,C0014,C0021,C0026,C0031,C0039` were training-only and `C1004` remained held out.
  Because the native Stage-1 renderer exhausted the 8 GB GPU on its first full-resolution update,
  a memory-bounded streaming driver fit one training view at a time through the supported
  StructSplat `cuda_tiled` backend. A non-scientific, single-view presentation-quality pilot chose
  640 fixed Gaussians and 100 updates; this post-outcome choice forbids method or capacity claims.
  The CUDA extension required
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. The effective config, per-view histories,
  input/calibration/source SHA-256s, environment, and artifact hashes are in
  `runs/dataset_viewer_fullres_20260716/fit_manifest.json`. The temporary streaming driver's exact
  source was not archived, so the run is explicitly not replay-complete. The saved train-only fits
  were lifted through the production CLI:

  ```bash
  .venv/bin/rtgs lift \
    --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
    --downscale 1 --max-images 8 \
    --fits runs/dataset_viewer_fullres_20260716/fits --fit-format native \
    --lifter carve --lifter-args '{"grid_res":48}' --device cpu \
    --out runs/dataset_viewer_fullres_20260716/gaussians_init.ply
  ```
- **Result**: All seven training fits consumed native 4608×5328 tensors; their mask crops ranged
  from 1353×3432 to 3295×3519. Masked training-view fit PSNRs were
  `21.693,22.331,22.709,18.346,22.352,17.098,19.624` dB (mean `20.593` dB). These are source-fit
  diagnostics, not held-out reconstruction metrics. Default-resolution Carve produced 835 finite
  degree-0 3D Gaussians. No 3D refinement or held-out 3D rendering ran: full-resolution Torch
  autograd is infeasible at this pixel count and the environment's GaussianImage `gsplat` fork
  lacks the repository's modern 3D rasterization API. Consequently `gaussians.ply` intentionally
  equals `gaussians_init.ply` (SHA-256
  `0bed5a18609d560371f621634aaae915ea3e6ac0f834584f729c616c9821059d`). Static full-resolution
  contact sheets were also omitted because the eight-view sheet alone would allocate about 2.2
  GiB. The native-resolution calibrated scene and saved initialization are live at
  `http://127.0.0.1:8080` through:

  ```bash
  .venv/bin/rtgs view \
    --gaussians runs/dataset_viewer_fullres_20260716/gaussians.ply \
    --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
    --downscale 1 --max-images 8 --device cpu --rasterizer torch \
    --snapshot-dir runs/dataset_viewer_fullres_20260716/viewer_snapshots \
    --host 127.0.0.1 --port 8080 --no-open
  ```
- **Conclusion**: Native-resolution masked Stage 1, train-only lifting, saved-Ply loading, and the
  live calibrated viewer handoff work on this host. This initialization-only integration run makes
  no held-out 3D quality, refinement, capacity-ranking, runtime, GPU-performance, default, or
  modern-gsplat claim. Viewer frustum thumbnails are intentionally display-sized; the selected
  reference panel retains the full-resolution image payload.
- **Independent audit**: `runs/dataset_viewer_fullres_20260716/AUDIT.md` confirms all input and fit
  hashes, native dimensions/crops, the seven finite 640-row fit archives, exact mean accounting,
  both identical finite 835-splat PLYs, and the live documented viewer process. It narrows split
  isolation and per-fit PSNR to non-replay-complete recorded diagnostics because the temporary
  streaming driver was not archived and the individual fit renders were not independently
  recomputed.
- **Follow-ups**: Add a tested streaming calibrated-fit CLI so future full-resolution runs preserve
  exact argv/source provenance. Use a clean environment with official modern 3D gsplat before
  attempting native-resolution refinement, held-out rendering, or exact viewer snapshots. Freeze
  capacity and validation policy prospectively before any decision-bearing real-data comparison.

## 2026-07-16 — Local calibrated-data and viewer workflow smoke (integration only)

- **Question**: Can the standing research workflow load the repository's local calibrated
  dataset with a strict held-out camera, save all reconstruction/preview artifacts, and launch the
  interactive viewer on this host?
- **Setup**: Revision `2dddca4aff59702341af9faceefa76ad2505dd83` in an explicitly dirty research
  tree. The CPU smoke used eight evenly sampled calibrated cameras from
  `dataset/2025_03_07_stage_with_fabric/frame_00008` at `--downscale 64`; the loader assigned
  seven cameras to training and `C1004` to held-out reporting. It ran native Stage 1 with 60
  fixed Gaussians/image for 15 updates, `carve(grid_res=24)`, three fixed-count Torch refinement
  updates, SH degree 0, and no density control:

  ```bash
  .venv/bin/rtgs run \
    --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
    --downscale 64 --max-images 8 --device cpu --fit-backend native \
    --initial-gaussians 60 --max-gaussians 60 --fit-iterations 15 \
    --lifter carve --lifter-args '{"grid_res":24}' --refine-iters 3 \
    --rasterizer torch --no-densify --target-sh-degree 0 \
    --out runs/dataset_viewer_smoke_20260716
  ```
- **Result**: The run recorded a non-decisional 1.278 s elapsed time and saved 129 initial/final 3D
  Gaussians, metrics/history, calibrated reference/init/final/error panels, and two novel-view
  animations. Held-out foreground PSNR was 16.850 dB at initialization and 17.287 dB after the
  three smoke updates; held-out crop PSNR was 21.763/22.206 dB and alpha IoU was 0.797/0.783. Viser
  1.0.30 then launched successfully, loaded the sibling initial model plus the calibrated scene,
  and served HTTP/WebSocket at `http://127.0.0.1:8080`:

  ```bash
  .venv/bin/rtgs view \
    --gaussians runs/dataset_viewer_smoke_20260716/gaussians.ply \
    --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
    --downscale 64 --max-images 8 --device cpu --rasterizer torch \
    --snapshot-dir runs/dataset_viewer_smoke_20260716/viewer_snapshots \
    --host 127.0.0.1 --port 8080 --no-open
  ```
- **Conclusion**: The mandatory local-data → strict split → saved reconstruction → live viewer
  handoff works on this machine. The deliberately tiny 1/64, three-update run is an integration
  smoke only and supports no quality, method-ranking, runtime, or default claim.
- **Independent audit**: Re-loading the scene confirmed the 7/1 split and that all sampled
  refinement views were training views. Independent Torch/CPU renders from both finite 129-vertex
  PLYs reproduced every recorded metric within `2.8e-7`; the viewer process remained live and
  returned HTTP 200. This is not replay-complete evidence: the run directory lacks full argv,
  fit/lifter/input/split/source hashes, environment provenance, and a persisted exact viewer
  snapshot. The shared environment also shadows the required modern 3D gsplat API with the
  GaussianImage 2D fork, so this smoke validates only the explicit Torch path, not CUDA/gsplat.
- **Follow-ups**: Require a frozen `dataset/` interaction and a viewer-ready output directory for
  every new research branch. Preserve synthetic scenes for deterministic mechanism gates. Run
  decision-bearing real-data experiments at useful resolution, use train-only validation for
  selection, keep held-out cameras reporting-only, and replicate beyond this frame before making a
  dataset-level claim. Persist at least one exact train and held-out viewer snapshot plus a complete
  provenance manifest for future durable handoffs.

## 2026-07-16 — Quaternion radial-gauge optimizer audit (invalid; no optimizer outcome)

- **Question**: Does the exact positive radial gauge of a normalized quaternion make ambient Adam
  materially representation-dependent, and, only if so, can entry canonicalization or a
  post-update unit/tangent retraction improve joint refinement?
- **Setup**: The original protocol in
  `benchmarks/results/20260716_quaternion_gauge_PREREG.md` froze CPU synthetic seeds 0/1/2, a
  top-128 anisotropic subset, radial scales 0.25/1/4, five 40-step quaternion-only policies, and a
  materiality gate before any 120-step joint-refinement arm. Its first sealed Phase-A attempt
  failed before a result because the producer formed one diagnostic as
  `float64(normalize_float32(q))` while validation used
  `normalize_float64(float64(q))`. The independent invalid-artifact audit is
  `benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md` (SHA-256
  `7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d`). A prospectively
  frozen append-only repair then made producer and validator share the same promote-first
  float64 diagnostic while leaving the native float32 projection, Adam update, replay, seeds,
  arms, schedules, and gates unchanged. Retry-2 is bound by
  `benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md`, its seal, and the consumed
  `benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_A_ATTEMPT.json`.
- **Result**: Retry-2 also failed closed before a materiality decision. Its invalid artifact is
  `benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid.json` (SHA-256
  `56df44d380ede52dba568b068685d9ffd1dbd625fe9ef92e8f31559660e0af0b`); the independent
  audit passed only its invalid disposition (SHA-256
  `b4492303d9dd688e1685eb886c90cbf94ceeefc698c48bb38667ea1cfd57d866`). All retained
  preparation and prerequisite records recomputed cleanly. However, native float32
  `F.normalize(c*q)` changed the direction seen by the later float64 covariance audit by
  **1.08e-8 / 2.06e-8 / 1.51e-8** across seeds. Step-zero covariance maximum errors were
  **6.13e-10 / 2.05e-9 / 9.53e-10** (relative **4.40e-10 / 1.20e-9 / 7.28e-10**), necessarily
  above the inherited `2e-12` absolute and relative gates for every scale. The fail-closed JSON
  contains no arms, trajectories, checkpoints, AUC, materiality decision, or Phase-B clearance.
- **Conclusion**: There is no evidence here that ambient Adam is or is not materially affected by
  quaternion radial gauge, and no evidence favoring any canonicalization, projection, or
  retraction policy. Phase B was forbidden and no default changed. The concrete finding is
  methodological: an exact algebraic gauge still needs a precision-aware feasibility contract
  when the intervention canonicalizes in float32 but validation compares a second normalization
  at float64-scale tolerances.
- **Follow-ups**: Do not relax the consumed threshold or reuse either marker. Any future retry
  needs a fresh preregistration and namespace, an analytically justified float32 covariance
  margin (or a different exact representation contract), and a pre-optimizer feasibility check.
  Prioritize the already-frozen Stage-1 appearance-parameterization and residual-responsibility
  allocation questions before spending another official attempt on this validity repair.

## 2026-07-16 — Gauge-invariant Stage-1-to-lifter semantic factorial (valid negative joint repair)

- **Question**: In a three-seed deterministic CPU-synthetic experiment, can the non-identifiable
  fitted `(weight,color)` boundary be replaced by the gauge-invariant scalar
  `m=max(weight*color)` and a source-observation RGB color while preserving or improving both
  Depth and Carve at matched per-view capacity and fixed refinement budget?
- **Setup and validity**: The frozen four-arm factorial crossed fitted weight versus `m` with
  fitted color versus sampled source RGB on fresh seeds 4409/5519/6637. Phase A independently
  cleared the frozen tolerance-bound source-render, coverage/retention, and ordinary-lift
  invariance gates before Phase B; separate product-preserving controls passed their frozen
  field/product tolerances.
  Phase B then built all six capacity cells and 24 matched initializations before optimization,
  ran 24 fixed-topology 120-step refinements, and unlocked 216 held-out render cells only after
  every model completed. The valid result JSON is
  `benchmarks/results/20260716T063637Z_cpu_stage1_semantic_factorial_utility.json` (SHA-256
  `005eabffc062e158c1ca510865fa40be799733bc5f9bc6c4c3444fff63fc0d9c`); its independent
  unqualified-PASS audit is the adjacent `_AUDIT.md` (SHA-256
  `4d197a040fa01cf105955db77adaa993b0d403426d29c3bd20287c4917401df6`). The reviewer
  validated and rehashed all 13,944 raw arrays, audited all 24 lift cells, regenerated all 24
  schedules, and recomputed metrics for all 1,080 training-checkpoint render cells and 216
  held-out render cells, plus every factorial estimand and decision, without replaying an official
  seed.
- **Result**: The full `m_amp__rgb_obs` candidate materially improved Depth in every seed: mean
  final PSNR/SSIM differences were **+3.127230 dB / +0.0248165**, with worst-seed PSNR
  **+2.702274 dB**. It failed Carve non-inferiority in every seed: mean differences were
  **-2.205314 dB / -0.0400166**, with worst-seed PSNR **-2.408451 dB**. The PSNR factorial means
  attribute Depth to color (**+3.127289 dB**) while its scalar effect was negligible
  (**-0.000059 dB**). Carve had a positive color effect (**+2.326600 dB**) overwhelmed by the
  scalar (**-4.531913 dB**) and interaction (**-2.516372 dB**) effects.
- **Conclusion**: The evidence is valid, but the proposed joint repair does not survive across
  backends: `repair_utility_survives=false`, `cross_backend_material_improvement=false`, and no
  default change is authorized. Carve's matched count and schedule were exact, but changing the
  scalar also changed coverage, retention, source-key availability, and tunnel placement; the
  Carve selected-set Jaccard with the fitted-scalar arm was only about 47%-66%. The scalar effect
  is therefore a total downstream effect, not a direct opacity coefficient at fixed
  correspondence.
- **Follow-ups**: Retain `w_fit__c_fit`. Do not tune this consumed namespace or select the positive
  color-only arm post hoc. A color-only replacement would need a fresh outcome-independent
  protocol, real-data transfer evidence, and interaction checks. Continue with the already-frozen
  Stage-1 fit-time 9p-versus-8p comparison before residual-responsibility density.

## 2026-07-16 — Stage-1 fit-time parameterization infrastructure (no scientific outcome)

- **Question**: Can the current learned `weight*color` appearance gauge be compared fairly with a
  bounded unit-weight RGB-amplitude parameterization using a common initialization, both with
  frozen geometry and during the ordinary joint native fit?
- **Implemented**: `FitConfig.appearance_parameterization` retains `weight_color_9p` as the default
  and adds native-only `unit_weight_bounded_8p`; a shared-initialization path, disabled geometry
  freeze, and detached read-only diagnostic snapshots support paired evidence collection. Focused
  masked and unmasked tests bind a test-local transcription of the frozen pre-change fitter and
  require bit-exact current outputs/history, common-forward candidate initialization, unit weights,
  absence of candidate weight optimizer state, frozen geometry, gradient-chain/Adam identities,
  callback isolation, finite raw rows, and rejection before a StructSplat import.
- **Fail-closed harness state**: `benchmarks/stage1_fit_parameterization.py` now declares the
  outcome-free implementation complete. It binds exact CLI provenance, seal/attempt chronology,
  target/initializer prerequisites, all per-update appearance-only equations, joint checkpoints,
  deterministic full-trajectory joint replay, strict raw recomputation, and finite/non-finite
  invalid-boundary evidence. Adversarial nonofficial tests close previously demonstrated joint
  checkpoint-splice and truncated/fabricated failure-evidence false accepts. Seal creation still
  requires an independent outcome-free implementation review, and a scientific run still requires
  that seal plus the sole exact process command.
- **Conclusion**: No seal, official seed, attempt marker, raw archive, decision, or scientific
  result was produced. The current representation remains the default; neither conditioning nor
  reconstruction benefit has been measured.

## 2026-07-16 — Stage-1 fit-time parameterization (valid negative result)

- **Setup and validity**: The once-only CPU-synthetic comparison used three fresh seeds per block,
  nine selected source views, 150 fixed components, eight checkpoints through 120 Adam updates,
  common initializations, and current `weight_color_9p` versus bounded
  `unit_weight_bounded_8p`. The valid artifact is
  `benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization.json`; the independent
  scientist audit is the adjacent `_AUDIT.md`, with exact machine bindings in
  `_SCIENTIST_REVIEW.json`. The reviewer rehashed all 360 raw arrays, regenerated all 54 source
  targets and 864 checkpoint renders, replayed gradient/Adam identities, and independently
  recomputed every frozen decision.
- **Appearance-only result**: The candidate lost in every seed. Mean candidate-minus-current PSNR
  AUC was **-1.330662 dB**, mean final PSNR was **-1.796120 dB**, and mean final SSIM was
  **-0.037330**. Current Adam updates nevertheless contained material local null-direction motion:
  the pooled null-energy ratio was **0.122921**, and **92.9508%** of eligible rows had null fraction
  at least 0.10. Both arms had zero weak-response rows, so the saturation guard passed. Because
  the curve gate failed, `fit_time_redundant_coordinate_interference_consistent=false`; the local
  projected motion is not evidence that a finite nonlinear update was globally wasted.
- **Joint-fit result**: The candidate again lost in every seed. Mean PSNR AUC was **-1.292971 dB**,
  mean final PSNR was **-1.501525 dB**, and mean final SSIM was **-0.048419**. Therefore
  `joint_stage1_noninferior=false` and `joint_stage1_material_improvement=false`.
- **Conclusion**: Retain the current nine-parameter fit and close this exact bounded unit-weight
  candidate on the deterministic CPU-synthetic fixed-count/fixed-budget setup without tuning. The
  one-scalar structural reduction authorizes no memory, runtime, bitrate, compression, real-image,
  downstream, CUDA, or default claim. Variable projection or another parameterization requires a
  fresh outcome-independent protocol.

## 2026-07-16 — Stage-1 weight/color gauge contract (qualified positive validity finding)

- **Question**: Can product-preserving changes to the native Stage-1 factorization leave every
  fitted source RGB reconstruction equivalent while materially changing the coverage, retention,
  Depth-lift, or Carve-lift boundary that consumes those fits?
- **Setup**: The once-only CPU synthetic protocol was frozen in
  `benchmarks/results/20260716_stage1_weight_gauge_PREREG.md` before implementation or outcome
  access. Seeds 0/1/2 used nine training views and 150 native fitted components per view. For each
  component with additive amplitude `a=w*c`, the audit compared the fitted identity against
  `unit_weight=(1,a)` and `peak_color=(max(a),a/max(a))`, with an exact zero case. All 54
  transformed source renders had to pass strict equivalence before coverage, retention, or an
  unmerged Depth/Carve lift could run. The frozen result is
  `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json` (SHA-256
  `e001d6efdfcf0beea30ae578069d6057350e47b3f3516ad95f216ae495793791`); its independent scientist
  pass is the adjacent `_AUDIT.md` (SHA-256
  `871c3235954f1025b05641385d70cd33c6160d200f74a26fb322dc20e390dfd6`).
- **Result**: Source equivalence passed with maximum raw RGB errors of **1.7881393e-7**
  (`unit_weight`) and **1.1920929e-7** (`peak_color`), minimum reported PSNR **120 dB**, and all
  `4050` components jointly changing weight and color. Nevertheless, pooled coverage
  delta/reference was **0.520168** and **0.705005**; the `0.40` coverage-threshold crossing
  fractions were **22.7993%** and **44.0619%**. Unmerged Depth render delta/signal was
  **0.581622** and **2.022173**, with peak-color output-key disagreement **9.5608%**. Unmerged
  Carve output-key disagreement was **9.2074%** and **64.1602%**, and render delta/signal was
  **0.589632** and **1.077597**. Each named transform independently passed all frozen materiality
  gates in 3/3 seeds and in its raw-sum pool for both backends.
- **Conclusion**: The current downstream boundary materially depends on a non-identifiable
  `(weight,color)` representative in this narrow setup even when Stage-1 RGB is unchanged. This
  does not identify a physically correct gauge, show held-out quality improvement, or authorize
  canonicalization or a default change. The independent verdict is **QUALIFIED** because the
  artifact stores decision-grade reductions, exact keys, and hashes rather than raw tensors; all
  derived decisions were independently recomputed, while tensor-level parity remains a sealed
  fail-closed assertion. There is no real-data, optimized, merged, CUDA/gsplat, speed, or memory
  claim.
- **Follow-ups**: Preregister a separate causal utility experiment that isolates scalar
  coverage/retention semantics from observed color semantics, proves the proposed boundary is
  invariant under the same gauges, and evaluates held-out quality at matched optimization and
  primitive budgets. Do not choose a canonical representative or alter a production default from
  this audit alone.

## 2026-07-16 — Fixed-topology 24-to-48 multiscale refinement (negative result)

- **Question**: Can a two-level 24-to-48 refinement schedule improve held-out quality or preserve
  it with fewer optimization raster pixels, and is a blocked coarse-to-fine order better than an
  exposure-matched interleaving?
- **Setup**: The protocol in
  `benchmarks/results/20260716_multiscale_refinement_PREREG.md` froze CPU synthetic seeds 3/4/5,
  nine train and three held-out views, one shared Carve initialization per seed, degree-zero SH,
  fixed topology, no density control, 120 updates, and checkpoints 0/30/60/90/120. Arms were full
  48x48 refinement, blocked camera downsampling, blocked loss-pyramid supervision, and an
  exposure-matched interleaved camera control. The official result is
  `benchmarks/results/20260716T003735Z_cpu_multiscale_refinement.json` (SHA-256
  `343263f3193871dbdae4f390d46ba9c305cb9c38bfead0dd5c7bc97448ce35fa`); the adjacent scientist
  audit passed with SHA-256
  `c736a0de3160f61f8b1df9113783576fab2f706d100687df34c0bac1a06cd394`.
- **Result**: Every candidate lost foreground-PSNR AUC to full resolution in every seed. Mean AUC
  deltas were **-0.338645 dB** (camera blocked), **-0.088758 dB** (pyramid blocked), and
  **-0.345927 dB** (camera interleaved); mean final foreground-PSNR deltas were **-0.263247**,
  **-0.203262**, and **-0.734998 dB**. Both camera arms used exactly **172800/276480 = 62.5%** of
  full optimization raster pixels, but failed quality noninferiority, so neither was exposure
  efficient. Blocked-minus-interleaved mean AUC was only **+0.007282 dB**, below the frozen
  attribution gate, and both arms were noninferior failures. The independently reconstructed
  decisions are no quality improvement, no exposure efficiency, and no blocked-order
  attribution.
- **Conclusion**: Close this exact 24-to-48, 60/60, fixed-topology, degree-zero CPU synthetic
  branch without scale, boundary, filter, loss, seed, or threshold tuning. The 37.5% raster-pixel
  reduction is exposure accounting, not a runtime speedup. This result does not reject
  parameter-specific schedules, adaptive density, full SH, real scenes, CUDA/gsplat, or
  multiscale methods generally, and it changes no default.
- **Follow-ups**: Do not combine this failed schedule with the density or gauge interventions.
  Any future multiscale question needs an independently motivated protocol, such as explicit
  geometry-versus-appearance parameter routing, rather than an outcome-tuned replay.

## 2026-07-16 — Carve equal-count merge controls (Phase-A materiality gate failed)

- **Question**: At the exact count produced by production Carve's voxel moment merge, does the
  merge preserve a meaningfully different allocation from two controls built from the same raw
  tensor—one representative per occupied voxel and one global top-weight prune—well enough to
  justify a held-out fixed-budget refinement comparison?
- **Setup**: The base protocol
  `benchmarks/results/20260715_carve_merge_controls_PREREG.md` froze seeds 0/1/2, twelve 48×48
  synthetic views (nine train, three held out), one native stage-1 fit per seed, one unmerged raw
  Carve tensor, production moment merging, two exact-count controls, construction identities, and
  a Phase-A materiality gate before any candidate refinement. The first sealed attempt stopped
  during seed 1 before artifact creation or scientific output because the producer's ordered
  binary64 left fold differed by one ULP from Python 3.12's compensated built-in `sum` in the
  validator. The failure, absent outputs, and access boundary are independently recorded in
  `benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_FAILURE_AUDIT.md` (SHA-256
  `861535cd6a99bca7ce4f49ddd66aefe3dd4965bcb40de3ed93d309363c5b7c5c`). Before replay, the
  outcome-neutral Retry-2 protocol
  `benchmarks/results/20260716_carve_merge_controls_iter2_PREREG.md` froze one explicit ordered
  left-fold representation in producer and validator, plus fresh artifact types and a fresh
  once-only marker. Its preregistration and seal SHA-256 values are
  `fd4361ab1a53a22760db72e99614abb04206c1b639602e0015d8debde91c1203` and
  `8d59df3310ad67e9e21e2979d491ab740894a6a923175c959c5bd687a91e92f8`.
- **Result**: Official Retry-2 evidence is
  `benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit.json` (SHA-256
  `1e1142b4a4301b7f05546f62d5868c64e976183b549dd305775fca43753a29cc`), independently reviewed
  in its matching `_AUDIT.md` (SHA-256
  `190a43465ac1108a7f4964766ac32e7b7cb890ff5df15486cac937cf66fd2d74`). Raw counts were
  **1156/1160/1155** and moment counts **1125/1129/1128**, only **2.68%/2.67%/2.34%** compression.
  There were **29/31/27** multi-member cells containing **5.19%/5.34%/4.68%** of raw primitives,
  below the frozen 50-cell, 15%-exposure, and 10%-compression floors in every seed. Moment versus
  voxel-control render-delta/residual ratios were **0.00567/0.00381/0.00282**; moment versus global
  control ratios were **0.01788/0.02405/0.01856**. All construction/parity checks passed, but every
  seed's complete gate was false, so `phase_b_authorized=false` and no candidate refinement ran.
- **Conclusion**: At the frozen production grid scale, Carve merging is too sparse to support the
  intended fixed-budget causal comparison. This result does **not** show that moment matching is
  worse than pruning; utility remains untested because Phase B was correctly withheld. Keep the
  current merge behavior and defaults unchanged. The evidence is CPU synthetic, fixed-scene,
  construction-level only and makes no real-data, density-control, CUDA/gsplat, speed, or quality
  claim.
- **Follow-ups**: Do not tune the consumed grid scale to force a pass. Audit earlier interfaces
  that can change what Carve receives—especially the exact stage-1 `weight*color` factorization—and
  test optimizer-coordinate and multiscale scheduling invariances under separately frozen
  protocols. A future merge study needs an independently motivated allocation mechanism or scene
  regime that produces material collisions before it can repeat an equal-count utility test.

## 2026-07-15 — Coarse visibility-margin support audit (Phase-A gate failed)

- **Question**: Does the Torch reference renderer's detached 3-sigma image-intersection cull omit
  enough genuine hard-kernel support (`q < 12`) to justify training with the exact conservative
  `sqrt(12)`-sigma visibility envelope?
- **Setup**: The original protocol
  `benchmarks/results/20260715_visibility_margin_PREREG.md` froze seeds 0/1/2, diffuse primary and
  view-dependent reporting conditions, twelve 48×48 views (nine train, three held out), one
  depth initialization per condition/seed, 120 fixed-topology CPU Torch-reference steps, and no
  density control. The first sealed Phase-A attempt completed diffuse seed 0, then failed closed
  during diffuse seed-1 initialization before output creation or candidate training: adding one
  support-safe primitive changed `torch.argsort`'s unspecified order for two current primitives at
  exactly equal float32 depth. Its seal and consumed marker remain
  `benchmarks/results/20260715_visibility_margin_SEAL.json` and
  `benchmarks/results/20260715_visibility_margin_PHASE_A_ATTEMPT.json`; the named failed JSON and
  result note are absent. Before recomputing any incidence, the retry protocol
  `benchmarks/results/20260715_visibility_margin_iter2_PREREG.md` froze a representation-only,
  baseline-preserving tie extension: keep the default current order, order new primitives
  separately, then stable-sort their concatenation. The complete replay was sealed by
  `benchmarks/results/20260715_visibility_margin_iter2_SEAL.json` and consumed
  `benchmarks/results/20260715_visibility_margin_iter2_PHASE_A_ATTEMPT.json`; official evidence is
  `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit.json`, independently
  recomputed in its matching `_AUDIT.md` note (SHA-256
  `21c262aad36f02cf9a6520d50c2d2a867a22758e0486daa35094cdd78b9eb928`).
- **Result**: Target-generation parity, current-set inclusion, support coverage, order preservation,
  finite tensors, all nine audited training views, and minimum support-pair counts passed for all
  three diffuse seeds. At the final diffuse states, only **4 of 2,480,463** pooled `q < 12`
  pixel/Gaussian pairs were omitted by 3-sigma culling, all from two Gaussian/view exposures. The
  pooled missed-pair fraction was **1.612602e-6** (gate `5e-4`), missed effective-mass fraction
  **1.646359e-8** (gate `5e-4`), and render-delta/residual ratio **3.986964e-8** (gate `1e-3`).
  Missed-count/exposure floors of 100/3 also failed. Every seed's material decision was false,
  the pooled decision was false, and `phase_b_authorized=false`. View-dependent incidence was
  recorded as reporting-only and cannot rescue the failed diffuse gate.
- **Conclusion**: In this CPU synthetic, depth-initialized, fixed-topology setup, the current
  3-sigma cull truncates genuine hard support only at immaterial incidence and mass. The frozen
  stop rule therefore forbids Phase B; no support-safe candidate was trained. Retain 3 sigma as
  the default and do not tune the margin. This does not establish real-scene behavior,
  density-control interaction, near-plane behavior, gsplat/CUDA culling parity or speed, or a
  general claim about other scenes/resolutions.
- **Follow-ups**: Close further smooth-color, smooth-kernel-tail, and visibility-margin variants on
  this setup. The subsequent Carve equal-count audit also stopped before utility because production
  grouping failed its materiality floors. Prioritize the Stage-1 representation and residual
  allocation interfaces rather than tuning another smooth gate or the consumed Carve grid scale.

## 2026-07-15 — Hard kernel-support C1 taper (mechanism passed, utility failed)

- **Question**: Does the hard reference-renderer kernel cutoff
  `exp(-q/2) * 1[q < 12]` suppress material loss-directed gradient in the immediately adjacent
  `12 <= q < 16` annulus, and, if so, do either a fixed outward C1 taper (`C=12`, `W=4`) or its
  hard-forward/taper-gradient attribution control improve held-out refinement?
- **Setup**: The original protocol
  `benchmarks/results/20260715_kernel_support_taper_PREREG.md` froze seeds 0/1/2, diffuse primary
  and view-dependent guardrail conditions, twelve 48x48 views (nine train, three held out), one
  depth initialization per condition/seed, 120 CPU Torch-reference steps, fixed topology, and no
  density control. Phase A passed and was independently cleared. The first Phase-B attempt then
  trained the diffuse seed-0 C1 arm but stopped before evaluation, aggregation, serialization, or
  result printing: an in-memory `list[tuple]` checkpoint schedule was compared directly with its
  semantically identical JSON-restored `list[list]` form. Its once-only marker remains
  `benchmarks/results/20260715_kernel_support_taper_PHASE_B_ATTEMPT.json`; the attempted result is
  absent. The retry protocol
  `benchmarks/results/20260715_kernel_support_taper_iter2_PREREG.md` froze a representation-only
  canonical-JSON comparison before a new seal, complete Phase-A replay, independent clearance,
  and fresh Phase B; renderer, trainer, initialization, and scientific choices were unchanged.
  Official retry evidence is
  `benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit.json` and
  `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`; the exact
  independent review notes are
  `benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit_AUDIT.md` and
  `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation_AUDIT.md`.
- **Result**: Phase A passed in all three diffuse seeds and pooled. From **48,290,887** pooled
  eligible observations, annulus upstream mass was **40.7745%** (1% gate), recoverable annulus
  mass **24.6717%** (10% gate), candidate-recovered mass over active hard q-gradient
  **0.252269%** (0.1% gate), and candidate-recovered mass over the hard boundary
  **5.43819%** (5% gate); all training views were sampled. Phase B rejected both arms under the
  common hard final renderer. C1-taper foreground-PSNR deltas were
  **-0.018741 / -0.013265 / -0.011443 dB** (mean **-0.014483 dB**, 0/3 wins), versus the frozen
  +0.10 dB and two-seed utility gates. The hard-forward control deltas were
  **-0.028500 / -0.013335 / -0.013576 dB** (mean **-0.018470 dB**, 0/3 wins), so both attribution
  gates also failed. All seven safety/replication guardrails passed: mean C1 SSIM delta was
  **-0.000386**, normalized depth-RMSE regression **+0.4106%**, alpha-IoU delta **+0.002942**,
  coverage delta **-0.000398**, and view-dependent PSNR deltas
  **[-0.009253,+0.011937,-0.011356] dB** (mean **-0.002891 dB**).
- **Conclusion**: The adjacent hard-support annulus carries material local loss-directed gradient
  under this exact CPU synthetic, depth-initialized, fixed-topology protocol, but exposing it with
  either prespecified arm did not improve common-hard held-out quality. Guardrails cannot rescue
  the failed primary and attribution gates. Reject this `C=12`, `W=4` taper branch, retain its
  implementation only as opt-in research/diagnostic infrastructure, and keep the hard kernel as
  the default. This is not evidence that every support smoothing is ineffective, and it makes no
  real-scene, density-control, gsplat/CUDA, speed, or production-default claim.
- **Follow-ups**: Do not tune width, shape, cutoff, loss, learning rate, schedule, iterations,
  seeds, or visibility margin from this result. If support-boundary work continues, preregister a
  separate hard-only incidence audit of the detached image-intersection visibility cull, comparing
  its current 3-sigma envelope with the support-safe `sqrt(12)` envelope. That audit has not run
  and must not be combined with another taper intervention.

## 2026-07-15 — SH color-floor incidence and SMU-1 (Phase-A gate failed)

- **Question**: During fixed-topology refinement, does the standard hard nonnegative SH-color
  floor suppress enough loss-directed gradient to justify testing SMU-1 or a hard-forward,
  negative-gradient-only SMU-1 control?
- **Setup**: The hard-only Phase-A protocol was frozen in
  `benchmarks/results/20260715_sh_activation_PREREG.md` and incorporated unchanged by the
  provenance retry `benchmarks/results/20260715_sh_activation_iter2_PREREG.md`. Seeds 0/1/2 used
  twelve 48×48 synthetic views (nine train, three held out), one pinned depth initialization per
  condition/seed, 120 CPU Torch-reference refinement steps, SH degrees 0–3, and no density control.
  The audit covered diffuse and deliberately view-dependent targets; only the latter could open
  Phase B. SMU-1 was
  fixed at `alpha=0`, `mu=2/255`, with no sweep. The first sealed attempt was consumed after six
  hard-arm trainings but failed before artifact creation when `.venv` Pillow modules were
  misclassified as repository source; it printed no diagnostic fraction or quality outcome. The
  retry used seal SHA-256
  `403ce133922f57fa45a3374be34cb92a85fb043d0a1a6ce188c82fc808370de0`; its official JSON and
  independent audit are
  `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit.json` and
  `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit_AUDIT.md`.
- **Result**: Every view-dependent seed failed all three materiality gates. Negative-channel
  incidence was **0.516717% / 0.245288% / 0.243935%** (pooled **0.336527%**) against 1%; recoverable
  blocked-gradient mass was **0.107542% / 0.142238% / 0.017500%** (pooled **0.090828%**) against
  5%; and fixed-SMU-1 recovered mass was **0.037962% / 0.030720% / 0.006035%** (pooled
  **0.025266%**) against 0.5%. All nine training views were sampled and per-seed observation counts
  were 226,236 / 224,226 / 219,321, so the coverage/count checks passed. The independently
  recomputed decision is seed passes `[false,false,false]`, pooled pass `false`, and
  `phase_b_authorized=false`.
- **Conclusion**: Under this exact CPU, fixed-topology, synthetic, depth-initialized protocol, the
  hard SH color floor was not a material optimization bottleneck. Phase B is permanently forbidden
  by the frozen stop rule, not awaiting a favorable review; neither SMU-1 nor its attribution
  control was trained. The retry is decision-usable but not fully replay-complete: its environment
  fingerprint omitted the Pillow version, and the first-attempt harness source/diff needed to
  independently prove a classifier-only retry is unavailable. The old seal/attempt artifacts are
  retained but were not themselves in the retry's sealed-path set. None of these caveats can turn
  the failed gate into evidence about a candidate arm.
- **Follow-ups**: Close SMU parameter/seed/schedule tuning on this setup. If smooth-support work
  continues, preregister a separate hard raster-support-cutoff incidence audit before any taper
  arm. This result makes no candidate-quality, real-scene, CUDA/gsplat, density-interaction,
  performance, or production-default claim; the hard activation remains unchanged.

## 2026-07-15 — Signed RGB-D occlusion attribution (development gate failed)

- **Question**: Does a denser construction-only T-depth z-buffer selectively remove
  behind-observed residuals strongly enough to attribute the prior TUM heavy tail to sparse
  occlusion handling, and thereby authorize a slow/fast-motion contrast?
- **Setup**: Before any archive/PNG access, the fixed protocol, official TUM
  `fr3/sitting_xyz`/`fr3/walking_xyz` source hashes, standalone harness, sealed base dependency,
  and 21 tests were bound by
  `benchmarks/results/20260715_tum_rgbd_signed_attribution_PREDECODE_SEAL.json`. Sitting used the
  same 48 T/eight V/eight H pose-only split and stride-16 oriented audit targets as the prior
  experiment. The new arm explicitly unioned those targets with valid stride-8 T-only points;
  neither visibility mask accepted V depth. Signed camera-z residuals and target-cluster bootstrap
  intervals were computed only after visibility. Full commands/provenance are in
  `benchmarks/results/20260715_tum_rgbd_signed_attribution_RESULT.md`.
- **Result**: The dense set retained 155,416/176,950 depth-valid pairs (87.83%) and 27,135
  two-view targets. Removed pairs were 30.11% positive versus 2.90% negative, capturing 32.48% of
  positive but only 5.52% of negative contradictions. Target-paired `E+=0.1381` with bootstrap
  interval `[0.1315,0.1444]`; p90 relative depth improved from 5.226% to 4.809%. Nevertheless,
  target-balanced positive rate fell only 1.424 pp (11.674% to 10.250%), below the frozen 1.751 pp
  floor, and removed/retained positive risk ratio was 1.7195 (interval 1.6767-1.7595), below 2x.
  Ten other support/selectivity/safeguard comparisons passed. Dense-visible far-minus-near
  contradiction increased 11.19 pp and remained +10.01 pp in the pose-conditioned sensitivity.
- **Conclusion**: Sparse construction visibility explains a real, sign-selective portion of the
  tail, but not the preregistered target-balanced amount. Reject attribution under this protocol;
  the result is partial mechanism evidence, not authorization for an oriented loss or utility
  run. The stopped decision left `fr3/walking_xyz` completely unopened and no confirmatory seal
  exists.
- **Follow-ups**: Do not tune density/tolerance or relax the consumed sitting gates. The next
  admissible attribution experiment should use new captures and compare pooled versus
  time-local/source-conditioned T-only visibility at matched pose baselines; the strong residual
  temporal effect makes scene-state aggregation the sharper hypothesis.

## 2026-07-15 — Real registered-RGB-D oriented points (confirmatory transfer failed)

- **Question**: Does a CPU-first pluggable registered-RGB-D backend produce metric points and
  depth-Jacobian normals that remain independently consistent across calibrated views of a second
  real static-scene sequence, strongly enough to authorize point-to-plane/shortest-axis utility
  testing?
- **Setup**: The target constructor, T/V/H pose-only split, nine metrics, development-transfer
  formulas, exact payload isolation, and one-shot desk stop rule were frozen in
  `benchmarks/results/20260715_tum_rgbd_oriented_validity_PREREG.md` before any PNG decode. Official
  TUM `fr1/xyz` supplied the one development run and `fr1/desk` the sole confirmatory run. Each
  phase selected 48 construction views, eight independent validation views, and eight sealed
  future-utility views. A harness-local backend decoded only T/V registered depth, estimated
  five-point normals, and used disjoint T-only and V-only capability maps. Commands and complete
  provenance are in `benchmarks/results/20260715_tum_rgbd_oriented_validity_RESULT.md`.
- **Result**: Development produced 40,341 eligible targets (`A=0.70036`) with 34,970 two-view
  oriented supports (`S=0.86686`), `R90=24.97 mm`, `D90=3.10%`, `C50=0.91516`, and
  `C10=0.67197`. Desk retained broad coverage/support (`A=0.68441`, `S=0.80359`) and passed
  `A`, `A_min`, `S`, `S_10`, `C50`, and `F`. It failed the transferred `R90` gate at
  **202.11 mm** (42.45 mm limit), `D90` at **25.19%** (5% limit), and `C10` at
  **0.50262** (0.52197 floor). The desk median residuals remained much smaller (15.42 mm surface,
  2.14% depth), so the rejection is driven by a broad heavy tail rather than missing support.
- **Conclusion**: Reject this exact registered-depth target/visibility protocol as a transferable
  prerequisite. The public backend/canonicalization API is retained as CPU-tested research
  infrastructure, but the TUM backend remains harness-local and all oriented loss defaults stay
  zero. No Phase-B optimization or utility claim is authorized. The result does not refute IGT's
  mechanism with a valid oriented-point source; it shows that plentiful local depth normals alone
  did not establish the required cross-view tail consistency here.
- **Follow-ups**: Do not rerun/tune desk, relax p90 gates, add V-depth filtering, or densify the
  target grid from this outcome. A later attempt requires new development/confirmatory sequences
  and a separately preregistered occlusion/rigidity attribution audit with signed depth
  discrepancies, construction-only visibility controls, and an ordinary extra-depth utility arm.

## 2026-07-15 — Local plane/shortest-axis targets (constructor rejected before optimization)

- **Question**: Can fixed local planes built only from corrupted metric train depth provide valid
  oriented-point supervision for IGT-style point-to-plane pulling and shortest-axis normal
  alignment in depth-backed Hybrid, and does the normal effect separate from a within-source
  shuffled-normal control?
- **Setup**: The target builder, clean audit, five Hybrid arms, coefficients, gates, and one-run
  stopping rule were frozen in `benchmarks/results/20260715_surface_plane_normal_PREREG.md` before
  outcome access. Seeds 0/1/2 used the established 40-Gaussian, twelve-view 48x48 scenes with
  held-out views `[3,7,11]`, shared 120-step train-only fits, block-corrupted metric train depth,
  and a zero-step retained Hybrid layout. Each target used the stable four nearest points from
  other train views, required at least two support views, passed PCA planarity/incidence/reachable-
  ray filters, and froze separate correct plane and shuffled alignment normals. The sole official
  command was `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
  benchmarks/surface_plane_normal_ablation.py --output
  benchmarks/results/20260715T110342Z_cpu_surface_plane_normal.json`.
- **Result**: Every structural floor passed: 318-339 targets covered **24.59%-26.02%** of retained
  nodes, with 67-98 corrupted targets, 24-31 minimum targets/source, farthest-neighbor p90
  **0.0638-0.0702** of extent, incidence p10 **0.1838-0.1926**, and shuffled-normal median
  separation **0.458-0.498**. The frozen clean audit nevertheless failed all seeds. All-target
  clean point-to-plane p90 was **0.1604/0.1745/0.1648** against the 0.10 ceiling; corrupted-target
  p90 was worse at **0.2391-0.2708**. Median clean-normal cosine also failed seed 1 overall and
  seeds 1/2 in the corrupted stratum. Every target was labelable. Per protocol, all five 90-step
  arms remained unrun and there are no loss-utility outcomes. Full audit:
  `benchmarks/results/20260715_surface_plane_normal_RESULT.md`.
- **Conclusion**: Compact, planar-looking cross-view neighborhoods were not accurate clean surface
  planes under block-corrupted depth. Reject this four-neighbor target constructor; do not tune its
  thresholds or run the withheld arms on these outcomes. This does not reject point-to-plane or
  shortest-axis losses supplied with valid oriented points. Both APIs remain opt-in with zero
  default coefficients, and no production behavior changes.
- **Follow-ups**: The next admissible plane/normal experiment needs an independently justified,
  pluggable oriented-point source on actual calibrated metric-depth/RGB-D data, audited before
  optimization. Do not repair this constructor with synthetic clean labels or another threshold
  sweep. IGT's oriented RGB-D assumption remains the scope boundary; RGB-only Gradient is not an
  honest target without an independent depth/normal backend.

## 2026-07-15 — Dense train-only patch/epipolar matcher (rejected before optimization)

- **Question**: Can a graph constructed only from train RGB patches and calibration cover enough
  retained primitives, with enough semantic precision, to fairly test whether the already-frozen
  position-consistency loss propagates its sparse local gains into whole-scene geometry?
- **Setup**: The matcher, graph/precision floors, paired 2-family x 3-arm design, and early stopping
  rule were frozen in `benchmarks/results/20260715_dense_train_position_PREREG.md`. A new
  pluggable pure-Torch `PatchEpipolarMatcher` uses raw bilinear 5x5 RGB patches, calibrated
  bidirectional epipolar distance <=2 px, reciprocal best/second ratio <=0.50, >=10-degree ray
  angle, positive closest-line triangulation, and <=1.5 px midpoint reprojection. It consumes only
  the nine physically subset train RGBs/cameras and detached fitted-center layout. The sole
  official command was `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  .venv/bin/python benchmarks/dense_train_position_ablation.py --output
  benchmarks/results/20260715T094311Z_cpu_dense_train_position.json`.
- **Result**: All structural floors passed: the three graphs had **165-187 edges**, represented
  **227-247 nodes / 17.99%-19.10%**, covered **34-35** camera-pair blocks and all train views, and
  reached **1.91x-2.47x** the sparse oracle's same-seed node coverage. Strict dominant-GT edge
  precision was only **9.04%/11.76%/10.91%**, however, versus the frozen 60% floor. Only
  17.15%-22.27% of represented nodes had a valid >=0.05-contribution, >=0.50-purity compositor
  label. Among pairs whose two endpoints were labeled, precision was 70.97%-80.00%; the problem
  was broad low-contribution/background coverage. The shuffled graph had zero strict true edges.
  Per protocol, the harness stopped before corrupted-depth construction and all 90-step arms, so
  there are no utility metrics to interpret. Full audit:
  `benchmarks/results/20260715_dense_train_position_RESULT.md`.
- **Conclusion**: Calibration and highly distinctive raw local RGB are insufficient to validate
  retained fitted-primitive identity; ratio confidence (median 0.738-0.848) was badly calibrated
  to meaningful surface contribution. Keep the matcher as a CPU-tested research reference, but do
  not use it as a correspondence source or claim that this run tests learned matching plus position
  consistency. No production default changes.
- **Follow-ups**: Honor the frozen stop: do not tune patch/matcher or position-loss thresholds on
  this outcome. Close this raw-patch position branch and test local plane pulling plus shortest-axis
  normal alignment next, initially only in depth-backed Hybrid where oriented depth points exist.
  Keep RoMa as a future optional real/calibrated backend, not a CPU/default dependency.

## 2026-07-15 — Fixed-match world-frame position consistency (locally positive, globally sub-threshold)

- **Question**: Does a robust position-only world-frame consistency loss on genuinely
  corresponding train-view primitives supply material geometry that inclusive photometric
  bounded-ray optimization lacks, and is any benefit specific to correct topology rather than a
  degree-matched graph regularizer? This is a repository-specific oracle adaptation motivated by
  MAC-Splat and EDGS, not a reproduction.
- **Setup**: The two-family x three-arm design, graph construction, gates, stopping rule, and
  one-run constraint were frozen and transparently amended before the official run in
  `benchmarks/results/20260715_world_position_consistency_PREREG.md`. Seeds 0/1/2 used the same
  40-Gaussian, twelve-view 48x48 scenes, held-out views `[3,7,11]`, shared 120-step fits, and 90
  bounded-ray steps as the LOSO experiment. Gradient and deterministic corrupted-depth Hybrid each
  compared inclusive `none`, privileged correct GT-identity edges, and an exact-degree/per-camera-
  pair cyclic derangement. The added loss was `0.25 * mean(Huber(||mu_i-mu_j||_1/extent,
  delta=0.05))`; no shape, appearance, merge, refinement, density, or loss sweep was included. The
  sole official command was `CUDA_VISIBLE_DEVICES='' .venv/bin/python
  benchmarks/world_position_consistency_ablation.py --output
  benchmarks/results/20260715T084557Z_cpu_world_position_consistency.json`.
- **Result**: The correct loss strongly engaged on represented primitives. Correct-edge p90 fell
  **91.11%** for Gradient and **86.44%** for Hybrid, and assigned-GT-center p90 fell **90.00%** and
  **82.42%**, all with 3/3 seed wins. Global gains were consistent but below every frozen threshold.
  Gradient held-out RMSE improved **0.896%** and all-source p90 **6.499%** versus required 2%/10%.
  Hybrid improved held-out RMSE **1.005%**, all-source p90 **5.865%**, and corrupted-source p90
  **5.689%** versus required 2%/10%/15%; every metric won 3/3 seeds. PSNR/coverage/IoU guardrails
  passed. The graph represented only **7.73%-9.43%** of retained primitives. Gradient's shuffled
  graph preserved 93.7% of the correct source-p90 gain, while Hybrid passed all frozen control-
  separation tests; neither rescues failed materiality. All graph, source-hash, schedule, count,
  bounded-ray, history, and cross-family invariants passed. Full audit:
  `benchmarks/results/20260715_world_position_consistency_RESULT.md`.
- **Conclusion**: Fixed correct edges can triangulate and localize their represented ray-bounded
  primitives, but this sparse oracle graph does not materially improve the whole reconstruction at
  the preregistered thresholds. Keep the inclusive production default, make no deployability claim,
  and stop coefficient/delta/norm/schedule sweeps. The result is locally positive and coverage-
  limited, not evidence that position consistency is globally sufficient.
- **Follow-ups**: Run one denser train-only matcher experiment with the same position loss and a
  pluggable, frozen mutual-confidence/reprojection/angle filter. Do not add shape/appearance yet.
  If denser coverage still fails global geometry, close this position branch and test the Scholar-
  grounded local plane/normal constraint.

## 2026-07-15 — Leave-one-source-view-out photometric supervision (negative result)

- **Question**: Do fitted splats reconstruct their own source view too easily to provide useful
  ray-depth gradients, and does removing that shortcut materially improve cross-view geometry?
  MAC-Splat grounds the underidentification diagnosis but uses direct matched 3D consistency; LOSO
  is a repository-specific diagnostic, not a reproduction.
- **Setup**: The two-family × three-arm protocol and stopping gates were frozen in
  `benchmarks/results/20260715_cross_view_supervision_PREREG.md`. On revision `2dddca4` plus the
  embedded dirty-worktree provenance, seeds 0/1/2 used 40-Gaussian synthetic scenes, twelve 48×48
  cameras, strict held-out views `[3,7,11]`, shared 150-Gaussian/view 120-step fits, and 90 lift
  steps. Pure Gradient and deterministic corrupted-metric Hybrid each compared inclusive
  supervision, target-own-source exclusion, and a globally balanced non-self dropout control. The
  latter matched each LOSO target's removed primitive count and scalar opacity while excluding
  every primitive exactly once across targets. Rotation/scale optimization, merge, refinement, and
  density control were disabled. The single official command was `CUDA_VISIBLE_DEVICES=''
  .venv/bin/python benchmarks/cross_view_supervision_ablation.py --output
  benchmarks/results/20260715T062601Z_cpu_cross_view_supervision.json`.
- **Result**: For Gradient, LOSO changed held-out depth RMSE from **0.154307 to 0.154077**
  (**0.149% better**, 2/3 wins) but all-source p90 from **0.211965 to 0.213057**
  (**0.515% worse**, 2/3 wins). PSNR changed **-0.0045 dB**. For Hybrid, LOSO changed held-out
  RMSE from **0.150330 to 0.150344** (**0.009% worse**, 1/3), all-source p90 from **0.163228 to
  0.167002** (**2.312% worse**, 0/3), and corrupted-source p90 from **0.205832 to 0.208639**
  (**1.364% worse**, 1/3); PSNR improved 0.0150 dB. Both material/attribution gates failed. All
  initialization, source-layout, balanced-exposure, opacity/count, schedule, output-count, finite,
  and provenance checks passed. The full audit is
  `benchmarks/results/20260715_cross_view_supervision_RESULT.md`.
- **Conclusion**: LOSO slightly improved the common cross-only training L1 (0.21% Gradient, 0.57%
  Hybrid) and nearest-GT median distance (1.72%/1.31%), so the intervention changed the intended
  mechanism. It did not yield material held-out or tail geometry; in Hybrid the all-source and
  corrupted tails worsened. Keep inclusive supervision as default and stop LOSO/dropout/schedule
  sweeps on this setup.
- **Follow-ups**: Pivot to a single direct robust world-frame position-consistency term between
  fixed train-view matches while retaining bounded-ray depth. Test position alone before shape or
  appearance consistency. A negative result here does not establish clean-prior or real-data harm.

## 2026-07-15 — Exact sampled-confidence attribution repair (negative result)

- **Question**: After removing both confounds in the first anchor experiment, does the spatial
  placement of already-sampled confidence weights produce a material, robust improvement over
  weighting every retained valid prior ray uniformly?
- **Setup**: The protocol, one-run rule, and stopping thresholds were frozen in
  `benchmarks/results/20260715_depth_anchor_attribution_PREREG.md` before implementation. On
  revision `2dddca4` plus the embedded dirty-worktree provenance, the CPU run used seeds 0/1/2,
  40 ground-truth Gaussians, 12 cameras at 48×48, 150 fitted Gaussians/view for 120 iterations,
  held-out views `[3,7,11]`, and 60 bounded-ray steps. Three step-0-identical arms used the same
  unjittered normalized Smooth-L1 anchor at lambda 0.01: unit weight on retained valid priors,
  sampled confidence, and an exact within-source-view permutation of those sampled valid weights.
  A separate RNG preserved the optimization/jitter stream. Refinement, merge, rotation, scale
  optimization, and density control were disabled. The official command and artifact were
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_anchor_attribution.py --output
  benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json` and
  `benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json`.
- **Result**: Confidence reduced mean held-out depth RMSE from **0.151705 to 0.149962**
  (**1.149%**, wins **3/3**), below the frozen 2% floor. It changed corrupted-source depth p90
  from **0.204933 to 0.206519** (**0.774% worse**, wins **1/3**), far from the required 15%
  reduction. PSNR was flat at **-0.0008 dB**, so the safety guard passed. Exact shuffled confidence
  reached **0.151792** RMSE and **0.205348** corrupted p90; confidence beat it on RMSE in 3/3 seeds
  and on p90 in 2/3, but the shuffle erased half the RMSE gain only. All step-0, lambda-zero RNG,
  invalid-zero, location-change, and exact sampled-weight multiset/moment invariants passed.
  Secondary signals were mixed: SSIM improved by 0.0041, all-source p90 by 8.26%, and nearest-GT
  median by 2.06%, while nearest-GT p90 worsened by 0.26%. The full audit is
  `benchmarks/results/20260715_depth_anchor_attribution_RESULT.md`.
- **Conclusion**: The repaired experiment is compatible with a small location-sensitive
  expected-depth effect, but it does not establish a material or robust confidence-anchor benefit.
  Both preregistered decision gates failed. Keep `legacy` as default and stop confidence-anchor
  loss/lambda/threshold/weighting sweeps on this setup; the auxiliary metrics cannot rescue the
  failed primary corrupted-tail criterion.
- **Follow-ups**: Pivot to the already-planned leave-one-source-view-out photometric ablation, then
  consider direct train-view geometric/correspondence consistency if photometric exclusion alone
  remains underidentified. Do not advance to train-derived confidence or tune on the held-out
  cameras from this result.

## 2026-07-15 — Confidence-weighted bounded-ray anchor (negative result)

- **Question**: Does a confidence-weighted, unjittered Smooth-L1 anchor in normalized bounded-ray
  coordinates let `HybridLifter` escape known-bad depth while preserving reliable depth seeds?
  This is a repository-specific adaptation of the confidence/local-anchor mechanisms in DP-GS and
  NoDrift3R, not a reproduction of either learned model.
- **Setup**: The protocol and thresholds were frozen in
  `benchmarks/results/20260715_depth_anchor_PREREG.md` before the official run. On revision
  `2dddca4` plus the recorded dirty-worktree source hashes, the CPU reference run used seeds 0/1/2,
  40-Gaussian synthetic scenes, 12 cameras at 48×48, training views
  `[0,1,2,4,5,6,8,9,10]`, held-out views `[3,7,11]`, 150 fitted 2D Gaussians/view for 120
  iterations, 60 bounded-ray steps, and 60 no-density refinement steps on the corrupted condition.
  Four step-0-identical arms compared legacy jittered raw-logit L2, unjittered normalized Smooth
  L1, confidence weighting, and confidence thresholding. Conditions were clean metric depth,
  deterministic 20%/−20% low-confidence block corruption, and a within-view shuffled-confidence
  negative control. Rotation, scale optimization, and merging were disabled. Exact command,
  configuration, per-seed values, source SHA-256s, and timings are in
  `benchmarks/results/20260714T224800Z_cpu_depth_anchor.json`; the command was
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_anchor_ablation.py --output
  benchmarks/results/20260714T224800Z_cpu_depth_anchor.json`.
- **Result**: On corrupted priors, mean held-out initialization PSNR for
  legacy/normalized/confidence/thresholded was **19.689/19.675/19.631/19.626 dB**. Confidence
  therefore trailed legacy by **0.058 dB**, won only **1/3** seeds, and changed low-confidence
  source-depth p90 error from **0.2066 to 0.2100** (**1.63% worse**, versus the predeclared 15%
  reduction). After refinement it reached **24.484 dB** versus legacy's **24.526 dB**
  (**−0.042 dB**). Its clean initialization regression was **0.090 dB**, inside the safety guard
  but without benefit. Against the normalized arm, calibrated confidence reduced held-out
  depth RMSE by only **0.394%** (0.15055→0.14996), while shuffled confidence worsened it by
  **0.929%** (0.15055→0.15195). Confidence improved the auxiliary corrupted initialization SSIM
  from 0.5863 to 0.6149 and median nearest-GT-center distance from 0.1291 to 0.1232, but neither
  outweighed the failed preregistered PSNR and source-depth criteria. A post-run audit found that
  the JSON's `confidence_location_causal=true` flag must not be read as causal attribution:
  normalized anchors include invalid-prior fallbacks that confidence excludes, and shuffling pixel
  confidences before bilinear sampling changed the retained-ray confidence distribution (730
  calibrated versus 351 shuffled low-confidence observations). The companion audit is
  `benchmarks/results/20260715_depth_anchor_AUDIT.md`.
- **Conclusion**: The primary hypothesis is not supported in this controlled synthetic
  configuration. That negative conclusion is independent of the flawed shuffled control. Robust
  normalized coordinates plus the tested confidence weighting did not improve the current
  photometric lift, and continuous weighting did not materially beat hard rejection. Keep `legacy`
  as the default. The other modes remain opt-in research controls; synthetic confidence cannot
  justify a production default, especially because the current Depth Anything/mock paths do not
  emit deployable confidence.
- **Follow-ups**: Before another anchor-loss sweep, test leave-one-source-view-out photometric
  supervision to strengthen cross-view identifiability. For the narrow attribution repair, compare
  valid-prior-uniform weighting with continuous confidence at the frozen lambda, and permute the
  already sampled weights among retained valid rays within each view so the exact multiset is
  preserved. Only then derive confidence using training-view consistency and evaluate actual
  monocular depth on calibrated held-out views. Do not tune on these three test cameras.

## 2026-07-14 — Three-iteration depth-covariance ablation
- **Question**: Does per-Gaussian footprint depth variance beat a train-selected global isotropic
  ray sigma, how does the current surface-Jacobian covariance compare, and are any rankings robust
  to perturbed depth plus normal merge/density refinement?
- **Setup**: CPU reference rasterizer, revision `2dddca4` plus the experimental working-tree
  changes, seeds 0/1/2, 40-Gaussian synthetic scenes with 12 cameras at 48×48, strict train views
  `[0,1,2,4,5,6,8,9,10]` and held-out views `[3,7,11]`, 150 fitted 2D Gaussians/view for 120
  iterations, SH degree 0, and fixed 0.1 opacity. `surface`, `footprint`, and one globally constant
  `isotropic` sigma shared identical means/count/opacity when merging was off. Isotropic sigma was
  selected on training views only from `{0.5,1,2}` times an RMS minor-footprint reference. The
  historical argv and complete effective configs are embedded in:
  `benchmarks/results/20260714T195446Z_cpu_depth_covariance_iter1.json`,
  `20260714T195655Z_cpu_depth_covariance_iter2_raw.json`,
  `20260714T195727Z_cpu_depth_covariance_iter2_robust.json`,
  `20260714T195902Z_cpu_depth_covariance_iter3_noise.json`,
  `20260714T195958Z_cpu_depth_covariance_iter3_production.json`, and
  `20260714T200107Z_cpu_depth_covariance_iter3_recovery.json`. The post-change canonical quick
  regression is `benchmarks/results/20260714T200932Z_cpu.json`. Because these ran against an
  evolving dirty worktree, `benchmarks/results/20260714_depth_covariance_REPLAY.md` supplies
  explicit effective replay commands and a SHA-256-bound patch against `2dddca4`. Runs forced CPU
  with
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py`; clean causal
  runs disabled merge/density, noise used masked 3×3 blur plus deterministic 2% multiplicative
  noise, and the corrected production interaction used `--merge --densify --refine-iters 100`.
- **Result**: On raw clean depth, mean held-out initialization PSNR was isotropic **20.98 dB**,
  footprint **20.72**, and surface **19.74**. Raw surface p99 covariance condition averaged
  **1.76M** and maximum scale reached 3.16× scene extent. Validity-aware finite differences raised
  surface initialization to **21.20 dB** and cut p99 condition to **721**; robust footprint reached
  21.08 and isotropic 20.98. After 60 no-density steps these converged to footprint **26.25**,
  isotropic **26.21**, surface **26.04**. With perturbed depth, initialization was surface
  **21.09**, isotropic **20.94**, footprint **20.88**, but 60-step refinement reversed the lead:
  isotropic **26.00**, surface **25.92**, footprint **25.90**. The first 80-step production run was
  invalid as a ranking because density surgery fired on its final step. With 20 recovery steps,
  isotropic won all seeds at **27.05 dB**, versus footprint **26.49** and surface **26.28**.
  The canonical seed-0 quick benchmark confirmed that the corrected default executes end to end:
  depth initialization/final PSNR rose from the preceding tracked 19.08/31.33 dB to
  **21.00/32.95 dB** (quality comparison only; PyTorch versions and machine load differed).
- **Conclusion**: The hypothesis that footprint variance beats isotropic sigma was not supported;
  clean differences after refinement were below the predeclared 0.25 dB effect. No covariance
  mode is a universal synthetic winner: surface is slightly stronger at robust initialization,
  while train-tuned isotropic is substantially better after the tested merge/density path.
  Differentiating across invalid zero-depth background was a real, replicated failure, so
  validity-aware gradients become the default. The covariance mode itself remains configurable
  and stays `surface` by default; synthetic evidence does not justify a real-scene ranking.
- **Follow-ups**: Repeat on calibrated held-out views using actual Depth Anything V2 predictions;
  learn or select global isotropic sigma without held-out leakage; test a surface-slope clamp or
  confidence-aware derivative on noisy real depth; never schedule density surgery at the final
  evaluation step.

## 2026-07-14 — gsplat density strategies, full-SH convergence, and novel-view repair
- **Question**: Were the poor novel views caused by correct state-of-the-art 3DGS refinement, or
  by missing density/appearance/geometry machinery; and do more iterations/resolution improve the
  compact 2D-Gaussian initialization?
- **Setup**: RTX 4090, PyTorch 2.12.0+cu132, gsplat 1.5.3, seed 0, the 23-train/3-test split of
  Janelle `frame_00008`, and the 3,015-splat fixed-640 StructSplat→carve initialization from the
  earlier experiment. Matched 1/16 runs used 2k iterations and a 30k cap with gsplat Default
  (AbsGS threshold 8e-4, revised opacity, reset) or MCMC (relocation/teleportation and position
  noise). The selected Default recipe then ran 7k iterations with complete masks, randomized
  backgrounds, explicit alpha loss, degree-3 SH, and antialiased rasterization at 1/8
  (666×576, cap 30k) and 1/4 (1332×1152, cap 45k). Several unrelated processes concurrently
  saturated the GPU, so elapsed times are not benchmark-valid; process peak-VRAM is retained.
- **Result**: At 1/16, Default and MCMC tied at **25.20 dB held-out foreground PSNR**; Default
  ended at 12,483 splats with 0.943 held-out alpha IoU, versus MCMC's 15,049 and 0.929. Default
  also fit training foreground better (28.27 versus 27.55 dB), so MCMC's extra relocation did
  not win this already-structured init. The 1/8 run ended at 21,202 splats and **25.67 dB
  held-out foreground / 32.52 dB crop / 0.9606 crop SSIM**, with 0.954 alpha IoU and 0.0040
  mean outside alpha; training crop reached 38.16 dB. Its held-out foreground curve peaked at
  25.79 around 4k and remained within 0.12 dB at 7k. Peak allocated VRAM was 0.27 GiB. The 1/4
  run reached 39,250 splats, 25.21/32.04/0.9580, 0.959 alpha IoU, 0.0034 outside alpha, and
  0.97 GiB peak VRAM; its held-out curve peaked at 25.51 around 3k while training crop reached
  38.24 dB. The true full-SH orbit and elevation-varying path are coherent; remaining artifacts
  are thin strands around hair, hands, and the dress hem rather than the former unconstrained
  novel-view splat cloud.
- **Bugs found/fixed**: The old 1k run never activated SH bands 1–3; gsplat always requested
  AbsGS gradients while applying the incompatible classic 2e-4 threshold; the custom controller
  was not gsplat Default or MCMC and had no relocation/teleportation; final metrics/previews
  silently recreated classic rasterization even after antialiased training; and Viser froze
  degree-0 RGB while the model used full SH. The repair uses canonical per-field optimizers,
  short-run-aware SH scheduling, gsplat strategy pre/post hooks, mask/alpha geometry loss,
  strategy-safe hard-budget surgery, persisted render configuration, view-dependent WebGL SH
  colors, and both in-plane and off-plane novel diagnostics.
- **Conclusion**: The compact initialization can now converge to a coherent object rather than
  merely fitting calibrated views. Default is the better choice on this scene; MCMC remains a
  useful initialization-robust alternative, not a universal improvement. More resolution and
  primitives sharpen appearance and silhouettes but do not automatically improve held-out
  PSNR; the 1/8 result is the best balanced reconstruction, while 1/4 is the higher-detail visual
  result. The performance numbers must be rerun on an idle GPU.
- **Follow-ups**: Repeat Default/MCMC on other frames and an SfM baseline, add LPIPS-VGG, record
  clean time-to-quality on an idle GPU, consider a train-only validation split/checkpoint policy
  rather than selecting on the three held-out test views, and target the residual hair/hem
  floaters with geometry-aware pruning or a stronger multi-view/depth initialization.

## 2026-07-14 — Compact 2D starts, strict held-out metrics, and CUDA Janelle ablation
- **Question**: Is 640 a useful configurable *start* rather than an image-wise final cap; which
  lift gives the best object-centric initialization; and how much 3D density growth is useful?
- **Setup**: RTX 4090, PyTorch 2.12.0+cu132, gsplat 1.5.3, seed 0. Janelle
  `2025_03_07_stage_with_fabric/frame_00008` has 26 calibrated RGB/mask views; every eighth view
  gives 23 train and 3 strictly held-out views. Runs used 1/16 resolution (333×288), 300 stage-1
  iterations, 1000 refinement iterations, and foreground/crop held-out metrics. StructSplat
  compared fixed 320, fixed 640, and adaptive 640→2000 against native fixed 640. Downstream
  controls used `carve(grid_res=96)` and density disabled, unrestricted, or stopped at iteration
  300 with a 15k cap. Full machine-readable result:
  `benchmarks/results/20260714T085148Z_cuda_janelle.json`. The CPU synthetic regression benchmark
  is `benchmarks/results/20260714T090516Z_cpu.json`.
- **Result**: Mean stage-1 foreground PSNR / 23-view wall time was StructSplat fixed-320
  **27.35 dB / 13.45 s**, fixed-640 **28.60 / 14.67**, adaptive-2000 **29.41 / 15.01**, versus
  native fixed-640 **25.66 / 47.29**. Fixed-640→carve initialized only 3,015 3D splats at
  **21.98 dB** held-out foreground PSNR, versus native-640's 3,613 at **20.16 dB** and
  adaptive-2000's 5,473 at **21.32 dB**. With the short 15k-capped density schedule, fixed-640
  reached **25.67 dB foreground / 32.08 dB crop / 0.9604 crop SSIM**. Fixed-320 reached
  25.50/31.89/0.9598, retaining a small deficit from its weaker init (-0.53 dB). For the same
  adaptive-2000 init, no density / unrestricted growth / capped growth reached 25.60 at 5,473 /
  25.04 at 70,485 / **25.76 dB at 15,000** splats. Depth-seeded bounded-ray `hybrid` improved
  initialization over direct monocular `depth` from **12.68 to 20.23 dB**; under the same 15k
  cap it refined to 23.41 versus depth's 23.20 dB, but `carve` remained better and faster.
  On the synthetic CPU benchmark, hybrid initialized at 21.61 dB and refined to 31.44 dB versus
  direct depth's 19.08/31.33, confirming the integration without claiming the real-depth ranking.
- **Conclusion**: 640 is a sound default start for this scene, not a ceiling. More per-image 2D
  splats improve the isolated image fit but did not improve the 3D initialization; compact
  structured fits plus carving were better. Adaptive 2k recovered 0.09 dB more final quality,
  while fixed 640 gave the strongest initialization with fewer splats. A short hard-capped 3D
  growth phase beat both no growth and unrestricted growth; the latter overfit the training views
  and expanded to 70k–100k splats. StructSplat fixed-640 was about 3.2× faster than native
  fixed-640 and improved held-out initialization by 1.82 dB. These are one-frame, low-resolution
  findings, not a cross-dataset ranking.
- **Follow-ups**: Repeat at 1/8 and 1/4 resolution, add LPIPS-VGG and peak-VRAM/time-to-quality,
  test `quadtree_wse` and GaussianImage at matched time/count, and compare the current density
  controller with gsplat MCMC/relocation. Inspect the saved contact sheet/GIF before choosing a
  high-resolution run.

## 2026-07-13 — Geometry/device correctness pass and calibrated Janelle smoke test
- **Question**: Do projection-correct covariance, bounded ray depths, independent opacity,
  color-independent carving coverage, and corrected density/timing plumbing improve the
  initialization pipeline; and does it run on the supplied calibrated object captures?
- **Setup**: `python benchmarks/run.py --quick --update-docs`, CPU, seed 0, synthetic 12-view
  48×48 scene, 150 2D gaussians/view, 120 fit + 150 refine iterations. Real-data smoke used
  `2025_03_07_stage_with_fabric/frame_00008`, four evenly sampled views at 1/64 resolution,
  real PNG masks, 60 gaussians/view, 15 fit + 5 bounded-ray + 3 refine iterations. The capture
  inventory is 26 RGB views in each stage frame and 30/32 in the two karate frames; the stage
  frames also contain per-camera masks. Result: `benchmarks/results/20260713T123616Z_cpu.json`.
- **Result**: Synthetic init/final PSNR changed versus the 2026-07-08 tracked run: depth
  17.05/28.53 → **19.08/31.36** dB; gradient 19.41/25.38 → **22.43/30.86**; carve
  17.48/29.13 → **20.31/31.91**. Gradient lift time fell 15.43 → 7.93 s. The comparison now
  includes the shared 3.90 s all-view fit cost and time/PSNR samples. On Janelle, the bounded-ray
  init reached **23.75 dB**, short refinement reached 23.83 dB, and ray-stage loss fell
  0.0124 → 0.0073. A mock relative-inverse-depth backend exercised the no-SfM bounds alignment
  and produced 46 finite splats at 18.49 dB. Coverage threshold 0.4 reduced the synthetic
  carving median center-to-GT distance from 0.276 to 0.236 (0.3 threshold vs 0.4).
- **Conclusion**: The repaired transfer now improves all three initializers on the integration
  benchmark, and both proposed depth routes execute on the real calibration format. This is a
  regression/integration comparison, not an isolated causal ablation; GPU quality and real
  held-out full-resolution quality remain unmeasured. Actual StructSplat/GaussianImage fields can
  now skip native stage 1 through the adapter rather than being conflated with 3D opacity.
- **Follow-ups**: Create a CUDA-enabled environment for the RTX 4090 and run the full 26-view
  frame at 1/4 resolution; compare StructSplat versus native versus GaussianImage at matched
  2D PSNR and primitive count; run real Depth Anything V2 Small and a depth→bounded-ray hybrid;
  report held-out PSNR/SSIM/LPIPS and time-to-quality against SfM when sparse points are available.

## 2026-07-08 — Refined `gradient` variant: depth+rot+scale along the ray, then merge
- **Question**: The staged idea "fit 2D → lift with a thin third axis → optimize each
  gaussian along its ray for position/rotation/scale → full 3DGS". Does optimizing
  rotation+scale (not just depth) and merging redundant gaussians beat the old depth-only
  `gradient` lifter, and is a literal thin "epsilon" a good along-ray init?
- **Setup**: synthetic ring scene (12 views, 48×48, 40 GT gaussians), 150 2D gaussians/
  view, `gradient` lift 100–120 iters, refine 150 iters, `rasterizer="torch"`, seed 0,
  CPU. Compared against the old depth-only behavior and across `ray_thickness`. Rev after
  this commit.
- **Result** (lift-only): depth-only `n=1800, med-dist-to-GT=0.355, init-PSNR=17.95`;
  depth+rot+scale `0.446, 18.69`; +merge(0.01) `n=1790` (barely changed). End-to-end
  (150 refine iters): OLD depth-only `init 16.98 → final 25.30, n 1800→3564`; NEW
  depth+rot+scale +merge(0.03) `init 17.95 → final 25.85, n 1634→3408`. Coarser merge
  does reduce count (voxel-frac 0.01/0.03/0.06 → n 1792/1634/1152). Thin
  `ray_thickness=0.05` raised init-PSNR (19.85) but **worsened** geometry (med-dist 0.519).
- **Conclusion**: (1) Optimizing rotation+scale + merging gives a **small but real** gain
  (init +1.0 dB, final +0.55 dB) from fewer, better-shaped gaussians. (2) **Merge barely
  fires at a fine voxel because the geometry is scattered** (median 0.35–0.45 to GT vs
  0.19–0.21 for `carve`/`depth`) — problems ④ (redundancy) and ⑤ (under-constrained depth)
  are coupled: cross-view merging only helps *after* depth converges onto surfaces, which
  single-view-sampled photometric descent does not achieve in ~100 CPU iters. (3) A literal
  thin "epsilon" trades appearance for geometry (higher init PSNR, worse 3D placement) and
  is numerically riskier — footprint-scaled thickness is the right default; the knob is
  clamped to ≥0.05× the footprint. End-to-end the `gradient` variant is still the weakest
  (25.85 vs carve 29.13, depth 28.53) precisely because of the scattered geometry.
- **Follow-ups**: (a) GPU run with many more ray-opt iterations + gsplat — does geometry
  converge enough for merge to matter? (b) Hybrid: `depth`/`carve` init → short `gradient`
  polish (start from good geometry so the ray stage refines rather than searches).
  (c) Real-scene measurement with train/test split + matched wall-clock vs SfM-init 3DGS
  (the actual step-5 comparison; synthetic numbers here are relative-only).

## 2026-07-07 — Pipeline v1 sanity on synthetic scenes
- **Question**: Do all three lifting variants beat random initialization on synthetic
  scenes, and does refinement converge from each?
- **Setup**: `python benchmarks/run.py --quick` at the initial commit (rev `eb437bb`);
  synthetic ring scene (12 views, 48×48, 40 GT gaussians), 150 2D gaussians/view,
  150 refine iters, seed 0. Result file: `benchmarks/results/20260707T115928Z_cpu.json`.
- **Result** (init PSNR → final PSNR, dB): `gradient` 18.05 → 25.31, `carve`
  17.48 → **29.13**, `depth` (GT depth backend) 17.05 → 28.53, `sfm` baseline
  19.95 → 28.67, `random` baseline **8.08** → 27.93. Lift wall-clock on CPU:
  depth 0.02 s, carve 0.11 s, gradient 8.8 s (it renders during optimization).
- **Conclusion**: The pipeline machinery works end-to-end and every variant initializes
  9-10 dB above random. Surprises worth noting: (1) `gradient` has the best init but the
  *worst* final PSNR — it keeps all per-view gaussians on their rays (1800), densification
  then balloons the count (6546) and short refinement can't clean it up; it likely needs
  cross-view merging like `carve` has. (2) `carve` refines best despite a mid init.
  (3) The `sfm` baseline init PSNR is inflated here because synthetic "SfM points" are
  sampled directly from GT gaussians. Nothing about real scenes is concluded yet (GT
  depth flatters `depth`; real monocular depth adds scale error).
- **Follow-ups**: M2 GPU validation; add merge step to `gradient` (or hybrid B→A:
  depth init + gradient polish); revisit densification budgets for dense inits (M3).
