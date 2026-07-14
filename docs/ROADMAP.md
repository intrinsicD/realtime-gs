# Roadmap

## M0 — Infrastructure (done)
- [x] Agent workflow: CLAUDE.md/AGENTS.md, skills (verify, bench, docs-sync, experiment)
- [x] Verification: ruff + pytest (CPU) + docs_sync, mirrored in CI
- [x] Benchmark harness with tracked JSON results and auto-updated docs table

## M1 — Pipeline v1 on CPU (done)
- [x] Core containers (2D/3D gaussians, cameras, SH, metrics) with PLY/NPZ IO
- [x] Differentiable 2D accumulated splatting + per-image fitting (gradient-magnitude init)
- [x] Reference 3D rasterizer (EWA projection, depth-sorted alpha compositing)
- [x] Lifting variants: `gradient`, `depth`, `carve` (+ `sfm`/`random` baselines)
- [x] 3DGS refinement loop with adaptive density control
- [x] Synthetic ground-truthed scenes; COLMAP text/binary parsing
- [x] Calibrated object-capture JSON, distortion/masks, held-out split, external 2D adapters
- [x] End-to-end tests and variant-comparison benchmark

## M2 — GPU validation
- [x] gsplat backend parity test green on RTX 4090; auto backend respects explicit CPU devices
- [x] Depth Anything V2 Small smoke test and bounds alignment on a calibrated Janelle capture
- [x] Optional StructSplat CUDA stage-1 backend with configurable progressive density growth
- [ ] Wire gsplat densification strategies as an alternative to `rtgs.optim.density`
- [ ] Benchmark on MipNeRF-360 `garden`/`bicycle` @ 7k iters: init-PSNR and
      time-to-quality vs SfM init (protocol in docs/RESEARCH.md §6)
- [ ] Fit-time target: stage 1+2 < 30 s for 200 images @ 1080p on one consumer GPU

## M3 — Research questions (log answers in docs/EXPERIMENTS.md)
- [x] First calibrated-capture comparison: compact `carve` wins Janelle frame 00008 at 1/16;
      repeat at higher resolution and on more scenes before treating this as general
- [ ] Does per-gaussian along-ray variance (footprint depth spread) beat isotropic σ_z?
- [ ] Carve: does moment-matched merging beat keep-all + prune-in-refinement?
- [x] Implement depth-seeded bounded-ray hybrid B→A; evaluate uncertainty and shorter schedules
- [x] Initial density ablation: a short 15k-capped schedule beats no-density and unrestricted
      growth on Janelle; repeat across scenes and compare gsplat MCMC/teleportation
- [x] Progressive/error-driven stage-1 allocation via StructSplat residual/tensor growth;
      compare `quadtree_wse` and GaussianImage at matched wall-clock/count
- [ ] Add LPIPS-VGG and novel-view geometry diagnostics to held-out evaluation
- [ ] Run full 26-view Janelle frame at 1/8 and 1/4 resolution with time/VRAM curves
- [ ] Feed-forward multi-view init (VGGT/MASt3R pointmaps) as a fourth variant

## M4 — Real-time ambitions
- [ ] CUDA kernel for stage-1 fitting (batched per-image, GaussianImage reports ~2k it/s)
- [ ] Streaming/incremental mode: add images to an existing scene without full re-fit
- [ ] Investigate skipping stage 3 entirely for preview-quality output
