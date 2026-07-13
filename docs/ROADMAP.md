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

## M2 — GPU validation (needs a CUDA box)
- [ ] gsplat backend parity test green on GPU; wire gsplat densification strategies as an
      alternative to `rtgs.optim.density`
- [ ] Depth Anything V2 backend smoke test + scale alignment on a real COLMAP scene
- [ ] Benchmark on MipNeRF-360 `garden`/`bicycle` @ 7k iters: init-PSNR and
      time-to-quality vs SfM init (protocol in docs/RESEARCH.md §6)
- [ ] Fit-time target: stage 1+2 < 30 s for 200 images @ 1080p on one consumer GPU

## M3 — Research questions (log answers in docs/EXPERIMENTS.md)
- [ ] Which lifting variant wins on init-PSNR? On time-to-30dB after refinement?
- [ ] Does per-gaussian along-ray variance (footprint depth spread) beat isotropic σ_z?
- [ ] Carve: does moment-matched merging beat keep-all + prune-in-refinement?
- [ ] Gradient variant: joint depth+opacity optimization vs depth-only; how few iterations
      suffice when seeded by the `depth` variant (hybrid B→A)?
- [ ] How much densification is still needed with dense 2D-gaussian init (can we disable
      cloning entirely and only prune)?
- [ ] Progressive/error-driven gaussian allocation in stage 1 (Image-GS-style) vs fixed N
- [ ] Feed-forward multi-view init (VGGT/MASt3R pointmaps) as a fourth variant

## M4 — Real-time ambitions
- [ ] CUDA kernel for stage-1 fitting (batched per-image, GaussianImage reports ~2k it/s)
- [ ] Streaming/incremental mode: add images to an existing scene without full re-fit
- [ ] Investigate skipping stage 3 entirely for preview-quality output
