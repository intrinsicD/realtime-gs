# ADR-002 carrier refinement on full-resolution masked Janelle — frozen protocol

**Frozen before any run of this harness:** 2026-07-27
**Evidence class:** outcome-exposed, single-scene, single-seed development/mechanism evidence
**Not authorized:** README claims, confirmatory language, default changes, pooling with held-out scenes

## Question

Does fixed-track covariance/opacity/SH0 repair followed by a carrier-preserving maturation schedule
extract more value from Beam Fusion's means than immediate standard 3DGS optimization, and which
phase accounts for any difference?

The central development hypothesis is:

> Beam Fusion means are useful, but poor covariance, opacity, and appearance initialization causes
> standard split/prune to replace them before they mature.

This experiment implements ADR-002 phases 0–7 and every runnable baseline/ablation named in
`docs/PAPER_PLAN_beam_fusion.md`. It does not promote the paper's main claim.

## Inputs and leakage boundary

- Compact source:
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.
- Compact manifest SHA-256:
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`.
- Raw source:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
- The harness must verify every selected raw RGB and mask against the source SHA-256 stored inside
  its strict `.rtgsv` container before decoding it.
- Resolution is native `5328 × 4608`; `downscale=1`.
- The outcome-independent Stage split rule already used by the masked development screen is reused
  verbatim:
  - train3: `C0001`, `C0014`, `C0028`;
  - validation: `C0031`, `C1000`, `C1002`;
  - report-only sealed: `C1001`, `C1004`.
- Only train3 compact captures enter Beam Fusion/repair. Only train3 RGB/masks enter dense
  optimization. Validation is evaluation-only. Report-only cameras must not be loaded or scored.
- Frame 00008 is already outcome-exposed. All results remain development-only regardless of size.

## Unavailable baseline

There is no COLMAP sparse point model and no `colmap` executable for this capture. Therefore:

- `Original 3DGS` is **unavailable**;
- `Original 3DGS with compressed RGB` is **unavailable** because its matched SfM initialization is
  unavailable;
- the harness runs matched `random-rgb` and `random-jpeg-q50` controls and must not call either one
  original 3DGS.

This absence is a limitation, not a license to substitute or relabel another initializer.

## Frozen initialization and repair

- Seed: `27027`.
- Beam Fusion: train3, `min_views=3`, `max_components=2400`, `source_chunk=256`,
  `seed_budget_multiplier=4`, NMS voxel `extent/100`, initial opacity `0.10`, existing production
  gates otherwise as encoded by the bound harness.
- Covariance repair:
  - means and all non-covariance fields frozen;
  - 120 float64 Adam steps, LR `0.03`;
  - robust observation-whitened reprojection loss, Huber delta `0.25`;
  - CI prior weight `1e-3`;
  - Cholesky SPD parameterization;
  - sigma bounds `[1e-4, 0.5*extent]`, aspect ratio at most `100`.
- Opacity repair:
  - mean/covariance/appearance frozen;
  - 80 float64 Adam steps, LR `0.05`;
  - optical-density-space Huber loss, delta `0.20`, prior `1e-3`;
  - opacity clamp `[0.005, 0.995]`.
- The fitted 2D component amplitude is explicitly an experimental opacity proxy, not identifiable
  physical per-carrier alpha. This caveat must appear in the receipt.
- Appearance repair: five robust amplitude-weighted IRLS steps, RGB Huber delta `0.10`; SH0 only,
  all higher bands disabled.

## Frozen optimization budget

All optimized arms receive exactly 160 dense-image updates. `beam-only` receives zero.

- Full schedule: warm-up 30, clone 40, higher-SH 30, standard 60.
- No-clone carrier schedule: warm-up 30, higher-SH 30, standard 100.
- No-warm-up: clone 40, higher-SH 30, standard 90.
- Clone-only and particle: warm-up 30, clone/particle 100, higher-SH 30, no standard handover.
- Immediate arms: 160 standard updates.
- CUDA gsplat, packed mode, masks on, black background, standard L1 + D-SSIM loss, final checkpoint,
  seed `27027`.
- Classic density events use the encoded 40-step cadence, opacity reset disabled, and a hard cap of
  9,600 primitives.
- Clone phase: no split, protected original prefix, covariance-frame jitter `0.10`, max 2.5× the
  initial carriers.
- Particle phase: parent retained, covariance-frame jitter at least `0.25`, child opacity `0.20×`,
  unsuccessful-child prune threshold `0.02`.
- Standard handover restores ordinary clone/split/prune and full SH.

## Arms

1. `beam-only`: Phase 0 state, no RGB optimization.
2. `beam-standard`: unrepaired carriers handed immediately to standard 3DGS.
3. `carrier-schedule`: repair + warm-up + higher SH + standard; no clone phase.
4. `carrier-schedule-clone`: complete ADR-002 schedule.
5. `no-covariance`: complete schedule with covariance repair off.
6. `no-opacity`: complete schedule with opacity repair off.
7. `no-warmup`: complete schedule with fixed-topology warm-up off.
8. `split-immediately`: repaired carriers handed immediately to standard split/prune.
9. `clone-only`: repaired carriers, protected clone maturation, no standard handover.
10. `particle`: optional carrier-guided low-opacity particle generation.
11. `random-rgb`: matched random means/simple remaining fields, original RGB.
12. `random-jpeg-q50`: bit-identical random initialization, second-generation JPEG quality 50
    train images.
13. `means-only`: Beam means with one global train color, isotropic median Beam scale, opacity
    `0.10`, then standard optimization.

The same deterministic Beam/repair state is shared wherever an arm definition permits. The two
random controls start bit-identically.

## Metrics

Primary endpoint:

- median per-view foreground-weighted masked PSNR over the three validation cameras, at native
  resolution.

Secondary quality/mechanism diagnostics:

- masked foreground-crop PSNR and SSIM (bounding box plus 5% full-image margin);
- AlexNet LPIPS on that masked crop, resized bilinearly so its longest side is at most 512 pixels;
- alpha IoU, inside alpha, outside alpha;
- initial and final primitive count;
- initial-carrier survival, roots with descendants, generations, original/all-row mean
  displacement, covariance log-determinant drift, opacity drift;
- fixed-track covariance and optical-density residuals before/after repair;
- cloned descendants and surviving original carriers.

Resource diagnostics:

- Beam/repair/training/end-to-end wall time;
- peak CUDA allocated and reserved memory;
- exact compact/raw/JPEG input byte counts and final PLY bytes;
- measured compact load and raw read+decode+undistort effective MiB/s.

LPIPS dependencies are pinned/bound as:

- `lpips==0.1.4`;
- LPIPS v0.1 Alex calibration weights SHA-256
  `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`;
- torchvision AlexNet weights SHA-256
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.

## Frozen interpretation gates

These are development dispositions, not confirmatory tests:

1. **Repair mechanism gate:** median fixed-track covariance and opacity optical-density residuals
   must each decrease by at least 25%. Failure invalidates the corresponding implementation claim.
2. **Carrier value gate:** `carrier-schedule-clone - beam-standard` primary PSNR must be at least
   `+0.25 dB`, with at least 50% of original carriers surviving through the clone phase. Passing
   supports further held-out evaluation; failing rejects this schedule on this scene.
3. **Clone contribution gate:** `carrier-schedule-clone - carrier-schedule` of at least `+0.25 dB`
   is a useful clone effect; differences inside ±0.25 dB are practically tied.
4. **Means gate:** `means-only - random-rgb` of at least `+0.25 dB` supports the narrow claim that
   Beam means carry downstream value under the matched simple-parameter control.
5. **Input-compression diagnostic:** compare only `random-jpeg-q50` to `random-rgb`; this is not the
   unavailable original-SfM baseline.
6. SSIM, LPIPS, alpha, time, memory, storage, and all other ablations are descriptive and must be
   reported even when they disagree with PSNR.

No result can change a default or close a roadmap question without a new held-out protocol.

## Bound implementation

Base commit before the working-tree implementation:
`dd84c28deb3378d57992cd10b20f08bb594f102a`.

SHA-256:

- `benchmarks/carrier_refinement_fullres.py`:
  `074462cb325a88de8d1e8ed7f08fa1ac41ff2e7f41be283cdbb8f8e5d4b30579`
- `src/rtgs/lift/carrier_refinement.py`:
  `bbd50a727da29663e22415376c8cabf269bf561e929a76ae86a4360dbb5590f5`
- `src/rtgs/optim/carrier_schedule.py`:
  `62de83cdd953a33fb1299ad023b2fb3c57b4f04aa872b44f017e39dae9280f69`
- `src/rtgs/optim/density.py`:
  `5c64ea245de69a63317c3c4564017ea82066184963af2984c7d89928af601ef8`
- `src/rtgs/optim/trainer.py`:
  `49c6493b1d9fe5ac45abdda1f3a16802dfbce112cc9f50d96f0590504804ce23`
- `tests/test_carrier_refinement.py`:
  `7cdc7376631d64adbb05c3e20cc716d8786884fbbd1e4788e5d56f0b4ccab7fb`
- `docs/ADR-002-carrier-refinement.md`:
  `61f51c88912a1ff2aaf450bb8cca322fb339df4fa2b95dbf9041c9fa028849ec`
- `docs/PAPER_PLAN_beam_fusion.md`:
  `069a34dc3db9a49ac6a8638572f28b6a85beaf5a4a6687e1a4a0474cca89f8e0`

Pre-freeze checks:

- `tests/test_carrier_refinement.py tests/test_calibrated.py`: 7 passed.
- `tests/test_pipeline.py tests/test_optim.py tests/test_init_preserving_density.py`: passed, with
  six expected CUDA skips and one pre-existing tensor-conversion warning.
- Ruff on all changed Python implementation/harness files: passed.

Any post-freeze change to a bound file requires a new protocol. Result presentation code may only
be changed after the run if the patch is isolated, disclosed, and cannot change stored metrics.

## Official command and required artifacts

```bash
PYTHONUNBUFFERED=1 .venv-cuda/bin/python benchmarks/carrier_refinement_fullres.py \
  --protocol benchmarks/results/20260727_carrier_refinement_fullres_PREREG.md \
  --out runs/carrier_refinement_fullres_frame00008_20260727
```

Required:

- root `summary.json`, `run_manifest.json`, `source_binding.json`,
  `repair_diagnostics.json`, `comparison_manifest.json`, `index.html`;
- per-arm initial/final PLY, metrics, training history, config, record, and previews;
- root representative calibrated bundle copied from `carrier-schedule-clone`;
- orbit viewer receipt with command, PID, URL, manifest hash, and HTTP smoke result;
- a results note and an independent scientist-pass audit.
