# Preregistration: geometric Stage-3 Gaussian arena on Janelle `frame_00008`

Date frozen: 2026-07-24 (Europe/Berlin), before any calibrated arena outcome was run.

## Question and scope

Does a live-shaped, geometrically growing Stage-3 parameter/Adam arena reduce the cost of
`gsplat-default` density surgery without changing its logical optimization result?

This is a single-scene, single-seed systems-development experiment. It may authorize retaining the
arena as an opt-in research path or a later multi-scene performance study. It cannot change the
default, establish general speed or memory claims, or establish MCMC/relocation behavior.

## Frozen input and split

- Raw scene:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`
- Calibration: `dataset/2025_03_07_stage_with_fabric/calibration_dome.json`
- Loader: downscale 16, at most eight calibrated views, `test_every=8`, masks and undistortion on.
- Ordered training cameras: `C0001,C0008,C0014,C0021,C0026,C0031,C0039`.
- Reporting-only held-out camera: `C1004`.
- Initialization:
  `runs/pool_structure_wse_frame00008_20260724/models/pool-structure-wse/gaussians_init.npz`,
  SHA-256 `f0e41c4c57289f08c8b7101898c1f06192e0b2085b10bb877d8a315e97971abb`,
  422 Gaussians.

The held-out camera is never used for fitting, density decisions, checkpoints, or model selection.

## Frozen arms and execution order

Each arm runs in a fresh Python process. Immediately before training, the worker resets
`torch.manual_seed(0)` and `torch.cuda.manual_seed_all(0)`.

1. `dynamic-a`: established exact-sized gsplat parameter replacement.
2. `geometric`: geometric arena, growth factor 2.0, initial capacity equal to the smallest power of
   two not below the initialization, hard maximum equal to `density.max_gaussians`.
3. `dynamic-b`: independent repeat of the established exact-sized path.

The arena keeps capacity-shaped parameter and Adam storage, exposes live-shaped leaf/moment prefix
views to autograd and Adam, and commits DefaultStrategy clone/split/prune ordering in one
transaction. It does not send inactive rows to the renderer. Dynamic A/B use the unchanged gsplat
operations. Scheduled density-event profiling synchronizes immediately before and after the whole
controller call in all arms.

## Frozen training protocol

- 10,000 fresh Stage-3 iterations; this is not a continuation.
- Same `TrainConfig` as the audited pool+structure+WSE 10k experiment except:
  `gaussian_storage_policy` selects the arm and `profile_density_events=True`.
- `gsplat-default`, density start 100, stop 1,000, every 100, absolute gradient threshold
  `0.0008`, revised opacity, hard live budget 20,000.
- SH degree 3, interval 250; masked random-background training; antialiasing on; `packed=False`.
- Training seed 0 and final-checkpoint policy.
- Evaluation every 100 steps. Complete detached states are saved at 2k, 4k, 6k, 8k, and 10k.
- Exact train and held-out metrics and calibrated renders are generated from saved states.

## Measurements

Primary systems measurements:

1. Sum of synchronized `event_seconds` across scheduled density events.
2. Native 10k elapsed time excluding checkpoint-callback serialization.
3. Approximate non-density time: native elapsed minus density-event sum.
4. Peak CUDA allocated and reserved bytes.
5. CUDA device allocation/free counts, allocation retries, OOM count, and inactive-split peak when
   reported by PyTorch.
6. Arena capacity/migration trajectory and copied bytes.

Correctness and quality measurements:

- sampled training-view sequence;
- density-event iteration and `n_before/n_after` trajectories;
- final Gaussian count;
- exact saved tensor hashes and maximum absolute field differences;
- train and held-out foreground PSNR, crop SSIM, alpha IoU, alpha inside, and alpha outside at every
  saved checkpoint.

## Frozen validity and decision rules

The comparison is invalid if an arm fails, inputs/source change during execution, saved artifacts
do not replay, sampled view sequences differ, density-event iteration sets differ, or dynamic A/B
disagree in final count. Dynamic A/B timing remains descriptive if their held-out foreground PSNR
differs by more than 0.02 dB.

Arena storage correctness passes only if:

- its complete event `n_before/n_after` trajectory and final count exactly match both dynamic
  controls;
- final train and held-out foreground PSNR lie inside the dynamic A/B envelope expanded by
  0.02 dB; and
- final held-out alpha IoU lies inside the dynamic A/B envelope expanded by 0.002.

The arena has a material density-event latency win only if its summed event time is at most 80% of
the faster dynamic control. It has an end-to-end win only if its native 10k elapsed time is at most
98% of the faster dynamic control. Its non-density path and each peak-memory measure are
non-inferior when no more than 2% and 10% above the worse dynamic control, respectively.

- Correctness pass + density-event win: retain the opt-in arena and authorize a repeated,
  multi-scene scaling study.
- Correctness pass without density-event win: log the negative systems result; do not optimize
  thresholds or change the default.
- Correctness failure: reject the current arena transaction implementation regardless of timing.
- No single outcome here changes `dynamic` as the default.

## Required artifacts

The run must save a source/input-bound plan and summary, per-arm histories and five checkpoint
models, exact metrics, calibrated train/held-out visuals, timing/memory tables, a viewer manifest,
`index.html`, HTTP smoke receipts for every local page target, an `rtgs view` smoke receipt, a
result note, and an independent scientist audit.
