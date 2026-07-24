# New opt-in variants on Janelle `frame_00008` — frozen development protocol

Date frozen: 2026-07-24 (Europe/Berlin)

Repository revision: `7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11` plus the explicitly
hashed experiment harness below.

This protocol was written after CPU-synthetic mechanism tests and CUDA renderer parity tests, and
before fitting or inspecting any Janelle outcome from these arms. It is a prospective,
single-scene, single-seed development comparison. It cannot authorize a default, generalization,
quality, or performance claim.

## Question and arms

Four opt-in seams landed after the 2026-07-23 Beam experiment. Do any improve stage-1 fit quality,
mask containment, or downstream reconstruction on a calibrated real capture?

The isolated stage-1 arms are:

- `baseline`: unchanged native gradient-magnitude initialization, fixed 640 rows;
- `pool`: baseline plus the fixed-capacity pool/free list, with 1,280 allocated rows and periodic
  32-row park/32-row residual-spawn recycling;
- `mask-containment`: baseline plus `mask_coverage_weight=5.0`;
- `structure-tensor`: feature-aware structure-tensor density, anisotropic weighted sample
  elimination, and oriented covariance initialization in place of gradient-magnitude sampling.

The value `5.0` is the only mask-containment strength exercised by its pre-run mechanism test. It
is frozen as a development treatment, not selected as a default.

Every stage-1 arm is lifted through the same carve configuration and receives the same refinement
schedule. The baseline refinement enables `checkpoint_policy=best_train_psnr`. Its captured final
iterate is reported as `baseline`; the checkpoint returned from the identical trajectory is
reported as `best-train-checkpoint`. Thus the checkpoint comparison changes model selection only,
not initialization, optimizer updates, view sampling, density surgery, or training budget.

Combinations among pool, containment, and structure initialization are outside this protocol.
Pool and containment are currently incompatible by validation, and a full factorial would answer
a different interaction question.

## Frozen implementation and pre-run gate

- Harness: `benchmarks/new_variants_frame00008.py`, SHA-256
  `dc2789f952e5972b0e7f32ae6799fe2670f52b838933cc0afd78d9740b6b351a`.
- Native fitter: `src/rtgs/image2gs/fit.py`, SHA-256
  `0533342c11b79209f099f14cdc0e50bfc740fd392e37c23bb01fb3799276fd3c`.
- Pool: `src/rtgs/image2gs/pool.py`, SHA-256
  `cad33b6def38e1c43011ae61b07292cb370ab876dc20773cb122f5fc96835999`.
- Structure initializer: `src/rtgs/image2gs/structure_init.py`, SHA-256
  `e5e8eb7906762f7563fb8bf7f03f12541557cbf0375761388cf8c651eab5c18c`.
- Stage-3 trainer: `src/rtgs/optim/trainer.py`, SHA-256
  `4dcfcf8d584fcce956c086844e6174b6993a1ff9305f91bf8128736c0a7fc36b`.
- Common carve lifter: `src/rtgs/lift/carve.py`, SHA-256
  `35135d6c93de3a836c9f9843fdfc63e3c08d973a080a9743372d3ee057a829eb`.
- Pre-run focused command:

  ```bash
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    .venv-cuda/bin/python -m pytest -q \
    tests/test_image2gs_pool.py \
    tests/test_image2gs_mask_contain.py \
    tests/test_image2gs_init_strategy.py \
    tests/test_structure_init.py \
    tests/test_checkpoint_policy.py \
    tests/test_renderer2d_cuda.py
  ```

  All 50 collected tests passed.
- Harness lint and syntax gates:

  ```bash
  .venv/bin/ruff check benchmarks/new_variants_frame00008.py
  .venv/bin/ruff format --check benchmarks/new_variants_frame00008.py
  .venv/bin/python -m py_compile benchmarks/new_variants_frame00008.py
  ```

  All passed.

Any change to a bound source or this protocol after outcome access requires a new protocol and a
fresh output directory. The run plan must record the hashes of every selected RGB/mask input,
loaded image/mask tensor, relevant source file, environment, and effective dataclass.

## Dataset, split, and isolation

- Object/capture: Janelle `2025_03_07_stage_with_fabric/frame_00008`.
- Raw RGB/masks:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
- Calibration:
  `dataset/2025_03_07_stage_with_fabric/calibration_dome.json`, SHA-256
  `51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`.
- The raw and checked-in calibration files were verified byte-identical before freezing.
- Loader: calibrated undistortion, downscale 16, `max_images=8`, `test_every=8`.
- Frozen view order:
  `C0001, C0008, C0014, C0021, C0026, C0031, C0039, C1004`.
- Training cameras:
  `C0001, C0008, C0014, C0021, C0026, C0031, C0039`.
- Reporting-only held-out camera: `C1004`.
- Resolution: 333×288 pixels.
- Seed 0, with the established per-view stage-1 offset (`seed + local_view_index`).

`C1004` must not enter stage-1 fitting, carve placement, optimizer updates, checkpoint selection,
hyperparameter choice, or stopping. It is rendered only for initialization/final reporting and
visual comparison. The other seven cameras are training views, not validation views.

## Stage 1

Common settings:

- native backend and experimental native CUDA renderer;
- 640 initial/live Gaussians;
- 300 Adam steps, LR 0.01 with the existing cosine schedule;
- current `weight_color_9p` appearance parameterization;
- crop-to-mask fitting and the existing weighted masked MSE;
- no early stopping.

Treatment settings:

- pool capacity 1,280; triage every 50 steps before the endpoint; park the 32 lowest-weight live
  rows and spawn at most 32 residual-peak rows; minimum one live row;
- containment weight 5.0, with no pool or batched views;
- structure-tensor defaults exactly as implemented; no containment or pool.

Save every fit and history. Independently render the restored full-image fit and report equal-view
means over the seven training cameras for:

- foreground and crop PSNR;
- foreground SSIM;
- PSNR against the full masked canvas;
- color-independent coverage IoU at threshold 0.10;
- mean coverage inside/outside the foreground;
- foreground support recall at coverage threshold 0.10;
- realized component count.

The representative stage-1 visual is frozen to `C0014`.

## Common lift and refinement

Carve settings:

- grid 32³ within half the calibrated scene extent;
- minimum two views; hull fraction 0.85;
- color standard-deviation sigma 0.20 and component color-match sigma 0.35;
- mask/coverage threshold 0.40;
- 48 samples per ray; minimum score/weight 0.05;
- voxel merge, opacity 0.10, SH degree 0.

Refinement settings:

- 2,000 CUDA steps with the modern `gsplat` backend, packed and antialiased;
- gsplat DefaultStrategy density control, absgrad threshold `8e-4`;
- density rounds at steps 100 through 1,000 every 100, under a 20,000-Gaussian cap;
- opacity reset interval 1,000, leaving 1,000 recovery steps after the last possible reset/surgery;
- masked standard training loss, random backgrounds, D-SSIM weight 0.2,
  mask-alpha weight 0.05, outside-alpha weight 0.01;
- target SH degree 3, one additional band every 250 steps;
- evaluation every 100 steps, seed 0.

Report initialization/final metrics separately over the seven training cameras and the untouched
`C1004` camera: full, foreground, and crop PSNR; SSIM; alpha IoU at 0.5; alpha inside; alpha
outside; and primitive counts.

Elapsed time, peak allocation, and placement time are saved only as execution diagnostics. The GPU
is not isolated, arm order is not randomized, and no speed, memory, or throughput comparison is
authorized.

## Frozen interpretation gates

All effect sizes are computed against `baseline`.

Pool mechanism validity requires capacity 1,280, exactly 640 live/output rows in every view, and
finite saved fits. A pool quality benefit requires:

1. mean stage-1 foreground PSNR at least +0.10 dB;
2. foreground-PSNR wins on at least five of seven cameras;
3. mean coverage outside no more than 10% higher.

Containment is useful at stage 1 only if:

1. mean coverage outside falls by at least 20%;
2. mean foreground PSNR is no more than 0.25 dB lower;
3. mean coverage inside is no more than 5% lower.

Structure initialization is useful at stage 1 only if:

1. mean foreground PSNR improves by at least 0.10 dB;
2. it wins on at least five of seven cameras;
3. mean coverage outside is no more than 10% higher.

Any stage-1 treatment is downstream-promising only if its final held-out foreground PSNR improves
by at least 0.10 dB, final held-out alpha IoU is no more than 0.01 lower, and final training
foreground PSNR is no more than 0.25 dB lower. Initialization-only gains do not override a failed
final gate.

The train-only checkpoint policy is promising only if it selects an earlier recorded step and,
against the byte-identical final-trajectory endpoint, improves held-out foreground PSNR by at
least 0.10 dB with alpha IoU no more than 0.01 lower. Selection at step 2,000 or a smaller gain is
neutral, not a success.

Passing a gate motivates a multi-seed/multi-scene replication only. Failure is a negative result.
No outcome from this protocol can change a production default.

## Artifacts, visuals, audit, and command

Primary output:
`runs/new_variants_frame00008_20260724`.

Required artifacts:

- pre-outcome `plan.json`;
- every stage-1 fit/history and aggregate metrics;
- exact initial/final NPZ plus viewer-ready PLY for all five report arms;
- complete training histories;
- per-arm calibrated comparison, calibrated-path animation, novel orbit, and novel elevation;
- cross-arm `C0014` stage-1 contact sheet;
- cross-arm `C0014` training and `C1004` held-out reconstruction contact sheets;
- comparison-viewer manifest and an HTTP smoke-test receipt;
- machine-readable summary, result note, and independent audit.

Official command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/new_variants_frame00008.py \
  --protocol benchmarks/results/20260724_new_variants_frame00008_PREREG.md \
  --out runs/new_variants_frame00008_20260724
```

Required viewer smoke:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_new_variants_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8784 --no-open
```

The viewer is qualitative and cannot reverse a metric gate. The audit must verify chronology,
source/input hashes, split isolation, effective configs, fit/model counts and finiteness,
checkpoint pairing/selection views, artifact hashes, metric arithmetic, decision gates, and
claim scope.
