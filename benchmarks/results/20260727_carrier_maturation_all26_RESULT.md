# All-26-view carrier maturation on full-resolution masked Janelle

Date: 2026-07-27
Base revision recorded by the run: `dd84c28deb3378d57992cd10b20f08bb594f102a`
Evidence class: outcome-exposed, single-scene, single-seed development and mechanism evidence

Disposition: **accept the completed artifact bundle as a valid fitted-view development
reconstruction, but reject the claim that every maturation phase converged. Change no default and
promote no paper, generalization, causal-stage, or performance claim.**

The canonical result bundle is
[`runs/carrier_maturation_all26_frame00008_20260727`](../../runs/carrier_maturation_all26_frame00008_20260727/)
and its browser overview is
[`index.html`](../../runs/carrier_maturation_all26_frame00008_20260727/index.html).
The frozen governing protocol is
[`20260727_carrier_maturation_all26_PREREG_V2.md`](20260727_carrier_maturation_all26_PREREG_V2.md).
The scientist pass is
[`20260727_carrier_maturation_all26_AUDIT.md`](20260727_carrier_maturation_all26_AUDIT.md).

## Recovery chronology

The first V1 launch stopped before Beam Fusion or any model-quality outcome because a compact-target
helper import failed. The first V2 process then reached the fixed-topology step-5,000 checkpoint
before PyCharm crashed. Its partial bundle is preserved unchanged at
`runs/carrier_maturation_all26_frame00008_20260727_crash_partial_step005000`.

The canonical bundle is a fresh restart, not an optimizer-state resume: the driver intentionally
refuses a non-empty output directory and does not serialize enough Adam/RNG state for exact
continuation. The V2 protocol hash, eight implementation hashes, seed, inputs, and command were
unchanged after the crash. Because partial repair and training outcomes had already been exposed,
this remains development evidence rather than an independent confirmatory repeat.

## Frozen setup

- Dataset: masked Janelle
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
- Native undistorted resolution: **5328 x 4608**; native masked RGB drives every optimization
  phase.
- Cameras: all 26 calibrated views are used for fitting, stopping, and reporting; held-out set is
  empty.
- Seed: `27027`.
- Hardware/software: NVIDIA GeForce RTX 4090, PyTorch `2.12.0+cu132`, gsplat `1.5.3`, CUDA runtime
  `13.2`.
- Starting topology: 5,000 Beam Fusion carriers.
- Final topology: 100,000 Gaussians after 128,000 optimizer updates.
- Wall time: 10,383.45 seconds for the canonical process.

Exact command:

```bash
.venv-cuda/bin/python benchmarks/carrier_maturation_all26.py \
  --out runs/carrier_maturation_all26_frame00008_20260727 \
  --protocol benchmarks/results/20260727_carrier_maturation_all26_PREREG_V2.md \
  --raw-frame /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008
```

## Final fitted-view endpoint

Values are arithmetic means over the same 26 fitted cameras.

| Target/evaluator | FG PSNR (dB) | Crop SSIM | LPIPS down | Alpha IoU |
| --- | ---: | ---: | ---: | ---: |
| Native masked RGB | **28.7641** | **0.973831** | **0.017323** | **0.980415** |
| Compact teacher | **31.7410** | **0.984937** | not recorded | **0.980415** |

The final native reconstruction is visually coherent across the inspected fitted-camera contact
sheet, with the largest residuals concentrated around the silhouette and fine extremities. The
saved free orbit is also coherent as an inspection aid, but it has no reference image and is not a
novel-view quality measurement.

## Phase stopping results

The frozen process required each fixed-topology maturation phase to reach its train-PSNR plateau
before handover, subject to a safety cap. Four phases reached the cap instead.

| Phase | Updates | Stop reason | Plateau result |
| --- | ---: | --- | --- |
| Fixed topology | 30,000 / 30,000 | `train_psnr_plateau` | converged; best step 24,000 |
| Clone recovery 1 | 15,000 / 15,000 | `max_iterations` | **not converged**; best step 15,000 |
| Clone recovery 2 | 15,000 / 15,000 | `max_iterations` | **not converged**; best step 12,000 |
| Clone recovery 3 | 15,000 / 15,000 | `max_iterations` | **not converged**; best step 14,000 |
| Higher SH | 15,000 / 15,000 | `max_iterations` | **not converged**; best step 15,000 |
| Standard growth | 30,000 / 30,000 | `max_iterations` | fixed budget by protocol |
| Standard settle | 8,000 / 40,000 | `train_psnr_plateau` | converged; best step 2,000 |

Raw stage-directory identifiers ending in `_converged` are preserved for reproducibility. The
reader-facing page and comparison manifest label the four failed boundaries as cap endpoints; the
phase `stop_reason` and `plateau.converged` fields are authoritative.

## Boundary measurements and mechanism limits

Native fitted-view FG PSNR moves from 11.546 dB at Beam initialization to 25.550 dB after the
30,000-update fixed-topology phase, 27.823 dB after the capped higher-SH phase, 28.724 dB after
standard growth, and 28.764 dB after the final settle.

Each tangent clone wave causes an immediate native-PSNR drop
(-0.125, -0.425, and -0.399 dB). Relative to the respective pre-clone boundary, the model is
0.539, 0.470, and 0.528 dB higher after the following 15,000-update recovery. These are sequential
measurements, not clone effects: there is no matched no-clone continuation, and each recovery
contains additional optimization.

Covariance repair changes native fitted-view PSNR by -0.443 dB and opacity repair then changes it
by +0.612 dB. This run contains no matched repair ablations, so neither change establishes
downstream causal value.

The standard-growth phase reaches the 100,000-Gaussian hard cap at its first 5,000-update
checkpoint. At the final endpoint, 2,975 of 5,000 original carriers survive directly (**59.50%**)
and **88.78%** of roots retain at least one descendant.

## Historical context, not a control

The bound prior all-view compact-target run selected 37.887 dB compact FG PSNR at step 69,000. The
current compact-teacher score is 31.741 dB, **6.146 dB lower**. This is useful context but not a
causal comparison: the optimization targets and schedules differ, and neither run is a randomized
matched control for the other.

## Scope and conclusion

The run proves that the instrumented all-26-view carrier-to-3DGS path can execute end to end and
emit a complete, auditable native-resolution fitted-view reconstruction. It does not prove the
intended convergence-between-stages behavior: three clone recoveries and higher-SH fail their
frozen plateau rule.

All cameras participate in training and selection, so there is no held-out or generalization
evidence. Every optimized phase consumes native masked RGB, and Original-3DGS and compressed-SfM
baselines are absent, so compact-only sufficiency and the paper's main claim remain untested.
Timing is descriptive only: this is one process with no idle-GPU protocol or repeats, and another
GPU process was observed. No default, paper, compact-only, novel-view, causal-stage, storage,
speed, bandwidth, or VRAM claim is authorized.
