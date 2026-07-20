# Independent scientist audit: multiscale fixed-topology refinement

## Verdict

**PASS for the frozen negative disposition.** The artifact is strict, finite, source-bound, and
internally consistent. I independently recomputed every serialized per-view metric from its named
raw evidence, all checkpoint and seed means, foreground-PSNR AUCs, paired deltas, exposure counts,
seed wins, and decision booleans without calling the harness's summary or decision helpers. The
result is unchanged: none of the three candidate/control schedules has a quality-improvement
result, neither camera schedule has an exposure-efficiency result, and blocked ordering is not
material under the frozen gate.

No blocking, major, or minor audit finding was identified.

Audited result:
`benchmarks/results/20260716T003735Z_cpu_multiscale_refinement.json`, SHA-256
`343263f3193871dbdae4f390d46ba9c305cb9c38bfead0dd5c7bc97448ce35fa`.

This verdict confirms only a negative CPU synthetic, fixed-topology, degree-zero result for the
exact 24-to-48, 60/60, 120-update protocol. It does not establish that multiscale optimization is
generally ineffective. It does not evaluate another scale or schedule, parameter-specific
geometry/appearance routing, adaptive density, full SH, real scenes, CUDA/gsplat, wall-clock
speed, FLOPs, memory, energy, or a production-default change.

## Chronology and provenance

- The append-only preregistration predates implementation and the official attempt. Its SHA-256 is
  `b4c17da489a6e66950ec15ecda78dd7ddb063d65055e25bb4f76df6e8cca0a59`.
- The independent pre-run implementation review has exact verdict `PASS` and predates sealing. Its
  SHA-256 is
  `d8836baef23abe46b556acc6d3ee2aa17590a3092d2f4c1226b43743a8a27955`.
- Seal SHA-256 is
  `a667d612bc6276dc5aeaee3c7fb406caddfa98dda8d94a5d23494bea9649936b`.
  Its timestamp is `2026-07-16T00:37:30+00:00`.
- The exclusive once-only marker was created at `2026-07-16T00:37:41+00:00`. Its file SHA-256 is
  `bac82ee3d3b421f28134dd100967f1734ecf264223f7b6a8302afa6805bff384`,
  and its canonical payload SHA-256 is
  `eb17036382277f4b35b46317179637990993c2365f3e355c97aa75b9f6c54537`.
  It binds the exact result path, command, seal, sealed source aggregate, and CPU environment.
- The result was written at `2026-07-16T00:39:42+00:00`, after the seal and marker. The adjacent
  result note has SHA-256
  `e545b0fe34b66ceabb13cd97e6767b5513a3ae0e75e5a5607f02fb06a05df490`;
  it binds the correct result hash and labels its values unaudited rather than making a speed or
  capability claim.
- All 75 path hashes in the seal match the audited files and reproduce sealed source aggregate
  `6d15d682e9de815eaa8b322a669b569fb713d0df3394b7f137bb8ac6569e8957`.
  All 38 repository sources recorded as loaded by the result are exact members of that manifest
  and reproduce loaded-source aggregate
  `7c5b57478479b247d7430c1dac4958e507e17491c3d6fe363c1fef8581ddb586`.
  No optional gsplat or StructSplat source appears in the loaded set.
- Seal and result agree on revision `2dddca4aff59702341af9faceefa76ad2505dd83` and tracked-diff
  SHA-256 `cedf9decbecf0a6caa9339b035d5ae986997bcad9e5e57af6d902b12e6c8563f`.
  The run was dirty but fully bound by the sealed manifest and diff; this is not an unrecorded
  provenance gap.
- The seal and marker/result environments agree apart from expected load-average observations:
  Python 3.12.9, Torch 2.9.0+cu128, NumPy 2.1.3, Pillow 11.3.0, Linux x86-64, 16 logical CPUs,
  Torch intra-op threads 4, deterministic algorithms enabled, `CUDA_VISIBLE_DEVICES=''`, and
  `OMP_NUM_THREADS=MKL_NUM_THREADS=4`. The CUDA-capable wheel does not imply CUDA execution.
- The seal's full Ruff, format, non-slow CPU pytest, docs-sync, and diff checks all returned zero;
  their stored stdout/stderr hashes and identical pre/post source snapshots reproduce. Seal
  verification SHA-256 recorded by the result is
  `082f3c49b2a2dd77facaa24643666e4eabeb305db505d20233e4ba1a3e64619b`.

The principal sealed implementation hashes also reproduce: harness
`7a70d315e7e8e5b1c0934e18f75b6c85452313ac39c549642afdbe0f190aa580`, focused test
`4a9beb35de2133709813a0263370cfb5c9af300f026aa7e671b573491016b1ea`, Trainer
`3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f`, and optimizer export
`1196f76c9386d808b88a0940f562b29a85b3598e182ba7e997ebf2f769e4d53a`.

## Independent reconstruction

Strict parsing rejected duplicate keys and non-standard numeric constants, and a recursive scan
found every serialized numeric value finite. I independently reproduced the canonical hashes for
all preparation, fit, truth, arm, checkpoint, per-view, exposure, and aggregate records.

The frozen seeds are `3,4,5`; training indices are `[0,1,2,4,5,6,8,9,10]`; held-out indices are
`[3,7,11]`. For every seed, fitting and Carve lifting operate on a physically subset training
scene whose depth and synthetic-GT Gaussian fields are removed before refinement. Held-out truth
is constructed only after initialization and arm schedules are frozen, stays in the read-only
callback closure, and is never passed to Trainer, the optimizer, loss construction, stopping, or
selection.

The isolated Torch generator schedule was rebuilt from `manual_seed(seed)` and exactly 120
`randint(0,9)` draws for each seed. Every arm reproduces the expected schedule hash. Execution
orders, step controls, 60/60 scale counts, last-step convention, and per-view half/full assignment
counts all match the preregistration. `full` uses the native `step_controls=None` path rather than
an alternate explicit schedule.

Every arm within a seed starts from exact detached clones. Initialization hashes match across all
arms; the shared step-zero checkpoint is byte-identical; primitive counts remain exactly 1070,
1140, and 1122 for seeds 3, 4, and 5 respectively; no topology operation occurs; and active SH
degree is zero at every checkpoint. Each arm has one persistent 120-update Trainer call and fresh
optimizers, rather than repeated short restarts.

For all 180 serialized per-view checkpoint records (three seeds, four arms, five checkpoints,
three held-out views), I recomputed foreground/full/crop MSE and PSNR, normalized depth RMSE,
alpha IoU, and foreground coverage directly from the named float64 sums, counts, crop bounds, and
extent. The reported crop SSIM equals its separately named raw SSIM evidence exactly. I then
recomputed all three-view arithmetic means, using no pixel pooling across views, and the five-point
step-normalized trapezoidal AUC at steps `(0,30,60,90,120)`.

The resulting foreground-PSNR AUC values are:

| Seed | Full | Camera blocked | Pyramid blocked | Camera interleaved |
|---:|---:|---:|---:|---:|
| 3 | 20.37898015511902 | 19.99569367560258 | 20.242359735770734 | 19.97038760657243 |
| 4 | 20.822568713007644 | 20.45144210566553 | 20.74099852185757 | 20.494136465390845 |
| 5 | 20.542503837052195 | 20.280982681332684 | 20.49442139748926 | 20.241748118804534 |

## Recomputed frozen decisions

All values below are paired against `full` and were rebuilt from per-view evidence rather than
accepted from the result summary.

| Arm | Mean AUC delta (dB) | AUC seed wins | Mean final foreground-PSNR delta (dB) | Quality noninferior | Quality improvement | Exposure efficiency |
|---|---:|---:|---:|:---:|:---:|:---:|
| `camera_blocked` | -0.33864474752602075 | 0 | -0.2632469132829118 | false | false | false |
| `pyramid_blocked` | -0.08875768335376577 | 0 | -0.20326181686616232 | false | false | false |
| `camera_interleaved` | -0.3459268381370168 | 0 | -0.7349981081173761 | false | false | false |

The signed per-seed AUC deltas are respectively:

- `camera_blocked`: `[-0.383286479516439, -0.3711266073421129,
  -0.2615211557195103]`;
- `pyramid_blocked`: `[-0.1366204193482865, -0.08157019115007458,
  -0.04808243956293623]`;
- `camera_interleaved`: `[-0.40859254854659, -0.32843224761679934,
  -0.300755718247661]`.

The result's complete PSNR, crop-SSIM, depth, alpha-IoU, and coverage guardrail values and every
criterion boolean reproduce. All three arms fail common quality noninferiority, and independently
also fail the preregistered AUC/final-PSNR improvement requirements. Thus unfavorable SSIM or depth
guardrails are not what creates the negative quality-improvement disposition.

`camera_blocked-camera_interleaved` AUC deltas are
`[0.02530606903015098, -0.042694359725313547, 0.03923456252815072]`, with mean
`0.00728209061099605 dB` and two blocked-arm wins. Although all three final-PSNR differences favor
blocked ordering, the mean AUC misses `+0.05 dB` and both camera arms fail noninferiority versus
`full`; therefore `blocked_order_attribution` is correctly `false`. No coarse-to-fine ordering
claim is permitted.

Exposure accounting also reproduces exactly:

- `full`: 276480 optimization render pixels and 276480 loss pixels;
- `camera_blocked`: 172800 render and 172800 loss pixels;
- `pyramid_blocked`: 276480 render and 172800 loss pixels;
- `camera_interleaved`: 172800 render and 172800 loss pixels.

Both camera schedules therefore have the exact descriptive render-exposure ratio `0.625`, or
37.5% fewer optimization raster pixels. Both exposure-efficiency booleans remain false because
quality noninferiority is false. Common native evaluation contributes 82944 pixels per arm,
held-out callbacks contribute 27648 per arm, and the shared manual step-zero evaluation contributes
6912 per seed; all are correctly excluded from the optimization-exposure ratio.

The frozen mechanism classification is consequently `no_quality_improvement`: neither blocked arm
passes. This is a gate-based negative classification, not proof of a unique failure mechanism.

## Timing and isolation

Trainer records native elapsed time before invoking the read-only checkpoint callback. Callback
seconds are separate, and pyramid construction is also outside the native timer. The result marks
wall/process, preparation, arm, native-evaluation, and callback timing descriptive only;
`timing_used_in_any_gate` independently reproduces as `false`. The recorded 121.22813604099792
wall seconds is not isolated performance evidence and must not be called a speed result.

## Evidence boundary

The JSON serializes decision-grade float64 sums/counts, SSIM evidence, mask hashes, renderer-field
hashes, truth hashes, and canonical record hashes, but not the full checkpoint Gaussian tensors or
rendered pixel arrays. Therefore this replay-free review can independently derive all decisions
from the serialized raw evidence and verify its sealed in-process binding, but cannot derive those
raw sums or SSIM anew from pixels without rerunning the consumed official computation. No official
seed, fit, arm, or outcome was rerun.

This is not a decision-critical qualification: every arm fails independently reconstructable
foreground-PSNR AUC and final-PSNR requirements, so no possible favorable reinterpretation of the
SSIM evidence can turn any quality-improvement, exposure-efficiency, blocked-order, or mechanism
gate positive. The limitation narrows renderer-pixel identity to the sealed run; it does not weaken
the audited negative disposition.

## Claim disposition

| Claim | Disposition | Evidence |
|---|---|---|
| Result is bound to the frozen protocol, PASS review, seal, marker, source, split, config, and CPU environment | Confirm | Hashes, chronology, manifests, command, environment, and verification records reproduce |
| All three schedules improve held-out quality over full-resolution refinement | Reject | Every independently recomputed mean AUC delta and final foreground-PSNR delta is negative |
| Either camera schedule is exposure-efficient under the frozen definition | Reject | Exact 0.625 exposure is confirmed, but both schedules fail required quality noninferiority |
| Blocked coarse-to-fine ordering is material | Reject | Mean blocked-minus-interleaved AUC is only +0.00728209061099605 dB and both camera arms are noninferiority failures |
| The exact fixed-topology 24-to-48 branch remains open on this synthetic setup | Retire/close | Neither blocked arm quality-improves and neither camera arm is exposure-efficient, triggering the frozen stopping rule |
| Multiscale refinement generally cannot help Gaussian splatting | Unverified and out of scope | No alternative schedule, adaptive density, full SH, real scene, or parameter-specific routing was tested |
| The 37.5% raster-pixel reduction is a speedup | Reject | Pixel exposure is exact, but timing is descriptive and no FLOP/hardware/runtime gate exists |

The allowed concise conclusion is: under the exact CPU synthetic fixed-topology degree-zero
protocol, the tested blocked and interleaved multiscale schedules regressed the preregistered
quality criteria relative to full-resolution refinement. The exact 24-to-48 branch closes on this
setup. Future work must come from a separately preregistered question, not outcome-tuning this
scale, boundary, filter, loss, seed set, or gate.

## Checks actually run

- Independent strict JSON, duplicate-key, finite-value, chronology, file/source hash, manifest,
  environment, git, verification-record, nested-record-hash, split, schedule, RNG, initialization,
  topology, SH, held-out-isolation, raw per-view metric, mean, AUC, delta, guardrail, exposure,
  order, mechanism, and timing-gate reconstruction.
- The sealed harness's read-only `validate_result_payload` path: pass.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q
  tests/test_multiscale_refinement.py`: 19 passed.
- Focused Ruff check and format check for the harness, test, Trainer, and optimizer export: pass.
- `git diff --check`: pass at audit time.

The official scientific fits and arms were not rerun. No CUDA/GPU benchmark was run, no result or
source file was modified, and no default or documentation claim was changed by this audit.
