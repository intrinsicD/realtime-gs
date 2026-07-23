# Masked native-anchor Beam partition covariance on Janelle — result

Date: 2026-07-23

Status: **mechanism valid; primary coverage hypothesis failed; narrow fitted-view convergence
benefit reproduced**

Protocol:
[`20260723_beam_partition_covariance_PREREG.md`](20260723_beam_partition_covariance_PREREG.md)

Machine-readable primary result:
[`runs/beam_partition_covariance_20260723/summary.json`](../../runs/beam_partition_covariance_20260723/summary.json)

Independent audit:
[`20260723_beam_partition_covariance_AUDIT.md`](20260723_beam_partition_covariance_AUDIT.md)

## What was implemented

Beam Fusion now preserves the exact implied camera-space depth beside every CSR contributor.
The new opt-in module `rtgs.lift.beam_partition` does the following for each selected reference:

1. obtains fixed native 2D anchors only from Beam Fusion's surviving
   `(view, source_component)` identities—there is no 3D-to-2D projection or rematching step;
2. samples all 5,000 original 2D Gaussians with frozen order-5 Gauss-Hermite quadrature;
3. discards every sample outside the exact packed foreground mask;
4. assigns retained density to the nearest native anchor (hard anchored Voronoi partition);
5. computes a density-weighted covariance about the original anchor mean;
6. lifts that covariance through the original source ray, original implied depth, and original
   Beam depth range, then repeats equal-weight CI for covariance only.

Two treatments isolate the effect:

- `pou-area`: original local 2D shape, determinant matched to the partition moment;
- `pou-full`: complete partition covariance, including orientation and anisotropy.

The unchanged `ci` result is the control. All arms initialize exactly **800 3D Gaussians**.
Their 3D means, opacity, SH/color, count, contributor identities, and contributor depths are
fixed; only quaternion/log-scale differ.

## Setup

- Janelle `frame_00008` compact Gaussian bundle.
- Global views `[0,3,6,9,12,15,18,21]`.
- All eight views are fitted; evaluation local views `[0,2,4,6]` are not held out.
- Seed 0; downscale 32 exact compact-field teachers and packed masks.
- Beam: minimum three views, 3-sigma pair/fold-in gates, color distance 0.35, color sigma 0.25,
  extent/100 NMS voxel, opacity 0.10, 800-output cap.
- Partition: 25 quadrature samples/source Gaussian, assignment chunks of 8,192,
  `1e-12` minimum partition mass, `1e-6 px²` numerical variance floor.
- Refinement: 1,000 Torch CPU steps, fixed topology, identical loss/view stream/schedules;
  no clone, split, prune, merge, teleport, or opacity reset.

Primary command:

```bash
.venv/bin/python benchmarks/beam_partition_covariance.py \
  --protocol benchmarks/results/20260723_beam_partition_covariance_PREREG.md \
  --out runs/beam_partition_covariance_20260723
```

The same command was rerun into `runs/beam_partition_covariance_20260723_repeat`.

## Mechanism checks

Beam produced 6,029 contributor links for 800 outputs. Deduplication within each view left 4,704
unique native anchors; 1,325 links reuse an anchor already partitioned for another 3D track.
Density is partitioned once per unique anchor and is not duplicated.

| View | Unique anchors | Mask-retained density | Median determinant-matching covariance multiplier | Maximum multiplier |
|---|---:|---:|---:|---:|
| C0001 | 592 | 93.83% | 0.4186× | 284.5× |
| C0006 | 594 | 95.84% | 0.3384× | 125.5× |
| C0012 | 630 | 95.96% | 0.3333× | 1,001.3× |
| C0019 | 591 | 93.53% | 0.4124× | 260.5× |
| C0022 | 598 | 95.81% | 0.3743× | 328.8× |
| C0028 | 596 | 93.06% | 0.5656× | 246.0× |
| C0031 | 536 | 94.55% | 0.3862× | 21,290.4× |
| C0039 | 567 | 94.71% | 0.8210× | 176.2× |

No partition was empty. Per-view partition mass error was at most `2.70e-16`, and the native
covariance/depth round trip reproduced stored CI covariance with maximum relative error
`1.23e-6` (frozen gate `1e-4`). All treatment covariances were finite/SPD and every frozen field
assertion passed.

The multiplier distribution is important: this is **not** a uniform upscaling heuristic. The
median scalar covariance multiplier needed to match 2D determinant is below one in every view,
while a small number of partitions are extremely large. Beam's existing 3D sigma cap bounds the
lifted outputs.

## Covariance diagnostics

Median residuals are evaluated across all 6,029 contributor links. “Own target” means native
covariance for CI, determinant-matched covariance for `pou-area`, and full partition covariance
for `pou-full`.

| Arm | Whitened residual vs own target | Whitened residual vs native target | Median min sigma | Median max sigma | Median condition |
|---|---:|---:|---:|---:|---:|
| `ci` | 0.6888 | 0.6888 | 0.00210 | 0.00980 | 15.04 |
| `pou-area` | 0.7839 | 0.6555 | 0.00229 | 0.00747 | 9.99 |
| `pou-full` | **0.5523** | 1.0778 | 0.00524 | 0.01024 | **3.89** |

`pou-full` is more consistent with the newly defined partition targets and substantially more
isotropic in 3D, but it is less consistent with the original local 2D Gaussian covariances. This
does not establish physical ground-truth 3D covariance.

## Rendering and convergence

Foreground-PSNR AUC is the trapezoidal mean over initialization and the 25-step checkpoints.

| Arm | Init FG PSNR | Init alpha IoU | Init alpha inside | Init alpha outside | Mean PSNR AUC | Final FG PSNR | Final alpha IoU | First reaches CI final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ci` | 12.3303 | 0.01073 | 0.13975 | 0.00103 | 21.8568 | 25.0083 | 0.93649 | 1000 |
| `pou-area` | 12.3128 | 0.00625 | 0.12274 | 0.00119 | 22.9184 (**+4.86%**) | 25.0982 | 0.93611 | 975 |
| `pou-full` | **12.5745** | 0.00886 | **0.15200** | 0.00212 | **23.3476 (+6.82%)** | **25.1263** | **0.93725** | **950** |

Both treatments fail the preregistered coverage gate:

- `pou-area` changes alpha-inside by −12.17% and alpha IoU by −41.80%;
- `pou-full` changes alpha-inside by only +8.77% (required +25%) and alpha IoU by −17.46%
  (required +10%).

Both pass the separate optimization gate. `pou-area` improves PSNR AUC 4.86%;
`pou-full` improves it 6.82%, ends +0.1180 dB over CI, and reaches the CI endpoint 50 steps
earlier. Relative to `pou-area`, the full shape adds 1.87% AUC and passes both subordinate
shape clauses. The overall preregistered “full partition adds value” decision still fails because
that decision required the primary coverage gate too.

The single-view Torch previews agree with the numbers: `pou-area` is the least visible initial
render, while `pou-full` is faint but more spatially coherent than CI. All three final renders are
qualitatively close. Each preview panel places the teacher on the left and the render on the
right. The orbit viewer remains the right way to inspect geometry rather than over-interpreting
that one camera.

## Conclusion

The requested construction is implementable and works mechanically: Beam survivors identify
native 2D anchors, the complete masked source mixture can be partitioned around them, and the
partition moments can be lifted without changing 3D positions or correspondences.

It does **not** solve the stated visible-coverage problem under the frozen 800-Gaussian,
opacity-0.10 screen. The full partition covariance nevertheless gives a reproducible early
fixed-topology optimization advantage. The evidence therefore supports a narrower interpretation:
the full covariance improves optimization conditioning/shape on these fitted views, not initial
mask coverage.

Keep CI and production defaults unchanged. Do not replace the method with blind heuristic
upscaling: the measured partitions mostly shrink and contain extreme outliers. The next justified
test is a multi-scene, multi-seed, held-out comparison of CI versus `pou-full`, followed—only if
that generalizes—by the production gsplat split/merge path with a separately frozen bound or blend
for the extreme partition tails.

## Reproducibility, audit, and visual comparison

- Exact repeat: all timing-free scientific JSON fields and every initial/final PLY were
  byte-identical.
- Independent audit: **70/70 checks passed**.
- Primary artifacts: `runs/beam_partition_covariance_20260723`.
- Repeat artifacts: `runs/beam_partition_covariance_20260723_repeat`.
- Static initial/final comparison:
  [`comparison_contact_sheet.png`](../../runs/beam_partition_covariance_20260723/comparison_contact_sheet.png).
- Viewer manifest:
  [`20260723_beam_partition_covariance_VIEWER.json`](20260723_beam_partition_covariance_VIEWER.json).
- Viewer receipt:
  [`20260723_beam_partition_covariance_VIEWER_RECEIPT.json`](20260723_beam_partition_covariance_VIEWER_RECEIPT.json).

To compare all three initial/final pairs with one synchronized orbit:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260723_beam_partition_covariance_VIEWER.json \
  --max-viewer-gaussians 800 --device cpu --port 8783 --no-open
```

The smoke test loaded all six 800-Gaussian models, returned HTTP 200, and used the CPU viewer
process. An unrelated Python process held 1,264 MiB on the GPU; it was not the viewer and no
timing claim is made. The viewer was stopped and port 8783 was closed after the test.
