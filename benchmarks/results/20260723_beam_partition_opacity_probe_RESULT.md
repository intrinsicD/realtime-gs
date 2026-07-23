# Post-hoc Beam partition optical-thickness probe — result

Date: 2026-07-23

Status: **strong fitted-view mechanism evidence; no opacity rule or default selected**

Machine-readable primary result:
[`runs/beam_partition_covariance_20260723/opacity_probe.json`](../../runs/beam_partition_covariance_20260723/opacity_probe.json)

Exact repeat:
[`runs/beam_partition_covariance_20260723_repeat/opacity_probe.json`](../../runs/beam_partition_covariance_20260723_repeat/opacity_probe.json)

Independent artifact audit:
[`20260723_beam_partition_opacity_probe_AUDIT.json`](20260723_beam_partition_opacity_probe_AUDIT.json)

## Question

The masked native-anchor partition experiment left `pou-full` with only 0.00886 initial alpha
IoU at the evaluator's hard `alpha > 0.5` prediction threshold. Does that result mean the
projected Gaussian footprints are missing most foreground pixels, or are they present but too
optically thin at Beam's fixed opacity 0.10?

This is explicitly a **post-hoc diagnostic**, not a preregistered treatment comparison. It may
localize a mechanism, but it cannot choose a multiplier, establish held-out utility, or justify a
default change.

## Setup

The probe loads the byte-identical saved `ci`, `pou-area`, and `pou-full` initial PLYs from the
primary and repeat partition runs. Every arm contains exactly 800 Gaussians. It performs no
optimization and freezes means, covariance, SH/color, and count. Only a uniform opacity
multiplier is applied at render time:

```text
opacity factor: 0.5, 1, 2, 4, 8
effective opacity: 0.05, 0.10, 0.20, 0.40, 0.80
alpha support thresholds: 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

The exact Torch CPU renderer and the same fitted evaluation views `[0,2,4,6]` were used. The
base revision is `c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df`; the probe source is bound to
SHA-256
`315a9d8cb4ada8a2d24f6ce066cccbd0a222b3fed93546a665cffa3153635018`.

Commands:

```bash
.venv/bin/python benchmarks/beam_partition_opacity_probe.py \
  --run runs/beam_partition_covariance_20260723
.venv/bin/python benchmarks/beam_partition_opacity_probe.py \
  --run runs/beam_partition_covariance_20260723_repeat
.venv/bin/python benchmarks/audit_beam_partition_opacity_probe.py
```

## Result

At the original opacity 0.10, `pou-full` already has broad low-alpha projected support:

| Alpha threshold | Foreground recall | Foreground precision |
|---:|---:|---:|
| 0.01 | 0.9437 | 0.8913 |
| 0.02 | 0.9099 | 0.9199 |
| 0.50 | 0.0089 | 1.0000 |

Changing no field except uniform opacity produces:

| Arm | Opacity | FG PSNR | Alpha IoU at 0.5 | Alpha inside | Alpha outside |
|---|---:|---:|---:|---:|---:|
| `ci` | 0.10 | 12.3303 | 0.01073 | 0.13975 | 0.00103 |
| `ci` | 0.80 | 14.4829 | 0.63090 | 0.59429 | 0.00715 |
| `pou-full` | 0.10 | 12.5745 | 0.00886 | 0.15200 | 0.00212 |
| `pou-full` | 0.80 | **16.4720** | **0.72233** | **0.67582** | 0.01501 |

The primary and repeat probes used byte-identical PLYs and reproduced every scientific arm field
exactly. The independent audit passed **15/15** checks.

## Interpretation

The near-zero standard alpha IoU does not mean that `pou-full` lacks projected footprint support
over the whole object. Most mask pixels receive a small alpha contribution, but the fixed
opacity-0.10 initialization rarely accumulates enough optical thickness to cross 0.5. The
opacity-only sensitivity therefore identifies optical thickness as the first bottleneck to test.

This does **not** establish that opacity is the only missing component. Low-threshold support
still leaves residual holes and leakage, and the probe used fitted views. A global opacity of 0.80
is not a selected initializer rule: it was chosen after seeing the original result, and outside
alpha rises with it.

## Next decisive experiment

Use a fresh preregistered blockwise bottleneck ladder with train-only fitting and untouched
held-out cameras:

1. freeze everything and fit per-Gaussian optical thickness only;
2. add covariance scale/orientation;
3. add 3D means;
4. only then enable split, merge, prune, and mask-invalid teleportation;
5. fit SH/color separately so appearance error is not mistaken for coverage error.

For each block, report the fraction of the initializer-to-full-fit alpha-IoU gap closed, the
outside-alpha guardrail, foreground PSNR, and held-out generalization. This attributes the missing
capability to optical mass, covariance, placement, topology/capacity, or appearance rather than
inferring it from a single final score.
