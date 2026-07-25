# Cover-consistent surfel initialization for Beam Fusion — preregistration

Date: 2026-07-24 (frozen before any outcome on this root was read)

Root: `dataset/2025_03_07_stage_with_fabric/frame_00009` — a **fresh** development root. The
all-view `frame_00008` screen is consumed by the 20260721/20260723 sessions and is used here
only for the mechanism diagnostic reported below, never for arm selection.

## Question

Beam Fusion returns good positions but a covariance that answers the wrong question. Its
covariance-intersection rule estimates *where a component is*, and the renderer needs *how much
surface a primitive covers*. Does replacing the localization covariance and the fixed opacity
with a cover-consistent surface-element rule (a) produce a usable initial image, (b) improve
refinement, and (c) make the initial Gaussians participate in density control?

## Mechanism diagnostic that motivated the treatment (already measured, both frames)

Measured on `frame_00009` and replicated on `frame_00008`, 800 components, 8 fitted views:

| quantity | frame_00009 | frame_00008 |
|---|---:|---:|
| mean \|cos\| between the fused short axis and a kNN surface normal | 0.531 | — |
| mean \|cos\| between the fused long axis and the mean view direction | 0.540 | — |
| random-orientation baseline | 0.500 | 0.500 |
| fused sigma_max / *smallest* contributor footprint (median) | 1.660 | 1.750 |
| fused sigma_max / *median* contributor footprint (median) | 0.443 | 0.551 |
| Spearman(sigma_max, smallest footprint) | 0.564 | 0.685 |
| Spearman(sigma_max, median footprint) | 0.399 | 0.500 |
| fused sigma_max / kNN-3 spacing (median) | 0.182 | 0.256 |
| 2D inputs per 3D output | 50.0 | 50.0 |

Reading: the orientation carries no surface information (both alignments sit at the random
baseline, because the contributing cameras span a median 161-degree arc so no direction is
systematically under-triangulated); a precision mean is dominated by the sharpest matched
observation rather than the typical one; and after 50x decimation the widest axis is roughly a
fifth of the distance to the component's own neighbours, so the primitives cannot tile. At
opacity 0.10, 6.58 overlapping primitives are needed to reach alpha 0.5.

## Treatment

`rtgs.lift.surfel_init.reconcile_covariances`, holding means, SH/color, and count bit-identical:

- **Local frame** from a chunked kNN PCA over the component centers (normal, tangents,
  spacing, planarity).
- **Tangential sigma** `max(0.5 * spacing, resolution_floor)`. The `0.5` is derived, not
  tuned: the peak-to-trough ripple of a hexagonal Gaussian cover
  (`rtgs.lift.surfel_init.hexagonal_cover_ripple`, direct summation) is 81.5% at 0.30, 13.3%
  at 0.40, **1.25% at 0.50**, 0.31% at 0.55. `resolution_floor` is the per-component median
  contributor footprint from Beam's own CSR lineage.
- **Normal sigma** = measured local out-of-plane spread, floored by the localization sigma
  resolved along that same normal, capped at `flatness = 0.5` times the tangential sigma.
- **Opacity** `1 - (1 - 0.9)^(1/S)` with `S = 2 pi sigma_t^2 / hexagonal cell area`, clamped to
  `[0.02, 0.95]`.

## Frozen arms

All arms share bit-identical means, SH/color, and count (800). Only quaternions, log-scales,
and opacity differ.

| arm | covariance | opacity | isolates |
|---|---|---|---|
| `ci` | unchanged Beam CI | fixed 0.10 | control |
| `ci-op` | unchanged Beam CI | coverage rule on CI's own effective sigma | optical thickness alone |
| `cover-iso` | isotropic at the cover sigma | fixed 0.10 | extent alone |
| `cover-iso-op` | isotropic at the cover sigma | coverage rule | extent + thickness, no orientation claim |
| `surfel` | oriented flat surfel | coverage rule | full treatment |

## Frozen protocol

- Beam Fusion: seed 0, 800 outputs, minimum 3 views, 3-sigma seed/fold-in gates, color
  distance 0.35 / sigma 0.25, extent/100 NMS voxel, opacity 0.10, source chunk 256, seed budget
  multiplier 4 — identical to the 20260721/20260723 configuration.
- Train views (global indices) `[0, 3, 6, 9, 12, 15, 18, 21]` = `C0001, C0006, C0012, C0019,
  C0022, C0028, C0031, C0039`. These are the only views used by Beam Fusion and by refinement.
- **Held-out views** (global indices) `[1, 13, 25]` = `C0004, C0025, C1004`. They enter neither
  Beam Fusion nor training and are read for reporting only. `C1004` is the extrapolative
  camera; `C0004` and `C0025` are interpolative. Both pooled and `C1004`-only numbers are
  reported.
- Teachers: exact compact-field queries at downscale-32 pixel centers with packed-alpha masks.
- Refinement: Torch CPU reference rasterizer, 1,000 steps, seed 0, identical loss
  (masked L1 + 0.2 D-SSIM + 0.05 mask-alpha + 0.01 outside-alpha), identical schedules.
- Modes: `fixed` (no topology change) and `adc` (classic controller, start 20, stop 500, every
  4, gradient threshold 3e-3, prune 0.005/0.1, opacity reset every 100 to 0.011, cap 8,000) —
  the exact configuration calibrated in the 20260723 convergence-dynamics screen, unchanged and
  identical for every arm.
- Checkpoints every 25 steps. No checkpoint selection: the final iterate is reported.

## Preregistered gates

Evaluated on the **held-out** pool unless stated. `ci` is the reference.

- **G1 coverage** — `surfel` initial alpha-IoU >= 0.25 absolute **and** initial alpha-outside
  <= 0.05.
- **G2 initial quality** — `surfel` initial foreground PSNR >= `ci` + 1.0 dB.
- **G3 optimization** — in `fixed` mode, `surfel` foreground-PSNR AUC >= `ci` + 3% relative
  **and** final foreground PSNR >= `ci` - 0.1 dB (non-inferiority).
- **G4 densification participation** — in `adc` mode, the fraction of the original 800 rows
  that meet the controller's own densification criterion (accumulated screen-space positional
  gradient above `grad_threshold`, evaluated on surviving original rows at each scheduled
  round) at least once is >= 5x the `ci` arm's fraction. The criterion is read from public
  controller state rather than reconstructed from `density.py` internals; realized
  clone/split/prune totals are reported alongside but are budget-capped and are therefore not
  the gate.
- **G5 attribution** — `surfel` initial alpha-IoU >= `ci-op` + 0.10 absolute. If G5 fails while
  G1 passes, the finding is "optical thickness was the whole story" and the covariance claim is
  not supported.
- **Guardrail** — no arm may end `adc` mode with held-out alpha-outside above 0.05.

## Decision rule

This is one scene, one seed, one device, and a reduced resolution. **No default change is
authorized by this screen under any outcome.** The outcome selects which mechanism is real and
what a confirmatory multi-scene, multi-seed, GPU protocol must control. A failed G5 closes the
covariance claim and reduces the finding to an opacity rule. A failed G1 with a passed G4 would
indicate that participation and coverage are separable and must be preregistered separately.

## Amendments and chronology disclosures

These are recorded rather than silently applied. Neither touches an arm definition, a metric, a
gate, or a threshold.

1. **Diagnostic correction (before any run outcome was read).** The `frame_00009` cell of
   "fused sigma_max / smallest contributor footprint" was first written as `1.750`, which is the
   `frame_00008` value; the measured `frame_00009` median is `1.660`. Corrected above.
2. **G4 wording.** The participation gate originally read "selected for clone or split at least
   once". It is stated above in the form the harness actually measures — the controller's own
   gradient criterion on surviving original rows — because realized clone/split counts are
   budget-capped and would confound the gate with the cap. Frozen before the run.
3. **Pre-run smoke.** After this protocol was frozen, a 50-iteration two-arm (`ci`, `surfel`)
   smoke was run to validate the harness. Initialization metrics are schedule-independent, so
   the `ci` and `surfel` **initial** held-out numbers were seen before the official run started.
   All gates were already frozen at that point and none was changed afterwards; no trajectory,
   final, participation, or other-arm value was observed.

## Official command

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/beam_surfel_init.py \
  --protocol benchmarks/results/20260724_beam_surfel_init_PREREG.md \
  --out runs/beam_surfel_init_20260724
```
