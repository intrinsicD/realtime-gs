# Compact-only carrier stage ablation — result — 2026-07-28

## Status

**Complete development result; passed independent audit with scope limits.**

- Protocol:
  [`20260728_compact_only_carrier_stage_ablation_PREREG.md`](20260728_compact_only_carrier_stage_ablation_PREREG.md)
- Machine result:
  `runs/compact_only_carrier_stage_ablation_20260728/result.json`
- Independent audit:
  [`20260728_compact_only_carrier_stage_ablation_AUDIT.md`](20260728_compact_only_carrier_stage_ablation_AUDIT.md)
- Command:
  `.venv/bin/python benchmarks/compact_only_carrier_stage_ablation.py`

This was a single-scene, real-data development experiment. It does not authorize a general
quality, default, speed, or VRAM claim.

## What actually ran

The process loaded 26 integrity-bound `.rtgsv` containers with `load_alpha=False`. The optional
packed-alpha member was not read or decoded. The live guard reported:

- zero image-file open attempts;
- zero forbidden image/dense-trainer imports;
- three successful negative-control denials; and
- no `PIL`, OpenCV, `rtgs.data.calibrated`, `rtgs.data.scene`, or
  `rtgs.optim.trainer` module crossing the boundary.

Beam Fusion used only the 19 fitting cameras and produced 5,000 carriers. Nine repair
initializations, 16 fixed-topology arms, three paired training roots, and four checkpoints were
then evaluated by point queries against frozen 2D Gaussian fields. Validation ranked arms;
held-out cameras were unlocked only for the frozen finalist set.

Compact training with 5,000 rows allocated at most 74.42 MiB and reserved 80.00 MiB in the
PyTorch CUDA allocator, with the compact teachers resident. That is an absolute diagnostic, not
a comparison to dense-image training and not yet a general VRAM claim.

## Validation result

Geometric means across the three roots:

| Arm | Validation `J_Q` | Validation `J_U` | Mean outside-center fraction |
| --- | ---: | ---: | ---: |
| corrected C + corrected A, all parameters | 0.00549373 | 0.00946172 | 0.007267 |
| corrected C, all parameters | 0.00549789 | **0.00944435** | 0.007183 |
| legacy A, all parameters | 0.01488894 | 0.01868734 | 0.006800 |
| Beam, means frozen | 0.01491001 | 0.01858216 | 0.007200 |
| Beam, all parameters | 0.01492036 | 0.01870341 | 0.006800 |
| legacy O, all parameters | 0.01666881 | 0.02005832 | 0.007000 |
| legacy C, all parameters | 0.02304407 | 0.02456085 | 0.006900 |
| Beam, means only | 0.03711604 | 0.03055237 | 0.006567 |

The numerical `corrected_CA_all` winner does not make corrected appearance a necessary stage:
relative to corrected C alone its `J_Q` edge was only 0.076%, while `J_U` was 0.184% worse.

## Stage decisions

### Legacy repair chain

Drop it.

- Legacy covariance repair worsened factorial-marginal validation `J_Q` by 51.63%, with 0/3
  paired wins. The old residual omitted renderer dilation and penalized expansion much more than
  collapse.
- Legacy opacity repair worsened `J_Q` by 9.78%, with 0/3 wins. Its normalized 2D mixture
  amplitude is not identifiable physical alpha.
- Legacy appearance repair improved `J_Q` by only 0.12%, below the preregistered 5% necessity
  floor. Cross-view amplitude weighting is also gauge-dependent.

### Corrected covariance repair

Retain it as the only currently material repair for the next development experiment.

The renderer-aware generalized-log residual reduced validation `J_Q` by 76.14% relative to
legacy covariance and by 63.15% relative to directly optimizing Beam, with 3/3 paired wins.
It uses the actual point-renderer covariance

`P = J Sigma J^T + 0.3 I`

and the reciprocal generalized-eigenvalue residual

`sqrt(mean(log(lambda(C^-1/2 P C^-1/2))^2))`.

This remains an initialization heuristic: some fitted 2D targets are sharper than the renderer's
0.3-pixel covariance floor and therefore cannot be matched exactly.

### Direct fixed-topology optimization

Keep compact all-parameter optimization; reject means-only.

- Means-only was 148.76% worse in `J_Q` and 63.35% worse in `J_U` than all-parameter Beam.
- Freezing means passed the 2% non-inferiority gate for the raw Beam initializer (`J_Q` ratio
  0.9993, `J_U` ratio 0.9935), but the corrected-covariance × means-freeze interaction was not
  tested here.
- Geometry-only and appearance-only were both materially worse. Scale/orientation and
  opacity/color must co-adapt even when center motion is unnecessary.

### Support / “no free floaters”

The tested soft barrier failed and must not be kept as implemented.

- Beam + support changed validation `J_Q` by +0.009%.
- Its outside-center fraction was 0.006817 versus 0.006800 without the barrier.
- Corrected CA + support reduced its parent outside fraction by only about 0.9%, far short of the
  50% gate.

The mathematical limit is more important than this negative result: requiring a projected center
to lie inside every fitted 2D Gaussian union approximates a visual hull. It can constrain an
outside projection, but it cannot detect a free Gaussian that lies inside that hull. Full
footprint containment would also bias legitimate boundary Gaussians inward.

The next experiment therefore tests two stronger, distinct invariants:

1. freeze Beam-derived means so every row retains its original at-least-three-view carrier
   support; and
2. deterministically prune any center outside any fitting-view Gaussian union, then recover only
   non-mean parameters so the containment proof cannot be undone.

## Math audit

| Stage | Disposition | Audit result |
| --- | --- | --- |
| Beam ray intersection | keep for development | Closest-ray signs and world/camera conventions are consistent. |
| Beam covariance intersection | keep for development | Equal-weight CI is internally consistent for correlated observations. |
| Tangent covariance lift | keep with renderer caveat | Lift is correct, but the student adds 0.3 EWA dilation that the fitted 2D field does not. |
| Legacy covariance repair | drop | Asymmetric whitened Frobenius residual plus omitted dilation biases collapse and worsened validation. |
| Corrected covariance repair | keep for next experiment | Symmetric generalized-log residual uses the rendered covariance and passed its replacement gate. |
| Opacity repair | drop | Normalized mixture amplitude is gauge-dependent and does not identify per-carrier opacity. |
| Appearance repair | drop as a separate stage | Robust SH0 averaging is a valid initializer, but its weighting is not physical and its effect was immaterial. |
| Compact point RGB loss | keep | Correct sampled objective for the actual 3D compositor versus queried 2D fields; it does not by itself identify opacity/support. |
| Soft ellipse-center support | drop at tested strength | Correctly has gradients outside clipped support and is opacity-independent, but did not engage materially. |
| Legacy clone/copy stage | do not retain untested | Copying parent opacity while retaining the parent changes coincident alpha from `a` to `1-(1-a)^2`. |
| Higher SH / later density | do not retain untested | Both can be supervised compactly, but neither is justified by this experiment. |

## Held-out descriptive check

The mechanically selected corrected-CA finalists replicated their low compact-field risk on the
three held-out cameras (`J_Q` approximately 0.00687–0.00702). These values did not select stages
and do not repair the single-scene or source-binding limitations.

## Provenance limitations

The producing manifest bound the preregistration, input containers, configs, banks, model
receipts, and environment, but it did not freeze the Git revision, dirty diff, exact command, or
an executed-source archive before running. The independent audit found no relevant source mtime
newer than the estimated run start and reproduced every gate from raw receipts, but that is
post-run corroboration rather than a cryptographic execution seal.

The result is suitable for choosing the next bounded development arms. It is not suitable for a
production default or a general “no images / quality / VRAM” claim without multi-scene replay
from a source-sealed harness.
