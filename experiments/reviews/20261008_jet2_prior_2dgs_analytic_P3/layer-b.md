# P3 packet, Layer B (intent): jet2 analytic pilot

Plan: `docs/TASK_jet2_synthetic_sphere_ellipsoid.md` (P3). Stage: **pilot / calibration**. Its
outputs inform whether and how P4 (base / jet1 / jet2 x 3 seeds) is run; they are not evidence
for or against the 2-jet prior and stay labelled pilot exposure.

## Questions

1. **Teacher.** At the activation step (7500) of a base 2DGS fit, does the Jet2Prior's own
   shape-operator estimator recover the analytic curvature better than `S = 0` (the quantity
   jet1 uses instead), and on how many splats does its gate let it act?
2. **Instrument.** Does the spectrum path (companion operator on fitted surfels) reproduce the
   reference spectrum for ideal surfels, and does it react to normal noise?
3. **Floor.** How large are seed-to-seed and run-to-run differences of every P4 candidate metric
   for the base arm (the detection limit P4 must exceed)?
4. **Cost.** Wall time and peak RSS of the operator on base fits.

## Known calibration before this run (deterministic, `tests/test_jet2_eval_controls.py`)

Ideal surfels, N = 4000, sigma = 0.75 sqrt(A/N): estimator relative curvature error 0.000
(sphere) / 0.007 (ellipsoid) with exact normals; 0.57 / 0.43 with every normal tilted by 5 deg.
Mechanism: over the estimator radius the true normal turns by about 0.1 rad, so a few degrees
of normal error is comparable to the curvature signal.

## Predictions (made before any P3 output)

- Teacher: the trained base normal median at 7500 is between 2 and 10 deg; by the calibration
  above, the estimator's median relative curvature error then lies between about 0.2 and 1.0.
  I do not predict which side of 0.5 it falls; the gate's masked fraction is below 0.5.
- Instrument: exact ideal surfels give max |rel. error| over lambda_1..lambda_10 <= 2 % (sphere)
  and <= 3 % (ellipsoid) at N = 15000; at N = 4000 at most twice that. 5 and 10 deg tilt raise it
  monotonically, by more than a factor 2 at 10 deg.
- Floor: seed-to-seed range of the base normal median <= 1 deg; run-to-run range smaller.
- Cost: operator <= 30 min and <= 12 GB per base model.

## What an ineffective estimator would show

Median relative curvature error near or above 1 (no better than `S = 0`), or a masked fraction
near 1 (the prior would act on almost nothing). In either case jet2 cannot differ from jet1 for
the reason the hypothesis gives.

## Decision rules (frozen)

- **Teacher go** on a surface iff, on base 9561 at step 7500, opacity > 0.3:
  `curvature_rel_error_median <= 0.5` and `estimator_masked_fraction <= 0.5`.
- **Instrument pass** iff exact ideal surfels at N = 15000 give `status = valid` and max
  |rel. error| over lambda_1..lambda_10 <= 3 % on that surface, and 10 deg tilt raises it.
  Without a pass the spectrum stays descriptive in P4 (not a decision metric).
- **P4 proceeds** on a surface only with teacher go. No go on both surfaces: P4 is not run in this
  form; the result (with the calibration) is recorded, and any redesign (e.g. a larger estimator
  radius, which trades noise for curvature resolution) is a new question with a new packet.
- **Detection limit** per metric m: `DL_m = max |m_a - m_b|` over the four base runs of a surface
  (seeds 9561, 9562, 9563 and the 9561 repeat). P4 thresholds must exceed `DL_m`; P4 freezes them.
- **Surprise rule**: a base normal median above 15 deg, a failed structural gate on a base fit,
  or an operator RSS above 16 GB voids the remaining plan until the packet is revised.

## Controls

Positive: exact ideal surfels (geometry and spectrum). Negative: 5 and 10 deg tilted ideal
surfels. Deterministic evaluation controls (tilt, inward shift, clones, opacity selection) pass
in the test suite. `S = 0` is the reference the teacher must beat (relative error 1 by definition).

## Replication and analysis

Replication unit: a fitting run (seed). Teacher: one run per surface (pilot, descriptive plus
the go rule). Floor: four base runs per surface. All numbers are reported with counts (subset,
cohort, curvature-eligible). Metrics and units: angles in degrees, distances in surface units
(R = 1, a1 = 1), curvature errors relative (1 = `S = 0`), spectra as relative errors of
lambda_1..lambda_10, raw and area-normalised. No selection, no aggregation beyond medians/p90 as
defined in `analytic_geometry`.
