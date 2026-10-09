# P3 packet, Layer B (intent): jet2 analytic pilot — revision 2

Stage: **pilot / calibration**. Outputs decide whether and how P4 (base / jet1 / jet2) runs; they
are not evidence for or against the 2-jet prior and stay labelled pilot exposure.

## Questions

1. **Teacher.** At activation (step 7500) of base fits, does the Jet2Prior's own estimator recover
   the analytic curvature better than `S = 0` (what jet1 uses), and on how many splats can it act?
2. **Instrument.** Does the spectrum path reproduce the reference for ideal surfels, react to
   normal noise, survive mixed surfel sizes, and differ between `S` and `S ≡ 0`?
3. **Spread.** Seed-to-seed and run-to-run differences of the base arm, per reported field.
4. **Cost.** Operator wall time and peak RSS on base fits.

## Known before this run

Deterministic controls (`pytest.txt`): with ideal surfels (N = 4000, σ = 0.75 sqrt(A/N)) the
estimator's relative curvature error is ~0 with exact normals and 0.43-0.57 with every normal
tilted by 5 deg (over the estimator radius the true normal turns by ~0.1 rad). Technical runs
(not read as results): operator on a 600-step sphere model (12.5k splats) 7 min, 6.4 GB; on 4000
ideal surfels 9 s, 0.8 GB.

## Predictions (before any P3 output)

- Teacher: trained base normal median at 7500 between 2 and 10 deg; estimator median relative
  curvature error between ~0.2 and 1.0; which side of 0.5 is not predicted. Masked fraction < 0.5.
- Instrument (raw `max_abs_rel_error` over λ_1..λ_10): exact N = 15000 ≤ 2 % (sphere), ≤ 3 %
  (ellipsoid); N = 4000 at most twice that; tilt 10 deg at least 2× exact; spread row within 2×
  exact; flat row differs from exact (sign not predicted on these convex shapes).
- Spread: seed-to-seed range of the base normal median ≤ 1 deg; repeat difference smaller.
- Cost: ≤ 30 min and ≤ 12 GB per base model (from the 12.5k technical run, roughly linear in N).

## Ineffective estimator signature

Median relative curvature error near or above 1, or masked fraction near 1: jet2 could then not
differ from jet1 for the reason the hypothesis gives.

## Decision rules (frozen in task `pilot`; missing values never pass)

- **Teacher go** on a surface iff for **both** teacher seeds (opacity > 0.3):
  `curvature_rel_error_median ≤ 0.5`, `estimator_masked_fraction ≤ 0.5`, `n_curvature ≥ 1000`,
  `collapse_fraction_at_activation ≤ 0.05`. The masked fraction over all splats is reported beside.
- **Instrument pass** iff row `n15000_tilt0` is `valid` with raw `max_abs_rel_error ≤ 0.03` and
  row `n15000_tilt10` is `valid` with at least 2× that error. Without a pass the spectrum is
  descriptive only in P4.
- **P4 runs** on a surface only with teacher go. No go on both: P4 is not run in this form; any
  redesign (e.g. a larger estimator radius, trading noise for curvature resolution) is a new
  question with a new packet.
- **Spread** is reported as the range over the four base runs (3 seeds + repeat) and separately
  the repeat difference. With n = 4 this is a descriptive floor, not a detection limit or a
  variance estimate; P4 must set its thresholds above it and state its own replication argument.
- **Surprise / stop rules** as in Layer A (enforced in code between phases).

## Controls

Positive: exact ideal surfels (geometry, spectrum). Negative: 5/10 deg tilt; mixed sizes (the
regime of trained splats); `S ≡ 0` spectrum. Deterministic geometry controls in the test suite
(tilt, inward shift, clones, opacity selection with a tail statistic, coverage hole, sampler
uniformity). Not run as spectrum controls, with reason: shift (λ scales with 1/R², covered by
area normalisation and absolute distances), clones (stopped earlier by the collapse fraction).

## Analysis (all summaries reported, none selected)

Per model: every field of `analytic_geometry` — medians, p90, means and fractions as defined
there, with counts `n_subset`, `n_cohort`, `n_curvature`; held-out PSNR / LPIPS / IoU means;
operator status, λ_0..λ_31, raw and area-normalised relative errors of λ_1..λ_10 and their max,
area error, G15a, n90, cond(M), resources. Units: degrees; surface units (R = 1, a1 = 1);
curvature error relative (1 = `S = 0`). `None` = unevaluated.

## Deviations from the TASK plan (rev. 3), recorded

Counts 5k/10k/20k → 4000/15000; perturbations 2/5/10 deg → 0/5/10; shift/clone/drop spectrum
controls → geometry tests; transported-normal comparison dropped (the estimator comparison against
`S = 0` answers the teacher question); Hoppe agreement not reported (orientation is not under
test; analytic orientation is used); closest point by bisection with a singular branch, not
Newton; operator thickness is `sqrt(σ_t1 σ_t2)/50` (companion loader), not `σ_t1/50`.
