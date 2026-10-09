# Plan: 2-jet prior on analytic surfaces (sphere, ellipsoid) before the cat

Status: plan rev. 3, 2026-10-08. P1 and P2 implemented and reviewed (Codex approve); P3 packet
in `experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3/`, awaiting two reviews. Nothing
results-bearing run. Task RTGS-030 continuation.
Rev. 1 was reviewed by Codex (thread 01a11b57-2016-7921-8ca1-1d2686f66b47) and a fresh Claude
session (11cc7558-abe8-46f9-9df8-6ce919fce3ee); both `revise`. Changes are marked [R2]. Rev. 2 exchange round: both `revise` for P1 with concrete
estimator requirements; rev. 3 implements them (section "P1 as implemented").

## Goal and scope of the claim

Decide, on surfaces where every target is known in closed form, whether a 2-jet (shape-operator)
consistency prior during 2DGS training gives better geometry than no prior and than the 1-jet
(S = 0) prior. All three arms keep the 2DGS normal-consistency anchor, so the claim is only the
**incremental effect with that anchor on** [R2]. A positive synthetic result licenses a
separately calibrated cat protocol, not a cat claim [R2].

## Ground truth

- Sphere R = 1: normal `x/|x|`, `S = I` (outward normals; surface `z = −½ sᵀSs` in a splat frame),
  LBO `l(l+1)` with multiplicities 1, 3, 5, 7.
- Ellipsoid axes (1, 0.75, 0.5): normal and `W = P∇²F P/|∇F|` (SplatDiffuseLBO `weingarten`);
  LBO λ1..λ10 from ContinuousLevelSetLBOGS `results/e12/evidence.json` (rel. unc. 3.1e-4),
  copied with its hash into the data seal [R2].
- Closest point: exact on the sphere; Newton on the Lagrange condition on the ellipsoid, with an
  asserted residual [R2]. All distances in absolute units (R = 1, a = 1) [R2].

## Expected mechanism (to be pinned by tests before any run) [R2]

On a sphere with symmetric weights, both priors exert ~zero torque on normals; jet1's height
term pulls centres toward neighbours' tangent planes (inward, a shrink), jet2's does not.
On the ellipsoid jet1 additionally biases normals by O(σ²|∇H|) (small). jet1 may also *help*
by smoothing correlated normal noise that jet2 absorbs into S. Hence the primary signal on
the sphere is signed distance (shrink), on the ellipsoid shrink plus curvature (`tr S_est/tr W`).

## Phases

**P1 — v3 prior + mechanism tests (code; tests, not runs).**
- Prior in `rtgs/optim/jet_prior.py`, reusing `field_estimate`'s pair/weight machinery and the
  `jet_residuals` algebra. [R2] Neighbourhood radius `3 σ_t1,i` with tangent-Gaussian weights
  `exp(−½ sᵀΣ_t,i⁻¹ s)` × density normalisation `w_j/ρ_j` — the same estimator as
  SplatDiffuseLBO `shape_estimated`, so training, teacher check and evaluation use one estimator;
  a surfel's own scale is not shrunk by clones (fixes the frozen-h0 failure). Detached `S_i`.
  `order=1` sets `S = 0`. Starts at 7500 (after densification); count change raises. Logs the
  fraction of ill-conditioned S fits (ridge turns them into jet1).
- Tests, assertions written before first execution:
  1. Sphere, exact surfels: both priors' normal torque ≈ 0; jet2 residual and centre gradient
     ≤ 1e-2 × jet1's (scaled bound, the paraboloid differs from the sphere by `|d|⁴/8R³`);
     jet1 centre descent points inward.
  2. Ellipsoid patch, prior-only gradient descent from exact surfels: jet1 lowers
     `tr S_est/tr W`, jet2 keeps it within 1 %.
  3. Perturbed normals (fixed 10°), prior-only descent: jet2 reduces the normal error.
  4. 8 coincident copies per splat, created before activation: residual unchanged (1e-6 rel).
  5. Half the normals flipped: residual unchanged. Rank-deficient neighbourhood: finite, flagged.

**P1 as implemented (rev. 3).** The estimator is defined in the module docstring of
`jet2_prior.py`, not a port of SplatDiffuseLBO `shape_estimated` (that one has no density
correction, anisotropic weights and no hemisphere flip). Isotropic weights at the frozen largest
tangential scale; detached hemisphere flip; density `rho` incl. zero-offset copies; zero-offset
pairs excluded from the fit; gate on neighbour mass `A_i >= 0.5` (a count-based gate was not
clone-invariant: found by the first test run) and design conditioning `>= 0.02`.
Tests (analytic only; empirical descent comparisons moved to P3 as Codex required):
exact sphere S = I and jet2 << jet1; `r_nu` vanishes identically on the sphere (rev. 2 predicted
O(sigma^6): a derivation error, corrected) and `r_c` converges at sixth order; 40 deg cap with
fixed rim: jet1 descent flattens, jet2's is < 1 % of it; half the normals flipped; 8 uniform
clones (residual and gate mass); 10:1 needle surfels; isolated splat; sparse well-conditioned
neighbourhood rejected by the mass gate; schedule, gradient paths and topology guard. Mutation
check: removing density normalisation, the zero-offset exclusion, the mass gate, the hemisphere
flip, S (jet2 -> S = 0) or flipping the height sign each fails at least one test.

**P2 — data + thin driver (technical test only, outputs not read).**
- `prepare_tosca_cat0.py` gets a surface argument only if that file is not bound by a registered
  task; otherwise a thin new prepare script imports its renderer [R2]. Procedural vertex colour,
  160 Fibonacci cameras, 512², 128/32 split. Sphere first, ellipsoid second.
- New thin driver imports the 20261008 driver (as v2.1 imports v1); no edits to files bound
  by the 20261008 task [R2]. Formally withdraw the 20261008 registration first [R2].
- LBO adapter [R2]: 2DGS normal width is 0 in training; evaluation freezes a positive thickness
  `σ_n = σ_t1/50` (SplatDiffuseLBO loader default), trial width `σ_χ = σ_t`, `measure="closed"`,
  GT orientation (Hoppe agreement reported separately).

**P3 — pilot packet (preflight + two reviews).** Superseded in detail by the packet
`experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3/` (rev. 2), whose Layer B lists the
deviations from this section. Original text: [R2]
- Evaluation calibration on frozen synthetic surfel sets (counts 5k/10k/20k, σ_t from
  `sqrt(area/N)`): exact; normals perturbed 2°/5°/10°; 8 clones; inward shift 0.002/0.005;
  worst 10 % dropped; S set to 0 in the LBO. Each must move its metric beyond a frozen threshold.
- Teacher check: base arm to step 7500, then the **actual training estimator** vs `W` and vs
  `S = 0`; transported normal vs trained normal vs GT. Go rule frozen in the packet.
- Base seed spread (3 seeds to 15000) = detection limit for P4.
- Operator memory/time at the largest base model.

**P4 — confirmation packet (preflight + two reviews), GPU.**
- Arms base / jet1 / jet2, seeds 9561–9563, sphere then ellipsoid; plus λ 0.03 and 0.3 for each
  prior, seed 9561, ellipsoid [R2]. Pre-activation parameter hash equal across arms per seed.
- Primary (estimator-free): normal angle to GT, signed distance (mean, fraction inside),
  coverage, on the frozen-cohort metrics of the 20261008 driver, with subset counts [R2].
  Secondary: `S_est` vs `W`, `tr S_est/tr W`, LBO λ1..λ10 raw and area-normalised.
  Gates: near-duplicates, coverage, PSNR floor; any seed violating → arm fails.
  Paired per-seed thresholds above the P3 detection limit; go / stop / inconclusive rules frozen.
  Unevaluated essential geometry is never a go.

## Done criterion

P1 tests green with the predicted signs; P3 and P4 packets each pass two reviews; runs complete
or explicitly unevaluated; decision recorded in the task record and `docs/EXPERIMENTS.md`.

## Verification

`./scripts/verify.sh`; `pytest tests/test_jet_prior.py`; `experiment_contract.py validate`;
`check_results_bundle.py` on each run root.

## Open

- Sphere and ellipsoid have h·κ far below the cat's high-curvature regions; a difference that
  only appears there will not show here. Kept as the user's chosen first step; noted as a limit.
- GPU shows 1.8 GB used by another client; jet cells may need 4–5 GB.
