# Preregistration: compact footprint occupancy scalar ablation

Status: outcome-free implementation and deterministic-core amendments frozen; execution requires
explicit root go-ahead

Frozen: 2026-07-17T02:01:58+02:00 (Europe/Berlin)

Outcome-free implementation amendment: 2026-07-17T02:05:41+02:00. Before any official metric or
run output, root review identified camera-only `torch.linalg.lstsq` center nondeterminism. The
harness now evaluates that center/extent exactly once and injects it as an explicit `bounds_hint`
into one Stage-B `ReconstructionInputs` clone consumed by every arm and failure diagnostic. Six
focused tests, rather than the original five, bind this intervention. No scalar, split, metric,
selector, arm, or threshold changed.

Outcome-free deterministic-core amendment: 2026-07-17T02:10:11+02:00. Before any official metric
or run output, the compact-Carve camera-center helper replaced the CPU-default `gelsy` solve with
an explicit hybrid: `gelsd` determines numerical rank and supplies the minimum-norm solution for
rank-deficient rigs, while full-rank systems use explicit `gels` to preserve the established
solution. Four compact-Carve CPU tests bind full-rank repeatability/equivalence, rank-deficient and
near-degenerate behavior, and explicit driver selection. The harness still freezes the resulting
center once for all arms. This source repair changed no scalar, split, metric, selector, arm,
threshold, or previously observed scientific outcome.

## Chronology and claim boundary

This protocol was frozen after the completed masked-lift mechanism screen and a read-only
footprint-sizing diagnostic, but before the official Stage-A occupancy metrics, temperature
selection, report-view occupancy metrics, or any Stage-B lift in this namespace. The prior screen
showed that the center-gated compact proxy had pooled precision `0.9870`, recall `0.7817`, IoU
`0.7737`, and AUC `0.9880`, and that its 835-center lift put `99.40%` of Gaussians on foreground in
at least two training views. The sizing diagnostic used the same deterministic 32 footprint
samples and observed that a Gaussian sample mean changed the count of components above scalar
`0.5` very little, while LSE beta 8/16 approached the hard-max activation ceiling. Those observed
facts motivated this exploratory experiment and prohibit treating it as outcome-blind
confirmatory evidence.

The question is narrow: can one mask-derived scalar per optimized 2D Gaussian better represent
footprint occupancy than the mask value at its center, while retaining exact masked compact color
teachers and fixed component-center rays? The experiment may support only a seven-training-view,
unrefined Stage-2 initialization mechanism claim. It cannot establish novel-view quality,
refinement convergence, memory/runtime superiority, generalization to another scene or 2D
producer, a production default, or state-of-the-art performance.

## Frozen source and inputs

- Harness: `benchmarks/compact_occupancy_scalar_ablation.py`, SHA-256
  `3be876d49885b5008baaf3afa42843da884e5ae144f05d3d2974b10b65f510a4`.
- Focused tests: `tests/test_compact_occupancy_scalar_ablation.py`, SHA-256
  `8b5f8918fabdcfe7a5e2c59fdaba1dec4228c03af64e5f45f0721bde3a376c89`.
- Deterministic compact-Carve source: `src/rtgs/lift/compact_carve.py`, SHA-256
  `810fd03f3ab057756ad5d730a93b7ee5b204956003b0fd31602612ff7373edc1`.
- Compact-Carve tests: `tests/test_compact_carve.py`, SHA-256
  `9f7c666044b3b10625717e93bc4bc7223c80d76aca2693e28019109ec710ebc6`.
- Reused audited helper harness: `benchmarks/compact_masked_lift_screen.py`, SHA-256
  `90b27af700ffe572bbafe4efbf93aca03d169b9a1b52725fbffd92b8b5443bf0`.
- Runtime transitive-source aggregate: SHA-256
  `02f910f711dacd7328e76528cac357abd6fed5598f0b3d792bc7440b35157f40`.
- Exact masked seven-view bundle manifest: SHA-256
  `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`.
- Prior screen result file: SHA-256
  `95f67619e523b6ff0192cb43c43b570fd9b64fdbb34341f5995009cfeca86a69`.
- Prior center-proxy PLY: SHA-256
  `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`.

The harness strict-loads and hashes every teacher archive and derived proxy bundle, hashes raw and
undistorted masks, binds transitive executed source, and refuses to overwrite an output. It writes
`plan.json` before decoding a mask or constructing any scalar, then writes observed tensor/proxy
bindings separately. Source RGB opens are denied for Stage A and Stage B. Stage B may query color
only from the exact masked compact fields; masks supply occupancy scalars and evaluation labels,
never color.

## Frozen scalar construction

For every optimized 2D Gaussian, draw 16 scrambled Sobol points transformed to standard normal
and append their 16 exact antithetic pairs. The seed is `18017`; the resulting float32 `(32,2)`
tensor hash is `7dbfe44552126414cfb54eaf5c366f5aff2f4ef8834771aded7068fcb5783667`.
Transform the shared points by each component's effective covariance and bilinearly sample the
undistorted binary mask. Persist exactly one scalar per component:

- `center`: mask at the Gaussian mean;
- `mean`: equal-weight sample mean, a deterministic quasi-Monte-Carlo estimate of Gaussian mask
  expectation;
- `lse_beta_{2,4,8,16}`: `(logsumexp(beta*m_k)-log(32))/beta`;
- `hard_max`: maximum sampled mask, used only as a non-smooth ceiling.

All scalars must be finite and in `[0,1]`. Geometry, support, filter variance, and fitted window
remain exact. Proxy colors are unused ones. Coverage uses `r=A*D/M` independently in each view,
so multiplying every scalar in a view by one common positive constant cancels. This tests the
relative occupancy distribution across components, not absolute scalar magnitude.

## Stage A: frozen split, metrics, and selection

Use 16,384 deterministic uniform full-canvas pixel centers per view (`seed=18018+view_index`) and
the existing coverage mapping `h=1-exp(-r)`. At fixed `h>=0.4`, report per-view and pooled
precision, recall, IoU, confusion counts, and average-tie AUC.

- Temperature-tuning views: `C0001`, `C0014`, `C0026`.
- Selection-report views: `C0008`, `C0021`, `C0031`, `C0039`.

Only the tuning metric object is accepted by the selector. The selector is serialized before the
report metrics are queried. The report views are held out only from official temperature
selection: their fields/masks exist, the earlier sizing diagnostic inspected component activation
fractions on all views, and Stage B later consumes all seven views. They are not novel-view or
scene-held-out evidence.

Set the precision floor to center pooled tuning precision minus `0.01`. Among LSE variants that
pass it, select maximum pooled recall, then pooled IoU, pooled AUC, then lower beta. If no LSE
passes, fail closed. Stage B always includes center, mean, and selected LSE. Include hard max only
if it passes the same precision floor and exceeds selected LSE by at least `0.02` recall or `0.01`
IoU on tuning views. Report metrics cannot change that decision.

## Stage B: fixed lift and completion rules

Pin PyTorch intra-op and inter-op CPU threads to one before plan creation. Evaluate the
camera-only compact-Carve center/extent once, serialize the exact center/extent/box, and inject the
center/extent as `bounds_hint` into one Stage-B input clone. Every arm and support diagnostic uses
that clone and cannot rerun the least-squares center. Every arm runs in one process with the bound
835-Gaussian configuration, `anchor_mode="component_centers"`, exact masked compact colors, and
only its proxy `weight_sum` substituted. Wrapper audits require bit-exact color, numerator, and
valid outputs. Save `gaussians_init.ply` and identical unrefined `gaussians.ply` for every feasible
arm.

Report candidate and eligible counts, unique source identities, source-center foreground rate,
background-in-all-views rate, foreground projection histograms, foreground in at least 2/6 views,
and common sampled masked-teacher uniform/foreground MSE and PSNR. Center and selected LSE must
pass; mean and optional hard max may be recorded as bounded `INFEASIBLE` only for insufficient
supported placements. Other exceptions fail the experiment.

The new center proxy tensors/query semantics must replay the prior proxy bit-exactly. The bound
solver now makes repeated current-source camera-center solves deterministic, and the harness then
injects that solve once for every arm. Cross-run 3D PLY bit equality against the earlier pre-repair
artifact is still explicitly not claimed: record byte and numeric differences, and fail only if
means exceed `1e-5` maximum absolute difference or covariance exceeds `5e-3`. Those legacy-reference
tolerances were frozen from the separate pre-repair nondeterminism audit. All scientific arm
contrasts are within the new one-thread process.

## Frozen command and stopping rule

After final implementation review and explicit root authorization, execute once:

```bash
PYTHONPATH=src .venv/bin/python -u benchmarks/compact_occupancy_scalar_ablation.py \
  --anchor-bundle runs/compact_masked_bundle_640_20260717/reconstruction_inputs \
  --out runs/compact_occupancy_scalar_ablation_20260717
```

Do not tune a beta, precision floor, sample count/seed, hard-max rule, lift configuration,
threshold, arm list, or tolerance after outcome access. A failed or partial output remains
consumed and is not overwritten.
