# Research portfolio: high-quality reconstruction from 2D Gaussian fields

Literature/search cutoff: 2026-09-07. Research proposal, not a measured reconstruction result.
Owner: RTGS-020. Codex inspected the repositories; Claude supplied an independent public-literature critique. The [collaboration receipt](research/2026-09-07-field-reconstruction/collaboration-receipt.json) records the exact scope and model provenance. Neither existing experiment is reopened by this proposal.

Sources searched: arXiv full texts, original author project pages, publisher papers, and the repository's previous literature survey, RTGS-016 audit and BENCH-019 audit. Query families covered Gaussian CT, exact ray integration, image-plane Gaussian lifting, multiview stereo, initialization/densification interaction, cryo-EM, astronomical deprojection, compression and inverse-problem validation. The goal is working reconstruction; there is no claim of a new method or exhaustive prior-art coverage.

## Answer and proposed direction

The user's tomography intuition is correct for **projected densities**. A Gaussian mixture is a useful parameterization of a shared 3D field, and several imaging disciplines reconstruct one from 2D observations. The missing condition is that the observations must follow the forward model being inverted.

For ordinary photographs, the most credible engineering route is:

1. Preserve enough appearance detail in the fitted 2D fields.
2. Estimate multiview geometry from **queries of those fields**, using spatial patches/features and calibrated rays, rather than assuming individual fitted ellipses are corresponding physical objects.
3. Initialize a shared 3D Gaussian scene near the supported surfaces.
4. Optimize it against the complete field colours with visibility-aware alpha rendering, adequate spatial capacity and a tested topology controller.
5. Evaluate independently against held-out photographs, reporting appearance, coverage and geometry separately.

This is field-supervised inverse rendering. It can consume only fields and cameras after Stage 1; rasterizing a field into a temporary image or querying a patch does not require reopening the original photograph. That distinction preserves the information boundary, but a method that decodes images must account for the memory and runtime. Learned matchers add an external prior and must be disclosed. A stronger field-only initialization and high-capacity quality remain hypotheses here.

## What “2D density” must mean

In ideal attenuation CT, after calibration and a logarithm,

\[
p_v(u)=-\log(I_v(u)/I_{0,v}(u))
      =\int \rho(o_v+t d_v(u))\,dt.
\]

The same nonnegative density contributes along each known ray. [R²-Gaussian](https://arxiv.org/html/2405.20693v2) fits radiative Gaussians using this operator and corrects the covariance-dependent amplitude factor in projection. Its CT results do not establish an RGB reconstruction result.

For a unit ray direction, an unnormalized 3D Gaussian with precision Q, peak a and displacement b=o-m has the full-line integral

\[
\int_{-\infty}^{\infty}a\exp[-(b+td)^TQ(b+td)/2]dt
=a\sqrt{2\pi/(d^TQd)}\exp[-(b^TQb-(d^TQb)^2/(d^TQd))/2].
\]

This follows by completing the square. Finite ray segments add Gaussian-CDF endpoint factors. Under parallel projection, a normalized 3D Gaussian marginalizes exactly to a normalized 2D Gaussian; general perspective projection is not exactly Gaussian. These are mathematical properties, not evidence that independent image fits are physically consistent projections.

For RGB, an emission–absorption model is instead

\[
C_v(u)=\int T_v(t)\sigma(r_v(t))c(r_v(t),d_v)dt+T_v(t_f)C_{bg},\quad
T_v(t)=\exp[-\int_{t_n}^{t}\sigma(r_v(s))ds].
\]

Standard 3DGS uses a projected, sorted alpha-compositing approximation. Occlusion, colour and viewing direction matter; brightness is not an attenuation integral. [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) supplies the practical renderer, while [Don't Splat Your Gaussians](https://arxiv.org/abs/2405.15425) studies volumetric primitives with a ray-based forward model. A more physical renderer can remove model error; it cannot reveal an opaque object's unseen interior.

For normalized image fields, C=N/(D+epsilon), where D is the summed kernel weight and N its colour-weighted numerator. D is a representation weight field, not measured material density. In the zero-epsilon idealization, rescaling N and D together leaves C unchanged. More generally, splitting, merging and changing a fitted decomposition can preserve image appearance while changing component identities. Copying these component weights to 3D opacity therefore needs independent justification. Even additive colour coefficients do not make the photographed object transparent.

Silhouettes constrain possible occupancy and free space, but cannot alone recover hidden concavities. Photometric evidence and priors are needed. The “2D Gaussians” in [2D Gaussian Splatting](https://surfsplatting.github.io/) are oriented surfaces in **3D space**, unlike the image-plane primitives in StructSplat.

## Frontier map

| Method/component | Primitive | Assumption | Mechanism | Evidence | Failure/open question |
|---|---|---|---|---|---|
| R²-Gaussian | Shared 3D attenuation kernels | Calibrated transmission data | Correct line-integral projection | External CT paper | RGB occlusion breaks the measurement model |
| [MGE astronomy](https://academic.oup.com/mnras/article/333/2/400/1019346) | 2D brightness / 3D luminosity Gaussians | Restricted shape and viewing assumptions | Analytic deprojection | External restricted precedent | Nonunique without structural assumptions; no ordinary opaque-surface model |
| [Random tomography](https://arxiv.org/abs/0909.0349) | Radial-mixture density profiles | Specific statistical projection model | Recovery from many profiles | External mathematical precedent | Does not solve arbitrary visible-light image-mixture association |
| [Cryo-EM GMM](https://pmc.ncbi.nlm.nih.gov/articles/PMC8363932/) | Shared 3D mixture | Known particle orientations / microscopy model | Fit projections of common mixture | External reconstruction precedent | No transfer of microscopy physics to RGB |
| [G²SR](https://arxiv.org/html/2607.14470v1) | Tracked image splats / 3D surface splats | Learned detection and correspondence | Track multiple footprint points; geometric lift | Direct neighbouring method | Independently fitted fields are not its learned detections; rendering coverage remains an issue |
| [EDGS](https://arxiv.org/html/2504.13204v2) | Dense matched points / 3DGS | Reliable calibrated RGB matches | Triangulate, distribute points, refine | External high-quality reconstruction | Applying its matcher to decoded fields needs testing; no direct ellipse-covariance evidence |
| [MVSGaussian](https://mvsgaussian.github.io/) | Stereo-derived Gaussians | Learned multiview geometry | Cost-volume/depth reconstruction | External adjacent evidence | Generalization to fitted-field queries is untested |
| field_refit / field_loss | Projected mixture proxy | Useful underlying density/numerator agreement | Analytic overlap optimization | Existing source and C42 | Not exact normalized RGB or full alpha visibility objective |
| field_sweep | Source rays with queried colours | Cross-view colour/support agreement identifies depth | Robust source-excluded sweep | Existing source, C32–C33 | Texture ambiguity and unsupported midpoint fallback |
| CompactTrainer | Shared 3DGS / exact teacher queries | Correct camera, colour and sampling semantics | Sampled alpha-rendered colour error | Existing source and tests | Calibrated high-capacity outcome not established by RTGS-016/019 |
| ClassicCompactDensityController | Clone/split/prune edits | Screen gradients guide useful capacity | Existing optimizer-state-safe adapter | Existing source and tests | Quality depends on initialization, sampling and controller interaction |

The repository already contains most of the seams. The proposed advance is an informative quality diagnosis followed by a targeted repair, not a new name for the existing field refitter.

## Repository findings and their limits

- [RTGS-016 audit](../benchmarks/results/20260906_tomography_source_constraints_haelyn_dome_AUDIT.md), especially findings 1–4 and limitations: 217/256 carriers, 100 proxy updates and 120 compact native updates did not produce a recognizable subject. Free footprints lowered MSE through drift and blur; a soft tether was inconsistent. Three downstream seeds did not replicate the deterministic proxy. C43 records this limitation. The later task handoff completed the separate delivery gates.
- [BENCH-019 worker](../src/rtgs/bench019_local_downstream.py): `n_init_3d=256`, `densify=False`, SH degree zero, 1000 steps, and Stage-1 acquisition downscale 8. The downstream worker uses the **original RGB trainer**. C45–C46 cover the accepted comparison; it is not an end-to-end field-only quality demonstration. The [handoff](../ara/evidence/tables/20260907_bench019_final_handoff/CONTINUE_AT_HOME.md) reports favourable contained means but a failing seed and a substantial initialization-support confound.
- [field_loss.py](../src/rtgs/lift/field_loss.py) explicitly describes its analytic density/numerator loss as a proxy for normalized, truncated/faded fields. [field_refit.py](../src/rtgs/lift/field_refit.py) refreshes centre visibility and permits hard/soft source footprints; this is not the complete per-pixel alpha renderer.
- [compact_trainer.py](../src/rtgs/optim/compact_trainer.py) already compares queried teacher colour with a point-rasterized student and exposes a topology-controller interface. [compact_density.py](../src/rtgs/optim/compact_density.py) already implements the classic clone/split/prune adapter. Neither needs to be invented to investigate capacity.
- [splat_sfm.py](../src/rtgs/lift/splat_sfm.py) already triangulates centers and covariances, but its component matching assumes more cross-view identity than independently optimized mixtures guarantee. Existing failed learned-depth routes in StructSplat also rule out treating “use VGGT” as an explanation by itself.

Low capacity is a plausible contributor, not a demonstrated sole cause. [The Role of Initialization in 3DGS, v3](https://arxiv.org/html/2603.20714v3) finds important interactions: dense initialization can improve geometry and off-trajectory behavior without consistently improving ordinary novel-view scores. Both coverage and geometry need measurement.

## Functional problem signature

Inputs: calibrated views of a static scene, stored as finite continuous colour fields with explicitly defined support, background and colour space. Hidden state: shared geometry, extent, opacity and direction-dependent appearance. Constraints are local along rays and globally coupled through visibility. Unknown correspondences, partial coverage and compression error limit identification. Gauge freedoms include mixture decomposition and unobserved depth/appearance tradeoffs. Desired outputs are high-quality held-out RGB and coherent geometry; speed and compression are separate secondary questions.

## Anti-library

Do not infer depth by intersecting every colour-compatible tube; a repeated colour is not a correspondence. Do not copy image weights into material density or alpha. Do not demand one physical splat per independent image ellipse. Do not mistake lower whole-frame MSE for recovered detail. Do not claim that more primitives, a depth network, a new regularizer or exact ray integration alone resolves the failure. Keep poor-support regions visible in metrics instead of deleting them from the denominator.

## Proposed reconstruction architecture

**Measurement adapter.** Preserve exact queried colour, finite support/fade, affine terms, background, alpha policy, intrinsics, distortion and colour-space conventions. The initial path uses pixel centers and the same declared colour convention on both teachers. Additional antialiasing is an explicit operator, not silently different targets. Inspect high-frequency and boundary fidelity during Stage 1; select budgets from training-only criteria. Never use the normalized weight sum as physical density.

**Geometry.** First establish a strong conventional baseline by running a calibrated patch matcher or dense-correspondence method on decoded training fields. Enforce cheirality, parallax, reprojection and reciprocal consistency; distribute accepted support spatially. This is field-derived geometry, even if the matcher internally uses dense tensors. Compare with current FieldSweep and a deliberately uninformative initialization. Keep unsupported hypotheses separate; a filled-in midpoint is a fallback, not observed depth. Occluded views must not all vote against a valid surface. Repeated patterns and broad low-texture regions need abstention or an explicitly declared prior.

**3D primitives.** Use geometry-supported centers and local surface frames. A projected footprint can inform initial extent only when the patch has supported local geometry; do not turn depth uncertainty into a thick rendered surface. Allow multiple 3D primitives under one image ellipse, and many source ellipses to supervise one surface. Full covariance recovery requires adequate rank; a surface-constrained fallback is a model assumption. No single-view covariance is an unrestricted 3D shape measurement.

**Refinement.** Optimize one shared scene against all training fields with the actual RGB compositor. Start with existing CompactTrainer and classic density control; validate sampled-loss and screen-gradient normalization before transplanting dense thresholds. Permit births/splits/pruning and geometry movement; avoid forcing all final projections to preserve arbitrary source ellipses. Use conservative geometry/appearance stages and add higher-order SH only as a controlled factor. Diagnose blur via residual structure, footprint growth, depth support and uncovered pixels. Faster CUDA point queries are a systems follow-up after reference correctness and utility.

**Alternative if geometry is paramount.** A 2DGS/GOF-like surface representation is a useful comparator, but changing the output representation is an architectural choice. [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772) demonstrates a route to geometry-aware Gaussian reconstruction. It does not establish that arbitrary image-field lifting works. Preserve ordinary 3DGS output in the first experiment to reduce confounding.

## Recommended first experiment

**First gate: isolate information loss before inventing another inverse.** Use the same adequate 3D initialization, renderer, loss, topology schedule, cameras and training samples; change only the target between source RGB and decoded/queryable field RGB. In a separate fully field-derived arm, replace the initialization with geometry obtained from field queries. The common reference initialization may use source-image SfM solely as a labelled diagnostic; its success cannot establish the strict field-only claim.

| Arm | Initialization | Supervision | Question |
|---|---|---|---|
| A | Frozen strong reference | Training photographs | Can this calibrated setup reconstruct well at all? |
| B | Same reference | High-fidelity fitted fields | Does the representation preserve enough signal? |
| C | Same reference | Existing low-budget fields | How much of the gap is compression? |
| D | Geometry estimated only from high-fidelity fields | Same fields as B | Does the full field-only route work? |
| E, conditional | Existing FieldSweep | Same fields as B | Does the new geometry method explain D's change? |

A–C are the first screen. D/E follow only once B makes the information boundary credible. C is a control using old-budget **settings**, not permission to overwrite old fields or outcomes. Reuse a single production-quality budget and all-parameter settings established on the training/development baseline; do not keep 256 primitives as a quality ceiling. Freeze exact counts/caps, resolution, optimizer, split roles, seeds and thresholds in a fresh protocol before running. If source RGB cannot meet the quality target at the budget, the test does not reject the field representation.

For B, verify decoded pixel-center values against direct queries and compare deterministic student losses/gradients on the same pixel set. Start with the same renderer and loss as A. If the dense reference uses SSIM windows, either implement identical field-sampled windows or freeze a shared pointwise loss; changing both renderer and loss would destroy the isolation. A streaming full-pixel query baseline can establish correctness before sparse sampling efficiency. Teacher colour space and mask/background treatment must match.

**Suggested development criteria, not frozen approvals:** on identical held-out views, B within 0.5 dB foreground PSNR and 0.02 LPIPS of a visually satisfactory A, with no predefined thin-detail or coverage regression; D then approaches B without source-RGB access. These are provisional engineering margins, not a discovered law or statistical significance test. Use multiple independent captures before generalization; separate within-capture camera splits from capture-level replication. Predefine detail regions without inspecting candidate outcomes. Report full-canvas, foreground and boundary scores, fixed-denominator coverage, calibrated geometric error where truth exists, and representative orbits. Record total acquisition, matching, refinement, bytes and peak resources.

**Decision tree.** A fails: repair calibration/data/baseline first. A passes but B fails: inspect representation fidelity, target semantics and optimizer interaction. B passes but D fails: prioritize geometric inference. D fits training views but fails held-out views: investigate ambiguity/visibility/overfitting. D passes: only then measure any advantage of Gaussian-specific footprint processing or optimize speed. An unsuccessful large run without these controls would be another ambiguous outcome.

**Cheapest mathematical control.** Separately generate a known small Gaussian volume under (i) additive projections, (ii) emission–absorption with increasing opacity and (iii) independently refitted image mixtures. Test matched forward models, then deliberately use the additive inverse on opaque observations. This probes the tomography analogy. A shared-renderer round trip is a correctness diagnostic, not independent scene evidence; use a different forward discretization/model and calibrated captures for subsequent validation. This proposal does not execute a result-bearing experiment.

## Productive recombinations

All candidate claims below are untested. Novelty classes are conservative working labels. Each success means bounded evidence for the stated mechanism, partial success narrows the boundary, and informative failure rejects that mechanism in the frozen scope.

### Candidate: exact field supervision with adequate capacity

- Central claim: high-fidelity fields support reference-quality reconstruction with fixed common geometry initialization.
- Novelty class: N1. Known foundation: 3DGS and CompactTrainer. Irreducible delta: isolate teacher information from initializer effects; this is useful integration, not a novelty claim.
- New prediction: B approaches A as training-field approximation improves. Null hypothesis: a large unexplained gap remains.
- Cheapest killing test: A–C above. Prior-art threats: ordinary training on compressed/decoded images. Novelty confidence: low; utility experiment. Highest reachable evidence maturity: bounded calibrated development.

### Candidate: match queried patches rather than mixture components

- Central claim: geometric support improves when correspondence follows field appearance rather than component identity.
- Novelty class: N1/N2. Known foundation: EDGS/MVS. Irreducible delta: test invariance to exact mixture splitting; known components, limited relationship hypothesis.
- New prediction: identical decoded fields give matching geometry despite changed decompositions. Null hypothesis: no improvement over FieldSweep.
- Cheapest killing test: D/E with common refinement. Prior-art threats: compressed-image MVS, G²SR. Novelty confidence: low. Highest reachable evidence maturity: calibrated development.

### Candidate: surface-supported covariance initialization

- Central claim: local geometry separates rendered extent from depth uncertainty and reduces thick floaters.
- Novelty class: N1. Known foundation: 2DGS/GOF and existing surfel code. Irreducible delta: preserve measured geometry while varying covariance alone; standard attribution.
- New prediction: depth thickness improves without losing coverage. Null hypothesis: equal or worse geometry/appearance.
- Cheapest killing test: fixed centers/count covariance control. Prior-art threats: existing surfel_lift and covariance repair. Novelty confidence: low. Highest reachable evidence maturity: mechanism then calibrated.

### Candidate: spatial coverage with geometry confidence

- Central claim: accepted-match coverage predicts final utility better than selecting only the most confident points.
- Novelty class: N1. Known foundation: EDGS. Irreducible delta: field-specific coverage diagnosis; not a new allocation principle.
- New prediction: matched-count spatial balancing improves missing-region coverage. Null hypothesis: confidence-only selection is equally good.
- Cheapest killing test: identical accepted pool and count, two selectors. Prior-art threats: EDGS sampling. Novelty confidence: low. Highest reachable evidence maturity: bounded mechanism.

## Exploratory candidates

### Candidate: multimodal ray support with abstention

- Central claim: retaining several supported depths until visibility resolves ambiguity avoids wrong midpoint/argmin commitments.
- Novelty class: N2. Known foundation: cost volumes and existing ray-posterior negatives. Irreducible delta: explicit unknown support and delayed commitment, not just softmax temperature.
- New prediction: less unsupported geometry at equal coverage. Null hypothesis: uncertainty merely spreads floaters.
- Cheapest killing test: repeated-texture/occlusion control with known depth. Prior-art threats: probabilistic MVS. Novelty confidence: low. Highest reachable evidence maturity: mechanism.

### Candidate: ray-based rendering as a model-error control

- Central claim: ray integration improves large-footprint/perspective cases where projected alpha splats disagree with the declared forward model.
- Novelty class: N1. Known foundation: volumetric Gaussian ray rendering. Irreducible delta: isolate renderer error; not a reconstruction guarantee.
- New prediction: advantage increases with footprint and perspective nonlinearity. Null hypothesis: geometry/representation dominates.
- Cheapest killing test: shared geometry, opacity sweep, two renderers. Prior-art threats: RayGauss and Don't Splat Your Gaussians. Novelty confidence: low. Highest reachable evidence maturity: synthetic mechanism.

### Candidate: training-only visibility consistency

- Central claim: geometry checked through neighbouring training-view depth ordering reduces contradictory surfaces without imposing all-view colour agreement.
- Novelty class: N1/N2. Known foundation: MVS and rendering transmittance. Irreducible delta: separate occlusion from disagreement.
- New prediction: improved concavity reconstruction without excess carving. Null hypothesis: erroneous depth creates a self-reinforcing bias.
- Cheapest killing test: foreground occluder over textured background. Prior-art threats: occlusion-aware MVS. Novelty confidence: low. Highest reachable evidence maturity: mechanism.

### Candidate: appearance freedom after geometric support

- Central claim: delaying view-dependent colour prevents a wrong geometry from absorbing evidence through appearance parameters.
- Novelty class: N1. Known foundation: staged inverse rendering. Irreducible delta: geometry-versus-appearance attribution under identical final freedom.
- New prediction: held-out geometry improves even if early colour loss is higher. Null hypothesis: schedule changes only convergence speed.
- Cheapest killing test: specular and Lambertian pair with known geometry. Prior-art threats: staged SH schedules. Novelty confidence: low. Highest reachable evidence maturity: mechanism.

## Transformational candidates

These are proposed formulation changes, **not established N3 innovations**. The prior-art audit downgrades all to N2 until a distinctive prediction survives. Their value is changing the question rather than adding machinery.

### Candidate: field representation equivalence classes

- Central claim: reconstruction decisions should depend on observable field values, not a nonunique decomposition.
- Novelty class: N2, candidate N3 formulation. Known foundation: inverse-problem gauges and existing exact-split invariance. Irreducible delta: test the entire initialization/refinement pipeline, not only analytic loss.
- New prediction: exact recomponentization preserves geometry and update decisions. Null hypothesis: topology-specific information is essential.
- Cheapest killing test: split identical components with conserved coefficients. Prior-art threats: C37/C42 already cover narrower invariance. Novelty confidence: low. Highest reachable evidence maturity: CPU contract, then mechanism.

### Candidate: compression error measured through reconstruction sensitivity

- Central claim: equal image distortion can have different geometry effects because residuals project differently through the reconstruction Jacobian.
- Novelty class: N2-T, candidate N3 observable. Known foundation: task-oriented compression and adjoint sensitivity. Irreducible delta: compare J-transpose-weighted teacher error with ordinary fidelity at fixed geometry.
- New prediction: matched-PSNR encodings differ in geometric gradient bias. Null hypothesis: scalar image error explains the gap.
- Cheapest killing test: frozen-state Jacobian-vector products on edges versus flat texture. Prior-art threats: task-aware rate-distortion. Novelty confidence: low. Highest reachable evidence maturity: diagnostic.

### Candidate: reconstruct supported equivalence instead of arbitrary physical density

- Central claim: explicitly separating supported surface state from unobserved volume avoids interpreting photometric null-space motion as recovered density.
- Novelty class: N2-T, candidate N3 problem statement. Known foundation: identifiability and system identification. Irreducible delta: report observable geometry and null-space freedom separately.
- New prediction: large density variation can coexist with identical training images in blind regions. Null hypothesis: all relevant geometry is already constrained.
- Cheapest killing test: opaque shell with differing hidden interiors. Prior-art threats: inverse-rendering ambiguity literature. Novelty confidence: low. Highest reachable evidence maturity: mathematical counterexample.

### Candidate: shared latent surface with independent image decompositions

- Central claim: soft field-level supervision can infer shared surfaces while permitting arbitrary split/merge decompositions in each view.
- Novelty class: N2, candidate N3 grammar. Known foundation: existing field refit, differentiable rendering, latent geometry. Irreducible delta: geometry and visibility are shared, component bookkeeping is not.
- New prediction: consistency survives independent image refits, whereas forced component tracks fail. Null hypothesis: exact image objectives already exhaust the benefit.
- Cheapest killing test: shared rendered scene, independently refitted views, controlled decomposition changes. Prior-art threats: inverse rendering already uses this formulation broadly. Novelty confidence: low. Highest reachable evidence maturity: mechanism.

## Cross-domain transfers

Donor families considered: medical imaging, structural biology, astronomy, information theory, numerical analysis and control/system identification. The first three have established Gaussian projection precedent; the others are diagnostic transfers, not claims that their mature theory already solves this problem.

### Transfer: CT forward-model calibration

Donor field and mechanism: attenuation physics calibrates ray measurements. Recipient mapping: scene → shared kernels, detector → field queries, projector → RGB renderer, residual → observation loss. Preserved causal structure: invert the measurement process actually used. Broken correspondence: transmission becomes occluded coloured emission. Required invention: correct visibility/appearance operator. Adoption barrier: efficient differentiable integration. Native baseline: ordinary 3DGS. Recipient-specific prediction: additive inversion fails as opacity increases. Counter-analogy: a black opaque object is not zero density. Cheapest killing test: matched/mismatched forward-model sweep. Prior-art threats: R²-Gaussian and volumetric rendering. Novelty confidence: low.

### Transfer: cryo-EM independent-half validation

Donor field and mechanism: independently reconstructed data halves expose overfitting. Recipient mapping: images → separated camera/capture groups, model → common coordinates, agreement → spatial diagnostics, independent truth → held-out evidence. Preserved causal structure: independent information tests repeatability. Broken correspondence: small camera subsets may observe different surfaces; shared priors induce correlated error. Required invention: visibility-overlap masks and external checks. Adoption barrier: reduced angular support. Native baseline: held-out RGB. Recipient-specific prediction: unsupported details are unstable across halves. Counter-analogy: agreement is not accuracy. Cheapest killing test: disjoint well-conditioned view groups. Prior-art threats: existing probabilistic_pipeline. Novelty confidence: low.

### Transfer: astronomy restricted Gaussian deprojection

Donor field and mechanism: shape assumptions regularize a nonunique brightness inversion. Recipient mapping: intrinsic shape → local surface family, projected ellipse → footprint, inclination → camera geometry, fit → constrained covariance. Preserved causal structure: explicit priors reduce null spaces. Broken correspondence: opaque visibility, arbitrary textures and perspective. Required invention: locally valid surface models. Adoption barrier: bias on curved or thin detail. Native baseline: unconstrained covariance. Recipient-specific prediction: lower variance only when the prior is valid. Counter-analogy: a symmetry prior can manufacture geometry. Cheapest killing test: planar versus curved patches. Prior-art threats: MGE/surfel reconstruction. Novelty confidence: low.

### Transfer: task-oriented compression

Donor field and mechanism: preserve information relevant to an inference objective. Recipient mapping: source image → field, code rate → complete bytes, task → geometry inference, distortion → reconstruction sensitivity. Preserved causal structure: finite representation allocates limited information. Broken correspondence: the downstream geometry is initially unknown and view-coupled. Required invention: training-only sensitivity proxy. Adoption barrier: expensive/unstable feedback. Native baseline: equal-byte PSNR fitting. Recipient-specific prediction: equal PSNR does not imply equal depth quality. Counter-analogy: downstream-aware selection may overfit a development scene. Cheapest killing test: matched-distortion encodings. Prior-art threats: [task-oriented compression](https://arxiv.org/abs/2405.04144). Novelty confidence: low.

### Transfer: numerical-analysis operator consistency

Donor field and mechanism: consistent forward/adjoint operators distinguish solver defects from model mismatch. Recipient mapping: forward solver → compositor, data → field queries, adjoint → gradients, discretization → sampled pixels/footprints. Preserved causal structure: wrong gradients solve a different problem. Broken correspondence: visibility/topology introduce nonsmooth changes. Required invention: frozen-state parity and separately tested topology events. Adoption barrier: renderer discrepancies. Native baseline: exact full-pixel reference. Recipient-specific prediction: discrepancies remain at fixed parameters before training. Counter-analogy: same-renderer success can be an inverse crime. Cheapest killing test: pixel and gradient parity, then independent renderer. Prior-art threats: standard inverse-problem testing. Novelty confidence: low.

### Transfer: system-identification observability

Donor field and mechanism: measurement sensitivity reveals hidden-state directions that cannot be estimated. Recipient mapping: state → depth/covariance, sensor → cameras, sensitivity → projection Jacobian, excitation → angular coverage. Preserved causal structure: deficient measurement rank leaves ambiguity. Broken correspondence: correspondence and visibility also change with state. Required invention: local rank diagnostics with uncertainty separated from rendered extent. Adoption barrier: costly large-state analysis. Native baseline: parallax/reprojection gates. Recipient-specific prediction: additional well-separated views help where repeated nearby views do not. Counter-analogy: strong priors can hide rank deficiency. Cheapest killing test: controlled camera-baseline sweep. Prior-art threats: existing field_observability. Novelty confidence: low.

## New-evidence discovery programs

### Program: teacher-information versus inverse failure

What varies: A–C teacher fidelity at common initialization. What is measured: target/gradient agreement, held-out appearance, detail and coverage. Surprising outcome: near-identical teachers lead to stable large quality differences. Conventional explanation: compression removes geometric features or optimizer branches diverge. Negative control: identical teachers under two serialization routes. Leakage/bug exclusion: fixed pixel samples, held-out isolation and parity. Raw evidence: fields, hashes, target differences, histories, models and fixed-view renders. Abandonment rule: no geometry claim until the reference works and semantics agree.

### Program: projection physics versus image-mixture semantics

What varies: opacity, perspective, texture, Gaussian decomposition and camera angles. What is measured: projection residual, recovered geometry, conditional covariance and invariance. Surprising outcome: additive inversion remains accurate under deliberately opaque images. Conventional explanation: easy scenes or strong priors mask mismatch. Negative control: hidden interiors with identical images. Leakage/bug exclusion: independent forward discretization, multiple instances and no ground-truth initialization except labelled controls. Raw evidence: generator, measurement definitions and complete outcomes. Abandonment rule: retain CT only as a donor if its projection assumption fails.

### Program: geometry versus allocation interaction

What varies: supported geometry versus current lift at matched initial count, with fixed and adaptive final topology. What is measured: error-to-time curves, coverage, thickness and birth lineage. Surprising outcome: worse starting geometry consistently reaches better final geometry. Conventional explanation: a controller destroys useful initialization or broadens footprints. Negative control: duplicated points with no new spatial information. Leakage/bug exclusion: same teachers, schedules and final scoring. Raw evidence: parameter/optimizer bindings, topology events, intermediate models. Abandonment rule: do not promote an initializer from step-zero metrics alone.

## Pareto set

Scores 0–5 are judgement, not measurements. Higher first-test-cost score means cheaper. No aggregate rank is implied.

| Candidate | Novelty | Falsifiability | Explanatory value | Importance | Feasibility | First-test cost | Interpretability | Baselines | Negative-result value | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher-information isolation | 1 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 2 |
| Field-query geometry | 1 | 5 | 4 | 5 | 4 | 3 | 4 | 5 | 4 | 2 |
| Covariance/surface support | 1 | 5 | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 2 |
| Representation invariance | 2 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 3 |
| Sensitivity-aware compression | 2 | 4 | 5 | 4 | 3 | 3 | 4 | 4 | 5 | 3 |
| Ray-based renderer control | 1 | 5 | 4 | 3 | 2 | 2 | 4 | 4 | 4 | 2 |

Fastest kill: representation invariance and teacher parity. Most useful engineering test: A–C. Strongest theory question: observability and representation invariance. Systems direction: efficient exact teacher sampling after parity. Higher-risk direction: sensitivity-aware compression. Most useful negative result: locating the failure between teacher information and geometric inference.

## Claude dialogue and adjudication

The [initial public critique](research/2026-09-07-field-reconstruction/claude-public-critique.md), [Codex follow-up question](research/2026-09-07-field-reconstruction/claude-followup-question.txt) and [Claude revision](research/2026-09-07-field-reconstruction/claude-public-followup.md) are preserved verbatim. They are discussion evidence, not automatically validated sources. Both requests completed without permission denials. Main answers used Fable 5.1; the CLI also reports auxiliary Haiku usage. Claude received public mathematics and literature only; the repository inspection was Codex's work.

Agreements after critique: CT and ordinary RGB need different forward models; independent fitted components need not correspond to physical primitives; exact field values can replace pixel targets under the same solver; covariance recovery must use actual numerical rank, not a requirement for non-coplanar camera axes; high opacity is not a justified universal initialization rule. Claude withdrew its unsupported greater-than-90-percent matching prediction and its assertion that an RGB image GMM was not a projection of anything.

Codex's final corrections to the revised response:

- **No geometry-error bound follows from small target error.** The E0 sentence suggesting bounded geometry differences is not adopted. Equality of targets gives equality of fixed-state losses and gradients; perturbations, ill-conditioning and topology changes can lead to very different solutions. The report's triangle inequality applies to image residuals only.
- **Finite-mixture identifiability is not the same as repeatable image fitting.** An ideal minimal mixture with distinct components may be identifiable; exact duplicate splitting creates a nonminimal representation, and approximate finite-pixel fits, normalized blending and independently partitioned patches have different assumptions. No blanket theorem of nonidentifiability is claimed here.
- **Analytic Gaussian overlap is a restricted tool.** Complete squared-distance evaluation includes within-mixture terms as well as cross terms, and typically costs quadratic pair work without additional structure. A generic exact 3DGS objective with clipping, sorting, finite support, normalized teachers or SSIM is not reduced to a cheap pairwise Gaussian formula. The exponential expansion discussed by Claude applies only under idealized unclipped Gaussian alpha terms and a fixed order; it is not the implemented renderer.
- **Footprint covariance is conditional evidence.** Neither a covariance residual nor a learned confidence alone establishes or invalidates a physical correspondence. Partial occlusion and independent fitting can change measured footprint shapes. Use appearance/geometry controls and report coverage.

These corrections do not require another run. The principal recommendation remains the A–C information-isolation screen, followed by D/E if the representation passes, with fresh protocol review before result-bearing execution.

## Audit limitations

No new reconstruction was run. Existing results are quoted only within C42–C46 and their original audits. Suggested tolerances, budgets and pipeline changes are hypotheses awaiting a fresh experiment contract and distinct prospective review. High-quality novel views are not a proof of physical volumetric density. Compression error bounds at training pixels are not bounds on unknown 3D geometry or unseen views.

The triangle inequality gives a useful limited statement: for the same sampled pixels, `||R-I|| <= ||R-F|| + ||F-I||`. For half squared-error loss at a fixed differentiable renderer state, changing target I to F changes the gradient by `J^T(I-F)`. These identities motivate parity and sensitivity checks; neither guarantees optimization convergence, stable topology or generalization.

The public-literature collaborator critiqued the mathematical proposal but has not reviewed private repository code or outcome artifacts. Codex's repository synthesis remains self-reviewed. The user authorized collaboration, but automatic approval review rejected an internal-context payload; the completed collaboration route is restricted to public text and web sources. No images, models, datasets or private repository content are supplied to it.
