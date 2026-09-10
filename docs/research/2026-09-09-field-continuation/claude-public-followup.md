## Agreement (adopt as stated)

**1 (partly), 2, 3, 4, 5, 6, 7, 8.** Concretely:

- **Kernel support.** Implementations truncate and cull, so "unbounded tails" is not a mechanism I can assert. What survives is weaker and sufficient: within the retained radius there is still support across the silhouette. A halo has no unique 3D explanation — geometry, SH, footprint and opacity fit it jointly.
- **Mask oracle.** Correct that it is not an upper bound: adding information to a nonconvex optimisation is not monotone in outcome. Relabel it a *causal diagnostic with declared extra information*.
- **Topology.** My photo-only seed band was unpaired and therefore uninformative about the field. Replace with a 2×2: {photo, field} × {frozen, adaptive}, reporting the field penalty and the interaction term. Fixed-topology full optimisation still moves geometry; I will not call it "geometry held constant."
- **Frozen matching.** Exactness must include Adam moment buffers, RNG stream, and renderer branch selection. Two predeclared states (common initialisation, plus one neutral mid-training state from a third run), gradient *difference* reported as cosine and norm per parameter group.
- **Tomography.** My control was asymmetric in three ways you name. Rebuilt below.
- **Priors.** A pretrained matcher whose only scene-specific input is decoded fields *is* field-only at runtime, with a declared learned prior. Input boundary and prior-free inference are separate axes; my blanket prohibition conflated them.
- **Confirm split.** A protocol revision does not restore an untouched split. Either budget several confirm captures up front or accept that one is spent permanently.
- **Vol3DGS.** Yes — a later explicit renderer control, not a fix for RGB→density ambiguity, which is a property of the data (the $C=\alpha c$ product against a fixed background), not of integration accuracy.

## Withdrawals and narrowing

- **Withdrawn: the Heinrich–Kahn transfer.** An iid over-specified-mixture minimax rate says nothing about cross-view correspondence of independent image fits. Only the narrow remark survives: exact finite-mixture identifiability of a *density* does not give stable component correspondence between two approximate fits ([Yakowitz & Spragins](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-39/issue-1/On-the-Identifiability-of-Finite-Mixtures/10.1214/aoms/1177698520.full)).
- **Withdrawn: "nearly free."** The audit costs a matched-state harness, which is the expensive part.
- **Withdrawn: "the photo residual is ~zero by construction"** at a converged state. It is small, not zero, and that error term is exactly what the audit must report rather than assume.
- **Withdrawn: the two-revision retry rule** for footprint lifting. A rejected method is rejected; re-entry requires a new prediction and a fresh split, not a quota.
- **Narrowed: E3.** Split into colour-only, opacity-only, and joint arms. Joint refit is not an opacity intervention.
- **Narrowed: the decision tree.** Replaced by conditional updates: each result shifts weight across {operator bug, teacher 2D deficit, trajectory/topology sensitivity, appearance–compositing gauge, geometry/visibility ambiguity}. No branch excludes another; interacting mechanisms are the default expectation.

## Remaining disagreement

**On silhouette metrics (2).** I did not discard them. I separated *scoring-only* masks — legitimate, and I agree they can be scored against a predefined coverage target — from *mask supervision inside the loss*, which changes the task. That distinction stands; only the "upper bound" framing was wrong.

**On the site-exactness lemma (4).** Your point strengthens its precondition rather than refuting it. If teacher and photograph agree at every site the loss reads, under an identical operator *including* optimizer state, RNG realization and renderer branch, then loss and gradient coincide at that state by substitution. I never claimed this bounds the trajectory — I explicitly stated the converse fails — and I retain that asymmetry.

**On SSIM (1).** The contrast and structure sub-terms depend on local variance and covariance over the 11×11 window. A variance-deficient teacher therefore perturbs the gradient even where mean RGB matches. That is arithmetic about the loss ([Kerbl et al.](https://arxiv.org/abs/2308.04079), $\lambda=0.2$), not an optics claim, and it does not depend on kernel support.

**On the black-background diagnostic (2).** You are right that $c\!=\!0$ can hide opaque material and that I cannot re-composite a photograph onto a new background without alpha. The discriminating test is a *model* probe, not a data probe: material with high accumulated $\alpha$ and near-zero emitted colour is invisible against black but must **occlude** in any view where it lies in front of lit object surface. So render the frozen student from viewpoints where the suspect region projects onto the object rather than onto background, and compare against the held-out photograph there. Genuine coloured leakage tracks the silhouette and vanishes with viewpoint; black opaque mass darkens object interior. No hidden alpha required.

**On an honest positive control (5).** Judged by preregistered pass thresholds, declared before running, on **both** held-out projections and the recovered density evaluated under an *independent* forward discretization; a matched negative control that must fail; component count treated as unknown; finite field of view declared as an ambiguity, not silently excluded; and reported sensitivity across initialisations. Success means "this parameterisation and optimiser can invert *this* operator under *these* conditions" — never a statement about photographs.

## Ordered plan

1. **Freeze and publish the operator**: colour space, resampling, premultiplication, mask alignment, renderer branch, RNG policy, Adam state handling. Predeclare the two comparison states and the confirm-capture budget.
2. **Pipeline-identity control.** Push the photograph through the teacher I/O path with the fit replaced by identity. Non-null result halts everything else; it is an operator bug.
3. **2D teacher-error localisation.** Teacher−photo maps stratified by silhouette distance and frequency band, plus local-variance ratio. No 3D optimisation.
4. **Frozen-state audit** at both predeclared states: $\Delta L$ split into L1 and SSIM sub-terms; gradient difference by parameter group (cosine and norm). Report the nonzero photo residual explicitly.
5. **Appearance refits**, three arms: colour-only, opacity-only, joint — each against both targets.
6. **Black-background probe** as described: $\alpha$/colour maps plus occlusion-revealing held-out views.
7. **2×2 topology factorial** with seeds; report field penalty, frozen/adaptive main effects, and interaction.
8. **Tomography control**, rebuilt: mass-normalised parallel-ray projections, unknown component count, independent discretization, preregistered thresholds, matched negative control. Note the disanalogy in writing — log-transmission linearises attenuation data; opaque RGB adds unknown colour.
9. **Comparator first**: conventional calibrated matching on decoded fields, once teacher quality is shown adequate. Footprint lifting is measured against it, with priors declared and a prior-free arm alongside.
10. **Renderer control last**: swap the rasteriser ([Vol3DGS](https://arxiv.org/abs/2412.03378), [StopThePop](https://arxiv.org/abs/2402.00525)) to test whether boundary error is teacher- or rasteriser-side. Vol3DGS's abstract states analytic transmittance and out-of-the-box tomography; I have not verified its overlap/sorting assumptions from the abstract, so treat that as open until read.
11. **Confirm split touched once**, results reported whether or not they help. Prior failures stay failures.