# Continuation after the field-target information screen

Date: 2026-09-09. RTGS-022 is a discussion of the existing proposal, not a new experiment
protocol, a novelty portfolio, or an approved implementation plan.

**Recommendation: first measure how target errors change the fitting signal at fixed
3D states. Then use controlled parameter and topology interventions to locate the
failure. Retain a separate positive tomography control.** Revisit field-derived
geometry only after an adequate photograph reference and its paired field test pass.

## What the completed experiment establishes

[C47](../ara/logic/claims.md#c47-dense-fields-improve-the-low-budget-result-but-miss-the-frozen-reconstruction-quality-prerequisites)
and the [independent RTGS-021 audit](../benchmarks/results/20260908_field_teacher_information_stage_frame00008_AUDIT.md)
remain the authority. Dense fields improved on the low-budget field family but failed
paired boundary, perceptual and silhouette margins. The photograph reference also
failed its foreground adequacy threshold. The excess-opacity diagnostic did not
identify a cause. The source/target sampling parity and equal-target mechanics tests
already passed; repeating those alone would add little evidence.

All arms shared RGB/mask-derived initialization, and fitting used a black background
without alpha supervision. Those facts make target-error, opacity/colour ambiguity,
initial geometry, and optimization sensitivity plausible interacting explanations.
None has been demonstrated as the explanation. The old D/E prerequisite remains failed;
a new discussion cannot relabel it as passed.

## First diagnostic: target residuals and their effect on gradients

Use training views only and the exact saved target operator. Compare photograph and
field targets at a predeclared set of common student states: the shared initial state
and, as explicitly exposed development diagnostics, the saved final photo and field
states. Do not choose a state because it gives an attractive discrepancy. Later runs
would need prospectively selected intermediate checkpoints; these were not saved by
the completed final-only protocol.

Measure teacher-minus-photo residuals separately in foreground interior, boundary
bands and exterior. Keep denominators fixed and report signed colour errors, local
contrast and the actual SSIM window support. Masks can define diagnostic strata
without becoming fitting inputs. These are proposed measurements, not findings from
this discussion.

For each common state, compare the exact L1 and SSIM objectives and gradients by
position, scale, rotation, opacity and SH groups. Report absolute and relative gradient
differences plus directions; an angle is undefined when a gradient is effectively
zero. Fix samples, renderer branches and stochastic state. If comparing optimizer
updates, also clone optimizer moments and scheduler state: equal gradients do not
imply equal Adam updates from different moments.

Keep an identity-target path as a negative control. Exact equality at every site read
by the same loss implies equal fixed-state loss and gradients by substitution. Small
target errors do not provide a uniform bound on a nonlinear training trajectory.
The new question is whether the *actual* approximation errors produce a concentrated
change in the fitting signal, beyond the already established exact-target parity.

Interpret this as evidence updating competing explanations. Boundary residuals and
opacity-gradient changes would motivate a boundary intervention; they would not prove
that geometry or visibility is irrelevant. A null gradient difference at one state
does not rule out later divergence. Stop and resolve a failed identity control before
interpreting the other measurements.

## Conditional interventions

Choose the smallest intervention supported by the diagnostic rather than launching
every combination immediately. Each executed experiment needs its own frozen inputs,
budget, controls and stopping rules.

| Question | Controlled comparison | What it can establish | What remains confounded |
|---|---|---|---|
| Does existing geometry trade colour against opacity? | At common frozen centers/covariances, compare colour-only, opacity-only and joint fits for both targets | Sensitivity to the permitted parameter groups | The imposed geometry and constraints can be wrong; joint fitting does not isolate opacity |
| Does adaptive topology amplify the field penalty? | Photograph/field targets crossed with fixed/adaptive topology; paired initial states and stochastic schedules | Difference in the field penalty between topology policies | Fixed topology still permits geometry motion; topology changes also change model capacity |
| Is a particular target region implicated? | Paired diagnostic targets replacing the same predefined boundary or exterior region, including an identity/sham control | Effect of that explicit target intervention | Replacing field values with source RGB adds oracle information and does not produce a field-only method |
| Does extra silhouette information help? | Identical photo/field arms with and without a training-mask constraint | Value of this added measurement under the specified optimizer | A mask is additional input, and its use is neither a guaranteed upper bound nor proof of the original cause |

No hard containment clip, opacity deletion or changed scoring mask may hide errors in
the original comparison. A successful oracle intervention can justify a new question
about obtaining that information from the allowed inputs; it cannot itself close the
strict field-only question.

Each paired parameter-group refit starts from identical values for every group,
including groups held fixed, with identical optimizer state. Otherwise an opacity-only
effect could inherit target-specific colours. Freeze the exact replacement and
compositing rule for a region intervention and measure seam effects: a hard splice
between photograph and field RGB can create an additional boundary of its own.

For a simple isolated layer on black, the compositing equation is
`C = alpha * colour`. Black material can therefore remain opaque while producing
black RGB. This is a conditional example of an ambiguity, not a diagnosis of the
saved models. Inspecting opacity and rendering the *same fixed model* over a different
display background can reveal such material. It does not supply a new ground-truth
training target. Re-compositing source RGB over arbitrary backgrounds requires
additional alpha/matting information. The existing alpha-IoU metric remains a valid
declared coverage test, even though it is not physical density or complete geometry
ground truth. The emission/absorption rendering context is described in
[NeRF](https://www.matthewtancik.com/nerf).

## Establish a stronger reference before claiming field-only reconstruction

Develop the photograph baseline on explicitly exposed development data. Check
calibration conventions and coverage, shared visual-hull limitations, sampling, and a
conventional geometry initializer. Do not simply lower the old quality threshold or
interpret more iterations as a demonstrated remedy. The existing audit already notes
image-edge masks and possible loss of geometry under the all-view in-frame hull rule.

Freeze the chosen recipe before a fresh confirmation capture or untouched evaluation
split. A new protocol version does not make previously inspected data untouched. Test
the field targets with the same recipe. Only if both prerequisites pass should the
next comparison change the source of initialization.

For that eventual comparison, a strong conventional control is calibrated multiview
matching on decoded field patches versus the same matcher on photographs. Hold
calibration, matching configuration, output budget and downstream refinement fixed;
label the photograph arm as an information-rich reference. COLMAP documents
[reconstruction from known poses](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses),
so recalibrating the scene is not intrinsically required. Querying or decoding patches
from fields can be field-only at runtime. A pretrained matcher or depth network is a
declared learned prior; it does not by itself violate that input boundary, nor does it
prove recovery from measurements alone. Original images, masks, image-derived bounds,
point clouds and scene-specific depth must not enter a strict field-only arm through
initialization or caches. Supplied calibration is an explicit allowed input.

Independent image-mixture components need not correspond to physical 3D primitives.
Do not revive previously closed component-matching/lifting variants under a new name
or infer correspondence from a general finite-mixture identifiability theorem. A
footprint-specific method would require a distinct hypothesis and a fresh comparison
against the conventional field-patch baseline.

## Give tomography its own positive control

The analogy is worth testing with the correct measurements. CT log transmission is a
line integral of attenuation, while ordinary RGB rendering also involves visibility
and colour. [R2-Gaussian](https://arxiv.org/html/2405.20693v2) demonstrates why the
Gaussian projection normalization matters for tomography. It does not establish that
opaque RGB colour fields are attenuation projections.

Propose a small synthetic volume with known nonnegative, mass-normalized 3D Gaussian
components. For orthonormal parallel-ray coordinates, a component of mass `m` projects
to a normalized 2D Gaussian of mass `m`, mean `P mu` and covariance `P Sigma P^T`.
An arbitrary peak-amplitude coefficient is not that mass. Compare the reconstructed
density and held-out projections, not the identities of independently refitted 2D
components. Check detector coverage and numerical truncation explicitly.

Use an independent numerical forward discretization and a conventional voxel inverse
as controls. Include a known-count well-conditioned case before testing unknown count,
noise or restricted angles. Evaluate several starts; sufficient measurements do not
guarantee a nonconvex optimizer finds the solution. Predetermine acceptable density,
mass and held-out projection errors after numerical feasibility work. A fitting loss
alone cannot declare success.

Then vary the observation model separately. Attenuation-only transmission with known
incident intensity can be log-linearized; emission/absorption RGB generally introduces
unknown colour as well. Model mismatch need not fail on every favourable phantom.
The experiment measures where the operators agree or differ; it must not be designed
to guarantee a dramatic failure. Perspective/cone-beam data requires its own ray
integral model; exact Gaussian projection on a plane is not generally preserved by
perspective division.

A positive result would validate a limited density-reconstruction mechanism. It would
not establish high-quality human reconstruction from ordinary photographs. A later
renderer comparison could examine
[Volumetrically Consistent 3D Gaussian Rasterization](https://openaccess.thecvf.com/content/CVPR2025/html/Talegaonkar_Volumetrically_Consistent_3D_Gaussian_Rasterization_CVPR_2025_paper.html),
whose analytic transmittance still uses overlap/sorting assumptions. Likewise,
[Mip-Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)
motivates a sampling/filter control. Neither paper is evidence that changing the
renderer fixes the current failure, and changing renderer and teacher together would
weaken attribution.

## Discussion provenance and unresolved points

Both public rounds completed successfully. The [initial critique](research/2026-09-09-field-continuation/claude-public-critique.md),
[Codex counterarguments](research/2026-09-09-field-continuation/claude-public-followup-prompt.txt),
[Claude revision](research/2026-09-09-field-continuation/claude-public-followup.md) and
[collaboration receipt](research/2026-09-09-field-continuation/collaboration-receipt.json)
are preserved. The CLI reported Claude Opus 5 for the answers and auxiliary Haiku
usage, with no permission denials. Claude received general public methods only; the
application to RTGS-021 above is Codex's synthesis from local evidence. This is not
Claude's independent audit of the private runs.

Claude withdrew the unbounded-kernel causal assertion, the mask-oracle upper-bound
claim, the photo-only topology comparison, an irrelevant statistical mixture-rate
argument, and a proposed two-revision retry rule. It accepted paired topology controls,
separate colour/opacity interventions, the declared-prior input distinction and fresh
confirmation data. Those are useful corrections, not evidence that a proposed repair
works.

Three qualifications remain after the second response. First, optimizer moments are
needed to compare optimizer updates; ordinary fixed-state loss/gradients do not depend
on Adam buffers. Second, dark opaque material might occlude a lit surface in an
appropriate observed view, but there may be no such camera. Coloured leakage need not
vanish with viewpoint either. A targeted held-out-view search would be exposed
development analysis, not untouched confirmation. Such a probe cannot uniquely separate
the causes. Third, a deliberately wrong observation model need not fail on every
phantom: the tomography experiment must report favourable exceptions and cannot demand
failure merely to support the analogy. An analytically invalid normalization can serve
as a specific implementation negative control, with its expected discrepancy checked
independently.

The recommendation is conditional and untested. No new reconstruction was executed,
no old gate was relaxed, and no production default was changed. The focused literature
check ended on 2026-09-09 and covered R2-Gaussian, volumetric Gaussian rendering,
aliasing and calibrated matching through author papers, project pages and official
documentation. It is not an exhaustive prior-art search or a novelty claim.
