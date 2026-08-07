# Research portfolio: compact 2D Gaussian fields as 3D reconstruction evidence

**Literature/search cutoff:** 2026-08-04  
**Sources searched:** arXiv, CVF Open Access, ECVA, ACM/TOG and publisher pages, official project
pages/repositories, and Scholar Inbox discovery followed by primary-source verification; query
families covering 2D image Gaussians, multiview Gaussian/ellipse lifting, Gaussian surfels, dense
correspondence initialization, feed-forward 3DGS, masked object 3DGS, visual hulls, covariance and
random tomography, astronomy MGE deprojection, cryo-EM GMMs, mixture transport/registration/
reduction, nonlinear uncertainty propagation, densification, validation, and time-to-quality  
**Companion synthesis:**
[`LITERATURE_REVIEW_2D_TO_3D_GAUSSIANS.md`](LITERATURE_REVIEW_2D_TO_3D_GAUSSIANS.md)

Novelty classes are used as working labels, not publication claims:

- **N0:** reproduction or protocol repair;
- **N1:** known mechanism transferred into this exact pipeline;
- **N2:** nontrivial recombination with a new causal prediction;
- **N3:** new representation or objective that changes the method family;
- **N4:** new problem grammar or capability;
- **-T:** cross-domain transfer whose causal correspondence still needs validation.

## Frontier map

| Method/component | Primitive | Assumption | Mechanism | Evidence | Failure/open question |
| --- | --- | --- | --- | --- | --- |
| [G²SR](https://arxiv.org/abs/2607.14470) | Predicted image-plane splat → thin metric 3D surfel | 2–3 posed RGB views and trackable flow | Five-sigma-point tracking, triangulation, affine normal/scale initialization, Hellinger Gauss–Newton fit | Direct; about 69–91 reconstructions/s in detailed tables and 115–203 MB in paper protocol | Low rendering coverage/LPIPS; not arbitrary pre-fitted fields; source RGB still used |
| [GPS-Gaussian](https://arxiv.org/abs/2312.02155) | Dense 2D Gaussian parameter maps → pixel-aligned 3DGS | Two rectified foreground-matted human views and a scan-trained depth prior | Binocular depth unprojection plus pixel-wise scale/rotation/opacity regression | CVPR 2024 adjacent masked feed-forward capability; authors report >25 FPS at 2K | Not sparse fitted fields; accurate matting and ground-truth training depth required; human-specific prior |
| Astronomy [MGE](https://arxiv.org/abs/astro-ph/0201430) | Fitted 2D Gaussian brightness mixture → analytic 3D Gaussian luminosity mixture | Assumed inclination and axisymmetric/triaxial intrinsic family | Gaussian PSF closure and analytic line-of-sight deprojection | Exact restricted prior art and cheap baseline | Additive distant-view brightness, often concentric components, non-unique deprojection, no visibility/alpha |
| [Random tomography](https://arxiv.org/abs/0909.0349) / [sparse continuation](https://arxiv.org/abs/1202.6475) | 2D mixture projections → 3D radial mixture center shape | Many random additive profiles; basic method uses distinguishable weights | Profile deconvolution, component labeling, projected Gram-matrix averaging | Direct statistical mixture-deprojection prior art | Radial/common kernels, no perspective/occlusion/color, global pose only up to orthogonal transform |
| Cryo-EM [e2gmm](https://www.nature.com/articles/s41592-021-01220-5) | 3D GMM ↔ known-orientation 2D particle images | Many raw additive projections and molecular density prior | Differentiable mixed-dimensional GMM optimization with latent variability | Peer-reviewed GMM projection/reconstruction precedent | No per-view fitted mixture, few-view visibility, alpha, or ordinary radiance |
| CT [R²-Gaussian](https://arxiv.org/abs/2405.20693) / [FaCT-GS](https://arxiv.org/abs/2604.01844) | Raw 2D detector projections → radiative 3D Gaussians | X-ray line-integral physics | Amplitude rectification, fast differentiable rasterization/voxelization, coarse-volume warm start | Strong forward-operator and systems evidence in tomography protocols | Additive measurements rather than alpha composition; runtime/quality do not transfer directly |
| Robertini et al. [surface-detail capture](https://pmc.ncbi.nlm.nih.gov/articles/PMC6979538/) | Image Gaussian ↔ coarse-mesh surface Gaussian | Calibrated synchronized views and tracked coarse topology | Gaussian overlap/color association; normal-only mesh displacement | Direct conceptual precursor for soft association | Mesh supplies depth, normal, topology, and temporal identity |
| [GaussianImage](https://arxiv.org/abs/2403.08551), [Image-GS](https://arxiv.org/abs/2407.01866) | Independently fitted additive/normalized image field | Each view optimized for image fidelity | Continuous Gaussian image representation and adaptive allocation | Strong Stage-1 fidelity/speed | No cross-view identity or downstream geometry evidence |
| [EDGS](https://arxiv.org/abs/2504.13204) | Dense matched image point → 3D Gaussian center | Many posed RGB views and RoMa matching | Dense triangulation, SH init, no densification | Direct evidence that dense initialization accelerates convergence | Points do not carry fitted footprint covariance; initialization costs time/VRAM |
| [Initialization × densification study](https://arxiv.org/abs/2603.20714) | Alternative 3D point starts | Posed images, multiple controllers | Six initializers crossed with five density controls | Strong interaction warning | Does a field-derived covariance survive or help any controller? |
| [G3Splat](https://arxiv.org/abs/2512.17547) | Pixel-predicted 3D Gaussian | Learned sparse-view prior | Ray and local-surface geometric priors | Shows photometric shape alone can be non-geometric | How to constrain field-derived covariance without losing appearance? |
| [Off The Grid](https://openaccess.thecvf.com/content/CVPR2026/html/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.html), [ATSplat](https://arxiv.org/abs/2607.20417) | Adaptive detected/token 3D primitives | Large trained geometry model and RGB features | Entropy/uncertainty-directed sparse allocation | Compact, fast feed-forward frontier | Can teacher-field uncertainty replace learned RGB uncertainty? |
| [GaussianObject](https://arxiv.org/abs/2402.10259) | Visual-hull-seeded 3D Gaussians | Calibrated RGB and masks | Intersect mask cones, then optimize and repair | Strong masked object initialization | Concavity and mask brittleness; long repair phase |
| [Probabilistic object reconstruction](https://arxiv.org/abs/2603.14316) | Foreground-probability 3D surfels | RGB, detector/SAM masks, SfM | Soft support, view filtering, probability pruning, self-consistent masks | Strong within-paper mask ablation | “2DGS” is already a 3D surfel; not field lifting |
| [Contour-aware image fields](https://arxiv.org/abs/2512.23255) | Region-constrained 2D Gaussian | Segmentation masks | Prevent cross-boundary blending after warm-up | Better low-budget boundaries | Does boundary respect improve track survival and 3D geometry? |
| [QuerySplat](https://arxiv.org/abs/2608.01186) | Geometry and appearance queries | Large pretrained geometry model; unposed RGB | Separate geometry and appearance branches | Very recent high-quality preprint evidence | Whether the separation transfers to compact fitted fields |

## Functional problem signature

The target system receives calibrated or calibratable views represented by compact 2D Gaussian
fields and must construct a 3D Gaussian radiance field. The desired output sits on at least six
axes: metric geometry, novel-view appearance, coverage, primitive count, total latency, and peak
memory. Improving one axis can harm another.

The causal bottlenecks are:

1. **non-canonical mixtures:** component order, splitting, merging, and amplitude/color gauges make
   independent view fields incomparable by index;
2. **projective null spaces:** one view leaves ray depth; two projected covariances do not identify
   unrestricted 3D covariance;
3. **semantic mismatch:** additive image amplitude is not alpha opacity, and image coverage
   covariance is not necessarily geometric uncertainty;
4. **visibility:** a component may cover a color region that is occluded, view dependent, or absent
   in another view;
5. **allocation:** rejecting unreliable tracks improves precision while lowering coverage;
6. **optimizer interaction:** conventional density control can overwrite a good initializer;
7. **mask asymmetry:** masks cheaply constrain support but are noisy and contain no concavity/depth
   evidence;
8. **evaluation leakage:** valid-pixel metrics can reward missing difficult regions, and final-only
   metrics hide convergence and preprocessing cost.
9. **forward-operator drift:** projected probability mass, additive image intensity, X-ray line
   integral, and alpha opacity have different normalization and visibility semantics.

The key falsifiable question is whether fitted field *shape* and *compact allocation* contribute
causally beyond centers, pixels, depth, and dense correspondence. If not, a point/depth initializer
is simpler and better supported.

## Anti-library

The portfolio explicitly rejects these weak idea patterns:

- **“Project each 2D Gaussian back along a ray.”** This creates a tube or arbitrary depth prior,
  not reconstructed geometry.
- **“Two views give six covariance equations for six unknowns.”** The restrictions share a
  duplicated line constraint; the generic system remains rank deficient without a prior.
- **“Copy 2D opacity/weight into 3D alpha.”** The source and target renderers have different
  compositing semantics.
- **“High Stage-1 PSNR means a better 3D lift.”** Image residual does not measure component
  repeatability, geometric conditioning, or boundary correctness.
- **“Hard-intersect every mask.”** A single false negative can remove true geometry; the resulting
  visual hull cannot express concavity.
- **“Run normal densification and compare final quality.”** A strong controller may erase the
  initializer and conceal both benefit and harm.
- **“Beat a valid-pixel PSNR table.”** Coverage and all-pixel error must be included.
- **“Combine the fastest components and inherit every speed claim.”** Preprocessing, model loading,
  hardware, code path, and interactions determine end-to-end performance.
- **“Rename 3D-embedded 2DGS surfels as image-plane Gaussians.”** They are different inputs and
  solve different inverse problems.
- **“Treat single-view completion as reconstruction.”** Unseen content is generated from priors.
- **“Claim Gaussian-mixture deprojection as new.”** MGE, random tomography, and cryo-EM GMM work
  already cover restricted additive forms; novelty must be in the few-view perspective radiance
  problem and its measured system advantage.
- **“Transfer CT's amplitude formula into alpha rendering.”** The useful lesson is to derive the
  recipient forward operator, not to reuse a correction from a different measurement model.

## Productive recombinations

### Candidate: correspondence-conditioned teacher fields

- **Central claim:** A 2D field optimized jointly for source reconstruction and cross-view
  repeatability will yield more valid analytic 3D tracks and better quality-time AUC than an
  independently fitted maximal-PSNR field at matched count and similar source fidelity.
- **Novelty class:** N2.
- **Known foundation:** GaussianImage/Image-GS field fitting, G²SR's randomized-order detector,
  contour-aware region constraints, and affine/sigma-point tracking.
- **Irreducible delta:** The field itself becomes a multiview observation interface with explicit
  track-consistency and boundary losses; it is not merely fitted per image and matched afterward.
- **Why this is not merely A+B:** The allocation/split decisions must be coordinated across views,
  changing which primitives exist and making Stage-1 PSNR subordinate to track survival.
- **New prediction:** At equal component count, jointly conditioned fields will lower source PSNR
  slightly but increase three-view track survival, reduce triangulation condition number, and
  improve held-out coverage and geometry before 3D refinement.
- **Cheapest killing test:** Use exact synthetic flow/poses to train or optimize matched two-view
  fields under a small repeatability term; compare with independent fits over three seeds and a
  fixed analytic lift.
- **Null hypothesis:** After controlling count and source PSNR, repeatability supervision does not
  improve accepted-track coverage, geometry, or convergence AUC.
- **Prior-art threats:** G²SR may already cover the detector-trained version; MAC-Splat and token-
  alignment work may contain a broader consistency objective. The remaining delta must be tied to
  arbitrary compact teacher-field semantics or field-only reuse.
- **Novelty confidence:** Medium for pre-fitted teacher-field integration; low for the broad
  detector-and-track concept because G²SR closes it.
- **Highest reachable evidence maturity:** E3 with public multiscene GPU experiments; E2 in current
  CPU synthetic scope.
- **Success / partial / informative-failure outputs:** Success establishes track-aware Stage 1;
  partial success identifies only boundary or baseline regimes; failure shows compact independent
  fields discard necessary identity and redirects work to points/features.

### Candidate: three-view covariance tomography with a surfel fallback

- **Central claim:** Three well-conditioned projected covariances can recover useful full 3D
  anisotropy, while an automatically detected rank/condition failure should fall back to a thin
  surfel and outperform a universal covariance rule.
- **Novelty class:** N2.
- **Known foundation:** Linearized 3D Gaussian projection, multiview ellipse/quadric reconstruction,
  G²SR's projected-Hellinger surfel fit, and local covariance regularization.
- **Irreducible delta:** A rank-aware estimator selects between full SPD tomography and constrained
  surfel geometry per track, and exposes identifiability diagnostics to allocation/refinement.
- **Why this is not merely A+B:** The solver's output family is chosen by the observed projection
  operator's numerical rank; uncertainty controls downstream coverage and freezing.
- **New prediction:** Full SPD recovery improves held-out footprint agreement for volumetric or
  oblique structures only when three-view angular diversity passes a condition threshold; the
  surfel fallback wins elsewhere.
- **Cheapest killing test:** Solve exact synthetic covariances across a grid of view angles, aspect
  ratios, noise, and two/three/four views; render held-out projections without appearance fitting.
- **Null hypothesis:** Rank-aware full covariance never improves held-out Hellinger error or normal
  accuracy enough to justify its instability over the surfel baseline.
- **Prior-art threats:** Astronomy MGE analytically deprojects 2D Gaussian mixtures under shape
  priors; Panaretos recovers radial 3D mixture shape from 2D mixture profiles; cryo-EM and CT
  optimize projected 3D GMMs. Classical ellipse/quadric work also contains related algebra.
- **Novelty confidence:** Low for mixture deprojection or the linear algebra; medium-low for a
  rank-aware, visibility- and renderer-calibrated estimator that demonstrably helps arbitrary
  independently fitted radiance fields.
- **Highest reachable evidence maturity:** E2 for mechanism; E3 after real calibrated multiscene
  evaluation.
- **Success / partial / informative-failure outputs:** Success defines a view-conditioned shape
  rule; partial identifies a narrow baseline/aspect regime; failure justifies surfels as the robust
  universal output.

### Candidate: uncertainty-directed coverage repair after analytic lifting

- **Central claim:** Track rejection residuals and projected coverage debt can target a short local
  addition phase that recovers G²SR-like holes without allowing global densification to erase
  geometric initialization.
- **Novelty class:** N2.
- **Known foundation:** G²SR's precision-first rejection, Image-GS residual allocation, Off The
  Grid/ATSplat uncertainty allocation, and initialization–densification interaction evidence.
- **Irreducible delta:** A causal “coverage debt” ledger links rejected 2D components and low-alpha
  pixels to bounded 3D hypotheses while freezing well-conditioned analytic tracks.
- **Why this is not merely A+B:** New primitives are authorized by provenance-consistent missing
  evidence, not generic render gradient; the controller distinguishes missing coverage from
  appearance residual on existing geometry.
- **New prediction:** At fixed added count and time, debt-targeted additions raise all-pixel
  coverage/LPIPS more than default densification while preserving depth and normal accuracy.
- **Cheapest killing test:** Take a fixed analytic synthetic reconstruction with injected rejected
  tracks; compare no additions, render-gradient additions, and debt-directed additions for 100
  equal-cost updates.
- **Null hypothesis:** Coverage debt provides no gain beyond ordinary error-driven additions at
  matched count/time, or it creates equivalent floaters.
- **Prior-art threats:** Image-GS residual growth, uncertainty-based feed-forward allocation,
  visibility-aware densification, and Structure-Aware Densification.
- **Novelty confidence:** Medium-low; provenance coupling is the likely delta.
- **Highest reachable evidence maturity:** E2 in synthetic; E3 on masked and unmasked scenes.
- **Success / partial / informative-failure outputs:** Success provides the missing precision/
  coverage bridge; partial helps only mask boundaries; failure favors dense hybrid anchors from the
  start.

## Exploratory candidates

### Candidate: dual covariance semantics

- **Central claim:** Maintaining separate localization and rendering/coverage covariances per
  source component improves association precision without creating holes after lifting.
- **Novelty class:** N2.
- **Known foundation:** statistical measurement covariance, Gaussian splat footprint covariance,
  anti-aliasing filters, and the repository's existing covariance controls.
- **Irreducible delta:** The representation admits that the covariance best for finding a center is
  not the covariance best for covering image signal; each has a separately audited downstream use.
- **Why this is not merely A+B:** It changes the observation schema and removes an implicit shared-
  covariance contract throughout matching, fusion, and rendering.
- **New prediction:** Sharp localization covariance improves track precision/triangulation, while
  broader coverage covariance preserves source reconstruction and reduces repair count.
- **Cheapest killing test:** Construct exact fields with broad smooth regions and sharp landmarks;
  compare one covariance versus analytically separated estimates under matched parameters.
- **Null hypothesis:** Separate covariances do not improve the precision–coverage Pareto frontier or
  the extra degrees of freedom destabilize fitting.
- **Prior-art threats:** uncertainty-aware keypoint detectors, anti-aliased point splatting, and
  Gaussian measurement models may already formalize the separation.
- **Novelty confidence:** Medium as a realtime-gs representation intervention; low as a general
  statistical concept.
- **Highest reachable evidence maturity:** E2 initially.
- **Success / partial / informative-failure outputs:** Success validates a new field contract;
  partial isolates only association or rendering benefit; failure supports one minimal covariance.

### Candidate: mask-core/boundary/background populations

- **Central claim:** Splitting masked components into confident core, uncertain contour, and
  optional background populations improves boundary fidelity and robustness versus one mask weight
  applied uniformly.
- **Novelty class:** N1.
- **Known foundation:** contour-aware 2D fitting, soft probability masks, visual hulls,
  probabilistic object pruning, and trimap matting.
- **Irreducible delta:** Each mask region receives a different geometry rule: core can triangulate
  and freeze, boundary keeps depth/association uncertainty, background is excluded or modeled
  separately.
- **Why this is not merely A+B:** The mask is used to choose observability and optimizer behavior,
  not only multiply a loss.
- **New prediction:** Boundary F-score and outside alpha improve under perturbed masks without
  reducing core geometry accuracy; boundary splats remain fewer but more uncertain.
- **Cheapest killing test:** Synthetic object with exact depth, thin structures, and controlled
  mask erosion/dilation; matched count and fixed lift.
- **Null hypothesis:** A calibrated scalar soft mask performs equally well, making population
  splitting needless complexity.
- **Prior-art threats:** probabilistic object reconstruction, alpha matting, and foreground/
  background layered NeRF/GS methods.
- **Novelty confidence:** Low-medium; likely an engineering transfer unless its geometry-specific
  prediction is strong.
- **Highest reachable evidence maturity:** E2 then E3.
- **Success / partial / informative-failure outputs:** Success yields a robust masked mode; partial
  only helps thin boundaries; failure favors a simpler soft-probability pipeline.

### Candidate: renderer-calibrated amplitude transfer

- **Central claim:** A small differentiable calibration from additive source amplitude to 3D alpha
  and color converges faster and avoids opacity artifacts compared with either copying weights or
  discarding them entirely.
- **Novelty class:** N1.
- **Known foundation:** GaussianImage additive accumulation, 3DGS alpha compositing, radiometric
  calibration, and product/gauge non-identifiability.
- **Irreducible delta:** Calibration explicitly renders the lifted geometry under the target
  compositing law while regularizing against source support and track confidence.
- **Why this is not merely A+B:** It treats renderer conversion as a system-identification problem
  with a frozen geometric scaffold, rather than a parameter copy or full scene optimization.
- **New prediction:** Independent low-alpha initialization plus calibrated color/alpha reaches a
  fixed all-pixel LPIPS sooner and with fewer opacity resets than copied source amplitude.
- **Cheapest killing test:** Exact synthetic geometry with known additive source fields; compare
  copy, constant, and 50-step calibration under identical target renderer.
- **Null hypothesis:** Source weights contain no useful information after geometry/support is known,
  so a constant-alpha initialization converges equally quickly.
- **Prior-art threats:** opacity initialization studies, alpha compositing inverse problems, and
  direct Gaussian transfer methods may already include equivalent fitting implicitly.
- **Novelty confidence:** Low as a standalone contribution; medium as a necessary protocol repair.
- **Highest reachable evidence maturity:** E2.
- **Success / partial / informative-failure outputs:** Success defines a safe renderer bridge;
  failure closes weight transfer and simplifies the interface.

## Transformational candidates

### Candidate: track-native multiview Gaussian field

- **Changed problem grammar:** Replace “fit one compact field per image, then discover
  correspondence” with “fit one sparse multiview field whose observations are per-view Gaussian
  projections and whose latent identity exists before 3D depth is solved.”
- **Central claim:** A track-native representation can retain the compression and continuous
  supervision benefits of 2D fields while eliminating most post-hoc mixture matching.
- **Novelty class:** N3.
- **Known foundation:** bundle adjustment, probabilistic feature tracks, multiview Gaussian
  projection, G²SR detection/tracking, and object-centric factor graphs.
- **Irreducible delta:** Component identity, visibility, and split/merge ancestry are first-class
  latent variables shared across views; 3D depth/covariance may remain unresolved during Stage 1.
- **Why this is not merely A+B:** It changes the unit of optimization from image primitives to a
  multiview factor graph and lets allocation be driven by cross-view evidence before committing to
  3D.
- **New prediction:** Track-native fields require fewer components than dense pixels, retain more
  usable correspondences than independent fields, and converge to 3D faster than post-hoc matching
  at equal source reconstruction error.
- **Cheapest killing test:** Two-view synthetic factor graph with known flow, explicit births/
  deaths, and only center/color before depth; compare track survival and optimization conditioning.
- **Null hypothesis:** Optimizing shared identity is as difficult and brittle as optimizing 3DGS
  directly, providing no compression or convergence advantage.
- **Prior-art threats:** G²SR may be interpreted as a detector-plus-track version; multiview
  Gaussian feature tracking and MAC-Splat-like consistency may cover much of the grammar.
- **Novelty confidence:** Medium-low until the latent pre-3D representation is distinguished from
  ordinary feature tracks and direct 3DGS.
- **Highest reachable evidence maturity:** E2/E3.
- **Success / partial / informative-failure outputs:** Success creates a new Stage-1/2 boundary;
  partial helps posed pairs only; failure recommends direct geometry prediction.

### Candidate: evidence-preserving field-only reconstruction

- **Changed problem grammar:** Treat the fitted fields plus their renderer/camera metadata as a
  lossy measurement archive and prohibit access to source pixels during lifting, so the experiment
  measures whether the field is a sufficient statistic for reconstruction.
- **Central claim:** A carefully designed field containing appearance, support, descriptors, and
  uncertainty can approach RGB-assisted geometry/quality while reducing storage and repeated
  image processing.
- **Novelty class:** N4.
- **Known foundation:** compressed-domain vision, Gaussian image representations, feature
  descriptors, and direct multiview lifting.
- **Irreducible delta:** Sufficiency of the field—not just its render fidelity—is the primary
  objective and evaluation axis.
- **Why this is not merely A+B:** It establishes a new systems boundary: source images can be
  discarded, so every required correspondence and uncertainty signal must survive compression.
- **New prediction:** Fields trained for multiview sufficiency close most of the RGB-assisted gap
  at a much smaller stored observation size; ordinary high-PSNR fields do not.
- **Cheapest killing test:** Encode three small calibrated scenes, delete RGB access for the lift,
  and compare field-only with RGB/flow at matched preprocessing time and stored bytes.
- **Null hypothesis:** The compact field omits texture/semantic cues needed for correspondence and
  cannot approach RGB-assisted coverage or geometry without becoming as large as the images.
- **Prior-art threats:** compressed-domain SfM, learned local-feature archives, G²SR's detector
  outputs, and Gaussian feature-map methods.
- **Novelty confidence:** Medium as a strict capability framing; technical success risk is high.
- **Highest reachable evidence maturity:** E3; a negative E2 result is valuable.
- **Success / partial / informative-failure outputs:** Success establishes an agent-/archive-native
  observation format; partial maps the byte-quality frontier; failure quantifies the information
  that Gaussian RGB fields discard.

## Cross-domain transfers

### Transfer: assumption-aware analytic mixture deprojection

- **Donor field and mechanism:** Astronomy's
  [Multi-Gaussian Expansion](https://arxiv.org/abs/astro-ph/0201430)
  analytically deprojects fitted 2D Gaussian brightness mixtures when viewing geometry and an
  axisymmetric/triaxial intrinsic family are assumed; Panaretos'
  [random tomography](https://arxiv.org/abs/0909.0349) estimates 3D radial-mixture center shape from
  many 2D profiles through component weights and projected Gram matrices.
- **Recipient mapping:** A restricted realtime-gs scene or local region supplies an additive 2D
  mixture, calibrated view geometry, and an explicit surface/volume symmetry prior; the analytic
  output is a seed and diagnostic, not final alpha radiance.
- **Preserved causal structure:** Gaussian marginalization closes analytically, and multiple
  projected second-order relations constrain intrinsic shape only after the missing orientation or
  symmetry is supplied.
- **Broken correspondence:** MGE typically uses concentric distant-view brightness components;
  random tomography uses radial/common kernels and many additive profiles. Arbitrary image fields
  have component splits, perspective, visibility, color, and alpha composition.
- **Required invention:** Detect when a local track satisfies a donor assumption, transform the
  analytic density/mass seed into target-renderer opacity, and expose assumption violations instead
  of returning a plausible but unsupported 3D Gaussian.
- **Adoption barrier:** The baseline is easy to implement but applies to a narrow subset; accidental
  use outside that subset can create deceptively smooth wrong geometry.
- **Native baseline:** Center triangulation plus G²SR-style thin-surface covariance.
- **Recipient-specific prediction:** On additive axisymmetric synthetic scenes, the analytic seed
  matches or beats iterative covariance fitting in error and runtime; on general scenes, its
  assumption residual reliably rejects use rather than silently harming held-out views.
- **Counter-analogy:** Image Gaussians chosen for texture coverage need not be marginal projections
  of any shared positive 3D density, so no analytic deprojection is physically meaningful.
- **Cheapest killing test:** Implement only the closed-form axisymmetric case and a Panaretos-style
  center-Gram toy problem; require exact forward round trips and correct rejection as inclination,
  perspective, and component labeling are perturbed.
- **Prior-art threats:** The donor methods are the prior art; the transfer itself is N0 unless the
  assumption diagnostics or renderer conversion yield a new measured system effect.
- **Novelty confidence:** N0 as deprojection, N1-T as a renderer-aware seed/diagnostic.

### Transfer: limited-angle covariance tomography

- **Donor field and mechanism:** Medical/physical tomography reconstructs an unknown field from
  projections, diagnoses null spaces by acquisition geometry, and uses priors only where the
  projection operator lacks rank.
- **Recipient mapping:** Each matched 2D Gaussian covariance is a local projected second moment;
  camera baselines are acquisition angles; surfel thickness or eigenvalue bounds are priors.
- **Preserved causal structure:** More diverse projection directions improve observability, while
  limited angle creates structured null directions and noise amplification.
- **Broken correspondence:** Tomography normally integrates a fixed global density on known rays;
  independently fitted Gaussian mixtures change component identity and renderer semantics between
  views.
- **Required invention:** Robust component association, visibility-aware moment equations, and a
  target-renderer calibration for opacity/color.
- **Adoption barrier:** The linear covariance solver is easy; proving that fitted footprints are
  measurements of one physical local density is hard.
- **Native baseline:** G²SR thin-surface fit and a fixed footprint/surface covariance rule.
- **Recipient-specific prediction:** An acquisition condition number predicts when a third view
  improves covariance and when the solver should remain surfel-constrained.
- **Counter-analogy:** If image fields allocate Gaussians for texture coverage rather than physical
  moments, adding views does not converge to one covariance.
- **Cheapest killing test:** Exact projected Gaussians followed by independently re-fitted mixtures,
  to separate solver identifiability from componentization error.
- **Prior-art threats:** Gaussian-mixture tomography and multiview ellipse/quadric reconstruction.
- **Novelty confidence:** N2-T for the rank-aware 3DGS adaptation; N0 for the underlying algebra.

### Transfer: probabilistic multi-target data association

- **Donor field and mechanism:** Radar/robotics tracking maintains association hypotheses, track
  births/deaths, gating, and uncertainty rather than committing every detection to one target.
- **Recipient mapping:** Per-view Gaussian components are detections; latent surface elements are
  tracks; occlusion is missed detection; split/merge is extended-target measurement ambiguity.
- **Preserved causal structure:** Premature hard matches create persistent state errors, while
  uncertainty-aware gating trades precision against coverage.
- **Broken correspondence:** Surface appearance and image decomposition change continuously with
  viewpoint; components are not independent sensor detections of fixed objects.
- **Required invention:** Split/merge-aware affine footprint likelihoods and visibility-conditioned
  track scoring for dense surface elements.
- **Adoption barrier:** Multiple hypotheses can destroy the speed and compactness advantage unless
  gating is extremely sparse.
- **Native baseline:** Mutual-nearest center/descriptor matching and G²SR forward/back rejection.
- **Recipient-specific prediction:** Keeping only two or three local hypotheses near occlusion
  boundaries improves final coverage without increasing core-region false geometry.
- **Counter-analogy:** Dense image fields may contain too many exchangeable components for target-
  tracking assumptions to remain computationally useful.
- **Cheapest killing test:** Synthetic occlusion and component split sequences with exact poses;
  compare hard mutual match and capped multi-hypothesis tracks.
- **Prior-art threats:** multiview feature-track graphs, Gaussian-mixture registration, and
  differentiable assignment methods.
- **Novelty confidence:** N2-T, low-medium.

### Transfer: epipolar-gated multi-marginal mixture transport

- **Donor field and mechanism:**
  [Mixture Wasserstein distance](https://arxiv.org/abs/1907.05254) reduces transport between GMMs
  to a discrete coupling over Gaussian components and supplies multi-marginal/barycenter variants;
  [JRMPC](https://arxiv.org/abs/1609.01466) and
  [Coherent Point Drift](https://arxiv.org/abs/0905.2635) use EM responsibilities, shared latent
  mixtures, and explicit robustness to noise, outliers, or missing points.
- **Recipient mapping:** Per-view image Gaussians are mixture mass; calibrated projection supplies
  epipolar gates; a latent 3D track is a multi-view transport cluster; unmatched/background mass is
  routed to a dustbin rather than forced into geometry.
- **Preserved causal structure:** Soft mass coupling tolerates permutation, missing observations,
  and component-count mismatch while using Gaussian mean/covariance geometry more fully than a hard
  center match.
- **Broken correspondence:** Donor methods register same-dimensional Euclidean measures; here the
  latent object is 3D, observations are perspective 2D footprints, visibility changes mass, and one
  physical patch may be decomposed differently in every view.
- **Required invention:** A projective component cost, sparse epipolar gating, capped one-to-many
  mass, visibility-conditioned dustbins, and track extraction that retains calibrated uncertainty.
- **Adoption barrier:** Dense multi-marginal transport can erase the latency advantage; sparse gates
  and a strict hypothesis budget are mandatory.
- **Native baseline:** Mutual-nearest five-sigma-point matching with forward/backward rejection.
- **Recipient-specific prediction:** At matched runtime or accepted-track count, soft transport
  improves precision×coverage and held-out geometry most near occlusion boundaries and asymmetric
  split/merge events, with little change in confident interiors.
- **Counter-analogy:** If independently fitted components are nearly exchangeable texture tiles,
  transport spreads mass among many equally plausible matches and produces no usable identity.
- **Cheapest killing test:** Generate exact multiview mixtures, then inject controlled component
  split, merge, disappearance, and background clutter; compare hard tracks, pairwise transport, and
  capped multi-marginal transport before any photometric refinement.
- **Prior-art threats:** Projective optimal transport, differentiable matching, probabilistic
  feature tracks, and multi-view mixture registration may already contain most of the algorithm.
- **Novelty confidence:** N1-T for soft mixture coupling; N2-T, low-medium, only for an efficient
  visibility-aware field-to-3D formulation with a new causal result.

### Transfer: optimal experimental design for next-view capture

- **Donor field and mechanism:** Active vision and experimental design choose the next observation
  to maximize information gain or reduce posterior uncertainty.
- **Recipient mapping:** Field-track rank, covariance condition number, visual-hull uncertainty,
  and coverage debt define where a new camera view is valuable.
- **Preserved causal structure:** Measurement geometry controls observability; another near-
  duplicate view adds less information than a diverse view that sees missing surfaces.
- **Broken correspondence:** Users may supply a fixed photo set, and appearance/occlusion make
  analytic information gain an imperfect proxy for final NVS quality.
- **Required invention:** A fast field-derived view score combining covariance-rank gain, expected
  track survival, mask-hull reduction, and disocclusion coverage.
- **Adoption barrier:** Requires interactive or robotic recapture and calibrated candidate poses.
- **Native baseline:** Evenly spaced orbit capture or largest angular-gap heuristic.
- **Recipient-specific prediction:** A rank/coverage score reaches target geometry with fewer views
  than uniform orbit sampling, especially for thin or concave masked objects.
- **Counter-analogy:** If correspondence is the dominant failure, a geometrically optimal view may
  add unusable detections and no effective information.
- **Cheapest killing test:** Offline next-view simulation on a synthetic object from a dense camera
  pool; compare uniform, random, angular-gap, and predicted information gain.
- **Prior-art threats:** active NeRF/3DGS view selection, next-best-view mesh capture, and visual-hull
  planning.
- **Novelty confidence:** N1-T for transfer; likely crowded, but useful as an evidence program.

### Transfer: projection-nonlinearity topology control

- **Donor field and mechanism:** Navigation/tracking adaptively splits Gaussian mixture components
  when a nonlinear transform makes one Gaussian inaccurate. Kulik and LeGrand's
  [moment-matching method](https://arxiv.org/abs/2412.00343) selects split direction using both
  uncertainty and transform nonlinearity; Runnalls'
  [mixture reduction](https://www.cs.kent.ac.uk/pubs/2007/2797/) merges pairs by second-moment
  matching with a KL-upper-bound criterion.
- **Recipient mapping:** Perspective projection is the nonlinear transform; a 3D Gaussian's spatial
  support is uncertainty extent; split direction follows projection curvature; merges operate on
  redundant lifted tracks after renderer-aware checks.
- **Preserved causal structure:** Large support along a high-curvature transform direction causes a
  single Gaussian approximation to fail, while selective splitting spends count only where it can
  reduce that approximation error.
- **Broken correspondence:** Donor densities are additive probabilities. In alpha compositing,
  order and visibility mean that mass-, mean-, and covariance-preserving topology changes can still
  change the rendered image.
- **Required invention:** A cheap projection-nonlinearity indicator, an exact or sampled projection
  oracle, geometric support mass distinct from opacity, and a merge veto based on multiview render
  and visibility error.
- **Adoption barrier:** Oracle evaluation and repeated topology changes may cost more than the
  Gaussians they save; stable differentiable alpha semantics are nontrivial.
- **Native baseline:** Gradient-threshold clone/split/prune and fixed footprint subdivision.
- **Recipient-specific prediction:** At fixed final count and wall-clock budget, nonlinearity-
  directed splits reduce held-out footprint error and extreme-perspective artifacts, while
  KL/moment-guided merges recover compactness without erasing geometry.
- **Counter-analogy:** If the dominant error is wrong depth or association rather than local
  projection approximation, splitting only duplicates incorrect geometry.
- **Cheapest killing test:** Project exact anisotropic Gaussians over a depth/aspect/view-angle grid;
  compare unsplit, longest-axis split, residual-gradient split, and nonlinearity-directed split,
  then test merges under both additive and alpha forward models.
- **Prior-art threats:** 3DGS structure-/frequency-/uncertainty-aware densification, antialiasing,
  analytic projection, and conservative remeshing may already cover most benefits.
- **Novelty confidence:** N1-T for the mechanism; N2-T, low-medium, for a projection-aware
  split/merge controller that wins the recipient count–quality–time tradeoff.

### Transfer: independent-half reconstruction stability

- **Donor field and mechanism:** Cryo-EM's
  [gold-standard half-map protocol](https://www.nature.com/articles/nmeth.2115) independently refines
  two models from disjoint data halves so shared noise or a shared refined reference cannot inflate
  agreement.
- **Recipient mapping:** Stratified camera halves replace particle halves; two independently lifted
  and refined Gaussian fields are compared only in mutually observed regions using geometry,
  projected covariance, and alpha-support agreement.
- **Preserved causal structure:** Structure genuinely constrained by independent observations should
  recur, while view-specific overfit, floaters, and unidentifiable covariance should decorrelate.
- **Broken correspondence:** Camera halves see different surfaces and baselines, unlike random
  particles of one molecular structure; disagreement can be caused by coverage rather than
  overfitting.
- **Required invention:** Angle/coverage-matched splitting, overlap masks, registration-invariant
  component/field comparison, and reporting that separates coverage loss from instability.
- **Adoption barrier:** It nearly doubles reconstruction cost and does not directly improve the
  production model.
- **Native baseline:** Held-out image/depth metrics, multi-seed variance, and source-view
  reprojection residual.
- **Recipient-specific prediction:** Half-field disagreement localizes regions with worse held-out
  geometry and higher seed variance, and detects overconfident covariance even when source-view
  photometric error is low.
- **Counter-analogy:** Strong shared pretrained priors can make both halves agree on the same
  hallucination, while legitimate disoccluded surfaces can appear in only one half.
- **Cheapest killing test:** On synthetic scenes with ground-truth geometry, measure whether
  coverage-adjusted half-field disagreement ranks injected correspondence, mask, and covariance
  failures better than source residual or ordinary seed variance.
- **Prior-art threats:** multi-view bootstrap, split-view consistency, ensemble uncertainty, and
  NeRF/3DGS cross-validation may already instantiate the recipient protocol.
- **Novelty confidence:** N0 as validation principle; N1-T as a calibrated field-specific evidence
  protocol.

### Transfer: ordered views with reversible freezing

- **Donor field and mechanism:** Tomographic
  [ordered-subsets EM](https://pubmed.ncbi.nlm.nih.gov/18218538/) accelerates early reconstruction by
  applying sequential view blocks; the Scholar Inbox surfaced
  [Incremental Online Scene Reconstruction](https://arxiv.org/abs/2607.10690), which freezes fully
  optimized historical Gaussian regions to reduce long-sequence compute.
- **Recipient mapping:** Angularly balanced camera blocks are ordered subsets; analytic low-residual,
  well-conditioned tracks are frozen; coverage-debt, disocclusion, or residual change triggers thaw.
- **Preserved causal structure:** Early updates need not process every observation, and repeatedly
  optimizing already stable variables wastes compute.
- **Broken correspondence:** OSEM's positive emission model has a structured likelihood, while
  alpha-rendered photometric optimization is nonconvex and view blocks have conflicting visibility;
  early errors can be locked in.
- **Required invention:** A block schedule that balances angle/appearance, freeze confidence tied to
  geometric conditioning, periodic all-view audits, and a deterministic thaw rule.
- **Adoption barrier:** State management and cache invalidation can overwhelm savings on small
  scenes, and subset cycles can harm convergence.
- **Native baseline:** Uniform randomized camera minibatches with every primitive trainable.
- **Recipient-specific prediction:** Ordered blocks plus reversible freezing reduce wall-clock
  time-to-target on long multiview sequences without changing final all-view quality, with the gain
  concentrated in stable historical regions.
- **Counter-analogy:** If appearance and geometry remain globally coupled, every new view changes
  gradients everywhere and no region is truly stable.
- **Cheapest killing test:** Replay a fixed long calibrated sequence with identical initialization;
  compare random views, angular blocks, irreversible freezing, and reversible freezing, all ending
  with the same full-view cleanup budget.
- **Prior-art threats:** ordinary SGD minibatching, keyframe selection, local/global bundle
  adjustment, SLAM marginalization, and online/frozen 3DGS systems make this a crowded transfer.
- **Novelty confidence:** N1-T, low; prioritize only as a performance ablation after geometry works.

## New-evidence discovery programs

### Program: local identifiability atlas

- **What varies:** View count and angular spread, mean depth, covariance eigenvalues/orientation,
  raster filter, noise, track error, and surfel/full-SPD parameterization.
- **What is measured:** Projection-operator singular values, recovered mean/covariance error,
  normal error, held-out Hellinger distance, solver iterations, and failure rate.
- **Surprising outcome:** Full SPD may outperform surfels only in a small non-surface regime, or
  two-view affine information may be more useful than the naive rank analysis predicts under the
  thin prior.
- **Conventional explanation:** More views and wider baselines improve triangulation and covariance
  recovery monotonically.
- **Negative control:** Exact ground-truth correspondences and covariances with no independent
  field fitting.
- **Leakage/bug exclusion:** Closed-form projection checks, finite-difference Jacobians, recovered-
  to-forward round trips, seed replay, and float64 reference.
- **Raw evidence:** Per-case operators, singular values, inputs, outputs, residual traces, and exact
  rendered projections.
- **Abandonment rule:** Stop full-SPD integration if no preregistered regime improves held-out shape
  or geometry over surfels by a material margin under stable conditioning.

### Program: source fidelity versus liftability frontier

- **What varies:** GaussianImage/Image-GS/contour-aware/correspondence-aware field, primitive count,
  source loss, and independent versus joint fitting.
- **What is measured:** Source PSNR/LPIPS, track survival/precision, split/merge rate,
  triangulation conditioning, 3D geometry, coverage, and time-to-target.
- **Surprising outcome:** A lower-PSNR field may reconstruct better 3D; or field type may cease to
  matter after a robust feature tracker.
- **Conventional explanation:** More accurate image approximation preserves more reconstruction
  information.
- **Negative control:** Raw RGB keypoints/dense flow and exact synthetic component identities.
- **Leakage/bug exclusion:** Same cameras/tracker/back end, matched field count, no held-out image in
  fitting, and shuffled-track controls.
- **Raw evidence:** Source fields, renders, match graph, rejected reasons, 3D output, checkpoints,
  and timing trace.
- **Abandonment rule:** If no field method beats centers/raw-image matching at equal stored bytes or
  total time across the protected scene set, stop claiming field shape as a convergence lever.

### Program: initializer–controller causal matrix

- **What varies:** SfM/dense point/field-center/field-covariance initialization crossed with fixed
  topology, short debt repair, classic ADC, and MCMC-like control at equal time/count budgets.
- **What is measured:** Geometry and appearance curves, primitive ancestry, displacement from
  initialization, split/prune/relocation counts, coverage, and final count.
- **Surprising outcome:** A field lift may win only without densification, or a controller may
  improve final quality by erasing the field geometry while losing convergence AUC.
- **Conventional explanation:** A better initializer should improve every training configuration.
- **Negative control:** Identical initializer duplicated under different labels/config plumbing;
  shuffled covariance and randomized centers.
- **Leakage/bug exclusion:** Common renderer, loss, seeds, view order, wall-clock checkpoints, and
  ancestry/hash validation.
- **Raw evidence:** Full trajectories, controller events, exact configs, environment, and all-pixel
  held-out renders.
- **Abandonment rule:** If the field advantage disappears under every controller or only appears
  after unmatched tuning, retain the representation as analysis infrastructure rather than a
  performance method.

## Pareto set

Scores are 1 (weak/high cost) to 5 (strong/low cost) and are prioritization judgments, not measured
results.

| Candidate | Novelty | Falsifiability | Value | Feasibility | First-test cost | Negative-result value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correspondence-conditioned teacher fields | 3 | 5 | 5 | 3 | 3 | 5 |
| Three-view covariance tomography | 3 | 5 | 4 | 5 | 5 | 5 |
| Uncertainty-directed coverage repair | 3 | 4 | 5 | 3 | 3 | 4 |
| Dual covariance semantics | 3 | 4 | 4 | 4 | 4 | 4 |
| Mask core/boundary/background populations | 2 | 5 | 4 | 4 | 4 | 3 |
| Renderer-calibrated amplitude transfer | 2 | 5 | 3 | 5 | 5 | 4 |
| Track-native multiview field | 4 | 4 | 5 | 2 | 2 | 4 |
| Evidence-preserving field-only reconstruction | 5 | 5 | 5 | 2 | 2 | 5 |
| Analytic mixture-deprojection baseline | 1 | 5 | 3 | 5 | 5 | 5 |
| Limited-angle tomography transfer | 3 | 5 | 4 | 5 | 5 | 5 |
| Multi-target association transfer | 3 | 4 | 4 | 2 | 2 | 4 |
| Multi-marginal mixture transport | 3 | 5 | 5 | 3 | 3 | 5 |
| Projection-nonlinearity topology control | 3 | 5 | 4 | 4 | 4 | 4 |
| Independent-half stability protocol | 2 | 5 | 4 | 4 | 3 | 5 |
| Ordered views with reversible freezing | 2 | 5 | 3 | 4 | 4 | 3 |
| Next-view experimental design transfer | 2 | 5 | 3 | 3 | 3 | 3 |

The immediate Pareto set is:

- **lowest-cost correctness baselines:** MGE-style analytic deprojection and Panaretos-style center
  Gram recovery on restricted synthetic cases;
- **lowest-cost distinctive mechanism test:** three-view covariance tomography;
- **highest decision value:** source fidelity versus liftability;
- **highest near-term system value:** correspondence-conditioned fields plus epipolar-gated mixture
  transport and coverage debt;
- **highest-risk/highest-upside capability:** evidence-preserving field-only reconstruction;
- **best masked extension:** core/boundary/background populations after the base lift works.

## Recommended first experiment

Run a protected **field-shape utility and identifiability** experiment before implementing a large
learned field-to-field model.

1. Create exact synthetic scenes of thin 3D Gaussians with three calibrated views and held-out
   projections. Sweep baseline angle, footprint aspect ratio, and small center/covariance noise.
2. First validate an MGE-style axisymmetric additive case and a Panaretos-style radial-mixture
   center-Gram case. These are restricted correctness baselines, not competitors on alpha-rendered
   scenes.
3. Produce exact projected fields and independently re-fitted fields so the study separates
   inverse-projection algebra from componentization/correspondence failure.
4. Compare center-only triangulation, center plus fixed pixel footprint, G²SR-like five-sigma-point
   surfel fitting, and three-view rank-aware full covariance with surfel fallback.
5. In a separate association arm, inject component splits/merges/missing mass and compare hard
   matching with epipolar-gated capped mixture transport. Do not conflate association gains with the
   covariance solver.
6. Hold track identity, primitive count, renderer, and any refinement budget fixed in the shape
   arm. Report mean,
   normal, covariance, and held-out Hellinger errors plus solver time and failure/coverage.
7. Add one matched real micro-capture only after the exact case passes; then compare independent
   GaussianImage/Image-GS fields against a correspondence-conditioned field at matched count.

The primary null is that source covariance adds no material held-out geometry or convergence
benefit beyond centers plus a projection-correct footprint. The permanent stopping rule should
close full-covariance or arbitrary-field claims if that null survives the well-conditioned exact
case. This experiment is cheap, falsifies the distinctive mechanism directly, and remains useful
even if negative.

## Audit limitations

- The portfolio is a hypothesis generator, not evidence that any candidate is novel, useful, or
  compatible with the existing implementation.
- G²SR closes the broad direct-lifting novelty claim. Several deltas above depend on the narrower
  meanings of independently fitted, compact, correspondence-conditioned, or field-only.
- MGE, random tomography, and cryo-EM GMMs close broad mixture-deprojection novelty under restricted
  additive forward models. Their assumptions must be represented explicitly in every claimed delta.
- Very recent preprints may change and an incomplete terminology search may miss Gaussian-mixture,
  ellipse, surfel, compressed-domain, or tomography work.
- Cross-domain mappings explicitly contain broken correspondence; a donor mechanism cannot be
  adopted until its recipient-specific prediction beats a native baseline.
- Scores in the Pareto table are subjective planning values. Only preregistered repository evidence
  can change defaults, promote claims, or establish maturity.
- The proposed first experiment does not establish real-scene NVS, masks, CUDA performance,
  density-control interaction, or a production default. Each requires a separate protected
  extension after the mechanism survives.
