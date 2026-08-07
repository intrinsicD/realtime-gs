# From 2D Gaussian fields to 3D Gaussian fields

## A state-of-the-art review for quality, speed, and convergence

**Literature cutoff:** 2026-08-04  
**Scope:** static scenes and objects reconstructed from one or more RGB views, with optional
calibrated cameras and foreground masks  
**Evidence policy:** quantitative values below are the source authors' reported results unless
explicitly marked as repository evidence; they have not been reproduced by this project

This review answers a narrower question than a general 3D Gaussian Splatting survey: what is known
about producing a renderable 3D Gaussian field from *2D Gaussian fields*, and what adjacent work
actually improves total reconstruction time, output quality, or convergence? The distinction is
important. The literature uses “2D Gaussian” for at least three different objects, and many
image-to-3DGS papers never construct an image-plane Gaussian field at all.

The short answer is:

1. **A direct modern method now exists.** [G²SR](https://arxiv.org/abs/2607.14470) predicts a
   reference-view field of 2D Gaussian splats, tracks each splat through other images, triangulates
   it, and analytically optimizes a thin metric 3D Gaussian surfel. It is the closest published
   method to the literal transformation in this review.
2. **G²SR wins a geometry/latency corner, not the whole Pareto frontier.** At 384×512 on an RTX
   4090 its detailed tables report about 69–91 reconstructions/s and 115–203 MB peak allocated GPU
   memory (the abstract summarizes the rate as 69–89/s), with strong
   metric depth and DTU geometry. Its three-view rendering has only 73.1–85.8% valid-pixel
   coverage and substantially worse LPIPS than the strongest learned radiance baseline in its own
   protocol.
3. **Restricted 2D-mixture-to-3D-mixture recovery is old prior art.** Astronomy's
   [Multi-Gaussian Expansion](https://arxiv.org/abs/astro-ph/0201430)
   analytically deprojects fitted 2D Gaussian surface-brightness mixtures under an assumed viewing
   geometry and shape family. [Random tomography](https://arxiv.org/abs/0909.0349) recovers a 3D
   radial mixture's center geometry from many 2D mixture projections even when the projection
   angles are unobserved. Cryo-EM's [e2gmm](https://www.nature.com/articles/s41592-021-01220-5)
   maps a learned 3D GMM into known-orientation 2D particle images. These works close any broader
   claim that Gaussian-mixture deprojection itself is new.
4. **No source found by the cutoff demonstrates arbitrary radiance-field conversion.** G²SR does not take
   independently optimized GaussianImage-style decompositions as its sole input. Its small
   detector is trained to emit boundary-respecting, correspondence-stable splats, and tracking
   still consumes the RGB images through optical flow. A converter for arbitrary pre-fitted 2D
   mixtures, especially after the source pixels are discarded, remains an open and narrower
   problem.
5. **Correspondence is harder than inverse projection.** Once a component has reliable
   cross-view identity, its mean can be triangulated and a surface-oriented covariance can be fit
   quickly. Independently fitted mixtures have permutation, split/merge, and gauge freedoms that
   make component identity non-observable from parameters alone.
6. **Two views do not identify an unrestricted 3D covariance.** Two generic 2D covariance
   projections share one redundant quadratic-form constraint and leave a one-dimensional null
   direction. Thin-surface, isotropy, normal, depth, or learned priors make the problem solvable;
   three well-conditioned views can constrain a full covariance.
7. **The best 2D image fit is not necessarily the best 3D initializer.** GaussianImage, Image-GS,
   Instant GaussianImage, and successors improve image PSNR and fitting time, but none shows that
   a photometrically optimal componentization is stable under multi-view matching. Boundary
   respect, repeatability, and geometric conditioning are separate Stage-1 objectives.
8. **Dense point correspondence has the strongest published convergence evidence adjacent to the
   exact problem.** [EDGS](https://arxiv.org/abs/2504.13204) triangulates dense RoMa matches, initializes
   color and spherical harmonics, removes densification, and reports original-3DGS quality in 15%
   of its training time—not 25%—plus better final LPIPS when optimized longer.
9. **Masks change allocation, not projective observability.** Calibrated masks provide visual-cone
   occupancy constraints and foreground semantics. They accelerate object reconstruction by
   avoiding background splats, but cannot recover concavities or assign depth by themselves.
   Soft probabilities and boundary uncertainty are safer than brittle hard intersections.
10. **For unmasked scenes, explicit geometry and background modeling are essential.** Low value in
   an additive image-Gaussian field is not evidence of empty 3D space. Use poses plus tracks,
   depth/point maps, or learned geometric priors, and represent distant/background content
   separately.
11. **There is no method that simultaneously maximizes quality, speed, convergence, compactness,
    geometry, and coverage.** A rigorous comparison must report a Pareto set and total pipeline
    cost, including 2D fitting, pose/mask/depth inference, tracking, lifting, and 3D refinement.

## 1. Definitions: four objects that must not be conflated

| Name in this review | Mathematical object | Typical source | Is it already in 3D? |
| --- | --- | --- | --- |
| **Image-plane Gaussian field** | A finite mixture or sum over `x ∈ R²`, with center `m`, covariance `C`, color/features, and an amplitude or weight | GaussianImage, Image-GS, a fitted teacher field, G²SR detector | No |
| **Projected 3D Gaussian** | The screen-space EWA footprint produced by projecting a 3D mean and covariance | 3DGS rasterizers and reprojection losses | The source primitive is |
| **2DGS surfel** | A planar Gaussian disk embedded in `R³`, normally with two tangent scales and a surface normal | [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888), Sparse2DGS | Yes |
| **Pixel-/token-predicted 3D Gaussian** | A 3D primitive regressed from an image pixel, feature token, depth plane, query, or 3D anchor | Splatter Image, pixelSplat, MVSplat, Off The Grid, QuerySplat | Yes after prediction |

The direct target is a field

```text
F_v(x) = Σ_i a_vi c_vi exp[-1/2 (x - m_vi)^T C_vi^-1 (x - m_vi)]
```

for each view `v`, transformed into a 3D set

```text
G = {(μ_j, Σ_j, α_j, appearance_j)}_j
```

whose projections and alpha-composited renders agree with the observations. The input fields may
be fitted independently, predicted jointly, or derived from masks. Those cases are not
equivalent:

- additive 2D amplitude `a` is generally **not** a volume-rendering opacity `α`;
- a covariance chosen to cover image texture need not be a localization uncertainty or a physical
  surface footprint;
- a low residual image decomposition is not guaranteed to produce the same components in another
  view;
- a mask is support/semantic evidence, not a depth map or a transmittance measurement.

### 1.1 The projection equations

For calibrated camera `v`, with world-to-camera rotation `R_v`, translation `t_v`, and perspective
map `π_v`, a locally linearized 3D Gaussian projects as

```text
m_v = π_v(R_v μ + t_v)
C_v ≈ J_v R_v Σ R_v^T J_v^T + C_filter,
```

where `J_v` is the `2×3` perspective Jacobian at the projected mean and `C_filter` is any renderer
low-pass term. The inverse problem therefore has four logically separate parts:

1. **association:** decide which 2D components across views describe the same surface element;
2. **mean geometry:** triangulate or otherwise infer `μ`;
3. **shape geometry:** infer `Σ`, or a constrained surface-normal/tangent parameterization;
4. **radiometry:** calibrate opacity and appearance for a compositing renderer.

Optimizing all four only through final RGB makes the system flexible but ill-conditioned. The
recent high-speed methods improve convergence by solving or strongly supervising the first three
before photometric refinement.

### 1.2 What is identifiable?

**Mean.** One calibrated view leaves the center anywhere on a ray. Two non-degenerate posed views
can triangulate a point; small baseline, repeated texture, rolling shutter, or poor association
make the estimate unstable. More views add robust consensus and coverage.

**Full covariance.** A symmetric `3×3` covariance has six independent values. Each observed `2×2`
covariance contributes three scalar equations. Counting six equations from two views is
misleading: each view observes the quadratic form only on a two-dimensional projection plane, and
the two planes share a line. Their restriction on that line is duplicated, so the generic linear
system has rank at most five. A third well-conditioned view, or a prior such as fixed normal
thickness, isotropy, known normal, or bounded aspect ratio, is needed to remove the remaining
freedom. Near-parallel views remain poorly conditioned even with more equations.

**Component identity.** A Gaussian sum is invariant to component order. Similar renders may also
be produced after a component is split, merged, widened, or reweighted. Independently optimizing
each view therefore produces no canonical index correspondence. Matching centers alone throws
away the most distinctive local affine information; matching full footprints is stronger but
still fails across occlusion, specularity, repeated patterns, and component-count changes.

**Opacity and color.** GaussianImage-style accumulation and 3DGS alpha compositing are different
forward models. Copying a source field's amplitude into `α` changes occlusion and view-dependent
transmittance. A safe conversion treats source amplitude as a proposal or confidence, initializes
3D opacity independently, and calibrates opacity/color with multi-view rendering loss.

These facts explain why the closest direct paper reconstructs *thin surfels*, not unrestricted
volumetric ellipsoids, and why it trains its image-plane detector for tracking rather than simply
using a maximal-PSNR image compressor.

## 2. Search protocol and evidence calibration

The search covered arXiv, CVF Open Access, ECVA, ACM/TOG and publisher pages, author project pages,
official repositories, and the authenticated Scholar Inbox personalized digest. Scholar Inbox was
used for discovery only; every included technical claim was then checked against a primary paper or
official source. Query families combined:

- `"2D Gaussian image representation" lift "3D Gaussian Splatting" multiview`;
- `"2D Gaussians" "3D Gaussian" reconstruction initialization`;
- Gaussian ellipse/affine-correspondence triangulation and Gaussian-mixture tomography;
- feed-forward, sparse-view, pixel-aligned, query-based, and anchor-based 3DGS;
- masked/object 3DGS, visual hull, probability mask, and silhouette reconstruction;
- 3DGS initialization, densification, convergence speed, compactness, and time-to-quality.
- multi-Gaussian deprojection in astronomy, random tomography, cryo-EM GMMs, mixture
  registration/transport/reduction, nonlinear uncertainty propagation, and independent-half
  reconstruction validation.

Inclusion required a primary paper, official proceedings copy, project page, or official code
source with enough detail to establish the input/output relation. Papers were classified as:

- **D — direct:** constructs image-plane Gaussians and geometrically turns corresponding splats
  into 3D Gaussians;
- **P — precursor:** matches/projectively relates 2D and 3D Gaussian primitives but requires an
  existing surface or solves a simpler geometric object;
- **A — adjacent:** images, pixels, masks, points, depth, or features are converted to 3DGS and the
  mechanism transfers, but a pre-fitted 2D Gaussian field is not the input;
- **S — Stage-1:** fits or predicts image-plane Gaussian fields without reconstructing 3D.

Conference papers are distinguished from arXiv preprints. “No method found” means no qualifying
source was found under this dated protocol; it is not a proof of absence. Cross-paper numbers are
not rankings unless dataset, split, resolution, hardware, metric mask, and timing boundary match.

### 2.1 End-to-end evidence and implementation snapshot

Availability was checked on the cutoff date and is not a license endorsement. “Not located” means
the search found no author-linked public implementation, not that one cannot exist.

| Method/class | Inputs and mask support | Geometry source | 3D output | Scene-specific optimization | Reported evidence emphasized | Author implementation at cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| G²SR / D | 2–3 posed RGB; no mask-specific pipeline | Tracked 2D splat affine frames and calibrated triangulation | Metric thin colored surfels | Up to 20 analytic Gauss–Newton iterations per splat | Geometry, memory, throughput; lower coverage/radiance quality | Preprint; public code not located |
| GPS-Gaussian / A | Two adjacent rectified human RGB views; accurate foreground matting is required preprocessing | Learned binocular depth supervised by scan depth; pixel-wise Gaussian parameter maps | One 3D Gaussian per foreground pixel, aggregated from two views | Feed-forward; no per-subject fitting | Authors report over 25 FPS at 2K and 27 ms source processing + 0.8 ms per novel view on RTX 3090 | [Official code](https://github.com/aipixel/GPS-Gaussian) |
| EDGS / A | Many posed RGB views; no required masks | Dense RoMa correspondences and triangulation | Dense 3DGS with initialized color/SH | 5K–30K gradient steps; densification off | Convergence and final LPIPS | [Official code](https://github.com/CompVis/EDGS) |
| GaussianObject / A | Usually four calibrated RGB views and object masks | Visual hull, monocular depth, floater filtering, learned repair | Object 3DGS | Coarse optimization plus repair, about 30 min reported | Masked sparse-view NVS quality | [Official code](https://github.com/GaussianObject/GaussianObject) |
| Probabilistic object reconstruction / A | RGB, YOLO/SAM soft masks, SfM | Mask-filtered points plus depth/normal/probability losses | Foreground-probability 3D surfels | 30K iterations in reported timing | Boundary robustness, count, object segmentation | Preprint; public code not located |
| MVSplat / A | Posed sparse RGB; no required masks | Plane-sweep cost volume | Pixel-aligned 3DGS | One forward pass | Amortized NVS speed/quality | [Official code](https://github.com/donydchen/mvsplat) |
| DepthSplat / A | Posed sparse RGB; no required masks | Monocular foundation features plus multiview cost volume | Pixel-aligned 3DGS | One forward pass | Depth and NVS, up to 12 input views | [Official code](https://github.com/cvg/depthsplat) |
| Off The Grid / A | Unposed RGB; no required masks | Fine-tuned geometry backbone, subpixel detector | Adaptive non-grid 3DGS | One forward pass | Compactness and pose-free NVS | [Official project and code](https://arthurmoreau.github.io/OffTheGrid/) |
| AnchorSplat / A | Multiview images plus 3D anchors from points/voxels/RGB-D | Explicit 3D geometric anchors | Anchor-aligned 3DGS | A few forward refinement passes | Compact pose-aware NVS | [CVPR paper/supplement](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html); recheck code |
| ATSplat / A | Up to 12 multiview images in its reported setup | Coarse depth/camera tokens and uncertainty expansion | Adaptive token-aligned 3DGS | One forward pass | Subsecond compact prediction | Preprint; public code not located |
| QuerySplat / A | Unposed RGB | Pretrained geometry branch plus separate appearance queries | Query-based 3DGS | One forward pass | Very recent DL3DV NVS claim | [Project/repository](https://inspatio.github.io/querysplat/) linked by preprint |
| Arbitrary fitted-field converter | Posed GaussianImage/Image-GS fields; desired masked/unmasked modes | Unresolved association and inverse projection | Desired full 3D radiance field | Unknown | No located end-to-end evidence | No established implementation |

## 3. The direct lineage

| Work | Class/status at cutoff | Input | 2D→3D mechanism | Output | Central limitation |
| --- | --- | --- | --- | --- | --- |
| [Multi-view Performance Capture of Surface Details](https://pmc.ncbi.nlm.nih.gov/articles/PMC6979538/) (Robertini et al., IJCV 2016) | P, peer reviewed | Synchronized calibrated RGB, coarse tracked mesh | Quad-tree image Gaussians are matched to projected surface Gaussians; vertices move only along their normals under appearance, spatial, and temporal terms | Refined mesh carrying surface Gaussians | It refines known topology and coarse motion; it is not free 3D reconstruction |
| [Projective Reconstruction of Ellipses from Multiple Images](https://doi.org/10.1016/j.patcog.2009.07.003) (Mai, Hung & Chesi, Pattern Recognition 2010) | P, peer reviewed | Multiple corresponding image ellipses | Multi-view projective geometry reconstructs a 3D ellipse, including partial/missing observations | Geometric 3D ellipse | No mixture identity, radiance, opacity, or 3DGS rendering |
| [Structure From Motion With Objects](https://openaccess.thecvf.com/content_cvpr_2016/html/Crocco_Structure_From_Motion_CVPR_2016_paper.html) (Crocco et al., CVPR 2016) | P, peer reviewed | Tracked object bounding ellipses | Factorizes affine cameras and 3D quadrics | Object-level quadrics and cameras | Object boxes are much coarser than Gaussian fields |
| [G²SR](https://arxiv.org/abs/2607.14470) (Gao, Sze & Karaman, 2026) | D, preprint | 2–3 posed RGB images; a learned reference-view 2D splat field and RGB optical flow | Tracks five sigma points, triangulates centers, initializes normal/scale from affine deformation, then minimizes projected-Gaussian Hellinger error | Metric thin 3D Gaussian surfels with color | Detector is task-trained; RGB/flow remain inputs; rejection creates holes; radiance quality is secondary |

### 3.1 The 2016 surface-detail precursor

Robertini et al. are an important conceptual ancestor because the paper makes Gaussian overlap the
bridge between images and a 3D surface. A coarse mesh vertex carries an isotropic 3D “Surface
Gaussian”; each calibrated frame is decomposed by a quad-tree into isotropic 2D “Image Gaussians.”
Closed-form spatial overlap and color similarity define data associations while the optimizer
moves each mesh vertex along its normal. Smooth spatial and temporal regularizers make detailed
performance capture stable, and the soft Gaussian overlap helps with partial occlusion.

What transfers is the use of footprint overlap, color, calibration, and robust visibility to make
correspondence differentiable. What does not transfer is observability: the mesh supplies topology,
surface normals, coarse depth, temporal correspondence, and a one-dimensional displacement space.
The authors report 2.7–11% optical-flow error improvement in their sequences, but also note that
few-camera capture needs stronger regularization. It cannot support a claim that arbitrary 2D
fields alone determine a 3D field.

### 3.2 G²SR: the closest literal solution

[G²SR](https://arxiv.org/abs/2607.14470), submitted 2026-07-16, separates the problem into a small
learned image-plane front end and an analytic geometric back end:

1. A 0.41M-parameter detector tiles a reference 384×512 image with colored, oriented 2D Gaussian
   splats. During detector training, splat order is randomized, discouraging opaque painter-order
   tricks and encouraging boundary-respecting primitives whose appearance is stable enough to
   track.
2. Off-the-shelf NeuFlowV2 tracks five sigma points per splat—the center and positive/negative
   principal-axis endpoints—to each other view. A least-squares affine map summarizes local
   deformation.
3. Tracks are rejected for out-of-bounds warps or forward/backward flow disagreement above three
   pixels. Direct linear transform triangulates the center. The affine mapping initializes a
   surface normal; projected footprint matching initializes tangent scale.
4. Up to 20 parallel Gauss–Newton iterations minimize the summed squared Hellinger distance
   between each observed 2D Gaussian and the projection of a metric 3D surfel. A small fixed normal
   thickness avoids the unrestricted-covariance ambiguity.
5. Color is view independent. There is no subsequent scene-specific neural training or long 3DGS
   densification loop.

Only the splat detector is trained, on RealEstate10K; the flow network is used off the shelf and no
fine-tuning is performed on Replica, ScanNet, or DTU. This is a compelling architectural result:
learn the ambiguous tasks—compact boundary-aware detection and correspondence—and solve calibrated
geometry explicitly.

#### Reported geometry, memory, and throughput

All values in the next two tables are from the G²SR paper on one RTX 4090 and an i9-14900K. Images
are 384×512 for depth; the DTU mesh protocol uses 480×640. Baseline depths are median-aligned to
ground truth, whereas G²SR is evaluated at its unaligned metric scale, which the authors describe
as an advantage granted to the baselines.

| Inputs / benchmark | Metric | G²SR | Peak allocated GPU memory | Reconstruction rate |
| --- | --- | ---: | ---: | ---: |
| 2 views, Replica | AbsRel | 4.0% | 115.4 MB | 89.4/s |
| 2 views, ScanNet | AbsRel | 6.8% | 115.4 MB | 91.1/s |
| 3 views, Replica | AbsRel | 4.4% | 203.2 MB | 71.5/s |
| 3 views, ScanNet | AbsRel | 7.7% | 203.2 MB | 72.2/s |
| 3 views, DTU TSDF mesh | Chamfer distance | 3.23 mm | 203.7 MB | 68.9/s, meshing excluded |
| 3 views, DTU Poisson on centers | Chamfer distance | 3.50 mm | 203.7 MB | 68.9/s, meshing excluded |

The paper reports 5–107× lower peak allocated GPU memory and 2.3–18× higher reconstruction rate
than the tested few-view neural baselines. Those factors are protocol-specific and do not include
camera-pose estimation or meshing.

#### Reported three-view rendering quality

Metrics are computed only where rendered opacity is at least 0.5; coverage is reported separately.
This matters because missing pixels do not lower PSNR, SSIM, or LPIPS.

| Method | Replica PSNR / SSIM / LPIPS | ScanNet PSNR / SSIM / LPIPS | Coverage Replica / ScanNet |
| --- | --- | --- | ---: |
| G²SR | 24.07 / .781 / .417 | 22.71 / .750 / .472 | 85.8% / 73.1% |
| MVSplat | 25.04 / .837 / .194 | 23.77 / .787 / .309 | 96.6% / 94.2% |
| FreeSplat | 30.53 / .906 / .137 | 26.57 / .821 / .241 | 97.4% / 94.5% |

G²SR therefore reports a strong **metric geometry, memory, and latency** point. It does
not establish best radiance-field quality. Aggressive track rejection protects geometry but loses
object boundaries and occluded regions; view-independent color and approximate 2D splats lose fine
texture. The authors suggest accumulating more views to recover coverage. For a quality-first
system, the geometry should be treated as an initializer or scaffold followed by coverage recovery
and appearance refinement.

### 3.3 What remains open after G²SR

The broad statement “no paper lifts 2D Gaussian splats into 3DGS” is obsolete as of the cutoff. A
more precise, search-qualified gap remains:

> No located method accepts arbitrary, independently fitted, sparse GaussianImage/Image-GS-style
> fields as its only observations, resolves cross-view mixture identity, and produces a
> high-coverage, high-fidelity 3D alpha-composited radiance field without returning to the source
> RGB images.

G²SR narrows the plausible solution space: direct analytic lifting works when the 2D
representation is trained for stable tracking, carries local affine shape, uses calibrated poses,
and adopts a surface prior. It simultaneously warns that image compression PSNR, radiance quality,
and geometric completeness are not interchangeable.

## 4. Stage-1 image fields: fidelity and speed before lifting

These methods produce the right *type* of 2D object but do not reconstruct 3D. Their evidence
answers how cheaply and accurately a view can be represented; it does not establish cross-view
repeatability.

| Work | Representation and allocation | Reported convergence/performance | Relevance and missing evidence |
| --- | --- | --- | --- |
| [GaussianImage](https://arxiv.org/abs/2403.08551) (ECCV 2024) | Additive colored Gaussians; center plus Cholesky covariance; fixed count; L2 optimization | Kodak, 70K Gaussians: 44.08 dB, 106.59 s, 2,092 FPS, 419 MiB on V100; 50K training steps | Establishes high-fidelity order-independent image fields. No multi-view correspondence or lift |
| [Image-GS](https://arxiv.org/abs/2407.01866) (SIGGRAPH 2025) | Gradient/uniform position sampling, inverse-scale parameterization, top-K normalized rendering, residual-guided additions | Authors' 45-image set: full method 31.77±4.73 dB vs random init 29.54±4.15; 95% of final quality before 400 steps and 99% before 2K; 10K Gaussians/1K steps at 2K² in 18.74 s on A6000 | Strong adaptive allocation and early convergence. Image residual, not multi-view geometric error, controls growth |
| [Instant GaussianImage](https://arxiv.org/abs/2506.23479) (ICCV 2025) | A network predicts count and Gaussian attributes, followed by optional fine-tuning | On three highlighted DIV2K images at 2 s: 46.68/37.41/36.79 dB vs random optimization 35.14/31.96/32.25; about 3–4 GB total memory in the reported setup | Useful amortized initializer; illustrative images are not a universal aggregate and geometry transfer is untested |
| [Fast 2DGS](https://arxiv.org/abs/2512.12774) (WACV workshop 2026) | Lightweight deep Gaussian prior and attribute network, then little tuning | Abstract reports over 1,000 FPS and contrasts a forward pass/minimal tuning with more than 10 s random fitting | Promising low-latency field creation; downstream lift is untested |
| [GaussianImage++](https://arxiv.org/abs/2512.19108) (2025 preprint) | Distortion-driven densification, context-aware filters, quantization | Improves rate/distortion and compact image representation in its protocol | Better compression does not imply more identifiable components |
| [SGI](https://arxiv.org/abs/2603.07789) (2026 preprint) | Learned seeds and MLP attributes with multiscale fitting for very large images | Authors report 1.6–6.5× faster optimization and up to 7.5×/1.6× compression over non-quantized/quantized baselines | Scale and speed insight; no multiview or field-lift result |
| [Contour Information Aware 2D Gaussian Splatting](https://arxiv.org/abs/2512.23255) (2025/2026) | Assigns each Gaussian to a segmentation region; region-constrained rasterization after warm-up | Improves low-budget boundary fidelity on synthetic charts and DAVIS in the authors' experiments | Most relevant masked-field design; it uses region labels, not 3D correspondence |

### 4.1 GaussianImage and Image-GS are different observation contracts

GaussianImage uses order-independent additive accumulation, fixes raster opacity to one in its
official Cholesky model, and directly optimizes the colored feature. Its paper reports that
accumulation outperforms alpha blending by about 0.8 dB while being faster, and that plain L2 is
the strongest tested PSNR objective. The output is a compact continuous image function, not an
occlusion model.

Image-GS emphasizes where primitives should be placed. It samples initial centers from a mixture
of gradient magnitude and uniform density, initializes colors from pixels, starts with half the
target count, and adds one eighth every 500 steps at high-error locations. It reports fast early
quality and strong gains over random placement. The lesson for lifting is allocation-aware
initialization—not that its normalized top-K renderer or amplitudes are physically compatible with
3DGS.

### 4.2 The downstream objective Stage 1 is missing

All of the following can increase image PSNR while harming liftability:

- placing a long Gaussian across an occlusion or semantic boundary;
- using different split patterns for the same texture in different views;
- allocating many tiny Gaussians to sensor noise or view-dependent highlights;
- permitting weight/color or scale/amplitude gauges that change downstream ranking;
- fitting covered/occluded pixels whose component has no valid partner in another view.

A geometry-aware field fitter should therefore measure at least:

- repeatability of center and affine footprint under known homographies or optical flow;
- agreement of projected component tracks across three or more views;
- boundary leakage and foreground/background confusion;
- conditioning of triangulated means and covariance recovery;
- useful coverage per retained component, not image PSNR per component alone.

The contour-aware paper and G²SR detector point in the same direction: prevent cross-boundary
blending and make components stable under reordering/tracking. This is a mechanism hypothesis, not
yet a demonstrated result for independently optimized teacher fields.

## 5. Adjacent reconstruction families

### 5.1 Dense correspondence and initialization

[EDGS](https://arxiv.org/abs/2504.13204) is the most important adjacent convergence result. It
matches dense image locations with RoMa, triangulates 3D points, initializes color and spherical
harmonics, and disables adaptive densification. Its primitive per observation is a point match,
not a 2D Gaussian with covariance, but it shows in its protocol that spending computation on dense, well-placed
geometry can remove a large part of the slow 3D optimization phase.

The default paper configuration samples 20K correspondences per reference image for 180 reference
views and two neighbors. On an A100, its reported initialization takes about 120 s—76 s matching,
11 s triangulation, and 15 s spherical-harmonic initialization—and peaks around 15 GB. The
MIP-NeRF 360 table reports:

| Method / steps | Total training time | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| EDGS, 5K | 8 min | 26.88 | .825 | .166 |
| EDGS, 10K | 12 min | 27.54 | .834 | .154 |
| EDGS, 30K | 27 min | 28.02 | .839 | .141 |
| Reproduced standard 3DGS, 30K | 26 min | 27.49 | .816 | .215 |

The paper's headline is that EDGS reaches the original 3DGS quality in **15% of the training
time**, and longer training gives 35% lower LPIPS in the paper's comparison. The 120 s
initialization is not free, and the result should not be shortened to “instant.” It also depends on
a large learned matcher and many RGB correspondences.

[The Role of Initialization in 3D Gaussian Splatting](https://arxiv.org/abs/2603.20714) evaluates
six initialization methods with five density controllers. Its central warning is that dense
initialization does not consistently improve novel-view synthesis on well-constrained scenes when
strong densification later reallocates the model. Dense geometry matters more off the camera
trajectory, in sparse settings, and for geometric metrics. The correct experimental unit is
therefore **initialization × density controller**, not initialization alone. Ordinary clone/split
or MCMC behavior can erase an initially compact, geometric field.

For a 2D-field lift this suggests a practical rule: evaluate no densification, a short
coverage-repair phase, and a strong conventional controller separately. If a field-derived
initializer only helps the first two, that is still a useful convergence result, but not a claim
of universal initialization superiority.

### 5.2 Pixel-aligned image-to-Gaussian predictors

These methods look superficially direct because an image grid or feature map emits Gaussians, but
the fitted 2D Gaussian field is not their input.

| Work | Pose assumption | Geometry mechanism | Allocation | What transfers to field lifting |
| --- | --- | --- | --- | --- |
| [Splatter Image](https://arxiv.org/abs/2312.13150) (CVPR 2024) | Single image or posed multiview variants | A 2D network predicts one 3D Gaussian per input pixel; unseen geometry is learned prior | Dense pixel aligned | Demonstrates 38 FPS prediction and 588 FPS rendering in its setup; single-view output is necessarily prior dominated |
| [GPS-Gaussian](https://arxiv.org/abs/2312.02155) (CVPR 2024 Highlight) | Two adjacent rectified source views selected for a target view | Iterative binocular stereo predicts depth; depth unprojects pixel-wise position maps while a network regresses rotation, scale, and opacity maps | One Gaussian per foreground pixel in each source view | A real masked 2D-parameter-map→3DGS system and an amortized speed ceiling; it needs human-specific training, ground-truth depth supervision, accurate matting, and source RGB rather than pre-fitted Gaussian mixtures |
| [pixelSplat](https://arxiv.org/abs/2312.12337) (CVPR 2024) | Image pairs with known cameras | Predicts a depth probability distribution along each ray and samples Gaussian means | Pixel aligned | Treat depth as a distribution; do not collapse ambiguous rays too early |
| [MVSplat](https://arxiv.org/abs/2403.14627) (ECCV 2024) | Posed sparse views | Plane-sweep cost volume plus cross-view features | Pixel aligned | Explicit multiview matching; authors report 22 FPS, over 2× pixelSplat speed and about 10× fewer parameters |
| [DepthSplat](https://arxiv.org/abs/2410.13862) (CVPR 2025) | Posed views | Monocular depth-foundation features plus multiview cost volume | Pixel aligned | Geometry prior and multiview evidence are complementary; authors report 12 views at 512×960 in 0.6 s |
| [FreeSplatter](https://arxiv.org/abs/2412.09573) (ICCV 2025) | Unposed sparse views | Transformer predicts pixel-aligned Gaussian maps and recovers cameras | Pixel aligned | Joint pose/scene inference and an object variant with foreground masks; model prior fills unseen content |
| [G3Splat](https://arxiv.org/abs/2512.17547) (2025 preprint) | Sparse multiview | Adds ray and local-surface priors to photometric learning | Pixel aligned | Critical negative lesson: photometric-only scales/orientations need not have geometric meaning |

GPS-Gaussian therefore **does fit this review, but as an adjacent masked feed-forward route**. Its
“2D Gaussian parameter maps” are dense predicted property maps, not a sparse mixture of elliptical
image primitives. The paper trains its depth module for 40K iterations and then jointly trains the
depth and Gaussian regressors for 100K iterations (about 15 hours in the reported setup); no
per-subject optimization is needed afterward. It reports more than 25 FPS at 2K and a runtime
breakdown of 27 ms for source processing plus 0.8 ms per novel view on an RTX 3090. The same paper
states that accurate foreground matting is necessary preprocessing and ground-truth depths are
required during training. It is consequently strong evidence for the quality/speed available to a
learned, masked, human-specific pipeline, but not evidence for unmasked general scenes or an
image-free converter of independently fitted fields.

The G3Splat result is especially relevant. A renderer can achieve plausible held-out RGB with
anisotropies that do not correspond to surfaces. Any claim that 2D covariance has been “lifted”
must therefore include geometry, normals, or reprojection-shape evaluation, not PSNR alone.

### 5.3 From pixels to adaptive 3D anchors, tokens, and queries

The 2026 frontier is moving away from one-Gaussian-per-pixel because that allocation scales with
input resolution and view count rather than scene complexity:

- [Off The Grid](https://openaccess.thecvf.com/content/CVPR2026/html/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.html)
  predicts subpixel primitive locations and varies count per patch using Shannon entropy. Its
  project reports roughly seven times fewer primitives and a 12-image reconstruction in about one
  second. The transferable principle is content-adaptive detection; its predictor still consumes
  RGB/features, not fitted fields.
- [AnchorSplat](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html)
  seeds the output with 3D geometric anchors such as sparse points, voxels, or RGB-D points, then
  runs a small number of forward refinements. This breaks output count away from the image grid,
  but makes quality depend on anchor accuracy and coverage.
- [TokenSplat](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TokenSplat_Token-aligned_3D_Gaussian_Splatting_for_Feed-forward_Pose-free_Reconstruction_CVPR_2026_paper.html)
  aligns features across views at token level and uses separate camera tokens and asymmetric
  decoding for pose-free reconstruction. It is evidence for cross-view representation alignment,
  not for parameter-space matching of fitted mixtures.
- [C3G](https://arxiv.org/abs/2512.04021) uses about 2K Gaussians and is an important compact
  query-based baseline. G²SR's own rendering table shows that compactness alone does not ensure
  good radiance quality.
- [ATSplat](https://arxiv.org/abs/2607.20417), a 2026-07 preprint, lifts coarse patch-level depth
  and camera cues to sparse 3D tokens, then expands high-uncertainty tokens. The authors report
  fewer than one second for 12 images at 512×960, 311K Gaussians, 1,136 FPS rendering, and over
  5.7× fewer Gaussians than dense feed-forward methods in their protocol.
- [QuerySplat](https://arxiv.org/abs/2608.01186), submitted only two days before this cutoff,
  explicitly separates a geometry branch using a pretrained vision-geometric model from an
  appearance branch. The authors report +2.30 dB over their best pose-free baseline and +1.04 dB
  over their best pose-required baseline on DL3DV. This is very recent preprint evidence, not an
  independently established default.

Across these methods, the durable design signals are: allocate by scene uncertainty rather than
pixels, initialize from explicit geometry, align cross-view features before prediction, and
separate geometry from high-frequency appearance. Their headline inference time excludes model
training and may include large pretrained backbones; peak VRAM, model weights, pose estimation,
and preprocessing must be counted before comparing them to G²SR or optimization-based methods.

### 5.4 “2DGS” surface reconstruction is already 3D

[2D Gaussian Splatting](https://arxiv.org/abs/2403.17888) replaces volumetric ellipsoids with
planar disks embedded in 3D and introduces a perspective-correct splatting and geometry objective.
Sparse2DGS, SparSplat, SurfelSplat, and related methods improve sparse-view surfaces using the same
kind of embedded surfel. They inform the *output constraint*—thin tangent-plane Gaussians are more
identifiable and mesh friendly—but they do not convert an `R²` image mixture to `R³`.

This naming collision is consequential for masked reconstruction: a paper saying it reconstructs
an object “based on 2DGS” may be optimizing 3D surfels from RGB and masks, not lifting image-plane
Gaussian components.

### 5.5 Cross-domain exact and near-exact precedents

Searching only for “Gaussian Splatting” misses much of the closest prior art. Astronomy,
statistical tomography, cryo-EM, target tracking, navigation, and silhouette reconstruction have
each solved a different part of the pipeline. The following table separates **novelty threats**
from **mechanism donors**; neither label means that a method can be dropped into an alpha-composited
perspective renderer unchanged.

| Field and primary work | Relation to 2D↔3D Gaussian fields | Role here | Preserved mechanism | Broken correspondence |
| --- | --- | --- | --- | --- |
| Astronomy: [Multi-Gaussian Expansion](https://arxiv.org/abs/astro-ph/0201430) (MGE; Cappellari 2002, building on 1990s work) | Fits a 2D surface-brightness image with Gaussians; PSF convolution and 3D luminosity-density deprojection are analytic for an assumed viewing geometry and intrinsic family | Strong restricted prior art; fast seed/baseline | Gaussian closure under convolution and line-of-sight marginalization | Usually concentric components, distant/orthographic additive light, prescribed inclination and axisymmetry/triaxiality; no occlusion, color, or alpha compositing |
| Statistics/cryo-EM: [random tomography](https://arxiv.org/abs/0909.0349) and its [sparse practical continuation](https://arxiv.org/abs/1202.6475) | Recovers a 3D radial mixture's center geometry from finitely many noisy 2D mixture projections, even with unobserved random view angles | Strong novelty threat and analytic initialization donor | Mixture weights persist through projection; projected center Gram matrices carry 3D shape information | Many random additive projections, radial/common kernels, restrictive labeling assumptions, recovery only up to an orthogonal transform; no perspective visibility or anisotropic radiance |
| Cryo-EM: [e2gmm](https://www.nature.com/articles/s41592-021-01220-5) | Maps a 3D GMM into 2D particle images at known orientations and learns continuous conformational variation | Mixed-dimensional GMM prior art and optimization donor | Compact differentiable 3D mixture with many projected observations | Raw additive density images, very many particles, molecular priors, and no alpha occlusion or ordinary scene appearance |
| X-ray CT: [R²-Gaussian](https://arxiv.org/abs/2405.20693), [exact Gaussian ray tracing](https://arxiv.org/abs/2602.01057), and [FaCT-GS](https://arxiv.org/abs/2604.01844) | Optimizes 3D anisotropic Gaussians from 2D detector projections with corrected or exact forward operators and fast voxelization | Projection-fidelity oracle, warm-start, and systems donor | Amplitude must obey the measurement operator; analytic Gaussian projection can make reconstruction fast | X-ray measurements are additive line integrals, not front-to-back alpha compositing; raw projections are not fitted 2D mixture components |
| Mixture geometry: [mixture Wasserstein](https://arxiv.org/abs/1907.05254), [JRMPC](https://arxiv.org/abs/1609.01466), and [Coherent Point Drift](https://arxiv.org/abs/0905.2635) | Softly couples components or observations without fixed indices; multi-marginal transport treats all views symmetrically | Association donor | Component-level Gaussian costs, soft responsibility, outlier/missed-detection handling, joint latent mixture | Native problems use same-dimensional Euclidean registration; camera projection, occlusion, component split/merge, and appearance must be added |
| Navigation/tracking: [nonlinearity-informed moment-matching splits](https://arxiv.org/abs/2412.00343) and [KL-guided mixture reduction](https://www.cs.kent.ac.uk/pubs/2007/2797/) | Selectively splits a Gaussian when a nonlinear map invalidates a single-Gaussian approximation, and merges components while preserving low moments | Quality/count controller donor | Spend components only where uncertainty and transform curvature demand them; conserve mass, mean, and covariance during topology changes | Moment conservation under an additive probability density does not guarantee invariant alpha rendering or visibility |
| Shape from silhouette: [Shape from Silhouette Probability Maps](https://openaccess.thecvf.com/content_cvpr_2013/html/Tabb_Shape_from_Silhouette_2013_CVPR_paper.html) | Reconstructs occupancy from binary or continuous foreground maps while treating false-positive and false-negative error symmetrically | Mask robustness donor | Delay hard labels; optimize agreement with all probability maps rather than hard-intersecting every silhouette | Voxel occupancy is not surface radiance, and silhouette evidence still cannot reveal hidden concavity |
| Cryo-EM validation: [gold-standard independent-half reconstruction](https://www.nature.com/articles/nmeth.2115) | Refines two reconstructions from disjoint data halves and compares them to expose overfitting | Evidence/measurement donor | Shared structure should survive independent reconstruction; shared initialization/reference can create spurious agreement | A camera split changes geometric coverage and visibility, so agreement is a stability diagnostic rather than a resolution certificate |
| Iterative tomography and online reconstruction: [ordered-subsets EM](https://pubmed.ncbi.nlm.nih.gov/18218538/) and [Incremental Online Scene Reconstruction](https://arxiv.org/abs/2607.10690) | Uses view blocks for rapid early updates; freezes fully optimized historical regions | Convergence scheduling donor | Avoid repeatedly spending full cost on already stable evidence | Nonconvex alpha rendering can cycle or lock in wrong early geometry; a final all-view cleanup and thaw rule are required |

#### Prior art that materially narrows novelty

MGE is the closest old **analytic 2D-Gaussian-mixture→3D-Gaussian-mixture** construction. For an
oblate, axisymmetric component with observed axial ratio `q'`, assumed inclination `i`, dispersion
`σ`, and luminosity `L`, the intrinsic flattening is fixed by

```text
q² = (q'² - cos² i) / sin² i,
```

and the 2D Gaussian deprojects to a 3D Gaussian with the same total luminosity. The construction is
defined only when the observed component is compatible with the assumed inclination; the flattest
component can therefore rule out candidate inclinations. Cappellari explicitly notes that general
deprojection is non-unique: MGE supplies one smooth solution under a structural prior. Gaussian PSF
convolution is also analytic, making MGE a useful **restricted baseline** for the repository: if an
axisymmetric additive synthetic case cannot match its closed-form seed, a more elaborate lifter has
an implementation or semantics problem.

Panaretos' random-tomography result is broader in component position and more restrictive in kernel
shape. It models the unknown 3D density as a finite mixture of radial location densities. Each 2D
profile remains a mixture with the same component weights, and under isotropically random rotations
the expected projected Gram matrix is a scaled version of the 3D center Gram matrix. This permits
consistent recovery of mixture shape without first estimating every viewing angle. The basic
labeling argument relies on distinguishable component weights; the practical continuation uses
LASSO deconvolution, clustering, weight/Gram-based labeling, and averaging, and documents projected
component collisions and flat/local-optimum likelihoods. It is direct evidence that **broad mixture
deprojection is not new**, and that label consistency—not covariance algebra—is the hard practical
step.

e2gmm supplies a modern differentiable counterpart: a compact 3D GMM is repeatedly projected into
known-orientation 2D cryo-EM images while a latent neural model captures structural variability.
It does not accept per-image fitted GMMs, but it invalidates any claim that optimizing one 3D GMM
against mixed-dimensional 2D observations is unique to 3DGS. The remaining defensible delta is the
combination of few calibrated perspective views, anisotropic footprints, partial visibility,
appearance and alpha semantics, and independently componentized input fields.

#### Mechanisms that can improve quality, speed, or convergence

**Use the correct forward operator before tuning the optimizer.** R²-Gaussian derives a
covariance-dependent amplitude rectification because a unit-peak EWA footprint does not conserve
X-ray line-integral density. It also sums contributions instead of alpha compositing. The exact-ray
tracing follow-up analytically integrates a full anisotropic Gaussian along arbitrary rays, avoiding
local affine projection error; FaCT-GS contributes optimized rasterization/voxelization and rapid
Gaussian fitting to a coarse reconstructed volume for warm starts. For realtime-gs, these are not
drop-in renderers. They support two tests: an exact or high-sample projection oracle for diagnosing
when the local Jacobian is inaccurate, and an explicit conversion layer between source mass and
target opacity. Cross-domain runtime numbers cannot be transferred because the measurement and
hardware protocols differ.

**Replace brittle one-to-one matching with sparse soft mixture coupling.** Delon and Desolneux
restrict Wasserstein couplings to Gaussian mixtures, reducing the comparison to a small discrete
transport problem over components and providing multi-marginal/barycenter formulations. JRMPC
treats multiple point sets symmetrically as observations of a shared latent GMM; CPD uses EM soft
responsibilities and explicitly handles outliers and missing points. A recipient implementation
would gate candidate pairs by epipolar geometry, then solve a capped multi-view transport with a
background/dustbin mass. It must permit one-to-many mass when independently fitted fields split the
same region differently. The killing test is simple: at matched track budget, transport must improve
track precision×coverage or held-out geometry over mutual-nearest sigma-point matching. Otherwise
its cubic/iterative overhead is unjustified.

**Split for projection nonlinearity, then merge by conserved evidence.** A single 3D Gaussian can
project non-Gaussianly when its support spans appreciable depth under perspective. Kulik and
LeGrand's navigation method preserves the original mean and covariance while choosing a split
direction from both uncertainty and nonlinear-transform structure, with whitening to remove unit
dependence. Runnalls' reduction replaces selected pairs with an exactly second-moment-matched
Gaussian and chooses merges using a computable KL upper bound. The proposed transfer is to estimate
Jacobian variation or sigma-point versus exact-projection error over each lifted Gaussian, split only
where that error predicts held-out residual, and later merge only when source-view mass/moments and
renderer output remain within tolerance. This is a candidate count–quality controller, not evidence
that moment preservation also preserves alpha compositing.

**Separate geometric support mass from appearance opacity.** The Scholar Inbox pass surfaced the
very recent [Manifold-GS](https://arxiv.org/abs/2608.00214) preprint, which explicitly distinguishes
appearance opacity from geometric quadrature mass and transports the latter conservatively during
refinement. This is conceptually aligned with source-mixture mass conservation and with MGE/Runnalls,
but its evidence covers only three DTU scenes and a certified-asset objective. The transfer should be
tested as bookkeeping for split/merge stability, not adopted as an established reconstruction
default.

**Use ordered evidence and freezing only as reversible accelerators.** Ordered-subsets EM processes
blocks of projection views and reported an order-of-magnitude early acceleration for its SPECT
simulation setting. The 2026 online-Gaussian work freezes optimized historical regions to reduce
long-sequence compute. A 3DGS analogue should use angularly balanced view blocks early, freeze only
well-conditioned low-residual primitives, periodically thaw them, and finish with all views. The
native baseline is ordinary randomized camera minibatching; if ordered blocks or freezing improve
only iteration count but not wall-clock time-to-held-out quality, the added state machinery should
be rejected.

#### A validation transfer, not another optimizer

Cryo-EM's independent-half protocol suggests a strong stability audit. Stratify cameras by angle
and baseline into two disjoint but coverage-matched sets, construct two 3D fields independently,
and compare their overlapping geometry, projected covariance spectra, and rendered alpha support.
Do not share learned scene parameters, correspondences, or a refined 3D seed between halves. High
agreement raises confidence that the result is data-supported; disagreement localizes unstable
regions. Because each camera half sees different surfaces, this cannot be called a 3DGS resolution
measure and should never replace held-out RGB/depth/geometry evaluation.

## 6. Masked inputs

A calibrated binary silhouette defines a generalized visual cone. Intersecting cones produces a
visual hull: an outer occupancy bound that can be computed before photometric optimization. It is
excellent for suppressing background and impossible depths, but it cannot reveal concavities that
do not change any silhouette. False-negative mask pixels can destroy valid geometry under hard
intersection; false positives enlarge the hull.

[Shape from Silhouette Probability Maps](https://openaccess.thecvf.com/content_cvpr_2013/html/Tabb_Shape_from_Silhouette_2013_CVPR_paper.html)
is a useful pre-3DGS corrective. It treats binary masks as a degenerate case of continuous
probability maps and optimizes a voxel labeling whose reprojections penalize false-positive and
false-negative disagreement symmetrically. The paper targets thin, textureless objects under mask
and camera-calibration error, precisely where visual-hull intersection is brittle. Its voxel solver
is not a Gaussian renderer, but its evidence supports soft mask likelihoods and leave-one-view-out
boundary checks rather than irreversible carving.

### 6.1 GaussianObject: visual-hull initialization plus repair

[GaussianObject](https://arxiv.org/abs/2402.10259) reconstructs an object from as few as four
calibrated RGB images and masks. It samples points in a 3D volume, retains points whose projections
lie inside all masks, averages observed colors, initializes scale from neighbor spacing, and then
optimizes RGB, mask BCE, and monocular-depth losses. K-nearest-neighbor floater elimination and a
learned visual repair stage address sparse-view artifacts.

The paper reports roughly one minute for the coarse representation at 779×520 and about 30 minutes
end to end on an RTX 3090: approximately one minute initialization, 15 minutes repair-model setup,
and 14 minutes repair. Four-view results include 24.81 dB/.935/4.98 LPIPS×100 on its MIP-NeRF 360
object protocol and 30.89 dB/.9756/2.07 on OmniObject3D. An ablation without visual-hull
initialization reports 15.95 dB versus 24.81 dB for the complete configuration, but that gap is
within one multi-component pipeline and should not be attributed to the hull alone.

The transferable mechanism is strong: use masks to bound allocation before optimization. The
limitations are equally important: the intersection is only a hull, four masks can be noisy, and
the long learned repair stage—not the Gaussian lift—provides much of the final visual quality.

### 6.2 Probabilistic object-level Gaussian splatting

[Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting](https://arxiv.org/abs/2603.14316),
a 2026 preprint, uses continuous YOLO/SAM foreground probabilities rather than treating binary
masks as exact. It filters SfM points using projected foreground probability, rejects poor views,
adds a learned foreground probability to each 3D surfel, and trains with mask-weighted RGB, depth,
normal, and probability losses. Low-probability/low-opacity primitives are pruned. After 7K
iterations the method replaces external SAM supervision with its rendered, cross-view-consistent
model mask to sharpen boundaries.

The paper's reported evidence includes:

| Protocol/result | Reported value | Interpretation boundary |
| --- | ---: | --- |
| NVOS object segmentation | 90.1 mIoU, 98.4 mAcc, 1,130 s | SAGA reports 90.9/98.3 and 3,320 s, but the pipeline scopes differ |
| Synthetic mask ablation | 56.71 binary → 84.33 soft probability → 93.47 with mask replacement → 95.63 with data refinement mIoU | Strong within-paper evidence for uncertain masks and self-consistent refinement |
| Truck primitive count | About 1/4 of standard 2DGS and about 1/10 of 3DGS | Scene-specific compression, not a universal ratio |
| Example training times | 25.3 vs 27.6 min and 22.0 vs 30.3 min | Modest-to-material savings depending on object/scene |

Here “2DGS” means planar 3D surfels. The paper does not lift GaussianImage fields, but its mask
handling is directly reusable: soft probability, quality filtering, foreground-aware pruning, and
late self-consistency are safer than an immutable hard silhouette.

### 6.3 A mask-aware field-to-field design

For **quality-first masked reconstruction**, the literature supports this staged design:

1. calibrate cameras and preserve a soft foreground probability plus an uncertainty band around
   the contour;
2. fit boundary-respecting 2D fields, preventing components from crossing foreground/background
   or semantic-region boundaries;
3. use a visual hull with slack as an outer depth bound, not as the reconstructed surface;
4. track centers and footprint axes across views; triangulate only well-conditioned consensus
   tracks and fit thin surface covariance;
5. maintain separate confident-core, uncertain-boundary, and optional background populations;
6. recover rejected/occluded coverage using depth/point-map anchors or uncertainty-driven local
   additions;
7. calibrate alpha and appearance with a short multiview render optimization, then prune by both
   support probability and rendered contribution.

For **speed-first masked reconstruction**, stop after visual-hull filtering, G²SR-like analytic
lifting, and a very short opacity/color calibration. This avoids fitting background and can be
extremely compact. The accepted losses are concavity, hidden-side completeness, fine specular
appearance, and holes where tracks are rejected.

Failure cases that must be tested explicitly include hair/thin structures, mask flicker, holes in
the foreground mask, transparent or reflective objects, severe self-occlusion, and masks that
include turntable/support geometry. Erode/dilate sweeps and boundary-F scores reveal whether an
apparent quality gain is merely a favorable mask threshold.

## 7. Unmasked inputs

Without masks, the system must infer both scene support and foreground/background allocation.
Three tempting shortcuts are invalid:

- additive field amplitude is not occupancy;
- an image-space gap in one view is not empty space in 3D;
- triangulating every textured component produces floaters on occlusions, reflections, sky, and
  independently moving objects.

### 7.1 Geometry-first unmasked reconstruction

The strongest evidence-backed path is:

1. obtain calibrated poses or estimate them jointly;
2. match each component using center plus affine footprint information, with RGB/feature flow as a
   fallback if the strict “field-only” constraint is relaxed;
3. triangulate means with baseline/condition-number checks and multi-view robust consensus;
4. fit a thin surface covariance from three-view projected shapes, or use two views with an
   explicit normal/thickness prior;
5. reject inconsistent tracks, but keep an uncertainty/coverage map so rejection triggers later
   recovery rather than silently improving valid-pixel metrics;
6. initialize appearance independently of alpha and run a short held-out-view render refinement;
7. represent far/unbounded content with a separate large-scale/background population or scene
   contraction rather than forcing foreground-scale splats onto the sky.

This is G²SR plus coverage and radiometry repair. It should be the default research baseline when
metric geometry, low memory, and low latency matter most.

### 7.2 Quality-first unmasked reconstruction

When RGB remains available, a pure field-only interface is unlikely to maximize final quality.
Quality-leading adjacent systems combine multiple cues: learned depth/point maps, plane-sweep or feature
correspondence, camera tokens/pose solvers, explicit 3D anchors, and separate appearance decoding.
A practical quality-first hybrid is therefore:

- field-derived, correspondence-tested surface anchors for compact geometry;
- dense depth or point-map anchors for textureless and rejected regions;
- a small uncertainty-driven residual population for high-frequency appearance;
- explicit background/far geometry;
- short joint refinement, with density growth restricted to missing-coverage regions.

The critical ablation is RGB/depth assistance versus field-only input at equal final primitive
count and wall-clock. Otherwise the fitted 2D field may be decorative rather than causally useful.

### 7.3 Single image

No deterministic method can recover unseen geometry or disoccluded appearance from a single 2D
field. Splatter Image and modern foundation-model predictors can produce a fast plausible 3DGS,
but the output is conditional generation from learned priors. A single-view result must be scored
for plausibility and visible-view consistency, not described as faithful reconstruction of
unobserved surfaces.

## 8. Performance and convergence engineering

### 8.1 Optimize total time, not only 3D iterations

Use the wall-clock decomposition

```text
T_total = T_pose + T_mask + T_2D-fit + T_track/depth + T_lift + T_3D-refine + T_export.
```

An instant analytic lift after a two-minute per-view field fit is not an instant pipeline. A
feed-forward predictor may have subsecond scene inference but multi-gigabyte weights and substantial
offline training. EDGS removes densification but spends about two minutes on dense initialization.
G²SR's reconstruction rate excludes camera-pose estimation and meshing. Report both cold and warm
runs, preprocessing, model loading, and peak end-to-end VRAM.

### 8.2 Three empirically supported convergence levers

**Better placement before optimization.** Image-GS shows that gradient/residual allocation speeds
2D fitting; EDGS shows dense triangulated placement accelerates 3D convergence; G²SR largely
solves geometry without scene optimization. This is the highest-confidence lever.

**Match density control to the initializer.** The initialization study shows that strong
densification can erase the advantage of a dense or geometric start. Evaluate fixed topology,
brief targeted additions, and a conventional controller. Do not tune the initializer with one
controller and generalize the result to all 3DGS training.

**Separate geometry from appearance.** G3Splat warns that photometric optimization can learn
non-geometric covariance. G²SR solves geometry first; QuerySplat uses separate geometry and
appearance branches. Geometry-first initialization followed by a smaller appearance phase should
converge more predictably, but the claim needs a controlled time-to-quality experiment.

Two additional cross-domain levers are mechanistically strong but not yet established in 3DGS:

- use angularly balanced ordered view subsets for fast early updates, then finish with full-view
  cleanup so subset cycles cannot define the final solution;
- split only when perspective nonlinearity makes a single Gaussian projection inaccurate, merge by
  moment/mass-preserving criteria, and freeze stable primitives with periodic thaw checks.

Both must beat ordinary randomized camera sampling plus the native density controller in
wall-clock quality–time AUC. Iteration-count improvement alone is not sufficient.

### 8.3 Fast 3D optimization after lifting

Fast training systems are complementary only after their interaction with the initializer is
measured:

- [DashGaussian](https://arxiv.org/abs/2503.18402) uses a coarse-to-fine schedule and reports
  roughly 200 s per scene in its setup.
- [FastGS](https://arxiv.org/abs/2511.04283) combines fast raster/training ingredients. On an RTX
  4090 its MIP-NeRF 360 table reports 1.93 min, 27.56 dB/.797/.261 with 0.40M Gaussians for FastGS,
  and 3.58 min, 27.93/.820/.216 with 1.15M for FastGS-Big, versus 20.93 min,
  27.53/.812/.221 for its 3DGS baseline. The smaller configuration makes a visible quality trade.
- [Faster-GS](https://arxiv.org/abs/2602.09999) consolidates implementation-level acceleration and
  reports up to 5× faster optimization while maintaining quality in its tested protocols.
- Structure-aware or frequency-aware densification is a plausible coverage-repair tool because it
  can split according to projected footprint versus image detail. It remains a 3D refinement
  mechanism, not evidence that a source 2D covariance has physical meaning.

The safest integration sequence is: reproduce the fast trainer on its native initializer, insert
the field-derived initializer without changing other settings, then jointly tune only after the
interaction is visible. A faster renderer or optimizer does not rescue wrong correspondence or
unobservable depth.

### 8.4 Pareto choices supported by current evidence

| Primary objective | Best-supported family | Expected strength | Accepted weakness |
| --- | --- | --- | --- |
| Metric geometry and minimum inference memory | G²SR-style detector/track/analytic surfel fit | About 69–91 reconstructions/s in its detailed tables and 115–203 MB in its protocol; metric scale | Lower texture fidelity and coverage; posed multiview RGB required |
| Fast convergence to optimized 3DGS quality | EDGS-style dense correspondence init, no/short densification | Strong time-to-LPIPS and final LPIPS evidence | About 120 s/15 GB initialization in default setup; points rather than 2D fields |
| Highest amortized few-view NVS quality | Modern geometry-prior feed-forward models such as DepthSplat, Off The Grid, ATSplat, QuerySplat | Subsecond/seconds prediction and learned completion | Training-domain dependence, large backbones, hallucination, not field input |
| Compact feed-forward scene | C3G, AnchorSplat, Off The Grid, ATSplat | Count decoupled from pixels; adaptive allocation | Anchor/query coverage and texture can fail; protocols differ |
| Masked object speed/compactness | Visual hull or soft-probability pruning plus analytic lift | Avoids most background primitives | Concavity and mask-error limits |
| Masked object quality | GaussianObject-style repair or hybrid geometry plus short refinement | Better completion and boundaries | Minutes of optimization/repair; generative components may alter fidelity |
| Arbitrary pre-fitted field conversion | No established end-to-end winner | Clear research opportunity | Correspondence, covariance, alpha semantics, and coverage all unresolved |

This table recommends families, not drop-in code. Availability, license, resolution, dataset,
camera assumptions, and hardware must be checked for the intended deployment.

## 9. Recommended field-to-field architecture

This is a synthesis of mechanisms with primary evidence; the complete combination has not been
validated and must not inherit the component papers' claims.

### 9.1 Observation contract

Store, for every source component:

- center and a numerically stable SPD covariance;
- the exact renderer semantics: additive/normalized/alpha, low-pass filter, truncation, and color
  parameterization;
- support probability and boundary/region label when masks exist;
- a stable local descriptor derived from source image features or field neighborhoods;
- source camera, resolution, and fitting uncertainty;
- both a **localization covariance** for association and a **coverage covariance** for rendering if
  they differ.

The last separation is essential. A broad Gaussian may efficiently cover smooth color but imply a
very uncertain center; a sharp landmark can localize depth precisely but need a broader surface
footprint to prevent holes. One covariance cannot serve both roles by assumption.

### 9.2 Association

Use a coarse-to-fine track graph:

1. predict candidate locations with epipolar geometry, optical/feature flow, or reprojection of a
   depth/point prior;
2. score center distance, Hellinger or Wasserstein footprint distance, color/feature similarity,
   region consistency, and forward/backward cycle error;
3. match five sigma points or an equivalent local affine frame rather than centers only;
4. use an epipolar-gated, capped soft transport or EM responsibility matrix with an explicit
   background/dustbin state; allow one-to-many mass so one broad component can correspond to
   several fine components;
5. form multi-view tracks with robust consensus; keep unmatched components as coverage debt rather
   than forcing bad geometry.

Strict field-only association can use rendered field samples and neighborhood descriptors, but it
should be compared against RGB/feature flow. If field-only tracking loses substantially, source
pixels contain information the decomposition did not preserve.

### 9.3 Analytic geometric lift

For each accepted track:

1. triangulate `μ` with uncertainty and reject small-baseline/large-condition tracks;
2. subtract the known raster low-pass covariance before shape fitting where stable;
3. with two views, fit a surfel with fixed/small normal thickness and initialize normal from the
   affine track; with three or more views, also test a full SPD covariance under eigenvalue/aspect
   bounds;
4. minimize robust projected-distribution distance, preferably Hellinger for normalized Gaussian
   shape plus a separate support/amplitude residual;
5. fit all views jointly and report the residual, rank, and covariance condition number;
6. compare the local-Jacobian projection against sigma-point sampling or an exact/high-sample
   reference; moment-preservingly split a primitive only when the discrepancy is both material and
   predictive of held-out error;
7. fuse duplicate tracks in world space only after checking their view sets and appearance, and
   require source-view render tolerance in addition to moment/KL criteria before merging.

This keeps G²SR's fast, well-conditioned core while making full-covariance recovery a controlled
three-view experiment rather than a default assumption.

For restricted additive synthetic cases, include MGE-style analytic deprojection and Panaretos-style
Gram recovery as sanity baselines. They are intentionally mismatched to perspective alpha rendering:
their purpose is to detect algebra, labeling, and forward-operator bugs before the full problem is
attempted.

### 9.4 Radiometric calibration and coverage repair

Initialize 3D opacity from a conservative constant or support confidence, not copied additive
amplitude. Estimate view-independent color from robust multiview samples, initialize higher-order
appearance at zero, then optimize opacity/color against all source views. For appearance that is
genuinely view dependent, unlock spherical harmonics only after means and principal axes are
stable.

Track a separate nonnegative **geometric support mass** through split/merge operations. It should
sum under refinement and control topology bookkeeping, while renderer opacity remains free to
calibrate visibility and appearance. Manifold-GS makes the same separation for a different asset
objective; its recent evidence motivates the ablation but does not establish the benefit here.

Build a coverage map from:

- source components with no accepted track;
- rendered low-alpha pixels inside valid masks or scene bounds;
- track rejection frequency;
- held-out-view residual and depth inconsistency.

Add primitives only where at least two signals agree. Candidate additions should come from dense
depth/point anchors in the hybrid system or bounded ray hypotheses in a field-only experiment.
Freeze or heavily regularize well-conditioned analytic geometry during this brief repair phase so
ordinary densification cannot erase it immediately.

### 9.5 Four operating modes

| Mode | Required inputs | Lift and refinement | Intended outcome |
| --- | --- | --- | --- |
| **Masked / speed** | Poses, RGB or fields, soft masks | Visual-hull bounds → sigma-point tracks → surfel fit → minimal alpha/color calibration | Small object model at low latency; incomplete concavities accepted |
| **Masked / quality** | Poses, RGB, fields, soft masks, optional depth | Above + boundary/core populations + uncertainty additions + short geometry/appearance refinement | High boundary fidelity and better completeness |
| **Unmasked / speed** | Posed multiview RGB/fields | G²SR-like direct geometry, view-independent color, explicit simple background | Metric visible surfaces with low memory; limited texture/coverage |
| **Unmasked / quality** | Poses or pose model, RGB, fields, multiview depth/point features | Analytic field anchors + dense geometry coverage + decoupled appearance + targeted refinement | Best expected NVS/coverage; higher compute and model dependence |

## 10. Evaluation protocol that can distinguish the methods

### 10.1 Dataset matrix

Use multiple regimes because a single bounded synthetic scene cannot establish the tradeoff:

| Regime | Suggested datasets/captures | Required variation |
| --- | --- | --- |
| Calibrated masked objects | DTU object crops, OmniObject3D, synthetic ground-truth objects | Exact/eroded/dilated/noisy soft masks; 2/3/4/8/16 views |
| Unmasked indoor scenes | Replica, ScanNet/ScanNet++ | Baseline, texture, depth range, occlusion, posed and perturbed poses |
| Unbounded NVS | MIP-NeRF 360 | Foreground/background allocation and far-field handling |
| Controlled identifiability | Synthetic ellipsoids/surfels with exact projected fields | Baseline angle, noise, split/merge, covariance aspect, two vs three views |
| Real fitted-field stress | At least two real calibrated captures | Specularity, thin structure, sensor noise, repeated texture |

Train/source views must be separated from held-out evaluation views. Methods using foundation
models need a disclosure of training-set overlap or at least the available contamination checks.

### 10.2 Metrics

**Geometry**

- metric depth AbsRel/RMSE and threshold accuracy without ground-truth scale alignment when metric
  cameras are claimed;
- Chamfer distance and F-score at stated thresholds;
- normal angular error and covariance shortest-axis alignment;
- projected center/covariance Hellinger error on source and held-out views.

**Appearance and coverage**

- full-canvas PSNR, SSIM, and LPIPS, with the LPIPS backbone named;
- valid-pixel metrics only as a secondary diagnostic, always paired with alpha coverage;
- foreground, foreground-crop, background, and boundary-band metrics;
- coverage-error curves over multiple alpha thresholds, so a method cannot improve PSNR by
  suppressing difficult pixels.

**Masks and support**

- foreground IoU, boundary F-score, outside-mask alpha, and precision/recall;
- robustness to mask erosion, dilation, holes, and calibrated probability noise;
- visual-hull violation and missing-concavity diagnostics where ground truth exists.

**Independent-half stability**

- build two models from disjoint, angle/coverage-stratified camera halves without shared scene
  parameters or refined seeds;
- compare only mutually observed regions using center/normal agreement, projected-covariance
  spectra, and alpha-support overlap;
- report the camera-coverage mismatch beside the agreement score; treat this as an overfitting and
  identifiability diagnostic, not a resolution metric.

**Efficiency and convergence**

- `T_pose`, `T_mask`, `T_2D-fit`, `T_track/depth`, `T_lift`, `T_refine`, and total wall-clock;
- peak end-to-end allocated/reserved VRAM, model-weight memory, host RAM, and final storage;
- primitive count before/after repair, rendered FPS at fixed resolution, and cold/warm latency;
- quality-versus-time and quality-versus-count curves, time to fixed PSNR/LPIPS/geometry targets,
  and area under a preregistered quality-time curve;
- failure rate and coverage at the target, not only successful-scene average.

### 10.3 Minimum causal ablations

1. source RGB pixels versus fitted 2D fields versus both;
2. centers only versus center+covariance sigma-point tracks;
3. independently fitted fields versus a correspondence-trained detector at matched count;
4. two views with surfel prior versus three-view full covariance;
5. hard masks versus soft masks versus soft masks plus uncertainty slack;
6. visual hull, mask-filtered SfM/depth, and no mask geometry;
7. copied field amplitude versus independent opacity calibration;
8. fixed topology, short targeted additions, default densification, and MCMC/strong density
   control;
9. no refinement, equal short wall-clock, and equal long wall-clock;
10. all-pixel versus valid-only metrics with coverage disclosed.
11. hard one-to-one matching versus epipolar-gated soft transport with dustbin and split mass;
12. local Jacobian versus sigma-point/exact projection, with and without nonlinearity-triggered
    moment-preserving splits;
13. ordinary randomized views versus ordered view subsets, stable-region freezing, and final
    all-view cleanup;
14. shared-view reconstruction versus two independent coverage-stratified camera halves.

The critical factorial is `initializer × density controller × time budget`. Without it, a result
cannot distinguish a better lift from a controller that compensates for or destroys the lift.

## 11. Cheapest decisive experiments

### Experiment A — is covariance information actually useful?

Generate exact calibrated views of thin ground-truth 3D Gaussians, fit or perturb the projected 2D
fields, and compare center-only triangulation, sigma-point affine tracking, and joint projected-
covariance fitting. Sweep two/three views, baseline, covariance aspect, and noise. Hold final
primitive count and refinement time fixed.

**Kill criterion:** if center+covariance does not improve held-out projected-shape error, normal
error, or time-to-geometry over center-only on well-conditioned three-view cases, the extra field
shape is not carrying usable geometry under the tested association model.

### Experiment B — does a high-PSNR fitted field preserve enough correspondence?

On two real calibrated scenes and one synthetic scene, compare independently optimized
GaussianImage/Image-GS fields with a G²SR-style correspondence-trained detector at matched
component count and source PSNR. Use the same tracker and analytic back end.

**Kill criterion:** if independent fields have materially lower track survival, coverage, and
held-out geometry even at equal source PSNR, stop treating Stage-1 image quality as the primary
selection metric and train/fix the field for repeatability.

### Experiment C — is the field causally better than dense points?

Compare point centers, centers plus isotropic pixel footprints, full lifted footprints, and EDGS-
style dense point initialization under fixed total time, primitive count, and density controller.

**Kill criterion:** if full footprints do not improve convergence AUC, geometry, or compactness
over a point/footprint baseline, use the simpler dense correspondence pipeline for production and
retain field lifting only as a research representation.

### Experiment D — do soft masks improve the actual 3D model?

Inject calibrated erosion/dilation/holes into ground-truth object masks. Compare hard visual-hull
intersection, soft-probability filtering, and soft probability with boundary slack/self-consistent
replacement.

**Kill criterion:** if the probabilistic method only improves rendered mask IoU but not geometry,
boundary appearance, or outside-alpha robustness, its benefit is segmentation calibration rather
than reconstruction quality.

## 12. Research gaps and novelty ledger

| Claim or direction | Status at 2026-08-04 | Strongest threat/evidence |
| --- | --- | --- |
| “Lift 2D Gaussian splats into 3D Gaussians” | **Closed as a broad novelty claim** | G²SR directly detects, tracks, triangulates, and optimizes 2D splats into metric 3D surfels |
| “Deproject a fitted 2D Gaussian mixture into a 3D Gaussian mixture” | **Closed under restricted forward models and shape priors** | Astronomy MGE performs analytic deprojection; Panaretos recovers radial 3D mixture shape from 2D mixture projections; e2gmm optimizes projected 3D GMMs in cryo-EM |
| Convert arbitrary independently fitted image-Gaussian mixtures | **No located end-to-end demonstration** | G²SR's detector is trained for correspondence and still uses RGB flow |
| Convert fields after discarding RGB pixels | **No located demonstration** | Existing direct/adjacent methods use RGB, features, depth, masks, or pretrained priors |
| Feed-forward masked human 2D parameter maps → 3DGS | **Established adjacent capability** | GPS-Gaussian unprojects depth-backed pixel-wise Gaussian maps from two foreground-matted source views; it is human-specific and depth-supervised |
| Recover unrestricted full 3D covariance from two views | **Not generically identifiable** | Projected quadratic-form restrictions leave a null direction; G²SR adopts a thin-surface prior |
| Three-view full covariance tomography for matched splats | **Mathematically plausible; narrow practical gap** | MGE/random tomography close broad mixture-deprojection novelty; renderer-aware anisotropic covariance, visibility, and arbitrary component association remain unresolved |
| Mask-aware boundary-stable field lifting | **Mechanism pieces exist; combination unvalidated** | Contour-aware 2D fitting, GaussianObject, probabilistic object reconstruction, G²SR |
| Field-derived initialization universally accelerates 3DGS | **Unsupported** | EDGS is point-based; initialization benefits interact strongly with densification |
| Stage-1 PSNR predicts Stage-3 quality | **Unsupported and mechanistically doubtful** | Component gauges/splits and G²SR's tracking-specific detector separate the objectives |
| Geometry/appearance decoupling improves a strict field lift | **Promising but untested** | G²SR, G3Splat, and QuerySplat support adjacent mechanisms |

The defensible research question is no longer “can 2D Gaussians be lifted?” It is:

> Can independently fitted, compact 2D teacher fields preserve enough cross-view identity and
> affine shape to beat point/depth initialization in the quality–time–count Pareto set, especially
> when masks and three-view covariance constraints are used explicitly?

That question is falsifiable and narrower than G²SR. It also allows a valuable negative result:
the fitted fields may be excellent image representations but discard the correspondence signal
needed for 3D.

After the cross-domain pass, the most defensible technical contribution would combine four deltas
that the older methods do not jointly cover: arbitrary compact per-view componentizations,
calibrated few-view perspective geometry, visibility/alpha-aware radiometry, and a measured
quality–time–count advantage over dense point/depth initialization. Analytic deprojection, mixture
transport, or moment-preserving topology control alone should be framed as transferred machinery,
not as a new problem class.

## 13. Implications for realtime-gs

This review changes documentation and research framing only; it does not change a default or
promote a capability claim.

- The previous broad novelty note in [`RESEARCH.md`](RESEARCH.md) must defer to G²SR.
- The repository's distinctive candidate is the use of **pre-fitted sparse teacher fields**, its
  explicit separation of localization and coverage behavior, and possibly an image-free lifting
  boundary—not the generic act of turning tracked 2D splats into 3D surfels.
- [`DESIGN_field_lift.md`](DESIGN_field_lift.md) remains a design proposal. G²SR should become the
  direct geometric baseline, while EDGS remains the dense point/convergence baseline.
- Existing results in [`EXPERIMENTS.md`](EXPERIMENTS.md) already warn that covariance choice has no
  universal winner and that ordinary density control can wash out initialization. Those local
  results are consistent with the literature but are not external validation.
- No experiment should maximize only final PSNR. The next protected comparison should measure
  association survival, coverage, geometry, time-to-quality, total time, and the interaction with
  the density controller.

The accompanying
[`RESEARCH_PORTFOLIO_2D_TO_3D_GAUSSIANS.md`](RESEARCH_PORTFOLIO_2D_TO_3D_GAUSSIANS.md) turns the
gaps above into falsifiable candidate programs and recommends the smallest first experiment.

## 14. Audit limitations

- G²SR, ATSplat, QuerySplat, Manifold-GS, the probabilistic object method, and several fast Stage-1 systems were
  recent preprints at the cutoff. Their claims may change after review, code release, or revision.
- Many papers report different datasets, resolutions, hardware, timing boundaries, metric masks,
  and pretrained models. Numbers retained here illustrate protocol-specific Pareto points; they
  are not a global leaderboard.
- Code and license availability were not the primary inclusion test and can change. Re-verify both
  before implementation or redistribution.
- Search terms may miss work described as Gaussian mixtures, ellipsoid/quadric reconstruction,
  surfel tracking, astronomy deprojection, cryo-EM, or tomography without “Gaussian Splatting” in
  the title. Scholar Inbox broadened discovery but does not establish exhaustiveness.
- No external result in this document has been reproduced in this repository. Repository claims
  require protected experiments and an independent scientist pass under the local workflow.
- The proposed combined pipelines are reasoned syntheses. Component evidence does not establish
  that the combination is additive, stable, or faster end to end.
