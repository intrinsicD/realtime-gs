# Can per-image 2D Gaussian fits support 3D Gaussian splatting? A physics-first critique

**Labels used:** [F] = verified from cited source; [D] = my own derivation from standard mathematics; [H] = hypothesis, not established.

## 1. The two forward models are not the same problem

The tomography analogy is inviting because in X-ray CT a 2D image *is* a projection of a 3D density. Beer–Lambert gives, for a ray r,

  −log(I(r)/I₀) = ∫ ρ(x) dℓ ,

which is **linear in ρ** after the log. [F] R²-Gaussian ([arXiv:2405.20693](https://arxiv.org/abs/2405.20693), NeurIPS 2024) and X-Gaussian ([arXiv:2403.04116](https://arxiv.org/abs/2403.04116), ECCV 2024) exploit exactly this: the density is a sum of 3D Gaussians and the rendering is additive, view-direction-independent, and occlusion-free.

For an unnormalised 3D Gaussian g_k(x) = ρ_k exp(−½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k)), the line integral along direction d under parallel (or locally affine) projection P is [D]:

  ∫ g_k dℓ = ρ_k · √(2π) σ_{d,k} · exp(−½(u−Pμ_k)ᵀ(PΣ_kPᵀ)⁻¹(u−Pμ_k)), with σ_{d,k}² = 1/(dᵀΣ_k⁻¹d).

So a 3D Gaussian projects to **exactly one** 2D Gaussian whose amplitude carries the along-ray extent σ_{d,k}. [F] R²-Gaussian reports an "integration bias" in vanilla 3DGS when used for CT; [D] the missing √(2π)σ_d factor is what one obtains if the projected 2D Gaussian is normalised to peak opacity as in 3DGS. The whole image is then a mixture of N 2D Gaussians, one per primitive, in every view. This is the regime in which "fit a 2D GMM per view, lift to 3D" is a coherent idea, and it is the regime that cryo-EM GMM methods already occupy [F] ([Chen & Ludtke, Nat. Methods 2021](https://www.nature.com/articles/s41592-021-01220-5); [CryoSplat, arXiv:2508.04929](https://arxiv.org/html/2508.04929)).

An ordinary RGB photograph obeys a different law. [F] 3DGS ([Kerbl et al., arXiv:2308.04079](https://arxiv.org/abs/2308.04079)) composites front-to-back:

  C(u) = Σ_i c_i α_i(u) Π_{j<i} (1 − α_j(u)),  α_i(u) = o_i · exp(−½(u−μ'_i)ᵀΣ'_i⁻¹(u−μ'_i)).

[D] Expanding the product, C(u) is a **signed** mixture of up to 2^N Gaussians: each term Π_{j∈S} α_j is itself a Gaussian (product of Gaussians is a Gaussian with precision Σ_{j∈S}Σ'_j⁻¹), but with sign (−1)^{|S|} and depth-order-dependent coefficients. So a compact 2D GMM fitted to an RGB image has no one-to-one relation to the 3D primitives that produced it; its components are effectively fitting the *front visible surface's radiance*, not the projection of a volume. Additionally [F] the ordering is per-view and 3DGS uses an affine approximation of perspective inherited from EWA splatting ([Zwicker et al., 2001](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf)); [F] Celarek et al. ([arXiv:2502.19318](https://arxiv.org/abs/2502.19318)) find that these approximations matter mainly when primitive count is low, and that optimisation with many Gaussians compensates otherwise.

**Bottom line for the analogy:** what makes CT linear is the absence of occlusion, not the fact that images are "2D densities". RGB images of opaque scenes are radiance of the first hit surface; the "density" of the object interior lies in the null space of the forward operator.

## 2. Visibility and identifiability

[F] From silhouettes alone, the recoverable shape is the visual hull ([Laurentini 1994](https://www.semanticscholar.org/paper/033cc3784a60115d758a11a765e764b86aca336c)); photometric constraints tighten this to visible surfaces but never expose interiors. Consequently the target 3D representation for RGB is a *shell* of Gaussians, and the only things a 2D image-plane Gaussian can encode are (a) a patch of surface radiance and (b) its screen-space footprint. Depth is absent per view and must come from cross-view parallax.

Two distinct identifiability problems arise for *independent* per-image fits:

1. **Mixture non-uniqueness.** [F] GaussianImage ([arXiv:2403.08551](https://arxiv.org/abs/2403.08551)) and successors ([GaussianImage++](https://arxiv.org/abs/2512.19108), [Instant GaussianImage](https://arxiv.org/abs/2506.23479)) fit thousands of 2D Gaussians to an image for compression; the components have no physical meaning, and the fit is a non-convex problem with many equivalent minima. [H] There is no reason a component fitted in view v corresponds to any component in view v′, even for a smooth Lambertian surface, because the fit partitions the image according to local gradient statistics that change with foreshortening, shading and occlusion boundaries.

2. **Covariance lifting is under-determined from two views.** [D] Under affine projection, Σ'_v = P_vΣP_vᵀ gives three linear equations in the six entries of Σ per view. Two views share the variance along the axis common to both image planes (the direction orthogonal to both optical axes), so the combined system has rank 5: **at least three non-coplanar views are needed to lift a covariance**, and the along-ray extent of any single view is invisible. In attenuation tomography the amplitude term √(2π)σ_d supplies that missing information; in RGB the amplitude is radiance and supplies nothing about extent.

The contrast with X-ray is sharp: there, a 3D Gaussian is a single 2D Gaussian in every view, so matching across views is a tracking problem in (μ', Σ', amplitude) space, and the linear forward model means one does not even need per-image fits — pixel-wise least squares on the additive model suffices, which is precisely what R²-Gaussian/X-Gaussian do [F].

## 3. Independent 2D fits versus shared projected 3D primitives

Given the above, the only physically sound design uses **shared 3D primitives whose projections are the 2D Gaussians**; independent per-image fits can serve as *data compression*, *initialisation*, or a *loss space*, but not as the latent representation. [F] The closest working relatives already exist and are instructive: Splatter Image ([arXiv:2312.13150](https://arxiv.org/abs/2312.13150)) and MVSplat ([arXiv:2403.14627](https://arxiv.org/html/2403.14627v1)) predict one Gaussian per pixel *with depth*, i.e., image-aligned but genuinely 3D, and MVSplat obtains that depth from a cross-view cost volume. Note also that "2D Gaussian Splatting" ([arXiv:2403.17888](https://arxiv.org/pdf/2403.17888)) uses 2D surfels placed in *world* space, not the image plane — it should not be confused with the user's proposal.

## 4. A credible architecture (proposal; all steps [H] unless marked)

**Stage A — per-view compression.** Fit a GaussianImage-style 2D GMM to each calibrated photo with a *modest* number of components (10²–10³) plus a residual, keeping colour, μ', Σ'.

**Stage B — epipolar matching.** Treat 2D Gaussian centres as anisotropic keypoints. Match across views with the epipolar constraint, colour similarity, and the covariance consistency test from §2 (a candidate 3D Σ must satisfy P_vΣP_vᵀ ≈ Σ'_v for all matched views; solve the 3V×6 linear system, reject high residual). Require ≥3 views per track.

**Stage C — lifting.** Triangulate μ from matched centres; solve Σ linearly; set opacity high (surface). This yields a 3D Gaussian initialisation that is already anisotropically aligned with image structure, which SfM point clouds are not.

**Stage D — standard 3DGS/ray-traced refinement.** Optimise all primitives against pixels with alpha compositing (or ray tracing, [F] [arXiv:2407.07090](https://arxiv.org/html/2407.07090v1), if perspective accuracy matters), with the per-view GMMs used only as an auxiliary multiscale loss (e.g., match rendered image against the GMM reconstruction at several σ), not as a target for primitive-level correspondence.

**What should work physically:** Stages C–D, because they never assume the RGB forward model is linear. **What is at risk:** Stage B, because 2D GMM components may not be repeatable across views (§2, item 1). If repeatability is poor the method degenerates to "3DGS with a different loss", which may still be fine but is not the user's idea.

**Alternatives.** (i) For genuinely additive media — smoke, fluorescence/light-sheet microscopy, emission nebulae, X-ray — use the attenuation/emission model directly; the 2D-fit-and-lift idea is then exact and the R²-Gaussian kernel with the √(2π)σ_d amplitude is the right renderer. (ii) For opaque scenes, replace the 2D GMM with a 2D *surfel* GMM in world space (2DGS) initialised from matched image Gaussians. (iii) Skip fitting entirely and use feed-forward pixel-aligned Gaussians (MVSplat) as initialisation — arguably the pragmatic baseline the proposal must beat.

## 5. A small falsification experiment

Synthetic scene: N = 50 known anisotropic 3D Gaussians, V = 24 calibrated views. Render two ways: (i) additive/attenuation with the exact projected amplitude; (ii) alpha-composited (3DGS). Fit per-view 2D GMMs with exactly N components and multiple restarts.

- **Test 1 (correspondence):** Hungarian-match fitted components to true projections using a Bhattacharyya distance on (μ', Σ'). Prediction from §1: in (i) >90 % of components match with small cost; in (ii) matching is poor and worsens as occlusion increases (control: vary primitive opacity from 0.05 to 1.0). *If (ii) matches as well as (i), the nonlinearity argument is falsified.*
- **Test 2 (lifting):** From matched tracks, solve for Σ using 2, 3 and 6 views. Prediction from §2: rank-5 failure with 2 views, recovery with ≥3. *Falsified if 2 views suffice for generic geometry.*
- **Test 3 (end-to-end):** On a public real dataset, compare final PSNR/LPIPS and convergence time of 3DGS initialised by (a) SfM points, (b) Stage-C lifted Gaussians, (c) MVSplat output. *The proposal is only worth pursuing if (b) beats (a) and is competitive with (c).*

## 6. Summary

Attenuation tomography with Gaussian primitives is linear, occlusion-free and exactly closed under projection; per-image 2D Gaussian fits are therefore meaningful there (and the CT/cryo-EM literature already uses this). Ordinary RGB photographs of opaque scenes are radiance of the first visible surface under a nonlinear, order-dependent compositing operator; a compact 2D GMM of such an image is not a projection of anything, and independent per-view fits are not identifiable across views. The physically defensible version of the idea keeps shared 3D primitives, uses per-view 2D fits only for compression, initialisation via epipolar matching plus linear covariance lifting (≥3 views), and auxiliary loss, and validates cross-view repeatability of the fits before anything else. No empirical success is claimed for any of this.

**Sources:** [Kerbl et al. 2023](https://arxiv.org/abs/2308.04079) · [Zwicker et al. EWA Volume Splatting](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf) · [Celarek et al. 2025](https://arxiv.org/abs/2502.19318) · [R²-Gaussian](https://arxiv.org/abs/2405.20693) · [X-Gaussian](https://arxiv.org/abs/2403.04116) · [GaussianImage](https://arxiv.org/abs/2403.08551) · [GaussianImage++](https://arxiv.org/abs/2512.19108) · [Instant GaussianImage](https://arxiv.org/abs/2506.23479) · [Chen & Ludtke 2021](https://www.nature.com/articles/s41592-021-01220-5) · [CryoSplat](https://arxiv.org/html/2508.04929) · [Splatter Image](https://arxiv.org/abs/2312.13150) · [MVSplat](https://arxiv.org/html/2403.14627v1) · [2D Gaussian Splatting (surfels)](https://arxiv.org/pdf/2403.17888) · [3D Gaussian Ray Tracing](https://arxiv.org/html/2407.07090v1) · [Laurentini, Visual Hull](https://www.semanticscholar.org/paper/033cc3784a60115d758a11a765e764b86aca336c)
