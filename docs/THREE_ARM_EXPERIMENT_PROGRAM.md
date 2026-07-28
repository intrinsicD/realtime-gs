# Two compact pipeline arms plus an RGB control

Status: task-first development design. No task below is an executed result.

The active registry is
`experiments/programs/20260728_three_claim_arms_stage_frames00008_00009.json`.
All three arms share camera splits, seeds, capture identity, artifact names, and report template.
They deliberately do not share input byte sets: compact tasks seal only calibration/`.rtgsv`;
the RGB task seals calibration/RGB/PNG masks.

```
frozen 2D Gaussian fields + calibration
       ├── direct compact arm ── compact carve → compact optimization
       │                         → 3D Gaussians + memory receipt
       └── Beam arm ── Beam Fusion → matched compact optimization
                       → 3D Gaussians + incremental Beam evidence

RGB + PNG masks + calibration + frozen matched initialization
       └── RGB control ── standard image-supervised 3DGS → quality/resource comparison
```

The RGB arm may render and score the immutable compact output on images. The compact processes
must never import an image loader, open an image suffix, accept `SceneData`, or call the dense RGB
trainer. This process separation is part of the claim, not a coding preference.

## Arm 1: direct compact-only VRAM claim without Beam

The reconstruction boundary starts at an integrity-checked `ReconstructionInputs`: calibrated
cameras and frozen per-view 2D Gaussian observation fields. It ends after the final 3D Gaussian
file is saved. RGB capture and RGB-to-2D fitting are upstream and excluded. Therefore the honest
claim name is **post-fit reconstruction VRAM**, not capture-to-model VRAM.

The draft primary process is:

1. balanced compact-carve initialization with a hard 3D count cap;
2. sampled compact-field optimization under a separately frozen no-Beam schedule;
3. compact-domain validation and standard `Gaussians3D` output.

This arm must never import or call Beam Fusion, Beam contributor lineage, carrier covariance
repair, or the Beam-selected carrier schedule. Its initializer is saved and reused byte-for-byte
by the RGB control, so the resource comparison changes supervision/input representation rather
than initialization. The exact compact optimizer, topology policy, and stopping rule remain draft
blockers and must be frozen before execution.

Measure in a fresh process on an otherwise idle named GPU:

- NVML process-used-memory peak sampled at a frozen rate;
- PyTorch allocated and reserved peaks;
- process RSS, input/output bytes, wall time, stage time, and Gaussian count;
- software/driver/GPU identity and background-memory baseline.

Run one warm-up and at least three fresh measured repeats for each paired seed/scene. Keep every
raw repeat; headline medians alone are insufficient. The RGB control must use the same device,
resolution, initialization, seeds, and resource boundary. CUDA allocated memory does not include
all driver/backend allocations, so it cannot be the sole VRAM number.

## Arm 2: Beam Fusion and stage value

### Beam geometry

For contributor beam \(k\), let \(\Lambda_k\) be its world-space precision and \(m_k\) its
world-space mean at the implied ray depth. The implementation uses equal-weight covariance
intersection:

\[
\Lambda = \frac{1}{K}\sum_k \Lambda_k,\qquad
m = \Lambda^{-1}\left(\frac{1}{K}\sum_k \Lambda_k m_k\right).
\]

The common \(1/K\) leaves the mean unchanged relative to a product but keeps the covariance
conservative when camera observations are correlated. Pair seeding is gated by ray separation in
the two transverse footprints and by fitted 2D color; later views fold in at most one component.
Consequently, Beam's main risk is false contributor association, not an algebraic missing factor.
Contributor count, per-view uniqueness, gates, covariance eigenvalues, and lineage hashes belong
in the report.

Beam value cannot be inferred from a Beam-only result. Compare, at exact count and compute:

- Beam Fusion (treatment);
- compact carve (strong image-free structural control);
- calibration-bounded random (lower bound).

Use the identical downstream compact optimizer and evaluation samples. Report initialization,
fixed checkpoints, risk-curve area, endpoint, paired seed wins, and both scenes. A converged tie
after an initialization win means Beam improves startup but not final capacity; say exactly that.

### Repair math and ablations

For a carrier covariance \(\Sigma_i\), contributor projection Jacobian \(J_{iv}\), target
2D covariance \(S_{iv}\), and renderer dilation \(dI\), covariance repair compares

\[
M_{iv}=S_{iv}^{-1/2}(J_{iv}\Sigma_iJ_{iv}^{T}+dI)S_{iv}^{-1/2}
\]

through the RMS log generalized eigenvalues of \(M_{iv}\), with a Huber data term, a prior to the
Beam covariance, and explicit SPD/aspect bounds. Means, opacity, appearance, lineage, and count
stay bit-frozen. The equation is internally consistent with the point renderer, but a lower local
residual does not establish downstream value when correspondences are wrong. Keep the stage only
if the matched no-repair ablation improves the frozen downstream risks.

Do not put opacity or appearance repair back into the principal pipeline without new evidence.
A fitted 2D component amplitude is not an identifiable per-3D-carrier opacity under
alpha compositing, and one SH0 color cannot in general explain view-dependent contributor colors.
They remain negative/attribution controls.

### Direct optimization and containment factorial

At minimum compare:

- Beam, no repair, all fields trainable;
- Beam + corrected covariance, all fields trainable;
- Beam + corrected covariance, means only;
- Beam + corrected covariance, means frozen;
- no containment;
- differentiable compact-support barrier with fixed count;
- hard post-fit containment without recovery;
- hard containment plus means-frozen recovery.

The compact objective is a fixed-attempt importance-corrected pointwise color MSE against frozen 2D
Gaussian fields. Evaluate separately with exact pixel risk \(J_\text{pixel}\) and fixed
four-offset area quadrature \(J_\text{area}\); training loss is not a substitute for either.

The recovery phase has a mathematical reason: deleting carriers changes transmittance and color
composition even when every survivor is unchanged. It is nevertheless an empirical stage and
must be dropped if the matched no-recovery arm is non-inferior.

## No-free-floater criterion

Raw masks are forbidden in the two compact arms. Their legal surrogate is the union of fitted 2D
Gaussian support ellipses. The current hard check retains carrier \(i\) only if, for every fitting
view \(v\),

\[
z_{iv}>z_\text{near}
\quad\text{and}\quad
\min_j q_{ivj}\le 3^2,
\]

where \(q_{ivj}\) is the projected-center Mahalanobis distance to fitted 2D component \(j\).
Phase-2 means are frozen and the condition is rechecked, so projected-center containment cannot
be undone.

For a count-preserving “push inside” variant, use a smooth penalty on the same compact support,
not a raw mask. For example,

\[
L_\text{support}=\sum_{i,v}
\operatorname{softplus}\!\left((\min_j q_{ivj}-9)/\tau\right)^2
+ \lambda_z\operatorname{softplus}((z_\text{near}-z_{iv})/\tau_z)^2.
\]

Use a smooth-min if gradients through component switches are unstable. To constrain the Gaussian
footprint rather than only its center, sample deterministic points on the projected 3-sigma
ellipse (or add a covariance-derived boundary margin) and apply the same support penalty. Freeze
this choice in the Beam task before using outcomes.

Silhouettes cannot prove that a Gaussian inside the multi-view visual hull lies on the physical
surface. Thus “no projected-center mask violations” is measurable; “no physical free-floating
Gaussians” is not established without depth, surface ground truth, or a stronger multi-view
occupancy test. Reports must preserve that distinction.

## RGB control: image-supervised 3DGS comparison

The primary comparison loads the exact frozen Beam-independent 3D initialization produced by the
direct compact arm,
then runs the repository's standard RGB/mask trainer. It includes standard densification under a
hard cap. Add a fixed-topology RGB control to separate supervision value from capacity growth and
a bounded-random RGB control as a lower bound.

Evaluate the RGB-trained model and the immutable compact model on exactly the same held-out
cameras with foreground/crop PSNR, SSIM, LPIPS, alpha IoU, exterior leakage, and per-view
distributions. The canonical PNGs carry soft 8-bit alpha, so retain that target for training and
freeze/report a threshold for binary IoU/leakage. Report final count and time-to-quality as well
as endpoint quality.

This is not Original 3DGS. Original 3DGS requires a real COLMAP sparse reconstruction
(`cameras`, `images`, and `points3D`) for initialization. A random or Beam-derived initialization
must not be relabeled as SfM.

## What is still needed for a claim

- A task-reviewed, executable harness for each compact arm and the RGB control; the current tasks
  correctly remain `draft`.
- A frozen material-effect/non-inferiority rule and multiplicity policy before outcomes.
- An NVML sampler and fresh-process resource wrapper shared by compact and RGB arms.
- At least one outcome-unseen calibrated scene. Frames 00008 and 00009 are prior-development
  scenes and cannot become confirmatory by renaming them.
- A real COLMAP model if Original 3DGS is part of the paper comparison.
- Independent source/data/metric audit and the mandatory viewer/results bundle.
- Disclosure of upstream 2D-fitting storage/time/VRAM even when it is outside the post-fit claim.
