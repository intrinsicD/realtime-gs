# Compact-only carrier stage ablation — preregistration — 2026-07-28

## Status and question

Status at creation: **FROZEN BEFORE NEW OPTIMIZER OUTCOMES**.

The previous all-26 carrier-maturation run is not evidence for the product's compact-only
claim: Beam Fusion and carrier repair consumed compact 2D Gaussian captures, but every later
optimizer phase consumed native masked RGB. This development experiment replaces that
supervision contract completely.

The questions are:

1. Which, if any, of covariance, opacity, and appearance repair improve a subsequent
   fixed-topology optimizer that sees only calibrated 2D Gaussian fields?
2. Can repair be skipped in favor of direct fixed-topology optimization?
3. What happens when only means move, when means are frozen, when only geometry moves, or when
   only appearance/opacity moves?
4. Does a differentiable support barrier derived only from the 2D Gaussian fields reduce
   projected-center violations without materially degrading compact-teacher validation risk?
5. Do renderer-dilation-aware symmetric covariance repair and view-uniform robust appearance
   repair outperform the legacy formulas they correct?

This protocol is a development mechanism study. Existing Beam/carrier outcomes and the prior
all-view scene are already known, so this is not confirmatory evidence.

## Absolute no-image boundary

The executable path may load only:

- the 26 capped `.rtgsv` containers in
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`;
- the Gaussian observation tensors, calibrated cameras, bounds, and integrity metadata inside
  those containers; and
- generated 3D Gaussian states and numeric receipts from this experiment.

It must not:

- decode or query `PackedAlpha`;
- open source RGB or mask files;
- materialize a dense RGB, alpha, mask, or teacher-replay image;
- import Pillow, OpenCV, `rtgs.data.calibrated`, or `rtgs.data.scene`; or
- call `Trainer`, `SceneData`, gsplat dense training, or the prior dense carrier schedule.

The optional packed alpha is deliberately excluded because the user requested no images at all.
The support treatment below is derived from the fitted 2D Gaussian primitives themselves.

## Camera split

The ordered manifest cameras are the population. Zero-based global indices `7,15,23` (every
eighth camera) are untouched held-out cameras. From the remaining ordered cameras, every fifth
entry starting at position four is validation; the rest are fitting cameras. Concretely, the
harness must persist the resulting ordered view IDs and fail if the manifest order or split
changes.

- Beam Fusion, repairs, proposal sampling, gradients, checkpoints, and selection use fitting
  cameras only.
- Validation cameras may rank arms and decide whether a later experiment is authorized.
- Held-out cameras are evaluated only after all arm states and the ranking rule are immutable.
  They do not change this experiment's ranking or any later-stage protocol.

## Shared Beam initialization

Run Beam Fusion once on fitting views with:

- `min_views=3`;
- `transverse_gate_sigma=3`;
- `max_color_distance=0.35`;
- `color_sigma=0.25`;
- `fold_in_gate_sigma=3`;
- `nms_voxel_size=bounds_extent/100`;
- `init_opacity=0.10`;
- `source_chunk=256`;
- `max_components=5_000`; and
- `seed_budget_multiplier=4`.

Persist the complete contributor CSR and implied depths so every repair arm uses exactly the same
carriers and links. Beam is deterministic; no pre-existing all-26 Beam PLY is reused because it
contains held-out-camera information.

## Repair math and frozen variants

Let carrier covariance be `Sigma_i`, contributor camera Jacobian be `J_iv`, fitted 2D covariance
be `C_iv`, renderer EWA variance be `d=0.3`, fitted 2D amplitude be `a_iv`, and contributor color
be `c_iv`.

### Legacy covariance

The existing repair uses

`r_legacy = RMS(C_iv^(-1/2) J_iv Sigma_i J_iv^T C_iv^(-1/2) - I)`.

This omits renderer dilation and is asymmetric: collapse toward zero has bounded residual while
arbitrary expansion does not. It remains an arm because it is the implemented historical stage,
not because the formula is endorsed.

### Corrected covariance

The corrected projected covariance is

`P_iv = J_iv Sigma_i J_iv^T + d I`.

Its residual is the affine-invariant generalized-eigenvalue log residual

`r_log = sqrt(mean_k(log(lambda_k(C_iv^(-1/2) P_iv C_iv^(-1/2)))^2))`.

This is symmetric under swapping scalar prediction and target, includes the exact EWA dilation
used by the point renderer, and remains finite after the existing SPD bounds. Targets sharper
than `d I` are reported as irreducible rather than silently double-dilated.

Both covariance variants retain 120 Adam steps, learning rate `0.03`, Huber delta `0.25`, prior
weight `1e-3`, world-sigma bounds `[1e-4, 0.5*extent]`, and aspect ratio at most `100`.

### Legacy opacity

The legacy repair maps `a_iv` to optical density `-log(1-a_iv)` and robustly fits one opacity per
carrier. Because normalized 2D Gaussian amplitudes are gauge-dependent mixture weights rather
than physical alpha, this stage is explicitly classified as **non-identifiable**. It remains only
as an ablation arm. No "corrected opacity repair" is invented.

### Appearance

Legacy appearance uses amplitude-weighted multivariate Huber IRLS. Corrected appearance uses the
same five IRLS steps and Huber delta `0.10` but gives each contributor view equal base weight;
normalized-renderer amplitudes do not determine cross-view confidence. Both output SH degree zero
and are initialization heuristics, not physical color recovery.

## Fixed-topology compact optimizer

Every arm uses `CompactTrainer` and fitting-view `ReconstructionInputs` only:

- three fresh training roots: `280701,280702,280703`;
- 380 updates (20 complete balanced cycles over 19 fitting views);
- 256 fixed attempts per update;
- continuous `area_gaussian` proposal;
- proposal-attempt target with uniform fraction `0.25`;
- explicit dataset bounds extent;
- `cuda:0`, float32;
- point chunk `256`, Gaussian chunk `512`, outer microbatch `128`;
- teacher query chunk `512`, tile size `16`;
- checkpoints `0,95,190,380`;
- hard degree-zero SH, hard 3-sigma EWA support, black background; and
- the existing Adam rates unless a parameter family is frozen.

A frozen family retains an Adam group with learning rate exactly zero, so topology/state layout
and all gradient diagnostics remain common. The harness must prove bit-exact zero motion for each
frozen family.

## 16 frozen arms

`C`, `O`, and `A` below mean legacy covariance, opacity, and appearance repair. Every repair
factorial arm then optimizes all degree-zero parameter families.

| Arm | Initialization / trainable families | Support barrier |
| --- | --- | --- |
| `beam_all` | Beam / means, quaternion, scale, opacity, SH0 | off |
| `legacy_C_all` | C / all | off |
| `legacy_O_all` | O / all | off |
| `legacy_A_all` | A / all | off |
| `legacy_CO_all` | C+O / all | off |
| `legacy_CA_all` | C+A / all | off |
| `legacy_OA_all` | O+A / all | off |
| `legacy_COA_all` | C+O+A / all | off |
| `corrected_C_all` | corrected C / all | off |
| `corrected_CA_all` | corrected C + corrected A / all | off |
| `beam_means_only` | Beam / means only | off |
| `beam_means_fixed` | Beam / quaternion, scale, opacity, SH0 | off |
| `beam_geometry_only` | Beam / means, quaternion, scale | off |
| `beam_appearance_only` | Beam / opacity, SH0 | off |
| `beam_all_support` | Beam / all | on |
| `corrected_CA_all_support` | corrected C + corrected A / all | on |

All arms begin from independently hashed but deterministically constructed snapshots. The three
seeds for an arm share its exact initialization. Within a seed, all arms share view and proposal
sample roots.

## Gaussian-only support barrier

For a projected 3D Gaussian center `u_iv` and fitted 2D component `j`, define

`q_ivj = (u_iv-mu_vj)^T C_vj^(-1) (u_iv-mu_vj)`.

Only positive-amplitude components participate. Let `q_min=min_j q_ivj`, `s_v` be the frozen
teacher cutoff (3 here), and

`z=max(0, sqrt(q_min)-s_v)/s_v`.

The per-center barrier is unit-beta smooth L1, `rho(z)`. On each update, a deterministic
independent stream selects 256 current 3D rows and applies

`L_support = 0.01 * mean rho(z)`

for the scheduled fitting view. The loss is independent of opacity, so a Gaussian cannot evade
the geometry constraint by becoming transparent. The minimum is evaluated in bounded component
chunks against the untruncated ellipses; unlike a hard clipped-support density, it therefore has
a gradient outside the union. A separate near-plane hinge is included for centers at or behind
the renderer near plane.

This is a soft projected-center visual-hull approximation, not a proof of surface occupancy.
Boundary footprints may legitimately cross a silhouette, so full covariance containment is not
required. A floater inside the multi-view visual hull can also satisfy every center constraint;
the experiment must not claim that the barrier alone removes all floaters.

## Frozen evaluation

Before training, create immutable 8,192-attempt continuous-uniform and 8,192-attempt
proposal-attempt banks for every camera from evaluation roots `281701,281702,281703`, with
domain-separated per-view/per-measure seeds. Every checkpoint is evaluated by sparse point
queries against those banks—never by image materialization.

For RGB-channel point MSE `ell`, report equal-view:

`J_U = mean_views (1/8192 sum_k ell_k)`

and

`J_Q = mean_views (1/8192 sum_k 1[active_k] ell_k)`.

Also evaluate every 3D row against every camera's Gaussian support union:

- fraction of projected centers with `sqrt(q_min)>3`;
- mean and p95 normalized violation `z`;
- fraction behind the near plane;
- per-Gaussian count of violating cameras; and
- train/validation/held-out aggregates.

Resource receipts include peak CUDA allocated/reserved bytes, peak RSS, teacher bytes resident,
3D parameter bytes, point/Gaussian chunk maxima, and zero forbidden-image opens/imports.
Measurements are descriptive on this one workstation and cannot establish a general VRAM claim.

## Frozen analysis

The primary stage score is the geometric mean across seeds of final validation `J_Q`; validation
`J_U`, log-AUC through the four checkpoints, support diagnostics, and parameter motion are
co-primary safety/mechanism reports.

- A repair factor is "necessary in this bounded study" only if its matched factorial marginal
  improves final validation `J_Q` by at least 5%, wins in all three paired seeds, and does not
  worsen validation `J_U` by more than 5%.
- A simpler parameter-family arm is non-inferior only if both validation `J_Q` and `J_U` are at
  most 2% worse than `beam_all`.
- A support arm succeeds only if projected-center violation falls by at least 50%, validation
  `J_Q` is at most 5% worse, validation `J_U` is at most 5% worse, and no behind-near fraction
  increases.
- Corrected covariance/appearance replace their legacy counterpart only if validation `J_Q`
  improves by at least 2% with validation `J_U` no worse than 2%.

After all arm files and rankings are immutable, held-out metrics are unlocked for the shared Beam
baseline, the lowest-validation-`J_Q` arm, the best support arm, and any simpler arm that passes
non-inferiority. Held-out results are descriptive replication checks and do not change selection.

No clone, split, birth, prune, higher-SH, or adaptive-density conclusion is allowed here. A second
preregistered experiment may use the immutable validation winner to test:

- continued fixed topology;
- optical-density-preserving tangent clone/split versus legacy opacity-copy cloning;
- degree-zero versus higher-SH compact refinement; and
- responsibility-selected local births plus support-based pruning.

The second protocol must be frozen only after this result is immutable and may not use this
experiment's held-out metrics to choose its stages or thresholds.

## Claim boundary

A successful run can establish only bounded single-scene development facts about compact-field
optimization, repair utility, parameter motion, and projected-center support. It cannot establish
source-RGB equivalence, exact mask containment, absence of all floaters, physical covariance or
opacity recovery, novel-scene transfer, production quality, speed, general VRAM savings, or a
default pipeline.
