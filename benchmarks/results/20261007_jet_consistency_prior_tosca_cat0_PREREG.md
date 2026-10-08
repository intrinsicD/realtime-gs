# Jet-consistency prior for 3DGS training — preregistration — 2026-10-07

## Status and question

Status at creation: **DRAFT — frozen once the protocol digest is approved by the reviewer
(Alexander Dieckman). No training outcome exists yet.** Content approved by the reviewer on
2026-10-07 ("Ich habe das PREREG gelesen. Und ich bin einverstanden."); the digest approval
follows once the task's structural fields are generated. Post-hoc numbers from the
companion repository `~/Documents/SplatDiffuseLBO` (SPEC3D.md §16) are known and are
quoted as the baseline to beat; they are not outcomes of this experiment.

Trained splat normals are noisy where the photometric loss has a null direction (a
needle rotated about its long axis, a sheet seen at grazing angles). Measured on a 3DGS
model of TOSCA cat0 (31 779 surfels; `SplatDiffuseLBO/data/cat_point_cloud.ply`)
against the source mesh: angle between splat normal (smallest axis) and true normal
**16.5° median, 90 % 49.5°**, needles 27° median; centres **0.028** (median, splat units)
off the surface; long axes tangent to 4°. Positions are good, orientations are not.

The question: does a **jet-consistency regularizer** — the normal field in a neighbourhood
must be the Weingarten field of one local quadric, with the shape operator `S_i` solved in
closed form per splat and detached — reduce normal noise and shell thickness during
training, and does that improve a downstream geometry operator (the splat
Laplace–Beltrami operator of SplatDiffuseLBO) against the mesh reference?

Prior art checked (2026-10-07, one search pass, not a review): QGS (ICCV 2025) trains
second-order primitives photometrically; GeoGaussian (ECCV 2024) imposes co-planarity of
nearest neighbours (first-order consistency); Gaussian Surfels and 2DGS use depth–normal
consistency; FlatCAD uses a Weingarten loss on neural SDFs. A neighbour-normal Weingarten
consistency on splats with closed-form `S` was not found.

## The regularizer (exact)

For splat `i` with centre `c_i`, normal `ν_i` (smallest axis of `R_i`), tangent frame
`(e_1, e_2)` (the two other axes), neighbours `j ∈ N_i` (k = 16 nearest centres, list
refreshed every 50 steps), `Δ_j = c_j − c_i`, `s_j = (e_1·Δ_j, e_2·Δ_j)`, `z_j = ν_i·Δ_j`,
weights `g_j = exp(−|Δ_j|²/2h_i²)`, `h_i` = mean distance to the 6 nearest centres:

1. `S_i = argmin_{S sym 2×2} Σ_j g_j |P_i(ν_j − ν_i) − S s_j|²` — closed form (3 unknowns,
   weighted least squares), computed **without gradient** (`detach`), sign convention as in
   SplatDiffuseLBO §11.2.
2. Residuals: `r_i^ν = Σ_j g_j |P_i(ν_j − ν_i) − S_i s_j|² / Σ_j g_j` (normal field is the
   Weingarten field of `S_i`) and `r_i^c = Σ_j g_j (z_j − ½ s_jᵀ S_i s_j)² / (h_i² Σ_j g_j)`
   (neighbour centres lie on the osculating paraboloid).
3. Loss term `λ_jet · mean_i (r_i^ν + r_i^c)`, added in `rtgs.optim.trainer` beside
   `opacity_reg` / `scale_reg`, gradients flow into `quats` (through `ν_j, ν_i, e_1, e_2`)
   and `means` (through `Δ_j`). `λ_jet = 0.1` (one value, fixed before outcomes; a
   sensitivity arm at 0.01 and 1.0 is descriptive only).
4. First-order control: the same with `S_i ≡ 0` (co-planarity of neighbours and parallel
   normals — the GeoGaussian-type prior). Separates what the 2-jet adds over the 1-jet.

The term must be CPU-importable (Hard Rule 1) and behind a flag (`--jet-lambda`,
default 0 = off). No change to defaults.

## Data (frozen)

- Mesh: TOSCA cat0 hires, `~/Dropbox/Work/Datasets/obj/tosca_hires/cat0.obj` with the
  per-vertex colours of `cat0_colored_v1.ply` (same directory), copied under
  `dataset/external/tosca_cat0/` with a provenance file.
- Rendering: pyrender (EGL offscreen, verified on this machine 2026-10-07), 160 cameras on
  a Fibonacci sphere of radius 2.2 × bbox diagonal around the mesh centroid, looking at the
  centroid, 512×512, yfov 0.5, three fixed directional lights, white background, alpha mask
  from the depth buffer; written in the repository's calibrated-scene layout
  (`rgb/C0001.png …`, `mask/mask_C0001.png`, `calibration_dome.json` with the 4×4 view
  matrix and camera matrix). Scene in **mesh units** (the splat-unit scale of the companion
  model is 1/20.208; the evaluation rescales eigenvalues by the square of the scale).
- Split: 128 training / 32 held-out views, evenly spaced on the Fibonacci index.
- Ground truth for geometry: the mesh itself (trimesh closest-point distances, face normals
  at the closest point) and the P2 surface-FEM spectrum `λ_1..λ_32`
  (`SplatDiffuseLBO/data/cat_reference.npz`, splat units; P1/P2 agree to 0.5 %).

## Arms (three paired seeds 9561, 9562, 9563)

| id | arm |
|---|---|
| `base` | standard `rtgs refine` from 20 000 random points in the object bounds, 15 000 steps, densification 500–7 500, SH 1, masks on; `λ_jet = 0` |
| `jet1` | base + first-order control (`S ≡ 0`) at `λ_jet = 0.1` |
| `jet2` | base + jet-consistency prior at `λ_jet = 0.1` |
| `jet2_s` | descriptive sensitivity: `λ_jet ∈ {0.01, 1.0}`, seed 9561 only |

Identical schedules, learning rates and densification across arms; only the loss term
differs. Fresh process per cell; one GPU job at a time on the RTX 3050 (8 GB).

## Metrics

Primary (geometry, against the mesh, final model, splats with opacity > 0.3):
1. **Normal angle** to the mesh normal at the closest point: median and 90 % (degrees).
2. **Centre-to-surface distance**: median and 90 % (mesh units / bbox diagonal).
3. **Operator accuracy**: `max_{k≤10} |λ_k/λ_k^{P2} − 1|` of the SplatDiffuseLBO operator
   (its §16 configuration, run by `splat_lbo_cat.py` on the exported PLY) and the same for
   the Zhou–Lähner point-cloud Laplacian on the centres.

Secondary (appearance, must not degrade): held-out PSNR / LPIPS inside the mask, as in the
repository's standard evaluation.

## Predictions (stated before any training)

- `base` reproduces the companion model's regime: normal angle median 12–20°, centre
  distance 0.4–0.6 h.
- `jet2` vs `base`, every seed: normal median **< 8°** (the prior closes the null
  direction; the post-hoc field estimate reaches 7.4° on the companion model), centre
  distance median **halved**, held-out PSNR within **−0.2 dB**.
- `jet1` improves normals less than `jet2` (co-planarity flattens curvature: expected
  worse at the ears/paws, visible as a larger 90 % angle than `jet2`).
- Operator accuracy: `max_{k≤10} |err|` on `jet2` **below** that on `base` (companion v2:
  14 %); no number predicted for the absolute level — the thick shell was the dominant
  error and the prior targets exactly it.
- `λ_jet = 1.0` over-smooths (90 % angle up or PSNR down > 0.5 dB); `0.01` is between.

## Decision policy

Pass iff, in every paired seed, `jet2` beats `base` on normal median by ≥ 4° and on centre
distance median by ≥ 30 % with held-out PSNR no worse than −0.2 dB. Reject iff `jet2` loses
on either geometry metric in every seed. Anything else: inconclusive, logged. A pass is a
screening signal (one object, synthetic renders); a real-capture confirmation is a separate
registration (Hard Rule 7). No default promotion.

## Where this lives

- Companion theory, operator and reference data: `~/Documents/SplatDiffuseLBO`
  (`SPEC3D.md` §11 jet principle, §16 cat, §17 pointer to this experiment).
- This repository: this file; task `experiments/tasks/20261007_jet_consistency_prior_tosca_cat0.json`
  (draft — structural fields to be completed with `scripts/experiment_contract.py` by the
  implementer, then reviewed); implementation target `src/rtgs/optim/jet_prior.py` +
  `--jet-lambda` in `rtgs refine`; data preparation `scripts/experiments/prepare_tosca_cat0.py`.

## Follow-up (registered intent, not part of this protocol)

If this screen passes, the same prior is to be tried on the real captures in this
repository — Janelle (`dataset/2025_03_07_stage_with_fabric`) and the Karate frames
(`~/Documents/structsplat/results/core019_…`) — as a separately registered confirmation
(Hard Rule 7). Not before the synthetic screen runs cleanly end to end.
