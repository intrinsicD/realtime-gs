# Jet-consistency prior, v2.1: collapse-resistant field residual on 2DGS surfels — preregistration — 2026-10-08

## Amendment history (disclosed; no screen outcome exists)

**v2.1 (2026-10-08, this text) replaces the v2 draft of the same day** after (i) the phase-1
implementation smoke and a λ_d diagnostic outside the protocol (seed 9561, 600 and 3600 steps,
training views only — these showed normal/centre/PSNR numbers of a b2 and a jet smoke cell and
are **pilot exposure**), and (ii) a static external review (Codex, read-only) of the prior, the
gsplat 2DGS kernel and the driver. Changes from v2, each with its reason:

1. **λ_d = 100 → 0.01·D/s.** gsplat's distortion is L1 in raw camera depth (scene in mesh
   units); the paper's is squared on normalized depth. `D` is divided by a frozen scene scale
   `s` (bbox diagonal of the sealed init), weight as in gsplat's 2DGS example. (LESSONS 1–2)
2. **Main prior = normal term only.** The centre term `r^c` pulls centres to the neighbourhood
   centroid, which lies *inside* a curved surface: a shrinkage force O(H) on every correctly
   placed splat. It is kept as a **descriptive ablation** (seed 9561), not in the treatment.
3. **Field tensor centred at the sheet centroid `m_i`**, not at `c_i` (sheet PCA; a far-off
   centre otherwise tips the smallest eigenvector into the tangent plane).
4. **Teacher guard:** residual only where the field estimate is well-posed — eigengap
   `(λ_2 − λ_1)/λ_3 ≥ 0.1` and effective support `Σ_j a_ij / max_j a_ij ≥ 4`; the fraction of
   guarded splats is reported per step bucket.
5. **"Collapse-proof" → "invariant under uniform replication at fixed bandwidth"** (what the
   test shows). The hard gate stays and is now **any-seed**: one seed above base + 1 point
   fails the arm (the v2 text contradicted its own code).
6. **Normal-consistency anchor starts at 7 500**, after densification, in every arm that uses
   it (otherwise it changes topology and `h⁰`, confounding the late-stage comparison).
7. **Metrics in fixed units**: centre gate in mesh units (bbox-diagonal fraction), not per-model
   `h̄` (that ratio rewards thinner coverage); added: mesh-area coverage (fraction of 20 000
   area-sampled mesh points with a splat within `0.5 h̄⁰`), signed normal displacement, normal
   p90 in the verdict, a **frozen step-7 500 cohort** evaluated beside the final subset,
   silhouette IoU on held-out views, collapse fraction before/after activation.
8. **Operator stage**: `b2` and `jet`, seed 9561 only, fixed CPU budget 8 h total; the
   companion's structural diagnostics (one zero mode, `cond(M)`, no localized modes) must
   pass or the spectrum does not count. Command `v2 a` deliberately: the operator sees the
   *trained* normals, which is the quantity under test.
9. **Declared, not fixed:** gsplat's opacity reset with `reset_every` never fires (known;
   `src/rtgs/optim/strategies.py`); all arms share this. The gsplat depth normal comes from
   per-surfel *centre* depth, not the ray–surfel intersection: the anchor is the
   "centre-depth consistency" proxy and its tilted-plane bias is measured on a tiled sheet
   before the run. The means learning rate is at 10 % by mid-run while the quaternion rate is
   constant: displacement and gradient magnitudes are logged so a null effect on centres is
   attributable.
10. **Pre-flight (LESSONS.md) is a stage of this protocol**: the review packet contains the
    ticked checklist, the teacher check (item 4 below) and the smoke, before the digest.

## Go / no-go before the screen: the teacher check

On the `b2` seed-9561 model at step 7 500 (one training run to that step, then stop): angle
to the mesh normal of (a) the trained `ν_i`, (b) the field normal `n_i` of the prior exactly
as implemented, (c) the fraction guarded. **Go** iff median(b) < median(a) − 3° and
p90(b) ≤ p90(a). Otherwise the screen is not run and this is the result.

## Primitive and anchor

**2DGS** via gsplat 1.5.3 `rasterization_2dgs` (new backend `rtgs.render.gsplat_2dgs_backend`
behind `rtgs.render.base.Rasterizer`; the model keeps `Gaussians3D` with the third scale
ignored and reported as `−∞`). 2DGS's own regularizers as in Huang et al. 2024: depth
distortion on the scale-normalized distortion `D/s`, `λ_d = 0.01` (amendment 1), and the
centre-depth normal-consistency proxy `λ_n = 0.05` starting at step 7 500 (amendments 6, 9),
both switchable.

## The field prior (v2, exact)

Frozen geometry of the neighbourhood: at the end of densification (step 7 500) compute
`h_i⁰` = mean distance to the 6 nearest centres and **freeze it** (it is never recomputed;
new splats after that step do not exist because densification is over).

At every step from 7 500 to 15 000, for each splat `i`, neighbours `j` within `3 h_i⁰`
(fixed radius, lists refreshed every 50 steps), `g_ij = exp(−|c_j − c_i|²/2(1.5 h_i⁰)²)`:
1. **Field structure tensor, density-normalized:** `C_i = Σ_j (w_j/ρ_j) g_ij (Σ_j + Δ Δᵀ)`,
   with `Δ = c_j − c_i`, `Σ_j = R_j diag(σ_j²) R_jᵀ` and `ρ_j = Σ_k g_jk` the local number
   density at `c_j` — a clone of `j` doubles `ρ` and halves each copy's weight: **duplicates
   do not count twice**. Field normal `n_i` = smallest eigenvector of `C_i` (detached), signed
   by `ν_i`; field sheet through the density-weighted centroid `m_i` of the same sum.
2. **Residuals** (per splat, no neighbour lists in the loss): treatment uses **`r_i^ν = 1 − (ν_i · n_i)²` only**;
   `r_i^c = ((c_i − m_i) · n_i)² / (h_i⁰)²` is the ablation term (amendment 2). Gradients flow into `quats_i` and `means_i` only
   (`n_i`, `m_i` detached). A copy of splat `i` has the same residual as `i`: cloning does not
   lower the mean.
3. Loss term `λ_jet · mean_i r_i^ν` (ablation: `+ r_i^c`), `λ_jet = 0.1`; no λ sensitivity arms
   (replaced by the ablation, amendment 2).
4. **Collapse gate (hard):** fraction of splats with a neighbour closer than `0.05 h_i⁰` at the
   end of training must not exceed the base arm's by more than one percentage point in any
   seed; a violating arm fails regardless of its other metrics.

No second-order term in this version: `S_i` is read off the field afterwards (§16.4 of the
companion), not trained. The 2-jet enters when the 1-jet is clean.

## Data (frozen, reused)

`dataset/external/tosca_cat0/` exactly as sealed for v1 (160 views, 128/32 split, same random
20 000-point initialization per seed). Ground truth: the mesh and
`SplatDiffuseLBO/data/cat_reference.npz`.

## Arms (seeds 9561, 9562, 9563)

| id | arm |
|---|---|
| `b2` | 2DGS with its own regularizers (`λ_d`, `λ_n`), `λ_jet = 0` — the control |
| `b2_noanchor` | 2DGS with `λ_d` only (`λ_n = 0`), `λ_jet = 0` — what the anchor does alone |
| `jet` | `b2` + field prior |
| `jet_noanchor` | `b2_noanchor` + field prior — is the anchor necessary? |
| `jet_c` | `jet` + centre term `r^c`, seed 9561, descriptive ablation |

15 000 steps, densification 500–7 500, SH 1, masks on, identical schedules; one GPU job at a
time; fresh process per cell.

## Metrics

Primary (opacity > 0.3, against the mesh): normal angle median and p90; centre-to-surface
median (in units of `h̄`); **collapse fraction** (gate); SplatDiffuseLBO operator
`max_{k≤10} |λ_k/λ_k^{P2} − 1}` with `CAT_PLY=… CAT_TAG=… python splat_lbo_cat.py v2 a`,
guard raised to 4 h per model, run **after** the screen, sequentially, on the seed-9561 models
only (CPU budget). Secondary: held-out PSNR/LPIPS, flatness `σ_min/σ_mid` (must be 0 by
construction — a check of the backend), Zhou–Lähner error on the centres.

## Predictions (before any run)

- `b2`: normals median **8–14°** (flat primitives + anchor; the downloaded surfel model had
  16.5° with unknown training), centre median `0.3–0.6 h̄`, flatness exactly 0.
- `b2_noanchor`: normals worse than `b2` by ≥ 3° in every seed (the anchor matters).
- `jet` vs `b2`, every seed: normal median **< 7°** and lower than `b2` by ≥ 3°; centre median
  ≤ `b2`; collapse fraction within the gate; PSNR within **−0.3 dB**.
- `jet_noanchor`: normals as good as `jet` **but** centre median not better than `b2_noanchor`
  — the prior aligns, the anchor positions. If `jet_noanchor` matches `jet` on both, the
  anchor is not necessary (that would be the surprising result, stated so it can be seen).
- Operator (seed 9561): `max_{k≤10} |err|` on `jet` below `b2`; absolute level not predicted
  (v2 of the companion on the downloaded model: 14 %).
- `jet_c` (centre term): centres move *inward* (signed displacement negative) at ears and paws; coverage not better than `jet`.

## Decision policy

Pass iff `jet` beats `b2` on normal median by ≥ 3° **and** on normal p90 in every seed, is no
worse on centre median (mesh units) or coverage in any seed, passes the collapse gate in every
seed, and PSNR ≥ −0.3 dB (≈ 7 % MSE; an engineering tolerance, stated as such). Reject iff `jet` loses on normals
in every seed or fails the collapse gate in every seed. Else inconclusive. (b) is answered by
`jet_noanchor` vs `jet` as stated above, descriptively. No default promotion; a real-capture
confirmation (Janelle / Karate) is a separate registration.

## Process notes carried over from v1

- The reviewer reads the generated task JSON fields before approving the digest; the review
  file is written only after that.
- Official `init-run` requires a clean worktree: the v1 implementation and this file are to be
  committed by the owner before this run, or the run is again a `--development` run (say which).
- The operator metric is a deferred, sequential CPU stage, not part of the GPU cells.

## Where this lives

Companion pointer: `~/Documents/SplatDiffuseLBO/SPEC3D.md` §17 (v1 result in §17.1).
Implementation targets: `src/rtgs/render/gsplat_2dgs_backend.py`, `src/rtgs/optim/jet_prior.py`
(`FieldPrior`), trainer flags `--rasterizer gsplat-2dgs --depth-distortion --normal-consistency
--jet-lambda --jet-start 7500`, driver `scripts/experiments/20261008_jet_field_prior_2dgs_tosca_cat0.py`.
