# Jet-consistency prior for 3DGS training — result — 2026-10-07

Task `20261007_jet_consistency_prior_tosca_cat0`, PREREG `benchmarks/results/20261007_jet_consistency_prior_tosca_cat0_PREREG.md`, run `runs/20261007_jet_consistency_prior_tosca_cat0/`
(development init-run, dirty worktree, digest `03e3b661…c35a1`). 11 cells, each a fresh process,
gsplat 1.5.3, RTX 3050. Machine record: `20261007_jet_consistency_prior_tosca_cat0_RESULT.json`; raw per-cell values
`runs/20261007_jet_consistency_prior_tosca_cat0/comparison.json`.

## Decision (frozen PREREG policy)

**Reject.** jet2 is worse than base on the normal median *and* on the centre median in every
paired seed (rule: reject iff it loses on either geometry metric in every seed). It also loses
2.9–3.7 dB of held-out PSNR. The first-order control jet1 fails the same way. No default changes.

## Measured table (opacity > 0.3 subset; mesh units; diagonal 223.537)

| arm | seed | N | N(op>0.3) | normal median ° | normal p90 ° | centre median (mesh u.) | centre median / diag | centre / h median | ZL max k≤10 err | held-out PSNR dB | held-out LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 9561 | 16861 | 11512 | 14.92 | 64.63 | 0.787 | 0.00352 | 1.35 | 0.387 | 37.39 | 0.0022 |
| jet1 | 9561 | 28132 | 12280 | 41.76 | 80.94 | 2.288 | 0.01024 | 62.89 | 0.427 | 33.83 | 0.0047 |
| jet2 | 9561 | 25734 | 11706 | 39.53 | 78.40 | 2.475 | 0.01107 | 31.52 | 0.633 | 33.65 | 0.0048 |
| base | 9562 | 16485 | 11347 | 14.58 | 65.80 | 0.787 | 0.00352 | 1.29 | 0.380 | 37.32 | 0.0022 |
| jet1 | 9562 | 27577 | 11613 | 41.65 | 82.01 | 2.282 | 0.01021 | 37.14 | 0.454 | 33.68 | 0.0046 |
| jet2 | 9562 | 25215 | 11364 | 38.29 | 78.79 | 2.272 | 0.01017 | 27.19 | 0.559 | 34.42 | 0.0040 |
| base | 9563 | 16140 | 11425 | 14.97 | 63.22 | 0.776 | 0.00347 | 1.28 | 0.389 | 37.41 | 0.0022 |
| jet1 | 9563 | 27084 | 11839 | 39.50 | 78.77 | 2.139 | 0.00957 | 51.17 | 1.000 | 34.22 | 0.0043 |
| jet2 | 9563 | 25025 | 11745 | 37.72 | 78.12 | 2.356 | 0.01054 | 30.59 | 0.563 | 34.41 | 0.0041 |
| jet2_l001 | 9561 | 16385 | 9455 | 16.83 | 68.66 | 0.968 | 0.00433 | 3.59 | 0.237 | 36.50 | 0.0026 |
| jet2_l1 | 9561 | 30614 | 11840 | 39.57 | 78.03 | 2.368 | 0.01059 | 36.18 | 1.000 | 32.61 | 0.0062 |

Paired gate, jet2 vs base (pass needs ≥ +4°, ratio ≤ 0.70, ≥ −0.2 dB):

| seed | normal-median gain ° | centre-median ratio | PSNR Δ dB | gate |
|---|---:|---:|---:|---|
| 9561 | -24.62 | 3.14 | -3.74 | fail |
| 9562 | -23.71 | 2.89 | -2.91 | fail |
| 9563 | -22.75 | 3.04 | -3.00 | fail |

jet1 vs base (descriptive, same quantities):

| seed | normal-median gain ° | centre-median ratio | PSNR Δ dB |
|---|---:|---:|---:|
| 9561 | -26.84 | 2.91 | -3.56 |
| 9562 | -27.06 | 2.90 | -3.65 |
| 9563 | -24.53 | 2.76 | -3.19 |

## Predictions, marked

| PREREG prediction | measured | mark |
|---|---|---|
| base normal median 12–20° | 14.58–14.97° | **right** |
| base centre distance 0.4–0.6 h | 1.28–1.35 h (0.78 mesh units ≈ 0.039 splat units) | **wrong** |
| jet2 normal median < 8° in every seed | 37.7–39.5° | **wrong** |
| jet2 centre median halved in every seed | ×2.89–3.14 (tripled) | **wrong** |
| jet2 held-out PSNR within −0.2 dB | −2.91 to −3.74 dB | **wrong** |
| jet1 improves normals less than jet2, larger 90 % angle than jet2 | jet1 p90 78.8–82.0° vs jet2 78.1–78.8° (larger in every seed), but neither improves: both worsen the median by 23–27° | **ordering right, premise wrong** |
| operator `max_k≤10` err on jet2 below base | not evaluated (operator deferred, see below); Zhou–Lähner on the same centres: base 0.380–0.389, jet2 0.559–0.633 | **open** |
| λ = 1.0 over-smooths (90 % angle up or PSNR down > 0.5 dB) | p90 78.0° vs 64.6°, PSNR −4.78 dB (seed 9561) | **right as stated; mechanism is collapse, not smoothing** |
| λ = 0.01 is between | normal 16.83° (base 14.92, jet2 39.53), PSNR 36.50 dB (37.39 / 33.65) | **right** |

## What happened (post-hoc diagnostic, not a protocol metric)

The loss has a degenerate minimiser the PREREG did not anticipate: **neighbour collapse**. Both
residuals vanish when the k = 16 neighbours of a splat are near-copies of it (Δ_j → 0, ν_j = ν_i),
and gsplat's clone step creates exactly such copies. With the prior on, the share of splats whose
nearest neighbour lies within 0.01 mesh units is 64–73 % (jet1, jet2, λ = 1) and 44 % (λ = 0.01),
against 2 % in base (seed 9561); the median 6-NN spacing h falls from 0.41 to 0.002–0.016, and the
population grows (25–31 k vs 16–17 k). Neighbourhoods then measure nothing geometric: the stacks
keep large, poorly oriented scales (max-scale median 1.5–1.6 vs 1.06), so normals and centres get
worse and appearance drops. `centre / h` of 27–63 in the treated arms is the same collapse seen
through h. A fix (a minimum neighbourhood radius, e.g. neighbours by radius ≥ the splat's own
tangential scale, or de-duplication of near-coincident centres) would be a new registration.

## Deviations and boundaries

- The SplatDiffuseLBO operator metric was not run (`deferred_metrics` in the task): its CLI reads a
  hard-coded PLY and the v3 assembly needs > 11 GB host RAM. Every cell exports
  `gaussians_splat_frame.ply` (companion frame) for a later run.
- Paraboloid sign implemented as `z_j + ½ s_jᵀ S_i s_j` (companion code convention, `S = I/R` on a
  sphere with outward normals); RGB as JPEG q100 4:4:4. Both fixed before outcomes.
- Training cells were launched directly with the coordinator's worker command (one at a time)
  after the user allowed training while the companion CPU job ran; evaluation and publication ran
  through the coordinator.
- One synthetic object, three seeds, one λ; development screen. The review record binds the
  reviewer's content approval of the PREREG to the generated digest; the audit below is a
  self-audit by the task owner, not an independent one.
