# Scientist pass — Beam-track covariance refit on Janelle

Date: 2026-07-23

Verdict: **qualified negative for covariance estimation; narrow positive for an accidental
scale/coverage mechanism; no default change**.

Machine-readable audit:
[`20260723_beam_covariance_refit_AUDIT.json`](20260723_beam_covariance_refit_AUDIT.json)

Protocol:
[`20260723_beam_covariance_refit_PREREG.md`](20260723_beam_covariance_refit_PREREG.md)

Result:
[`20260723_beam_covariance_refit_RESULT.md`](20260723_beam_covariance_refit_RESULT.md)

## Claim inventory and disposition

| Claim | Kind and scope | Evidence | Disposition |
| --- | --- | --- | --- |
| Beam Fusion produces partial correspondences | Implemented/measured, compact Janelle development | Deterministic replay: 800 CSR rows, 6,029 links, at most one source Gaussian/view/row | **Confirm**, specifically Gaussian-to-Gaussian lineage rather than dense pixel correspondences |
| Beam tracks improve 3D covariance estimation through linear LS | Causal mechanism claim, fixed means/tracks | 635/800 raw results non-SPD; bounded result median whitened residual 13.4478 versus CI 0.6888 | **Retire** for the tested solver |
| Robust covariance descent repairs LSQ and improves over CI | Causal mechanism/pipeline claim | Median residual 0.6350, only 7.80% below CI; all three pipeline gates fail | **Retire** |
| LSQ improves visible initialization and fixed-topology convergence | Measured, fitted-view CPU development only | Alpha IoU 0.01073→0.55056; PSNR AUC +9.108%; final FG PSNR +0.5569 dB | **Confirm narrowly** |
| LSQ's useful effect is evidence for a physical 3D covariance | Interpretation | Median max sigma 4.59× CI, median min sigma at floor, condition 178,541, failed reprojection gate | **Retire**; effect is anisotropic inflation after invalid-SPD repair |
| The result supports a production default | Capability/default claim | One scene, one seed, all cameras fitted, Torch CPU, 800 fixed topology | **Not authorized** |

## Independent checks

The audit passed **75/75** checks:

- verified the compact manifest plus all 26 payload hashes and byte counts;
- verified protocol, executed-harness reconstruction, synthetic-test, imported-helper, and
  post-run-fix source bindings;
- reran deterministic Beam Fusion and recovered exactly 800 components and 6,029 links;
- checked every arm has 800 finite initial/final PLY rows and exact repeated PLY hashes;
- checked all 40 checkpoints/arm, timing-free dynamics, and PSNR trajectories repeat exactly;
- checked CI replay equals the saved initialization and that means/opacity/SH are bit-identical
  across arms;
- independently recomputed relative and whitened covariance residual distributions from the
  saved initial PLYs;
- independently replayed the raw LS system, 635 non-SPD count, eigenvalue projection, AUC,
  first-threshold iteration, and every frozen decision gate.

Audit command:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/audit_beam_covariance_refit.py \
  --primary runs/beam_covariance_refit_20260723 \
  --repeat runs/beam_covariance_refit_repeat_20260723 \
  --out benchmarks/results/20260723_beam_covariance_refit_AUDIT.json
```

## Referee findings

1. **The direct hypothesis fails before downstream metrics can rescue it.** Track-LSQ passes the
   coverage and optimization gates but fails the primary covariance-consistency gate by a large
   margin. It must not be labeled a better covariance estimator.
2. **SPD projection is the intervention that matters.** The unconstrained six-parameter solution
   is invalid for 79.375% of tracks. Clamping negative axes to `1e-4` while retaining a wide
   positive axis creates surface-aligned needles and destroys the raw least-squares
   interpretation.
3. **Robust descent falsifies the “just optimize covariance” proposal in this exact form.** As the
   whitened objective removes LSQ inflation, visible coverage and convergence return to CI.
4. **Lineage coverage is limited.** Only 4,704 unique fitted 2D Gaussians, 11.76% of the selected
   input pool, occur in the 6,029 links. This is lineage for retained 3D hypotheses, not complete
   surface coverage.
5. **All quality metrics are fitted-view metrics.** No held-out, multi-scene, CUDA/gsplat,
   adaptive-density, split/merge, or teleport behavior was tested. CPU timings are
   non-decisional.
6. **Source binding is qualified.** The official harness and tests were preregistered by hash, but
   its dirty imported convergence helper was not separately pre-hashed. The unchanged post-run
   hash and exact repeat make the development result usable, not fully pre-bound.
7. **The viewer repair is non-scientific and isolated.** The original harness wrote
   `../../../runs` instead of `../../runs` into its comparison manifest. The adjacent
   reverse-applicable patch changes only these two strings and reconstructs the exact executed
   harness hash.

## Corrections enforced

- Replace “correspondences give better 3D covariance” with “Beam Fusion exposes lineage, but the
  tested constrained covariance inference failed.”
- Replace “LSQ covariance improves coverage” with “invalid-SPD LSQ repair acts as a useful
  anisotropic scale heuristic on this fitted-view screen.”
- Do not call robust descent a partial win: its 7.80% median residual reduction misses the frozen
  20% gate, its tail mean is worse, and all pipeline gates fail.
- Do not promote LSQ post hoc because its downstream numbers are attractive.
- Frame explicit covariance/scale inflation as the next preregistered experiment, with
  outside-mask alpha and held-out replication, not as an already established fix.

## Evidence still needed for promotion

A promotion candidate needs multiple scenes/seeds, a train-only selection role with untouched
held-out cameras, the actual production gsplat/density path, matched count/budget, mask leakage
guards, and an independently audited replay-complete source bundle. A covariance estimator must
first pass PSD and reprojection gates; a coverage heuristic should be named and evaluated as such.
