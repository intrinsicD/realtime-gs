# Result: four-point cross-view plane constructor rejected before optimization

The sole official artifact is
`20260715T110342Z_cpu_surface_plane_normal.json` (1,316,129 bytes; SHA-256
`07694e7388971936f918dd6c6d187df781e03f4bc44430f3099ea165728a9fd2`). The frozen
protocol, its transparent pre-outcome amendments, target gates, five withheld arms, and stopping
rule are in `20260715_surface_plane_normal_PREREG.md`.

## Official command and outcome

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/surface_plane_normal_ablation.py \
  --output benchmarks/results/20260715T110342Z_cpu_surface_plane_normal.json
```

All three corrupted-training-depth target sets passed every structural floor. All three then failed
the post-freeze clean-target audit. As preregistered, the harness emitted the sole stopped artifact
without evaluating initialization invariants or running any of the five 90-step Hybrid arms.
`arms`, `invariants`, and `summary` are empty for every seed. There are therefore no plane-loss,
normal-loss, held-out, source-depth, PSNR, coverage, IoU, or shuffled-control optimization outcomes
to interpret.

## Structural target audit

| seed | retained | targets | coverage | corrupted targets | min/source | farthest-neighbor p90 / extent | incidence p10 | lambda-mid/sum min | scale gap p10 | shuffle separation median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,303 | 339 | 26.02% | 98 | 26 | 0.07017 | 0.18511 | 0.01332 | 6.5368 | 0.4601 |
| 1 | 1,293 | 318 | 24.59% | 79 | 24 | 0.06383 | 0.18378 | 0.01488 | 6.5357 | 0.4984 |
| 2 | 1,262 | 326 | 25.83% | 67 | 31 | 0.06648 | 0.19264 | 0.01408 | 6.5223 | 0.4583 |

Every seed exceeded the frozen floors of 300 targets, 23% coverage, 60 corrupted targets,
20 targets per query source, 0.08 farthest-neighbor p90, 0.10 incidence p10, 0.01 middle-
eigenvalue ratio, 5.0 shortest-axis scale separation, and 0.25 shuffled-normal separation. All
nine local train views occurred as query, candidate-pool, and selected-support views. The correct
and per-source shuffled target tensors were frozen and hashed once before clean labels were used.

## Post-freeze clean-target audit

The plane criterion required p90 clean-point-to-target-plane residual at most `0.10 * extent` in
both the all-target and corrupted-target strata. The normal criterion required median absolute
cosine with the validity-aware clean source normal of at least `0.50` in both strata.

| seed | all plane p90 | all normal median | corrupted plane p90 | corrupted normal median | seed pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.17451 (fail) | 0.58694 (pass) | 0.24711 (fail) | 0.64439 (pass) | no |
| 1 | 0.16042 (fail) | 0.48165 (fail) | 0.23910 (fail) | 0.37262 (fail) | no |
| 2 | 0.16475 (fail) | 0.52651 (pass) | 0.27077 (fail) | 0.49066 (fail) | no |

All 983 targets were clean-labelable, so missing labels did not cause the rejection. Instead, the
all-target plane p90 exceeded the frozen limit by 60%-75%, and the corrupted-target p90 exceeded it
by 139%-171%. Normal agreement was also unstable: only seed 0 passed both normal strata. The local
PCA filters therefore selected spatially compact, numerically planar, reachable neighborhoods that
were not sufficiently faithful to the clean surface planes, especially where the input depth was
corrupted.

## Validity and provenance

- The target builder received only corrupted metric training depths, the nine physically subset
  train cameras, the detached retained layout, and synthetic bounds derived from allowed sparse
  points. It did not receive clean depth, corruption masks, ground-truth identities, or held-out
  data.
- Clean depth was introduced only after the correct/control tensors and hashes were frozen. It
  audited those tensors but did not filter, repair, or weight them.
- Revision `2dddca4aff59702341af9faceefa76ad2505dd83` was dirty. The artifact records the full
  status, command/config/environment, tracked-diff hash
  `60ca73b89e930b841065492b0bf1d98d5eab8e806c081e9c35b411acc4021cf0`, and loaded source-tree
  hash `0ffe69b81cc27f23445a8e6651d7326dfd87a2e559a3df143d2ab674083344c2`.
- An independent post-run audit reproduced every structural/clean check, confirmed zero arms ran,
  and verified that the recorded preregistration, harness, `surface.py`, `gradient.py`, and
  `hybrid.py` hashes still matched the files used for the run.
- A pre-official static review found two decision-labeling deviations. They were repaired before
  any target construction or outcome access: local/global insufficiency now includes every frozen
  gain and seed-win requirement, and a thin-surface-only gain has its own covariance-initialization
  attribution. No arm, coefficient, threshold, seed, fit, or schedule changed.

## Conclusion and stopping decision

Reject this deterministic four-neighbor cross-view PCA target constructor. Its structural
diagnostics are not reliable proxies for plane correctness under the tested block-corrupted metric
depth. Do not tune neighbor count, radius, planarity, incidence, audit thresholds, loss weights, or
schedule on these three outcomes, and do not run the withheld arms.

This result does **not** test whether point-to-plane pulling or shortest-axis normal alignment can
help when supplied with valid oriented points. The losses remain CPU-tested, pluggable, and disabled
by default, but no production behavior changes and no positive utility claim is authorized.
Incremental Gaussian Triangulation assumes oriented RGB-D surface points; this repository-specific
nearest-neighbor construction was a deliberately weaker adaptation, not a reproduction.

The next valid evidence should use an independently justified oriented-point source on actual
calibrated metric depth/RGB-D data, exposed through a pluggable backend and audited before loss
optimization. It should not be another threshold sweep or a clean-label-assisted repair of this
synthetic constructor.
