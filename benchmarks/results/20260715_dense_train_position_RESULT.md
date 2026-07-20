# Result: dense train-only patch/epipolar matcher rejected before optimization

The sole official artifact is
`20260715T094311Z_cpu_dense_train_position.json` (SHA-256
`653af09c20659f3d102810e6ec1e55a2058c1ea66e5c648e03acc299ad1ecee2`). The protocol,
transparent pre-freeze probes, exact matcher, graph/precision gates, and stopping rule are in
`20260715_dense_train_position_PREREG.md`.

## Official command and outcome

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/dense_train_position_ablation.py \
  --output benchmarks/results/20260715T094311Z_cpu_dense_train_position.json
```

All three train-only graphs passed every structural floor, then all three failed the frozen strict
semantic precision floor. As preregistered, the harness stopped before constructing corrupted
depth inputs or running any Gradient/Hybrid optimization arm. Consequently there are no position-
engagement, local-localization, held-out, source-depth, PSNR, coverage, IoU, or control-utility
outcomes to interpret.

## Graph structure and precision

| seed | edges | represented nodes | coverage | vs sparse oracle | blocks | labeled represented nodes | strict true edges | strict precision | precision among labeled pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 177 | 239 | 18.34% | 2.25x | 35 | 41/239 (17.15%) | 16/177 | 9.04% | 80.00% |
| 1 | 187 | 247 | 19.10% | 2.47x | 35 | 55/247 (22.27%) | 22/187 | 11.76% | 70.97% |
| 2 | 165 | 227 | 17.99% | 1.91x | 34 | 49/227 (21.59%) | 18/165 | 10.91% | 78.26% |

The positive graphs exceeded the frozen floors of 160 edges, 220 nodes, 17.5% coverage, 1.85x
same-seed sparse coverage, 34 blocks, all nine views, and 16 nodes/view. Their strict precision was
only **9.04%-11.76%**, far below the required 60% in every seed. The shuffled graph had zero strict
true edges in all seeds, so semantic separation existed but could not rescue the failed absolute
precision gate.

The distinction between strict and labeled-pair precision explains the implementation pilot's
optimism. When both endpoints had a valid dominant reference-compositor label, 70.97%-80.00% of
positive pairs shared an identity. But only 17.15%-22.27% of represented nodes were validly
labeled; the preregistered audit counts every edge with an unlabeled endpoint as false. The matcher
therefore broadened coverage mainly through primitives with negligible/background GT contribution,
not through sufficiently many meaningful surface correspondences.

Raw-patch ratio confidence was not calibrated to that failure: median diagnostic confidence was
0.848/0.767/0.738 for seeds 0/1/2. The positive graph nevertheless satisfied its explicit
epipolar, angle, cheirality, and reprojection filters. This combination shows that calibration plus
near-identical local RGB patches is insufficient to establish that retained fitted primitives
represent the same scene element, especially in low-signal/background regions.

## Invariants and audit

- Matcher inputs were only the nine physically subset training RGB tensors, their cameras, and the
  detached retained center layout. Its API cannot receive `SceneData`, GT/sparse points, depth,
  scene bounds, Hybrid corruption, or held-out data.
- Positive/control tensors and hashes were frozen before the literal reference compositor was used
  for the synthetic audit. GT labels never selected, filtered, or weighted an edge.
- Positive/control degree, per-block endpoint multisets, camera-pair counts, confidence multiset,
  and edge count were exact; exact edge overlap was zero.
- Every graph floor passed. Reference-contribution parity stayed within `1.79e-7` alpha and
  `7.15e-7` depth max absolute error.
- Raw pairs/confidence, graph/input hashes, positive/control appearance, epipolar, ray, step-zero
  residual/Huber covariates, complete source hashes, command/config, environment, and dirty-tree
  provenance are serialized in the 359,063-byte artifact.
- Independent pre-official review found no release-blocking issue. Focused matcher/lifter tests,
  Ruff, formatting, compilation, and diff checks passed. The complete CPU-reference command
  `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passed, including the full non-slow suite and docs
  sync. Running the same script without hiding CUDA exposes a pre-existing host issue: an unrelated
  local `GaussianImage_plus/gsplat` shadows the expected gsplat package, causing six CUDA-path
  failures; no matcher code is involved.

The cyclic control is deliberately much less geometrically feasible (for example, seed-0 median
reprojection error 9.59 px versus 0.435 px positive, and mean step-zero Huber 0.0611 versus 0.0302).
This remained an acknowledged attribution limitation, but no optimization ran and the limitation
does not affect the precision rejection.

## Conclusion and stopping decision

This deterministic raw-patch CPU backend is a useful pluggable reference and tested protocol
boundary, but it is not a valid correspondence source for the frozen position loss. The experiment
does **not** test whether a learned high-quality train-only matcher plus position consistency would
improve global geometry, because the prerequisite graph-quality gate failed. It also does not
weaken the preceding result that privileged sparse correct edges localize their represented nodes.

Per the frozen stopping rule:

- do not tune patch radius, epipolar/ratio/reprojection thresholds, confidence, position lambda,
  Huber delta, norm, or schedule on this outcome;
- do not run the withheld 2x3 optimization arms and do not change the production default;
- close this raw-patch position-consistency branch rather than interpreting unlabeled/background
  coverage as useful density;
- pivot next to the Scholar-grounded local plane-pulling and shortest-axis normal-alignment
  constraint, scoped first to depth-backed Hybrid because Incremental Gaussian Triangulation's
  oriented-point assumption does not transfer honestly to RGB-only Gradient.

An optional RoMa-style learned matcher remains a future real/calibrated validation backend, not the
next synthetic threshold sweep and not a permissive CPU dependency.
