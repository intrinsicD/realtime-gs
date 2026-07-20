# Dense train-only position matcher gate

## Frozen artifact

- Official JSON: `benchmarks/results/20260715T094311Z_cpu_dense_train_position.json`
- Artifact SHA-256: `653af09c20659f3d102810e6ec1e55a2058c1ea66e5c648e03acc299ad1ecee2`
- Loaded-source tree SHA-256: `cbb1d699f112906c10056247c092d5347f4cf116f028637be753100d9844c058`
- Protocol: `benchmarks/results/20260715_dense_train_position_PREREG.md`
- Audit: `benchmarks/results/20260715_dense_train_position_RESULT.md`

## Structural and strict semantic gates

| seed | edges | nodes | node coverage | prior coverage multiplier | blocks | labeled nodes | strict true/total | strict precision | labeled-pair precision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 177 | 239 | 18.342% | 2.255x | 35 | 41/239 | 16/177 | 9.040% | 80.000% |
| 1 | 187 | 247 | 19.103% | 2.470x | 35 | 55/247 | 22/187 | 11.765% | 70.968% |
| 2 | 165 | 227 | 17.987% | 1.908x | 34 | 49/227 | 18/165 | 10.909% | 78.261% |

Every graph passed the frozen structural floors: at least 160 edges, 220 nodes, 17.5% node
coverage, 1.85x same-seed sparse coverage, 34 camera-pair blocks, all nine train views, and 16
nodes/view. Every graph failed the frozen 60% strict semantic precision floor. Shuffled strict true
edge count was zero for all seeds. The protocol therefore withheld all Gradient/Hybrid arms; the
artifact intentionally contains no optimization or utility summaries.

## Mechanism diagnosis

- Valid dominant-contribution labels covered only 17.15%/22.27%/21.59% of represented nodes.
- Positive precision conditional on both endpoints being labeled was 80.00%/70.97%/78.26%.
- Median ratio-derived confidence was nevertheless 0.848/0.767/0.738, so the confidence was not
  calibrated to meaningful surface contribution.
- Positive seed-0 median reprojection error was 0.435 px versus 9.59 px for the cyclic control;
  mean step-zero Huber was 0.0302 versus 0.0611. These serialized differences preserve the known
  control limitation but do not affect the pre-optimization precision rejection.

## Integrity

- Matcher inputs: physically subset train RGB, train cameras, detached retained `(xy, view ids,
  ranges)` only. The matcher API cannot receive scene GT, depth, sparse points, bounds, corruption,
  or held-out data.
- Positive and control hashes were fixed before reference-compositor identity audit.
- Exact degree, per-block endpoint multisets, camera-pair counts, confidence multiset, and edge
  count were preserved; positive/control exact-edge overlap was zero.
- `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passed the complete CPU-reference verification and
  docs sync. An unhidden run exposed an unrelated local incompatible gsplat shadow on CUDA paths.

## Decision

Reject the raw-patch matcher as a correspondence source; retain its pluggable CPU reference and
tests; do not tune matcher or position-loss settings; pivot to depth-backed local plane pulling and
shortest-axis normal alignment.
