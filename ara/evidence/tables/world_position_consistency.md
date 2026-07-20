# Fixed-match world-frame position consistency

## Forensic binding

- Official artifact: `benchmarks/results/20260715T084557Z_cpu_world_position_consistency.json`
- Preregistration: `benchmarks/results/20260715_world_position_consistency_PREREG.md`
- Post-run audit: `benchmarks/results/20260715_world_position_consistency_RESULT.md`
- Artifact SHA-256: `5d04fa3793d6fd3064f2b85b92088de1c9245c8ae490e05118e79214f9b0f1d6`
- Source-tree SHA-256: `709aecb5fda9b72058aca4d496f847f293de5e3f6c958c4ce24444983575a89f`
- Revision: `2dddca4aff59702341af9faceefa76ad2505dd83` plus embedded dirty-worktree provenance
- Protocol: CPU reference renderer, four threads, seeds 0/1/2, Gradient and corrupted-depth
  Hybrid, 90 lift steps, none/correct/degree-shuffled arms, lambda 0.25, Huber delta 0.05

Three independent post-run audits reproduced all 37 source hashes, graph/layout hashes, 18 arm
histories, six 90-step schedules per seed, 120 summary cells, and every decision field.

## Primary utility

| Family | Metric | None | Correct | Shuffled | Correct gain | Seed wins | Frozen floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient | held-out RMSE/extent | 0.154307 | 0.152924 | 0.154651 | 0.896% | 3/3 | 2% |
| Gradient | all-source abs-rel p90 | 0.211965 | 0.198190 | 0.199052 | 6.499% | 3/3 | 10% |
| Hybrid | held-out RMSE/extent | 0.150330 | 0.148820 | 0.150548 | 1.005% | 3/3 | 2% |
| Hybrid | all-source abs-rel p90 | 0.163228 | 0.153655 | 0.159025 | 5.865% | 3/3 | 10% |
| Hybrid | corrupted-source abs-rel p90 | 0.205832 | 0.194122 | 0.200271 | 5.689% | 3/3 | 15% |

PSNR, coverage, and alpha-IoU guardrails passed. Material utility and correspondence attribution
failed in both families because all applicable geometry floors are conjunctive.

## Local mechanism and graph

| Family | Correct-edge p90 gain | Wins | Assigned-GT p90 gain | Wins |
| --- | ---: | ---: | ---: | ---: |
| Gradient | 91.11% | 3/3 | 90.00% | 3/3 |
| Hybrid | 86.44% | 3/3 | 82.42% | 3/3 |

- Edges: 169 / 140 / 175; represented nodes: 106 / 100 / 119.
- Represented-node fraction: 8.14% / 7.73% / 9.43%; represented GT identities: 27 / 30 / 33.
- Correct closest-ray gap p90: 3.24% / 3.34% / 3.06% of extent.
- Correct midpoint-to-assigned-GT p90: 4.03% / 4.00% / 4.97% of extent.
- Gradient control separation failed because shuffled topology preserved 93.7% of the correct
  all-source gain; Hybrid control separation passed, but materiality still failed.
- The shuffled graph matches degree, endpoints, camera-pair counts, and baselines, not feasibility
  or force magnitude: shuffled closest-ray gap p90 was 48.3%-51.2% of extent.

## Frozen outcome

`denser_train_only_matcher_without_loss_sweep`; no production-default change, deployability claim,
or coefficient/delta/norm/schedule sweep is authorized.
