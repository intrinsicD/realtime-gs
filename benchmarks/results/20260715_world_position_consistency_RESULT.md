# Fixed-match world-frame position-consistency result

This is the post-run audit for
`20260715T084557Z_cpu_world_position_consistency.json`. The protocol and its transparent
pre-official amendments are in `20260715_world_position_consistency_PREREG.md`.

## Validity

- The sole official command was `CUDA_VISIBLE_DEVICES='' .venv/bin/python
  benchmarks/world_position_consistency_ablation.py --output
  benchmarks/results/20260715T084557Z_cpu_world_position_consistency.json`.
- The artifact is 510,468 bytes with SHA-256
  `5d04fa3793d6fd3064f2b85b92088de1c9245c8ae490e05118e79214f9b0f1d6`.
- Revision `2dddca4aff59702341af9faceefa76ad2505dd83` was dirty; the artifact embeds the full
  status, tracked-diff hash, command, environment, exact config, every loaded repository source
  hash, and aggregate source-tree hash
  `709aecb5fda9b72058aca4d496f847f293de5e3f6c958c4ce24444983575a89f`.
- Seeds 0/1/2 used the frozen Gradient/Hybrid x none/oracle/degree-shuffled design. Every arm
  passed step-zero, lambda-zero, finite-history, bounded-ray, inclusive-render, full-view-schedule,
  output-count, pair-hash, graph, and cross-family layout/schedule/count invariants.
- The correct graphs had 169/140/175 edges, 106/100/119 represented primitives, 32/31/31
  non-singleton camera-pair blocks, and 27/30/33 represented GT identities. Exact endpoint degree,
  per-block endpoint multisets, camera-pair counts, and zero semantic/exact edge overlap held for
  the shuffled controls.
- The CPU verification command `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passed after the run.
  All embedded hashes for the harness, preregistration, and lifter sources still matched during
  this audit.

The artifact is valid for the amended preregistered decision. The oracle topology remains a
privileged synthetic upper bound and cannot authorize production or deployability claims.

## Primary result

### GradientLifter

| Metric | None | Correct position | Shuffled position | Correct gain | Wins | Required | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Held-out depth RMSE/extent | 0.154307 | 0.152924 | 0.154651 | 0.896% | 3/3 | >=2%, >=2/3 | no |
| All-source absolute-relative p90 | 0.211965 | 0.198190 | 0.199052 | 6.499% | 3/3 | >=10%, >=2/3 | no |
| Held-out PSNR | 19.6894 dB | 19.6769 dB | 19.6334 dB | -0.0125 dB | -- | >=-0.10 dB | yes |

Coverage stayed 1.0 and alpha IoU changed by +0.00016. The position term engaged, but the two
global geometry thresholds failed, so the material-effect and correspondence-attribution decisions
are false.

### Corrupted-depth HybridLifter

| Metric | None | Correct position | Shuffled position | Correct gain | Wins | Required | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Held-out depth RMSE/extent | 0.150330 | 0.148820 | 0.150548 | 1.005% | 3/3 | >=2%, >=2/3 | no |
| All-source absolute-relative p90 | 0.163228 | 0.153655 | 0.159025 | 5.865% | 3/3 | >=10%, >=2/3 | no |
| Corrupted-source absolute-relative p90 | 0.205832 | 0.194122 | 0.200271 | 5.689% | 3/3 | >=15%, >=2/3 | no |
| Held-out PSNR | 19.6725 dB | 19.6891 dB | 19.6502 dB | +0.0166 dB | -- | >=-0.10 dB | yes |

Coverage stayed 1.0 and alpha IoU changed by -0.000002. All three geometry metrics improved in
every seed, but none reached its frozen materiality threshold. Material effect and correspondence
attribution are therefore false.

## Mechanism and control evidence

The intervention itself worked strongly on graph-supported primitives:

| Family | Correct-edge p90, none -> correct | Gain / wins | Assigned-GT p90, none -> correct | Gain / wins |
| --- | ---: | ---: | ---: | ---: |
| Gradient | 0.680237 -> 0.060468 | 91.11% / 3/3 | 0.407361 -> 0.040727 | 90.00% / 3/3 |
| Hybrid | 0.509400 -> 0.069089 | 86.44% / 3/3 | 0.261013 -> 0.045890 | 82.42% / 3/3 |

Both the engagement and local-geometry gates pass in both families. The graph is geometrically
plausible: correct-ray closest-distance p90 was 3.06%-3.34% of scene extent and triangulated
midpoint-to-assigned-GT p90 was 4.00%-4.97%. Yet only 7.73%-9.43% of retained primitives were graph
nodes. Global nearest-GT p90 was essentially flat for Gradient and slightly worse for Hybrid,
consistent with a strong local correction that was too sparse to move whole-scene metrics enough.

The degree-shuffled control gives mixed attribution evidence. In Gradient it preserved 93.7% of
the correct arm's all-source p90 gain, so that source-tail change is not correspondence-specific
under the frozen control. In Hybrid, correct topology beat shuffled topology in 3/3 seeds on every
primary geometry metric and the shuffled arm preserved at most 47.5% of the correct gain, so the
control-separation test passes. Neither observation rescues the failed material-effect gates.

The control matches graph degree, endpoints, camera-pair counts, and baselines, but deliberately
does not match geometric feasibility or initial residual/gradient magnitude. Correct closest-ray
gap p90 was 3.06%-3.34% of extent versus 48.3%-51.2% for shuffled edges; family-specific step-zero
correct-edge p90 was 0.49-0.57 versus 0.93-0.95 for shuffled Gradient, and 0.51-0.55 versus
0.98-1.07 for shuffled Hybrid. Any later positive claim must remain scoped to this degree-matched
derangement.

## Decision

The position-only oracle loss is **locally effective but coverage-limited**. It does not satisfy
the preregistered global utility gates, so it does not change the inclusive production default and
does not establish a deployable correspondence method. Apply the frozen stopping rule:

- stop coefficient, Huber delta, norm, and schedule sweeps for this position loss;
- do not add shape or appearance consistency yet;
- run one denser, train-only matcher experiment with the same bounded-ray position loss, using a
  pluggable matcher and frozen mutual-confidence/reprojection/angle filtering; and
- if denser train-only coverage does not propagate the local mechanism to global geometry, stop
  this position branch and pivot to the local plane/normal constraint branch.

This result supports the narrower claim that correct fixed cross-view edges can pull represented
ray-bounded primitives near their physical synthetic identities. It does not show that the tested
sparse topology materially improves the reconstruction as a whole, that GT-free matching will
retain this purity, or that the effect transfers to real monocular-depth scenes.
