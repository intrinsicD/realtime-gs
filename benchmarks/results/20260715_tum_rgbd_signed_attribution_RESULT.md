# Result: signed TUM RGB-D occlusion/rigidity attribution

The official development artifact is
`20260715T160300Z_cpu_tum_rgbd_signed_attribution_sitting.json` (SHA-256
`ca62497b71cef78bf046d5d4b463a2a71652ac1fc80ca4ce41e3b85b17cb5dfb`). The frozen decision is
`20260715_tum_rgbd_signed_attribution_DECISION.json`; it records
`confirmation_authorized=false`. The walking attempt seal does not exist, and no walking archive
member, manifest, RGB PNG, or depth PNG was opened.

## Frozen execution

The preregistration, source acquisition record, implementation/dependency hashes, and 21 focused
synthetic/legacy tests were bound before the first sitting PNG decode by
`20260715_tum_rgbd_signed_attribution_PREDECODE_SEAL.json` (SHA-256
`6605ad557b819d9b1f64a39f84a17badfd5fd59ed5909a5adcd79fdb5f645671`). Its implementation
aggregate is `75b0dbb6a84548852dd6de4c5623557c78168ccc305d93ff1a4664f4f4347f3d`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_tum_rgbd_signed_attribution.py tests/test_tum_rgbd_oriented_validity.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src \
  .venv/bin/python benchmarks/tum_rgbd_signed_attribution.py \
  --phase development \
  --archive /home/alex/.cache/rtgs/tum-rgbd/rgbd_dataset_freiburg3_sitting_xyz.tgz \
  --output benchmarks/results/20260715T160300Z_cpu_tum_rgbd_signed_attribution_sitting.json \
  --decision-output benchmarks/results/20260715_tum_rgbd_signed_attribution_DECISION.json \
  --threads 4
```

The source SHA-256 was
`05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205`. The harness selected 64
of 66 pose keyframes as 48 T, eight V, and eight H views. It decoded exactly the 56 selected T/V
depth members once, zero RGB and zero H payloads, in 12.67 seconds. It constructed 29,268 eligible
audit targets and a 165,633-point dense T-only set (the exact targets plus valid stride-8 T
points).

## Outcome

The development gate **failed 2 of 12 comparisons**, so the experiment stopped before walking:

| quantity | sparse | dense T-only | change / gate |
| --- | ---: | ---: | --- |
| depth-valid pairs | 176,950 | 155,416 | 87.83% retained; pass |
| supported targets | 27,525 | 27,135 | dense support pass |
| target-balanced positive rate | 11.674% | 10.250% | -1.424 pp; **fails** required -1.751 pp |
| target-balanced negative rate | 6.347% | 6.468% | +0.121 pp; pass <=+1 pp |
| total contradiction rate | 18.021% | 16.718% | -1.303 pp |
| target median-relative-depth p90 | 5.226% | 4.809% | ratio 0.920; pass |

The construction control removed 21,534 depth-valid pairs (12.17%). Removed pairs were strongly
sign-skewed: 30.106% positive and 2.898% negative. It captured 32.48% of all sparse positive
contradictions but only 5.52% of negative contradictions. Among 10,473 targets having both removed
and retained observations, `E+=0.13813` (one-sided-95%/two-sided-90% interval
`[0.13148,0.14436]`) and positive selectivity was the same because `E-=-0.00743`; both gates
passed. The target-balanced removed/retained positive risk ratio was only **1.7195**, however,
with log-ratio interval `[0.51673,0.56503]`, or ratio `[1.6767,1.7595]`, entirely below the frozen
2.0 floor. The common-target positive reduction was positive but only about 1.23 pp (interval
`[1.163,1.304]` pp).

The secondary time audit found a dense-visible far-minus-near contradiction increase of
**11.19 pp** across 11,884 paired targets, interval `[10.63,11.79]` pp. The preregistered
pose-conditioned sensitivity was estimable in four cells and remained +10.01 pp. This is evidence
that elapsed capture state still matters in the slow-motion sequence after the dense visibility
control, but it cannot distinguish moving people, view-dependent missing depth, and remaining
pose/visibility differences without the withheld cross-regime confirmation.

## Interpretation and stop

The result is **partial mechanism evidence but a preregistered rejection**. Densifying the T-only
z-buffer preferentially found behind-observed pairs and modestly improved the heavy tail; this is
consistent with some construction-predicted occlusion. It did not deliver the target-balanced
effect size required to claim that sparse visibility explains enough of the tail. Pair-weighted
enrichment alone would overstate the result because residuals cluster by target, which is why the
target-cluster gate was primary.

The decision manifest forbids walking confirmation under this implementation. Do not relax the
2x or 15% floors, change stride/tolerance, rerun sitting as confirmatory, or decode walking for a
post-hoc contrast. No oriented-plane or ordinary-depth utility run is authorized.

The most informative next experiment is not another density sweep. It should preregister a
**time-local, source-conditioned T-only visibility model** on new development/confirmation
captures and compare it with the pooled dense cloud at matched pose baselines. The strong
pose-conditioned temporal delta suggests that aggregating construction points across the whole
capture conflates visibility with scene state; a local-time control tests that mechanism directly.
