# Exact sampled-confidence attribution result

This is the post-run audit for
`20260715T052539Z_cpu_depth_anchor_attribution.json`. The experiment was frozen in
`20260715_depth_anchor_attribution_PREREG.md` before implementation and execution.

## Validity

- Command: `CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_anchor_attribution.py
  --output benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json`
- Revision: `2dddca4aff59702341af9faceefa76ad2505dd83`, with the dirty status, tracked-diff
  hash, loaded source hashes, environment, and complete effective config embedded in the JSON.
- The official run used the frozen seeds `0/1/2` and the three declared arms only:
  `valid_uniform`, `confidence`, and `confidence_shuffled`.
- Every seed passed exact step-0 equality and the two-iteration `depth_prior_lambda=0` main-RNG
  equality check.
- Every source view passed exact valid-weight multiset, sum, and squared-sum preservation under
  the sampled-weight shuffle. Invalid weights remained zero and every nonconstant valid layout
  changed location.
- Primitive count, valid-ray layout, and resolved anchor stiffness were shared across arms within
  each seed. No refinement, merge, rotation, scale optimization, or density control was enabled.

The artifact is therefore valid for the preregistered decision. This audit was written after the
official run and is intentionally not part of that run's embedded source hash set.

## Primary result

| Comparison | Valid uniform | Confidence | Change | Seed wins | Required | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Held-out depth RMSE | 0.151705 | 0.149962 | 1.149% lower | 3/3 | >=2%, >=2/3 | no |
| Corrupted-source depth p90 | 0.204933 | 0.206519 | 0.774% worse | 1/3 | >=15%, >=2/3 | no |
| Held-out PSNR | 19.6318 dB | 19.6310 dB | -0.0008 dB | -- | >=-0.10 dB | yes |

The shuffled arm reached 0.151792 held-out depth RMSE and 0.205348 corrupted-source p90.
Confidence beat it on RMSE in 3/3 seeds and on p90 in 2/3, but the shuffled control erased at
least half the RMSE gain only. It did not satisfy the corrupted-tail attribution condition.
Consequently both `material_effect_pass` and `confidence_location_attribution_pass` are false.

Secondary, non-decisive signals were mixed: confidence improved SSIM by 0.0041, all-source depth
p90 by 8.26%, and nearest-ground-truth median distance by 2.06%, while nearest-ground-truth p90
worsened by 0.26%. These metrics cannot override the frozen primary criteria.

Two scope limitations do not change the paired decision. The clean-ground-truth source diagnostic
contains 16/29/34 more rays than the valid anchor mask across seeds, which can dilute sensitivity
but is fixed across arms. The explicit main-RNG invariant covers two optimization iterations;
separate-generator code inspection, exact multi-arm equality on all seeds, and targeted tests
provide the remaining evidence. The deterministic fallback that forces an unchanged nonconstant
shuffle to move is an exact permutation, but it is not a uniformly conditioned random permutation.

## Decision

The result is compatible with a small location-sensitive expected-depth effect, but it does not
support a material or robust confidence-anchor benefit. Apply the preregistered stopping rule:

- keep `legacy` as the default;
- retain the new modes as explicit research/attribution controls;
- stop confidence-anchor loss, lambda, threshold, and weighting sweeps on this setup;
- do not advance to train-derived confidence on this evidence;
- pivot to leave-one-source-view-out photometric supervision or direct cross-view
  geometric/correspondence consistency.
