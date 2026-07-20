# Scientist audit: I1 correspondence-confidence gate

Date: 2026-07-20
Audited result:
[`20260720_dense_confidence_gated_init_i1_RESULT.md`](20260720_dense_confidence_gated_init_i1_RESULT.md)
Disposition: **accept implementation/count reproduction; exploratory quality only**

## Chronology and leakage

The classifier was frozen in
[`20260720_dense_confidence_gated_init_i1_PREREG.md`](20260720_dense_confidence_gated_init_i1_PREREG.md)
after E1 and before easy-only render quality was opened. Threshold selection
used only placement/merge signal distributions on the seven training compact
views. The 60-cluster expectation was recorded before implementation. No
held-out image, optimizer outcome, or downstream density-control result was
used.

The subsequent implementation reproduced exactly 60 retained clusters. No
threshold was changed after opening easy-only quality.

## Claim inventory

| Claim | Evidence class | Audit disposition |
|---|---|---|
| The gate deterministically retains 60/2,319 clusters | Calibrated count reproduction + complete records | Accepted |
| Typed diagnostics and off-by-default CLI are implemented | Source inspection + CPU fixture/harness tests | Accepted |
| Easy-only starts above top-K on these seven views | Exploratory same-view init screen | Accepted, training-view scope only |
| Easy-only is a better downstream/default initializer | No downstream or held-out result | Rejected/not measured |
| The accelerated evaluator is numerically exact | GPU parity is close, not bit-exact | Rejected; retain CPU E1 as correctness anchor |

## Independent checks

- Raw JSON SHA-256:
  `9980d91536a622808acd33076c7325707385d160d5c375363cafbc24d60986c4`.
- Dense/top-K PLY hashes match E1 exactly, showing that enabling the audit/gate
  did not alter either upstream initialization.
- Recomputed from raw JSON:
  easy-only minus top-K `+0.4504885 dB` mean foreground PSNR,
  `+0.0017161` mean SSIM, worst view `+0.3376160 dB`, and primitive ratio
  `0.348837`.
- The stored keep count is 60, drop count 2,259, and record count 2,319.
- CPU tests cover every frozen threshold, exact audit/lineage matching,
  malformed group rejection, and a known three-view-target/single-view-decoy
  split.
- Synthetic `--gate` ran end to end. Its zero-cluster result exposed and now
  regression-tests empty degree-0 PLY serialization.
- Artifact-only viewer smoke returned HTTP `200`.

## Confounds and limits

1. All init-quality numbers are from the seven views used to construct and
   parameterize the confidence signals. They are not a validation or held-out
   estimate.
2. The 60-count reproduction validates code/protocol fidelity, not classifier
   optimality.
3. Failure counts overlap and must not be summed as disjoint populations.
4. GPU metrics differ from the CPU reference by up to roughly `0.004 dB` in
   the E1 parity replay; that is small for diagnosis but prevents an
   exact-equivalence claim.
5. The runtime is one local mixed CPU/GPU observation, without repeated
   warmups or an idle-host guarantee.
6. The current compact bundle has calibrated teachers but no independently
   usable held-out teacher in the same bundle. E2 must freeze and document its
   train/validation/held-out source before any optimizer run.

## Decision

I1 meets its definition of done: the classifier was preregistered,
implemented CPU-first, fixture-tested, emitted complete diagnostics, and
reproduced both the E1 distribution and expected 60 kept clusters. The
training-view quality screen is promising enough to proceed to E2 under a
separately frozen schedule and split. It does not authorize a default change.

Repository-wide verification and final docs-sync must pass before merge.
