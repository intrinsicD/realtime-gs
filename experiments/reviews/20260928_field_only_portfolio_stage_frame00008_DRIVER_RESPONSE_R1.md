# Driver response to the V1 prospective review

- Task ID: `20260928_field_only_portfolio_stage_frame00008`
- Rejected digest: `4dd390c27ff30d7db4b5fdb4be0d59f86c849f8be1dd910930adbd756868236c`
- Revised digest: `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68`
- Driver: `Claude-Code-Opus-5.5-driver`
- Review record: `20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`
- Source binding: 119 files, aggregate `2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e`

No condition, seed, split, configuration, threshold or execution order changed.

## Required changes

- **1. Two-operator evaluation.** `evaluate` now stores, for every cell and held-out view, a
  second metric set `ds8_point` from a point-sampled render at the downscale-8 held-out camera
  (colour and alpha as in RTGS-028), scored by the unchanged `mask_scores` on the same reference,
  mask, bands and LPIPS crop, with its own per-cell mean `mean_ds8_point`. `publish` recomputes
  that mean from the per-view rows with the same finiteness and row-order checks and reports
  `groups_ds8_point` in `comparison.json` and RESULT.json. `nb_ds4` is gated under both operators
  (pass iff the written rule passes under both in every seed, reject iff it reverses under both
  in every seed, otherwise inconclusive); the five downscale-8 treatments keep the primary gate
  with the point operator reported descriptively, and both descriptive pairs are reported under
  both operators. Task JSON: `scoring.evaluation_render` describes both operators and the 0.3 px^2
  pixel-unit dilation mechanism; `decision_policy.per_arm` carries the two-operator clause;
  `primary_metrics` are unchanged and a `secondary_metrics` block defines the point set;
  `claim_boundary` states both operators and qualifies the quadrature coincidence as positional.
  Tests: a two-operator case (primary-only pass gives inconclusive for `nb_ds4`; other arms stay on
  the primary operator).
- **2. Disclosures in the frozen protocol.** (a) and (b) are in the `nb_30k_d6` and
  `nb_30k_d6_lr` comparator purposes with the stated means-LR factors and the post-8000 regime;
  (c) is in the `nb_ds4` purpose (dilation and SSIM-window scale, doubled pixel-unit gradients,
  effectively lower grow threshold, possible cap); (d) is appended to the hypothesis; (e) is in the
  `nb_v11` purpose.

## Verification

7 protocol tests pass on the GPU (the CUDA box-render test skips on CPU); `validate` passes; the
non-protocol smoke evaluation was re-run with the two-operator code on the existing smoke models
(training views standing in for held-out) and produced both metric sets.
