# First target-gradient diagnostic — 2026-09-10

The diagnostic is complete and independently audited. The cached 2D field images produce substantially different position gradients from the photographs at identical saved 3D models. Most target error is near or outside the silhouette. This identifies a measurable mismatch to investigate; no improved reconstruction was trained. The bounded finding is [C48](../ara/logic/claims.md#c48-cached-field-and-photograph-targets-produce-differing-gradients-at-the-same-saved-states).

| Measurement | Audited result |
|---|---|
| Coverage | 9 saved states × 22 training views = 198 comparisons |
| Position-gradient cosine, initial | 0.273494 |
| Position-gradient cosine, photograph-final | 0.352577 |
| Position-gradient cosine, field-final | 0.415299 |
| Boundary plus exterior share of target error | 93.2569% |
| Boundary / interior target MAE | 0.04856269 / 0.00344643 |

A cosine of 1 means the gradients point in the same direction. Values average views, then the three inherited seeds; every comparison holds the model state fixed. Error shares are means of per-view shares, not pooled pixels. All 198 position comparisons pass the frozen observed-repeat precision flags. Two repeats do not provide a statistical confidence interval or bound systematic error. Initial rotation and inactive higher-order SH cosines remain undefined.

The first attempt, RTGS-023, failed its declared numerical identity check and remains inconclusive. A separately reviewed successor, RTGS-024, measured photo-only render/adjoint/backward repeatability before comparing field targets through a common render graph. Its raw controls and independent reductions passed. Both attempts and the original failure remain preserved; the old thresholds were not relaxed.

The next useful experiment is a separately registered **outside-mask target correction**: replace only field pixels outside the frozen training mask with their cached photograph values, then compare gradients against the original field, photograph and sham controls at the same saved states. This tests the outside-mask signal's contribution to the gradient difference. It adds photograph/mask information, so it is an attribution probe rather than a field-only reconstruction method. It has not been executed.

Gradient agreement alone would not establish better reconstruction. A subsequent paired fitting trial would still need an adequate photograph reference and held-out quality evaluation. The previous quality failures in C47 remain unchanged, as do the unavailable conditional initialization arms. No tomography experiment or physical-density inference was performed here.

The [generated report](../runs/20260910_field_gradient_adjoint_stage_frame00008/index.html) is also served at [localhost:8765](http://127.0.0.1:8765/index.html); the [viewer](http://127.0.0.1:8880/) displays the unchanged, preselected field-final model. Both were exercised in the actual browser. The [independent audit](../benchmarks/results/20260910_field_gradient_adjoint_stage_frame00008_AUDIT.md) and [evidence table](../ara/evidence/tables/20260910_field_gradient_adjoint.md) retain raw/source bindings, precision limits and final delivery receipts. No new private results were sent to Claude in this experiment turn; his public continuation discussion is recorded in the [preceding note](RESEARCH_2026-09-09_CONTINUATION.md).
