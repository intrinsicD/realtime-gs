# Driver response to prospective review 1

Protocol SHA-256: `3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7`. Source binding: `3a3a7f0c1e1dc92c300c43e7a0ca53aa3b1e780fb2dcd8bfb39e93174d0c1430` across 122 files, unchanged patterns. The task is draft, its distinct review is pending, and no reconstruction run exists. This response is Driver-authored orientation, not an approval.

The rejected V1 review is preserved verbatim in `20260906_tomography_source_constraints_haelyn_dome_PROTOCOL_REVIEW_V1_REJECTED.md`. Its protocol/source hashes remain historical; this is a new exact-digest review.

| Finding | Resolution |
|---|---|
| B1 | Resolve run paths in main, worker, coordinator and aggregate. A relative-argv regression test exercises actual aggregate output; completion receipt is written only after all producer sources and RESULT files. |
| B2 | Clear the stale checklist while retaining draft/pending review. User authorized Fable review, and independent V1 explicitly recommended this bounded edit. In-memory negative control proves empty checklist with ready/pending still fails the independent approval gate. No validator weakened. |
| R1 | Field workers reject any placement_fallback_reason before saving their initial state; summaries record the absence. Warmup follows the same path. |
| R2 | Coordinator aborts the entire task at the first failure and emits validated v2 failure sources. Partial producer sources are moved with exact bytes into failure_preserved before replacement. Existing attempts are refused outside the catch; a regression test verifies failed rendering inputs and byte preservation on retry. No retry is authorized for this task ID. |
| R3 | Reject nonfinite proxy objective/elapsed histories and empty/nonfinite saved Gaussian states. Record accepted_steps and retain the existing finite-step rollback policy. |
| R4 | Correct the metric description to half-integer teacher-window centers and equal-weighted views. The suggested unequal-denominator interpretation was disproved: all 47 sealed teachers are full-canvas, including masked exports. Native fitting restores global coordinates; the preparation adapter call omits fit_window and exports full-canvas. Added a worker invariant check for train, validation and heldout, plus a negative test. Do not infer this receipt is a quality outcome. |
| R5 | Explicit alpha_policy and worker boundary records: masked field arms use compact-embedded alpha for source support; maskless and Beam load none. Raw masks remain report-only. |
| R6 | Claim boundary explicitly limits maskless Haelyn to a uniform black rendered background, not cluttered-photo robustness. |
| O1–O5 | Read frozen warmup iteration counts; retain evaluation-only observer subtraction as a secondary group time and paired ratio; declare five extra Adam/global-clipped coordinates, fixed initial opacity's proxy diagnostic limits, Beam's absent alpha, and no retry. Stage-runtime chart now reports each actual stage separately. |
| O6 | No commit requested or performed. The prescribed development lock binds the dirty source state. |

Preflight evidence: `ara/evidence/tables/20260906_tomography_revision1_preflight/`. Focused protocol and historical contract tests pass (44). Full-canvas metadata and the independent-review gate negative control are copied there. Both compact/external seals and measured effective configuration remain unchanged. Rehashed 1,703 historical files: no changes. Prior full CPU logs describe the earlier source state; the current verify log is separate and must be inspected for its own exit status.

The focused tests include fresh-process hard/soft/free/Beam workers on tiny fixtures; relative aggregation; failed producer source validation and retry byte preservation; fatal guards; frozen warmup counts; and main argv resolution. Fixture values are not capture results.

No algorithm, sealed input or comparative budget changed to chase outcomes. No real warmup, worker, phase evaluator, protected init-run, result audit or calibrated comparison has run. Fable 5.1 effort max is the only authorized independent reviewer; no nested agents or fallback model.
