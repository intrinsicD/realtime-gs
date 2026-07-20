# Compact point-training independent scientist audit

Verdict: PASS

Unresolved findings: none

CALIBRATED_INTEGRATION_AUTHORIZED: YES

## Disposition

The one-shot CPU synthetic experiment is protocol-valid, and an independent implementation of
the frozen reducers reproduces the committed RESULT. The scientific outcome is
`NO_GLOBAL_SAMPLING_WIN`: the discrete-pixel pair is `NEUTRAL_OR_NEGATIVE`, while the
continuous-area pair is `NONINFERIOR` but not a material sampling win. The Gaussian proposal
must therefore not be promoted as a default or described as a general convergence improvement.

The authorization above applies only to the preregistered, bounded, non-decision-bearing
calibrated integration. It does not override or soften the synthetic result.

## Claim audit

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | All twelve seed--arm jobs completed the frozen fixed-topology CPU mechanism protocol. | Measured; synthetic, CPU, fixed topology | RAW records and sealed source | Confirmed: 120 updates and 15,360 attempts per record; `N_init,3D=N_opt,3D=4`; six Adam clocks equal 120; all five effective parameter families moved. |
| 2 | Gaussian importance sampling improves discrete-pixel convergence. | Measured primary estimand; synthetic only | RAW checkpoint risks | Retired: `G_AUC=1.0245665262`, `G_final=1.0681355694`; uniform had lower log-AUC in all three seeds. Label `NEUTRAL_OR_NEGATIVE`. |
| 3 | Gaussian importance sampling improves continuous-area convergence. | Measured confirmatory secondary; synthetic only | RAW checkpoint risks | Narrowed: `G_AUC=0.9910818462`, `G_final=0.9873547158`; all three AUC directions favor the mixture, but the frozen 0.95 materiality floor is not met. Label `NONINFERIOR`. |
| 4 | The experiment establishes a global sampling win. | Frozen decision | Independently recomputed domain labels | Retired: both pairs are not material wins, so the exact decision is `NO_GLOBAL_SAMPLING_WIN`. |
| 5 | The artifacts support production quality, scale, speed/memory, CUDA, density control, novel-view quality, or a default change. | Asserted extension beyond protocol | Preregistration claim boundary | Retired/not tested. No such claim is licensed. |
| 6 | The bounded calibrated path may be run as an integration diagnostic. | Protocol authorization, not an outcome claim | PASS lifecycle plus fail-closed calibrated protocol | Confirmed only for the frozen seven-view, 640-teacher/view, 835-Gaussian fixed-topology integration; its outputs cannot alter the official decision. |

## Independent statistic recomputation

Every checkpoint risk was recomputed from its three per-view float64 SSE/count records, averaging
view MSEs rather than pooling views. The checkpoint `J_pixel`/`J_area`, embedded risk fields,
aggregate scalar counts, range fractions, and the separate final streaming evaluation all agree
within `1e-12` relative tolerance. The domain-matched risks are:

| Domain / seed | Uniform risks at steps 0,30,60,120 | Gaussian-mixture risks at steps 0,30,60,120 |
|---|---|---|
| pixel / 74101 | 0.069922720066, 0.061743855649, 0.054595044912, 0.038348322501 | 0.069922720066, 0.062014734385, 0.054852834843, 0.040229462758 |
| pixel / 74102 | 0.069441573790, 0.061291651456, 0.054099307377, 0.037541519669 | 0.069441573790, 0.061887246007, 0.055282895588, 0.040581865485 |
| pixel / 74103 | 0.070824869518, 0.062997002321, 0.056365005587, 0.040565525146 | 0.070824869518, 0.063520239610, 0.057566659542, 0.043593146300 |
| area / 74101 | 0.060783126199, 0.052689541288, 0.045137658186, 0.030475676768 | 0.060783126199, 0.052125078270, 0.044592454885, 0.029893409972 |
| area / 74102 | 0.060305534113, 0.052253733149, 0.044881938590, 0.029840277002 | 0.060305534113, 0.051841644580, 0.044603027582, 0.030293883135 |
| area / 74103 | 0.061684850813, 0.054021787881, 0.047545691777, 0.032584027790 | 0.061684850813, 0.053709542737, 0.047001855840, 0.031495621806 |

An independent trapezoidal implementation of the preregistered normalized log-AUC produced:

| Domain | Seed | Uniform AUC | Mixture AUC | Delta AUC | q_init | q_final | Direction |
|---|---:|---:|---:|---:|---:|---:|---|
| pixel | 74101 | -0.2740618187 | -0.2592286932 | 0.0148331255 | 0.5484386544 | 1.0490540429 | uniform better |
| pixel | 74102 | -0.2785942102 | -0.2485923278 | 0.0300018824 | 0.5406202311 | 1.0809862212 | uniform better |
| pixel | 74103 | -0.2542393127 | -0.2262654551 | 0.0279738576 | 0.5727582052 | 1.0746353250 | uniform better |
| area | 74101 | -0.3199178939 | -0.3319903824 | -0.0120724886 | 0.5013838325 | 0.9808940487 | mixture better |
| area | 74102 | -0.3224899971 | -0.3230353412 | -0.0005453441 | 0.4948182193 | 1.0152011368 | mixture better |
| area | 74103 | -0.2903469649 | -0.3046036080 | -0.0142566431 | 0.5282338753 | 0.9665969477 | mixture better |

Label precedence, the per-seed direction counts, `G_init`, `G_final`, `G_AUC`, thresholds, record
summaries, and global decision exactly reproduce the RESULT.

## Mechanism, accounting, and RNG controls

- Each record contains exactly 120 sequential steps and 128 attempts per step. Independently
  summed active/null counts equal attempts; uniform/Gaussian branches equal attempts; Gaussian
  accepted/rejected branches equal Gaussian attempts; every attempted coordinate reached both
  teacher and student; and both used four outer query calls per step.
- Record-level active fractions, ESS per attempt, maximum importance, and branch totals reproduce
  exactly from the 1,440 step records. Uniform arms are fully active with unit importance and ESS.
  Mixture maxima are exactly 4.0, within the frozen `1/eta` cap. Pixel-mixture active fractions are
  0.31960, 0.31445, and 0.31445; area-mixture fractions are 0.99974, 0.99967, and 0.99974.
- All recorded proposal invariants are true, including no null resampling, active-domain
  containment, and zero invalid-row densities/importance. Teacher hashes are identical before
  and after training and independent final evaluation. RGB/source access count is zero.
- Initial state, teacher set, exact step-0 risks, and view schedule match across all four arms for
  each seed. The view schedules and all 360 per-step sample seeds were regenerated independently
  from the frozen SHA-256 seed maps; they match RAW exactly and each seed's 120 step streams are
  unique. Coordinate/activity/importance/component tensors are retained as bound digests rather
  than raw vectors, so their numerical contents were not replayed; sealed estimator/RNG regression
  tests supply the outcome-safe implementation control.
- The fixed-attempt estimator itself is sealed to `(loss * marginal_importance).sum()/S` and the
  focused controls confirm null retention, all-null differentiability, marginal rather than joint
  density, later-step RNG isolation, and streamed risk parity. No one-shot fixture or outcome was
  replayed.

## Chronology, provenance, and environment

The seal (`2026-07-16T20:51:56Z`) precedes the exclusive attempt marker
(`2026-07-16T20:52:08Z`), which precedes the committed RAW/RESULT
(`2026-07-16T20:53:23Z`). The seal self-digest, 115-path source manifest, 114-path reviewed
manifest, preregistration reviews, implementation review, attempt, RAW, and RESULT cross-bind.
Every currently bound source file still matches the seal; HEAD and the tracked binary diff also
match. The sealed run is CPU-only with Python 3.12.9, PyTorch 2.9.0+cu128, NumPy 2.1.3, four Torch
threads, one interop thread, deterministic algorithms enabled, and `CUDA_VISIBLE_DEVICES=''`.
Recorded wall time and RSS are descriptive only and were not used for a performance claim.

## Frozen bindings

```text
preregistration_sha256: 865f86d35805c265d27caf4b5f6e02b99e4679f53162dbdd23c17681354065ea
seal_file_sha256: b04875a86d16aae00a656b7dd5bb28c482f9f18654139dcd5938bae1ddb85f4b
attempt_sha256: 130fa471c57d90efd366261d47acf75ed848ad630cfb8b5a3f2484af37ff687f
raw_sha256: 077024a732d4d77ce2b9f1444043e0616af25b571d6336dcb49579c76b2a8a2e
result_sha256: 2339dd308304d2401572d7786d1c9f2d46fd0d1caf19e44cac4e7e6a88e9ed93
source_aggregate: 345fcf71e80655878e93ec274d7ac58eff46773e64eeb75ea8e80783bad0a95a
seal_payload_sha256: 5376b3a9650cf46750ca9994bc12f5ab20c0c5774b8fc4b179db5ef87782be16
preregistration_review_sha256: 203c6d1f7f4b9c5a636f42ba9082b782187cc8a51371649e668893a48c53898e
implementation_review_sha256: 9f18bb65274cbaba74ab72de2c721cc5b858e58c0aecf53e7722b23663eeaf7a
```

## Commands executed

```text
sha256sum benchmarks/results/20260716_compact_point_training_{PREREG.md,PREREG_REVIEW.md,IMPLEMENTATION_REVIEW.md,ATTEMPT.json,RAW.json,RESULT.json,SEAL.json} benchmarks/compact_point_training.py

jq ... benchmarks/results/20260716_compact_point_training_{SEAL,ATTEMPT,RAW,RESULT}.json

git rev-parse HEAD
git diff --binary HEAD | sha256sum

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
# Independent standard-library/Torch audit: strict JSON and digest validation; source-manifest
# verification; SSE/count checkpoint reducers; AUC, ratios, labels and decision; fixture/hash,
# topology, optimizer, proposal, query, accounting, RNG schedule and sample-seed assertions.
PY

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q \
  tests/test_observation2d.py::test_mixture_sampling_is_seeded_and_has_bounded_importance \
  tests/test_observation2d.py::test_discrete_gaussian_sampling_matches_exact_marginal_and_keeps_null_attempts \
  tests/test_compact_trainer.py::test_schedule_and_per_step_seeds_are_mode_independent_and_stable \
  tests/test_compact_trainer.py::test_later_step_stream_is_independent_of_earlier_rng_consumption \
  tests/test_compact_trainer.py::test_all_null_fixed_attempt_loss_is_differentiable_zero \
  tests/test_compact_trainer.py::test_streamed_pixel_and_area_evaluation_match_materialized_reference \
  tests/test_compact_point_training.py::test_normalized_log_auc_matches_literal_hand_calculation \
  tests/test_compact_point_training.py::test_domain_labels_follow_frozen_precedence \
  tests/test_compact_point_training.py::test_result_is_recomputed_from_strict_raw_bindings \
  tests/test_compact_point_training.py::test_audit_authorization_needs_literal_line_and_every_binding
```

The ten focused tests passed. CUDA, GPU timing, dataset RGB, calibrated artifacts, and the
one-shot official fixture were deliberately not accessed or executed during this audit.

## Claim boundary

This audit confirms only a synthetic CPU fixed-topology point-training mechanism and its neutral
or negative sampling result. It establishes no production scale, reconstruction quality,
speed/memory advantage, CUDA behavior, density-control behavior, novel-view quality, or default
method choice.
