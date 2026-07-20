# Compact point-training preregistration executability review

Verdict: PASS

Unresolved findings: none

## Final outcome-blind amendment review

The passing review binds
`benchmarks/results/20260716_compact_point_training_PREREG.md` at SHA-256
`865f86d35805c265d27caf4b5f6e02b99e4679f53162dbdd23c17681354065ea`.

The final lifecycle-only amendment resolves the otherwise impossible combination of immutable
exclusive RAW creation and rewriting a committed PASS RAW after a later RESULT-stage failure.
Before the RAW commit boundary, caught failures produce FAIL RAW and RESULT artifacts. After a
successful PASS RAW flush/fsync, that evidence remains immutable and a later failure produces only
a FAIL RESULT binding the RAW, recording `failure_phase='post_raw_commit'`, and containing no
scientific decision. This changes no fixture, objective, statistic, threshold, or scientific
interpretation. The reviewer returned PASS with no unresolved finding and required explicit
commit-state tracking plus fault tests on both sides of the boundary.

The calibrated-only `evaluate_checkpoint_risks=False` exception is literal and bounded. It keeps
the `(0,10,20,40)` detached checkpoint snapshots, hashes, and callback events while preventing an
exhaustive full-resolution risk pass from dominating compact training. The decision-bearing
synthetic experiment still evaluates every frozen checkpoint. Calibrated training risk remains a
separate 4,096-sample-per-view diagnostic, the calibrated result remains non-decision-bearing,
and it cannot alter the official RAW or RESULT artifacts.

The reviewer found no unresolved contradiction in the amendment. Implementation must use the
exact flag spelling and explicitly prove both zero exhaustive checkpoint-evaluator calls and all
four callback events in the calibrated path. No official seed was invoked, no official fixture
was constructed, and no outcome was accessed during this amendment review.

## Earlier passing review chronology

The calibrated-scalability amendment PASS bound SHA-256
`fed1b194af41e072f9ad1216f556e8c2676354dd1c7203c9a16a1b456001530f`.

The pre-scalability-amendment PASS bound SHA-256
`0bf2298261eee727419467537d4c1b3ac61ab7d0c189f508ec03ae60df8139f1`.

The amended protocol resolves the initial review's exact-statistic, matched-work wording,
tolerance, provider-schema, supervision-boundary, artifact-path, archive/index-cap,
calibrated-plan, fresh-process RSS, renderer, and optimizer blockers. The discrete and continuous
pairs have domain-matched uniform controls; fixed attempts retain rejection nulls; implementation,
one-shot synthetic execution, audit, full-resolution acquisition, raw-RGB-denied training, and
viewer handoff have executable gates and exclusive artifact paths.

That reviewer found no remaining contradiction that prevented outcome-blind implementation. The
literal fixture arrays, cameras, and seed mapping may be implemented only as review-bound source
before sealing. No official seed was invoked, no official fixture was constructed, and no outcome
was accessed during either review.

## Preserved initial review chronology

The first draft at SHA-256
`990a530ec2943875c4cb062d13a1c5e26fdb310038851cb0adbcf6ccad02b040` received FAIL before any
implementation or outcome access. Its blockers were:

1. ambiguous final/viability ratios and label precedence;
2. an incorrect matched point--Gaussian-work statement;
3. missing numeric mechanism tolerances;
4. no source-compatible truthful synthetic provider;
5. ambiguity between trainer supervision and CompactCarve coverage use;
6. unnamed raw/audit/calibrated artifacts;
7. non-literal archive and tile-index caps;
8. a selectable calibrated split/configuration;
9. incomparable same-process peak RSS; and
10. incomplete renderer and Adam bindings.

All ten were amended before this PASS review. The initial reviewer likewise did not invoke an
official seed, construct an official fixture, or access an outcome.
