# Prospective Protocol Review

- Task ID: `20260806_gaussian2d_image_refinement_janelle_frame00008`
- Protocol SHA-256: `56178a3e48eb12829a66476c7bac7b2f22fdd7273ebcaae6102e43d032fb48b5`
- Reviewer: `Volta-protocol_review`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This is the owner-authorized, bounded V4 prospective review of the exact RTGS-013 protocol for
six independent, image-backed Janelle Gaussian2D-to-3DGS experiments. The V4 scope was limited to
the sole V3 blocker: the ordering and failure handling of the two terminal completion writes. I
reviewed that repair, its outcome-free one-shot failure and same-root recovery tests, the unchanged
six-folder scientific matrix, and the existing V1/V2 authorization, leakage, receipt, resource,
publication, and six-viewer guards.

If the protected matrix later completes and passes its independent results audit, this protocol
may support development-only, within-frame descriptions of masked versus unmasked field lifting
and RGB refinement for six pre-existing decompositions of one 26-camera Janelle capture. Approval
does not establish any result. The protocol cannot establish cross-scene generality,
state-of-the-art quality, GPS-Gaussian reproduction, real-time performance, a production default,
or complete-field preservation by the bounded carrier.

## Checks

- Recomputed the prospective digest and obtained exactly
  `56178a3e48eb12829a66476c7bac7b2f22fdd7273ebcaae6102e43d032fb48b5`.
  Recomputed the live 103-file source binding and obtained exactly
  `b10eb15c38bd44da97ad42464870fee64eb5a158f722e3b2cf3a6a1d77f4445a`.
- Confirmed that the three rejected review artifacts remain byte-identical. Their V1, V2, and V3
  SHA-256 values are respectively
  `ac15bbb7e1f27194e1f3c816db5d968fad0535c66ed5af069380e6b76d5be2b2`,
  `512f2f2005403f6b02e8c114c9fa04496f188ffb0fb9c132e0ebe8c30e885d22`, and
  `acd84e5a430f50589a5bb228a95cbbb4a17ce7a0ba1a7e41c0f630337de707d5`.
- Re-ran task-contract and complete sealed-data validation; both returned `OK`. The unchanged data
  seal has SHA-256
  `1199a410a7070e23126d51c55f5f5039cd0f505ff3f2a8a9b0d8e503b4ac5a63` and binds 215 files
  totaling 490,153,435 bytes: 156 compact views, six manifests, 26 Janelle JPGs, 26 lossless
  masks, and one calibration file.
- Reconfirmed the unchanged scientific matrix: exactly six owner-selected folder units, masked
  and unmasked arms, seeds 80601/80602/80603, 20 optimizer cameras, three reporting-only
  validation cameras, three final held-out cameras, 1,500 fixed refinement iterations, all 15
  primary metrics, no early stopping, and prospective seed 80601 presentation selection. The
  matrix contains 36 independently receipted measured cells plus one held-out-forbidden warmup.
- Inspected `_run_parent` at the post-matrix boundary. Completed progress is mutated and written
  inside the caught block; `RUN_COMPLETE` is logged next; and the completed `run_receipt.json`
  write is the final fallible success action. No call, write, logging operation, or other fallible
  action follows the successful receipt before the non-fallible return.
- Reproduced the parameterized one-shot completed-write counterexamples for both `progress.json`
  and `run_receipt.json`. Each injected failure returns nonzero, leaves both terminal documents in
  `failed` state with `failure_phase: post_matrix_publication`, leaves no false completed receipt,
  and then completes on the same canonical-root retry.
- Replayed the existing post-matrix aggregation failure and recovery test, authenticated worker
  and review-lock guards, strict transitive cell-bundle replay, source/data revalidation, complete
  resource and distinct-clock curves, exact-content evidence recovery, retry-safe six-viewer
  launch, and exact six-entry browser/WebGL/visible-content/orbit requirements.
- Ran the 99-test focused outcome-free collection over
  `tests/test_janelle_image_experiment_protocol.py`, `tests/test_experiment_contract.py`, and
  `tests/test_optim.py`; all passed. Python compilation, Ruff lint and formatting, workflow
  validation, docs-sync, ARA validation, script-layout validation, and `git diff --check` also
  passed.
- Reconfirmed immediately before the verdict that the task remained `draft`, the canonical official
  run root did not exist, and no official worker, held-out outcome, RESULT/AUDIT record, model,
  metric, preview, report, or viewer had been created or inspected.

## Findings

The exact protocol is **rejected** for one administrative freeze blocker. The bounded V4 source
repair itself closes the sole V3 code blocker: both terminal writes are covered by the common
fail-closed post-matrix boundary; completed progress precedes completion logging; the successful
run receipt is the final fallible success action; and both one-shot terminal-write failures
converge to canonical failed state before a same-root retry completes. The prior V1/V2 guard
surfaces and scientific matrix remain sufficiently protected by the unchanged focused tests and
frozen task fields.

However, the reviewed task still freezes the non-empty blocker “Obtain fresh prospective
independent protocol approval under the repaired digest with no protected outcome access.” The
contract intentionally includes `blockers` in `protocol_sha256` and excludes only `status` and
`protocol_review`, while a `ready` task is forbidden to retain any blocker. Clearing this blocker
changes the prospective digest from the reviewed
`56178a3e48eb12829a66476c7bac7b2f22fdd7273ebcaae6102e43d032fb48b5` to
`61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b`; retaining it makes the
administrative `ready` state invalid. Therefore no valid approved ready state exists under this
exact reviewed digest.

Do not initialize this task. If the owner authorizes one administrative V5 loop, clear the
approval-only blocker prospectively while the task remains draft, freeze the resulting digest,
and obtain a fresh outcome-unseen review. No code, data, matrix, configuration, metric, command,
or scientific field needs to change. Approval under the changed digest would still establish no
quality, performance, convergence, viewer, or capability result.

## Protected Actions Not Taken

I did not invoke `init-run`, execute the canonical coordinator, invoke an authenticated official
worker or scratch cell, enumerate or read a protected run, open any held-out outcome, write or
inspect any RESULT/AUDIT artifact, model, metric, preview, child report, browser-smoke record, or
orbit output, or launch a viewer. All failure reproductions used pytest temporary directories,
mocked cell validation/aggregation/viewer/evidence operations, and synthetic status receipts.
Outcome Access remained `none` throughout.
