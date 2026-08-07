# Prospective Protocol Review

- Task ID: `20260806_gaussian2d_image_refinement_janelle_frame00008`
- Protocol SHA-256: `c3a334db13b3cc51cbe2634cd44f5d04339cdd51115c6420eea5902497ac7b05`
- Reviewer: `Volta-protocol_review`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was a fresh V3, outcome-blind prospective review of the exact second-repair RTGS-013
protocol for six independent, image-backed Janelle Gaussian2D-to-3DGS experiments. I reviewed the
frozen task and design, task-specific producer, `Trainer` observer timing, generic experiment
contract, result-bundle gate, targeted counterexample tests, live source binding, complete hybrid
data seal, and durable V3 handoff. I specifically replayed all five V2 findings and then searched
for new authorization, receipt, publication, metric-clock, and all-six-viewer bypasses.

If repaired and executed successfully, the protocol could support development-only, within-frame
descriptions of masked versus unmasked lifting and RGB refinement for six pre-existing
decompositions of one 26-camera capture. It cannot establish cross-scene generality,
state-of-the-art quality, GPS-Gaussian reproduction, real-time performance, a production default,
or complete-field preservation by the bounded carrier.

## Checks

- Recomputed the prospective digest and obtained exactly
  `c3a334db13b3cc51cbe2634cd44f5d04339cdd51115c6420eea5902497ac7b05`.
  Recomputed the live 103-file source binding and obtained exactly
  `6b1f0469735e28e404c6d45d28026cf19c67551fc37035af9055e8d63f246ba4`.
- Confirmed that both prior rejection artifacts remain unchanged. V1 has SHA-256
  `ac15bbb7e1f27194e1f3c816db5d968fad0535c66ed5af069380e6b76d5be2b2`; V2 has SHA-256
  `512f2f2005403f6b02e8c114c9fa04496f188ffb0fb9c132e0ebe8c30e885d22`.
- Ran task-contract and complete sealed-data validation without initializing a run; both returned
  `OK`. The data-seal SHA-256 remains
  `1199a410a7070e23126d51c55f5f5039cd0f505ff3f2a8a9b0d8e503b4ac5a63`, binding 215 files
  totaling 490,153,435 bytes: 156 compact views, six manifests, 26 Janelle JPGs, 26 lossless
  masks, and one calibration file.
- Reproduced V2 repair 1. `_official_lock` now requires the canonical review path, a regular
  non-symlink file, exact lock keys, and the live artifact SHA-256. Both ticket issuance and the
  authenticated worker call that helper, and the ticket itself binds the review digest. The
  deletion, byte-drift, wrong-path, malformed-lock, and numeric-ticket-type counterexamples pass.
- Reproduced V2 repair 2. Generic `_cell_bundle_errors` now requires exact policy, root, entry,
  receipt, input-binding, and artifact-record keys; exact ordered warmup/measured identities;
  strict integer types excluding Boolean aliases; frozen iterations, partition, effective
  configuration, input bindings, mode-specific artifact inventories; completed-summary endpoint
  semantics; and current transitive hashes. Omitted semantics, `true` as an integer, coherent
  artifact-inventory removal, copied identities, and changed artifact bytes are rejected.
- Reproduced the substantive portion of V2 repair 3. Exit integrity, aggregation/previews,
  six-viewer startup, and exact-content RESULT publication share a caught post-matrix block.
  Preview/aggregation failure produces failed-run v2 sources; a same-root retry reuses validated
  cells; exact partial evidence completes idempotently; conflicting evidence fails closed; and a
  one-viewer startup failure can relaunch only that viewer while reusing the other five. The final
  terminal completion transition still has the blocker below.
- Reproduced V2 repair 4. Each dataset summary loops over all 15 frozen primary metrics, including
  field-lift, native optimizer, validation-observer, worker-start endpoint, input-open endpoint,
  CUDA allocated/reserved, RSS, and final-capacity measurements, and plots every seed for both
  arms. Five validation-quality curves use an explicitly labelled observer-excluded native
  optimizer clock. The separate history charts use worker-start cell-wall timestamps and explicit
  stage boundaries. The six-child renderer test confirms both clock labels and every metric ID on
  every child page.
- Reproduced V2 repair 5. The bundle checker derives exactly one child report and exact viewer argv
  per frozen dataset. Schema-v2 `viewer_smoke.json` must contain six ordered entries, each with its
  dataset identity, child HTTP-200/local-target result, browser identity, WebGL2, a live canvas,
  visible non-background pixels, an orbit camera change, no client errors, and classified
  warnings. Missing sixth entries and a failed sixth orbit are rejected.
- Ran the 97-test focused outcome-free collection over
  `tests/test_janelle_image_experiment_protocol.py`, `tests/test_experiment_contract.py`, and
  `tests/test_optim.py`; 91 passed and six hardware-specific tests skipped on CPU. Python
  compilation, Ruff lint/format, workflow validation, docs-sync, and `git diff --check` passed.
- Reconfirmed immediately before writing this artifact that the task remained `draft`, the
  canonical official run root did not exist, and no official worker, held-out outcome, RESULT or
  AUDIT record, model, metric, preview, report, or viewer had been created or inspected.

## Findings

The protocol is **rejected**. All five V2 repair surfaces are materially present, but a new
fail-closed terminal-state counterexample prevents approval.

1. **Blocking / critical -- the canonical completion transition remains outside the terminal
   failure boundary.** In `_run_parent`, the caught post-matrix block ends immediately after
   RESULT publication. The completed `run_receipt.json` write and the subsequent completed
   `progress.json` write occur after that `try`/`except`. An exception in either write therefore
   escapes instead of producing failed-run sources. More seriously, success is written to
   `run_receipt.json` before progress is finalized. In an outcome-free temporary fixture, I
   injected a one-shot exception only when writing completed progress. `_run_parent` raised
   `OSError("injected final progress write failure")`, while `run_receipt.json` remained
   `status: completed` and `progress.json` remained `status: running`. Thus a nonzero coordinator
   exit can leave the canonical receipt claiming completion; the current bundle validators do not
   reconcile progress with the process terminal state.

   Put both terminal writes inside the same caught post-matrix boundary. Finalize completed
   progress first and publish the successful run receipt last, so no fallible action follows the
   success commit. If either write fails, return a nonzero code with a schema-valid failed run
   receipt and failed progress while retaining the validated cells and exact-content evidence for
   same-root retry. Add outcome-free injections for failure of each terminal write, especially the
   progress-after-receipt ordering counterexample, and prove that neither can leave a completed
   receipt; then prove a subsequent same-root retry completes using the unchanged evidence.

Because repairing this finding changes a behavior-bearing source byte, the owner must freeze a
new source binding and prospective digest and obtain another outcome-unseen review. The task must
remain non-executable. V1, V2, and this V3 rejection must remain unchanged provenance.

## Protected Actions Not Taken

I did not transition the experiment task, invoke `init-run`, execute the canonical coordinator,
invoke an authenticated official worker or scratch cell, enumerate a protected run, read any
protected RESULT/AUDIT artifact, inspect any protected model, metric, preview, child report, or
orbit output, launch a viewer, or consume a held-out image outcome. The terminal-write
counterexample used only temporary directories, mocked cell validation/aggregation/viewer/evidence
operations, and synthetic status receipts. Outcome Access remained `none` throughout.
