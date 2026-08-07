# Prospective Protocol Review

- Task ID: `20260806_gaussian2d_image_refinement_janelle_frame00008`
- Protocol SHA-256: `19a47a047479e3817d78c55d3d4875716caa0b88404446201aa34d767d1ed898`
- Reviewer: `Volta-protocol_review`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was a fresh, outcome-blind prospective review of the exact post-V1-repair six-folder,
image-backed Janelle Gaussian2D-to-3DGS protocol. I reviewed the current task, pipeline design,
task-specific driver, `Trainer` changes, experiment-contract and result-bundle code, focused
tests, live source binding, complete hybrid data seal, durable task handoff, aggregation/report
ordering, and all six deferred viewer commands. I did not assess a result or whether either arm
or any Gaussian2D folder is favorable.

If repaired and executed successfully, this protocol could support development-only, within-frame
descriptions of masked versus unmasked lifting and RGB refinement for six pre-existing
decompositions of the same 26-camera capture. It cannot establish cross-scene generality,
state-of-the-art quality, GPS-Gaussian reproduction, real-time performance, a production default,
or complete-field preservation by the bounded structural carrier.

## Checks

- Recomputed the prospective digest. It exactly matched
  `19a47a047479e3817d78c55d3d4875716caa0b88404446201aa34d767d1ed898`.
- Confirmed that the V1 rejection remains unchanged at
  `experiments/reviews/20260806_gaussian2d_image_refinement_janelle_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`,
  with SHA-256 `ac15bbb7e1f27194e1f3c816db5d968fad0535c66ed5af069380e6b76d5be2b2`.
  This fresh review does not rewrite or reinterpret that artifact.
- Ran task-contract and sealed-data validation without initializing a run; both returned `OK`.
  The live 103-file source binding exactly matched
  `2fa9a8932d3e4c85985833c8e9e867839036c1bc0e2091bb418e4be17caeca87`.
  The data-seal SHA-256 remained
  `1199a410a7070e23126d51c55f5f5039cd0f505ff3f2a8a9b0d8e503b4ac5a63` and bound 215 files
  totaling 490,153,435 bytes: 156 compact views, six compact manifests, 26 Janelle JPGs, 26
  lossless masks, and one calibration file.
- Reconfirmed the exact six owner-selected folder units, paired masked/unmasked arms, and seeds
  80601--80603, yielding 36 measured cells. Every unit retains the same disjoint 20 optimizer,
  three validation, and three held-out cameras. The single 50-iteration warmup remains fixed to
  `gaussians2d` / `masked_pipeline` / seed 80601 and structurally skips held-out loading.
- Adversarially rechecked the V1 worker blocker. The former public `--worker` CLI is rejected;
  scratch output is confined below `.scratch/`; scratch and warmup take the held-out-inaccessible
  branch; and measured/warmup children require an HMAC-authenticated canonical ticket that binds
  task/run lock, protocol/data digests, dataset, arm, integer seed, integer iteration count, mode,
  output, and nonce. A tampered ticket fails authentication before task or scientific input access.
- Adversarially rechecked the V1 validation-timing blocker. `Trainer` receives a 20-camera
  optimizer-only `SceneData`; `internal_checkpoint_evaluation=False` prevents the duplicate
  evaluation pass; train-metric selection is rejected in that mode; and the external observer
  accounts for checkpoint snapshot construction and validation rendering separately from
  `history["elapsed"]`. The final endpoint remains fixed and validation cannot stop or select it.
- Adversarially rechecked the V1 resource blocker. CUDA peaks reset once before compact-field
  access, `Trainer` is forbidden from resetting them, and allocated/reserved peaks plus total cell
  wall freeze after synchronized final PLY/NPZ serialization and before held-out input hashing or
  rendering. Held-out, report, preview, and viewer work is excluded from that endpoint.
- Adversarially rechecked resume and aggregation. The task-specific validator requires exact cell
  keys, identity, split digest, effective configuration, input bindings, mode-specific artifact
  inventory, and current artifact hashes before either resume or aggregation. Copied-arm and
  modified-PLY counterexamples fail. The root receipt then hashes one warmup and all 36 measured
  cell receipts in frozen order. The independent generic bundle replay is weaker, however, as
  described below.
- Adversarially rechecked live input enforcement. Source bytes are verified at coordinator and
  worker entry and by final contract validation. The complete seal is rehashed at coordinator
  entry and after all measured cells and again at bundle validation; every worker additionally
  hashes the compact, calibration, JPG, and PNG bytes it actually opens. Camera alignment includes
  rotation, translation, image dimensions, focal lengths, and principal point.
- Confirmed that six distinct comparison viewers are now actually spawned with `--open` on ports
  8400--8405 only after all measured resource endpoints and preview generation. This closes the V1
  non-launch defect and prevents viewer GPU/server activity from contaminating measured cells.
- Passed 88 focused outcome-free tests across
  `tests/test_janelle_image_experiment_protocol.py`, `tests/test_optim.py`, and
  `tests/test_experiment_contract.py`; compiled the driver, Trainer, contract, and bundle checker;
  and reran the workflow and docs-sync checkers successfully after the reviewer handoff was made
  structurally valid.
- Reconfirmed immediately before writing this artifact that the task remained `draft`, the
  canonical official run root did not exist, and no authenticated/scientific worker, scratch
  cell, protected report, viewer, model, metric, or held-out outcome had been opened or executed
  during this review. Focused tests exercised only temporary outcome-free guard/report fixtures.

## Findings

The protocol is **rejected**. The six-folder scientific matrix and the core V1 execution repairs
are materially improved, but the exact executable is not yet approval-worthy. The following
fail-closed and deliverable defects require repair and another fresh outcome-unseen review.

1. **Blocking / critical -- official execution does not revalidate the hashed prospective-review
   artifact.** `init-run` stores `protocol_review_artifact_sha256`, and final contract validation
   checks it, but the producer's `_official_lock` checks only the review envelope copied into the
   task/lock. It never checks that the current review file is a regular file whose bytes still
   match the lock. `_write_worker_ticket` and `_run_ticket_worker` both rely on this incomplete
   helper, so the coordinator can spend the protected run and open held-out outcomes after the
   approval artifact is deleted or changed; only a later bundle check notices. Make the producer
   revalidate the exact review-artifact path and locked SHA-256 before coordinator entry and again
   before accepting an official worker ticket. Add deletion, byte-drift, wrong-path, and malformed
   lock counterexamples that fail before scientific input access.
2. **Blocking / high -- final cell-bundle validation is transitive over bytes but not over the
   promised semantics.** The task-specific resume/aggregation validator is strict, but
   `experiment_contract._cell_bundle_errors` accepts a cell receipt without the experiment schema,
   exact top-level keys, iteration count, partition digest, effective-config digest, or input
   binding. It also accepts any non-empty artifact list instead of the frozen warmup/measured
   inventories. The current generic contract test explicitly constructs such a minimal receipt
   and expects it to pass. Therefore the final canonical gate cannot independently establish the
   design's claim that the root receipt binds exact configuration, split, inputs, and every
   required artifact. Share or replay the exact task-specific receipt validator at bundle time,
   including strict JSON types and exact artifact sets. Test omitted fields, Boolean/integer and
   integer/float aliases, copied identities, modified summaries/models, and an artifact removed
   coherently from both the cell and root receipts.
3. **Blocking / critical -- late publication and presentation failures are neither canonical nor
   retry-safe.** `_aggregate_run` is outside any terminal failure handler. It writes the canonical
   `RESULT.json`/`RESULT.md` before root metrics, history, configuration, boundary/resource
   receipts, environment, viewer launch, and success receipt. Any preview, aggregation, evidence,
   or serialization exception can therefore leave `progress.json` stuck at `running`, no valid
   failed-run sources, and partial canonical evidence; rerunning then raises because RESULT
   evidence already exists. The one caught viewer-launch failure changes only `run_receipt.json`
   to failed while retaining completed-run metrics, non-empty charts, dataset summaries, and four
   success-evidence links. That state is rejected by the v2 failed-run schema and cannot render an
   inspectable failure report, and the existing RESULT files also prevent a retry. Put all
   post-matrix work behind one exact terminal boundary, publish canonical evidence only after all
   success-critical artifacts exist (or use exact-content idempotent publication), and make every
   late failure produce schema-valid failure sources without destroying the validated 36 cells.
   Add outcome-free counterexamples for preview/aggregation failure, partial RESULT publication,
   and failure of any one of the six viewer startups followed by a same-root resume.
4. **Blocking / requirements -- the child pages do not plot all declared resource/performance
   metrics, and the convergence clock is mislabeled.** `_dataset_summary` creates final seed curves
   only for `task["primary_metrics"]`. The frozen resource protocol and design also promise
   field-lift time, validation-observer time, total cell wall, peak CUDA reserved memory, peak RSS,
   and the measurement endpoint; these remain only in raw JSON. The child page therefore cannot
   provide the requested all-metric curve comparison. In addition, validation/training records are
   positioned using observer-excluded native optimizer time while the generic report labels the
   axis simply `elapsed time (s)` and overlays actual cell-wall stage boundaries. Preserve both
   clocks explicitly: add every promised resource metric to per-seed curves/tables and label native
   optimizer time separately from input-open-to-endpoint wall time. Add a rendered-child test that
   enumerates the required metric/clock IDs for all six datasets.
5. **Blocking / presentation evidence -- only one of the six opened viewers is required to be
   browser-smoked.** The launch receipt proves six processes survived a one-second probe, but the
   canonical bundle checker validates only root `viewer_smoke.json` against the first dataset's
   `commands.viewer`. It cannot prove that the other five WebGL viewers reached ready state,
   rendered visible content, and responded to an orbit, even though the durable success criterion
   requires the page and viewer for every completed folder to be opened and browser-smoked. Freeze
   one aggregate six-entry browser receipt or one exact receipt per dataset, bind each to its child
   report and viewer argv, and require all six in the bundle gate. Process liveness and `--open`
   alone are not a browser-render attestation.

The task must remain non-executable. Any correction to a digest-bearing task field or bound source
byte requires a new prospective digest and another fresh review. The preserved V1 rejection and
this rejection must not be edited into approvals.

## Protected Actions Not Taken

I did not transition the experiment task, invoke `init-run`, execute the canonical coordinator,
invoke an authenticated official worker or scratch cell, create or enumerate the protected run,
read a protected RESULT/AUDIT artifact, inspect any protected model, metric, preview, child report,
or orbit output, launch a viewer, or consume a held-out image outcome. Checks were limited to the
exact task/design/source, sealed input metadata and hashes, static control-flow/API inspection,
compilation, workflow/schema validation, and temporary outcome-free tests. Outcome Access remained
`none` throughout.
