# Prospective Protocol Review

- Task ID: `20260806_gaussian2d_image_refinement_janelle_frame00008`
- Protocol SHA-256: `7a470a851444f69ce236ed30636741835708886812888b16361f2a5d13129744`
- Reviewer: `Volta-protocol_review`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the exact six-folder, image-backed Janelle
Gaussian2D-to-3DGS protocol. I reviewed the task, design, driver, hybrid RGB/Gaussian2D seal and
contract changes, focused tests, Trainer interaction, result aggregation, child-report surface,
and viewer commands. I did not assess a result or whether any method is favorable.

If repaired and executed successfully, the protocol could support development-only, within-frame
descriptions of masked versus unmasked lifting and RGB refinement for six pre-existing
decompositions of the same 26-camera capture. It cannot establish cross-scene generality,
state-of-the-art quality, GPS-Gaussian reproduction, real-time performance, a production default,
or complete-field preservation by the bounded structural carrier.

## Checks

- Recomputed the prospective digest. It exactly matched
  `7a470a851444f69ce236ed30636741835708886812888b16361f2a5d13129744`.
- Ran task-contract validation and sealed-data validation without initializing a run; both returned
  `OK`. The hybrid seal SHA-256 is
  `1199a410a7070e23126d51c55f5f5039cd0f505ff3f2a8a9b0d8e503b4ac5a63`. It binds 215 files
  totaling 490,153,435 bytes: 156 compact views, six compact manifests, 26 Janelle JPGs, 26
  lossless masks, and one calibration file. Every dataset advertises calibration, Gaussian2D,
  RGB, and mask modalities.
- Confirmed the exact owner-selected six folder units, paired masked/unmasked arms, and seeds
  80601--80603, yielding 36 measured cells. Every unit has the same disjoint 20 optimizer, three
  validation, and three held-out camera roles. The one 50-iteration warmup is prospectively fixed
  to `gaussians2d` / `masked_pipeline` / seed 80601 and is excluded from aggregation by path.
- Confirmed that the field lifter receives only the 20 optimizer compact views. The masked arm
  decodes source-bound packed alpha and uses hard support; the unmasked arm requests no alpha and
  uses no support mask. Both arms use the same deterministic 2,048-component cap and the lifter
  emits original/used counts, selection rule, and selection digest in its diagnostics.
- Confirmed that RGB gradient sampling uses only optimizer indices, while the declared validation
  views are used for fixed checkpoint reporting and the three held-out JPGs are loaded only after
  the final PLY and NPZ have been saved on the ordinary coordinator path. The final checkpoint
  policy and disabled early stopping prevent validation/test selection on that path.
- Confirmed the declared 1,500-step CUDA gsplat configuration, density schedule, 50,000-Gaussian
  cap, SH-degree schedule, image downscale, fixed seeds, final endpoint, Janelle image metrics,
  per-view held-out records, per-folder seed curves, six comparison manifests, and canonical child
  `index.html` renderer.
- Passed all 41 focused outcome-free checks in
  `tests/test_janelle_image_experiment_protocol.py` and `tests/test_experiment_contract.py` and
  compiled the experiment driver and contract module. These tests confirm the positive structure
  but do not exercise the execution-boundary, timing, peak-memory, or resume counterexamples below.

## Findings

The protocol is **rejected**. The scientific matrix is clear, but the exact executable is not
approval-worthy until the following fail-closed defects are repaired and independently reviewed
under a new digest.

1. **Blocking / critical -- the worker CLI can expose held-out outcomes outside the protected
   run.** The direct `--worker --scratch` branch passes `scratch=True` but does not force
   `warmup=True`, does not constrain the output below `.scratch/`, and deliberately disables the
   ready-task check. A caller can therefore request an arbitrary short scratch cell with
   `warmup=False`; it falls through to the held-out loader and writes test metrics even while the
   task is still draft. The non-scratch worker is likewise lockless and accepts arbitrary output
   paths and iteration counts. Make workers private/authenticated children of the canonical
   coordinator, bind them to the locked cell identity and output path, reject non-frozen official
   iterations, and make every scratch mode structurally held-out-inaccessible.
2. **Blocking / high -- validation is evaluated twice and contaminates the frozen optimizer-time
   axis.** The scene passed to `Trainer.train` contains the three validation cameras as
   `testing_views`. At every evaluation checkpoint, Trainer evaluates those views before recording
   `history["elapsed"]`; it subtracts only checkpoint-callback time. The driver then evaluates the
   same validation views again in its callback. Consequently `rgb_refinement_wall_seconds` and the
   convergence AUC x-axis include an unreceipted internal validation pass, while
   `validation_observer_seconds` omits that pass and snapshot-construction time. The internal cost
   is also arm-dependent because Trainer sees masks only in the masked arm. Provide a train-only
   Trainer scene and a supported no-internal-evaluation mode, or explicitly isolate and subtract
   every evaluation/snapshot cost. Add a counterexample test proving native elapsed checkpoints
   exclude all validation work.
3. **Blocking / high -- CUDA and total-wall receipts do not measure the declared interval.** The
   driver performs an initial validation render before training, but Trainer resets CUDA peak
   statistics at training entry. The driver samples the reported peak only after held-out
   evaluation. Thus the claimed compact-open-through-final-PLY peak excludes pre-training CUDA
   allocations and can include post-endpoint test rendering. Likewise `total_cell_wall_seconds` is
   sampled after held-out/presentation work although the frozen scope ends at the synchronized
   final PLY, and the RGB stage's `end-start` includes save/serialization work that its `seconds`
   field omits. Freeze one unambiguous interval per metric, capture its endpoint before held-out
   access, and test peak-reset, interval, and stage-additivity semantics.
4. **Blocking / critical -- resume and aggregation permit stale or copied cell substitution.** A
   warmup is skipped for any existing `summary.json`. A measured cell is skipped for any summary
   whose only checked field is `status == "completed"`; aggregation checks only `status` and
   `warmup`. It never verifies that the payload's task, dataset, arm, seed, split, effective
   configuration, carrier receipt, or artifacts match the canonical path, and it does not hash the
   cell bundle. A copied completed cell can therefore stand in for another seed/arm/folder and
   still contribute to the alleged 36-cell result. Require a type-strict per-cell receipt bound to
   the task lock, exact identity/config/splits/input digests, and hashes of all required cell
   artifacts; validate it both before resume and before aggregation, then prove copied/stale/tampered
   counterexamples fail.
5. **Blocking / high -- the new source/data binding is recorded but not enforced at execution.**
   The contract now hashes tracked and untracked development source at `init-run`, but
   `source_diff_sha256` is subsequently checked only for syntactic shape; neither the producer nor
   `check-run` compares the live source state with it. The coordinator checks only the lock's task
   id and command. The sealed-input bytes are also rehashed at initialization but not at producer
   entry or final validation. In the worker, compact-file size and expected metadata are checked,
   but the loaded outer digest is not compared with the seal; actual JPG/mask bytes are loaded
   without rehashing. Bind the exact reviewed behavior-bearing source tree, verify that binding at
   coordinator/worker entry and bundle validation, revalidate the complete data seal before outcome
   access, and compare actual compact, JPG, mask, calibration, and full camera records (including
   principal point and dimensions) with their frozen values.
6. **Secondary / requirements -- the orbit viewers are generated but not opened.** The coordinator
   writes comparison manifests and prints their paths; `_viewer_command` explicitly supplies
   `--no-open`, and no process launches the six viewers. This does not corrupt the science, but it
   does not fulfill the requested inspection behavior. Freeze a presentation policy that launches
   or clearly hands off all six viewers only after all measured resource intervals are closed, so
   viewer GPU/server activity cannot contaminate later cells.

The task must remain non-executable. Any correction to these protocol-bearing surfaces requires a
fresh prospective digest and a new outcome-unseen review; the present artifact must not be edited
into an approval.

## Protected Actions Not Taken

I did not transition the task, initialize or execute the producer, invoke a worker or scratch cell,
create or enumerate the protected run, open a result/evidence artifact, inspect any model, metric,
preview, child report, or orbit output, launch a viewer, or consume a held-out image outcome. Checks
were limited to the frozen task/design/source/seal, outcome-free hash and schema validation, static
control-flow/API inspection, compilation, and focused structural tests. Outcome Access remained
`none` throughout.
