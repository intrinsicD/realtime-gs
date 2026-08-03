# Prospective Protocol Review

- Task ID: `20260801_paper_three_provider_fullres_stage_frame00008`
- Protocol SHA-256: `6115bb16b1ef170e9c93095de3a40b60f79cf6ef898b82ace86b0b56c2d76a0d`
- Reviewer: `Goodall`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This review covers only the frozen prospective design, sealed compact inputs, image-free mechanism,
execution contract, metric alignment, resource boundary, and required evidence bundle. It does not
assess any experimental outcome. The design may ultimately support a source-bound, single-frame,
development comparison of the three registered compact providers and three registered initializers.
It cannot support causal provider ranking, generalization, physical-geometry accuracy, a production
default, or a VRAM claim.

## Checks

- Recomputed the exact prospective digest above, validated the experiment task, and validated the
  sealed data without initializing a run.
- Confirmed all 85 sealed files and 24,831,997 bytes are present and hash-bound, including the three
  provider manifests, production manifests, 78 compact fields, and calibration.
- Independently loaded the compact inputs with alpha disabled. Each provider has the frozen 26-view
  partition with 23 train and three disjoint held-out views. GaussianImage and mask-contained each
  contain 11,000 primitives per view; no-boundary contains 5,000 to 8,592.
- Confirmed the provider semantics are frozen as native/additive for GaussianImage and
  normalized/StructSplat for both StructSplat variants, all at 5328x4608 with the registered sigma,
  fade, and antialias settings.
- Recomputed the train-only initializer feasibility without training or writing outputs. The common
  matched count is 3,293; mask-contained and no-boundary independently admit 3,857 and 3,727
  Splat-SfM candidates, and Beam admits 5,000 for every provider. Held-out views were excluded.
- Confirmed the live no-image guard rejects the four registered negative controls and observed no
  image open or forbidden import attempt during the mechanism preflight.
- Confirmed reconstruction uses train inputs only, all held-out evaluation occurs after training,
  the three measured seeds and one warmup are frozen, and the nine provider/initializer cells share
  the registered optimizer and topology settings.
- Focused outcome-free verification passed: 48 experiment-contract, compact-view, initializer,
  density, and driver tests; Ruff passed for the driver and its test; relevant whitespace checks
  passed.

## Findings

Revision required before initialization. The following protocol-bearing gaps are blocking:

1. The directional hypothesis that mask containment reduces boundary leakage is not tested by the
   frozen primary metric. `heldout_sampled_j_area` measures each reconstruction against its own
   provider teacher and does not define boundary leakage or a common target. Add a prospectively
   frozen boundary-leakage endpoint with a valid common reference and protected evaluation boundary,
   or narrow the hypothesis and interpretation to provider-native compact-field fidelity.
2. The driver does not emit or enforce several frozen resource fields: process-level NVML peak,
   background device allocation, GPU idle-state comparability, driver version, and complete input
   and output byte accounting. It also constructs all three initializers inside every arm worker,
   so arm resource measurements include unrelated initializer work. Align measurement scope and
   implementation, retain every repeat, and expose the frozen fields before approval.
3. The exact command cannot satisfy the mandatory v2 preview gate. It writes per-cell previews but
   not the required root reconstruction contact sheet and orbit/elevation GIFs. Produce the promised
   root artifacts through the frozen image-free presentation stage, or revise the task and gate
   explicitly; bypassing previews would contradict the current protocol.
4. The frozen viewer smoke command opens only the representative top-level PLY, although the driver
   publishes a nine-method comparison manifest. Make the exact command consume that comparison
   manifest and verify all registered provider/initializer outputs are inspectable and synchronized.
5. Worker failure currently leaves only a temporary `WORKER_FAILED.txt` and re-raises. It does not
   atomically publish the required v2 failed-run receipt, source records, and structured failure
   evidence. Freeze a fail-closed publication path that remains auditable without presenting partial
   diagnostics as a completed result.
6. Canonical source binding accepts a dirty development lock because `development=true` bypasses the
   clean-source condition. The exact protected execution must reject development/dirty locks and
   require the reviewed base commit plus the frozen source hashes.
7. The task promises an identical sample stream, but provider-conditioned fit windows and proposal
   distributions make realized coordinates provider-specific. Either freeze common evaluation
   coordinates where the intended comparison permits them or state precisely that only RNG seeds
   and algorithms are shared. Separately document the all-initializer worker overhead if it remains
   part of every cell's timing boundary.

Any correction changes the reviewed protocol or its executable evidence contract and therefore
requires a new digest and a fresh outcome-unseen prospective review.

## Protected Actions Not Taken

The reviewer did not initialize or execute the experiment, access a task run or result path, train
any cell, inspect downstream outcomes, render results, or open protected RGB or mask inputs. Checks
were limited to sealed compact-input QA, static contract inspection, focused tests, and an
outcome-free mechanism preflight.
