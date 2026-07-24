# Scientist pass — pooled structure/WSE 10k checkpoint study

Date: 2026-07-24

Machine-readable audit:
[`20260724_pool_structure_wse_10k_frame00008_AUDIT.json`](20260724_pool_structure_wse_10k_frame00008_AUDIT.json),
SHA-256 `84a8924084fe08af5f58ee5f6abdc906176d0a1c626952b0ad5551ffac0089e3`.

Disposition: **16/16 checks passed; WSE has a sustained scoped downstream win, but 10k itself does
not improve held-out quality**

## Claim inventory and disposition

| # | Claim | Kind and scope | Disposition |
|---|---|---|---|
| 1 | WSE beats no-WSE density sampling under pooling over the 10k trajectory | Preregistered single-scene/seed downstream claim | **Confirm narrowly: gate passes at all five checkpoints; +0.7737 dB held out at 10k** |
| 2 | WSE beats pooled gradient over the 10k trajectory | Preregistered single-scene/seed downstream claim | **Confirm narrowly: gate passes at all five checkpoints; +0.3702 dB held out at 10k** |
| 3 | Density without WSE beats pooled gradient | Preregistered downstream claim | **Retire: fails all five checkpoints; −0.4035 dB at 10k** |
| 4 | Training beyond 2k improves held-out quality | Checkpoint-trajectory claim | **Retire: every arm peaks at the first 2k reporting snapshot** |
| 5 | The best selectable checkpoint is 2k | Selection claim | **Not authorized: `C1004` is reporting-only and no validation selection ran** |
| 6 | WSE is an end-to-end combined winner | Stage-1 plus downstream claim | **Not established: this run reuses parent initials; both parent structure stage-1 gates failed** |
| 7 | A default should change or the result generalizes | Production claim | **Not authorized** |
| 8 | Timings or memory are performance evidence | Performance claim | **Not authorized: unreserved, unrepeated GPU** |

## Chronology, binding, and isolation

The protocol predates the plan and binds the question, exact parent states, fresh 10k schedule,
checkpoint times, metrics, thresholds, interpretation, and artifact contract. The official result
binds revision `7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11`, the dirty-tree status/diff,
all executed source hashes, calibration and raw inputs, loaded tensors, environment, parent
summary, and output artifacts.

The copied initial NPZs and PLYs are byte-identical to their audited parent states. The only
effective refinement changes from the parent final config are
`iterations: 2000→10000` and `schedule_iterations: None→10000`. Camera order replays exactly with
seven train-only cameras and reporting-only `C1004`; all 30,000 sampled training-view ids are in
`[0,6]`, and the three sampled sequences are identical.

## Independent state, history, and metric replay

- All initial, checkpoint, and final NPZ/PLY states load finite with their reported counts.
- Every returned final state is tensor-identical to its captured 10k state.
- Exact CUDA/gsplat replay of initialization plus 15 checkpoint metric records differs from the
  stored values by at most `0.0` under the audit tolerance.
- Each history has exactly 10,000 losses and sampled views plus 100 evaluation/count entries.
- Density count changes end at completed step 1,000 for every arm, followed by 9,000 recovery
  steps.
- Final policy is used; no selected checkpoint exists.

## Gate and trajectory replay

| Contrast | 2k FG Δ | 4k FG Δ | 6k FG Δ | 8k FG Δ | 10k FG Δ | 8k+10k sustained |
|---|---:|---:|---:|---:|---:|---|
| density vs gradient | −0.4276 | −0.4472 | −0.4701 | −0.3680 | −0.4035 | no |
| WSE vs gradient | +0.2246 | +0.2934 | +0.3743 | +0.3586 | +0.3702 | **yes** |
| WSE vs density | +0.6522 | +0.7406 | +0.8443 | +0.7266 | +0.7737 | **yes** |

Alpha-IoU and train-PSNR guardrails pass for both WSE contrasts at every checkpoint. They cannot
turn this into a general or end-to-end claim; they establish only the preregistered downstream
observation from the fixed parent initializations.

All three reporting-only held-out trajectories decline after 2k:

- gradient: 22.5578→22.1985 dB (`−0.3593`);
- density: 22.1302→21.7950 dB (`−0.3352`);
- WSE: 22.7824→22.5687 dB (`−0.2137`).

This is evidence against a 10k quality benefit, not permission to select 2k on the held-out
camera. A shorter production schedule needs a new train-only validation protocol.

## Visual, viewer, and page audit

All 39 raster visual artifacts decode, including the per-arm five-frame progress GIFs. The
trajectory SVG contains the expected six series. The required summary-bound `index.html` has 15
checkpoint cards, 18 metric rows, and 60 relative links. Before this audit wrote its own JSON,
all non-self-referential local targets existed; the final HTTP receipt checks the completed link
set:
[`20260724_pool_structure_wse_10k_frame00008_INDEX_RECEIPT.json`](20260724_pool_structure_wse_10k_frame00008_INDEX_RECEIPT.json).

The CPU checkpoint viewer loaded all 30 manifest model entries, returned HTTP 200 with a
2,888,259-byte response, owned the listening socket, used no NVIDIA compute process, and stopped
with its PID gone and port closed.

## Corrections and restrictions

The parent 2k endpoints must not be treated as repeatability controls. The current 2k states use a
10k means-LR schedule, while the parent endpoints used a 2k schedule, so their differences are
expected treatment-by-schedule differences rather than an estimate of CUDA drift.

Visual differences are subtle relative to the metric deltas. The checkpoint sheet supports a
modest WSE edge and persistent edge/hair/hand/dress residuals; it does not support a categorical
quality leap.

## Final decision

Keep all defaults unchanged. Preserve WSE as the existing structure initializer behavior and
density-prefix as a research control. The new evidence justifies replication of the WSE 10k
downstream advantage, not promotion. Required next evidence is a train-only validation checkpoint
rule plus paired multi-seed, multi-scene, multi-heldout-camera runs. Runtime and memory require a
separate idle-GPU repeated benchmark.
