# Independent Results Audit

- Task ID: `20260927_silhouette_relocation_index_decode_stage_frame00008`
- Protocol SHA-256: `27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815`
- Reviewer: `Claude-Code-Fable-5.1-reviewer` (model `claude-fable-5-1`, effort max)
- Driver: `Claude-Code-Opus-5.5-driver` (distinct label)
- Self-reviewed: no
- Outcome Access: full post-run results audit
- Verdict: `accepted_with_limits`
- Audit date: 2026-09-27
- Preview authorization: user, 2026-09-27 ("Ja bitte"), recorded in `.scratch/<task_id>/claude_results_audit/authorization.json`. The payload manifest hashes to `61a778d3…`, as the authorization record states. All 85 listed items, including the 13 dome-derived contact sheets, match their listed SHA-256 on disk.

## Scope and stance

This is the separate post-run referee pass over the run that the same reviewer label approved prospectively at digest `27f7b23a…`. Read in full: the 85 manifest items (run-level records, 12 cell records, 13 contact sheets, the RESULT Markdown and JSON, the task JSON, the approved review, the predecessor task JSON and its three review records, the failed-run README), the driver, the report module, `rtgs.optim.silhouette_relocation`, the trainer seam (lines 896-912), `rtgs.data.field_targets`, the `visualize` contact-sheet helpers, the protocol test file, the gsplat 1.5.3 `DefaultStrategy.step_post_backward`, the rtgs strategy wiring, the experiment lifecycle section of `experiments/README.md`, and the task record. RESULT.md, RESULT.json, the `metrics.json` summary and decision, and every Driver handoff were treated as claims and re-derived from the raw cell records.

Not opened: photographs, masks, `.rtgsv`, `.npz`, `.ply`, `targets/`, `renders/`, GIFs, logs, `gaussians.config.json`, the archived source copies under `source_snapshot/files/`, the consumed predecessor root, and the RTGS-025 RESULT and AUDIT records.

## 1. Chronology and single invocation

| Event | Time (UTC) | Evidence |
|---|---|---|
| Approval of this protocol committed (`70e4bc6`) | 2026-09-27 12:45:58 | `git log` |
| Lock written by `init-run` | 12:46:04.98 | `task.lock.json` |
| Launch context written (GPU shared) | 12:46:04 | `launch_context.json` |
| `prepare` | +8.2 s to +68.4 s | `preparation.json` |
| `initialize` | +69.4 s to +71.9 s | `initialization.json` |
| 12 `fit` cells in the frozen order | +73.5 s to +1905.1 s | 12 cell receipts |
| `evaluate`, 12 cells | +1912.7 s to +1938.8 s | 12 `evaluation.json` |
| Run receipt `completed`, exit 0 | 13:18:24.81 (+1939.8 s) | `run_receipt.json` |

Findings. One continuous timeline under one lock; the cells ran in exactly the frozen `execution_order` with 1.5 to 2 s process gaps. No `execution_failure.json`, `attempts/`, `production/`, `index.html`, `README.md`, `manifest.json` or `viewer_smoke.json` exists under the run root, so there was no re-entry, no failure path, no production phase and no render yet. The two RESULT files are untracked in git and regenerate byte-identically from `comparison.json` plus the task JSON, so the write-once rule was respected. The predecessor's failed run (12:20:08 to 12:20:50 per the preserved README) precedes the retry registration commit `492996f` (12:26:56); its root is different and was not entered.

## 2. Source, protocol and data binding

| Check | Result |
|---|---|
| HEAD versus lock | `70e4bc6` equals lock `source_commit`; the worktree differs only by the appended task-record handoff and the two untracked RESULT files |
| Source snapshot manifest | 122 entries (120 bound files, the seal, the review); every entry equals the live tree by SHA-256 and byte count; `git_diff_sha256` is the SHA-256 of the empty string, so the tracked diff was empty at snapshot; lock `source_dirty` false, `development` false |
| Executed source equals reviewed source | Snapshot digests of the driver `d2f68c21…`, report `7c234f01…`, protocol test `57202cdb…`, mechanism `e61981e0…`, trainer `e226d015…`, `field_targets.py` `59901d82…`, contract `15f12939…`, bundle checker `0f5e1293…` and `pyproject.toml` `e5b15e72…` equal the identities in the approved review |
| Task bytes | `fdcfca2b…` equals the lock; `status: ready`; verdict `approved`; digest in the task equals the lock |
| Review artifact | `a58ce3e9…` equals the lock |
| Data seal | `d01254e7…` equals the lock and is byte-identical to the predecessor and RTGS-025 seal |
| `review-digest` | prints `27f7b23a…` |
| `validate` / `validate-data` | OK / OK |
| Input integrity | entry and exit receipts: 81 sealed files, all match; exit split digest `0ab039b7…` recomputed from the task splits |
| Source-binding aggregate `8dacd07f…` | not reproduced (no allowed tool encodes it); bound instead by snapshot-versus-live equality and by `init-run`'s own fail-closed check |
| Environment | Python 3.12.9, torch 2.9.0, numpy 2.1.3, gsplat 1.5.3, lpips 0.1.4, CUDA 12.8, NVIDIA GeForce RTX 3050; `environment.json` is byte-identical (`48df639a…`) to the predecessor's |
| Predecessor task diff | 20 changed lines: identifiers, paths, title suffix, decoder wording, the appended failure disclosure, the aggregate; nothing else |
| Focused CPU tests | 17 passed (11 mechanism, 6 protocol) |

## 3. Input boundary per phase

| Phase | Dataset opens recorded | Denied | Reading |
|---|---|---|---|
| `prepare` | 24 training photographs and masks; all 26 compact views, the two held-out views only in the hull loop | none | No held-out photograph or mask PNG was opened. The CPU selftest inside `prepare` denied its 4 forbidden probes |
| `initialize` | none | none | Hull and three seeded initializations built from cached targets only |
| `fit` (12 cells) | none | none | Only the cell's own target family, training masks, hull cache and metadata are permitted |
| `evaluate` | C0001 and C0029 photographs, masks and compact views | none | First held-out colour access; the driver first requires all 12 receipts to read `completed` |

Held-out views contributed only packed alpha, camera and recorded source digests before `evaluate` (driver 497-511). The held-out `.rtgsv` files are parsed whole by `CompactView.load`; that their colour fields are unused rests on the code read here and in the prospective reviews, not on the audit hook. Decoder parity: index-versus-reference max abs error 5.96e-8 to 1.79e-7 on all 24 training views against the frozen 1e-5 abort; `make_query_backend` is called once with `"index"` and every `decode_compact_view` call passes `backend="index"`, so no CUDA decoder ran. Hull: 26 dilated masks, 21391 occupied voxels of side 0.01749, bounds from the 24 training views only; initial hull-rejected fractions 0.9816, 0.9805, 0.9810 for seeds 9361, 9362, 9363. Teacher consistency (decoded field versus photograph inside training masks): mean 30.54 dB over 24 views, range 24.90 to 37.68 dB, descriptive only.

## 4. Cells, configuration and relocation events

Every cell: initial digest equals `initialization.json` for its seed, `executed_iterations` 8000, `stop_reason` max_iterations, 8000 finite losses, effective `train_config` equal to the frozen resolved config, hull and relocation config equal to the task, 24 training view ids, `gsplat-default` density with `dynamic` storage, SH interval 1000, 8000 sampled training draws over exactly 24 views, `dataset_opens` empty.

| Cell | Final N | Fit wall s | Peak alloc MB | Events | First event moved / jitter fallbacks | Second event | Total moved (events 3-66) | Last event | Final hull-rejected |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| nb_ms / 9361 | 45529 | 149.7 | 147.0 | 0 | | | | | 0.0537 |
| nb_ms_reloc / 9361 | 46565 | 147.7 | 156.6 | 66 | 19393 / 6227 | 1214 | 29821 (9214) | 17 | 0.0015 |
| ph_ms / 9361 | 48183 | 151.3 | 150.7 | 0 | | | | | 0.0431 |
| ph_ms_reloc / 9361 | 48424 | 152.2 | 160.2 | 66 | 19396 / 6147 | 1230 | 29319 (8693) | 7 | 0.0010 |
| nb_ms_reloc / 9362 | 44658 | 152.1 | 156.9 | 66 | 19397 / 6210 | 1221 | 29653 (9035) | 12 | 0.0019 |
| ph_ms / 9362 | 48352 | 150.4 | 151.0 | 0 | | | | | 0.0451 |
| ph_ms_reloc / 9362 | 47768 | 152.7 | 158.8 | 66 | 19402 / 6271 | 1193 | 29187 (8592) | 12 | 0.0013 |
| nb_ms / 9362 | 43821 | 150.5 | 145.0 | 0 | | | | | 0.0540 |
| ph_ms / 9363 | 47591 | 150.6 | 149.0 | 0 | | | | | 0.0440 |
| ph_ms_reloc / 9363 | 48206 | 152.1 | 161.3 | 66 | 19406 / 6252 | 1070 | 29233 (8757) | 17 | 0.0012 |
| nb_ms / 9363 | 44287 | 148.8 | 145.3 | 0 | | | | | 0.0562 |
| nb_ms_reloc / 9363 | 45901 | 152.6 | 157.7 | 66 | 19405 / 6149 | 1157 | 29609 (9047) | 14 | 0.0021 |

Relocation schedule: all six relocation cells record exactly the 66 events at completed steps 500, 600, ..., 7000 and the six no-relocation cells record none. Jitter fallbacks are recorded per event (about a third of the first event's rows, 15 to 25 percent later). Count invariance: the `n_gaussians` value of every event equals the trainer's checkpoint count at the same completed step in all six cells, and the trainer seam raises if a callback changes the count.

New mechanism finding from the density statistics. The first gsplat refinement (global iteration 600, completed step 601) found 19149 to 19297 of the 20000 initial Gaussians at opacity 0.005 or below and pruned them in every one of the 12 cells, relocated or not, leaving 867 to 1030. The count then regrows through cloning and splitting to the final 43.8k to 48.4k by step 6000 (54 density events from 600 to 5900, no budget prunes). The hull-guided placement of the random start, disclosed in the claim boundary as part of the method, is therefore mostly transient: about 96 percent of the relocated start points are gone 100 iterations later. The lasting relocation effect acts on the roughly 1000 survivors and on the 8.6k to 9.2k relocations of grown Gaussians over events 3 to 66. The claim-boundary sentence stays true, but "hull-guided placement" should not be read as a persistent initialization.

The opacity-reset disclosure is confirmed on this run's own data: the density record at iteration 3000 shows `n_before == n_after` in every cell checked, and the loss traces show no spike after iterations 3000 or 6000 (windows 2980-3000 versus 3000-3020 and 5980-6000 versus 6000-6020 are flat or lower). The installed gsplat 1.5.3 line reads `if step % self.reset_every == 0 & step > 0:`, which never executes; rtgs wires `reset_every=3000` and `pause_refine_after_reset=100`.

## 5. Independent recomputation from per-view rows

Recomputed means over the two held-out views and the three seeds equal `comparison.json` and RESULT.json to 1e-12; the saved per-cell means equal their per-view rows; view order is C0001, C0029 in every cell.

| Condition | foreground_psnr (dB) | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha | hull_rejected_fraction |
|---|---:|---:|---:|---:|---:|---:|
| nb_ms | 23.157871 | 0.167610 | 0.001782 | 0.001470 | 0.998727 | 0.054677 |
| nb_ms_reloc | 23.220908 | 0.164909 | 0.001314 | 0.001026 | 0.998808 | 0.001825 |
| ph_ms | 23.717539 | 0.142658 | 0.001695 | 0.001403 | 0.998775 | 0.044078 |
| ph_ms_reloc | 23.698802 | 0.141619 | 0.001368 | 0.001077 | 0.998893 | 0.001171 |

H1 in its written form (pass iff, for every seed, reloc PSNR >= base PSNR + 0.1 and reloc LPIPS <= base LPIPS + 0.005; reject iff, for every seed, reloc PSNR <= base PSNR - 0.1; otherwise inconclusive):

| Seed | nb_ms PSNR | nb_ms_reloc PSNR | Delta PSNR | nb_ms LPIPS | nb_ms_reloc LPIPS | Delta LPIPS | Pass | Reverse |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 9361 | 23.069614 | 23.297866 | +0.228251 | 0.170700 | 0.163625 | -0.007075 | yes | no |
| 9362 | 23.231079 | 23.215189 | -0.015890 | 0.168418 | 0.163751 | -0.004667 | no | no |
| 9363 | 23.172918 | 23.149668 | -0.023251 | 0.163713 | 0.167350 | +0.003637 | no | no |

Recomputed verdict: inconclusive. It equals the producer's verdict and every RESULT.json row.

Per-view decomposition of the H1 pairs:

| Seed | View | nb_ms | nb_ms_reloc | Delta PSNR | Delta LPIPS |
|---|---|---:|---:|---:|---:|
| 9361 | C0001 | 20.7207 | 21.1311 | +0.4105 | -0.0117 |
| 9361 | C0029 | 25.4186 | 25.4646 | +0.0461 | -0.0024 |
| 9362 | C0001 | 20.9467 | 20.9709 | +0.0242 | -0.0120 |
| 9362 | C0029 | 25.5155 | 25.4595 | -0.0560 | +0.0027 |
| 9363 | C0001 | 20.9162 | 20.9333 | +0.0171 | +0.0036 |
| 9363 | C0029 | 25.4297 | 25.3660 | -0.0636 | +0.0037 |

The single passing seed is carried by C0001; C0029 moves by less than 0.07 dB in every seed, negative in two.

Descriptive pairs (delta PSNR dB / delta crop LPIPS, seeds 9361, 9362, 9363), all matching RESULT.json:

| Pair | 9361 | 9362 | 9363 |
|---|---|---|---|
| ph_ms_reloc minus ph_ms | +0.037866 / -0.000911 | -0.006865 / -0.004303 | -0.087214 / +0.002097 |
| nb_ms_reloc minus ph_ms_reloc | -0.410354 / +0.021843 | -0.552173 / +0.025363 | -0.471156 / +0.022664 |
| nb_ms minus ph_ms | -0.600739 / +0.028007 | -0.543147 / +0.025726 | -0.535120 / +0.021124 |

Under photograph targets the two views disagree in sign in every seed (C0001 +0.288, +0.070, +0.027 dB; C0029 -0.212, -0.084, -0.202 dB). The in-sample alpha descriptors move the same way in all six view pairs of the field arms: outside-mask alpha mass and floater fraction fall, and the final hull-rejected fraction falls from about 5.5 percent to about 0.2 percent.

RESULT.md, RESULT.json (summary, decision, claim boundary, command, embedded source lock) and `metrics.json` (summary, decision, notes, four declared evidence files, viewer command) were regenerated or cross-checked from `comparison.json` and the task JSON and agree. `training_history.json` holds 384 records (12 cells, 2 metrics, 16 checkpoints) and 96 stage markers.

## 6. Visual disposition

Viewed: the 12 cell contact sheets and the root sheet (identical bytes to `cells/nb_ms_reloc/9361`, as the frozen preview policy requires). Each sheet shows, per held-out view, the mask-multiplied photograph, the render of the random-ball initialization, the final render and a fourfold error image.

Disposition: adequate at contact-sheet scale, with no reconstruction failure in any cell. All 12 final renders reproduce the subject's silhouette, pose and colours for both held-out views; the initial panels show the expected grey ball. Every arm shows a bright rim along the whole silhouette in the error panels (render-versus-mask-edge disagreement common to all arms) and faint interior texture error, denser for C0001 than for C0029, consistent with C0001's roughly 4.5 dB lower PSNR. A few isolated bright specks in the black background are visible in the field arms without relocation, mainly around C0001, and fewer in the relocation arms. No arm shows a halo, ghost or detached fragment outside the silhouette beyond that rim, and no arm shows a collapsed or missing body part.

Limit: the reader rendered the 5328 by 600 sheets at 2000 by 225 pixels. An attempt to crop native-resolution panels into the session scratchpad was denied by the sandbox and was not retried, and the individual renders are excluded by the payload policy. Floater and halo differences between arms are therefore not visually decisive here; the quantitative basis is the in-sample alpha descriptors above. Visual adequacy for the viewer handoff still requires the browser smoke recorded in `viewer_smoke.json`.

## 7. Metric semantics, in-sample status and required disclosures

Metric semantics verified in the driver (`mask_scores`): held-out masks come from the four-site quadrature of the lossless mask, binarized at 0.5; foreground PSNR is the channel-mean squared error of the clamped black-background render against the photograph over mask pixels only (C0001 27927 pixels, C0029 30518); outside band is the complement of the 3-pixel-dilated mask (350419 and 348081 pixels), interior band the 3-pixel-eroded mask (22941 and 25742); floater fraction counts outside-band pixels with alpha at or above 0.5; crop LPIPS uses AlexNet with `normalize=True` on the mask-multiplied render and photograph cropped to the mask box padded by 8 pixels; hull-rejected fraction applies the frozen 26-view hull to the final means once per cell. The field-consistency PSNR against the CPU-index-decoded held-out field is recorded as a diagnostic only and does not enter any gate.

In-sample status: the 26-view hull includes the two held-out silhouettes, so outside-mask alpha mass, floater fraction, hull-rejected fraction and also interior alpha are in-sample descriptors of what the relocation objective directly targets. The primary-metric label of `interior_alpha` lacks the "(in-sample)" tag the other three carry; it should be treated as in-sample. Held-out colour inside the mask is the only novel-view measurement, and in the relocation arms it is novel colour under in-sample silhouettes, because the held-out masks shaped the geometry through the hull.

Disclosures required by residual condition 5 of the approved review, and where each is carried:

| Disclosure | RESULT.md | This audit |
|---|---|---|
| Frame 00008 is outcome-exposed (RTGS-021/024/025) | yes | yes |
| Held-out alpha, floater and hull-rejection metrics are in-sample | yes | yes, extended to interior alpha |
| The first relocation event re-initializes most random start points onto the hull | yes | yes, quantified (19393 to 19406 of 20000) and qualified (about 96 percent pruned at the first refinement in every arm) |
| gsplat refinement after each event uses screen-gradient statistics accumulated partly at pre-relocation positions | no (task JSON `relocation.definition` only) | yes |
| The installed gsplat 1.5.3 never executes its opacity reset, also for the RTGS-025 baseline | no (task JSON only) | yes, confirmed on the loss and density traces |
| `heldout_mask_alpha` is a declared reconstruction input | in substance (all 26 masks define the hull) | yes |
| First pre-review smoke (60 iterations, training stand-ins) exposed relocation counts, hull size and a meaningless gate value; no threshold changed | yes | yes |
| Second pre-review smoke (`prepare` plus `initialize` over the frozen split) exposed per-view index parity, the hull voxel count, per-view teacher PSNR and initial hull-rejected fractions; no threshold changed | no | yes; the official values (parity at most 1.79e-7, 21391 voxels) agree with the handoff's smoke figures, and none of them enters the gate |
| Predecessor task failed closed at its CUDA parity gate; receipts at `ara/evidence/tables/20260927_silhouette_relocation_failed_run/` | yes, without the path | yes |
| Targets, held-out diagnostic and production inputs are decoded with the exact CPU index; the `nb_ms` arm is an in-task re-fit whose targets differ from the CUDA-decoded targets of the published RTGS-025 run at float32 rounding level and by up to about 1.9e-4 at near-empty normalized-blend sites; RTGS-025 numbers are not comparators | partly ("the only design change is exact CPU-index decoding") | yes |

RESULT.md is generated from the frozen claim boundary and is write-once, so it cannot be amended. The `docs/EXPERIMENTS.md` entry and any ARA rows must repeat the four disclosures that RESULT.md lacks.

Resources and timing. `launch_context.json` records the GPU shared at launch with pid 3896722 (`experiments.latent_agent`, about 3.6 GB). Fit wall times of 147.7 to 152.7 s per cell, prepare 60.3 s, initialize 2.5 s, evaluate 2.1 to 2.8 s per cell, peak allocated 145.0 to 161.3 MB, peak reserved 211.8 to 245.4 MB and about 1.8 GiB host RSS are descriptive only; `resource_receipt.json` sets `performance_inference` false and no producer statement claims speed, throughput or memory advantage. None may be derived from this run.

## 8. Claim dispositions

| # | Producer statement (where) | Kind and scope | Evidence | Disposition | Basis and boundary |
|---|---|---|---|---|---|
| P1 | "12 paired development cells completed." (RESULT.md/json, metrics.json) | measured; real data; development; GPU | 12 receipts, histories | confirm | 12 receipts `completed`, 8000 iterations each, one invocation |
| P2 | Held-out foreground PSNR inside the mask: nb_ms 23.158, nb_ms_reloc 23.221, ph_ms 23.718, ph_ms_reloc 23.699 dB (RESULT) | measured; one outcome-exposed frame; one split; 3 seeds | recomputed from 24 per-view rows | confirm, narrowed | In-task numbers only. Not comparable to the published RTGS-025 numbers (different decoder); no absolute quality floor was frozen |
| P3 | "H1 ... inconclusive" (RESULT decision) | measured gate | recomputed in written form | confirm | Seed 9361 passes (+0.228 dB, LPIPS -0.007); seeds 9362 and 9363 neither pass nor reverse |
| P4 | "No production model is trained; the frozen rule is not met." (RESULT) | asserted state | run root listing, `production()` guard | confirm | No `production/` directory; gate inconclusive; production refuses without `pass` |
| P5 | Table rows for crop LPIPS, outside alpha mass, floater fraction, interior alpha, hull-rejected fraction (RESULT) | measured; alpha and hull rows in-sample | recomputed | confirm, narrowed | Mechanism descriptors. Interior alpha is also in-sample although its label lacks the tag |
| P6 | "Relocation lowered the final hull-rejected Gaussian fraction from about 5.5 percent to about 0.2 percent" (handoff) | measured; in-sample | 0.0547 to 0.0018; per seed 0.0537/0.0540/0.0562 to 0.0015/0.0019/0.0021 | confirm, narrowed | A compliance descriptor of the relocation objective under the frozen hull, not a quality statement; relocation stops at 7000, so the residual is post-schedule drift |
| P7 | "in-sample outside-mask alpha by about a quarter" (handoff) | measured; in-sample | 0.001782 to 0.001314 (-26 percent); lower in all six view pairs | confirm, narrowed | In-sample; says nothing about floaters seen from views whose masks are absent from the hull |
| P8 | "the first event moved about 19400 of 20000 start points" (handoff) | measured | 19393 to 19406 per cell | confirm, qualified | 19149 to 19297 of the 20000 initial Gaussians were pruned at the first refinement in every cell; the placement is mostly transient |
| P9 | Per-seed deltas +0.228, -0.016, -0.023 dB (handoff) | measured | recomputed | confirm | Seed-9361 gain is carried by C0001 (+0.41 dB); C0029 changes by less than 0.07 dB in every seed |
| P10 | "the only design change is exact CPU-index decoding" (claim boundary) | asserted | task diff, driver decode calls, parity records | confirm | 20-line task diff of identifiers, paths, wording, disclosure and aggregate; `backend="index"` at every decode; parity gate index-versus-reference only |
| P11 | Run locked at `70e4bc6`, clean, non-development, one invocation, exit 0, 12/12 cells, GPU shared (handoff) | asserted | lock, receipts, launch context | confirm | Section 1 and 2 |
| P12 | Descriptive photograph and field-versus-photograph pairs (RESULT.json gates) | measured; descriptive; no verdict | recomputed | confirm | No direction claim may be attached; the photograph pair's two views disagree in sign in every seed |
| P13 | "Numeric gates precede the independent audit and do not assert visual adequacy." (RESULT) | asserted | this audit | confirm | Visual disposition in section 6, with its resolution limit |
| P14 | Any reading that relocation improves novel-view colour or reduces novel-view floaters | hypothesis | H1 inconclusive; alpha metrics in-sample | unresolved | Needs a hull-initialized no-relocation arm, masks of views excluded from the hull, more frames and seeds, or a longer budget |

No producer statement is retired. Nothing in RESULT.md overstates its artifact.

## 9. Corrections, limitations and what promotion would need

Corrections to carry forward in prose and ledgers:

- Label interior alpha as in-sample wherever the alpha descriptors are repeated.
- Repeat in `docs/EXPERIMENTS.md` and any ARA row the four disclosures absent from RESULT.md (post-event refinement statistics, inert opacity reset, the second pre-review smoke, and non-comparability with the RTGS-025 published run).
- State that the hull-guided placement of the random start is mostly pruned at the first refinement, so H1 mainly measured ongoing relocation of the survivors and of grown Gaussians.
- Report the capacity difference alongside colour: the field relocation arms ended with 2.3, 1.9 and 3.6 percent more Gaussians than their baselines (photograph arms +0.5, -1.2, +1.3 percent), a mild confound that the design did not control.

Limitations of the evidence:

- Development screen on one outcome-exposed frame, one split, two held-out colour views, three paired seeds, one 8000-step budget at downscale 8. No default, SOTA, generalization, physical-geometry, speed or VRAM claim can follow.
- H1 is inconclusive; the production phase was correctly not run and no all-view model exists.
- All alpha and hull descriptors are in-sample by design; only held-out colour is novel, under in-sample silhouettes in the relocation arms.
- GPU contended at launch; every timing and memory number is descriptive.
- Visual inspection at contact-sheet scale only; the browser viewer smoke is still pending.
- The source-binding aggregate was not reproduced by the auditor; binding rests on snapshot-versus-live equality and `init-run`.
- The held-out compact views are parsed whole in `prepare`; alpha-only use is code discipline.

Evidence that would promote P14 to a supported or refuted claim: a preregistered arm initialized inside the hull without ongoing relocation, hull masks drawn only from training views with the held-out silhouettes withheld, at least one additional calibrated frame, more paired seeds, and the same per-seed written rule.

## 10. Commands executed, not executed, protected actions not taken

Executed (read-only, CPU):

```text
git status --short; git rev-parse HEAD; git log --oneline -8
git log --format='%h %cI %s' -4; git diff --stat HEAD; git diff HEAD -- .agents/state/current-task.md
git diff --no-index [--stat] experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_silhouette_relocation.py tests/test_silhouette_relocation_index_decode_protocol.py
.venv/bin/python -c <payload manifest hashes versus disk and authorization>
.venv/bin/python -c <source_snapshot/manifest.json versus live tree; task, review and seal bytes versus task.lock.json>
.venv/bin/python -c <per-cell receipt, effective_config, history and evaluation summary>
.venv/bin/python -c <group means, paired deltas, H1 verdict and descriptive pairs from per-view rows>
.venv/bin/python -c <RESULT.md regeneration from comparison.json; metrics.json and training_history.json checks>
.venv/bin/python -c <relocation events versus checkpoint counts; count trajectories; density statistics; loss windows at 3000 and 6000>
.venv/bin/python -c <split digest and run wall time>
rg / sed over src/rtgs/optim/trainer.py, src/rtgs/optim/strategies.py, src/rtgs/render/gsplat_backend.py, src/rtgs/visualize.py, scripts/experiment_contract.py, .venv/.../gsplat/strategy/default.py, docs/, ara/, experiments/README.md
```

Denied by the sandbox and not worked around: compound invocations of the contract commands and pytest (re-run plainly), a combined snapshot-and-copies hash script (re-run without the archived copies), and a PIL script that would have cropped native-resolution panels of the authorized contact sheets into the session scratchpad.

Not executed: `init-run`, the driver's `run`, `prepare`, `initialize`, `fit`, `evaluate`, `production` and `selftest` (the protocol test file runs `selftest` as a CPU subprocess), `render`, `check-run`, `scripts/check_results_bundle.py`, the viewer, `verify.sh`, the full test suite, any GPU work, and any reproduction of the source-binding aggregate. No file under the repository or the run root was created, modified or deleted; the RESULT files were read only; no checkpoint was selected; the owner's task record was not written.

## 11. Remaining handoff checks

1. Persist this audit verbatim as `benchmarks/results/<task_id>_AUDIT.md` and the JSON block below as `<task_id>_AUDIT.json`; commit both RESULT files unchanged with them.
2. Run `render` once, serve the report, exercise the viewer in a real browser with the frozen viewer command, write `viewer_smoke.json`, rerun `render`, then pass both `check-run` and `scripts/check_results_bundle.py`. The production sequence is not triggered because the gate is inconclusive.
3. Append the dated outcome to `docs/EXPERIMENTS.md` with the disclosures of section 7 and the corrections of section 9; add ARA observations for the in-sample descriptors and the first-refinement prune finding, and a `hypothesis` row for H1 bound to the RESULT and AUDIT artifacts; update `docs/ROADMAP.md` status if it lists RTGS-026.
4. Close the task record: record this verdict under the Reviewer label, move Maturity Reached to Calibrated only after the viewer receipt and both gates pass, and archive to `docs/tasks/` on closeout.
5. Later tasks, not conditions: the hull-initialized no-relocation arm, hull masks without held-out silhouettes, the CUDA query kernel's near-zero-weight normalization as an ARA observation, a backend-aware field-target cache key, and the `interior_alpha` label.

```json
{
  "schema_version": 1,
  "task_id": "20260927_silhouette_relocation_index_decode_stage_frame00008",
  "reviewer": "Claude-Code-Fable-5.1-reviewer",
  "model": "claude-fable-5-1",
  "self_reviewed": false,
  "verdict": "accepted_with_limits",
  "protocol_sha256": "27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815",
  "outcome_access": "full post-run results audit",
  "audit_date": "2026-09-27",
  "authorization": {
    "record": ".scratch/20260927_silhouette_relocation_index_decode_stage_frame00008/claude_results_audit/authorization.json",
    "payload_manifest_sha256": "61a778d388e6e7933395ed71b0f5c44611a6f37662d41ad1d278065ca53fd377",
    "payload_items": 85,
    "payload_items_verified": 85,
    "dome_derived_previews_viewed": 13
  },
  "run_binding": {
    "run_root": "runs/20260927_silhouette_relocation_index_decode_stage_frame00008",
    "source_commit": "70e4bc6dcb09442b99e511fcd9a2ffa036275a06",
    "source_dirty": false,
    "development": false,
    "task_sha256": "fdcfca2bad4ebce826a2d822b8ef68c94dd44f1581e1856cb765dd28ed9a867d",
    "protocol_review_artifact_sha256": "a58ce3e924fe2c8d1305dcd2437f108c9848ac32a76c194a97a96c36bb62f400",
    "data_seal_sha256": "d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087",
    "source_snapshot_files": 122,
    "source_snapshot_live_mismatches": 0,
    "snapshot_tracked_diff_empty": true,
    "source_binding_aggregate_reproduced": false,
    "review_digest_reproduced": true,
    "validate": "OK",
    "validate_data": "OK",
    "focused_tests": "17 passed",
    "started_at_utc": "2026-09-27T12:46:04.984296+00:00",
    "finished_at_utc": "2026-09-27T13:18:24.806813+00:00",
    "wall_seconds": 1939.8,
    "exit_code": 0,
    "single_invocation": true,
    "re_entry": false,
    "execution_failure_present": false,
    "production_directory_present": false,
    "render_artifacts_present": false,
    "split_sha256": "0ab039b757b99ce9ae2eeb308013f3923263742fc3a417d4809b7c65ccbcc160",
    "input_integrity_files_checked": 81,
    "gpu_shared_at_launch": true,
    "environment": {"python": "3.12.9", "torch": "2.9.0", "gsplat": "1.5.3", "lpips": "0.1.4", "cuda": "12.8", "device": "NVIDIA GeForce RTX 3050"}
  },
  "input_boundary": {
    "prepare_heldout_photograph_or_mask_opens": 0,
    "initialize_dataset_opens": 0,
    "fit_dataset_opens": 0,
    "evaluate_first_heldout_colour_access": true,
    "denied_opens_all_phases": 0,
    "selftest_probes_denied": 4,
    "decoder_backend": "index",
    "parity_index_vs_reference_max": 1.7881393432617188e-07,
    "hull_occupied_voxels": 21391,
    "initial_hull_rejected_fraction": {"9361": 0.9816, "9362": 0.9805, "9363": 0.981}
  },
  "cells": {
    "count": 12,
    "all_completed_8000_iterations": true,
    "all_initial_digests_match": true,
    "all_configs_equal_frozen": true,
    "relocation_events_per_relocation_cell": 66,
    "relocation_schedule": "completed steps 500..7000 every 100",
    "first_event_relocated_range": [19393, 19406],
    "first_refinement_dead_before_range": [19149, 19297],
    "first_refinement_survivors_range": [867, 1030],
    "final_counts": {
      "nb_ms": {"9361": 45529, "9362": 43821, "9363": 44287},
      "nb_ms_reloc": {"9361": 46565, "9362": 44658, "9363": 45901},
      "ph_ms": {"9361": 48183, "9362": 48352, "9363": 47591},
      "ph_ms_reloc": {"9361": 48424, "9362": 47768, "9363": 48206}
    },
    "count_invariance_verified": true,
    "opacity_reset_inert_confirmed_on_traces": true
  },
  "official_gates": {
    "h1_relocation": {
      "rule": "pass iff for every paired seed nb_ms_reloc foreground PSNR >= nb_ms foreground PSNR + 0.1 dB and nb_ms_reloc crop LPIPS <= nb_ms crop LPIPS + 0.005; reject iff for every paired seed nb_ms_reloc foreground PSNR <= nb_ms foreground PSNR - 0.1 dB; otherwise inconclusive",
      "verdict_recomputed": "inconclusive",
      "verdict_producer": "inconclusive",
      "rows_match_result_json": true,
      "rows": [
        {"seed": 9361, "nb_ms_foreground_psnr": 23.069614, "nb_ms_reloc_foreground_psnr": 23.297866, "delta_foreground_psnr": 0.228251, "nb_ms_crop_lpips": 0.1707, "nb_ms_reloc_crop_lpips": 0.163625, "delta_crop_lpips": -0.007075, "pass": true, "reverse": false},
        {"seed": 9362, "nb_ms_foreground_psnr": 23.231079, "nb_ms_reloc_foreground_psnr": 23.215189, "delta_foreground_psnr": -0.01589, "nb_ms_crop_lpips": 0.168418, "nb_ms_reloc_crop_lpips": 0.163751, "delta_crop_lpips": -0.004667, "pass": false, "reverse": false},
        {"seed": 9363, "nb_ms_foreground_psnr": 23.172918, "nb_ms_reloc_foreground_psnr": 23.149668, "delta_foreground_psnr": -0.023251, "nb_ms_crop_lpips": 0.163713, "nb_ms_reloc_crop_lpips": 0.16735, "delta_crop_lpips": 0.003637, "pass": false, "reverse": false}
      ]
    },
    "production_gate": {"required": "h1_relocation == pass", "passed": false, "production_run": false, "production_model_exists": false},
    "descriptive_pairs": {
      "photographs_relocation": [
        {"seed": 9361, "delta_foreground_psnr": 0.037866, "delta_crop_lpips": -0.000911},
        {"seed": 9362, "delta_foreground_psnr": -0.006865, "delta_crop_lpips": -0.004303},
        {"seed": 9363, "delta_foreground_psnr": -0.087214, "delta_crop_lpips": 0.002097}
      ],
      "field_vs_photographs_with_relocation": [
        {"seed": 9361, "delta_foreground_psnr": -0.410354, "delta_crop_lpips": 0.021843},
        {"seed": 9362, "delta_foreground_psnr": -0.552173, "delta_crop_lpips": 0.025363},
        {"seed": 9363, "delta_foreground_psnr": -0.471156, "delta_crop_lpips": 0.022664}
      ],
      "field_vs_photographs_without_relocation": [
        {"seed": 9361, "delta_foreground_psnr": -0.600739, "delta_crop_lpips": 0.028007},
        {"seed": 9362, "delta_foreground_psnr": -0.543147, "delta_crop_lpips": 0.025726},
        {"seed": 9363, "delta_foreground_psnr": -0.53512, "delta_crop_lpips": 0.021124}
      ],
      "all_match_result_json": true
    }
  },
  "recomputed_groups": {
    "nb_ms": {"foreground_psnr": 23.157871, "crop_lpips": 0.16761, "outside_alpha_mass": 0.001782, "floater_fraction": 0.00147, "interior_alpha": 0.998727, "hull_rejected_fraction": 0.054677},
    "nb_ms_reloc": {"foreground_psnr": 23.220908, "crop_lpips": 0.164909, "outside_alpha_mass": 0.001314, "floater_fraction": 0.001026, "interior_alpha": 0.998808, "hull_rejected_fraction": 0.001825},
    "ph_ms": {"foreground_psnr": 23.717539, "crop_lpips": 0.142658, "outside_alpha_mass": 0.001695, "floater_fraction": 0.001403, "interior_alpha": 0.998775, "hull_rejected_fraction": 0.044078},
    "ph_ms_reloc": {"foreground_psnr": 23.698802, "crop_lpips": 0.141619, "outside_alpha_mass": 0.001368, "floater_fraction": 0.001077, "interior_alpha": 0.998893, "hull_rejected_fraction": 0.001171},
    "match_result_json_tolerance": 1e-12,
    "in_sample_metrics": ["outside_alpha_mass", "floater_fraction", "interior_alpha", "hull_rejected_fraction"],
    "novel_view_metrics": ["foreground_psnr", "crop_lpips"],
    "result_md_regenerated_byte_identical": true
  },
  "claim_dispositions": [
    {"id": "P1", "statement": "12 paired development cells completed.", "source": "RESULT.md/json, metrics.json", "disposition": "confirm", "basis": "12 receipts completed, 8000 iterations each, one invocation"},
    {"id": "P2", "statement": "Held-out foreground PSNR inside the mask: nb_ms 23.158, nb_ms_reloc 23.221, ph_ms 23.718, ph_ms_reloc 23.699 dB", "source": "RESULT.md/json", "disposition": "confirm", "narrowing": "in-task development numbers on one outcome-exposed frame; not comparable to the published RTGS-025 run (different decoder); no absolute floor"},
    {"id": "P3", "statement": "H1 all-mask floater relocation versus the masked objective alone: inconclusive", "source": "RESULT decision", "disposition": "confirm", "basis": "recomputed in written form: seed 9361 pass, seeds 9362 and 9363 neither pass nor reverse"},
    {"id": "P4", "statement": "No production model is trained; the frozen rule is not met.", "source": "RESULT decision", "disposition": "confirm", "basis": "no production/ directory; gate inconclusive; production() refuses without pass"},
    {"id": "P5", "statement": "Table rows for crop_lpips, outside_alpha_mass, floater_fraction, interior_alpha, hull_rejected_fraction", "source": "RESULT.md", "disposition": "confirm", "narrowing": "mechanism descriptors; alpha and hull rows in-sample, including interior_alpha whose label lacks the tag"},
    {"id": "P6", "statement": "Relocation lowered the final hull-rejected Gaussian fraction from about 5.5 percent to about 0.2 percent", "source": "task record handoff", "disposition": "confirm", "narrowing": "in-sample compliance descriptor of the relocation objective under the frozen hull; not a quality statement"},
    {"id": "P7", "statement": "in-sample outside-mask alpha by about a quarter", "source": "task record handoff", "disposition": "confirm", "narrowing": "0.001782 to 0.001314 (-26 percent), lower in all six field view pairs; in-sample only"},
    {"id": "P8", "statement": "the first event moved about 19400 of 20000 start points", "source": "task record handoff", "disposition": "confirm", "qualification": "19393 to 19406 per cell; 19149 to 19297 of the 20000 initial Gaussians were pruned at the first gsplat refinement in every cell, so the hull-guided placement is mostly transient"},
    {"id": "P9", "statement": "seed 9361 passes at +0.228 dB; seeds 9362/9363 at -0.016 and -0.023 dB", "source": "task record handoff", "disposition": "confirm", "basis": "recomputed; the seed-9361 gain is carried by C0001 (+0.41 dB), C0029 changes by less than 0.07 dB in every seed"},
    {"id": "P10", "statement": "the only design change is exact CPU-index decoding", "source": "claim boundary", "disposition": "confirm", "basis": "20-line task diff of identifiers, paths, wording, disclosure and aggregate; backend index at every decode; parity gate index-versus-reference only"},
    {"id": "P11", "statement": "Run locked at 70e4bc6 (clean, non-development); one run invocation, exit 0, 12/12 cells; GPU shared", "source": "task record handoff", "disposition": "confirm", "basis": "lock, receipts, launch context, run root listing"},
    {"id": "P12", "statement": "Descriptive photograph relocation pair and field-versus-photograph pairs", "source": "RESULT.json gates", "disposition": "confirm", "narrowing": "descriptive only; the photograph pair's two views disagree in sign in every seed; no direction claim"},
    {"id": "P13", "statement": "Numeric gates precede the independent audit and do not assert visual adequacy.", "source": "RESULT.md", "disposition": "confirm", "basis": "visual disposition given in this audit with its resolution limit"},
    {"id": "P14", "statement": "Relocation improves novel-view colour or reduces novel-view floaters", "source": "hypothesis (not asserted by the producer)", "disposition": "unresolved", "basis": "H1 inconclusive; floater metrics in-sample; needs a hull-initialized no-relocation arm, hull masks without held-out silhouettes, more frames and seeds or a longer budget"}
  ],
  "visual_disposition": {
    "status": "adequate_at_contact_sheet_scale",
    "sheets_viewed": 13,
    "root_preview_equals_selected_cell": true,
    "reconstruction_failures": 0,
    "observations": [
      "all 12 final renders reproduce silhouette, pose and colours of both held-out views",
      "bright silhouette-rim error in every arm; interior error denser for C0001 than C0029",
      "a few isolated background specks in the no-relocation field arms, fewer in the relocation arms",
      "no halo, ghost, detached fragment or missing body part in any arm"
    ],
    "limits": [
      "sheets rendered at 2000x225 from 5328x600; native-resolution crops denied by the sandbox and not retried",
      "individual renders excluded by the payload policy",
      "floater and halo differences between arms not visually decisive; quantitative basis is the in-sample alpha descriptors",
      "browser viewer smoke still pending"
    ]
  },
  "disclosures_carried": [
    "frame 00008 is outcome-exposed (RTGS-021/024/025)",
    "held-out alpha, floater, hull-rejection and interior-alpha metrics are in-sample",
    "the first relocation event re-initializes most random start points onto the hull (19393 to 19406 of 20000) and about 96 percent of them are pruned at the first gsplat refinement in every arm",
    "gsplat refinement after each event uses screen-gradient statistics accumulated partly at pre-relocation positions",
    "the installed gsplat 1.5.3 never executes its opacity reset, also for the RTGS-025 baseline; confirmed on loss and density traces",
    "heldout_mask_alpha is a declared reconstruction input",
    "first pre-review smoke (60 iterations, training stand-ins) exposed relocation counts, hull size and a meaningless gate value; no threshold changed",
    "second pre-review smoke (prepare plus initialize over the frozen split) exposed per-view index parity, the hull voxel count, per-view teacher PSNR and initial hull-rejected fractions; no threshold changed",
    "predecessor task failed closed at its CUDA parity gate; receipts at ara/evidence/tables/20260927_silhouette_relocation_failed_run/",
    "targets, held-out diagnostic and production inputs are CPU-index decoded; nb_ms is an in-task re-fit not comparable to the published RTGS-025 run",
    "RESULT.md lacks four of these disclosures (post-event statistics, inert reset, second smoke, RTGS-025 non-comparability); EXPERIMENTS.md and ARA rows must repeat them"
  ],
  "limitations": [
    "development screen on one outcome-exposed frame, one split, two held-out colour views, three paired seeds, one 8000-step budget at downscale 8",
    "H1 inconclusive; no production model; no default, SOTA, generalization, physical-geometry, speed or VRAM claim",
    "alpha and hull descriptors are in-sample; held-out colour is novel under in-sample silhouettes in the relocation arms",
    "GPU contended at launch; all timing and memory numbers descriptive only",
    "field relocation arms ended with 2.3, 1.9 and 3.6 percent more Gaussians than their baselines (capacity confound not controlled)",
    "visual inspection at contact-sheet scale only",
    "source-binding aggregate not reproduced by the auditor",
    "held-out compact views are parsed whole in prepare; alpha-only use is code discipline, not hook-enforced",
    "interior_alpha label lacks the in-sample tag"
  ],
  "commands_executed_by_reviewer": [
    "git status --short; git rev-parse HEAD; git log --oneline -8",
    "git log --format='%h %cI %s' -4; git diff --stat HEAD; git diff HEAD -- .agents/state/current-task.md",
    "git diff --no-index [--stat] experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json",
    ".venv/bin/python scripts/experiment_contract.py validate",
    ".venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json",
    ".venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json",
    ".venv/bin/python -m pytest -q tests/test_silhouette_relocation.py tests/test_silhouette_relocation_index_decode_protocol.py",
    ".venv/bin/python -c (payload manifest hashes versus disk and authorization)",
    ".venv/bin/python -c (source_snapshot/manifest.json versus live tree; task, review and seal bytes versus task.lock.json)",
    ".venv/bin/python -c (per-cell receipt, effective_config, history and evaluation summary)",
    ".venv/bin/python -c (group means, paired deltas, H1 verdict and descriptive pairs from per-view rows)",
    ".venv/bin/python -c (RESULT.md regeneration from comparison.json; metrics.json and training_history.json checks)",
    ".venv/bin/python -c (relocation events versus checkpoint counts; count trajectories; density statistics; loss windows at 3000 and 6000)",
    ".venv/bin/python -c (split digest and run wall time)",
    "rg and sed over trainer.py, strategies.py, gsplat_backend.py, visualize.py, experiment_contract.py, installed gsplat strategy/default.py, docs/, ara/, experiments/README.md"
  ],
  "sandbox_denied_not_worked_around": [
    "compound contract and pytest invocations (re-run plainly)",
    "combined snapshot-and-archived-copies hash script (re-run without the archived copies)",
    "PIL crop of native-resolution contact-sheet panels into the session scratchpad"
  ],
  "not_executed": [
    "init-run",
    "driver run / prepare / initialize / fit / evaluate / production / selftest",
    "render, check-run, scripts/check_results_bundle.py, viewer",
    "scripts/verify.sh and the full test suite",
    "any GPU work",
    "reproduction of the source-binding aggregate",
    "reading of photographs, masks, .rtgsv, .npz, .ply, targets/, renders/, GIFs, logs, gaussians.config.json, archived source copies, the predecessor run root, or the RTGS-025 RESULT/AUDIT records",
    "any file creation, modification or deletion under the repository or the run root"
  ],
  "remaining_handoff_checks": [
    "persist AUDIT.md and AUDIT.json verbatim under benchmarks/results/ and commit them with the unchanged RESULT.md and RESULT.json",
    "run render once, serve the report, perform the browser viewer smoke with the frozen viewer command, write viewer_smoke.json, rerun render",
    "pass check-run and scripts/check_results_bundle.py on the run root",
    "append the dated outcome to docs/EXPERIMENTS.md carrying all disclosures and corrections listed here",
    "add ARA observations for the in-sample descriptors and the first-refinement prune finding, and a hypothesis row for H1 bound to RESULT and AUDIT; update docs/ROADMAP.md status if RTGS-026 is listed",
    "close the task record with this verdict under the Reviewer label; move Maturity Reached to Calibrated only after the viewer receipt and both gates pass; archive to docs/tasks/ on closeout",
    "production sequence not triggered (gate inconclusive); no production invocation may run"
  ]
}
```
