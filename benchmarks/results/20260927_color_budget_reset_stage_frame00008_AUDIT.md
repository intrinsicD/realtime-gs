# Independent Results Audit

- Task ID: `20260927_color_budget_reset_stage_frame00008` (task record RTGS-028)
- Reviewer: `Claude-Code-Fable-5.1-reviewer` (model `claude-fable-5-1`); the same label approved the prospective protocol, this is the separate post-run results audit
- Self-reviewed: no (Driver `Claude-Code-Opus-5.5-driver`)
- Protocol SHA-256: `7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9`
- Outcome access: full post-run results audit
- Audit date: 2026-09-28
- Verdict: `accepted_with_limits`

## Verdict in one paragraph

The run executed the approved protocol exactly once, from the approved commit, on a clean tree, with every held-out photograph, mask and compact view closed until `evaluate`. All 18 cells completed their frozen budget from the digest-matched shared initialization with the frozen configuration. Recomputing every group mean, paired delta and verdict from the 72 per-view rows reproduces the producer's `comparison.json` and `RESULT.json` to 1e-12: H1 (30000-step budget) is **rejected** under the written rule, with every seed and every held-out view losing colour, and H2 (intended opacity reset at 30000 steps) is **inconclusive**, with all three seeds inside the ±0.1 dB dead zone. The intended reset fired on exactly the frozen gsplat iterations and never in the upstream arms; no cell reached the 100000-Gaussian cap. All 18 previews are visually adequate. The limits: the RESULT note's frozen boundary text omits the means-learning-rate horizon that condition 5 required, so this audit carries that disclosure and the Driver must propagate it; the Hard Rule 7 bundle (page, manifest, page and viewer receipts, bundle-checker pass) does not exist yet, so the run is not results-bearing until it does; and the result is a development-level negative on one exposed frame whose mechanism is not identified.

## Scope, inputs and what was not opened

Read in full: `CLAUDE.md`; the results-audit skill; the task JSON; the approved protocol review (residual conditions 1-8); `docs/tasks/RTGS-027-gsplat-opacity-reset.md`; `.agents/state/current-task.md` (treated as claims); the driver and report modules; `IntendedOpacityReset` and `enforce_budget` in `src/rtgs/optim/strategies.py`; the two bound test files; `RESULT.md` and `RESULT.json` (treated as claims); and every record in the payload manifest: lock, run receipt, environment, preflight, launch context, entry/exit integrity, preparation, initialization, comparison, metrics, training history, resource and input-boundary receipts, the source-snapshot manifest, and for all 18 cells the receipt, evaluation, effective configuration and history. Read in part: `src/rtgs/optim/trainer.py` (callback seam, means-LR decay, loop head, lines 305-330, 486-491, 656-658, 878, 904-909, 966-981) and `scripts/experiment_contract.py` (`build_source_binding` 1022-1052, `_development_source_state` 1115-1158, `init_run` 1161-1216). Viewed all 18 cell `reconstruction_contact_sheet.png` previews; the root preview is byte-identical to `cells/nb_30k_rs/9461/reconstruction_contact_sheet.png` (SHA-256 `d886241e…`), so it was assessed by identity.

The payload manifest hashes to `f473cc3d437747e5a1aec553d9801b26f76ea71a6b1dd976474fb7ec901219b0`, equal to the authorization record. All 111 manifest items (92 records, 19 previews) exist on disk with matching SHA-256.

Not opened: raw photographs, masks, `.rtgsv`, `.npz`, `.ply`, `runs/**/targets/`, `runs/**/renders/`, GIFs, `logs/`, `source_snapshot/git-status.txt` and `tracked.diff`, `gaussians.config.json` (not in the manifest), the `.scratch` smoke directory beyond the two authorization files, and the RTGS-025/026 RESULT and AUDIT records. CUDA is not visible in this session; no GPU work was done.

## 1. Chronology and single invocation

| Event | Time (UTC) | Source |
|---|---|---|
| Draft protocol committed (`a496016`) | 2026-09-27 20:19:50 | `git log` |
| Prospective approval recorded (`1428175`, HEAD) | 2026-09-27 20:52:49 | `git log` |
| Driver launch note (GPU free after pid 4057089 exited) | 2026-09-27 20:53:41 | `launch_context.json` (Driver-written note, unverifiable post hoc) |
| `init-run` lock written, `source_commit` `1428175`, `source_dirty` false | 2026-09-27 20:53:42.073 | `task.lock.json` |
| `prepare` | +3.09 s to +58.47 s | `preparation.json` |
| `initialize` | +59.42 s to +61.10 s | `initialization.json` |
| 18 `fit` cells, frozen order, strictly sequential and disjoint | +62.55 s to +3791.78 s | cell receipts |
| `evaluate` (all 18 cells) | +3803.36 s to +3844.91 s | cell evaluations |
| Run receipt `completed`, exit 0 | 2026-09-27 21:57:48.603 (+3846.5 s) | `run_receipt.json` |

Findings: the lock was created 53 s after the approval commit and its `started_at_utc` equals the run receipt's. Every cell receipt's `wall_seconds` equals its fit interval to 1e-6 s, and each fit interval starts after the previous one ends, in the frozen `execution_order`. The four `evaluate` opens of held-out data begin after the last fit ended. No `execution_failure.json` exists; `runs/` contains exactly one root for this task id among 18 roots (no `_v2`, `_final` or `latest` sibling). The coordinator's re-entry refusal is in the bound driver and test-pinned. The prepare and initialize intervals are identical in all 18 receipts, as the report note says. Condition 6 (one invocation) holds; condition 3 (runtime) was met with a wide margin: 8000-step cells took 77.3-85.1 s and 30000-step cells 301.0-365.2 s against the 3600 s limit. The task record's pre-start claims (prior 8000-step walls 142-187 s; waiting for the unrelated GPU process) cannot be verified retroactively and are not used as evidence; the observed walls are consistent with them. Timings remain descriptive.

## 2. Source, protocol and data binding

- `review-digest` prints `7a55197b…`, equal to the lock, the task's `protocol_review`, and the approved review; `validate` and `validate-data` print OK.
- The lock's `task_sha256` (`8f7c6fc3…`), `protocol_review_artifact_sha256` (`331b1db6…`) and `data_seal_sha256` (`d01254e7…`) equal the live files. The task JSON differs from the reviewed `53768705…` only by the recorded approval and `status: ready` (condition 1), which the digest excludes.
- The lock's `source_diff_sha256` (`448d83d6…`) equals SHA-256 of an empty tracked diff plus an empty untracked manifest (`b"\nRTGS-UNTRACKED-SOURCE-MANIFEST\0[]"`), so at `init-run` the tree was clean including untracked files (condition 2). The snapshot manifest records `git_diff_sha256` equal to SHA-256 of the empty string.
- All 122 snapshot entries (120 bound source files plus seal and review) match the live tree byte for byte. Replicating `build_source_binding` by plain hashing over the frozen patterns gives 120 files and aggregate `28b7fb80b74d1b520e9ec176e0e9161d6c81ec699868340c43f6736158486af5`, equal to the frozen value. `git rev-parse HEAD:<path>` reproduces the object ids in the approved review (`src/rtgs` tree `1cebc4bc…`, `strategies.py` `8c4e99a2…`, `trainer.py` `81f88540…`, driver `fc265e6c…`, report `24177606…`). `git diff --stat 1428175 HEAD` is empty; the only untracked files are the two RESULT records.
- Data: 81 unique sealed paths (81 seal entries; the 28 field entries are all duplicated inside the seal with identical digests), including the held-out photographs, masks and compact views; entry and exit receipts report 81 files, `all_match` true. The exit `split_sha256` (`150a3659…`) equals SHA-256 of the task's `splits` JSON.
- Environment: RTX 3050, torch 2.9.0, CUDA 12.8, gsplat 1.5.3, lpips 0.1.4, Python 3.12.9 (`environment.json`, `preflight.json`). LPIPS AlexNet weights are unbound, as disclosed prospectively.
- Bound CPU tests: 15 collected, 14 passed, 1 skipped (the CUDA seam test). The canary test asserts that the installed gsplat still carries the inert expression `step % self.reset_every == 0 & step > 0`, which is the basis of the "upstream (inert) reset" control (condition 4, CPU form; the Driver's GPU canary receipt is unverified by me).

## 3. Input boundaries per phase

- `prepare`: 110 dataset opens, all of them the 22 training views' `rgb/*.jpg`, `mask/mask_*.png` and `no_boundary/*.rtgsv`; no held-out file; no denials. Decoder parity index-versus-reference max 1.79e-7 (gate 1e-5) on all 22 views; packed-alpha equality and fit-window checks passed (the phase would have aborted otherwise). Teacher mean foreground PSNR on training views: 30.79 dB (decoded field versus photograph), a preparation quantity.
- `initialize`: zero dataset opens; three 20000-point seeds with digests `abfca731…` (9461), `973cbbd3…` (9462), `6eebd595…` (9463).
- `fit`: zero dataset opens and zero denials in all 18 receipts; `sampled_train_views` indices lie in 0..21 for every iteration; `history["psnr"]` is empty (no held-out evaluation during training; `internal_checkpoint_evaluation` false, `checkpoint_policy` final); `stop_reason` `max_iterations`. No held-out selection anywhere.
- `evaluate`: runs only after all 18 receipts read `completed` (driver 664-667); its guard lists exactly the four held-out views' photograph, mask and compact view; no denials.

Held-out masks and colour therefore never entered preparation, initialization or fitting; the alpha, floater and coverage descriptors are out-of-sample in the sense the protocol defines (novel-view silhouettes), while training views were supervised with mask-derived packed alpha.

## 4. Cells: completion, initialization, configuration, reset events, cap

For all 18 cells: `status` completed; receipt condition/seed match the directory; `initial_sha256` equals the seed's initialization digest; `effective_config.train_config` equals the frozen `resolved_training_configs[budget][seed]` by dict equality; `target_family`, `budget` and `intended_opacity_reset` match the driver's condition table; `train_view_ids` equal the frozen 22 training views; scene bounds equal the initialization bounds; `executed_iterations` equals `iterations` (8000 or 30000) and the receipt; `loss` has exactly that many finite entries; `density_strategy` gsplat-default with `dynamic` storage.

| Cell | Seed | Final n | Max live n | Cap pruning | Reset gsplat iterations (completed steps) | Wall s | Peak MiB | Held-out fg PSNR | Crop LPIPS | Outside alpha |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| nb_8k_up | 9461 | 43504 | 43504 | 0 | none | 79.8 | 139 | 23.4806 | 0.14422 | 0.001147 |
| nb_8k_rs | 9461 | 37410 | 37410 | 0 | 3000 (3001) | 77.3 | 130 | 23.4881 | 0.14531 | 0.001082 |
| nb_30k_up | 9461 | 73969 | 73969 | 0 | none | 360.3 | 189 | 22.7455 | 0.16663 | 0.000919 |
| nb_30k_rs | 9461 | 35733 | 53595 | 0 | 3000, 6000, 9000, 12000 (3001, 6001, 9001, 12001) | 301.0 | 149 | 22.7148 | 0.16608 | 0.000942 |
| ph_8k_up | 9461 | 47004 | 47004 | 0 | none | 79.0 | 143 | 23.8041 | 0.12975 | 0.001095 |
| ph_30k_rs | 9461 | 39203 | 59293 | 0 | 3000, 6000, 9000, 12000 | 302.7 | 159 | 23.2337 | 0.14067 | 0.000866 |
| nb_30k_up | 9462 | 73039 | 73039 | 0 | none | 365.2 | 188 | 22.7767 | 0.16585 | 0.000957 |
| nb_30k_rs | 9462 | 37282 | 51263 | 0 | 3000, 6000, 9000, 12000 | 307.2 | 146 | 22.7298 | 0.16511 | 0.000950 |
| ph_8k_up | 9462 | 46639 | 46639 | 0 | none | 81.3 | 141 | 23.9570 | 0.12915 | 0.001121 |
| ph_30k_rs | 9462 | 42144 | 57337 | 0 | 3000, 6000, 9000, 12000 | 333.0 | 155 | 23.2285 | 0.14096 | 0.000917 |
| nb_8k_up | 9462 | 43303 | 43303 | 0 | none | 85.1 | 139 | 23.5256 | 0.14462 | 0.001162 |
| nb_8k_rs | 9462 | 35677 | 35677 | 0 | 3000 (3001) | 77.3 | 127 | 23.5390 | 0.14783 | 0.001151 |
| ph_8k_up | 9463 | 48023 | 48023 | 0 | none | 83.4 | 143 | 23.8740 | 0.12884 | 0.001071 |
| ph_30k_rs | 9463 | 41178 | 57684 | 0 | 3000, 6000, 9000, 12000 | 320.0 | 157 | 23.3205 | 0.13902 | 0.000860 |
| nb_8k_up | 9463 | 44433 | 44433 | 0 | none | 80.5 | 138 | 23.4600 | 0.14736 | 0.001096 |
| nb_8k_rs | 9463 | 36512 | 36512 | 0 | 3000 (3001) | 83.8 | 129 | 23.6303 | 0.14386 | 0.001074 |
| nb_30k_up | 9463 | 72118 | 72118 | 0 | none | 358.5 | 187 | 22.7836 | 0.16402 | 0.000905 |
| nb_30k_rs | 9463 | 37244 | 52609 | 0 | 3000, 6000, 9000, 12000 | 323.9 | 147 | 22.7492 | 0.16887 | 0.000934 |

Reset mechanism: `opacity_reset_events` in every `_rs` cell equals exactly the frozen set `0 < g < stop_iter`, `g % 3000 == 0` (8k: `g = 3000`; 30k: `g = 3000, 6000, 9000, 12000`, with 15000 excluded by `stop_iter`), recorded at completed steps `g + 1` as the trainer's clock implies (trainer 657-658, strategies 401-404). Every `_up` cell records an empty list. The clamped counts show that nearly every Gaussian was above the cap at each event (for example 35747 of 36493 at `g = 3000` in nb_30k_rs/9461). The reset value is `2 * prune_opacity = 0.01` from `IntendedOpacityReset.from_density` (strategies 394-399; test-pinned), so the cap is `logit(0.01)`; the frozen `opacity_reset_value: 0.011` is a classic-controller field unused on this path.

Cap: `density_stats.pruned_to_budget` sums to 0 in all 18 cells; the largest live count in any history is 73969 (nb_30k_up/9461), below the 100000 cap. Least-significant-row pruning was never an active mechanism in any arm, so the cap confound raised in the prospective review did not materialize.

Timing and memory are descriptive (`performance_inference` false). Peak allocated memory is about 187-189 MiB in the 30000-step upstream arms and 146-149 MiB in the 30000-step reset arms, following the count.

## 5. Independent recomputation of the gates

Recomputed from the 72 per-view rows (18 cells times four held-out views `C0001, C0018, C0029, C1002`, in the frozen order in every cell). Per-cell means equal the saved `evaluation.mean` to 1e-10; three-seed group means, both verdicts and all descriptive deltas equal `comparison.json` and `RESULT.json` to 1e-12. The RESULT summary string reproduces from the recomputed groups. Mask-derived invariants (foreground, outside and interior pixel counts and the LPIPS crop box) are identical across all 18 cells for each view (C0001: 27927/350419/22941, crop (17,237,470,366); C0018: 20427/359379/16487; C0029: 30518/348081/25742; C1002: 18830/360498/14804).

Group means (three seeds of four-view means):

| Condition | Foreground PSNR dB | Crop LPIPS | Outside alpha mass | Floater fraction | Interior alpha |
|---|---:|---:|---:|---:|---:|
| nb_8k_up | 23.488748 | 0.145397 | 0.001135 | 0.000834 | 0.999002 |
| nb_8k_rs | 23.552471 | 0.145665 | 0.001102 | 0.000849 | 0.999054 |
| nb_30k_up | 22.768617 | 0.165498 | 0.000927 | 0.000789 | 0.999061 |
| nb_30k_rs | 22.731253 | 0.166686 | 0.000942 | 0.000817 | 0.998985 |
| ph_8k_up | 23.878357 | 0.129246 | 0.001095 | 0.000845 | 0.999033 |
| ph_30k_rs | 23.260886 | 0.140215 | 0.000881 | 0.000785 | 0.998997 |

H1, `nb_30k_up` versus `nb_8k_up` (pass iff every seed PSNR ≥ +0.2 dB and LPIPS ≤ +0.005; reject iff every seed PSNR ≤ −0.2 dB):

| Seed | ΔPSNR dB | ΔLPIPS | ΔOutside alpha | Pass | Reverse |
|---|---:|---:|---:|---|---|
| 9461 | −0.735 | +0.0224 | −0.000228 | no | yes |
| 9462 | −0.749 | +0.0212 | −0.000205 | no | yes |
| 9463 | −0.676 | +0.0167 | −0.000190 | no | yes |

Verdict **reject**, as written. Per-view PSNR deltas are negative in all 12 view-seed pairs (range −0.381 to −1.231 dB; C0029 loses the most, −1.03 to −1.23 dB). Crop LPIPS also worsened in every seed, beyond the +0.005 margin. Outside-mask alpha was slightly lower at 30000 steps.

H2, `nb_30k_rs` versus `nb_30k_up` (pass iff every seed PSNR ≥ +0.1 dB, LPIPS ≤ +0.005 and outside alpha ≤ +0.005; reject iff every seed PSNR ≤ −0.1 dB):

| Seed | ΔPSNR dB | ΔLPIPS | ΔOutside alpha | Pass | Reverse |
|---|---:|---:|---:|---|---|
| 9461 | −0.031 | −0.00055 | +0.000023 | no | no |
| 9462 | −0.047 | −0.00075 | −0.000007 | no | no |
| 9463 | −0.034 | +0.00486 | +0.000029 | no | no |

Verdict **inconclusive**, as written. All three PSNR deltas are small and negative but inside the ±0.1 dB dead zone; per-view deltas are mixed (−0.191 to +0.094 dB). LPIPS and outside-alpha margins were respected (seed 9463's LPIPS delta of +0.00486 is just inside +0.005). The reset roughly halved the final count (72-74k to 36-37k) with no measurable colour or floater change.

Descriptive pairs (no verdicts):

| Pair | ΔPSNR dB by seed (9461, 9462, 9463) | ΔLPIPS by seed |
|---|---|---|
| Reset at 8000 steps, `nb_8k_rs` − `nb_8k_up` | +0.008, +0.013, +0.170 | +0.0011, +0.0032, −0.0035 |
| Fields − photographs at 8000 steps, `nb_8k_up` − `ph_8k_up` | −0.323, −0.431, −0.414 | +0.0145, +0.0155, +0.0185 |
| Fields − photographs at 30000 steps with reset, `nb_30k_rs` − `ph_30k_rs` | −0.519, −0.499, −0.571 | +0.0254, +0.0241, +0.0299 |
| Photographs 30000 with reset − 8000, `ph_30k_rs` − `ph_8k_up` | −0.570, −0.729, −0.554 | +0.0109, +0.0118, +0.0102 |

The photograph pair couples budget and reset (no `ph_30k_up` arm exists), so its −0.55 to −0.73 dB cannot be attributed to the budget alone by this run, although the reset's near-zero effect in the field arms at 30000 steps makes the budget the plausible driver (hypothesis). Its per-view deltas are negative in all 12 pairs. The reset-at-8k pair is dominated by seed 9463 (+0.170 dB); the other two seeds are within 0.02 dB.

## 6. Metric semantics

Foreground PSNR is the masked MSE of the black-background render against the held-out photograph inside the fixed held-out binary mask, converted per view and averaged over the four views; crop LPIPS is AlexNet LPIPS on mask-multiplied render and photograph cropped to the padded mask box; outside-alpha mass, floater fraction and interior coverage use the held-out mask dilated or eroded by 3 output pixels (driver 161-198). Held-out masks are produced by the four-site quadrature of the lossless mask binarized at 0.5 and never entered fitting, so the alpha descriptors are out-of-sample novel-view silhouette agreement. Interior coverage is saturated (0.9989-0.9991 everywhere) and does not discriminate. Floater fractions are below 0.003 in the hardest view (C0001) and below 0.0003 in the other three views in every cell; floaters are rare in all arms. The gsplat rasterizer scored all cells; the diagnostic field-consistency PSNR (render versus decoded held-out field) also fell with the 30000-step budget (for example C0029, seed 9461: 27.19 to 25.88 dB), so the longer budget moved the models away from the held-out fields as well as from the photographs.

## 7. Visual adequacy and floaters

All 18 contact sheets show, for each held-out view, the masked reference photograph, the initial 20000-point gray ball, the final render and a four-times error map. In every cell the final render is a complete silhouette of the reclining figure with plausible colour and no missing limbs, halos, or gross geometric failures. A handful of faint isolated specks outside the silhouette are visible in several 8000-step cells (for example ph_8k_up/9462, nb_8k_up/9462, ph_8k_up/9463 in the C0001 and C0029 panels); the 30000-step cells look slightly cleaner outside the silhouette, consistent with their lower outside-alpha mass (about 0.0009 versus 0.0011). Error maps are dominated by silhouette rims and fabric-texture residual; the leftmost hem region of C0001 carries a magenta and white rim in every cell of every arm, so it is a shared boundary error, not a treatment effect. The 30000-step error maps show visibly brighter interior residual in C0001 and C0029 than the 8000-step maps, consistent with the 0.7 dB loss. The root preview is the prospectively selected nb_30k_rs/9461 model. Visual disposition: adequate for a development screen in all 18 cells; no cell's metrics are invalidated by a visual failure. Differences between arms are subtle at contact-sheet scale and the WebGL viewer remains a diagnostic.

## 8. Required disclosures (condition 5) and conditions 1-8

- Frame 00008 is outcome-exposed (RTGS-021/024/025/026): stated in the RESULT boundary. Confirmed.
- H1 as a package: the RESULT boundary names the densification window (stop 15000 versus 6000) but not the means-learning-rate horizon, and the metrics note "scale only iterations and the densification stop" is silent on the derived schedule. Disclosure carried here: `means_gamma = 0.01 ** (1 / iterations)` gives 0.9994245 (8k) and 0.9998465 (30k), verified in each cell's `history.json`; the 30000-step arm therefore keeps 29.3 percent of the initial means learning rate at step 8000, 10 percent when densification stops at 15000 and 1 percent at 30000, whereas the 8000-step arm is at 1 percent at step 8000. H1 measures iterations, densification window and means-LR horizon together. The Driver must carry this into the task record, `docs/EXPERIMENTS.md` and the ARA row; the once-only RESULT files must not be edited.
- 100000 cap: never reached (pruned_to_budget 0 in all cells; max live 73969). Carried here.
- Reset events per `_rs` cell: listed in section 4. Carried here.
- Reset value 0.01 (`2 * prune_opacity`), not the inert `opacity_reset_value` 0.011: carried here.
- H2 floater clause: gated by outside-mask alpha mass; `floater_fraction` is descriptive. Carried here.
- Photograph budget pair couples budget and reset: carried here; the RESULT summary's "photographs 23.878 / 23.261 dB" must be read as `ph_8k_up` / `ph_30k_rs`.
- Pre-review smoke: stated in the RESULT boundary as the protocol froze it; not verifiable from tracked state; no threshold changed after it (the frozen rules are those approved before the run).
- Timings on a possibly shared GPU are descriptive: stated; the Driver's note that the GPU was free at launch is a claim, not a receipt.
- gsplat version: `preflight.json` and `environment.json` record 1.5.3, the basis of the inert upstream control; the CPU canary passes on this box.
- RTGS-025/026 numbers are not comparators in any gate: confirmed; every gate is a within-run paired comparison of the frozen seeds. The task record's "0.4-0.6 dB" is motivation only.
- Conditions 1 (recording), 2 (clean bound source at init-run), 3 (runtime margin), 4 (canary), 6 (one invocation), 7 (no late abort occurred) and 8 (nothing beyond the 18 cells was trained; the root model is nb_30k_rs/9461) are satisfied on the evidence above.

## 9. What the result may and may not support

May support, at development maturity on frame 00008, the RTGS-025 split, downscale 8, three paired seeds: (a) the 30000-step 3DGS convention package (iterations 30000, densification to 15000, slower means-LR decay) gives worse held-out colour inside the mask than the 8000-step RTGS-025/026 configuration for field-only distillation, by 0.68-0.75 dB per seed and in every view, with worse crop LPIPS; (b) the RTGS-027 intended opacity reset at 30000 steps has no colour effect at or above 0.1 dB and no floater cost, while halving the final Gaussian count; (c) the field-versus-photograph gap at 8000 steps is 0.32-0.43 dB and 0.50-0.57 dB at 30000 steps with the reset; (d) descriptively, photograph supervision also loses 0.55-0.73 dB under the 30000-step-with-reset package.

May not support: any default change; any generalization beyond this frame, split, resolution or scene; iteration count alone as the cause of the H1 loss (three factors are coupled and no 30000-step arm with densification stop 6000 or fixed LR horizon exists); an "overfitting" mechanism. On the last point, the records contain only training-objective evidence: for nb_30k_up/9461 the weighted L1 term fell from 0.00043 (steps 7500-8000) to 0.00023 (last 500 steps) and weighted DSSIM from 0.00115 to 0.00041 while held-out PSNR fell 0.7 dB, and the 8000-step arm ended at a training objective of 0.00184 versus 0.00075 for the 30000-step arm. This is consistent with a tighter fit to the 22 training targets at the expense of novel views, but `record_train_metrics` was false, so no training-view PSNR exists, and the objective includes alpha terms and random-background compositing; overfitting remains a hypothesis. Also unsupported: that the reset is "harmless" beyond the measured metrics, and any speed or VRAM claim (peak memory follows the count and is descriptive). The nb_30k_rs arm reaching the same held-out colour as nb_30k_up with half the Gaussians is a descriptive observation, not a claim.

## 10. Claim dispositions

| # | Producer statement (source) | Disposition | Basis |
|---|---|---|---|
| P1 | "18 paired development cells completed." (RESULT, metrics summary) | confirm | 18 completed receipts at the frozen iterations from digest-matched initializations; all evaluated. |
| P2 | Field-only held-out foreground PSNR 23.489 / 22.769 / 22.731 dB at 8000 / 30000 / 30000+reset; photographs 23.878 / 23.261 dB (RESULT summary) | confirm numbers, narrow wording | Reproduced to 1e-12 from per-view rows. Three-seed means of four-view means on one exposed frame; "23.261" is `ph_30k_rs`, which couples budget and reset. |
| P3 | "H1 longer budget (30000 vs 8000 steps, field-only): reject." (RESULT decision) | confirm verdict, narrow scope | Every seed ≤ −0.2 dB (−0.676 to −0.749) and every view negative; LPIPS worse in every seed. Refutes the 30000-step package on this frame, not iteration count alone; mechanism unidentified. |
| P4 | "H2 intended opacity reset at 30000 steps: inconclusive." (RESULT decision) | confirm | All seeds inside the ±0.1 dB dead zone (−0.031 to −0.047); LPIPS and outside-alpha margins met. Benefit untested; the reset halves the count with no measurable colour or floater change. |
| P5 | Group table of five metrics for six conditions (RESULT) | confirm | Reproduced to 1e-12. Interior alpha is saturated and non-discriminating. |
| P6 | Descriptive pairs (reset at 8k; fields vs photographs at 8k and at 30k+reset; photograph budget pair) (RESULT.json gates) | confirm numbers | Reproduced to 1e-12; no verdicts; the photograph pair couples budget and reset; the reset-at-8k mean is driven by seed 9463. |
| P7 | "The 30000-step budget also extends densification to step 15000 (the 3DGS convention)" (claim boundary) | narrow | True but incomplete: the means-LR decay horizon also changes (29.3 percent at step 8000, 10 percent at 15000, 1 percent at 30000 versus 1 percent at 8000). |
| P8 | "The 30000-step cells scale only iterations and the densification stop; see the task." (metrics note) | narrow | Same as P7; true of the explicit configuration, silent on the derived schedule. |
| P9 | "upstream gsplat 1.5.3 never resets" (claim boundary) | confirm | Preflight/environment 1.5.3; CPU canary passes; RTGS-027 evidence; `_up` cells record no events. |
| P10 | "held-out masks are evaluation-only, so alpha metrics are out-of-sample" (claim boundary, input-boundary receipt) | confirm | Guards and receipts show no held-out open before `evaluate`; training sampled only indices 0..21. |
| P11 | "Targets are decoded with the exact CPU tile index" (claim boundary) | confirm | `backend="index"` in the bound driver; parity max 1.79e-7 on all 22 views. |
| P12 | "Timings on a possibly shared local GPU are descriptive" (claim boundary); "No timing advantage is inferred" (metrics note) | confirm | `performance_inference` false; walls reported only descriptively here. |
| P13 | Pre-review smoke disclosure (claim boundary) | unresolved (carried as disclosed) | Not verifiable from tracked state; no threshold changed after it. |
| P14 | Task-record pre-start claims: GPU canary 9 tests with gsplat 1.5.3; prior 8k walls 142-187 s; run waited for pid 4057089 (current-task.md, launch_context.json) | unresolved (not used as evidence) | No GPU here; CPU canary reproduced; preflight version confirmed; observed walls consistent but timing is descriptive. |
| P15 | "Root preview is the prospectively selected nb_30k_rs/9461 model." (metrics note) | confirm | Root and cell contact sheets are byte-identical; root PLYs not opened, to be covered by the bundle checker. |
| P16 | "Visual adequacy and any promotion require the independent audit; no default changes." (RESULT decision) | confirm | Visual disposition: adequate in all 18 cells; no promotion beyond a development-level negative result. |

Recommended ledger handling (Driver): promote a staging observation (next id after O182) and a claim row whose Statement records H1 as a refuted development expectation with the per-seed deltas and H2 as untested/inconclusive with the per-seed deltas; Status `refuted development expectation (H1); untested reset benefit (H2 inconclusive)`; Proof the RESULT.json, this AUDIT.md and AUDIT.json; Boundary: one exposed frame, one split, three seeds, downscale 8, the budget package confound including the means-LR horizon, the coupled photograph pair, no training-view PSNR, no cap engagement, timings descriptive. C49 and C51 stay as they are (lineage only; their numbers were not comparators here). Append a dated `docs/EXPERIMENTS.md` entry with the same boundary.

## 11. Commands executed, denied, and not executed

Executed (read-only, or the allowed CPU tests):

```text
sha256sum .scratch/20260927_color_budget_reset_stage_frame00008/claude_results_audit/payload_manifest.json
.venv/bin/python -c <verify all 111 manifest item SHA-256 against disk>
git status --short; git log --oneline -8; git rev-parse HEAD
git log -4 --format='%h %cI %aI %s'; git rev-parse HEAD:<src/rtgs, strategies.py, trainer.py, driver, report, task JSON>
git diff --stat 1428175 HEAD; git status --porcelain --untracked-files=all
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_color_budget_reset_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_color_budget_reset_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_color_budget_reset_protocol.py tests/test_gsplat_opacity_reset.py -rs
.venv/bin/python -c <recompute per-cell means, group means, H1/H2 rules, descriptive pairs, per-view deltas, mask invariants, summary string from the 18 evaluation.json files>
.venv/bin/python -c <per-cell receipt/effective_config/history checks: status, init digest, frozen config equality, iterations, loss finiteness, reset events vs schedule, pruned_to_budget, max live count, sequential fit intervals, wall equality, loss means>
.venv/bin/python -c <snapshot manifest vs live tree, lock task/review/seal hashes, lock source_diff digest vs clean-tree constant, split digest>
.venv/bin/python -c <seal/field-input unique count and held-out coverage, means-LR gamma and factors, history means_lr_gamma, sampled_train_views range, held-out psnr entries, stop_reason>
.venv/bin/python -c <replicate build_source_binding by hashing the 120 pattern files; training_history record and marker counts; loss_terms decomposition>
Glob/Grep (names and text only): run-root file names, runs/*/task.lock.json, bundle file presence, RTGS-028 mentions in docs/ara, C49/C51 rows, O18x ids
```

Denied by the review sandbox and re-done in an allowed form, touching no protected material: one heredoc Python script (re-run as `python -c`) and one combined `python -c` that imported `scripts/experiment_contract.py` for `build_source_binding`, `verify_source_binding` and `validate_task` (replaced by plain-hashing scripts that replicate `build_source_binding` byte for byte and by the CLI `validate`/`validate-data`).

Not executed: `scripts/check_results_bundle.py`; `experiment_contract.py init-run`, `check-run`, `render` or any report rendering; the driver's `run`, `prepare`, `initialize`, `fit`, `evaluate` (the protocol test suite runs `selftest` and a guarded `coordinate` on a temporary root, as approved); any GPU work (CUDA not visible; the CUDA seam test skipped); `./scripts/verify.sh`, ruff, `docs_sync.py`, `check_ara.py`, `check_agent_workflow.py`; `rtgs view`; no file was modified, no run root or sibling created, no checkpoint selected, no RESULT record touched.

## 12. Remaining handoff checks (Driver)

1. Persist this audit verbatim as `benchmarks/results/20260927_color_budget_reset_stage_frame00008_AUDIT.md` and the JSON block as `…_AUDIT.json`; commit both together with the untracked RESULT files unchanged (`RESULT.md` SHA-256 `1ddef7f2722ba7de413fe54a9fb9e5ad4891a1dc2f8d5d848949b737d6c68a19`, `RESULT.json` `c2a8dca1602c79223799fe132fd6dbacff80266a204f7a23e5a7a4e70fbccd9f`).
2. Generate `index.html`, `README.md` and `manifest.json` from the shared metrics, history, configuration, environment and receipt schemas (none exists in the run root yet), record the page smoke receipt and the `rtgs view` receipt for the root `gaussians.ply` / `gaussians_init.ply` (nb_30k_rs/9461), then run `python scripts/check_results_bundle.py runs/20260927_color_budget_reset_stage_frame00008` and cite its pass. Until then the run is not results-bearing under Hard Rule 7.
3. Confirm the bundle checker's treatment of the Driver-written `launch_context.json` in the run root; it is provenance, not a receipt.
4. Carry the section 8 disclosures (means-LR horizon with the factors, cap never reached, reset events and value 0.01, floater clause via outside alpha mass, coupled photograph pair, gsplat 1.5.3 from preflight, descriptive timings, RTGS-025/026 not comparators) into the task record handoff, the `docs/EXPERIMENTS.md` entry and the ARA observation/claim; do not edit the once-only RESULT files or the frozen task.
5. Append the post-run Driver handoff and this verdict to `.agents/state/current-task.md` with distinct labels, then run `check_agent_workflow.py`, `check_ara.py`, `docs_sync.py` and `./scripts/verify.sh` before closing RTGS-028 at development maturity; no default changes.
6. For a follow-up that can resolve the mechanism: a 30000-step arm with densification stop 6000 and a fixed means-LR horizon, a `ph_30k_up` arm to decouple the photograph pair, and `record_train_metrics` enabled so training-view PSNR is on record.

```json
{
  "schema_version": 1,
  "task_id": "20260927_color_budget_reset_stage_frame00008",
  "task_record": "RTGS-028",
  "reviewer": "Claude-Code-Fable-5.1-reviewer",
  "model": "claude-fable-5-1",
  "self_reviewed": false,
  "verdict": "accepted_with_limits",
  "protocol_sha256": "7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9",
  "outcome_access": "full post-run results audit",
  "audited_at": "2026-09-28",
  "authorization": {
    "payload_manifest_sha256": "f473cc3d437747e5a1aec553d9801b26f76ea71a6b1dd976474fb7ec901219b0",
    "manifest_items": 111,
    "previews_viewed": 19,
    "all_item_hashes_match": true
  },
  "bindings": {
    "task_sha256": "8f7c6fc33ac1b098e473bedc0cc53ee441900476a23723e39f50cb0a6553b7c8",
    "protocol_review_artifact_sha256": "331b1db6b829ca36b41513de6d8c8de0a0a598798850902d8251e657f4736141",
    "data_seal_sha256": "d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087",
    "source_commit": "14281753c4544a0841236aa80ead3006e35e9b0e",
    "source_dirty": false,
    "lock_source_diff_sha256_equals_clean_tree_no_untracked": true,
    "snapshot_entries_verified_against_live_tree": 122,
    "source_binding_file_count": 120,
    "source_binding_aggregate_sha256": "28b7fb80b74d1b520e9ec176e0e9161d6c81ec699868340c43f6736158486af5",
    "sealed_inputs_verified": 81,
    "split_sha256": "150a365952b69296c9dab14a15787064b03e2415caef044e117d1d73306b2452",
    "review_digest_reproduced": true,
    "contract_validate": "OK",
    "contract_validate_data": "OK",
    "bound_cpu_tests": "15 collected, 14 passed, 1 skipped (CUDA)",
    "result_md_sha256": "1ddef7f2722ba7de413fe54a9fb9e5ad4891a1dc2f8d5d848949b737d6c68a19",
    "result_json_sha256": "c2a8dca1602c79223799fe132fd6dbacff80266a204f7a23e5a7a4e70fbccd9f"
  },
  "environment": {
    "gpu": "NVIDIA GeForce RTX 3050",
    "torch": "2.9.0",
    "cuda": "12.8",
    "gsplat": "1.5.3",
    "lpips": "0.1.4",
    "python": "3.12.9"
  },
  "chronology": {
    "approval_commit": "1428175 at 2026-09-27T20:52:49Z",
    "lock_started_at_utc": "2026-09-27T20:53:42.073180+00:00",
    "run_finished_at_utc": "2026-09-27T21:57:48.602962+00:00",
    "prepare_s": [3.086, 58.469],
    "initialize_s": [59.421, 61.098],
    "fit_cells_s": [62.553, 3791.776],
    "evaluate_s": [3803.365, 3844.914],
    "single_invocation": true,
    "execution_failure_present": false,
    "sibling_run_roots": 0,
    "fit_intervals_sequential_and_disjoint": true,
    "heldout_opens_before_evaluate": 0
  },
  "official_gates": {
    "recomputed_from": "72 per-view rows in cells/<condition>/<seed>/evaluation.json",
    "agreement_with_producer": "group means, per-seed deltas and verdicts equal comparison.json and RESULT.json to 1e-12",
    "group_means": {
      "nb_8k_up": {"foreground_psnr": 23.488748, "crop_lpips": 0.145397, "outside_alpha_mass": 0.001135, "floater_fraction": 0.000834, "interior_alpha": 0.999002},
      "nb_8k_rs": {"foreground_psnr": 23.552471, "crop_lpips": 0.145665, "outside_alpha_mass": 0.001102, "floater_fraction": 0.000849, "interior_alpha": 0.999054},
      "nb_30k_up": {"foreground_psnr": 22.768617, "crop_lpips": 0.165498, "outside_alpha_mass": 0.000927, "floater_fraction": 0.000789, "interior_alpha": 0.999061},
      "nb_30k_rs": {"foreground_psnr": 22.731253, "crop_lpips": 0.166686, "outside_alpha_mass": 0.000942, "floater_fraction": 0.000817, "interior_alpha": 0.998985},
      "ph_8k_up": {"foreground_psnr": 23.878357, "crop_lpips": 0.129246, "outside_alpha_mass": 0.001095, "floater_fraction": 0.000845, "interior_alpha": 0.999033},
      "ph_30k_rs": {"foreground_psnr": 23.260886, "crop_lpips": 0.140215, "outside_alpha_mass": 0.000881, "floater_fraction": 0.000785, "interior_alpha": 0.998997}
    },
    "h1_budget": {
      "treatment": "nb_30k_up",
      "control": "nb_8k_up",
      "rule": "pass iff every seed dPSNR >= 0.2 and dLPIPS <= 0.005; reject iff every seed dPSNR <= -0.2",
      "verdict": "reject",
      "rows": [
        {"seed": 9461, "d_foreground_psnr": -0.735061, "d_crop_lpips": 0.022410, "d_outside_alpha_mass": -0.000228, "pass": false, "reverse": true},
        {"seed": 9462, "d_foreground_psnr": -0.748869, "d_crop_lpips": 0.021235, "d_outside_alpha_mass": -0.000205, "pass": false, "reverse": true},
        {"seed": 9463, "d_foreground_psnr": -0.676461, "d_crop_lpips": 0.016658, "d_outside_alpha_mass": -0.000190, "pass": false, "reverse": true}
      ],
      "per_view_psnr_deltas_negative": "12 of 12",
      "per_view_psnr_delta_range": [-1.231, -0.381]
    },
    "h2_reset": {
      "treatment": "nb_30k_rs",
      "control": "nb_30k_up",
      "rule": "pass iff every seed dPSNR >= 0.1, dLPIPS <= 0.005 and dOutsideAlpha <= 0.005; reject iff every seed dPSNR <= -0.1",
      "verdict": "inconclusive",
      "rows": [
        {"seed": 9461, "d_foreground_psnr": -0.030763, "d_crop_lpips": -0.000548, "d_outside_alpha_mass": 0.000023, "pass": false, "reverse": false},
        {"seed": 9462, "d_foreground_psnr": -0.046946, "d_crop_lpips": -0.000746, "d_outside_alpha_mass": -0.000007, "pass": false, "reverse": false},
        {"seed": 9463, "d_foreground_psnr": -0.034384, "d_crop_lpips": 0.004859, "d_outside_alpha_mass": 0.000029, "pass": false, "reverse": false}
      ],
      "per_view_psnr_delta_range": [-0.191, 0.094]
    },
    "descriptive": {
      "reset_at_8k_descriptive": {"pair": ["nb_8k_rs", "nb_8k_up"], "d_foreground_psnr": [0.007533, 0.013394, 0.170242], "d_crop_lpips": [0.001092, 0.003214, -0.003500], "d_outside_alpha_mass": [-0.000066, -0.000010, -0.000022]},
      "field_vs_photographs_8k_descriptive": {"pair": ["nb_8k_up", "ph_8k_up"], "d_foreground_psnr": [-0.323472, -0.431396, -0.413960], "d_crop_lpips": [0.014471, 0.015466, 0.018515]},
      "field_vs_photographs_30k_reset_descriptive": {"pair": ["nb_30k_rs", "ph_30k_rs"], "d_foreground_psnr": [-0.518954, -0.498674, -0.571270], "d_crop_lpips": [0.025413, 0.024150, 0.029851]},
      "photographs_30k_reset_vs_8k_descriptive": {"pair": ["ph_30k_rs", "ph_8k_up"], "d_foreground_psnr": [-0.570342, -0.728536, -0.553535], "d_crop_lpips": [0.010920, 0.011806, 0.010182], "couples_budget_and_reset": true, "per_view_psnr_deltas_negative": "12 of 12"}
    },
    "cells": {
      "completed": 18,
      "frozen_iterations_reached": true,
      "initialization_digest_matched": true,
      "effective_config_equals_frozen": true,
      "training_sampled_only_train_indices_0_to_21": true,
      "heldout_evaluation_during_training": 0,
      "final_counts": {
        "nb_8k_up": [43504, 43303, 44433], "nb_8k_rs": [37410, 35677, 36512],
        "nb_30k_up": [73969, 73039, 72118], "nb_30k_rs": [35733, 37282, 37244],
        "ph_8k_up": [47004, 46639, 48023], "ph_30k_rs": [39203, 42144, 41178]
      },
      "wall_seconds_range_8k": [77.3, 85.1],
      "wall_seconds_range_30k": [301.0, 365.2],
      "timeout_s": 3600,
      "timings_descriptive": true
    },
    "cap_100000": {"reached_in_any_cell": false, "pruned_to_budget_total": 0, "max_live_count": 73969, "max_live_cell": "nb_30k_up/9461"},
    "reset_events": {
      "clock": "gsplat iteration g = completed_step - 1; events recorded at completed steps g + 1",
      "value": 0.01,
      "value_source": "2 * prune_opacity via IntendedOpacityReset.from_density; opacity_reset_value 0.011 unused on this path",
      "nb_8k_rs": [3000], "nb_30k_rs": [3000, 6000, 9000, 12000], "ph_30k_rs": [3000, 6000, 9000, 12000],
      "nb_8k_up": [], "nb_30k_up": [], "ph_8k_up": [],
      "matches_frozen_schedule_in_all_cells": true
    },
    "means_lr_horizon": {
      "means_gamma_8k": 0.9994245193792801,
      "means_gamma_30k": 0.9998465061085267,
      "factor_30k_at_8000": 0.293, "factor_30k_at_15000": 0.10, "factor_30k_at_30000": 0.01, "factor_8k_at_8000": 0.01,
      "disclosed_in_result_md": false,
      "disclosed_in_this_audit": true
    },
    "training_objective_descriptive": {
      "nb_8k_up_9461_final_loss": 0.00184,
      "nb_30k_up_9461_loss_at_7500_8000": 0.00172,
      "nb_30k_up_9461_final_loss": 0.00075,
      "note": "training objective fell while held-out PSNR fell; no training-view PSNR recorded; overfitting is a hypothesis"
    }
  },
  "claim_dispositions": [
    {"id": "P1", "statement": "18 paired development cells completed", "disposition": "confirm"},
    {"id": "P2", "statement": "Field-only 23.489/22.769/22.731 dB; photographs 23.878/23.261 dB", "disposition": "narrow", "note": "numbers confirmed; three-seed means on one exposed frame; 23.261 is ph_30k_rs (budget and reset coupled)"},
    {"id": "P3", "statement": "H1 longer budget: reject", "disposition": "narrow", "note": "verdict confirmed; refutes the 30000-step package (iterations, densification to 15000, means-LR horizon) on this frame, not iteration count alone; mechanism unidentified"},
    {"id": "P4", "statement": "H2 intended opacity reset at 30000 steps: inconclusive", "disposition": "confirm", "note": "all seeds inside +/-0.1 dB; margins met; count halved with no measurable colour or floater change"},
    {"id": "P5", "statement": "Group metric table", "disposition": "confirm", "note": "interior alpha saturated and non-discriminating"},
    {"id": "P6", "statement": "Descriptive paired deltas", "disposition": "confirm", "note": "no verdicts; photograph pair couples budget and reset; reset-at-8k mean driven by seed 9463"},
    {"id": "P7", "statement": "30000-step budget also extends densification to step 15000", "disposition": "narrow", "note": "also changes the means-LR decay horizon"},
    {"id": "P8", "statement": "30000-step cells scale only iterations and the densification stop", "disposition": "narrow", "note": "silent on the derived means-LR schedule"},
    {"id": "P9", "statement": "upstream gsplat 1.5.3 never resets", "disposition": "confirm"},
    {"id": "P10", "statement": "held-out masks are evaluation-only; alpha metrics out-of-sample", "disposition": "confirm"},
    {"id": "P11", "statement": "targets decoded with the exact CPU tile index", "disposition": "confirm"},
    {"id": "P12", "statement": "timings descriptive; no timing advantage inferred", "disposition": "confirm"},
    {"id": "P13", "statement": "pre-review non-protocol smoke disclosure", "disposition": "unresolved", "note": "not verifiable from tracked state; no threshold changed after it"},
    {"id": "P14", "statement": "task-record pre-start claims (GPU canary, prior walls, GPU free at launch)", "disposition": "unresolved", "note": "not used as evidence; CPU canary and preflight version reproduced"},
    {"id": "P15", "statement": "root preview is the prospectively selected nb_30k_rs/9461 model", "disposition": "confirm", "note": "contact sheets byte-identical; root PLYs left to the bundle checker"},
    {"id": "P16", "statement": "visual adequacy and any promotion require the independent audit; no default changes", "disposition": "confirm"}
  ],
  "visual_disposition": {
    "status": "adequate",
    "cells_viewed": 18,
    "root_preview_identity": "byte-identical to cells/nb_30k_rs/9461/reconstruction_contact_sheet.png",
    "floaters": "a few faint isolated specks outside the silhouette in several 8000-step cells; 30000-step cells slightly cleaner outside, consistent with lower outside-alpha mass",
    "halos": "none",
    "shared_artifact": "magenta/white rim at the C0001 hem region in every cell of every arm (not treatment-related)",
    "arm_differences": "subtle at contact-sheet scale; 30000-step error maps show more interior texture residual"
  },
  "limitations": [
    "Development screen on one previously outcome-exposed frame (00008), one split (22 train / 4 held-out), downscale 8, three paired seeds; no generalization, default, SOTA, speed or VRAM claim.",
    "H1 measures the 30000-step convention as a package: iterations, densification window (15000 vs 6000) and the derived means-LR horizon (29.3 percent at step 8000, 10 percent at 15000, 1 percent at 30000 vs 1 percent at 8000); the RESULT boundary text omits the LR horizon and cannot be edited, so this audit and the downstream records carry it.",
    "The photograph budget pair couples budget and reset (no ph_30k_up arm).",
    "No training-view PSNR was recorded; the overfitting interpretation of the budget effect rests only on the falling training objective and remains a hypothesis.",
    "H2 is inconclusive: any reset effect on colour is below the 0.1 dB floor in all three seeds; benefit untested, harm not shown.",
    "The Hard Rule 7 bundle (index.html, README.md, manifest.json, page smoke receipt, rtgs view receipt, check_results_bundle pass) does not exist yet; the run is not results-bearing until it does.",
    "Timings on a local desktop GPU are descriptive; the Driver's launch note that the GPU was free is a claim, not a receipt.",
    "LPIPS AlexNet weights and installed packages are recorded but not bound; the CUDA seam test and the Driver's GPU canary receipt were not reproduced here."
  ],
  "commands_executed_by_reviewer": [
    "sha256sum .scratch/20260927_color_budget_reset_stage_frame00008/claude_results_audit/payload_manifest.json",
    ".venv/bin/python -c <verify all 111 payload manifest item SHA-256 against disk>",
    "git status --short; git log --oneline -8; git rev-parse HEAD",
    "git log -4 --format='%h %cI %aI %s'; git rev-parse HEAD:<bound paths>; git diff --stat 1428175 HEAD; git status --porcelain --untracked-files=all",
    ".venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_color_budget_reset_stage_frame00008.json",
    ".venv/bin/python scripts/experiment_contract.py validate",
    ".venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_color_budget_reset_stage_frame00008.json",
    ".venv/bin/python -m pytest -q tests/test_color_budget_reset_protocol.py tests/test_gsplat_opacity_reset.py -rs",
    ".venv/bin/python -c <recompute per-cell and group means, H1/H2 rules, descriptive pairs, per-view deltas, mask invariants, summary string from the 18 evaluation.json files>",
    ".venv/bin/python -c <per-cell receipt, effective_config and history checks: status, initialization digest, frozen config equality, iterations, loss finiteness, reset events vs schedule, pruned_to_budget, max live count, sequential fit intervals, wall equality, loss means>",
    ".venv/bin/python -c <snapshot manifest vs live tree, lock task/review/seal hashes, lock source_diff digest vs clean-tree constant, split digest>",
    ".venv/bin/python -c <seal and field-input unique count, held-out coverage, means-LR gamma and factors, history means_lr_gamma, sampled_train_views range, held-out psnr entries, stop_reason>",
    ".venv/bin/python -c <replicate build_source_binding by hashing the 120 pattern files; training_history record and marker counts; loss_terms decomposition>",
    "Glob/Grep name and text listings: run root files, runs/*/task.lock.json, bundle file presence, RTGS-028 mentions in docs/ara, C49/C51 rows, O18x ids",
    "Read: CLAUDE.md, results-audit skill, task JSON, protocol review, RTGS-027 record, current-task.md, driver, report, strategies.py 300-440, trainer.py excerpts, experiment_contract.py 1022-1101 and 1100-1219, both bound tests, RESULT.md/RESULT.json, all manifest run and cell records, 18 cell contact sheets"
  ],
  "denied_and_redone_in_allowed_form": [
    "one heredoc Python script (re-run as python -c)",
    "one combined python -c importing scripts/experiment_contract.py for build_source_binding/verify_source_binding/validate_task (replaced by plain-hashing replication and the CLI validate/validate-data)"
  ],
  "not_executed": [
    "scripts/check_results_bundle.py",
    "experiment_contract.py init-run / check-run / render or any report page rendering",
    "driver run/prepare/initialize/fit/evaluate (protocol tests ran selftest and a guarded coordinate on a temporary root only)",
    "any GPU work (CUDA not visible; CUDA seam test skipped)",
    "./scripts/verify.sh, ruff, docs_sync.py, check_ara.py, check_agent_workflow.py",
    "rtgs view",
    "opening raw photographs, masks, .rtgsv, .npz, .ply, targets/, renders/, GIFs, logs/, source_snapshot/git-status.txt, tracked.diff, gaussians.config.json, the .scratch smoke directory, RTGS-025/026 RESULT and AUDIT records",
    "any file modification, run root creation, checkpoint selection or RESULT edit"
  ],
  "remaining_handoff_checks": [
    "Persist AUDIT.md and AUDIT.json verbatim; commit with the untracked RESULT files unchanged (hashes recorded in bindings).",
    "Generate index.html, README.md and manifest.json from the shared schemas; record the page smoke receipt and the rtgs view receipt for the root nb_30k_rs/9461 PLYs; run scripts/check_results_bundle.py on the run root and cite its pass.",
    "Confirm the bundle checker's treatment of the Driver-written launch_context.json (provenance note, not a receipt).",
    "Carry the condition-5 disclosures (means-LR horizon factors, cap never reached, reset events and value 0.01, floater clause via outside alpha mass, coupled photograph pair, gsplat 1.5.3 from preflight, descriptive timings, RTGS-025/026 not comparators) into the task record, docs/EXPERIMENTS.md and the ARA observation/claim; do not edit the RESULT files or the frozen task.",
    "Append the post-run handoff and this verdict to .agents/state/current-task.md; run check_agent_workflow.py, check_ara.py, docs_sync.py and ./scripts/verify.sh before closing RTGS-028 at development maturity; no default changes.",
    "Optional follow-up task to resolve the mechanism: 30000-step arm with densification stop 6000 and fixed means-LR horizon, a ph_30k_up arm, and record_train_metrics enabled."
  ]
}
```
