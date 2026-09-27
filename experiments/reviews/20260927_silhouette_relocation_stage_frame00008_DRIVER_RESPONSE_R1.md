# Driver response to the V1 prospective review

- Task ID: `20260927_silhouette_relocation_stage_frame00008`
- Rejected digest: `57370d132b561b48b5bc95b7f7c15a8bfd8b3b01c129e55d200f209ce084f76f`
- Revised digest: `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad`
- Driver: `Claude-Code-Opus-5.5-driver`
- Review record: `20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`
- Source binding: 120 files, aggregate `f7877b92c3655feb1efde420c6ca37e27d6abe2f711308ba1323c11a6bcf0e8f`

No hypothesis, threshold, seed, split, comparator, metric, budget or execution order changed.

## Required changes

- **B1.** Adopted option (b). `production.sequence` freezes: run completes, the independent
  results audit is persisted, the report is rendered with its viewer receipt, `check-run` and
  `check_results_bundle.py` pass, and only if `h1_relocation == pass` production runs once,
  followed by one rerender and both gates again. `production/` is never evidence and never enters
  RESULT or AUDIT. The report notes now restrict "held-out colour never enters fitting" to the
  twelve evidence cells and describe a possible `production/` directory; `publish` writes them
  unconditionally.
- **B2.** `coordinate` raises before writing anything when `targets/` or `preparation.json`
  exists; `test_coordinator_refuses_a_consumed_run_root` checks the error and that the root is
  unchanged. `decision_policy.stopping` states the refusal.

## Optional changes

- **R1.** Adopted as disclosure in `relocation.definition`: post-event gsplat refinement uses
  statistics partly accumulated at pre-relocation positions, and the installed gsplat 1.5.3 never
  executes its opacity reset (`step % self.reset_every == 0 & step > 0` evaluates as a chained
  comparison ending in `0 > 0`), so `opacity_reset_every` is inert in every arm and was in RTGS-025.
  The Driver reproduced the source line. A repository-wide correction is outside this task.
- **R2.** Adopted. `load_hull` passes `threshold`; `SilhouetteRelocationConfig` no longer carries
  the unused `dilation_px`/`near` fields (the hull owns them).
- **R3.** Adopted. `prepare` checks the recorded photograph and mask SHA-256 of every compact view
  used for the hull, including the two held-out views, against the seal.
- **R4.** Adopted in part: the nearest-target `cdist` chunk is 256 rows (about 21 MB per chunk for
  the smoke-sized hull); no lattice index was added.
- **R5.** Adopted: `heldout_mask_alpha` is listed in `input_policy.reconstruction_allowed`.
- **R6.** Adopted in part: tests that relocation leaves non-position rows byte-identical and that
  the seam observes post-surgery rows under CPU classic densification. The `initialize` face and
  minimum-voxel aborts remain covered by code review only.
- **R7.** Adopted: each relocation event records `jitter_fallbacks`.
- **R8.** Adopted: claim-boundary wording for photograph arms; production is a single manual
  invocation outside the cell limit; the dead production timeout branch is removed.
- **R9.** Adopted: `evaluate` asserts the held-out compact-view camera equals the calibrated one.
- **R10.** Deferred to a later task, as suggested.

## Verification

Focused CPU tests (22 across the relocation, protocol and conditional-density files) pass. A
repeated non-protocol GPU smoke (60 iterations, one seed, two training views standing in for
held-out) exercised every phase with the new prepare checks and fallback records.
