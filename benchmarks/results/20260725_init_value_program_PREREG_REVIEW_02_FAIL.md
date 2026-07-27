# Initialization Value Program — amended master protocol, second independent review

Reviewed on 2026-07-26 as a non-author review of the repaired document.

Reviewed preregistration: `benchmarks/results/20260725_init_value_program_PREREG.md`, SHA-256
`e8f75dd72425abcb5b2edf33a88ba7c456b69576b97500083fae064ebae5f8cc`
(working-tree content; the file is modified and uncommitted at repository base
`d6876bc1f1e78cbcc32f712c3a9d4706fa94ec0e`, which matches the base recorded in its header).

Prior review: `20260725_init_value_program_PREREG_REVIEW_INITIAL_FAIL.md`, SHA-256
`a7d0f2ac365fb675ece0be9ae8511df97036e563044b6e6f64d6f9a06385f201`.

Verdict: **FAIL — BOUNDED AMENDMENT REQUIRED (B1–B4)**.

This is a different verdict in degree, not in kind, from the first review. The nine initial
blockers are substantively addressed and most of the document is now a usable confirmatory
protocol. Four defects remain that would change a confirmatory decision if left in place: the
information control is not actually an information control, the time endpoint can score a success
that no speed difference produced, one headline guardrail metric does not exist in this
repository, and one metric threshold contradicts the implementation the protocol leans on. Each is
localized to §3.1, §4.2, or §5.1.6. The rest of the document should not be reopened.

No experiment, validation screen, seal, default change, or claim promotion is authorized by this
review.

## Reviewer independence and exposure

I am a separate session from the repair author (`ara/trace/sessions/2026-07-26_001.yaml`) and took
no part in drafting the amendment. I am not an outcome-blind external reviewer: I am the same
agent family operating in the same repository, with read access to every pilot artifact
enumerated in §1.1. Treat this as an internal non-author review. It satisfies "not the producing
session"; it does not satisfy "independent laboratory."

## What I verified as correct

Unlike the first round, the amended document makes checkable factual assertions. All of them
reproduce:

| Assertion | Check | Result |
| --- | --- | --- |
| §1.3 four manifest SHA-256 identities | `sha256sum dataset/*/frame_*/gaussians2d/manifest.json` | all four match exactly |
| §1.3 four semantic digests | `semantic_digest` field in each manifest | all four match exactly |
| §2.2 validation split IDs | membership in each frame manifest | all 14 IDs per scene exist; the four roles are disjoint; 16 unused views in `frame_00005`, 18 in `frame_00060` |
| §1.2 "Karate compact manifests do not contain packed alpha" | `has_alpha` over all views | Karate 0/30 and 0/32; Stage `frame_00008` 26/26 — the masked-endpoint exclusion is factually forced, not a convenience |
| §3.3 "no reviewed train-only SfM initializer is registered" | `rtgs.lift.baselines.SfMLifter` | accurate: it consumes a preexisting `scene.points` cloud and never runs SfM |
| §7.1 `2*2*2*4*3 = 96`; §7.2 `3*3*5*1 = 45` | arithmetic | both correct; checkpoint grid `0..1500` step `100` is 16 points |
| §4.2 foreground-MSE definition | `rtgs.core.metrics.masked_psnr` | exact match, including the `3 * mask-weight sum` denominator and the `1e-12` clamp |
| Feasibility of `N in {2400,5000}` on the validation screen | manifest `n_gaussians` | every view carries 5000 fitted 2D gaussians, so a 3-view train set supplies 15000 roots before adaptation |

Two additional facts, not asserted in the protocol, that support it:

- **Karate RGB recovery is fully determined today.** `20260716_compact_point_training_SEAL.json`
  records SHA-256 for all 62 Karate RGB JPEGs and `calibration_dome.json`. §6.2 can be bound to
  those hashes rather than to an unspecified "recovered" payload.
- **The Karate source has no masks either.** That seal lists zero alpha/mask files. §7.1's
  full-canvas `Qval` is therefore forced by the source capture, not chosen to avoid work, and no
  recovery effort can convert Karate into a masked-endpoint scene.

Repository hygiene checks at review time: `scripts/check_ara.py` → `OK (24 claims)`;
`git diff --check` → clean.

## Blocking findings

### B1. `random-count-matched` is not an information control as currently bound (§3.1)

§3.1 specifies the arm as "the existing seeded random initializer, restricted to the same
train-derived bounding volume and exact count." The existing initializer is
`rtgs.lift.baselines.RandomLifter`, and it differs from `beam-cover` in three ways that are not
initialization *information*:

- **Appearance.** It assigns a constant gray `0.5` to every primitive. `beam-cover` inherits Beam
  Fusion appearance, and the SfM arm samples color from training views via `_colors_from_views`.
- **Extent.** It uses `0.5 * extent / N**(1/3)`, not a cover-consistent extent.
- **Volume.** It samples a *sphere of half the scene extent* around `scene.center_and_extent()`,
  which is not a bounding volume and not train-derived (see A1).

Under §3.2 fixed topology, birth/clone/split/prune are disabled, so a bad appearance
initialization is not recoverable by densification and only partially recoverable by SH gradient
descent within 1500 updates. Gate §5.1.3 could therefore pass on color initialization alone,
which is exactly the confound this arm exists to exclude. A "the gain comes from initialization
information rather than optimizer variance" conclusion does not follow.

**Required amendment.** Freeze `random-count-matched` so it matches `beam-cover` on every factor
except mean placement: identical count, identical initial opacity `0.10`, identical extent rule,
and an identical train-view color-assignment procedure. State explicitly whether the random arm
receives cover-consistent extents. If any factor is deliberately left unmatched, name it and
state that the contrast is then a bundle contrast, not an information control.

### B2. The time endpoint contains a censoring asymmetry that can manufacture a success (§4.2, §5.1.4–5)

`tau(s,k) = Q_colmap(s,k,1500) - 0.25 dB`, and reaching a target requires persistence across the
next two scheduled checkpoints. The last checkpoint is `1500`. Combining these:

> COLMAP is right-censored **exactly when** `Q_colmap(1300) < tau`, i.e. when COLMAP gains more
> than `0.25 dB` over its final 200 updates.

That is a property of the comparator's optimizer schedule, not of any initializer, and for a
1500-update run it is entirely plausible. When it happens, §4.2 clause (a) awards the candidate a
paired time success merely for reaching a target the comparator "never reached" — a target defined
by the comparator's own final quality. Five such seeds satisfy §5.1.4 with no measured speed
difference at all, and §5.1.5 has the same structure against random.

The protocol elsewhere is careful that "reachability first, time second"; this clause silently
converts non-reachability of the comparator into a candidate win.

**Required amendment.** Choose one and freeze it:

1. treat comparator censoring and joint censoring as **non-evaluable** pairs rather than
   successes, and require `>=4/5` *evaluable* pairs to satisfy the ratio rule; or
2. apply the persistence requirement only where it is observable and define comparator crossing at
   the final two checkpoints without a persistence test, stating the resulting asymmetry; or
3. extend the checkpoint grid two checkpoints past the last decision checkpoint so persistence is
   observable at `1500`.

Option 1 is the smallest change and preserves the document's own stated logic. In all three cases,
add one sentence: target-reach dominance may be reported descriptively but does not by itself
satisfy §5.1.4 or §5.1.5. Note also that §3.2 can make `tau` undefined entirely (see A5), since it
is derived from a COLMAP arm the protocol permits to be unavailable.

### B3. The LPIPS guardrail is not executable in this repository (§4.2, §5.1.6, §6)

§5.1.6 makes "LPIPS not worse by more than `0.010`" a conjunctive headline condition and §4.2
pins it to AlexNet LPIPS v0.1. This repository contains no LPIPS implementation, no LPIPS
dependency in `pyproject.toml`, and no weight file; `grep -rn lpips -i src/ tests/ benchmarks/`
returns nothing. Hard Rule 1 additionally requires CPU-first, import-safe, deterministic scoring,
so a network-fetched weight file is not acceptable as-is, and §6's ten implementation items do not
mention LPIPS at all — a gate condition with no gate.

**Required amendment.** Either drop LPIPS from §5.1.6 (SSIM and alpha IoU already cover the
perceptual/geometric guardrail role and both exist), or add an eleventh §6 item requiring a
vendored, hash-pinned, offline, CPU-deterministic LPIPS backend behind a registered interface with
its own test. Do not leave the gate naming a metric no command can compute.

### B4. The alpha IoU threshold contradicts the implementation the protocol relies on (§4.2)

§4.2 thresholds "both the ground-truth mask and predicted alpha at `0.05`". The repository's
existing evaluator (`src/rtgs/optim/trainer.py:736-741`) thresholds both at `0.5`, and
`rtgs.core.metrics.masked_crop` takes its bounding box at `> 0.5`. A 10× looser threshold admits
faint alpha halos into the predicted foreground and changes what the `0.020` tolerance in §5.1.6
measures; the same document simultaneously invokes "the repository's masked foreground bounding
box" at the `0.5` convention one sentence earlier.

Related and cheap to fix in the same edit: §4.1 requires **median** aggregation over held-out
views, while `trainer.evaluate` means over views. The confirmatory scorer must not be that
function.

**Required amendment.** State the intended threshold deliberately (`0.05` with a stated reason, or
`0.5` to match the implementation), acknowledge the two different thresholds if both are kept, and
add to §6.8 that the scorer implements §4.2/§4.1 definitions directly and may not reuse
`trainer.evaluate`.

## Findings routed to the validation addendum (non-blocking)

These do not require touching the master protocol, but the addendum cannot pass without them.

**A1. `bounds_hint` is not train-derived.** Every scene bound available today comes from the
frame-level `bounds_hint` in the compact manifest — a single dataset-wide constant covering all
30/32 views, returned by `SceneData.center_and_extent()` before any camera geometry is consulted.
Any arm that reads it (random volume, field placement, depth alignment) inherits information
derived from validation and report-only views. The addendum must recompute bounds from train views
only and record how.

**A2. Staging necessarily changes the frozen manifest identity.** `CompactDataset.load` verifies
*every* view listed in the manifest and has no allow-list parameter, so §2.1's staged train-only
directory requires a new manifest — whose SHA-256, `semantic_digest`, and (per A1) `bounds_hint`
will differ from the §1.3 identities by construction. Freeze both: the source-frame identity *and*
the derived per-cell staged identity. Also freeze the view-ID ↔ source-file mapping. I verified it
is bijective for both frames (`rgb_N.jpeg -> C%04d`; 30/30 and 32/32); per-file hashes cannot
detect a *role* mis-assignment, so the mapping needs its own frozen record and negative test.

**A3. §4.4's reference-gap ratios have no arm to bind to.** `R_Q` and `R_T` are defined against an
"oracle/reference run," but no reference arm appears in §3.1 or §7.1. The only reference
construction in the document is §8.2's correspondence-matched synthetic means. Scope §4.4 to the
synthetic factorial or name the arm; as written it is inert.

**A4. The timed path does not exist as one command.** §4.3 forbids cached arm-specific
preprocessing, but the checked-in dataset *is* cached Stage 1 output and the tree contains no RGB
at all. The confirmatory `beam-cover` timing therefore requires an RGB → `fit-images` → lift →
optimize path executed in one measured process. §6 should require that exact command plus a
negative test asserting no prebuilt `.rtgsv` is read during a timed run.

**A5. §2.3 and §3.2 can jointly make the program undecidable.** Eligibility "is based only on
acquisition and calibration QC, never on method output or COLMAP success," while a COLMAP yield
below `N=2400` on any one of three scenes both records `insufficient_initial_points` and removes
the headline — and (per B2) leaves `tau` undefined on that scene. This is honest and fail-closed,
but its most likely outcome is a null that reflects comparator yield rather than initializer
value. Freeze *now*, before scene acquisition, either a second lower anchor evaluated identically
or an explicit predeclared disposition for the under-production case. Any such rule invented after
counts are observed is post-hoc.

**A6. The run-order rule is underspecified.** "Deterministic seed-keyed Latin-square rotation" does
not define an order for 3 arms × 5 seeds. Publish the explicit 5×3 order table.

**A7. Provenance hygiene before the addendum.** The reviewed master protocol, both reviews, the two
2026-07-25 pilot preregistrations, and their harnesses (`benchmarks/gpu_init_cost_to_target.py`,
`benchmarks/gpu_init_refit_ceiling.py`) are uncommitted or untracked. §1.1 uses those pilots as the
outcome-access ledger; an untracked file cannot anchor a chronology claim. Commit them before the
addendum is hashed.

**A8. Bind §6.2 to the existing seal.** Karate RGB recovery should be verified against the 62 file
hashes plus `calibration_dome.json` already recorded in
`20260716_compact_point_training_SEAL.json`, not against a fresh unverified copy.

## What the amendment got right and should not be reopened

- The outcome-access ledger in §1.1 and the pilot/validation/confirmation role table in §1.2. The
  document states plainly that it was written after results existed and does not retroactively
  license prior runs.
- Retiring `Omega` from all decision rules (§4.5) rather than repairing it.
- Direction-normalized quality and time gaps with explicit undefined cases (§4.4), and the
  renaming of `P_2D` to a self-reconstruction baseline (§8.1).
- Synchronized end-to-end seconds as the primary cost estimand with iterations demoted to a
  diagnostic (§4.3).
- Fail-closed comparator availability (§3.3): a missing COLMAP arm is reported as unavailability,
  never converted into a coverage win.
- Conjunctive per-scene gates with no pooled rescue, and the explicit licensed-wording split when
  only one co-primary passes (§5.1).
- The null and narrowing outcomes in §5.4, and the one-repair-family reset rule in §8.5.
- Keeping optimizer utility and physical covariance validity as separate licences, which preserves
  the existing observability boundary rather than overwriting it.

## Minimal route to a PASS

1. Amend §3.1 to match the random arm to `beam-cover` on every non-placement factor (B1).
2. Amend §4.2 to remove the censoring asymmetry, and add the "reach dominance is descriptive"
   sentence to §5.1.4–5 (B2).
3. Resolve LPIPS in §5.1.6 — drop it, or add its §6 implementation item (B3).
4. Fix the alpha IoU threshold in §4.2 and add the scorer-independence sentence to §6.8 (B4).
5. Re-hash the document and obtain a fresh non-author review of the amended sections. The
   remainder of the protocol carries this review's assessment forward unchanged.

Item §6.2 (Karate RGB recovery) is a data-restoration action fully determined by the 2026-07-16
seal and carries no scientific risk; it may proceed in parallel with the amendment. No other §6
implementation work, and no validation screen, should start before the PASS.

## Checks performed

Read-only: `git log`, `git status`, `git rev-parse HEAD`, `git diff --check`, `sha256sum` on the
reviewed documents and all four frame manifests, direct JSON inspection of the four manifests and
of `20260716_compact_point_training_SEAL.json`, a Python cross-check of the §2.2 split IDs, role
disjointness, `has_alpha`, `n_gaussians`, and the RGB-to-view-ID bijection, source reading of
`rtgs/lift/baselines.py`, `rtgs/lift/__init__.py`, `rtgs/core/metrics.py`, `rtgs/data/scene.py`,
`rtgs/data/compact_views.py`, `rtgs/optim/trainer.py` (evaluation path), and `pyproject.toml`, plus
`grep` sweeps for LPIPS, alpha IoU, farthest-point adaptation, fixed-topology enforcement, and
COLMAP usage. Executed `.venv/bin/python scripts/check_ara.py`.

Not done: no training, rendering, lifting, benchmark, or GPU run; no CPU test suite run (this is a
document review, not a results audit or an implementation change); the reviewed preregistration was
not modified; no sealed or unopened data was inspected; no untracked user file was altered.
