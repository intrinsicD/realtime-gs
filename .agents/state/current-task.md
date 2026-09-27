# Current Task

## Title

Inert gsplat Default opacity reset: evidence, disclosure and opt-in intended reset

## Task ID

RTGS-027

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: reviewer

## Mode

Implement

## Risk

Standard

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal

Record that gsplat 1.5.3 `DefaultStrategy.step_post_backward` never executes its opacity reset
(`step % self.reset_every == 0 & step > 0` is a chained comparison ending in `0 > 0`), correct the
architecture wording, and provide an opt-in intended reset usable through the existing
`parameter_step_callback` seam without changing defaults or historical configurations.

## Motivation

Found by the Fable 5.1 RTGS-026 prospective review and confirmed by the Driver; the user asked on
2026-09-27 to handle it. The repository's geometric-arena path already performs the reset, while
the default dynamic path relies on the inert upstream call.

## Success Criteria

- A CPU test demonstrates the upstream precedence defect independent of gsplat installation.
- `rtgs.optim.strategies.IntendedOpacityReset` clamps opacity logits and zeroes the opacity Adam
  moments exactly at the intended steps, CPU-tested; a CUDA test exercises it inside gsplat-default
  training; `chain_parameter_callbacks` composes it with other seam users.
- ARCHITECTURE wording, a staging observation and the experiment log disclose the inert reset.
- No default, TrainConfig or DensityConfig field changes; historical protocol tests stay green.

## Constraints

CPU-first imports; no gsplat import in the new CPU path; no historical evidence edits.

## Non-Goals

Changing the default, patching the installed gsplat, rerunning historical experiments.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

None

## Current Evidence

RTGS-026 review V1 (R1) and audit; gsplat 1.5.3 `strategy/default.py` source; the arena path in
`rtgs.optim.strategies.GsplatStrategyController._arena_default_step`.

## Minimal Plan

1. Tests for the upstream defect and the intended reset.
2. Implement the opt-in reset and callback chaining.
3. Docs, staging observation, verify, self-review.

## Status

In review

## Human Decisions

### Question
How to handle the inert gsplat opacity reset?
### Options
Document only; opt-in intended reset; change the default.
### Recommendation
Document plus opt-in intended reset; defaults unchanged.
### Decision
User asked to handle it (2026-09-27: "mach alle 3").
### Date
2026-09-27

### Question
After two consecutive Revision required verdicts: which clock should IntendedOpacityReset use?
### Options
gsplat's own iteration clock (`g = step - 1`); keep the completed-step clock but bound the window by
the run end; keep it and document the final-iteration reset as a known divergence.
### Recommendation
gsplat's own iteration clock (reviewer and Driver).
### Decision
User chose gsplat's own iteration clock.
### Date
2026-09-27

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-27, for independent code review)

#### Objective
Independent review of the RTGS-027 diff.

#### Reviewed state
Branch `rtgs-027-gsplat-opacity-reset`, diff against `main` at `271cce5`.

#### Changes
`upstream_default_reset_fires`, `IntendedOpacityReset`, `chain_parameter_callbacks` in
`rtgs.optim.strategies`; `tests/test_gsplat_opacity_reset.py`; ARCHITECTURE wording; O182;
experiment-log note.

#### Evidence
6 tests pass on the local GPU (incl. gsplat-default training with the reset firing at step 20);
5 pass and 1 skips on CPU; `./scripts/verify.sh` exit 0.

#### Assumptions
Intended semantics follow gsplat `reset_opa` (clamp to `2 * prune_opa`, zero opacity moments)
within `0 < step < refine_stop_iter`, as in the repository's arena path.

#### Uncertainties
No quality effect measured.

#### Review Focus
Semantics match upstream intent and the arena path; no default/config change; CPU-first imports.

#### Protected actions not taken
No default change, no merge.

#### Recommended Next Action
Fable 5.1 code review.

### Review (2026-09-27, Fable 5.1 code review round 1, verbatim)

#### Verdict
Revision required

#### Self-reviewed
No

#### Correctness
- **Precedence claim verified.** Installed gsplat 1.5.3 at `.venv/lib/python3.12/site-packages/gsplat/strategy/default.py:195` reads `if step % self.reset_every == 0 & step > 0:` (sha256 of the file `8f9cf7d5…a622`). The AST is one `Compare` node with ops `[Eq, Gt]` and comparators `[0 & step, 0]`, so the expression is `(step % reset_every == (0 & step)) and ((0 & step) > 0)`. `0 & step` is `0` for every int, so the last term is `0 > 0`. Brute force over `reset_every` 1..50 and `step` −100..4999 gives zero hits. The docstring of `upstream_default_reset_fires` states this correctly.
- **Reset semantics match upstream.** gsplat `reset_opa` (`ops.py:214`) clamps logits to `logit(value)` computed in float32 and zeroes every optimizer-state tensor except `step`. `IntendedOpacityReset` clamps to the same cap and zeroes every state tensor whose shape equals the parameter shape. Adam `step` is shape `()`, so it survives, as upstream. Upstream replaces the Parameter object; the callback mutates in place, which is safe and keeps optimizer-state identity. The arena path `GeometricParameterArena.reset_opacity` (`arena.py:317`) is the same clamp-plus-zero in place.
- **Window matches in count, not in clock.** Upstream returns early at `step >= refine_stop_iter` and would reset at `step % reset_every == 0`, `step > 0`; the arena branch does exactly that at `strategies.py:296`. `due()` implements the same predicate. But the trainer hands the gsplat controller `global_it` (`trainer.py:899`) and the callback `completed_step = global_it + 1` (`trainer.py:658`, `:907`). So the callback resets at the end of iteration `global_it = k*reset_every − 1`, one iteration before the upstream and arena clock and on the same clock as the classic CPU controller (`density.py:352`). Reset count per run is identical and the refine pause still follows each reset. This is not a defect, but the docstring's "matching … the geometric-arena path" and the handoff's "as in the arena path" are imprecise.
- **Ordering.** The callback runs after `optimizer.step()` and after density control in the same iteration, so at a coincident refine/reset step grow/prune precede the reset, as in upstream `step_post_backward`. Caveat: under opt-in `conditional_density` the callback still fires during non-unit phases where density hooks are suppressed (`trainer.py:668`, `:880`); not a default.
- **No default or config change.** The diff touches only `strategies.py` additions, one new test file, docs, staging, and the task record. `DensityConfig`, `TrainConfig`, and `trainer.py` are unchanged.
- **CPU-first.** `strategies.py` has no module-level gsplat import; the test imports gsplat only through `importorskip`.
- **Style.** `opacities.data.clamp_(max=cap)` uses `.data` under `@torch.no_grad()`; the `.data` is unnecessary and rtgs-review §2 says to avoid it. Existing precedent at `silhouette_relocation.py:167`, so not blocking.

#### Evidence Quality
- Reproduced on CPU: `pytest -q tests/test_gsplat_opacity_reset.py tests/test_silhouette_relocation.py tests/test_field_only_distillation.py tests/test_silhouette_relocation_index_decode_protocol.py` gives 33 passed, 1 skipped.
- Not reproduced: the CUDA test. `torch.cuda.is_available()` is False in this session, so the handoff's "6 tests pass on the local GPU" is unverified by me. `./scripts/verify.sh` was outside the allowed command set and is likewise taken on the Driver's word.
- Historical statement: RTGS-025 and RTGS-026 have tracked receipts with gsplat 1.5.3 (`ara/evidence/tables/*_final_handoff/preflight.json`) and the RTGS-026 audit confirmed the inert reset on loss and density traces. RTGS-021 is gsplat-default dynamic per its task JSON, but no tracked receipt in `benchmarks/results/` or `ara/evidence/tables/` records its gsplat version; the claim rests on `uv.lock` pinning 1.5.3. Acceptable, but say so or cite the lockfile.
- O182 binds to N249, which exists. The `refuted`/`supported` ledger is untouched; correct, since nothing is promoted.
- **Wording inaccuracy (required).** ARCHITECTURE and O182 say `opacity_reset_every` "is inert on the default dynamic path, only pausing refinement". Counterexample: gsplat `_prune_gs` gates large-scale pruning on `step > self.reset_every` (`default.py:320`), and the adapter maps `opacity_reset_every <= 0` to at least 1e9 (`strategies.py:62-66`), which silently disables large-scale pruning on the dynamic path. The reset is inert; the parameter is not. The arena branch has the same gate (`strategies.py:270-274`).
- The pause wording is accurate: with `pause_refine_after_reset = every` the pause skips exactly the refine event coincident with each would-be reset.

#### Simplicity
- The callback is the smallest opt-in under the no-config-change constraint, and reuses the existing seam. Good.
- The simpler faithful design for any future default change is to perform the reset in the dynamic branch of `GsplatStrategyController.post_backward` exactly as the arena branch does; that would also keep the `global_it` clock. It needs a config flag, so it is correctly out of scope here.
- `upstream_default_reset_fires` is library code that exists only to be tested; it could live in the test module. `chain_parameter_callbacks` is fine.

#### Missing Cases
- No CPU trainer-level test of `IntendedOpacityReset` through `Trainer.train` with the torch rasterizer; the only trainer-level test is CUDA and skips in CI. A 12-iteration run with `reset_every=5`, `stop_iter=11`, `densify=False` should yield events at `[5, 10]` and unchanged count.
- No assertion that Adam `step` state survives the reset.
- The one-iteration clock offset versus the arena path is neither documented nor tested.
- `from_density` raises when `opacity_reset_every <= 0`, whereas the adapter treats that value as "disabled"; returning `None` would compose with `chain_parameter_callbacks`.
- The canary test `test_installed_gsplat_carries_the_defective_expression` will fail, not skip, on a GPU box after a gsplat upgrade that fixes the bug. That is the intended signal, but a version check or explicit message would make the failure readable.

#### Required Changes
1. In `docs/ARCHITECTURE.md` (rtgs/optim row) and `ara/staging/observations.yaml` O182, replace "`opacity_reset_every` is inert … only pausing refinement" with wording that the reset is inert while the parameter still (a) pauses refinement for `every` steps after each would-be reset and (b) gates gsplat's large-scale pruning onset via `step > reset_every`, so a non-positive value disables large-scale pruning on the dynamic path.
2. In the `IntendedOpacityReset` docstring, the `docs/EXPERIMENTS.md` note, and the handoff assumption, state the clock: the callback evaluates the trainer's completed step, one iteration before the gsplat-strategy and arena `global_it` clock and on the same clock as the classic controller; clamp and moment semantics match, reset count matches.

Both are wording-only; no code change is required for acceptance.

#### Optional Improvements
- Drop `.data`: `opacities.clamp_(max=cap)` under `no_grad` is sufficient.
- Add the CPU trainer-level test and the Adam `step` preservation assertion described above.
- Have `from_density` return `None` for non-positive `opacity_reset_every`, or document the raise.
- Move `upstream_default_reset_fires` into the test module.
- Guard the canary test with the installed gsplat version so an upgrade produces a clear message.
- Record the GPU test receipt (device, torch, gsplat) in the task record's Evidence, cite `uv.lock` for the RTGS-021 version, and bind "Reviewed state" to commit `8779de5`.
- Note the `conditional_density` coarse-phase caveat in the docstring.

### Handoff (2026-09-27, revision 1 for code review round 2)

#### Objective
Address the round-1 required wording changes and selected optional improvements.

#### Reviewed state
Round 1 reviewed commit `8779de5`; revision on branch `rtgs-027-gsplat-opacity-reset` (next commit).

#### Changes
Required 1: ARCHITECTURE and O182 now say the reset is inert while `opacity_reset_every` still
pauses refinement after each would-be reset and gates gsplat's large-scale pruning onset (a
non-positive value disables large-scale pruning). Required 2: the `IntendedOpacityReset`
docstring and the EXPERIMENTS note state the completed-step clock (one iteration before the
gsplat/arena clock, same as the classic controller; reset count matches). Optional: `.data`
removed; `from_density` returns `None` for non-positive periods; CPU Trainer-seam test (events at
steps 5 and 10, count unchanged); Adam `step` preservation assertion; readable canary message with
the gsplat version; conditional_density caveat in the docstring; O182 cites `uv.lock` for RTGS-021.
Not adopted: moving the probe into the test module (kept as documented library probe).

#### Evidence
GPU receipt: RTX 3050, torch 2.9.0+cu128, gsplat 1.5.3 — 7 tests pass; CPU: 6 pass, 1 skip.

#### Assumptions
Clamp and moment semantics follow gsplat `reset_opa`; clock differs by one iteration (documented).

#### Uncertainties
No quality effect measured.

#### Review Focus
The two required wording changes and the new tests.

#### Protected actions not taken
No default change, no merge.

#### Recommended Next Action
Fable 5.1 code review round 2.

### Review (2026-09-27, Fable 5.1 code review round 2, verbatim)

#### Verdict
Revision required

#### Self-reviewed
No

#### Correctness
- **Round-1 required changes are done and accurate.** `docs/ARCHITECTURE.md` and O182 now say the reset is inert while the parameter still pauses refinement and gates large-scale pruning onset. Both statements hold against the installed source: `default.py:170` pauses via `step % reset_every >= pause_refine_after_reset` with the adapter passing `pause_refine_after_reset=config.every` at `strategies.py:76`, and `default.py:320` gates large-scale pruning on `step > self.reset_every`, with a non-positive period mapped to a huge value at `strategies.py:62-66`. The docstring, the EXPERIMENTS note and the handoff assumption now state the completed-step clock.
- **Precedence claim re-verified.** The installed file hashes to `8f9cf7d5cb02999cdb3abfb85d796e6989d5249c25ed7b08ee58538a4288a622`, identical to round 1. The AST of `default.py:195` is one `Compare` with ops `[Eq, Gt]` and comparators `[0 & step, 0]`. A brute force over periods 1 to 50 and steps from minus 100 to 4999 gives zero hits.
- **Clamp and moment semantics match upstream.** gsplat's `_update_param_with_optimizer` skips the `step` key and zeroes every other state tensor. The callback zeroes tensors whose shape equals the parameter shape, so Adam's scalar `step` survives, and the new assertion covers it. The clamp cap is the same float32 `logit(value)`. The `.data` access is gone and the in-place clamp under `no_grad` is correct on a leaf Parameter.
- **Ordering.** The callback runs at `trainer.py:904-909` after the optimizer step and after density control, so grow and prune precede the reset within an iteration, as upstream.
- **New counterexample, terminal step (required).** Upstream evaluates `global_it`, which never exceeds `iterations - 1`, so it never resets on a run's final iteration. The callback evaluates `completed_step = global_it + 1` and fires whenever the final completed step is a multiple of the period below `stop_iter`. `Trainer.train` then returns `build().detach()` at `trainer.py:1033` with every opacity clamped to at most twice the prune opacity and no recovery iterations. With `from_density` and the `DensityConfig` default `stop_iter` of ten million, this hits runs of 6000, 15000 or 30000 iterations. The same offset applies at a resumed segment: if `iteration_offset` is itself a reset multiple, upstream resets at the end of the first iteration and the callback does not. So the sentence "the reset count is the same" in the docstring and in `docs/EXPERIMENTS.md` is false at both run boundaries. This corrects my round-1 statement that the count per run is identical; the Driver adopted that wording at my request.
- **Classic clock claim still true.** `density.py:206` and `density.py:352` share the terminal-step trait, so "on the classic controller's clock" is accurate. But the class exists to restore gsplat's intended reset, and gsplat's intent excludes a last-iteration reset.
- **No default or config change.** The branch diff against `main` touches only additions in `strategies.py`, one new test file, docs, staging and the task record. `density.py`, `trainer.py`, `arena.py`, `cli.py`, `pyproject.toml`, `uv.lock`, `experiments/`, `benchmarks/` and `scripts/` are untouched.
- **CPU-first.** `strategies.py` has no module-level gsplat import. The canary test imports gsplat only through `importorskip` and runs here because gsplat imports on CPU.

#### Evidence Quality
- Reproduced on CPU with the allowed command over the four files: 34 collected, 33 passed, 1 skipped. The skip is the CUDA test with reason "requires a CUDA GPU". The new file alone: 6 passed, 1 skipped in under one second.
- O182's version statements check out: `uv.lock` pins gsplat 1.5.3, and both `preflight.json` receipts under `ara/evidence/tables/` for RTGS-025 and RTGS-026 record `"gsplat": "1.5.3"`. O182 binds to N249, which exists.
- Not reproduced: the CUDA test, `./scripts/verify.sh` and ruff, all outside my allowed commands or hardware. The Driver's GPU receipt names torch 2.9.0+cu128 and gsplat 1.5.3, which match the venv, so it is plausible but unverified by me.
- `git diff --check` is clean. The commit message names the review and carries attribution. No closed `docs/tasks/RTGS-027*` record exists yet, which is correct while the task is active.
- Determinism: the conftest autouse fixture seeds torch, and the new CPU seam test uses a seeded synthetic scene.

#### Simplicity
- The callback remains the smallest opt-in under the no-config-change constraint.
- The simplest correct fix is also simpler than the current design: evaluate upstream's own predicate on upstream's clock inside the callback, using `g = step - 1`, so the reset fires at the end of exactly the iteration gsplat would have reset. That removes the three-place "one iteration earlier" caveat instead of documenting it, matches the pause relation exactly, and fixes both boundaries with no run-length argument.
- Keeping `upstream_default_reset_fires` in the library is a Driver judgment call and acceptable.

#### Missing Cases
- No test covers the terminal boundary. A CPU seam run with `iterations=10`, `reset_every=5`, default `stop_iter` and `densify=False` currently fires at completed step 10 and returns a clamped model.
- No test or wording covers a resumed segment whose `iteration_offset` is a reset multiple.
- The callback composed with `gaussian_storage_policy="geometric"` double-resets on consecutive iterations, since the arena already resets. Not guarded or mentioned in the docstring.
- The callback's events record completed steps while `GsplatStrategyController.stats` records `global_it`, so logs of the same event differ by one.

#### Required Changes
1. Make the fired set equal upstream's. Preferred: in `due`, evaluate `g = step - 1` with `g > 0 and g < stop_iter and g % reset_every == 0`, record both the completed step and `g` in each event, and adjust the three existing step expectations. Alternative: keep the completed-step clock and bound the window by `min(stop_iter, iteration_offset + iterations)`, documenting the offset-start case.
2. Add a CPU trainer-seam test for the terminal boundary: a run whose length is a reset multiple below `stop_iter` must not reset on its final iteration, asserting the returned Gaussians' maximum opacity exceeds the reset value.
3. Replace "the reset count is the same" in the `IntendedOpacityReset` docstring and `docs/EXPERIMENTS.md` with a statement that holds for the chosen fix. With the preferred fix, also drop the "one iteration before the arena path" clause from `docs/ARCHITECTURE.md`.

This is the second consecutive `Revision required`. Per `docs/AGENT_WORKFLOW.md`, the Driver should record the choice under Human Decisions and set `Blocked on human decision` unless the human accepts a fix directly. The options are: adopt the gsplat-clock predicate, bound the window by the run end, or keep the current clock and document the terminal-step reset as a known divergence. I recommend the first.

#### Optional Improvements
- Add one docstring sentence stating the callback must not be combined with the geometric arena, which already performs the reset.
- Bind the handoff's "Reviewed state" to commit `9be3f4a` and the next revision's hash before closeout.
- Consider naming the boundary in the O182 context line so the staging record does not inherit the "count matches" claim.

### Handoff (2026-09-27, revision 2 for code review round 3 after human decision)

#### Objective
Implement the user's chosen gsplat-clock predicate and close the round-2 findings.

#### Reviewed state
Round 2 reviewed `9be3f4a`; this revision is the next commit on `rtgs-027-gsplat-opacity-reset`.

#### Changes
`IntendedOpacityReset.due` evaluates `g = step - 1` with `0 < g < stop_iter` and `g % reset_every == 0`;
events record `step` and `gsplat_iteration`; docstring, ARCHITECTURE, EXPERIMENTS and O182 drop the
one-iteration/count caveat and state the gsplat clock (no final-iteration reset); docstring warns
against combining with the geometric arena. New CPU seam test: a 10-iteration run with period 5
resets only at gsplat iteration 5 and returns max opacity above the reset value. Existing step
expectations adjusted (CUDA test observes completed step 21 = gsplat iteration 20).

#### Evidence
GPU (RTX 3050, torch 2.9.0+cu128, gsplat 1.5.3): 8 tests pass; CPU: 7 pass, 1 skip.

#### Assumptions
Resumed segments now follow gsplat's clock automatically (the predicate is on the global step).

#### Uncertainties
No quality effect measured.

#### Review Focus
Predicate equivalence with gsplat/arena at both run boundaries; tests.

#### Protected actions not taken
No default change, no merge.

#### Recommended Next Action
Fable 5.1 code review round 3.
