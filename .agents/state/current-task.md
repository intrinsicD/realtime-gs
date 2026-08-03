# Current Task

## Title

Provider-neutral StructSplat BENCH-019 downstream receipt exporter

## Task ID

RTGS-009

## Role Assignment

- Driver: Codex-bench019-exporter-driver
- Reviewer: Codex-independent-bench019-reviewer
- Turn: driver

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Pipeline-integrated
- Reached: Pipeline-integrated

## Goal

Implement a strict, CPU-first realtime-gs exporter for
`structsplat.bench019.cell.v1`. The exporter must bind one frozen StructSplat BENCH-019 protocol
cell to its exact field semantics, shared downstream factor, raw metric sources, and inspectable
artifacts without executing or interpreting the StructSplat analysis. It must also inventory a
six-capture development/confirmation data portfolio without opening a formal outcome phase.

## Motivation

StructSplat BENCH-019 already owns protocol freezing, A/A validation, clustered ranking, and
portable reporting, but it intentionally does not execute realtime-gs. The current realtime-gs
RTGS-008 draft is a dirty, single-frame paper experiment and cannot serve as a general exporter.
The owner selected the general-scope route: isolate this work in a clean worktree, preserve the
active RTGS-008 worktree, and prepare independent capture groups before any downstream conclusion.

## Success Criteria

- A reusable CPU-only module and bounded task driver emit the exact
  `structsplat.bench019.cell.v1` row shape without importing StructSplat or CUDA-only packages.
- Frozen/review protocol integrity, stable cell identity, field-manifest hash, semantic digest,
  result metric names, and source artifact hashes fail closed on drift.
- The downstream-factor digest is derived only from frozen protocol bindings plus frame, seed, and
  initializer, is identical across field families, and is identical for the A/A replay.
- Successful rows extract every Stage-1 predictor and downstream response from declared JSON
  pointers into sealed source artifacts; constants, missing values, booleans, non-finite values,
  and post-export transformations are rejected.
- Successful rows bind field, history, config, target, reconstruction, and error artifacts by
  exact path, SHA-256, and byte length. Error rows remain explicit and cannot masquerade as
  successful cells.
- An assembler orders cells exactly as the frozen protocol declares, rejects duplicates and
  undeclared cells, and fails closed on missing cells unless explicitly producing an incomplete
  diagnostic inventory.
- Deterministic tests cover valid export, normalized/additive semantic preservation, A/A factor
  identity, family-factor drift, metric-source tampering, artifact tampering, duplicate/missing
  cells, and append-only output behavior.
- A calibrated, non-result-bearing diagnostic exercises the exporter boundary without accessing a
  protected BENCH-019 outcome or changing a production default.
- A source-bound inventory records three development and three confirmation capture groups. Any
  unavailable field-family production remains visible rather than being inferred as complete.
- Focused tests, docs/workflow checks, self-review, and `./scripts/verify.sh` pass before handoff.

## Constraints

- Preserve `/home/alex/Documents/realtime-gs` and all of its uncommitted RTGS-008, ARA, IDE, and
  experiment files unchanged. Work only in `/home/alex/Documents/realtime-gs-bench019`.
- Preserve BENCH-019's exact additive versus normalized labels; never convert a normalized field
  and relabel it additive.
- The exporter is passive evidence infrastructure. It does not choose a Stage-1 surrogate, loss,
  field representation, initializer, or realtime-gs schedule.
- No formal execution before StructSplat's distinct prospective protocol review and clean frozen
  protocol. Diagnostic review-state exports must be labelled and cannot become claim-ready rows.
- Development and confirmation capture identities are disjoint. Confirmation outcomes remain
  unopened until the upstream gate authorizes them.
- All output writes are new-file or empty-directory only; no existing receipt or result bundle is
  repaired in place.

## Non-Goals

- Completing or publishing RTGS-008, modifying the dirty primary realtime-gs worktree, or adopting
  its single-frame outcome as general evidence.
- Running the BENCH-019 correlation analysis, promoting a scientific claim, or selecting a Field
  V2 semantic contract.
- Producing missing 2D field families, tuning downstream hyperparameters, or consuming a sealed
  confirmation split in this implementation task.
- Adding another HTML/report framework; StructSplat remains the BENCH-019 report authority.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- rtgs-bench
- rtgs-docs-sync
- rtgs-review
- rtgs-verify
- realtime-gs-results-audit

## Experiment Contract

None

## Current Evidence

- Clean branch `rtgs/009-bench019-exporter` starts at realtime-gs commit `4c1a7a5` in an isolated
  worktree; the primary worktree remains dirty and untouched.
- StructSplat commit `c61db61` defines `structsplat.bench019.cell.v1`, its exact row fields, the
  six required cell artifacts, frozen cell order, and cross-family downstream-factor equality.
- `rtgs.bench019` and its task-local CLI now validate protocol/review integrity, exact field
  semantics, verified global factors, sealed JSON pointers, run bindings, six cell artifacts,
  append-only writes, export receipts, and stable receipt-required assembly.
- The committed pre-outcome portfolio binds 88 exact source files across three development groups
  (Janelle Stage, TUM `fr1/xyz`, TUM `fr1/rpy`) and three disjoint confirmation groups (Janelle
  Karate, TUM `fr1/desk`, TUM `fr1/desk2`). All four TUM archives match pinned official bytes and
  SHA-256 values.
- Stage has two complete-but-unbound 26-view field families and a snapshotted incomplete 13/26
  mask-contained family. Karate has no frozen mask policy; every TUM group still needs a keyframe,
  camera, split, and mask adapter. No confirmation field family has been produced or opened.
- A synthetic frozen success row and a calibrated Stage review-state intentional-error row both
  passed StructSplat's independent `validate_result_rows` with zero problems. The calibrated row
  bound field manifest SHA-256 `e6daeb328000e08343dca2d36a1b817549d632d6b1298de58dd830ad5dffcdcb`
  and semantic digest `1981bf163e67d51485a6eb80afe0f451b9eaf5a1e24c4db6a51e63d6dc0ece84`.
- Eighteen focused exporter/portfolio tests pass. The canonical `./scripts/verify.sh` gate and the
  additional complete CPU pytest suite pass from the isolated worktree; only the two established
  PyTorch warnings are emitted.
- The implementation is committed as `d3e76fe44f3afe2044505fdcf1e7043657a4e4b1`; StructSplat
  binds that still-unaccepted checkpoint in commit `5f1b1fb516a10e48af14081c82017ac7388167fb`.
- Revision `f582c58` closes all three independent-review findings: formal frozen protocols now
  mirror StructSplat's portable v1 repository, analysis, path, seed, metric, and A/A invariants;
  assembly reproduces rows from the receipt-bound source manifest; and portfolio verification
  re-enforces the four official TUM archive identities.
- Twenty-six focused tests pass, including cooperative receipt/source mutations and a
  self-consistent TUM substitution. One exact frozen positive fixture is accepted by both the
  realtime-gs and StructSplat validators, while both reject the digest-consistent no-repository
  negative fixture with the same reason.
- The canonical `./scripts/verify.sh` gate and the additional complete CPU pytest suite pass on
  the repaired revision; only the two established PyTorch warnings are emitted.
- No general BENCH-019 downstream row or correlation result exists.

## Minimal Plan

1. Completed: freeze the exporter/source/factor contracts against the StructSplat row validator.
2. Completed: implement the CPU-only modules, bounded CLIs, receipt-required assembler, and
   adversarial tests.
3. Completed: exercise synthetic-success and calibrated-error diagnostics and update public docs.
4. Completed: bind three development plus three confirmation source groups without fitting or
   accessing confirmation outcomes.
5. Revision required: the original three findings are closed, but the acceptance-pass differential
   audit found that formal identity still fails to rehash every frozen artifact and accepts some
   whitespace-only strings that StructSplat rejects. Complete one bounded parity repair before
   resuming the same independent acceptance pass.

## Status

Revision required

## Human Decisions

### Question

Should BENCH-019 remain a frame-00008 workload-specific diagnostic or prepare a general-scope
cross-capture run?

### Options

Use only frame 00008, or build the provider-neutral exporter in a clean realtime-gs worktree and
obtain independent capture groups.

### Recommendation

Use the clean general-scope route so later Field V2 choices are not selected from one exposed
capture.

### Decision

(Owner, in chat.) Do the general-scope clean-worktree route.

### Date

2026-08-03

### Question

Should the remaining validator-parity findings stop the task after the first repair, or receive one
bounded systematic fix before the same acceptance pass resumes?

### Options

Stop and reopen the design with the owner, or complete the exact portable-v1 artifact traversal and
stripped-string parity fix without changing scope or opening outcomes.

### Recommendation

Complete the bounded parity fix because the reviewer has isolated exact missing upstream checks;
this is evidence-boundary correctness, not a new method or result loop.

### Decision

(Owner, standing direction in chat.) Continue with the next steps and do the independently reviewed
recommended work. The reviewer deliberately returned these findings as continuation feedback and
did not record a second `Revision required` verdict; keep this to the exact parity repair.

### Date

2026-08-03

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (provider-neutral exporter and source portfolio)

#### Objective

Independently review the general BENCH-019 realtime-gs evidence boundary and six-group source
portfolio before StructSplat binds this revision or anyone freezes and executes a formal protocol.

#### Reviewed state

Commit `d3e76fe44f3afe2044505fdcf1e7043657a4e4b1` on branch
`rtgs/009-bench019-exporter`, based on clean realtime-gs commit `4c1a7a53`.

#### Changes

- Added the passive CPU-only cell exporter, exact downstream-factor derivation, source/run binding,
  row validator, provenance receipt, and receipt-required stable assembler.
- Added a source-portfolio contract and task-local create/verify driver with pinned official TUM
  archive identities and a hard confirmation-outcome boundary.
- Added adversarial contract tests and documented the cross-repository ownership boundary in the
  README, architecture, and repository map.
- Closed a self-review finding that had allowed the assembler to accept a row without its export
  receipt; assembly now rehashes the protocol, row, source manifest, metric sources, cell
  artifacts, factor record, and paired receipt.

#### Evidence

- `tests/test_bench019_exporter.py` and `tests/test_bench019_portfolio.py`: 18 passed.
- Source portfolio local rehash: three development groups, three confirmation groups, 88 files.
- Synthetic frozen success diagnostic: `status=ok`, integer metric preserved, zero StructSplat row
  validation problems.
- Calibrated Stage diagnostic: review-state `status=error` by design, exact GaussianImage field and
  semantic hashes preserved, zero StructSplat row validation problems.
- `PYTHONPATH=/home/alex/Documents/realtime-gs-bench019/src
  PY=/home/alex/Documents/realtime-gs/.venv/bin/python ./scripts/verify.sh`: passed.
- Complete CPU suite with the isolated worktree on `PYTHONPATH`: passed.
- `git diff --check`: passed. The dirty primary realtime-gs worktree was inspected read-only and
  remains outside this branch.

#### Assumptions

The four TUM sequences are treated as distinct acquisition trajectories at the source-inventory
level. That does not by itself establish the stronger statistical independence or domain diversity
needed for a general scientific claim; StructSplat's prospective review remains authoritative.

#### Uncertainties

Five groups have no matched Stage-1 families. TUM keyframes/adapters and the Karate mask policy are
unfrozen. Stage mask-contained production was 13/26 at the inventory snapshot and is deliberately
not bound. The portfolio contains host-local absolute source paths, so another executor must stage
or reacquire byte-identical files before use. No convergence, quality, speed, or compression
outcome has been measured by this task.

#### Review Focus

Audit canonical JSON compatibility with StructSplat, additive/normalized semantic preservation,
factor family/A/A invariance, strict metric provenance, receipt-required assembly, confirmation
isolation, pinned archive authenticity, and whether the six acquisitions support the intended
capture-cluster interpretation.

#### Protected actions not taken

No formal protocol was frozen or executed, no confirmation field or downstream outcome was
opened, no correlation/report was produced, no scientific claim/default was changed, no active
RTGS-008 file was modified, and nothing was pushed.

#### Recommended Next Action

Commit this verified driver checkpoint, obtain a distinct implementation verdict, and only then
bind the accepted realtime-gs commit and portfolio digest into StructSplat BENCH-019. The next
implementation task should freeze the TUM/Karate adapters and produce matched development fields;
confirmation production must remain unopened until the upstream gate authorizes it.

### Review (provisional self-review)

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

Re-read the complete exporter/portfolio diff from `4c1a7a53` through `d3e76fe`, traced successful
and error-cell construction through receipt-required assembly, and checked exact-key validation,
source and artifact rehashing, family/A/A factor invariance, semantic labels, strict JSON parsing,
and append-only writes. No correctness defect was found in the committed task scope.

#### Evidence Quality

Re-executed all 18 focused tests and a full 88-file portfolio rehash from the committed revision.
The earlier canonical verify and complete CPU suite remain implementation evidence only; neither
establishes GPU behavior, capture independence, matched-field availability, or a downstream
scientific result.

#### Simplicity

The implementation keeps the exporter passive and CPU-only, shares canonical artifact helpers,
uses one explicit source manifest per cell, and leaves analysis/report authority in StructSplat.
No competing executor, report framework, CUDA import, or result registry was introduced.

#### Missing Cases

No distinct reviewer has reproduced the canonical-JSON compatibility or six-capture independence
judgment. Five capture groups lack matched fields, TUM and Karate adapters are unfrozen, and the
formal protocol and confirmation outcomes remain unopened.

#### Required Changes

No code change is required by this self-review. A distinct reviewer must still inspect and accept
exact commit `d3e76fe` before RTGS-009 can receive independent acceptance or become formal
BENCH-019 execution authority.

#### Optional Improvements

After independent acceptance, open a distinct RTGS task for deterministic TUM/Karate adapters and
the Stage-1 predictor collector. Keep field production development-only until its own protocol is
prospectively reviewed.

### Review (independent evidence-boundary review)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The canonical JSON bytes are identical to StructSplat's v1 implementation, the emitted row has the
exact upstream field set, additive and normalized semantic digests are preserved, and the shared
factor omits family and replicate as intended. The independent review nevertheless reproduced
three fail-closed defects in commit `d3e76fe44f3afe2044505fdcf1e7043657a4e4b1`:

- Formal export accepted a digest-consistent frozen-looking protocol with no repository bindings;
  StructSplat's `validate_protocol(..., require_frozen=True)` rejected the same object with
  `BENCH-019 must bind both StructSplat and realtime-gs repositories`. The local validator also
  omits the upstream analysis, path, minimum-seed, predictor/response-disjointness, and complete
  A/A contract checks. A formal flag therefore does not currently establish that the input is a
  valid StructSplat BENCH-019 protocol.
- Assembly accepted a modified export receipt after `downstream_metrics` was deleted from
  `source_artifacts`. `_validate_export_receipt` rehashes whichever descriptors remain but does not
  reload and reconcile the sealed source manifest, exact metric-source set, metric bindings, run
  binding, or source artifact descriptors to the row. Receipt presence is therefore not yet proof
  that the assembled row is the export of its named source manifest.
- Portfolio verification accepted replacing the `tum_fr1_xyz` official archive with an unrelated
  valid calibration file after its self-descriptor and `source_digest` were recomputed. The pinned
  official TUM bytes and SHA-256 values are enforced by `create` only, not by the reusable
  `validate_capture_portfolio` / `verify` path.

#### Evidence Quality

All 18 focused exporter/portfolio tests passed, all 88 committed source files rehashed, and the
complete CPU suite passed with only the two established PyTorch warnings. The committed portfolio
currently contains 88 unique paths, the declared 3+3 IDs are disjoint, all confirmation families
remain `not_produced`, and its four current TUM descriptors match the constants in the creator.
A synthetic formal row also produced zero StructSplat `validate_result_rows` problems. Those green
checks establish current row compatibility and current-file integrity, but the three negative
controls above show that the verification boundary can accept substituted provenance.

#### Simplicity

The passive module split, CPU-only imports, exact row schema, raw finite JSON-pointer extraction,
six-artifact binding, append-only writes, and upstream-owned analysis boundary are appropriately
small. The required repairs can stay within the existing validators and tests; no new executor,
report framework, or dependency is needed.

#### Missing Cases

The test suite lacks an exact upstream-valid frozen protocol fixture and paired negative corpus,
receipt source-set/source-manifest deletion and substitution cases, and TUM-pin verification
mutation cases. Capture IDs are source-disjoint in the committed inventory, but the two TUM groups
within each split do not by themselves establish statistical independence or domain diversity.
Five groups still lack matched fields, and no protected or confirmation outcome was accessed by
this review.

#### Required Changes

1. Make formal protocol acceptance fail closed on the complete StructSplat v1 protocol invariants,
   either by mirroring the portable validation rules locally or by verifying a suitably exact
   upstream validation artifact without importing StructSplat. Add an upstream-valid frozen
   fixture plus cross-repository positive and negative compatibility tests.
2. During assembly, strictly reload the receipt-bound source manifest and require the exact source
   artifact set, metric bindings and raw values, run binding, six cell artifacts, cell identity,
   and protocol/factor bindings to reproduce the row. Add deletion, substitution, extra-source,
   and source-manifest mutation tests.
3. Move the capture-specific official TUM archive identities into the reusable portfolio contract
   and enforce them in both create and verify. Add a test proving that a self-consistent archive
   substitution is rejected.

#### Optional Improvements

Order assembly-receipt `row_sources` by frozen cell order rather than caller order, and make the
standalone `factor` command reject undeclared frame, seed, or initializer values before downstream
work begins. Keep the portfolio's source-disjointness distinct from any later statistical
independence claim.

### Handoff (evidence-boundary repair)

#### Objective

Re-review RTGS-009 after closing the three independent-review blockers without widening the
exporter's passive scope or opening any protected outcome.

#### Reviewed state

Implementation revision `f582c58` on branch `rtgs/009-bench019-exporter`, following the independent
`Revision required` record in `35c107c`.

#### Changes

- Mirrored the complete portable StructSplat v1 formal invariants for repository bindings,
  downstream paths and seeds, predictor/response separation, analysis gates, and A/A tolerances;
  review-state diagnostics remain intentionally lightweight.
- Made assembly reopen the receipt-bound source manifest and reproduce its exact load-bearing
  source set, metric pointers and raw values, run binding, cell identity, factor, and six cell
  artifacts. Caller-order receipt records are now emitted in frozen cell order, and standalone
  factor construction rejects undeclared coordinates.
- Moved all four official TUM identities into the reusable portfolio module, so both creator and
  verifier enforce the same bytes, SHA-256, source kind, source ID, and official origin.
- Added positive formal export, source/receipt deletion/substitution/extra/cooperative-mutation,
  undeclared-factor, and TUM-substitution tests; updated the public contract description.

#### Evidence

- Focused exporter/portfolio suite: 26 passed.
- Committed portfolio verification: 3 development groups, 3 confirmation groups, 88 source files;
  every file rehashed and all official TUM pins matched.
- Cross-repository check: the exact frozen positive fixture passed both validators; the exact
  digest-consistent no-repository fixture failed both with `BENCH-019 must bind both StructSplat
  and realtime-gs repositories`.
- `PYTHONPATH=/home/alex/Documents/realtime-gs-bench019/src
  PY=/home/alex/Documents/realtime-gs/.venv/bin/python ./scripts/verify.sh`: passed.
- Complete CPU pytest suite with CUDA hidden: passed with only the two established warnings.
- `ruff`, formatting, docs sync, ARA, task workflow, experiment-contract, and `git diff --check`
  gates passed.

#### Assumptions

The portable v1 validation rules are mirrored locally because the passive exporter must not import
StructSplat. The cross-repository positive and negative fixtures guard semantic drift, while each
repository remains responsible for updating its copy deliberately if the versioned schema changes.

#### Uncertainties

This repair establishes evidence-chain integrity, not capture independence, field completeness,
predictor validity, reconstruction quality, convergence, speed, or compression. Five capture
groups still lack matched fields and the Stage mask-contained production remains incomplete.

#### Review Focus

Reproduce the three original counterexamples; compare the positive frozen fixture against the
actual StructSplat validator; audit unused-source rejection, error-row replay, and whether any
formal upstream invariant is still absent or materially stricter/looser.

#### Protected actions not taken

No formal BENCH-019 protocol was created or executed, no confirmation field or downstream outcome
was opened, no analysis/report/claim/default changed, and the dirty primary realtime-gs worktree
was not modified.

#### Recommended Next Action

If the revision is accepted, archive RTGS-009 and open RTGS-010 for deterministic source adapters
and a fail-closed Stage-1 predictor collector. Keep production development-only until its own
prospective protocol review.

### Acceptance-pass feedback (no second verdict recorded)

The bounded re-review at `1494676` reproduced all original counterexamples as closed, passed the 26
focused tests, portfolio rehash, error-row/frozen-order controls, and full verification. Before
acceptance it found two related parity omissions: formal identity did not rehash non-selected
capture/family artifacts, and local nonempty-string checks did not reject whitespace like the
upstream validator. The reviewer returned the clean tree without editing the task or recording a
second verdict so the driver can make one exact systematic repair and resume this same acceptance
pass.
