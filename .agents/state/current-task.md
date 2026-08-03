# Current Task

## Title

Provider-neutral StructSplat BENCH-019 downstream receipt exporter

## Task ID

RTGS-009

## Role Assignment

- Driver: Codex-bench019-exporter-driver
- Reviewer: pending-independent-reviewer
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
- No general BENCH-019 downstream row or correlation result exists.

## Minimal Plan

1. Completed: freeze the exporter/source/factor contracts against the StructSplat row validator.
2. Completed: implement the CPU-only modules, bounded CLIs, receipt-required assembler, and
   adversarial tests.
3. Completed: exercise synthetic-success and calibrated-error diagnostics and update public docs.
4. Completed: bind three development plus three confirmation source groups without fitting or
   accessing confirmation outcomes.
5. In progress: commit the verified driver revision, obtain independent implementation review,
   then bind the accepted revision from StructSplat BENCH-019 before any formal protocol review.

## Status

In progress

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

Verified working-tree implementation on branch `rtgs/009-bench019-exporter`, based on clean
realtime-gs commit `4c1a7a53`. The exact handoff commit is pending this record and the required
pre-commit verification gate.

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
