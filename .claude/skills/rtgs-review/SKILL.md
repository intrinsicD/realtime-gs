---
name: rtgs-review
description: Pre-commit self-review checklist for a realtime-gs diff. Use before every commit or PR, after a refactor, or when asked to review changes. Covers the five hard-rule gates (CPU-first imports, backend pluggability, test determinism, docs sync, benchmark/experiment logging), gradient-safety and numerical-stability review for differentiable code, and the claim-hygiene check that keeps prose in step with ara/logic/claims.md. Do not use to audit result numbers or promote a quantitative claim; use realtime-gs-results-audit for that.
---

# Review (pre-commit self-review)

Run this before `git commit` on anything non-trivial. It is cheaper than a failed
`./scripts/verify.sh` and much cheaper than a retracted claim.

`rtgs-verify` proves the tree is green. This skill asks whether the change is *right*.

## 0. Establish the diff

```bash
git status --short
git diff                      # unstaged
git diff --cached             # staged
git diff --stat origin/main...HEAD
```

Review the whole diff, not just the file you were last editing. If the diff has grown
beyond the task you started, split it — mechanical moves and semantic edits do not belong
in one commit.

## 1. Hard-rule gates (CLAUDE.md)

- **CPU-first imports.** No new module-level `import gsplat`, `import transformers`, or other
  CUDA-only dependency. Confirm with `grep -n "^import\|^from" <changed files>`; GPU imports
  live inside functions, behind a guard, with a CPU fallback or a clear raise.
- **Backend pluggability.** New rasterizers implement `rtgs.render.base.Rasterizer`, sparse
  point renderers `rtgs.render.point_base.PointRasterizer`, depth estimators
  `rtgs.depth.base.DepthBackend`. No `if backend == "gsplat"` branching in pipeline code.
- **Test determinism.** New tests seed RNGs via the `tests/conftest.py` helpers. Quality
  thresholds are floors, not snapshots — a lowered threshold needs a dated
  `docs/EXPERIMENTS.md` entry justifying it, in the same commit.
- **Docs sync.** New/removed subpackage, CLI command, lifter, or skill → `docs/ARCHITECTURE.md`
  and the `CLAUDE.md` map updated in the same commit.
- **Benchmarks and experiments.** A performance claim cites `benchmarks/run.py` output; a
  research finding (including a negative one) has a dated `docs/EXPERIMENTS.md` entry.

## 2. Differentiable-code review

For anything touching `render/`, `image2gs/`, `lift/`, or `optim/`:

- **Gradient safety.** No in-place mutation of a tensor that requires grad; no `.data`;
  `detach()` is deliberate and commented where it changes what learns.
- **Numerical stability.** Divisions and `sqrt`/`log` have an epsilon or a positivity
  guarantee. Normalizing rasterizer weights, exponentials, and covariance inversions are the
  usual suspects. Check behavior at zero opacity and at a degenerate covariance.
- **Reference parity.** `rtgs.render.torch_ref` is the correctness anchor. A new fast path
  ships a parity test against it; a CUDA path ships `@pytest.mark.cuda` plus a CPU-reference
  counterpart where feasible.
- **Shape and dtype.** Batch dims flow through unchanged; no silent float64 promotion; device
  is taken from the input, never hardcoded.
- **Cost.** Test scenes stay tiny (≤64×64, ≤300 gaussians, ≤200 iters) and the suite stays
  under ~3 minutes on a 4-core box.

## 3. Claim hygiene

Prose is a claim surface. If the diff adds a number or a capability statement to `README.md`,
`docs/`, or a docstring:

- Is it bound to a row in `ara/logic/claims.md` with a `Proof` entry that exists on disk?
- Is the wording no stronger than the evidence class — synthetic vs calibrated, CPU vs GPU,
  development vs confirmatory, initialization vs refinement?
- Does a changed default cite the experiment entry that justifies it?

`python scripts/check_ara.py` checks the structure. It cannot check whether the sentence
overstates the artifact — that is this step, and `realtime-gs-results-audit` for anything
promoted.

## 4. Scope and hygiene

- Commit message says what changed and why, and names the experiment/claim ID when relevant.
- No stray debug prints, commented-out code, `TODO` without an owner, or scratch files.
- No new one-off script at the top level of `scripts/` — those belong in
  `scripts/experiments/` (see `scripts/experiments/README.md`).
- No results-bearing artifact overwritten. Official JSON, seals, receipts, and audit notes in
  `benchmarks/results/` are append-only.
- `git diff --check` is clean (no whitespace errors, no conflict markers).

## 5. Gate

```bash
./scripts/verify.sh
```

Then, if the change is substantial and touched `src/rtgs`, run the slow suite once:

```bash
.venv/bin/pytest -q
```

If the change is results-bearing, hand off to `realtime-gs-results-audit` before the claim
enters public prose.
