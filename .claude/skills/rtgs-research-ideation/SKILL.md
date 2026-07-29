---
name: rtgs-research-ideation
description: Generate and adversarially audit falsifiable, potentially publishable research directions for realtime-gs. Use for open-ended 2D-to-3D Gaussian lifting ideas, novel representations or objectives, cross-domain mechanism transfer, new evidence programs, high-risk/high-reward research portfolios, or requests to find directions beyond incremental component swaps. Do not use for routine feature brainstorming or direct implementation of an already selected method.
---

# Research ideation

Produce a diverse falsifiable portfolio, not a list of fashionable components. This skill proposes
and audits candidates; it does not modify production code, execute experiments, or create roadmap
commitments without a selected candidate and normal task authorization.

Read `CLAUDE.md` first. Then read:

- `references/repository-context.md` to build a repository-specific frontier;
- `references/novelty-and-transfer.md` before classifying novelty or importing a donor mechanism;
- `references/prior-art-and-evidence.md` before making a novelty statement or proposing evidence;
- `assets/research-portfolio.md` when producing the final deliverable.

Verify every repository summary against the current checkout.

## 1. Map the frontier

Inspect `docs/RESEARCH.md`, `docs/ROADMAP.md`, `docs/EXPERIMENTS.md`, relevant ADR/design notes,
current implementations, tests, experiment contracts, and negative results. Record for each
important method:

- primitives and representation;
- assumptions and information boundary;
- objective and core operator;
- training/inference procedure;
- strongest evidence class;
- known failure, confound, or unresolved anomaly.

Do not generate final ideas until this map distinguishes dense RGB/depth paths, compact-only paths,
initialization, fixed-topology refinement, adaptive density, and evaluation-only inputs.

## 2. Write a functional problem signature

Remove 3DGS vocabulary. State what enters, what hidden state is inferred, what is transported or
conserved, which variables are local/global and continuous/discrete, which symmetries or gauge
freedoms exist, and what limits identifiability, stability, quality, memory, or speed.

Use this signature—not superficial terminology—to search donor fields.

## 3. Build an anti-library

List default suggestions that do not count as research contributions on their own: another loss
term, attention, a larger network, generic uncertainty weighting, multiscale processing, more
iterations, or an unscoped combination of existing lifters. A candidate using one must retain a
specific irreducible scientific claim after subtracting it.

## 4. Generate independent lanes

Keep lanes separate through the first pass:

1. **Productive recombination:** at least three bounded combinations that repair an observed
   failure using existing components.
2. **Assumption surgery:** at least three candidates that remove, reverse, localize, infer, or
   explicitly model failure of a shared assumption.
3. **Primitive/grammar invention:** at least three candidates, including two new observables,
   operators, equivalence classes, measures, or implicit problem statements.
4. **New-evidence programs:** at least two investigations that can reveal a phenomenon absent from
   current artifacts, including negative controls and leakage checks.
5. **Cross-domain transfer:** at least four mechanisms from at least three donor fields, including
   two fields not routinely paired with Gaussian splatting and one transfer of measurement or
   experimental practice rather than an algorithm.

For each transfer state the donor mechanism, recipient map, preserved causal structure, broken
correspondence, required invention, adoption barrier, and recipient-specific prediction.

## 5. Attack transformation and prior art

Apply the tests in `references/novelty-and-transfer.md`. Switch to an adversarial evaluator role
and search primary sources by exact terms, synonyms, mathematical form, older terminology,
functional description, repositories, theses, patents, donor fields, and bridge fields.

Record sources searched and the cutoff date. Use only calibrated language:

- "apparently unexplored under the stated search";
- "known components with a possibly new relationship";
- "strongest prior-art threat";
- "insufficient evidence".

Never claim absolute novelty or guaranteed publication.

## 6. Score without premature collapse

Score 0–5 independently for apparent novelty, falsifiability, explanatory value, scientific
importance, feasibility, first-test cost, interpretability, baseline strength, value of a negative
result, and publication potential.

Keep a Pareto set containing the fastest kill, strongest theory direction, strongest systems
direction, highest-risk/high-reward direction, and most useful negative-result direction when
possible.

## 7. Design the cheapest killing test

For every surviving high-novelty candidate specify:

- null hypothesis;
- predicted signature if correct;
- signature under the strongest conventional explanation;
- smallest implementation, proof, or diagnostic;
- required data, compute, and baseline;
- decisive metric/plot/lemma;
- confounders and negative controls;
- abandonment rule;
- highest evidence maturity the test can reach.

Prefer a small mechanism screen that can kill the claim over a production implementation.
Synthetic evidence may open a branch; it cannot close a calibrated/default/generalization claim.

## 8. Deliver and route

Use `assets/research-portfolio.md`. Include productive, exploratory, transformational,
cross-domain, and new-evidence candidates; prior-art threats; and one recommended first killing
experiment.

When the user selects a candidate:

1. use `rtgs-task-workflow` to record scope and maturity;
2. add an ADR only for a hard-to-reverse, non-obvious trade-off;
3. create the immutable experiment contract before result-bearing execution;
4. use `rtgs-experiment` or `rtgs-bench`;
5. run `realtime-gs-results-audit` before claim/default promotion.

For a saved Markdown portfolio, run:

```bash
python .claude/skills/rtgs-research-ideation/scripts/validate_portfolio.py <portfolio.md>
```

The validator checks structure and calibrated language, not scientific novelty.
