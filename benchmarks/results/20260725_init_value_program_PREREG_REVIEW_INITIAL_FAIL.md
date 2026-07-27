# Initialization Value Program preregistration — independent review (initial FAIL)

Reviewed on 2026-07-25 as a non-author review.

Reviewed preregistration:
`benchmarks/results/20260725_init_value_program_PREREG.md`, SHA-256
`2336b6362b0602a488fc10ed74b526eba91f8fd05a19db621e6cc42a1cef2408`.

Verdict: **FAIL PENDING PROTOCOL REPAIR**.

The program asks the right high-level question and its decomposition is useful, but the current
document is not yet a claim-safe preregistration. The headroom arithmetic is wrong for a minimizing
metric, the proposed `Omega` synthesis contains contradictory directional predictions, held-out
outcomes are reused for selection, and the primary "speed" gate measures iterations despite local
pilot evidence that fewer iterations can take more time. These are decision-changing defects, not
editorial ambiguities. No program experiment, seal, default change, or confirmatory claim is
authorized by this review.

## Scope and chronology

I read the complete preregistration, `CLAUDE.md`, the repository results-audit procedure, the
linked same-day cost-to-target and refit-ceiling preregistrations and harnesses, their local summary
artifacts, the shared train/held-out scene builder, the prior 2026-07-21 and 2026-07-23 result
dispositions, and the relevant ARA claim boundary. I did not run a scientific arm, construct a new
initialization, open a new scene, train, render, modify the reviewed preregistration, or inspect an
unopened/sealed outcome.

This review is not outcome-blind with respect to the same-day pilot work. The umbrella protocol was
first committed at `127f471` and amended at `7d62f50`, after closely related local cost-to-target
and refit-ceiling artifacts existed. More importantly, the refit preregistration says it was frozen
before the cost-to-target run was read, but later in that same frozen file states the observed
500-versus-750-step and 14.5-versus-10.6-second result. Its hash in
`runs/gpu_init_refit_ceiling/summary.json` matches the current file. This does not prevent a
prospective validation on untouched scenes and seeds, but the umbrella document must be described
as **pilot-informed**, enumerate every outcome already seen, and exclude those pilot cells from
confirmatory evidence. The current "zero data existed" wording is too broad to support a blindness
claim.

## Blocking findings

### 1. `H` and `F` have the wrong direction for `T@tau`

Sections 1.4 and E0b define, for every metric,

`H = metric(oracle) - metric(random)` and
`F = (metric(arm) - metric(random)) / H`.

That is correct only for higher-is-better metrics. For `T@tau`, lower is better, so a useful oracle
produces a negative `H`; the gating rule would then classify the strongest speed headroom as no
positive headroom. The direction-normalized definitions must be frozen separately:

- quality: `H_q = q_oracle - q_random`,
  `F_q = (q_arm - q_random) / H_q`;
- cost: `H_t = t_random - t_oracle`,
  `F_t = (t_random - t_arm) / H_t`.

Unreached targets are censored observations and cannot be substituted into these equations as
ordinary finite values. The protocol needs a frozen censoring rule or a restricted-time/AUC
estimand.

The document also calls the constructed oracle both the "maximum advantage" and a lower bound from
one optimizer-fixed-point construction, and explicitly permits `F > 1`. Those statements cannot
all be true. Rename this quantity a **reference-oracle gap** and `F` a **reference-gap ratio**.
Failure of this one oracle to beat random may stop work as a resource decision, but it cannot
support the universal claim "no initialization can matter."

### 2. `Omega` cannot currently be the governing variable

With

`Omega = model parameters / observed RGB scalars`,

three interventions move `Omega` in ways that conflict with the stated predictions:

- increasing the Gaussian cap raises `Omega`, while E4 predicts initialization value falls;
- decreasing the number of views raises `Omega`, while E5 predicts initialization value rises;
- reducing resolution raises `Omega`, while the downscale-16 motivation predicts a null.

E0b additionally labels "headroom small at high `Omega`, large at low `Omega`" as expected under
H-P2, while E5 predicts the largest gap at three views, which has higher `Omega` than the
corresponding 26-view cell. A pooled collapse cannot be a confirmatory central claim under these
opposed directions.

Keep budget, angular coverage, resolution, and optimization budget as separate causal factors.
Treat an `Omega` plot as exploratory unless the preregistration freezes a model, direction,
collapse statistic, acceptance threshold, and held-out predictive check. At minimum, use a metric
that includes time-varying population and mask-visible observations rather than nominal maximum
parameters and full-canvas pixels.

### 3. Held-out outcomes are used for selection and then reused for claims

The standing contract says held-out views are reporting-only, but later stages:

- populate full-arm cells only when held-out oracle headroom passes;
- choose the "best structured initializer";
- choose the two best E4 budget cells for E5 and E6;
- set `tau` from a held-out reference plateau;
- form the final pooled claim from the same selected E4/E5/E6 outcomes.

Predeclaring an adaptive rule makes the adaptation auditable; it does not remove selection bias or
turn the same held-out set back into untouched evidence. Add a three-way role split:

1. training views for fitting, lifting, refitting, and optimization;
2. validation views/scenes for oracle gating, arm/cell selection, and target calibration;
3. a once-opened test set for the locked confirmatory comparison.

Alternatively label E0b–E6 as development only and repeat one fully frozen arm/cell comparison on
new scenes. Exact scene IDs, camera IDs, seed lists, and exclusions must be frozen; "`>= 3`" leaves
outcome-dependent room to add, drop, or headline cells. Optional `top-K` and "best" arms are not
confirmatory definitions.

### 4. The noise-floor and cross-scene decision rules are not defined statistically

E0 measures an undefined "spread" for Beam Fusion on one scene and two schedules, then applies
twice that quantity to every arm, scene, resolution, view count, and metric. Range, standard
deviation, MAD, and a paired-difference interval have different meanings; seed variance also
changes with cell and controller. `T@tau` is discrete and may be censored.

Freeze:

- exact paired seeds shared across arms;
- the dispersion/interval estimator and its unit for every primary metric;
- whether decisions use ratios of seed medians or paired per-seed contrasts;
- the scene-level aggregation rule and required direction of replication;
- multiplicity handling for arms, caps, two co-primaries, scenes, and target levels;
- a sample-size or precision justification.

Reporting median/min/max is descriptive and does not supply these decision semantics. "Replicated
on at least three scenes" also needs a rule: all scenes, a majority, or a hierarchical lower bound
can yield different verdicts.

### 5. Iterations-to-target is not a speed estimand

E4 labels `T@tau` the speed axis and uses it for the "skip the cold start" core claim. Initial
scale, footprint, population trajectory, and controller can materially change per-step cost. The
already-produced local pilot demonstrates the failure directly:
`cover-iso` reached 21 dB in 500 steps versus `ci` in 750, but required 14.49 seconds versus
10.60 seconds in the primary density cell. A step win was a time loss.

The confirmatory cost outcome must include:

- post-initialization optimizer GPU/wall time to target;
- end-to-end time to target including Stage 1, lift, covariance/appearance repair, and any refit;
- iterations-to-target as a mechanism diagnostic;
- interleaved A/B order, warmup, synchronized timing, fixed hardware/software state, and repeated
timings.

If Stage 1 is intentionally amortized, report both end-to-end and post-Stage-1 costs and state the
amortization scenario. E11 throughput does not repair a core speed gate scored only in iterations.

The `T@tau` definition also needs repair: "stays within 0.1 dB of it for 200 iterations" could
reject an arm for improving more than 0.1 dB above `tau`, and cannot be observed when evaluation
cadence exceeds 200 steps. Define it as "does not fall below `tau - 0.1 dB` during the next
specified checkpoints," freeze the cadence, and define end-of-run censoring.

### 6. Input isolation and the COLMAP comparison are not fail-closed

The linked scene builder correctly restricts Beam Fusion inputs to explicit training views, but
the umbrella protocol does not freeze that allowlist or require input hashes for every lifter.
M1 creates Stage-1 caches for every view, which makes accidental held-out use especially easy.
Every initializer, refit, density-selection step, threshold, and source hash must be checked against
the train allowlist before held-out release.

`colmap-sfm` is underspecified. Reusing sparse points from a reconstruction made with all images
would pass held-out RGB information into an initializer even if the camera poses are accepted as
calibration infrastructure. Build the comparison points from training images only, with poses
fixed if necessary, and freeze COLMAP settings, version, point filters, and minimum usable count.
A degenerate arm is evidence about that frozen baseline in that cell, not automatically proof of
the general H-P1 coverage claim.

E1c's dense COLMAP/MVS reference shares texture and matching failures with the COLMAP arm and with
the hypothesis under test. It cannot independently establish coverage precisely where SfM is
empty. Restrict strong coverage claims to synthetic or genuinely independent depth/scan reference,
or report real MVS agreement as a bounded diagnostic stratified by reference confidence and image
texture.

### 7. `Delta LE` and the E1b substitutions do not identify the claimed mechanism

`P_2D` is a useful input-fidelity baseline, but it is not the ceiling of the 3D pipeline: multi-view
optimization can outperform a per-view Stage-1 approximation, and independently fitted 2D views
need not be jointly realizable by one 3D state. Therefore a large `Delta LE` establishes an
iteration-0 render mismatch, not that the optimizer receives "no warm start." The downstream E1b
logic already acknowledges that these can invert; E1's interpretation table must use the same
narrow wording.

The GT/lift component substitutions are not executable or causal until the protocol defines:

- one-to-one topology and correspondence between GT and lifted primitives;
- how GT scales, rotations, opacity, and SH are constructed at lifted means;
- how count mismatches and unmatched primitives are handled;
- whether the estimand is a full `means x packaging` factorial or an order-dependent cumulative
  ladder.

Packaging is position-dependent, so copying it between unmatched means can create an invalid state.
The two crossover arms cannot support "the whole loss" language in the presence of interactions.

### 8. Optimizer utility and physical covariance validity must remain separate claims

It is reasonable to retest the wide track-LSQ state as an optimizer preconditioner under an
SPD-by-construction coverage parameterization. It is not valid to say that SPD parameterization
retires the prior physical-validity finding. The prior audit found both non-SPD matrices and a very
large whitened reprojection residual, and the repository has a supported rank result that two
generic EWA views leave one covariance coordinate unobservable.

Preserve two dispositions:

- **utility**: a nonphysical coverage prior may pass if it improves held-out cost/quality;
- **geometry**: it is not a recovered physical covariance unless it passes the independent
  reprojection/observability criteria.

If the utility arm wins, name it a coverage or surfel prior. Do not supersede the physical
covariance rejection merely because downstream optimization benefits.

### 9. The program is not frozen tightly enough to be executable as one preregistration

The document does not bind exact scenes, seeds, full resolved configs, code/harness hashes,
commands, target-evaluation cadence, random-arm construction, pairing, aggregation, or a
machine-readable decision function. M0 records a Git SHA after execution, but a dirty run remains
non-replayable unless the exact source tree is archived and hashed.

The cost estimate is also not an upper bound. Six arms times five caps times three scenes times
three seeds is 270 E4 runs before the required five-seed cells, absolute-count normalization, E0b,
E5, E6, diagnostics, and full-length reruns. E0b is described as a one-arm sweep but requires both
oracle and random over a larger grid. Freeze a feasible cell table and run budget before opening
outcomes.

M4 contains a smaller logical conflict: with cap equal to the initial count, pruning can make the
final count lower; maintaining exactly the initial count requires replacement births and hence
churn. Specify either fixed topology, prune-without-refill, or capped prune-and-replace semantics
and test the selected invariant.

## What should be retained

The following design choices are strong and should survive the repair:

- separating available reference gap, init-render delivery, and downstream regime;
- requiring held-out evaluation, multiple scenes/seeds, negative-result dispositions, and an
  independent audit;
- adding random and train-only COLMAP baselines and reporting absolute matched-count comparisons;
- gating covariance/appearance/density repairs on diagnostics rather than bundling them;
- reporting compactness separately from convergence cost;
- preserving artifacts, viewer handoff, and machine-readable raw evidence.

## Minimal route to a PASS

1. Recast E1/E1b/E1c/E2/E3 and the existing same-day runs as pilot-informed diagnostics.
2. Use validation scenes/views for E0b and a small factorial screen over exact low/high view count,
   exact low/high absolute cap, and exact resolution; do not claim on that screen.
3. Freeze one primary structured arm, random and train-only COLMAP controls, one or two cells, exact
   scenes and paired seeds, and a once-opened test set.
4. Make synchronized end-to-end seconds-to-target and held-out quality at an exact absolute count
   the confirmatory co-primaries; keep steps, `Delta LE`, lineage, geometry, and `Omega` as mechanism
   diagnostics.
5. Direction-normalize the oracle-reference gap, define censoring and paired inference, and rename
   it so no universal-ceiling claim follows from a lower-bound construction.
6. Add a machine-readable decision function and a fresh independent PASS review before any new
   confirmatory outcome is opened.

This is a repairable protocol, but it is currently a thoughtful research roadmap rather than a
sealed confirmatory preregistration.

## Checks performed

Read-only checks included `git log`, `git status`, `git diff --check`, SHA-256 verification, text
searches over `ara/`, `docs/`, `benchmarks/results/`, the linked harnesses, and direct inspection of
the local JSON decision summaries. No CPU/GPU verification suite or scientific replay was run
because this was a document/protocol review, not an implementation or results audit. Existing
untracked user files were not modified.
