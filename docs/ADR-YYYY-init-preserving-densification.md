# ADR-YYYY — Init-Preserving Densification and Trust Schedule

**Status:** proposed
**Date:** 2026-07-27
**Implements:** PREREG `20260725_init_value_program` E9 (split-inheritance repair) and the concrete
levels of the E12 factor grid (S1–S5)
**Depends on:** ADR-XXXX (surfel lift) — operators below assume the `(q, log s)` surfel frame and
the appearance-matched α from that ADR; M3 lineage tracking must be present.
**Renumber:** replace YYYY with the next free ADR number on merge.

---

## Context — four mechanisms by which vanilla ADC destroys a good init

1. **Split samples children from N(μ, Σ).** Position information is replaced by a draw from the
   parent's own uncertainty. For the old CI covariance this scattered children along the
   unresolved ray axis (E3's predicted failure); even with correct surfel covariance, sampling is
   noise where a deterministic placement is available.
2. **Clone and split are not appearance-preserving.** A clone at the same position with copied α
   changes the composite from α to 2α−α² at the peak; a split copies α to both children and
   divides scales by the ad-hoc 1.6. Every densification event therefore *perturbs the render*,
   and the iterations that follow are repair, not progress. With a cold init this is invisible —
   everything is repair anyway. With a warm init it is exactly how the warm start leaks away, one
   event at a time.
3. **The opacity reset erases the α-solve.** Clamping all α to ≤0.01 every 3k iterations is an
   escape mechanism for bad configurations. Applied to an appearance-matched init it deletes, by
   construction, the one parameter block ADR-XXXX §3 computed.
4. **The gradient criterion conflates need with conflict, and the LR schedule assumes wrong
   positions.** Accumulated view-space position gradients fire on large, correctly-placed
   primitives with residual color error; the high initial position LR exists to move wrong means
   and is a destroyer of right ones.

**Design invariant for everything below:** *every topology operation either preserves the rendered
image (within a tested tolerance) or is explicitly an insertion of new capacity into a region the
current model does not explain.* Growth changes capacity, never the current solution.

---

## Decision

### §1 Three growth channels with distinct semantics and budgets

| Channel | Trigger (all error-driven, §4) | Placement | Lineage |
| --- | --- | --- | --- |
| **CLONE** — extend a surface | structured residual at the *rim* of a primitive's footprint (coverage gap in the tangent plane) | deterministic tangential step toward the residual | inherits `root_id` and full surfel frame |
| **SPLIT** — refine a surface | structured residual *inside* the footprint (detail finer than the primitive) | deterministic ± offsets along the in-plane residual axis | both children inherit `root_id` and frame |
| **INSERT** — new surface | residual mass in a region with transmittance ≈ 1 along the ray (nothing there to clone from) | residual importance sampling; depth from median transmittance | `NULL_ROOT` |

Budgets are **per channel**, not pooled. This is what lets a good init express its value: an
accurate init consumes its clone/split budget and leaves the insert budget idle; a bad one starves
inheritance and forces insertion. The channel usage ratio is itself a diagnostic (feeds E2/E3
attribution) and must be logged per densification round.

### §2 Appearance-preserving CLONE

Parent `(μ, q, s, α, sh)`, tangent frame `(t₁, t₂, n)` from `q`, residual direction `d̂ ∈ span(t₁,t₂)`
(unit, toward the rim residual):

```
μ_child   = μ ± δ·d̂            δ = 0.5·σ_d   (σ_d = parent sigma along d̂; parent keeps μ)
q_child   = q                   (frame inherited — this, not the mean, is the surfel inheritance)
σ_n,child = σ_n                 (the ρ prior is inherited, never re-derived from the child's own
                                 neighborhood, which would be contaminated by its parent)
σ_t: both parent and child shrink along d̂ to preserve the second moment of the pair:
      σ_d' = sqrt(max(σ_d² − (δ/2)², (0.3σ_d)²))        [floor prevents needle collapse]
α:    α' = 1 − sqrt(1 − α)      for both               [peak-composite invariance for the
                                                         coincident-pair limit; exact at the peak,
                                                         approximate under the tangential offset —
                                                         tolerance tested, A1]
sh:   copied
```

Rejected alternative: vanilla clone at the identical position with copied α. Coincident twins have
identical gradients and separate only by numerical noise, while locally doubling opacity — the
subsequent "repair" is what the opacity reset then papers over.

### §3 Appearance-preserving SPLIT

In-plane residual axis `d̂`, parent replaced by two children:

```
μ_{1,2} = μ ± d·d̂               d = 0.5·σ_d, deterministic — never sampled from N(μ, Σ)
σ_d,child = sqrt(σ_d² − d²)      = 0.866·σ_d for d = 0.5σ_d   [mixture second-moment preservation;
                                  replaces the ad-hoc /1.6, which over-shrinks and, combined with
                                  sampling, is mechanism 1+2 above]
other axes, q, σ_n: inherited unchanged
α:    α' = 1 − (1 − α)^(1/2)     per child (N-child generalization: 1 − (1−α)^(1/N))
sh:   copied
```

For `NULL_ROOT` primitives (no trusted frame), the residual axis falls back to the largest
covariance axis and the same formulas apply — the operators are global; only the *frame trust*
distinguishes lineages. There is one growth code path, not two.

**Honesty note:** exact appearance invariance for offset children is not closed-form. The formulas
above are exact at the peak in the coincident limit and second-moment-exact in the offset; the
residual event error is bounded by acceptance test A1 rather than assumed away.

### §4 Error-driven trigger (replaces the gradient-magnitude criterion)

Per densification round, per tile: accumulate |residual| over the recent window (held views never
included). A primitive is a CLONE/SPLIT candidate iff structured residual (tile residual above the
E0-derived noise floor, spatially clustered — the PREREG E4/E12 "structured" test) overlaps its
footprint; rim vs. interior overlap selects the channel (§1). INSERT triggers on structured
residual tiles whose accumulated transmittance along corresponding rays exceeds 0.9 (nothing
present to grow).

The vanilla gradient criterion is retained **only** as an E12 S3 level for comparison. It is not
the default of the init-trusting profile.

### §5 Trust schedule (E12 S4/S5 made concrete)

Per-primitive LR multipliers, all as **smooth ramps, never steps** (the S5 "watch for"):

```
trust(t) = β + (1 − β) · min(1, t / T_trust)         β = 0.3, T_trust = 2000
applies to init-rooted primitives only (root_id ≠ NULL_ROOT); children inherit the CURRENT
trust state of their parent at creation time, not a fresh β.
```

| Parameter | init-rooted LR multiplier | Rationale |
| --- | --- | --- |
| position | trust(t) | means are the measured asset |
| σ_t1, σ_t2 (tangential log-scales) | trust(t) | tangential scales are *measured* (ADR-XXXX §2.1) |
| σ_n (normal log-scale) | **1 (no trust)** | σ_n is a deliberate prior, not a measurement — it must be free to correct itself immediately; trusting it would defend our own guess against the data |
| rotation q | ĉ·trust(t) + (1−ĉ)·1 | trust the frame only where the PCA planarity ĉ justified it |
| α, SH | 1 | appearance parameters carry the photometric fit; throttling them starves the only error signal the trusted geometry receives |

**Not a freeze.** β > 0 always; a wrong init must remain overridable by the data. If E2 shows
trusted primitives migrating anyway (median normalized displacement > 3 despite trust), that is
evidence *against the init*, and the correct response is ADR-XXXX diagnostics — not more trust.

### §6 Opacity reset → selective relocation

The global reset is removed from the init-trusting profile. Its two legitimate functions are
replaced surgically:

- **Dead-weight recycling:** primitives with α < 0.005 for 3 consecutive checks are *relocated*
  (MCMC-style) to INSERT-channel targets, with `root_id ← NULL_ROOT` (relocation is destruction
  plus creation; M3 rule). This consumes INSERT budget.
- **Floater escape:** floaters are primitives with high α and near-zero structured-residual
  overlap support across views; they are pruned directly rather than reset-and-hoped-away.

The global reset is retained as E12 S1 level "on" for the comparison arms only.

### §7 Warmup

With an appearance-matched init the residual is meaningful at iteration 0, so the error-driven
trigger (§4) needs only its accumulation window (~100–200 iterations), not the vanilla 500-plus
warmup. The vanilla warmup remains as the E12 S2 comparison level. For cold inits the longer
warmup stays the sensible default — this profile is *init-conditioned*, which is the entire E12
thesis.

---

## Relationship to the PREREG (binding)

This ADR **implements levels; it does not select them.** The init-trusting profile
(§2+§3 operators, §4 trigger, §5 trust, §6 no-reset, §7 short window) is a named *candidate
configuration* for P-adapt. Whether it wins — per arm, including random — is decided by E12's
tuning under §1.5 fairness rules, and the interaction contrasts remain the experiment's output.
Hard rules:

1. No factor outside the E12 grid may be introduced here or during implementation. New knobs
   require a PREREG amendment first.
2. The E12 comparison levels (vanilla clone/split, gradient trigger, reset on, full warmup) must
   remain implemented and selectable; deleting the baseline is the classic way to make the new
   thing unbeatable.
3. §1's channel budgets are set by the E4 cap machinery (M4); this ADR adds the per-channel split
   but the total remains the experiment's variable.

Mapping: §2/§3 = E9 variants 2+3 (merged: tangential constraint + appearance-preserving split);
§4 = E12 S3 "error-driven"; §5 = E12 S4/S5; §6 = E12 S1 "off" + E9 variant 4 (relocation);
§7 = E12 S2.

---

## Acceptance criteria

**A1 — Event invariance.** Unit test: on a fixed scene state, apply each operator (clone, split)
to a batch of primitives; PSNR between pre- and post-event renders of affected tiles ≥ 40 dB, no
event with local Δ > 1 dB. Vanilla operators are run through the same harness and their (large)
event error is recorded as the baseline the test is protecting against.

**A2 — Second-moment preservation.** Numerical check of §2/§3 scale formulas against the analytic
mixture moment, per event, tolerance 1e-4 in log-scale.

**A3 — Lineage integrity.** Property tests against M3: clone/split children carry the parent's
`root_id`; relocation assigns `NULL_ROOT`; channel budgets never exceeded; channel usage logged.

**A4 — Degeneration guards.** With trust enabled and a *deliberately corrupted* init (E4b harness,
σ = 2), the run must still reach within the E0 noise band of the random-init run — i.e. trust must
not be able to defend a wrong init to the end. This is the anti-freeze test and it reuses M5.

**A5 — Profile isolation.** With all §-features disabled, the trainer is bit-identical to the
incumbent pipeline (same seeds → same states). The comparison arms of E12 must literally be the
old code path, not a re-implementation of it.

## Consequences

- Densification becomes a set of image-preserving rewrites plus one explicit insertion channel;
  a warm start can no longer be destroyed silently, only spent visibly (channel logs).
- The surfel frame — not just the mean — is what inheritance propagates; this is the mechanism by
  which the ADR-XXXX construction "keeps playing" through the whole run, which was the question.
- The trust asymmetry (σ_n untrusted, tangentials trusted) encodes which parts of the init are
  measurements and which are priors; blanket trust would defend our own assumptions against data.
- Cost: the error-driven trigger needs per-tile residual accumulation (memory: one float per tile
  per window; negligible next to the compact-capture savings) and the NNLS-style bookkeeping at
  events is O(children), trivial.
- Risk accepted: A1's 40 dB bound is a heuristic tolerance; if E12's H-S1/H-S2 effects are smaller
  than the accumulated event error allows detecting, tighten the bound before blaming the
  hypothesis.
