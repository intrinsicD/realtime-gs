# Compact residual-responsibility birth allocation iter2 — preregistration

Date frozen: 2026-07-17

Status: **FROZEN, NOT YET IMPLEMENTATION-REVIEWED, NOT SEALED, NOT RUN**

## Chronology and reason for the fresh namespace

This is an outcome-neutral lifecycle restart of the matched-count parent-allocation experiment.
The preceding namespace was closed before sealing because an implementation test consumed its
official split root `77201` in a real generator before the Phase-B marker. The failure is bound by:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_FAILURE_AUDIT.md
SHA-256 5524a274937502587a3e41a0ecffd12ba66c2cf4aaa1a853874cb99e230f8044
```

No Phase-A score, selection, bank, arm, utility metric, or quality result existed in the failed
namespace. This iter2 restart changes no scientific question, arm, budget, score, topology rule,
metric, threshold, gate, input, teacher, initialization, optimizer, renderer, checkpoint, claim
boundary, or stopping rule. It changes the lifecycle namespace and every random root, and adds
fail-closed evidence requirements discovered without observing an outcome.

Immediately before this file was created:

- repository HEAD was `2dddca4aff59702341af9faceefa76ad2505dd83`;
- exact-word repository-wide searches found zero occurrence of every fresh root below;
- none of the fresh roots had been passed to a generator, scheduler, sampler, trainer, bank,
  score, selection, split, evaluator, seal, attempt marker, or result; and
- every iter2 lifecycle and run path named below was absent.

The failed namespace is historical evidence only and can never be reopened or pooled with iter2.

## Normative scientific protocol imported unchanged

The complete scientific protocol is imported by reference from:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_PREREG.md
SHA-256 e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427
```

The following sections and their formulas, arithmetic, order, gates, and claim limits are
normative and unchanged:

- question, `G` / `R` / `U` arms, and matched `16 clone + 16 split` count;
- literature grounding and adaptation boundary;
- frozen full-resolution compact teacher, center-occupancy proposal, 835-Gaussian initialization,
  and RGB-free authority boundary;
- the 140-step iter3-D compact configuration and six independent Adam groups;
- the literal front-to-back compositor and one-VJP residual/support responsibility seam;
- exact `L_t`, `R`, `S`, `G`, equal-view arithmetic, and assigned-residual fraction;
- eligibility, four matched strata, eight parents per stratum, and shuffled complete rank labels;
- one-wave clone/split arithmetic, raw split draws, revised opacity, persistent IDs, and optimizer
  surgery;
- exact step/checkpoint order and the common `35_pre` / arm-specific `35_post` distinction;
- Phase-A mechanism gates and independent Phase-A audit requirement;
- Phase-B replay, paired samples, fixed banks, `J_Q`, `J_U`, recovery log-AUC, and evaluation points;
- both comparator primary/safety gates and the ordered terminal decision map;
- structural, finite-value, count, state, round-trip, complexity, and accounting invariants;
- post-result native-resolution gsplat visualization and live viewer handoff; and
- interpretation and stopping rules.

The append-only executability amendment in that file is also imported for exact atom encoding,
ordered terminal decision precedence, and the elementwise final-PLY tolerance. Where a path,
artifact name, random root, arm-order key, or seed-domain literal differs below, this iter2 file is
authoritative. No failed-namespace seed, bank, score, state, or result may be reused.

## Fresh frozen roots and domains

The fresh, pairwise-disjoint roots are:

```text
official training roots:        78101, 78102, 78103
official evaluation-bank roots: 78201, 78202, 78203
official split-noise roots:      78301, 78302, 78303
official shuffle roots:          78401, 78402, 78403

focused-only training roots:     994001, 994002, 994003
focused-only evaluation roots:   994101, 994102
focused-only split roots:        994201, 994202, 994203
focused-only shuffle roots:      994301, 994302, 994303
```

Official mode rejects every focused root and every root from the failed namespace. Focused mode
rejects every official iter2 root. Development tests that merely exercise library mechanisms use
ordinary development roots outside all frozen sets. In particular, no test may pass an official
iter2 root to `manual_seed`, `Generator`, a schedule, sampler, bank, split, shuffle, trainer, or
worker before the matching exclusive marker.

The exact atom encoder remains:

```text
encode_atom(value):
  if type(value) is int and value >= 0: ASCII decimal without sign/whitespace
  if type(value) is str: UTF-8
  otherwise: reject
```

The iter2 domain payload is:

```text
b"rtgs.compact-responsibility-birth.iter2.v1\0" +
encode_atom(label) + b"\0" + encode_atom(root) +
b"".join(b"\0" + encode_atom(part) for part in parts)
```

and the derived seed is the little-endian first eight SHA-256 bytes masked to 63 bits.

For replicate `j`:

```text
split:   domain_seed("split", split_root[j], 1)
shuffle: domain_seed("shuffle", shuffle_root[j], 1, stratum_code)
bank:    domain_seed("evaluation_bank", evaluation_root[j], view_name, measure_name)
```

The evaluation metadata domain is exactly
`rtgs.compact-responsibility-birth.iter2.eval.v1`. Every derived seed, generator-state-before
hash, generator-state-after hash, realized permutation/raw-draw hash, and bank tensor hash is
serialized. There is no redraw, repair, rejection, replacement, or cross-domain generator.

Phase-B arm order is cyclic:

```text
78101: G, R, U
78102: R, U, G
78103: U, G, R
```

## Outcome-neutral fail-closed implementation requirements

These requirements make already-frozen semantics executable and independently auditable; they do
not add or change a scientific gate.

1. Every score window must contain exactly five visits to each of exactly seven views. Phase A
   fails if this is not exact.
2. Each step serializes the native float32 visible residual and support VJP vectors, their hashes,
   the visible-to-global mapping, and the divided float64 global vectors. The assigned numerator
   is the actual native VJP numerator reduced in native float32 and then cast to float64; it may
   not be reconstructed by undoing the `/128` division.
3. The enabled research VJP is compared against a disabled no-op path through a complete optimizer
   update. Forward values, scalar loss, six gradients, `means2d.grad`, six updated tensors, RNG,
   and all established history fields except explicit diagnostic/timing fields must match exactly.
4. After surgery, fail closed unless every parameter and Adam moment has the expected shape,
   device, dtype, and finite values; every newborn moment is exact zero; all scalar clocks remain
   35; and group order, name, LR, betas, epsilon, weight decay, and flags are unchanged.
5. Variable-cardinality summaries include original-survivor motion and newborn summaries split by
   clone, split-child-0, and split-child-1 lineage. No positional comparison across changed rows is
   permitted.
6. Selection finalization requires finite scales, exactly 835 pre-surgery rows, 32 unique eligible
   parents, 16 small/clone and 16 large/split parents, and exact persistent IDs `0..834`.
7. Focused verification and the seal statically and dynamically prove that no official iter2 root
   reaches a generator before its marker. Finding such use permanently closes iter2.
8. The complete local loaded-source closure is sealed. Importing a prior benchmark helper does not
   exempt its transitive local modules, tests, or data-contract modules from source binding.

## Frozen lifecycle and paths

```text
harness:
  benchmarks/compact_responsibility_birth_allocation_iter2.py
focused tests:
  tests/test_compact_responsibility_birth_allocation_iter2.py
preregistration:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG.md
preregistration review:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG_REVIEW.md
implementation review:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_IMPLEMENTATION_REVIEW.md
seal:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_SEAL.json
Phase-A attempt:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_ATTEMPT.json
Phase-A result:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_RESULT.json
Phase-A audit:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_AUDIT.json
Phase-B attempt:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_B_ATTEMPT.json
final result:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_RESULT.json
executed sources:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_EXECUTED_SOURCES.tar
run directory:
  runs/compact_responsibility_birth_allocation_iter2_20260717
visualizer:
  benchmarks/visualize_compact_responsibility_birth_allocation_iter2.py
```

Before implementation migration, an independent outcome-free review must bind this file, the
imported preregistration and reviews, and the failure audit, then write exact `Verdict: PASS` with
no unresolved findings. Before sealing, a separate independent implementation review must bind the
complete source aggregate and report exact PASS with no unresolved findings.

The public commands are:

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py seal

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py phase-a

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py phase-b \
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_RESULT.json \
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_AUDIT.json
```

Each command exclusively creates and re-reads its marker before the first matching official root
can reach any generator or schedule. There is no overwrite, retry, resume, implicit latest-file
discovery, alternate output path, copied seal, replacement worker, or seed substitution.

Phase B remains forbidden unless an independent Phase-A audit recomputes all score, mapping,
stratum, permutation, selection, gate, seed, identity, optimizer, runtime, source, and RGB evidence
and reports exact `verdict=PASS` with an empty `unresolved_findings` list.

## Claim boundary and stopping rule

All original claim limits remain unchanged. A valid positive can establish only that sampled
residual-responsibility is a promising one-wave parent-allocation signal at a matched 32-birth
budget on this one scene. It cannot establish that birth beats fixed topology, that repeated
birth/pruning is beneficial, that the method scales to larger per-view 2D sets or 3D populations,
or that it should become a default.

Any lifecycle, seed, source, runtime, RGB, evidence, finite-value, structural, Phase-A, or audit
failure yields `UNAVAILABLE` or the frozen valid negative and stops without tuning. No metric,
threshold, quota, scale boundary, event step, root, input, or arm may be amended after any iter2
official root, score, selection, bank, or arm is observed.
