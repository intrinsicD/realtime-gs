# Independent preregistration review: compact residual-responsibility birth allocation iter2

Verdict: PASS

Unresolved findings: none

## Bound artifacts and review scope

I independently reviewed the complete fresh preregistration before implementation migration or
fresh-root use. The reviewed artifact bindings are:

| Artifact | SHA-256 |
| --- | --- |
| iter2 preregistration | `e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c` |
| imported amended preregistration | `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427` |
| preserved concurrent premature review | `2ec29eeb5b0d5824bc7ec3c234fe4f01fa8c23a9fcb8dc164fccd54395c6d214` |
| preserved initial FAIL review | `804036c7fdcd1c82a163f7551c34a134d4e6cd4a0f6bd4d00ceb851ff8550b66` |
| amendment-1 independent PASS review | `93b1858be05f75a32ba17e07fc208c1bd2ea3369720ad49adaf9b6ac5db91ee5` |
| failed-namespace lifecycle audit | `5524a274937502587a3e41a0ecffd12ba66c2cf4aaa1a853874cb99e230f8044` |

The prerequisite iter3 preregistration, result, and audit hashes also match the three bindings
cited by the imported protocol. I did not inspect any unpermitted outcome, implement or migrate a
harness, invoke a fresh or historical root, construct a generator, schedule, sample, split,
shuffle, score, bank, selection, trainer state, seal, marker, or result, or run a test. This
review file is the only file written by the reviewer.

## Freshness and lifecycle checks

At review start, repository HEAD was
`2dddca4aff59702341af9faceefa76ad2505dd83`. Exact-word searches found every fresh iter2 root
only in the iter2 preregistration, and no iter2 seed-domain or namespace literal elsewhere. The 23
fresh roots are pairwise distinct and have no overlap with the failed namespace's official or
focused roots. The harness, focused test, visualizer, review, implementation review, seal, both
attempt markers, both phase results, Phase-A audit, executed-source archive, and run directory
were all absent.

The failed namespace remains terminally unavailable and is neither reopened nor pooled. Iter2
uses a complete fresh namespace, fresh official and focused roots, a fresh domain prefix,
fresh evaluation metadata domain, fresh reviews, fresh seal, fresh exclusive markers, and fresh
outputs. The required static and dynamic pre-seal root-use checks directly cover the failure mode
that closed the first namespace: any official iter2 root reaching a generator or related
mechanism before its marker permanently closes iter2.

The lifecycle remains fail closed. Implementation requires a separate source-bound review;
sealing requires the complete local loaded-source closure and pre-marker root-use proof; Phase B
requires a fresh exclusive marker plus an independent PASS audit that recomputes Phase A from raw
evidence. There is no retry, resume, overwrite, implicit discovery, seed substitution, or
cross-namespace reuse.

## Scientific and executable-protocol checks

The imported protocol is exact at its scientific scope. Iter2 changes no question, arm,
matched-count intervention, score, eligibility rule, stratum, quota, birth arithmetic, optimizer,
step/checkpoint order, teacher, proposal, initialization, metric, threshold, primary or safety
gate, terminal decision, interpretation, or claim boundary. Its only substitutions are the
explicit namespace paths, root sets, arm-order keys, seed-domain literal, and evaluation metadata
domain, plus outcome-neutral evidence requirements prompted by the lifecycle failure.

The seed derivation is executable without hidden representation choices: nonnegative integers use
unsigned ASCII decimal, strings use UTF-8, other atom types reject, separators and the iter2
prefix are literal, SHA-256 truncation and masking are fixed, and split, shuffle, and evaluation
bank calls have exact labels and ordered parts. The evaluation domain and cyclic arm order are
also explicit, with no redraw or shared generator.

All decision-bearing arithmetic carries over unchanged: the literal compositor VJP, native
float32-to-float64 reduction boundary, equal-view `R`/`S`, native screen-gradient `G`, assigned
residual fraction, Phase-A distinction gates, fixed-bank `J_Q`/`J_U`, recovery log-AUC,
per-comparator primary and safety booleans, ordered exhaustive decision map, population guards,
and structural invariants. The added exact five-visits-per-view check, native VJP serialization,
full-update no-op parity, surgery-state validation, lineage-aware variable-cardinality summaries,
selection-finalization checks, root-use proof, and transitive source sealing strengthen evidence
closure without changing an estimand or acceptance gate.

This PASS authorizes implementation migration and focused nonofficial verification only. It does
not authorize sealing, Phase A, Phase B, fresh official-root use, or any scientific claim.
