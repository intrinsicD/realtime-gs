# Compact residual-responsibility birth allocation iter2 — implementation review

Date: 2026-07-17

Verdict: PASS

Unresolved findings: none

Reviewed source aggregate SHA-256: 79c8f374e416a93a6572d262a09dfa41b4bd851d15596f49c5ac80e3ffa5b5de

## Scope and disposition

This is an independent, outcome-free implementation and protocol review of the final pre-seal
source. It covers the complete 44-file execution closure, the literal compositor VJP, native
screen-gradient comparator, matched G/R/U selections, one-wave clone/split surgery, raw-state
evidence, Phase-A recomputation, Phase-B replay and bank evaluation, terminal decision arithmetic,
source/input/runtime binding, once-only lifecycle, and RGB denial.

This PASS authorizes creation of the iter2 seal only while the reviewed aggregate and all bound
inputs, prerequisites, reviews, runtime fields, and namespace-absence conditions remain exact. It
does not authorize Phase A or Phase B by itself, does not report an official score or utility
result, and does not establish that responsibility allocation improves fitting. It also does not
establish birth-versus-fixed utility, repeated density control, pruning, optimal Gaussian count,
source-RGB or novel-view quality, gsplat-training parity, scaling, speed, memory, or a production
default.

Several fail-closed gaps were found during review and repaired before the aggregate above was
computed:

1. current inputs were recorded but were not initially compared exactly with the frozen iter3
   result's complete input-binding object;
2. the current runtime was recorded but its source-independent ABI and normalized import-path
   projection was not initially compared with the hash-frozen iter3 runtime;
3. Phase-A raw recomputation and parent publication checks initially trusted several duplicated
   history, raw-state, snapshot, command, and binding receipts too deeply;
4. Phase-B artifact validation initially did not fully bind canonical paths, raw state to the
   frozen initialization and Phase-A state, snapshots to raw tensors, or the final PLY to its
   source snapshot;
5. strict bank reload initially did not recompute active/null population metadata and raw
   finite, nonnegative, half-open, direct-uniform, and inactive-null invariants;
6. stored Phase-B checkpoint metrics were initially accepted without exact recomputation from the
   strict raw bank and snapshot; and
7. combined test collection exposed two transitive local modules, `rtgs.optim.strategies` and
   `rtgs.optim.trainer`, that were not initially present in the explicit source closure.

All seven findings are closed in the reviewed snapshot, with focused corruption or closure tests
where applicable.

## Claim disposition

| Claim under review | Kind and scope | Independent evidence | Disposition |
| --- | --- | --- | --- |
| The implementation computes the frozen literal residual/support responsibility without materializing an attempts-by-Gaussian matrix. | Implementation claim; first 35 compact-training updates. | Direct source and formula review, literal compositor tests, off/on optimizer parity, raw VJP/alpha recomputation. | Confirmed for the reviewed source. |
| G, R, and U are distinct matched-count parent-allocation policies with exact 16-clone/16-split budgets. | Causal-design implementation claim; one scene and one birth wave. | Selection reconstruction, tie-heavy complete-rank permutation tests, operator-count gates, CPU surgery tests, real-input focused CUDA structural smoke. | Confirmed for implementation and seal eligibility. |
| Phase B replays the common prefix and evaluates immutable fixed banks with the frozen risks and decision map. | Protocol implementation claim. | Exact replay/pairing records, raw bank and snapshot validation, exact metric-recomputation regression, log-AUC/geometric-mean/precedence tests. | Confirmed for the reviewed paths. |
| The official R arm improves compact fitting over G and U. | Scientific result claim. | No official iter2 root, score, bank, arm, or result exists. | Withheld pending the once-only phases and independent result audit. |
| The approach scales, improves novel views, or beats ordinary 3DGS. | General capability/performance claim. | Outside this single-scene compact-teacher experiment. | Withheld. |

## Bindings and chronology

The reviewed lifecycle documents independently matched:

- iter2 preregistration:
  `e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c`;
- iter2 preregistration review:
  `59b60d6516ee3547978bb41cf5faa51fc2353f262c136feec71fc6a14def22a5`;
- imported scientific preregistration:
  `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427`;
- imported preregistration review:
  `2ec29eeb5b0d5824bc7ec3c234fe4f01fa8c23a9fcb8dc164fccd54395c6d214`;
- imported initial-fail review:
  `804036c7fdcd1c82a163f7551c34a134d4e6cd4a0f6bd4d00ceb851ff8550b66`;
- imported executability-addendum review:
  `93b1858be05f75a32ba17e07fc208c1bd2ea3369720ad49adaf9b6ac5db91ee5`;
  and
- failed-namespace audit:
  `5524a274937502587a3e41a0ecffd12ba66c2cf4aaa1a853874cb99e230f8044`.

The hash-frozen prerequisite checks additionally bind the iter3 preregistration, result, and audit
at `5b3f7213...`, `c0a278a8...`, and `44836994...`, plus the older unexecuted residual
preregistration at `f65b4afe...`. The complete current input object must equal the frozen iter3
result object, not merely reproduce selected file hashes. Its canonical SHA-256 is
`682ea803dd928abcf27e4ddca5367b2bf9518365914e8dbb1a13528e5ba23f1d`.

The failed namespace remains historical evidence only. No failed or official iter2 root was
passed to a generator, scheduler, sampler, bank, trainer, score, selection, split, or evaluator
during this review.

## Literal responsibility and score arithmetic

The point renderer exposes the exact activated front-to-back `colors_v` tensor only when the
default-false compositing-basis request is enabled. The controller performs one
`torch.autograd.grad` contraction from rendered color to that basis:

- channel 0 receives detached native-float32 `active * point_loss`;
- channel 1 receives detached native-float32 `active`;
- channel 2 receives exact zero.

The contraction therefore returns the literal compositing-weight residual and support vectors.
It does not populate parameter `.grad` fields, the basis is cleared immediately, and the normal
loss backward remains unchanged. Tests compare the complete enabled and disabled optimizer update:
forward values, loss, all six gradients, retained `means2d.grad`, updated parameters, RNG state,
and established history agree exactly. Empty visibility and empty query paths remain explicit.

For each score step, the implementation serializes native residual/support vectors, their
front-to-back visible-to-global mapping, native sums, alpha contractions, active/error/alpha
vectors, and divided float64 global vectors. The independent Phase-A path recomputes:

- native-float32 VJP sums and their alpha identities;
- float64 global index-add after division by exactly 128 attempts;
- five visits for each of exactly seven views;
- equal-step-per-view and then equal-view R/S reductions;
- the assigned numerator from the native-float32 parent sum before division and its native
  active-error denominator; and
- G from the retained native `means2d.grad` norm multiplied by
  `max(width,height) * 0.5`, accumulated with native float32 indexed-add and divided by
  coarse-visible count.

Active values are exact binary values; point loss, alpha, R, S, and G evidence is finite and
nonnegative. The small upper alpha tolerance only accommodates the renderer's established
`1-alpha+1e-10` streaming arithmetic and does not alter any score or gate.

## Eligibility, shuffled control, and surgery

Eligibility exactly requires finite G/R/S, positive equal-view support, positive support in at
least two views, and at least one coarse-visible score observation. Scale family is read from the
post-update step-35 raw log-scales. Within each family, ascending `(S,persistent_id)` order is
split at `floor(n/2)` into the four frozen exhaustive strata.

G and R use descending score and ascending persistent-ID ties. U lists recipients by ascending
persistent ID, draws one complete permutation, assigns each recipient one complete canonical
R/source-ID rank label, and selects the eight recipients carrying ranks 0 through 7. The mapping
is a uniform conditional eight-member subset even when every residual scalar is tied. Fixed
points are retained; there is no redraw or recipient-ID tie repair. Tests cover a complete unique
rank permutation and a tie-heavy reversed-ID fixture.

Every arm is sorted into ascending selected persistent-ID order before surgery and contains
exactly eight rows from each stratum: 16 small clone parents and 16 large split parents. The
selected-birth seam:

- retains survivors in current physical order;
- appends clones in ascending selected-parent ID;
- appends split child 0 and then split child 1 in ascending selected-parent ID;
- consumes exactly two complete `(16,3)` native-float32 standard-normal draws;
- uses the current quaternion rotation, parent scale, scale division by 1.6, and revised opacity;
- assigns 48 monotonically increasing birth IDs with explicit lineage; and
- preserves surviving Adam moments bitwise, initializes every newborn moment to exact zero, keeps
  all six scalar clocks at 35, and preserves optimizer group identity and fields.

The exact physical count is `835 - 16 + 16 + 32 = 867`, with 819 survivors, 48 newborn rows, net
growth 32, and no pruning, reset, relocation, or additional density wave. Variable-cardinality
motion is identity-aware and never subtracts unequal physical row positions.

## Phase-A and Phase-B evidence

Phase A writes raw parameter and optimizer arrays for `0` and `35_pre`, plus a separate NPZ
Gaussian snapshot. Strict loading rejects duplicate or unexpected members and recomputes every
dtype, shape, value hash, metadata digest, group field, Adam clock, persistent ID, and semantic
state record. Label 0 is compared bitwise with tensors reconstructed from the frozen INIT PLY.
The step-35 snapshot is reconstructed from raw parameters rather than trusted as a duplicate.

The independent authorization path replays all 35 proposal samples with the exact trainer working
inputs and query chunk, recomputes history aggregates, VJP identities, G/R/S, eligibility,
strata, all shuffle seeds/permutations/mappings, selections, all nine gates, state bindings, RGB
receipts, worker commands, and parent artifact receipts. A Phase-A audit must equal this canonical
recomputation object exactly.

Each Phase-B arm is one persistent 140-update trainer call. Before surgery it must reproduce the
complete Phase-A replay and raw `0`/`35_pre` state. Across arms, all 140 view/sample fields and
both raw split draws are paired exactly. Checkpoint/state labels are exactly
`0,35_pre,35_post,70,105,140`; their raw tensors, optimizer states, persistent IDs, NPZ snapshots,
and final PLY paths and semantics are cross-bound.

Evaluation banks are created before the first arm. Strict reload verifies the derived seed and
initial generator-state hash, after-state hash schema, every tensor descriptor and draw digest,
active/null counts and fractions, finite and nonnegative arrays, direct fully active half-open
uniform rows, proposal active-inside behavior, and exact zero inactive joint/target/importance
values. A focused corruption test rewrites archive arrays and all dependent metadata hashes and
still confirms rejection.

Every stored checkpoint metric is recomputed from the strict raw bank and raw-bound snapshot
before parent decision and again before publication. `J_U` divides the float64 uniform loss sum
by 4096. `J_Q` masks inactive proposal rows but still divides by 4096, never active count. Views
are averaged equally. The recovery log-AUC uses fixed abscissae
`0,1/3,2/3,1`, and geometric means use the arithmetic mean of logs. The terminal map applies
structural/population availability first, then both comparator primary and safety gates in the
frozen exhaustive precedence order.

## Source, runtime, lifecycle, and RGB boundary

The 44-file reviewed closure includes the harness, library seams, all seal-invoked tests, scripts,
configuration and guides, lifecycle documents, and every repository-local transitive module
observed under combined test collection. Loaded `rtgs` origins must equal their repository paths;
unbound loaded local sources fail both runtime and source binding.

The source-independent current runtime projection is compared exactly with the hash-frozen iter3
runtime whose canonical SHA-256 is
`165f117243cbc93fa5374cc2b58c257d96ee5d62ca0e49b57f49dc7902772489`.
This includes Python, NumPy, PyTorch and CUDA builds, gsplat, device/driver identity, deterministic
and matmul settings, normalized `sys.path`, Torch's generated-import receipt, `PYTHONPATH`, and the
effective preload ABI. Iter2 module/source origins are validated separately so the new bound
source closure does not masquerade as the old source list.

Official root gateways are statically and dynamically guarded. Marker creation is exclusive and
immediately re-read before the first matching official mechanism. Worker failures are appended to
parent command evidence before success is required. Seals, markers, results, raw archives, banks,
snapshots, PLYs, and executed-source archives are exclusive or immutable and are revalidated
before publication.

Parent and worker execution remains inside the inherited live RGB boundary. Compact field
parameters are allowed; calibrated source images, source masks, image decoders, scene loaders,
and source-RGB metrics are denied. Required receipts include three firing negative controls and
zero source-RGB opens, forbidden imports, or forbidden modules at entry and exit.

## Checks actually executed

- Read the canonical repository guide, experiment protocol, results-audit procedure, iter2
  preregistration and review, imported preregistration/reviews/addendum, failed-namespace audit,
  final harness, library seams, and focused/affected tests.
- Ran the final exact affected four-suite pytest set under the frozen preload and focused
  environment: 141 tests passed (`33` iter2 experiment tests plus `108` point-render,
  compact-trainer, and optimizer tests).
- Ran Python compilation, Ruff lint, Ruff format-check, and `git diff --check` over the changed
  harness, tests, trainer, density, and point-renderer surfaces; all passed.
- Independently loaded a valid focused bank through the product-aware strict path, then rewrote
  arrays and all dependent descriptors/draw/metadata hashes. A half-open coordinate violation and
  non-finite color were both rejected. The committed regression additionally rejects a
  self-consistent negative proposal density and a malformed generator initial-state hash.
- Verified the checkpoint-metric corruption regression: a `1e-16` stored-value change is rejected
  by exact recomputation, and the Phase-B parent validator invokes that path.
- Ran a fresh-process, real-input CUDA structural smoke under
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33`, using only focused training root
  `994001`, split root `994201`, and shuffle root `994301`. It used the frozen 140-step config but
  stopped after the R-arm step-35 surgery. It confirmed `835 -> 867`, five visits to each view,
  exact 16/16 operator counts, 48 lineages, raw state labels `0,35_pre,35_post`, all six Adam
  clocks `0,35,35`, unchanged teacher/proposal digests, unique persistent IDs, and no official
  artifact.
- The same CUDA smoke compared source, complete frozen input, and runtime receipts exactly before
  and after training. The runtime receipt SHA-256 was
  `abb5e88b3915b538a25119d07e27d596b213e02b201480cb7867a06aeef7ed2c`.
  The live RGB receipt passed with three negative-control denials and zero source-image opens,
  forbidden imports, or forbidden modules.
- The first focused CUDA review process had already completed its training/surgery but was
  discarded when the reviewer compared the in-memory tuple `(16,3)` to its JSON list
  representation. It created no artifact and observed no decision metric. A new process with the
  normalized structural check produced the complete passing receipt above.
- Recomputed the final 44-file non-self aggregate as
  `79c8f374e416a93a6572d262a09dfa41b4bd851d15596f49c5ac80e3ffa5b5de`
  and confirmed the official seal, markers, results, executed-source archive, and run directory
  were absent immediately before this review was written.

No official root, official bank, official Phase-A score or selection, official arm, quality
metric, viewer output, timing claim, seal, marker, result, executed-source archive, or official
run directory was created during this review. The CUDA run is structural development evidence
only; its elapsed time is not performance evidence.

