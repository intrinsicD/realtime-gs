# Compact occupancy-point refinement factorial iter2 — failure audit — 2026-07-17

## Verdict

**PROTOCOL FAIL / SCIENTIFICALLY INDETERMINATE.** The once-only iter2 attempt is consumed and
must not be resumed or retried. All three evaluation-bank archives were completed successfully,
and the endpoint-safe continuous-uniform repair passed every archived bank invariant. The first
worker then completed all 140 arm-A updates, all four checkpoint evaluations, and artifact
serialization, but its final provenance check rejected a normal PyTorch import side effect:
constructing Adam appended a generated temporary-module directory to `sys.path`, while the seal
required raw `sys.path` equality before and after execution.

This is a deterministic runtime-receipt false positive, not evidence of source/input drift, RGB
access, endpoint failure, bank corruption, CUDA failure, timeout, or numerical failure. It still
invalidates the experiment because only arm A of seed 76601 ran, no checkpoint metric was
persisted, none of the paired or cross-arm invariants can be checked, and no D/B decision can be
computed. The terminal `scientific_decision=UNAVAILABLE` and `promotion_authorized=false` are the
correct append-only disposition. No density follow-up, default change, viewer claim, scheduling
claim, or target-measure claim is authorized.

## Claim inventory and disposition

| # | Claim | Kind and scope | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| 1 | The iter2 preregistration, review, seal, attempt, and result are mutually hash-bound and the result failed closed. | Proven lifecycle fact | Immutable lifecycle artifacts and recomputed canonical hashes | **Confirm.** Every link and the seal payload digest pass; the attempt is consumed. |
| 2 | The endpoint-safe uniform-coordinate repair generated a complete valid official bank population. | Measured mechanism/protocol diagnostic, one real compact bundle | Three immutable bank archives plus manifest | **Confirm narrowly.** All 86,016 uniform attempts are active, direct, finite, inside, and strictly half-open. This validates the repaired bank path, not refinement quality. |
| 3 | The complete proposal-bank population passes the two active-mass guards. | Measured protocol diagnostic | The 21 immutable proposal banks | **Confirm narrowly.** Fractions span `0.9912109375` to `0.997802734375`; max/min is `1.0066502463054188`. These gates alone cannot authorize anything. |
| 4 | Arm A, seed 76601, completed its frozen fixed-topology optimization. | Measured execution diagnostic | History, four NPZ snapshots, final PLY | **Narrow to diagnostics.** The history has 140/140 steps and six groups each report 140 optimizer steps; all artifacts are finite and fixed at 835 Gaussians. This is not efficacy evidence. |
| 5 | Balanced-cycle scheduling improves equal-view fitting. | Intended factorial scientific claim | Requires B/A and D/C across three paired seeds | **Unavailable.** No balanced arm ran. |
| 6 | Proposal-attempt targeting improves occupancy-region fitting relative to uniform correction. | Intended authorizing factorial claim | Requires D/B final and AUC-derived `J_Q`, `J_U` safety, paired wins, and all invariants | **Unavailable.** B, C, and D never ran; no fixed-bank checkpoint metric was persisted even for A. |
| 7 | The factorial authorizes a density-control follow-up. | Intended decision | Requires a complete passing 12-worker result | **Retire for iter2.** `promotion_authorized=false`; a fresh experiment is required. |
| 8 | Iter2 establishes quality, timing, memory, scaling, source-RGB equivalence, novel-view quality, or production readiness. | Unsupported capability/performance assertions | None | **Reject.** The protocol failed and the preregistration excluded these scopes even on success. |

No public documentation or ARA claim found during this audit states a positive iter2 outcome. The
first attempt's previously retired `NO_REFINEMENT_TARGET_PROMOTION` string remains non-scientific;
iter2 correctly uses `UNAVAILABLE` instead.

## Immutable lifecycle and executed-source preservation

| Artifact | SHA-256 |
| --- | --- |
| Iter2 preregistration | `da4ef58a620c687e6eccfae959113c7e1bf7f25242f2d2f4a05b885c26047278` |
| Implementation review | `a7d0a9ffc136992a2fcf383d537bf0d6f31fff4be220064df188ba650d2e6c00` |
| Seal | `c3b6c665b1255b1021fc1393dae978e36dc8f8f43ea025fdc9080d5c87cb2c01` |
| Attempt | `dee6d681acf0170ed249a6c792432d9fca5e72ab072fc3e84e99e10bae4ba2f6` |
| Terminal result | `fdc4b5aa5f1b7cd69e32237cfd6d49ec1c1bc624cc6ed29f0650c0c7fa162a6f` |
| Bank manifest | `d71dfbff59eda32a9df85b2d97e5363c873a8ffab4b4108996f700f2f60239d3` |
| Failed worker receipt | `13854409dba3877898c3933a21d71c9c85de0ef12f9f2070a0347c8cde590397` |

The seal payload digest recomputes to
`d4c0794ed0bd33a0c228c2e60b77ae51e7719843b1ec39165a4913f49cf3c80b`.
Its config canonical hash matches the attempt's
`fd686c5392a76dc4804b59cc9b9d11321fc34500ddfa4f42bb2ec2f3a42304bc`.
All 24 sealed live source paths still match their recorded hashes, and their canonical aggregate
is `0b85fa70fe35b779a8c8fcbfb68dea95f2bdc6e913912fe15a8d09e1e4954eb3`.
The current input binding also equals the sealed input binding exactly.

Before any future harness repair, the exact 24-member source set was preserved without overwrite
as `20260717_compact_occupancy_refinement_factorial_iter2_EXECUTED_SOURCES.tar`, SHA-256
`2ffdae72f066d4936a640e66283070ee101d0a8c5bd8f59f8dfc3fee34d653ac`.
An independent tar read verified that it has exactly the seal's member names, every member is an
ordinary file, every member hash equals `seal.source_hashes`, and the reconstructed source
aggregate is the same `0b85fa70...4954eb3` value.

## Chronology

| UTC | Event |
| --- | --- |
| `2026-07-17T02:42:52Z` | Immutable iter2 seal published after the independent review and focused verification. |
| `2026-07-17T02:43:18Z` | Exclusive once-only attempt token published. |
| `02:43:19.100Z` | Evaluation archive 76701 published. |
| `02:43:19.973Z` | Evaluation archive 76702 published. |
| `02:43:20.841Z` | Evaluation archive 76703 published. |
| `02:43:20.847Z` | Complete bank manifest published before any worker artifact. |
| `02:43:25.838Z`–`02:43:25.851Z` | Arm-A seed-76601 checkpoints, final PLY, and history published after training and in-memory evaluation. |
| `02:43:25.875Z` | Worker failure receipt published with `worker bindings changed during execution`. |
| `2026-07-17T02:43:26Z` | Parent terminal result published as `FAIL/UNAVAILABLE`, then execution stopped before arm B. |

The harness computes all four checkpoint metrics before `_save_snapshot_artifacts`, so the
presence of every saved arm-A artifact establishes that evaluation executed. Those metric values
existed only in process memory and were discarded when the final binding check raised. This audit
did not reconstruct or report them.

## Exact cause of `worker bindings changed during execution`

At worker entry, `verify_seal()` successfully compared the live input and runtime receipts with
the seal. The sealed runtime receipt contains the raw nine-entry `sys.path` list. The compact
trainer subsequently constructs six `torch.optim.Adam` objects. In PyTorch `2.9.0+cu128`, the
first Adam construction imports `torch._dynamo`, whose import closure includes
`torch.distributed.nn.api.remote_module` and
`torch.distributed.nn.jit.instantiator`. The installed instantiator:

1. creates a `tempfile.TemporaryDirectory()` at module import;
2. appends that random directory to `sys.path`; and
3. generates `_remote_module_non_scriptable.py` there.

The installed instantiator source has SHA-256
`567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615`.
The generated 2,355-byte Python source independently reproduced with SHA-256
`8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`.

A fresh synthetic diagnostic matched the seal's initial `sys.path`, created one CPU scalar
parameter, and constructed Adam without calling `step`, backward, CUDA, a trainer, or any official
seed. The parameter remained unchanged and optimizer state remained empty. The only runtime
receipt key that changed was `sys_path`, which gained one `/tmp/tmp*/` entry containing the file
above and its bytecode cache. All other bound runtime fields remained equal. Post-failure
`input_bindings()` remains exactly equal to `seal.inputs`.

Therefore the final combined check
`input_bindings() != sealed["inputs"] or runtime_binding() != sealed["runtime"]` was guaranteed to
fail on raw `runtime.sys_path` after normal optimizer construction. The worker receipt did not
retain a key-level diff, and the parent error omitted the worker's stdout, but the sealed code
path plus the no-step causal reproduction uniquely isolate this deterministic mismatch.

## Bank and endpoint audit

The bank manifest is bound to the immutable attempt hash. All three NPZ files pass ZIP integrity,
file size/hash checks, metadata canonical-digest checks, exact member allowlists, tensor
descriptor hashes, shape/dtype checks, and finite/non-negative density checks.

| Evaluation seed | File SHA-256 | Semantic SHA-256 | Proposal active range |
| --- | --- | --- | --- |
| 76701 | `32a401d41ed0ef94618c14b8c4a354c5080428987b29f27af2ef549b0e7e58e7` | `c2350d23cff8955de68cd83ea31b746e736153bda2ea8771efd6d22875fd7efe` | `0.9931640625`–`0.997314453125` |
| 76702 | `6030eaeeab1612c584fe00fb5ede0ed98528cbfece7a2fab33dadf7608b240fa` | `2106110e76d3f519c0b912fb8537f549719562a90e1cce7a556702b4ab2d5c6a` | `0.993896484375`–`0.9970703125` |
| 76703 | `31835c967bc6a76a5082636aa2ce40fc61788bf4135cd86de5f6ea9c36b7f922` | `3bd69c43bffd91e20e9f3bb09a46e7935606141b442e77677677ab64ea128818` | `0.9912109375`–`0.997802734375` |

Across three seeds, seven views, and 4,096 attempts per bank, every uniform attempt is active,
inside the fitted window, direct (`component_id=-1`), finite, and strictly below both exclusive
upper coordinates. The closest archived coordinate remains `0.000244140625` pixel below its
upper endpoint. Uniform proposal and joint densities are exactly equal. Proposal-bank active rows
imply inside-window rows, inactive joint density is zero, and every archived teacher color is
finite. The parent RGB-denial receipt records zero source-RGB opens/imports and all three negative
controls firing.

The iter2 bank inodes and hashes differ from both consumed iter1 partial banks; metadata carries
the fresh `official_iter2` domain and evaluation seeds. There is no evidence of partial-bank
reuse. The endpoint repair and bank-generation phase may be carried forward as a validated
mechanism, but these diagnostics do not answer either refinement question.

## Arm-A artifact audit and limits

The completed history is 246,583 bytes, SHA-256
`6f8de9b5c01b159d6ec8aa3ae028700e1ef9089cb72e35dd705be3d2c6a05736`.
It records the frozen `iid/uniform` arm, seed 76601, 140 ordered steps, 128 attempts per step,
fixed cardinality 835, unchanged teacher/proposal digests, finite values, and 140 steps for every
optimizer group. Its four checkpoint evaluations are deliberately `null`, as required for the
trainer-internal route.

| Step | NPZ SHA-256 | Semantic SHA-256 |
| --- | --- | --- |
| 0 | `a9afeb0ebfadf23dbcf37e0c1cfb7aeb7297bac3efbac06e50af8c3537ed1c15` | `4f6a7295f37ea98b2c3fdcdb41b1a93a5feefe5a8e2520474445594e46dde67c` |
| 35 | `480ab562413a7828d61b8610a8554c5c7fc27eb962778fa5147d2aaa0896ba01` | `bca7c11f69a4b623697cfb2161e35219d6aa086cc4029345ba62fd0292854620` |
| 70 | `daaf977a75ae9009c0ed897c2801512f8e20cc75112712ef038d8b422d9d351b` | `24baf489dc192d457344ab0e69226fd8c4e005fb74d7f656dcbec679096404f0` |
| 140 | `9916c12208f5926c825f439dd67d7558bccfd66dddf043bb8782f90445ee2178` | `fd0a3f2bdb8a6410260fc882376eca3360fb0afb3f961a1c4799cbf912b02daf` |

Step 0 is semantically identical to the common initialization. The final PLY is 57,193 bytes,
SHA-256 `aef095836b366edfa342813e1d0adf9a38ff5debee1ee144c7be24b22241cfbb`,
finite with 835 Gaussians, and round-trips against step 140 with zero maximum absolute difference
except opacity at `5.960464477539063e-08`.

These artifacts establish that one optimizer path executed and moved parameters. They cannot
support a quality or causal claim: there is no persisted fixed-bank metric, no B/C/D arm, no
paired A/C sample comparison, no B/D comparison, no all-arm step-zero check, no multi-seed
contrast, and no decision arithmetic. Sampled training losses are not substitutes for the frozen
evaluation risks. Post-hoc evaluation would be an audit-generated diagnostic and was deliberately
not performed.

## Required append-only disposition and iter3 conditions

1. Preserve every iter2 artifact, this audit, and the executed-source tar. Do not modify, resume,
   copy into a decision aggregate, or overwrite the iter2 run directory.
2. Treat all iter2 training and evaluation seeds as consumed. Any new attempt needs fresh,
   disjoint train/evaluation/test seeds, newly generated banks, a new namespace/run directory,
   preregistration, independent review, seal, attempt, and result.
3. The endpoint clamp, structured bank failures, RGB denial, bank budgets, scientific arms,
   metrics, and gates may remain unchanged if frozen before any iter3 outcome. Iter2 does not
   justify threshold or arm changes.
4. Replace raw mutable `sys.path` equality with a stable provenance contract. At minimum bind the
   base import roots and exact module origins, normalize only explicitly allowlisted PyTorch
   temporary-instantiator entries, and validate each such directory by purpose, member allowlist,
   and generated-source hash. Never ignore arbitrary added import paths.
5. Add a focused no-step test that constructs the actual six Adam groups and proves the post-
   optimizer runtime receipt passes while an unrelated path injection still fails.
6. Persist pre/post input and runtime receipts plus a structured key-level diff in every worker.
   The parent failure receipt must retain the worker file hash and both stdout/stderr tails so a
   child diagnostic cannot collapse to an empty message.
7. Bind the relevant external PyTorch instantiator behavior or source hash in the runtime receipt,
   because this failure arose outside repository-local source despite a fixed package version.
8. Require all three fresh bank archives and their manifest before workers, all twelve fresh
   worker PASS receipts, the preregistered paired/step-zero/cardinality/RGB checks, and the exact
   original decision gates before reporting either scientific decision.
9. Run a new independent results audit before any density follow-up, documentation outcome,
   gsplat/contact-sheet/viewer claim, or default change.

## Checks actually performed

- Read `CLAUDE.md` and `.claude/skills/realtime-gs-results-audit/SKILL.md` in full.
- Read the iter2 preregistration, review, seal, attempt, result, bank manifest, worker receipt,
  harness, sampler, trainer, compact input loader, history, and relevant public/ARA claim surfaces.
- Recomputed every lifecycle linkage, seal payload/config/source aggregate, every sealed live
  source hash, and the current sealed input binding.
- Created the exact executed-source tar with exclusive refusal-to-overwrite behavior, then read it
  back and verified all 24 members and hashes against the seal.
- Independently validated all existing bank archives and all archived endpoint/active-mass
  invariants without invoking any bank generator or RNG seed.
- Independently loaded and hashed the existing history, snapshots, initialization, and final PLY;
  checked finite tensors, cardinality, semantic checkpoint hashes, history accounting, and PLY
  round trip. No quality metric was evaluated or reconstructed.
- Reproduced the runtime mutation with a fresh parameter-only Adam constructor: no optimizer step,
  backward, CUDA, trainer, official seed, or parameter mutation occurred.
- Inspected the installed PyTorch source responsible for the temporary directory and `sys.path`
  append.
- Ran `unzip -t` on all three bank archives and `git diff --check` after audit publication.

No official seed was replayed, no bank was regenerated, no new optimization ran, no quality metric
was reconstructed, and no GPU timing/scalability claim was made. Full repository verification is
left to the producing session after any iter3 implementation changes; the immutable iter2 seal
already records its pre-attempt focused verification as passing.
