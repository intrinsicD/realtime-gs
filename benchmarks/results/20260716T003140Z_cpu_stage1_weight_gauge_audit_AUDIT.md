# Independent results audit: Stage-1 weight/color gauge contract

Verdict: **QUALIFIED**

The frozen decision is numerically and procedurally supported at its preregistered narrow scope:
on seeds `0,1,2` of the CPU synthetic setup, both named product-preserving gauges pass the
source-reconstruction prerequisite and independently pass the Depth and Carve materiality gates
in all three seeds and in the raw-sum pool. The valid conclusion is therefore that the current
Stage-1 `(weight,color)` factorization boundary is materially representation-dependent for both
tested unoptimized lifters in this setup.

The result does **not** identify a correct gauge, authorize canonicalization or a default change,
establish reconstruction-quality utility, or provide real-data, held-out, CUDA/gsplat,
performance, or memory evidence.

The qualification is evidentiary rather than an observed result error: the JSON contains the
preregistered raw per-view reductions, exact keys, summaries, and hashes, but not the fitted,
render, production-output, or independent-reconstruction tensors themselves. Because this is a
consumed once-only phase and the official configurations were not replayed, tensor-level
amplitude, pixel-map, render, and production/sidecar parity cannot be recomputed independently
from the artifact. Their zero-error assertions remain bound to a reviewed, sealed harness. All
derived ratios, set arithmetic, seed decisions, pools, bindings, routing, and the companion note
were independently recomputed below.

## Claim disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | Both transformed gauges preserve the fitted source reconstructions within the frozen tolerances. | Measured; CPU synthetic, 3 seeds, 9 training views, native Stage 1. | JSON `source_equivalence`, 54 transform-view records. | **Confirm, qualified:** all serialized raw gates pass; raw pixels are not present for independent remeasurement. |
| 2 | The consumed coverage/retention inputs are materially gauge-dependent. | Mechanism diagnostic only; same synthetic training inputs. | JSON `coverage_and_retention`. | **Confirm narrowly:** both transforms pass 3/3 seeds and the raw pool. |
| 3 | The unmerged Depth lift is materially gauge-dependent. | Measured unoptimized-lift boundary; no refinement or held-out quality claim. | JSON `depth`. | **Confirm narrowly:** both transforms pass 3/3 seeds and the pool. |
| 4 | The unmerged Carve lift is materially gauge-dependent. | Measured unoptimized-lift boundary; no merge/refinement claim. | JSON `carve`. | **Confirm narrowly:** both transforms pass 3/3 seeds and the pool. |
| 5 | This audit authorizes no default change and has no Phase B. | Protocol constraint. | JSON `decision` and result note. | **Confirm.** |
| 6 | One tested gauge is physically correct or improves final reconstruction. | Not measured. | None. | **Retire/withhold.** A separately preregistered utility experiment is required. |

No matching result claim was found in `README.md`, `docs/`, or `ara/` at audit time. The result is
therefore not yet public-project or ARA evidence. A dated append-only `docs/EXPERIMENTS.md` entry
and durable tracking of the evidence set remain required before promotion.

## Chronology, binding, and routing

- Final preregistration SHA-256:
  `ec2bdaea7362649392da915af2d44e7aa47a8a1825546f8487f6afa3067b9489`.
- Outcome-free implementation review SHA-256:
  `004f3f2e4a46cf217eafbd1153c17890d3f87cef5001e54b2922a467c62e371a`
  with verdict `PASS`.
- Seal SHA-256:
  `ce8edb8f908bb7d83996bf485e4a020d2881d0701fbe5d98059e6d4fd834f5cd`.
- Once-only marker SHA-256:
  `ff8ca4abc9e744bb7c09bf73efe66fd11cf2ad133f8f1966d29d70747f8fbe47`.
- Official result SHA-256:
  `e001d6efdfcf0beea30ae578069d6057350e47b3f3516ad95f216ae495793791`.
- Harness SHA-256:
  `86dc68315e3a7f6bae9099edf2af1d2fbb7608e67e5e059478ecd0b746f0d1b1`.
- Focused-test SHA-256:
  `d3d1d9a94f26d9bf60264cc9b3bc129d1acc1b348047a1d6a8fdda530976a196`.
- Companion result-note SHA-256:
  `3571852101a8bee3ee4455117353fc8960cc2814bc0f66f44e582e360e04d78a`.

Chronology is valid. The last preregistration clarification was frozen at
`2026-07-16T01:45:29+02:00` (`2026-07-15T23:45:29Z`), followed by the seal at
`2026-07-16T00:31:32Z`, the atomic marker at `00:31:45Z`, and the completed result at
`00:32:03Z`. The prospective filename was preflighted and bound by the marker; neither invalid
sibling exists. The marker binds the same seal, preregistration, complete-source aggregate,
loaded-source aggregate, output, and note as the result.

All `74/74` sealed paths still exist and match their sealed digests. Recomputing the canonical
source-hash map gives the sealed aggregate
`91023f0c59a9a725635f309b0071268aab086bd5c70a48752f017283cff4cd88`.
The result's 34 loaded repository sources are an exact digest-matching subset and recompute to
`31068339323bbb7343b5edc7d3afa9624712d5005f92910fd6452e3d7edfc9a4`.
The dirty run is adequately source-bound: revision
`2dddca4aff59702341af9faceefa76ad2505dd83`, tracked-diff digest
`cedf9decbecf0a6caa9339b035d5ae986997bcad9e5e57af6d902b12e6c8563f`, and every
repository Python/test input used by the audit are preserved exactly.

The seal records the exact five preregistered verification commands, all with return code zero,
and their stored stdout/stderr hashes recompute. Environment bindings agree across seal, marker,
and result: Python 3.12.9, PyTorch `2.9.0+cu128`, CPU device, hidden CUDA, four Torch intra-op/
OMP/MKL threads, and deterministic algorithms enabled. This is CPU evidence only.

## Independent raw reductions

### Transform and source-equivalence gates

Every seed has `9 x 150 = 1,350` components. Per-view component/bin counts, local/original view
maps, and source-key order all reconcile. Identity field hashes equal the fitted field hashes;
all gauges have identical `xy/chol` hashes. `unit_weight` amplitude hashes equal identity
bit-for-bit. `peak_color` has pooled maxima no larger than `1.4901161e-08` absolute and
`1.1319273e-07` relative, below the `1e-7`/`1e-6` gates. Both transforms jointly change weight
and color for `1,350/1,350` components in every seed.

All 54 expected `(seed,view,transform)` source checks are present and unique. The raw L1 ratios
recompute exactly from each serialized numerator/denominator; the two transforms share the same
identity denominator within every source view. All 81 render-hash names are present and their
canonical aggregate recomputes.

| Transform | Max RGB absolute delta | Max `delta/identity` | Min reported PSNR | Gate |
|---|---:|---:|---:|---|
| `unit_weight` | `1.7881393e-07` | `2.4883569e-08` | `120.0 dB` | PASS |
| `peak_color` | `1.1920929e-07` | `2.8055970e-08` | `120.0 dB` | PASS |

The stored MSE is not serialized, but the maximum error in every record is already small enough
to imply an unfloored PSNR above the `100 dB` gate; the reported `120 dB` is consistent with the
frozen `1e-12` MSE floor.

### Coverage and retention

Each row was recomputed from the nine per-view raw sums/counts and exact retained-key lists.
`Sym/union` is retention symmetric difference over union. The joint-change fraction is `1.0` in
every row.

| Seed | Transform | Coverage delta/reference | Crossing fraction | Sym/union | Material |
|---|---|---:|---:|---:|---|
| 0 | `unit_weight` | 0.522425 | 0.235340 | 0/1350 | PASS |
| 1 | `unit_weight` | 0.518518 | 0.222319 | 0/1350 | PASS |
| 2 | `unit_weight` | 0.519546 | 0.226321 | 0/1350 | PASS |
| pool | `unit_weight` | 0.520168 | 0.227993 | 0/4050 | PASS (3/3) |
| 0 | `peak_color` | 0.700623 | 0.432967 | 520/1350 | PASS |
| 1 | `peak_color` | 0.714170 | 0.458430 | 551/1350 | PASS |
| 2 | `peak_color` | 0.700251 | 0.430459 | 540/1350 | PASS |
| pool | `peak_color` | 0.705005 | 0.440619 | 1611/4050 | PASS (3/3) |

The `unit_weight` input gate is carried by coverage, not retention; this is permitted by the
frozen disjunction. No threshold-edge pass is present.

### Depth lift

Exact ordered output keys agree with the independent-mask records, are unique, and reproduce all
stored intersections, unions, and symmetric differences. Per-view render numerators and
denominators reproduce every seed ratio and the seed-tagged raw pool. `Sym/union` refers to output
source keys.

| Seed | Transform | Sym/union | Set disagreement | Delta/signal | Delta/residual | Material |
|---|---|---:|---:|---:|---:|---|
| 0 | `unit_weight` | 0/864 | 0.000000 | 0.581127 | 1.045292 | PASS |
| 1 | `unit_weight` | 0/872 | 0.000000 | 0.581378 | 1.082140 | PASS |
| 2 | `unit_weight` | 0/837 | 0.000000 | 0.582350 | 1.103669 | PASS |
| pool | `unit_weight` | 0/2573 | 0.000000 | 0.581622 | 1.076599 | PASS (3/3) |
| 0 | `peak_color` | 68/864 | 0.078704 | 2.026260 | 3.644701 | PASS |
| 1 | `peak_color` | 102/872 | 0.116972 | 2.151051 | 4.003828 | PASS |
| 2 | `peak_color` | 76/837 | 0.090800 | 1.891383 | 3.584544 | PASS |
| pool | `peak_color` | 246/2573 | 0.095608 | 2.022173 | 3.743105 | PASS (3/3) |

`unit_weight` has unchanged keys and passes through the preregistered render gate; `peak_color`
also passes the set gate. This is not transform mixing: each named transform independently passes
3/3 seeds and its own raw pool.

### Carve lift

For every seed/gauge, all 1,350 source records are present once in canonical
`(seed,local_view,component)` order. Recomputed `keep`, `valid_ray`, and `placed` counts agree;
unkept records have nullable downstream fields, placed records are kept valid rays, and the
ordered placed keys exactly equal the output-key sequence. Source-record, coverage-hash-list, and
13-volume-hash-list aggregates recompute. Shared-key tunnel-score and selected-depth deltas
recompute from the raw source records. Set and render pools again use seed-tagged keys and raw
per-view sums.

| Seed | Transform | Sym/union | Set disagreement | Delta/signal | Delta/residual | Material |
|---|---|---:|---:|---:|---:|---|
| 0 | `unit_weight` | 137/1293 | 0.105955 | 0.596981 | 0.934322 | PASS |
| 1 | `unit_weight` | 120/1280 | 0.093750 | 0.590380 | 0.965827 | PASS |
| 2 | `unit_weight` | 95/1250 | 0.076000 | 0.581413 | 0.912740 | PASS |
| pool | `unit_weight` | 352/3823 | 0.092074 | 0.589632 | 0.937200 | PASS (3/3) |
| 0 | `peak_color` | 761/1156 | 0.658304 | 1.004268 | 1.571758 | PASS |
| 1 | `peak_color` | 792/1160 | 0.682759 | 1.069506 | 1.749650 | PASS |
| 2 | `peak_color` | 674/1155 | 0.583550 | 1.160237 | 1.821414 | PASS |
| pool | `peak_color` | 2227/3471 | 0.641602 | 1.077597 | 1.712803 | PASS (3/3) |

Both transforms independently pass both set and render mechanisms with wide margins.

## Parity and isolation

- All nine Depth `(seed,gauge)` production/independent parity summaries report exact zero maximum
  error for means, covariance, opacity, and SH. All shared transformed/identity Depth keys have
  zero serialized maximum mean, covariance, and opacity deltas under the required geometry
  control.
- All nine Carve `(seed,gauge)` production/sidecar parity summaries report exact zero maximum
  error for means, covariance, opacity, and SH. Output counts and ordered sidecar keys reconcile.
- These parity values cannot be recalculated from tensors because neither independent output is
  serialized. Code inspection confirms that parity is a fail-closed prerequisite before any
  comparison, and the sealed focused tests exercise the behavior.
- The standalone coverage diagnostic hashes each tensor with the domain name `coverage`, whereas
  the Carve sidecar uses `coverage_0` through `coverage_8`. Since the hash domain is part of the
  digest, the two sections' maps cannot be cross-compared post hoc even though sealed code calls
  the same renderer on the same gauge/view. This does not alter either independently recomputed
  gate, but it prevents an otherwise useful identity invariant.
- Preparations contain exactly seeds `0,1,2`, physical training indices
  `[0,1,2,4,5,6,8,9,10]`, held-out indices `[3,7,11]`, and nine aligned image/camera/depth hashes.
  The lifter-facing scene records `gt_gaussians=None`, no masks, and the frozen training-scene
  extent cache. Source order is exactly nine blocks of 150 immutable keys per seed.
- Code inspection confirms one full synthetic-scene construction per seed, immediate physical
  subsetting, deletion of the full scene, one native fit per training image, fresh unmerged
  lifters per gauge, and no held-out evaluation or optimization. The retained points and extent
  are the explicitly preregistered world-space priors. No Carve merge-control artifact is loaded.

## Severity-ranked findings

1. **Moderate / qualification — raw tensors are not reviewable post hoc.** The artifact satisfies
   the preregistered requirement for raw per-view numerators, counts, keys, hashes, and configs,
   but it does not contain enough data to independently recompute transform amplitude errors,
   pixel-level source/coverage/render reductions, or production/diagnostic tensor parity without
   replaying the consumed official configuration. This is not a detected gate error or a
   preregistration violation, but it prevents an unqualified tensor-level confirmation.
2. **Minor — coverage hashes are cross-section incompatible.** Domain-separated names differ
   between the standalone and Carve hash sites, preventing a direct invariant check on maps that
   should otherwise be identical.
3. **Minor / process — evidence is not yet durably integrated.** The result, seal, marker,
   preregistration, harness, and focused test are currently untracked, and no dated
   `docs/EXPERIMENTS.md`/ARA disposition exists. Do not cite the result as project-level evidence
   until the evidence set and this audit are preserved and the append-only ledger is updated.

No critical or scientific-decision-reversing finding was found.

## Commands and checks actually executed

The once-only scientific command was **not** rerun. Independent checks used the existing JSON and
sealed sources only.

```text
sha256sum <result> <seal> <marker> <prereg> <review> <harness> <focused-test> <result-note>
.venv/bin/python -  # independent JSON reductions and invariant checks; no artifact writes
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_stage1_weight_gauge_audit.py
# 11 passed
.venv/bin/python -m ruff check \
  benchmarks/stage1_weight_gauge_audit.py tests/test_stage1_weight_gauge_audit.py
# All checks passed
.venv/bin/python -m ruff format --check \
  benchmarks/stage1_weight_gauge_audit.py tests/test_stage1_weight_gauge_audit.py
# 2 files already formatted
git diff --check
```

The seal's already-recorded repository-wide CPU verification was independently integrity-checked,
including command order, return codes, and stdout/stderr hashes. CUDA tests were skipped there as
appropriate and no GPU work was performed here.

## Required next evidence

- To remove the auditability qualification, use a new append-only preregistered namespace that
  serializes immutable raw tensor sidecars (or independently decodable tensor archives), hashes
  production and diagnostic outputs separately, and uses one common domain label for identical
  coverage maps. The consumed attempt must not be overwritten or replayed as a replacement.
- To select a gauge or change a default, preregister a separate causal utility experiment with a
  candidate justified independently of this outcome, held-out evaluation, matched primitive and
  optimization budgets, multi-seed minimum effects, and a new scientist pass.
- Real-data, CUDA/gsplat, speed, and memory claims each require their own evidence; this CPU
  representation audit supplies none.
