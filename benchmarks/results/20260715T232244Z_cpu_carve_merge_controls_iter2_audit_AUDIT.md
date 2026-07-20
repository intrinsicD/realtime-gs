# Independent scientist audit: Carve merge controls Phase A, Retry 2

## Verdict

**PASS for the frozen fail-closed Phase-A disposition.** The artifact is strict, finite,
source-bound, and internally consistent, and its complete decision was independently recomputed
from the serialized construction arrays and ordered per-view materiality evidence. The seed passes
are `[false, false, false]`; therefore `phase_b_authorized` is `false` and Phase B is forbidden.

No `*_AUDIT.json` machine-clearance artifact was created. The required clearance type is only
valid when Phase A authorizes Phase B, which this result does not.

Audited result:
`benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit.json`, SHA-256
`1e1142b4a4301b7f05546f62d5868c64e976183b549dd305775fca43753a29cc`.

This verdict confirms a CPU synthetic construction/materiality failure only. It does not evaluate
fixed-topology optimization utility, real scenes, density interaction, CUDA/gsplat behavior, or a
default change. Held-out initialization metrics were checked only for schema, split, and truth-hash
binding; they were not used to select, repair, or recompute any gate.

## Chronology and provenance

- The base protocol was frozen before implementation, with clarifications ending at
  `2026-07-15T21:55:00Z`. Its SHA-256 is
  `4eda7a69442bddc25cd5edce85125942f91adc52f3d62806f050a64b854b3efe`.
- The first once-only attempt failed before an artifact or scientific decision was written. Its
  seal, marker, and independent failure audit remain present and hash-bound; its named failed JSON
  and result-note targets remain absent.
- Retry 2 was born at `2026-07-15T22:56:46Z` and its outcome-neutral execution clarification was
  complete before the repaired seal. Its SHA-256 is
  `fd4361ab1a53a22760db72e99614abb04206c1b639602e0015d8debde91c1203`.
- The independent failure audit's post-review chronology correction predates the Retry-2 seal.
  The repaired implementation seal was written at `23:22:39Z`, the exclusive marker at
  `23:22:53Z`, and the result at `23:23:46Z`.
- Reverse-applying the sealed repair patch in a temporary tree reconstructs both predecessor files,
  all predecessor source hashes, and predecessor source aggregate
  `21ca5d47a4cad54c8cdf446339f174febc48018f6cee45b193569aebd40694cf`
  exactly. The patch SHA-256 is
  `6e50856b116035ba59772556ff8f37daf982df82812bc66bdaeb24afc9c288d0`.
- Retry-2 seal SHA-256:
  `8d59df3310ad67e9e21e2979d491ab740894a6a923175c959c5bd687a91e92f8`.
  Exclusive Phase-A marker SHA-256:
  `89b5f573fb5604c1db08222d5f24b72c58496c8042dbb627e5cc0a391779954e`.
- All 75 sealed path hashes and their canonical aggregate independently reproduce
  `d7752cc39fc2f1bacf5e28cb75215cff95639bf5f80e74a9b9fa74fe7f0269d4`.
  All 37 loaded repository-source hashes are a matching subset of the seal and reproduce loaded
  aggregate `67eee53170015e7b3aae7559ca89017e58f77fc68a2c40fc1ecc9a1a97f8c7b4`.
- The seal and result agree on revision `2dddca4aff59702341af9faceefa76ad2505dd83`,
  tracked-diff hash, defaults, and environment. The runtime status differs from the seal only by
  the newly created Retry-2 seal, seal note, and Phase-A marker, as expected.
- The recorded environment is Python 3.12.9, Torch 2.9.0+cu128, CPU device,
  `CUDA_VISIBLE_DEVICES=''`, four Torch/OMP/MKL threads, and deterministic algorithms enabled.
  A CUDA-enabled wheel is installed, but no CUDA execution or performance claim is made.

The Retry-2 seal binds the independent failure audit that records the pre-seal review correction.
There is no separate machine-readable implementation-review identity in the seal; this is a minor
durability limitation. It cannot weaken the fail-closed decision, and it would not substitute for
the separate result clearance that Phase B would have required after a passing Phase A.

## Independent reconstruction

Strict parsing rejected duplicate keys and non-standard numeric constants; a recursive pass found
all serialized floats finite. Seeds, split, nine-view native fit histories, fit configuration, raw
Carve configuration, artifact types, command, defaults, and environment match the frozen protocol.
The split is training `[0,1,2,4,5,6,8,9,10]` and held-out `[3,7,11]` for seeds 0, 1, and 2.

For every seed, the unique-key, group-ID, group-count, native-weight, float64 audit-weight,
representative-index, and global-index tensor hashes were independently rebuilt byte-for-byte.
Group IDs are contiguous, counts equal `bincount(group_ids)`, keys are unique and canonically
ordered, representatives are the lowest-index native-weight maxima in their groups, and global
indices are the stable lexicographic top-K set. The serialized native and float64 weight arrays are
finite and at least `1e-12`; their maximum absolute differences are
`8.255308308962542e-12`, `7.541440982436996e-12`, and
`5.973565954766388e-12` for seeds 0, 1, and 2 respectively. Maximum relative differences are
`2.035391496800446e-7`, `2.012273947275538e-7`, and
`1.9542902036756263e-7`.

| Seed | Raw N | K | Compression | Multi groups | Raw in multi | Raw multi fraction | Control Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1156 | 1125 | 0.026816608996539815 | 29 | 60 | 0.05190311418685121 | 0.948051948051948 |
| 1 | 1160 | 1129 | 0.026724137931034453 | 31 | 62 | 0.05344827586206897 | 0.9482312338222606 |
| 2 | 1155 | 1128 | 0.023376623376623384 | 27 | 54 | 0.046753246753246755 | 0.9549393414211439 |

The ordered Python-float left fold was independently applied to all nine training-view terms. Its
totals and ratios match the artifact by exact binary64 equality:

| Seed | Raw residual denominator | Moment-vs-voxel numerator | Voxel ratio | Moment-vs-global numerator | Global ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 3610.4076117890227 | 20.45873384109008 | 0.005666599464915377 | 64.54249678620685 | 0.017876789472595 |
| 1 | 3387.260652190689 | 12.902527867299057 | 0.0038091334538883003 | 81.47863794935577 | 0.024054434044412844 |
| 2 | 3539.4294235216485 | 9.991477129699476 | 0.0028229061620215026 | 65.69790481980716 | 0.018561721949646705 |

The serialized moment-error extrema are finite and imply the frozen tolerance passes. Maximum
absolute errors `(mean, covariance, SH, opacity)` are:

| Seed | Mean | Covariance | SH | Opacity | Minimum covariance eigenvalue |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.960464488641293e-8 | 1.1787804439153393e-8 | 1.692639497452575e-7 | 7.450580610801616e-9 | 6.126078302788085e-5 |
| 1 | 7.862984008344398e-8 | 1.3276431396966326e-8 | 2.709393360778023e-7 | 7.450580610801616e-9 | 5.857247347867695e-5 |
| 2 | 6.150305581487103e-8 | 1.1957076133037314e-8 | 2.0813673629049845e-7 | 7.450580610801616e-9 | 5.284638155478226e-5 |

All covariance symmetry maxima are zero and all mean-bound violations are at most
`5.960464477539063e-8`. The moment arm hash equals the independent `merge=True` parity hash, and
the raw reporting-arm hash equals the raw Carve hash in every seed.

One evidence boundary remains: the JSON serializes raw/order/arm hashes, group and weight arrays,
and error extrema, but not the raw and moment Gaussian tensor fields. Therefore this review can
verify the reported tolerance disposition and hash relationships, but cannot rederive the actual
moment tensor errors, voxel keys from raw means, or control-field bitwise subsets without rerunning
the one-shot scientific preparation. No rerun was performed. This narrows the construction-identity
claim to the sealed in-process audit. It does not affect the decision: all seeds independently fail
three count/materiality gates using fully serialized arrays alone.

## Recomputed frozen gates

- Seed 0 fails compression `>=0.10`, multi-member groups `>=50`, and raw multi-member fraction
  `>=0.15`.
- Seed 1 fails those same three gates and moment-vs-voxel render ratio `>=0.005`.
- Seed 2 fails those same three gates, control Jaccard `<0.95`, and moment-vs-voxel render ratio
  `>=0.005`.
- Structural/tolerance, raw-count, group-count, strict count reduction, and moment-vs-global ratio
  criteria pass in all three seeds. Seed 0 also passes Jaccard and moment-vs-voxel materiality;
  seed 1 passes Jaccard.

Recomputed decision: `seed_passes = [false, false, false]` and
`phase_b_authorized = false`. The dominant result is not that moment merging is bad; it is that at
the frozen grid scale only 2.34%–2.68% of raw primitives merge, with 27–31 multi-member cells and
4.68%–5.34% raw multi-member exposure. The preregistered mechanism/utility test is therefore not
material enough to proceed.

## Claim disposition

| Claim | Disposition | Evidence |
|---|---|---|
| Retry-2 result is bound to the frozen protocol, seal, source, split, config, and CPU environment | Confirm | Hash, chronology, source-subset, command, config, and environment checks above |
| Serialized grouping, exact-count selections, count summaries, and materiality reductions are internally consistent | Confirm | Independent byte hashes, bincounts, stable selections, and exact left folds |
| Native moment construction satisfies all stated identities | Narrow | Sealed in-process checks and serialized extrema pass, but raw/moment tensors are not present for replay-free independent reconstruction |
| Phase A passes | Retire/reject | Every seed fails; independently recomputed |
| Phase B may execute | Retire/reject | Frozen authorization is false; no clearance JSON exists |
| Moment merging has fixed-budget refinement utility | Unverified | Phase B was correctly stopped before training |

## Checks actually run

- Independent strict JSON, duplicate-key, finite-value, schema, chronology, source, split, config,
  tensor-hash, group, selection, weight, left-fold, gate, and decision reconstruction.
- Reverse application of `20260716_carve_merge_controls_iter2_repair.patch` in a temporary tree and
  exact reconstruction of the predecessor source aggregate.
- The sealed harness's own `load_and_verify_seal` and `validate_phase_a_audit` path: pass.
- `python -m pytest -q tests/test_carve_merge_controls_ablation.py`: 19 passed.
- Focused Ruff check and format check for the harness/tests: pass.
- `git diff --check`: pass at audit time.

The seal already records a passing full Ruff check, full format check, complete non-slow CPU test
suite, and docs-sync run. This review did not rerun the scientific experiment, did not run Phase B,
did not use held-out quality for selection, and did not run CUDA/GPU work.

## Phase-B clearance

**DENIED by the frozen result.** No machine-clearance JSON was created. Any invocation of Phase B
for this protocol would violate the preregistered stopping rule.
