# Compact occupancy-point refinement factorial iter3 — preregistration — 2026-07-17

## Status and question

This is the third and final fresh attempt at the already frozen fixed-topology factorial asking:

1. does equal-view `balanced_cycle` scheduling improve compact-teacher refinement over `iid`; and
2. does optimizing the active Gaussian proposal-attempt risk improve occupancy-region fitting
   over uniform-area importance correction without materially harming uniform-area risk?

The first attempt is consumed by a float32 exclusive-upper-endpoint failure. Iter2 is consumed by
a runtime-receipt false positive: the first normal Adam construction imported PyTorch's remote
module instantiator, which appended one generated random temporary directory to raw `sys.path`.
Iter2 completed all fresh banks and one arm-A optimization, but no fixed-bank checkpoint metric
was persisted or reconstructed and no paired decision was available. Its independent audit
therefore says **PROTOCOL FAIL / SCIENTIFICALLY INDETERMINATE**. Iter3 does not reuse either
attempt's banks, workers, seeds, metrics, or outcome strings.

No scientific arm, metric, threshold, optimization budget, input, renderer, or quality gate is
changed. The only semantic implementation change is a preregistered normalization of one exact
framework-owned temporary import path plus stronger structured binding diagnostics.

## Runtime-path repair

Every seal, parent, and worker runtime snapshot must first import `torch._dynamo`. The import is
required to leave CPU and CUDA RNG states byte-identical and to expose the same framework side
effect that Adam would otherwise trigger only after the entry snapshot.

Exactly one random path may be normalized. It must:

- be the unique final `sys.path` entry;
- equal both
  `torch.distributed.nn.jit.instantiator.INSTANTIATED_TEMPLATE_DIR_PATH` and that module's
  `_TEMP_DIR.name` exactly;
- be an absolute, non-symlink, mode-0700 directory owned by the current uid and outside this
  repository;
- originate from the installed bound PyTorch package, whose instantiator source SHA-256 is
  `567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615`;
- contain exactly one Python source,
  `_remote_module_non_scriptable.py`, of 2,355 bytes and SHA-256
  `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`, plus only its ordinary
  `__pycache__` directory and matching bytecode file; and
- be the exact parent of the loaded `_remote_module_non_scriptable.__file__`.

Only that validated path is replaced in the receipt by the stable sentinel
`<torch.distributed.nn.jit.instantiator-temp>`. No glob, general `/tmp` allowance, path sorting,
deduplication, prefix matching, or arbitrary framework exemption is allowed. All other import
paths and their order, `PYTHONPATH`, effective `LD_PRELOAD`, CUDA flags, deterministic setting,
device/driver identity, package versions, and numeric runtime fields remain literal.

The runtime records the complete expected repository `rtgs` module-origin allowlist, while live
loaded modules must be a matching subset with no shadowed origin. At worker exit, source hashes,
unbound local-source closure, module origins, inputs, normalized runtime, and frozen config must
all be recomputed against the seal. A mismatch must retain a bounded structured key-level diff,
the failed worker receipt hash, and both subprocess tails. It may never collapse to an empty
parent error.

Focused controls use no official seed and must prove:

- priming followed by six exact CPU Adam constructors and one step preserves the normalized
  runtime path binding;
- CPU and CUDA RNG states are unchanged by priming;
- two fresh processes have different raw temporary paths but identical normalized path/runtime
  subrecords;
- a foreign suffix, duplicate or reordered valid entry, symlink alias, wrong owner/mode,
  provider-origin spoof, source tamper, extra Python source, or non-final position fails closed;
- unrelated path injection and shadowed/unbound repository-local modules still fail; and
- `LD_PRELOAD`, `PYTHONPATH`, deterministic algorithms, and CUDA matmul precision are not
  normalized.

## Frozen inputs and compact proposal

- Exact RGB-free teachers/cameras:
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`.
- Exact center-occupancy proxy:
  `runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center`.
- Common initialization:
  `runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply`, SHA-256
  `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`.
- Ordered views: `C0001,C0008,C0014,C0021,C0026,C0031,C0039`.
- Current counts are $m_{\mathrm{opt},i}^{2D}=640$, but every interface and receipt retains the
  variable per-view list, its sum, $N_{\mathrm{init}}^{3D}$, and every resulting
  $N_{\mathrm{opt}}^{3D}$ without an equality assumption.
- The proposal field is the exact aligned elementwise amplitude product of the compact teacher
  and center-occupancy proxy; proposal colors are never queried.
- Continuous uniform samples retain the iter2 endpoint-safe affine map clamped only to the dtype
  predecessor of the exclusive upper endpoint. There is no resampling, discarding, duplication,
  active-count normalization, acceptance normalization, or null reinterpretation.

Source RGB, calibrated image loaders, and image files are forbidden in the parent and every
worker. RGB may be used only after an immutable `status=PASS` result for viewer/reference
diagnosis and cannot affect fitting, metrics, selection, or the decision.

## Frozen factorial, seeds, runtime, and budget

| Arm | View schedule | Target measure |
| --- | --- | --- |
| A | `iid` | `uniform` |
| B | `balanced_cycle` | `uniform` |
| C | `iid` | `proposal_attempt` |
| D | `balanced_cycle` | `proposal_attempt` |

Official training seeds are `76801,76802,76803`; paired evaluation seeds are
`76901,76902,76903`. Focused-only training seeds are `992601,992602,992603` and focused-only
evaluation seeds are `992701,992702`. These domains are disjoint from each other, the excluded
dry seed, both consumed attempts, and all previous focused domains. Focused processes reject any
official seed and label banks `focused_test`; official workers accept only `official_iter3`.

Every arm runs 140 updates with 128 attempts/update, `eta=0.25`, checkpoints `0,35,70,140`, and
exactly 20 complete seven-view cycles in balanced arms. Every seed/view has fresh 4,096-attempt
uniform and proposal banks. All three archives and their manifest precede all twelve fresh
workers; no prior bank path may be copied, loaded, or queried by iter3.

Workers use PyTorch float32 on `cuda:0`, NVIDIA GeForce RTX 3050 capability 8.6,
`TorchPointRasterizer`, point/gaussian chunks 256, outer microbatch 128, teacher query chunk 640,
tile size 16, degree-zero hard SH, hard EWA support, visibility margin 3.0, and black background.
The explicit extent is `1.5469313859939577`. Learning rates, six Adam groups, betas, epsilon,
decay, and mean-LR schedule remain byte-for-byte the iter2 frozen config. Built-in dense
checkpoint evaluation remains disabled.

Paired A/C and B/D histories must match all 140 steps on view, sample seed, coordinates,
active/inside flags, component IDs, proposal density, and joint density. Target-density and
importance hashes must differ on all 140 steps. All four arms must have exactly equal step-zero
semantic snapshots and fixed-bank metrics within each seed. Callback snapshots at all four
checkpoints must hash-match trainer history, whose built-in evaluations remain `null`.

## Frozen evaluation and sole decision

The bank-seed derivation, continuous risks, equal-view averaging, null semantics, float64 loss
accumulation, and log-AUC equation remain exactly iter2. Uniform banks must be fully active,
direct, finite, and strictly half-open. Proposal banks retain every null.

For RGB-channel MSE $\ell$:

$$J_U=\frac1{4096}\sum_k\ell_k,\qquad
J_Q=\frac1{4096}\sum_k\mathbf1[\mathrm{active}_k]\ell_k.$$

The sole authorizing contrast is D/B on $J_Q$. It passes only if every original gate passes:

- geometric-mean final D/B $J_Q\le0.95$;
- geometric-mean AUC-derived D/B $J_Q\le0.97$;
- strict D final-$J_Q$ wins in at least two of three seeds;
- geometric-mean final D/B $J_U\le1.05$ and every seed ratio $\le1.10$; and
- all 21 proposal active fractions are at least 0.95 with global max/min at most 1.03.

Only then is `scientific_decision=AUTHORIZE_DENSITY_FOLLOWUP` and
`promotion_authorized=true`. A completed negative result is
`NO_REFINEMENT_TARGET_PROMOTION`. Any runtime, binding, bank, worker, serialization, or protocol
failure is `scientific_decision=UNAVAILABLE` with promotion false. Secondary contrasts are
non-authorizing.

## Once-only lifecycle and claim limits

The namespace is `*_factorial_iter3_*` with run directory
`runs/compact_occupancy_refinement_factorial_iter3_20260717`. The seal binds this preregistration,
both prior failure audits, both executed-source archives, all immutable prior lifecycle/run
artifacts, current source/tests, exact compact inputs, initialization, effective runtime/config,
and a fresh independent implementation review. Bindings are checked before and after focused
verification and immediately before exclusive seal publication. The exclusive attempt token
precedes every bank. Workers are bounded to 180 seconds. No retry or overwrite is allowed.

After an immutable passing result, all four seed-76801 final PLYs must be rendered on the same
seven native 5328x4608 compact-bundle cameras with repository gsplat, `packed=false`, and
`antialiased=false`, assembled into a labelled contact sheet, and exposed in a smoke-tested live
viewer. A failed result produces no decision-bearing viewer artifact.

Even a positive result remains single-scene, single-producer, fixed-topology, finite-budget
development evidence. It does not establish source-RGB equivalence, novel-view accuracy,
multi-scene transfer, fitting scalability, speed/memory superiority, ordinary-3DGS superiority,
GaussianImage_plus integration, variable-$N_{\mathrm{opt}}^{3D}$ quality, or a production default.
