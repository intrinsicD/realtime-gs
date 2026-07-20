# Compact occupancy-point refinement factorial iter3 — implementation review — 2026-07-17

Verdict: PASS

Reviewed source aggregate SHA-256: 98621da45ae6d2a9bade89f719895ab5b982c3030ac59929dbe5453dd4a2e320

## Scope and disposition

This is an independent adversarial pre-seal implementation review, not an experiment result. It
covers the iter3 TorchDynamo runtime-path repair, unchanged factorial science, source/input/runtime
closure, focused controls, RGB denial, fresh seed/bank namespace, worker failure evidence, and
once-only lifecycle. It authorizes only creation of the iter3 seal while the aggregate above,
preregistration, inputs, runtime, prior-attempt provenance, and official namespace still match.

It does not authorize a scientific conclusion, density-refinement follow-up, default change,
quality claim, scaling claim, speed or memory claim, source-RGB-equivalence claim, or viewer
result. Those remain unavailable until a complete immutable official result receives a separate
post-result audit.

Four pre-seal gaps were found and repaired before the aggregate above was computed:

1. focused controls established a clean local-module baseline but did not negatively inject
   shadowed and unbound repository-local modules;
2. the controls did not compare complete fresh-process runtime records, expose a foreign import
   path inserted before the still-valid final Torch path, or reject a perturbed effective
   `LD_PRELOAD`;
3. `load_bank_archive` did not call the focused-process official-seed firewall before opening an
   archive; and
4. a return-zero worker with a rejected passing receipt could lose its artifact hash and
   subprocess tails at the parent boundary.

The final snapshot adds test-only negative controls for each case. Archive loading now invokes
the firewall before `np.load`. A rejected passing receipt produces a bounded key-level
entry-versus-exit diff and is wrapped in a parent `WorkerProcessError` retaining the worker
SHA-256/status/binding record and both process tails. The seed-firewall test uses a spied
test-only sentinel and never passes an official iter3 seed.

## Claim disposition

| Claim under review | Kind and scope | Independent evidence | Disposition |
| --- | --- | --- | --- |
| Iter3 changes runtime normalization and diagnostics, not factorial science. | Implementation claim; single-scene fixed-topology protocol. | Direct iter2 executed-tar comparison; exact AST equality for the frozen config, proposal/bank, evaluation, metrics, pairing, and decision functions; byte-identical sampler and trainer. | Confirmed for the reviewed source. |
| The one permitted random Torch import path is narrowly validated and reproducibly normalized. | Runtime-contract claim on the bound local CUDA environment. | Full fresh-process runtime equality, six-Adam control, exact source/origin/member checks, RNG check, and preregistered negative controls. | Confirmed for seal eligibility. |
| Runtime/source/input mismatches and worker receipt failures fail closed with useful evidence. | Protocol/lifecycle claim. | Structured diff tests, shadowed/unbound source probes, rejected-return-zero worker composition test, and source/input/runtime entry/exit code review. | Confirmed for the implemented paths. |
| The official D/B contrast improves compact fitting or authorizes density refinement. | Scientific result claim. | No official iter3 bank, worker, metric, or result exists. | Withheld pending the once-only run and post-result audit. |
| The approach scales, improves novel views, or beats ordinary 3DGS. | General capability/performance claim. | Outside this fixed-topology single-scene experiment. | Withheld. |

## Unchanged scientific contract

The exact iter2 executed-source tar was compared with the final iter3 source:

- `src/rtgs/core/observation2d.py` is byte-identical at
  `3a4c67871ab07813dfc15206d67a034175f0d913572b824642579b78e2e98153`.
- `src/rtgs/optim/compact_trainer.py` is byte-identical at
  `81a2b538f68623c39e2d17a513b3d43b41a0c7a6ea8a5f72355dd326e288378c`.
- The iter2 and iter3 ASTs are exactly equal for `_frozen_config`, config/optimizer recording,
  evaluation-bank seed derivation, log-AUC and geometric means, the sole decision function,
  proxy alignment/product construction, bank invariant/sample/archive construction, snapshot
  evaluation/storage, paired-history validation, step-zero validation, and secondary contrasts.
- Arms A–D, ordered views, checkpoints `0,35,70,140`, 140 updates, 128 attempts per update,
  4,096 bank attempts, `eta=0.25`, extent `1.5469313859939577`, 180-second worker bound,
  renderer/chunks, hard degree-zero SH, hard support, six Adam groups, learning rates, and all
  authorizing gates are unchanged.

Harness differences are confined to the fresh iter3 artifact/seed namespace, binding of both
consumed attempts, the exact preregistered Torch generated-import-path normalization, stable
module-origin/source closure, structured binding diagnostics, archive-load firewall, and
parent/worker receipt evidence.

## Runtime repair and negative controls

Every runtime snapshot imports `torch._dynamo` before recording state and verifies byte-identical
CPU and CUDA RNG states across that priming operation. Exactly one unique final path may be
replaced by `<torch.distributed.nn.jit.instantiator-temp>`. The raw path must equal both
instantiator-owned path fields, be an absolute non-symlink mode-0700 directory owned by the
current uid outside the repository, and originate from the bound Torch package.

The instantiator source is bound to
`567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615`.
The directory must contain only `_remote_module_non_scriptable.py`, its ordinary cache
directory, and one matching bytecode file. The generated source is exactly 2,355 bytes with
SHA-256 `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`,
and the loaded module origin must resolve to that source.

Focused controls prove:

- six exact CPU Adam constructors and one step leave the normalized path binding unchanged;
- two fresh processes have different raw directories but identical complete runtime records;
- foreign suffixes, duplicate/reordered/non-final valid entries, symlink aliases, wrong
  owner/mode, provider-origin spoofing, source tampering, and extra Python members fail closed;
- a foreign path inserted immediately before the still-final valid Torch entry remains literal
  and produces a key-level runtime diff;
- shadowed `rtgs` origins and unbound repository-local Python sources fail both source and runtime
  bindings;
- `PYTHONPATH`, deterministic algorithms, CUDA matmul precision, and effective `LD_PRELOAD` are
  not normalized; and
- a perturbed `LD_PRELOAD` is rejected before runtime acceptance.

The independently observed bound runtime was Python 3.12.9, PyTorch `2.9.0+cu128`, CUDA 12.8,
gsplat 1.5.3, NVIDIA GeForce RTX 3050 capability 8.6, driver 590.48.01, and effective preload
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33`. Live module-origin violations and unbound local
source sets were empty.

## Inputs, isolation, lifecycle, and RGB boundary

The strict input aggregates independently matched:

- compact teacher bundle:
  `56a02fbdf3f4f2d61d9358f486c90f6c963449c0642533859395b0c6e2f21db7`;
- center-occupancy proxy:
  `73e070fdfab42147501f94561a47681f79d26b7ff98450e31d4bf0a8d6084176`;
- common 835-Gaussian initialization:
  `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`; and
- consumed iter2 run-directory aggregate:
  `52643df0cd254f6fe48701929bcddf3fe2b23e36391e3d54f9870ac2fc6739ee`.

All fourteen immutable lifecycle/audit/executed-source artifacts from iter1 and iter2 match their
hard-coded hashes. In particular, the two failure audits are
`67bf419e696273a7b47d729b7e0c07f5afb468e297568bfc694e6ddec5c0ccc7` and
`747b093f41518513f7f0881482df515b92d5028169010fae0be7b481466e29d3`;
the executed-source archives are
`a4dbc184a4288cb50253b40421d7216f8aae585c0870d87a8e1ff98e893fde49` and
`2ffdae72f066d4936a640e66283070ee101d0a8c5bd8f59f8dfc3fee34d653ac`.

Official train/evaluation seeds `76801..76803` and `76901..76903`, focused train/evaluation seeds
`992601..992603` and `992701..992702`, the excluded dry seed, and both consumed attempts are
disjoint. Focused paths reject official seeds; official bank loading requires
`official_iter3`. The new run directory is created only after the exclusive attempt token, all
three bank archives and their manifest precede every worker, and no prior bank path is referenced.

The seal, attempt, result, and run-directory paths are append-only/exclusive. Seal bindings are
checked before and after focused verification and immediately before publication. Parent and
worker entry/exit states recompute source hashes, local-source closure, module origins, inputs,
normalized runtime, and frozen config. Passing worker receipts must have identical entry and exit
receipts; rejected receipts retain structured evidence at the terminal parent boundary.

Parent and worker fitting/evaluation operate inside the filesystem/import RGB denial boundary.
Strict compact teachers and cameras, proxy scalars, and the initialization are allowed; source
images, the dataset tree, PIL/OpenCV/imageio, and calibrated/source-scene loaders are denied.
Proposal colors are never queried. The denial boundary includes three live negative controls and
requires zero real RGB/import attempts at exit.

## Checks actually executed

- Read the repository guide, experiment skill, results-audit skill, both failure audits, iter3
  preregistration, exact prior lifecycle artifacts, complete final harness, sampler, compact
  trainer, and all three focused test files.
- Ran the exact focused pytest set under the frozen effective preload with
  `RTGS_FACTORIAL_FOCUSED_TEST=1`: 94 tests passed (36 factorial, 29 trainer, 29 sampler/query).
- Ran the exact Ruff check and Ruff format-check over the eight sealed
  harness/sampler/trainer/test/init files; both passed. `git diff --check` passed.
- Ran the real strict compact teacher/proxy bank path under the RGB guard with test-only
  evaluation seed `992702` and a temporary archive. All seven views retained 640 components and
  had 4,096/4,096 active, direct, inside, strictly half-open uniform attempts. The proposal active
  fraction minimum was `0.99462890625`, max/min was `1.0019636720667648`, bank semantic SHA-256
  was `97a9309d0c44e9e22135fe8313b61df5021574433e148e51a33a48f6d29f3a5a`, all RGB/import
  attempt counters were zero, and the temporary archive was deleted.
- Ran one exact 140-update CUDA training smoke on the real strict inputs using only test training
  seed `992601`, arm A, and no output artifact. All six Adam clocks reached 140; callback/history
  checkpoints matched at `0,35,70,140`; teacher and proposal digests were unchanged;
  `N_init=N_opt=835`; the RGB/import attempt counters were zero; and the complete normalized
  runtime record was exactly equal before and after training at SHA-256
  `2c19569e89f78ef9e8c84e1872fe8aa60ab40dba71f3f2effd6a00b82c126494`.
- Recomputed the 25-file non-self reviewed aggregate, exact input/prior bindings, runtime
  identity, source/module closure, and empty official namespace after all checks.

No seal, attempt, result, official run directory, official bank, official worker, official
training seed, or official evaluation seed was created or invoked during this review. Before
this file was published, the iter3 namespace contained only the preregistration. After
publication it contains only the preregistration and this review, which is the required clean
state for one seal.
