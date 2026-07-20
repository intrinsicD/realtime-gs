# Compact occupancy-point refinement factorial — implementation review — 2026-07-17

## Disposition

Verdict: PASS

Reviewed source aggregate SHA-256: 714a834f6b8b86fbb8fa678acae5cadf69f8e5fd0a50c9d13908d664d9c1103a

This is an independent adversarial pre-seal review of the implementation against
`20260717_compact_occupancy_refinement_factorial_PREREG.md`. The aggregate above is computed over
every entry in the harness's frozen `SOURCE_PATHS` except this self-referential review file. The
seal binds this review separately and rejects a later change to either the reviewed aggregate or
the review bytes.

No official seal, attempt, result, bank, worker, or run-directory artifact existed when this
review began or when it ended. Neither reviewer invoked the `seal`, `run`, or private `worker`
operation. This disposition authorizes only the preregistered one-shot development experiment; it
does not predict its outcome or authorize a default.

## Reviewed claims and evidence

| # | Frozen requirement | Evidence inspected | Disposition |
| --- | --- | --- | --- |
| 1 | Once-only seal/attempt lifecycle | Namespace predicates, pre/post binding snapshots, exact review gate, `O_EXCL` writes, terminal fallback | Confirmed |
| 2 | Exact source/input/runtime/config binding | Full source closure and origin checks, both directory bindings, init PLY hash, GPU/software receipt, effective config receipt | Confirmed |
| 3 | Raw proxy alignment before scalar copy | `validate_proxy_alignment`, strict bundle loads, mismatch tests | Confirmed |
| 4 | Variable per-view $m_{\mathrm{opt},i}^{2D}$ | List-valued accounting and unequal-count tests in harness and trainer | Confirmed |
| 5 | Frozen bank construction and exact colors | Isolated seed derivation, fixed attempts, null retention, tensor hashes, original-teacher query | Confirmed |
| 6 | No source RGB or RGB-capable loader access | Lazy package imports, file/import denial boundary, entry/exit module checks, negative controls | Confirmed |
| 7 | Fresh fixed-budget factorial workers | Twelve fresh bounded subprocesses, frozen arm/seed pairing and config | Confirmed |
| 8 | Paired streams and step-zero equality | Per-step sample hashes, all-step target-hash contrast, callback/checkpoint binding | Confirmed |
| 9 | $J_U$, $J_Q$, log-AUC, and decision arithmetic | Fixed-attempt reductions, equal-view aggregation, geometric ratios, strict wins and active-mass gates | Confirmed |
| 10 | Finite replay artifacts and PLY handoff | Snapshot NPZ round trips, final PLY count/finiteness/hash, history/bank hashes | Confirmed |

## Protocol checks

### Lifecycle and binding

- The official namespace must be empty before sealing. The seal takes source, input, runtime, and
  semantic-config snapshots before focused verification, repeats all four afterward, rechecks the
  review and namespace, validates the unwritten seal again, and publishes with an exclusive write.
- The review gate accepts exactly one standalone passing-disposition line and exactly one current
  reviewed-source aggregate line. A source or test change after this review invalidates the gate.
- The run verifies the immutable seal and rechecks its namespace immediately before exclusively
  publishing the sole attempt token. `execute_attempt` creates banks only after that token exists.
- Every `(training seed, arm)` is run by a fresh Python subprocess with a 180-second bound. Workers
  revalidate the seal, attempt token, seed pairing, bank hash, inputs, runtime, and source origins.
- Non-finite result serialization is converted to a bounded finite failure artifact, and the parent
  reports failure with exit status 2. Existing terminal artifacts are never overwritten.
- The sealed local import closure includes the point renderer's execution-critical
  `render/base.py` and `render/torch_ref.py`, both focused test files, package initializers, the
  preregistration, and all other loaded local `rtgs` modules. Live module paths must resolve to the
  corresponding repository files; the runtime receipt also records module origins, `sys.path`, and
  `PYTHONPATH`.

### Proxy, proposal, and scalability semantics

- Both bundles are strict-loaded. Before multiplying any amplitudes, the harness checks ordered
  view names/IDs, cameras, canvas/window, blend and coordinate-relevant semantics, `n_init`, dtype,
  component count, means, log-scales, rotations, optional filter variance, cutoff/fade/AA, and
  epsilon. Proxy scalars must be finite and lie in `[0,1]`.
- The proposal amplitude is constructed as the exact float32 product of teacher amplitude and
  center occupancy scalar. Proposal colors are constants and gradients are absent; color targets
  are queried only from the original teacher. Trainer poison-color tests independently enforce
  that separation.
- No equality assumption is made across views. Harness tests retain `[3,2,2,2,2,2,2]`, trainer
  tests retain unequal `[2,3]`, and official receipts use list-valued counts plus
  $\sum_i m_{\mathrm{opt},i}^{2D}$.
- The real frozen bundles independently strict-loaded as seven ordered views with
  `m_opt_i_2d=[640,640,640,640,640,640,640]`; the product-amplitude check passed on all 4,480
  components. The initialization PLY contains 835 finite Gaussians and matches the preregistered
  SHA-256.

### Banks, optimization, and metrics

- Each evaluation-bank seed is the masked little-endian integer from the first eight SHA-256
  bytes of the exact frozen seed string. Banks use an isolated CPU generator per
  `(evaluation seed, view, kind)`.
- Uniform banks are fully active fitted-window continuous draws. Proposal banks retain all 4,096
  attempts, including nulls. Coordinates, flags, component IDs, densities, and exact teacher
  colors are hashed in the archive and shared by all arms/checkpoints for the seed.
- Training uses the same product-density field for paired arms. A/C and B/D must match every
  scheduled view and all frozen sample hashes on all 140 steps; target-density and importance
  hashes must differ on all 140 steps. Balanced arms must visit each of seven views exactly 20
  times.
- Built-in exhaustive checkpoint evaluation is disabled. The callback is required to capture only
  steps `0,35,70,140`, and its semantic hashes must match the trainer checkpoint history. All four
  arms must have identical step-zero semantics and both fixed-bank risks within each seed.
- `J_U` is the RGB-channel MSE mean over all 4,096 uniform draws. `J_Q` is the active loss sum over
  all 4,096 proposal attempts, never the active count. Checkpoint values are equal-view means.
  Log-AUC uses frozen abscissae `(0,0.25,0.5,1)`, and D/B aggregates use geometric means of
  per-seed final or AUC-derived ratios. The strict win count, uniform-risk guards, and 21 unique-bank
  active-mass guards match the preregistration.
- The result reports numeric teacher/proposal tile-overlap preflights, proposal normalizers,
  per-view counts, bank/training attempts, and fixed `N_init^3D=N_opt^3D=835`, while explicitly
  withholding a scaling claim.

### RGB denial and artifacts

- Fresh harness import leaves `PIL`, calibrated-scene, and scene-data modules unloaded. The guarded
  region wraps built-in, `io`, and OS opens plus built-in and `importlib` imports; it rejects image
  suffixes and every path under the repository dataset. Entry/exit `sys.modules` checks prevent a
  preloaded or bypass-imported RGB-capable module from being reported as clean.
- Parent and each worker must record zero real image/import attempts and three successful negative
  controls. Source images, masks, PIL, and calibrated scene loaders are not available to bank
  construction, optimization, or metric evaluation.
- Each worker writes four hash-bound semantic NPZ snapshots and a final viewer-ready PLY. NPZ
  tensors must round-trip exactly; PLY tensors must retain cardinality and remain finite, with
  per-family round-trip error recorded. Full-resolution gsplat visualization remains correctly
  deferred until after the immutable official result.

## Adversarial findings resolved before this disposition

The first candidate was not accepted. Review found and the implementation owner repaired all of
the following before the aggregate above was frozen:

- incomplete point-renderer/package/test source binding and no live module-origin enforcement;
- eager pre-guard imports of calibrated-scene code and Pillow, plus an `importlib` denial bypass;
- a terminal serialization fallback that could write failure while returning success;
- paired target hashes required to differ on only one step rather than every step;
- no machine binding from the passing review to the exact reviewed source aggregate;
- verification-before-hashing chronology and missing pre-publication namespace rechecks;
- an incomplete semantic config receipt; and
- tile-overlap accounting that pointed at CUDA direct-backend `None` diagnostics rather than the
  numeric CPU preflight.

No blocker remains in the reviewed snapshot.

## Commands and independent observations

Executed read-only:

```text
.venv/bin/python -m pytest -q tests/test_compact_occupancy_refinement_factorial.py tests/test_compact_trainer.py
.venv/bin/python -m ruff check benchmarks/compact_occupancy_refinement_factorial.py tests/test_compact_occupancy_refinement_factorial.py src/rtgs/optim/compact_trainer.py tests/test_compact_trainer.py src/rtgs/data/__init__.py src/rtgs/optim/__init__.py
.venv/bin/python -m ruff format --check benchmarks/compact_occupancy_refinement_factorial.py tests/test_compact_occupancy_refinement_factorial.py src/rtgs/optim/compact_trainer.py tests/test_compact_trainer.py src/rtgs/data/__init__.py src/rtgs/optim/__init__.py
```

Results: 48 focused tests passed; Ruff and format checks passed. A separate strict real-bundle
preflight confirmed the seven view IDs/counts, bit-exact product amplitudes, input aggregate hashes,
835-Gaussian initialization, empty unbound-source/origin-violation sets, PyTorch `2.9.0+cu128`, and
the bound NVIDIA GeForce RTX 3050. No official optimization arm was executed during review.

Principal reviewed file hashes:

| File | SHA-256 |
| --- | --- |
| `benchmarks/compact_occupancy_refinement_factorial.py` | `7099a1662a01909efab5c6effd66b9a4182b87b9b50112ef57acc9db2ca4de64` |
| `tests/test_compact_occupancy_refinement_factorial.py` | `b14abd4910fab5159a6019b818402b03ce9638d4e987853906f816a6ee262f2f` |
| `tests/test_compact_trainer.py` | `8fa4c00abf9da6e675d90e8ccd7ab1fd04725e6263abd951db99ab9e9fb89f5e` |
| `src/rtgs/optim/compact_trainer.py` | `81a2b538f68623c39e2d17a513b3d43b41a0c7a6ea8a5f72355dd326e288378c` |
| `src/rtgs/core/observation2d.py` | `6e729ee825497a954d3653fc9bab3823fb7d6473a1337b869aa4f33a01c8806e` |
| `src/rtgs/render/torch_points.py` | `c9b6441addbe19cb06f2bf65ec6140be4c61110c4c8f4704c149183d9c8b3696` |
| Preregistration | `72553e528cbd12185b3845e63ab5367d4e78af3711acfc850383bebd7519f2bf` |

Input evidence observed during review:

- teacher-bundle aggregate: `56a02fbdf3f4f2d61d9358f486c90f6c963449c0642533859395b0c6e2f21db7`;
- center-proxy-bundle aggregate: `73e070fdfab42147501f94561a47681f79d26b7ff98450e31d4bf0a8d6084176`;
- initialization PLY: `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`.

The preregistered claim limits remain binding: even a positive scientific result is single-scene,
fixed-topology development evidence and can authorize only a later variable-$N_{\mathrm{opt}}^{3D}$
experiment.
