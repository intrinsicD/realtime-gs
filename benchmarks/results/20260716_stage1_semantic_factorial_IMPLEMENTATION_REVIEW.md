# Independent implementation review: Stage-1 scalar/color semantic factorial

Verdict: PASS

Reviewed outcome-blind on 2026-07-16 after the preregistration and its independent review were
frozen and before any official semantic-factorial seed, seal, attempt marker, mechanism result,
utility result, or result review existed. I did not invoke the harness `seal`, `mechanism`, or
`utility` actions and did not inspect a scientific outcome. This verdict authorizes only creation
of the outcome-free implementation seal after its own fresh, snapshot-bound verification run. It
does not establish mechanism validity, utility, physical semantics, real-data behavior, CUDA
behavior, performance, or a production/default change.

## Bound reviewed snapshot

```text
f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce  benchmarks/results/20260716_stage1_semantic_factorial_PREREG.md
72596ee50731e6b8c55e4e54f83ce53339f85e0672c0dd55483959759b822e7a  benchmarks/results/20260716_stage1_semantic_factorial_PREREG_REVIEW.md
6baf5455da4f3901ff97e305ba498ea91c956157baf74392c7c1c1622d27e4a7  benchmarks/stage1_semantic_factorial.py
28841fb5e4bd482647dfd68b6f0328613211b30f5f76c86e650349cb9d2953e6  tests/test_stage1_semantic_factorial.py
8007136a7381382d36fdadba15c218f78bbd1e794b9b42a6172a42b3b7542402  src/rtgs/image2gs/fit.py
434bb918804849e738e77490a803660d0076c1cbd5a786ba6fb96d8f06921ec2  tests/test_stage1_fit_seam.py
```

The seal remains authoritative for the complete repository-owned source/test manifest, loaded
source subset, revision and dirty state, environment, protocol record, commands, and this review.
Any drift from the bytes above requires a fresh implementation review rather than an addendum that
silently inherits this verdict.

## Disposition of the native-fit source drift

The preregistration review inspected `src/rtgs/image2gs/fit.py` at
`2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18`; the current reviewed file is
`8007136a7381382d36fdadba15c218f78bbd1e794b9b42a6172a42b3b7542402`. I inspected that drift rather
than treating the earlier hash as current. It adds an opt-in native-fit research seam: detached
diagnostics, supplied-initialization support, geometry freezing, and the candidate
`unit_weight_bounded_8p` appearance parameterization. The established default remains
`weight_color_9p` with geometry trainable.

The semantic-factorial harness uses ordinary `fit_views` with its fully frozen native
`FitConfig`; it does not request diagnostics, supplied initialization, frozen geometry, or the
candidate appearance parameterization. The current `fit_views` call continues to invoke the
ordinary per-view `fit_image` path. The 12/12 focused fit-seam contract tests passed. They include
masked and unmasked, implicit-default and explicit-`weight_color_9p` comparisons against a
test-local transcription of the `2a9b76d...` algorithm; final Gaussian fields and histories are
bit-exact, and initialization, RNG states, raw parameters, losses, learning rates, and update
events agree. The remaining seam tests cover paired initialization/gradient/Adam behavior,
callback isolation, nonfinite rejection, and native-only backend enforcement. I therefore dispose
the hash drift as reviewed, additive, and non-interfering with this protocol's frozen Stage-1 fit.

## Scientific implementation audit

- **Frozen design and isolation — PASS.** Seeds, twelve-view construction, nine training indices,
  three held-out indices, fit/lift/train configurations, thresholds, factorial codes, renderer,
  metrics, effects, and decision rules match the preregistration. Training scenes are physical
  subsets with ground-truth Gaussian fields removed. Held-out RGB/cameras/depth remain behind the
  one-way guard until every natural lift, matched initialization, and final model is complete,
  finite, count-checked, and archived; no fitting, selection, schedule, capacity, or refinement
  path can address held-out tensors.
- **Phase-A chronology and algebra — PASS.** All seed preparation, fitted-field validation, gauge
  construction, and source-equivalence checks precede candidate downstream semantics. The
  implementation checks the frozen amplitude, scalar/color routes, exact-product controls,
  boundedness, common coverage, strict retention, immutable source keys, production/independent
  Depth and Carve parity, output fields, render parity, completion counts, and the global
  all-cells decision. A failed prerequisite raises `ProtocolInvalid`; there is no averaging,
  backend rescue, tolerance repair, or later scientific statistic.
- **Phase-B global pre-refinement gate — PASS.** The harness constructs all four frozen arms and
  completes every natural production lift, independent parity check, canonical coverage record,
  common `rho` table, six seed/backend capacity cells, matched selection, and immutable initial
  copy before the first refinement call. It then requires exactly 24 completed models (three
  seeds, two backends, four arms), identical matched counts within each seed/backend, exact shared
  schedules, fixed topology, and complete per-seed accounting before held-out unlock.
- **Metrics and decisions — PASS.** Held-out PSNR and SSIM use the frozen renderer, clamping,
  camera aggregation, and seed-as-replicate convention. Both metrics expose per-seed scalar main
  effects, color main effects, interactions, and full-candidate differences with mean/minimum/
  maximum summaries; driver labels are limited to the preregistered PSNR rule. Non-inferiority and
  materiality use the exact conjunctive thresholds. A completed run whose final-PSNR validity
  floor fails remains a valid negative execution artifact with decision gates false, rather than
  being mislabeled a protocol-invalid triple.
- **Raw scientific evidence — PASS.** Decisive fitted, transformed, routed, coverage, lift,
  selection, model, schedule, render, target, metric, and completion tensors are archived in an
  uncompressed, pickle-free numeric NPZ. Logical array names, canonical little-endian
  dtype/shape/data digests, collection digest, canonical coverage references, finite-array checks,
  source-key order, and explicit nullable defined masks are independently checkable. Nonfinite
  failure evidence is converted to finite strict JSON without losing its NaN/+Inf/-Inf class.

## Provenance, authorization, and fail-closed audit

- **Seal chronology — PASS.** The preregistration hash is hard-coded. Seal creation snapshots the
  preregistration, this implementation review, the complete sealed path set, loaded repository
  sources, git revision/dirty bytes, environment, and protocol before and after the five frozen
  verification commands and refuses any drift. Seal loading requires the sole frozen path,
  revalidates the complete current sealed manifest, review, protocol, verification rows/output
  digests, and full recorded environment fingerprint.
- **Once-only routing — PASS.** Each scientific action derives and preflights every valid and
  invalid JSON/NPZ/note sibling, exclusively creates the canonical marker before constructing an
  official seed, binds prospective paths and source/seal inputs, and rechecks the marker and seal
  after observation. Ordinary `ProtocolInvalid` failures route only to the invalid namespace;
  completed valid and invalid artifacts have strict raw/JSON/note mutual bindings.
- **Interruption semantics — accepted exactly as preregistered.** The three siblings are written
  sequentially with exclusive creation and fsync, not as one filesystem transaction. An
  unexpected I/O interruption after the marker is claimed may therefore leave a partial triple
  (for example raw only, or raw plus JSON). That interruption consumes the once-only marker under
  the preregistration: the partial files are not a result, and the run may not resume, retry,
  redirect, overwrite, or mix labels. This is an explicit protocol disposition, not a claim of
  atomic triple creation.
- **Phase-A authorization — PASS.** Phase B requires the canonical valid Phase-A JSON, raw NPZ,
  result note, independent audit Markdown, and machine review; rejects invalid siblings; checks
  exact artifact and marker identities, timestamps, prospective paths, raw manifest/finiteness,
  mutual hashes, reviewer identity, exact nine decisive gate names, recomputation/pass booleans,
  and nonempty finite evidence. All authorization inputs are rebound in the Phase-B marker and
  rehashed after execution.
- **Scientific-process boundary — PASS.** During either scientific compute region the scoped
  guard replaces Python socket construction, DNS, and connection helpers and installs a Python
  audit hook rejecting every `socket.*` event plus `subprocess.Popen`, `os.system`,
  `os.posix_spawn[p]`, `os.fork`, `os.forkpty`, `os.exec`, and `pty.spawn`. A real direct-fork
  probe raised `ProtocolInvalid`; focused tests cover DNS, sockets, subprocesses, the direct audit
  events, restoration of all patched callables and guard globals, and successful subprocess use
  after scope exit. The current sealed scientific call graph has no internal broad exception path
  that can consume this failure.

## Verification performed

All commands below ran from the repository root with
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`. The exact frozen sequence
ran in order against the bound harness/test/source hashes above:

```text
.venv/bin/python -m ruff check .
```

Exit `0`: `All checks passed!`

```text
.venv/bin/python -m ruff format --check .
```

Exit `0`: `94 files already formatted`.

```text
.venv/bin/python -m pytest -q -m "not slow"
```

Exit `0`: the complete non-slow CPU suite reached 100%; the displayed skips were the expected
environment-gated cases.

```text
.venv/bin/python scripts/docs_sync.py
```

Exit `0`: `docs_sync: OK`.

```text
git diff --check
```

Exit `0` with no diagnostics.

The additional focused run covered 47/47 semantic-factorial tests and 12/12 native-fit seam tests
(59/59 combined). A nonofficial direct `os.fork` probe was blocked inside the scientific guard and
a post-scope subprocess succeeded. These were toy/control checks only; they did not construct,
fit, transform, lift, refine, render, or inspect any official semantic-factorial seed.

At review completion there was no semantic-factorial seal, Phase-A marker, Phase-B marker,
mechanism JSON/NPZ/note, utility JSON/NPZ/note, or scientist review. No official seed or scientific
command was run. PASS is therefore implementation authorization only. The fresh seal must rerun
and bind its own exact verification sequence, and every future scientific artifact remains subject
to the independent audits and interpretation limits frozen in the preregistration.
