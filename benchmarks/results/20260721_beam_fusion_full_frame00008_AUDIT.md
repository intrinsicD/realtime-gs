# Full `frame_00008` beam-fusion experiment — independent results audit

## Verdict

**Accept as a valid single-scene development result, with repository-verification and viewer
caveats.** The bounded initializer evaluated the declared full ray-pair space, produced the
declared 5,000 outputs, and the beam-initialized fit reached the preregistered compact-target
plateau. The count-matched initialization result is negative: beam fusion trails top-K by
0.280314 dB on the primary metric. No default, generalization, downstream-superiority, exhaustive
full-beam-equivalence, or training-speed claim is authorized.

The machine-readable result SHA-256 is
`63afd7534f1fe7fe5d186788f24f6e8c147e487ce68b2e1ab5f9a482f4ab293d`.

## Claim table

| Claim | Kind and scope | Disposition | Independent evidence |
| --- | --- | --- | --- |
| Bounded beam fusion consumed the full 26×5k bundle. | Measured, real single-scene initialization | **Confirm narrowly.** | Placement records 325 pairs and 8,125,000,000 evaluated ray pairs, exactly `325 × 5000²`. This means every pair was tested, not every splat survived. |
| Beam fusion initialized 5,000 3D Gaussians. | Measured | **Confirm.** | Placement, PLY reload, and CSR offsets independently agree on 5,000. |
| Beam fusion is a better initializer than top-K. | Comparative, fitted-view | **Retire for this scene.** | Recomputed all-view FG PSNR is 11.582614 versus 11.862928 dB, beam − top-K = −0.280314 dB; crop SSIM and alpha IoU also decline. |
| The beam-initialized fit converged. | Measured, fitted compact targets | **Confirm only under the frozen stopping definition.** | Both last-six and five-transition rules report plateau at the 70k assessment; selection chose 69k. This is not a global-optimum claim. |
| Beam fusion causes strong downstream quality. | Causal | **Not tested.** | Only beam received a new downstream fit; densification grew 5,000 to 44,222 and may account for recovery. There is no matched top-K fit. |
| The result generalizes or has held-out/novel-view quality. | Generalization | **Not tested.** | All 26 views entered initialization, fitting, selection, and stopping. `V` is fitted-view validation. |
| The bounded path is equivalent to exhaustive full-data beam fusion. | Algorithmic equivalence | **Not tested at full scale.** | The ideal fixture matches the exhaustive reference, but the full path deliberately performs lossy seed-voxel selection and no feasible full exhaustive oracle was run. |
| The CPU watcher has no performance impact. | Performance | **Retire.** | It owns no CUDA allocation, but measured CPU/RSS and checkpoint I/O are nonzero; no controlled on/off repeat exists. |
| The CPU watcher avoids CUDA contention. | Capability, observed process | **Confirm narrowly.** | Training held the only observed CUDA process; watcher and final viewer were launched with `--device cpu`. This does not prove zero system-level interference. |
| The 91.7-second top-K run closes the linked CSR task. | Task acceptance | **Not established.** | It clears the wall-time numbers once, but lacks the required frozen-reference discrete parity, repeated microbenchmark, CSR payload/RSS breakdown, and tracked benchmark update. |

## Protocol chronology and isolation

Both initialization configs embed preregistration SHA-256
`1fa29697ccc729e4caab4a0dff4e8528d244cced13c32da53d73f24aa7c7a126`. The parent source
snapshot contains the same protocol bytes from run entry. The working copy's modification time is
later because a mistaken append was immediately reverted; its current bytes still match the
run-bound hash, so the protocol content used by the run did not change.

The uninterrupted parent exposed a pre-existing recovery-specific continuation preflight. The
addendum was frozen after the 30k parent result but before any 30k→40k checkpoint or metric. It
authorizes only a fail-closed clean-parent preflight and changes no scientific parameter or gate.
`polish_start.json` binds its SHA-256
`19d4f643f02beb47fd7abc7cdd8171ff01d52e67444949ae4763f7fa52bf237b`; this is acceptable for
continuation evidence, with the timing explicitly disclosed.

The top-K and beam source manifests contain the same 55 files with identical hashes. Their compact
target receipts differ only in per-view render elapsed seconds; every tensor/hash field is equal.
At every PLY continuation, targets replayed twice with the same deterministic identity
`139b3a3803e4422ef43c31cf7026b262a54fd550d31b96ad89174d502bd08c00` and exact original-tensor
equivalence. Continuations correctly declare `continuation_exact=false` because PLY omits Adam and
RNG state.

Source RGB was not available to fitting or selection. `evaluation_request.json` was written at
10:46:30 UTC after the 69k PLY and model-selection hash were frozen; only then did the evaluation
path hash and load source RGB/masks. Those metrics remain reporting-only.

## Independent recomputation and invariant checks

Raw JSON was reopened rather than copied from console output. The audit reproduced the
initialization delta, selected step, final metrics, and joint plateau decision. Placement CSR
checks found 5,001 offsets, 108,361 contributor links, valid component/view indices, no duplicate
view within a component, positive finite weights, and 18–26 views per output. Reloaded
`gaussians_init.ply` tensor hashes match the placement receipt.

All 70 checkpoint PLYs plus the beam initialization and selected final PLY were loaded on CPU.
Their steps are exactly 1k, 2k, …, 70k; every Gaussian tensor is finite; counts range from 5,000
to 44,222 and remain below the 100,000 cap. The selected PLY hash is
`733843ae79e4464bb5c43d2174a17d329ebaf31ee45bf03cec4d1ada70699c63`.

The dirty parent is usable because revision
`d74c9a623cba8af4694e0112753927407c7fdab5` and a 57,126-byte binary working-tree patch are
preserved alongside 55 source/protocol files. Independent rehashing found zero mismatches. This is
strong local-source reconstruction evidence; compiled dependency binaries are version-bound, not
archived, so it is not a bit-replay guarantee for the full environment.

## Performance and viewer corrections

The 138.326-second beam placement and 91.665-second top-K placement are single executions, not
warm repeated benchmarks. Training-side callbacks used 0.8960 seconds across 70 saves versus
2,163.649 seconds of native optimizer work. That ratio describes callback accounting only; it is
not a viewer-on/off speedup or overhead measurement. The formal watcher sample used 11.4% of one
logical CPU and 680,532 KiB RSS, and the final idle viewer sample used 3.4% and 603,872 KiB.

The formal watcher loaded full checkpoints but kept the displayed slider at its initial 5,000
unless the user expanded it. A post-result UX repair now follows a growing checkpoint count when
the user was showing the whole previous model, and documentation now says partial writes are
retried rather than calling them atomic. The repaired code passed an HTTP-200 smoke against the
saved 70k checkpoint directory; it did not receive a new live-training overhead trial. This
repair does not affect any reconstruction metric.

## Verification

Task-related CPU tests passed in both environments. The main focused collection comprised 150
tests: 145 passed and five optional-dependency/CUDA tests skipped. Ruff check, Ruff format check,
docs sync, and `git diff --check` pass.

The repository-wide non-slow CPU gate is **not green**: 16 of 1,502 collected tests fail. None is
in the modified beam, viewer, trainer, or continuation paths. They are fail-closed historical
harness checks whose frozen environment/artifacts are absent on this workspace:

- six occupancy-factorial tests reject changed system `libstdc++`/PyTorch source bindings;
- two responsibility-birth tests reject the same frozen `libstdc++` binding;
- one capacity-crossover test rejects its frozen native ABI; and
- seven G2SR diagnostic tests require the absent
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs/manifest.json`.

Therefore the focused implementation and experiment evidence is usable, but any statement that
the entire repository verification suite passes must be withheld.

## Commands and skipped work

The audit executed:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_beam_fusion.py tests/test_splat_sfm.py tests/test_viewer.py \
  tests/test_full_compact_recovery.py tests/test_optim.py tests/test_multiscale_refinement.py

CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/python -m pytest -q \
  tests/test_beam_fusion.py tests/test_viewer.py tests/test_full_compact_recovery.py \
  tests/test_optim.py::test_checkpoint_callback_is_isolated_and_none_preserves_default_exactly \
  tests/test_multiscale_refinement.py::test_resolution_transition_keeps_optimizer_state_and_full_resolution_observers

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -m 'not slow' --tb=no
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python scripts/docs_sync.py
git diff --check
```

It also rehashed every declared source/result artifact, compared both target receipts, validated
placement CSR structure, loaded all 72 relevant PLYs, recomputed raw-result deltas and convergence,
and performed CPU-viewer HTTP/CUDA-process checks.

The expensive 8.125-billion-pair/70k experiment was not rerun during audit. No exhaustive
full-data beam reference, matched top-K downstream fit, held-out/novel-view evaluation, repeated
performance benchmark, or controlled viewer A/B was performed. Those are the exact missing pieces
needed to promote the corresponding claims.
