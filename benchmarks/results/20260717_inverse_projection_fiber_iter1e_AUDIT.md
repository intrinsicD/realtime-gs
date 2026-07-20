# Inverse-projection fiber fitting, Iteration 1e — independent results audit

## Verdict

**ACCEPT AS A VALID NEGATIVE RESULT, with replay and metric-granularity caveats.** The publication
transaction is committed and internally consistent. The scientific result is correctly `FAIL`.
No default, production path, appearance model, or topology action is authorized.

## Claim table

| Claim | Disposition | Independent evidence |
| --- | --- | --- |
| The Iteration 1e transaction committed a scientific result. | **Confirm.** | Lifecycle is the sole `COMMITTED` marker; result, terminal, lifecycle, and validation agree on `FAIL`. Cross-hashes and recorded current inodes match, with no publication uncertainty or error. |
| The exact fiber preserves the spawning 2D Gaussian. | **Confirm, synthetic-only.** | Fiber-conic source-center maximum is `7.105e-15 px`; relative covariance maximum is `7.855e-16`. |
| Independent hard-min fitting recovers the hidden tracks and 3D geometry. | **Retire for this protocol.** | Train correct counts are `60/96,80/96,55/96`; held-out `39/64,54/64,38/64`; correct tracks `19/32,26/32,16/32`; all Gate-2 roots fail. |
| The fiber result approaches the oracle. | **Retire.** | Oracle center p90 is `3.90e-8,1.15e-8,2.42e-8`; fiber is `0.6388,0.2941,0.5575`, above every frozen `0.01` gate. |
| Fiber fitting separates from the globally wrong shuffled control. | **Retire.** | Relative center improvement `0.0047616` and held-out accuracy gain `0.2239583` both miss `0.50`. |
| Exact fibers are non-inferior or superior to free 3D geometry. | **Retire as uninterpretable.** | Free-source center and covariance drift fail attribution in every root; Gate 5 is correctly `UNINTERPRETABLE`. |
| All validity logic is fail-closed. | **Narrow.** | Every observed sentinel passed, but the combined boolean omitted three sentinel families. A post-result repair and focused regression test close this for future code; they do not rewrite the official artifact. |
| Global true correspondence, appearance, topology, or real-data fitting is established. | **Not tested.** | These were explicitly excluded. |
| Runtime or memory performance is established. | **Not tested.** | CPU timings and cumulative RSS were diagnostic, without a performance protocol. |

## Chronology and immutable evidence

| Artifact | SHA-256 |
| --- | --- |
| Preregistration | `7b2f52631355e15f5ef1c2098309af4bcfa6f91a6250813d009a01fc83737a06` |
| Preregistration review | `fda1e4dd700d87888c3c0965e11e3a988720f76801cc8d0805d5f347c4b7bef5` |
| Verification receipt | `71a98f3d6cdb1aaeb1ebca22e751d997525fe20d4f20c9111a5209c2948d7037` |
| Implementation review | `5c9417d8e7c073cbbc071922e470abe1a4f9b65d530fb6a3c58312feb1040f1a` |
| Result | `2601a45d19d1d8a636d3c0db5ef8b14adf5f4137baaf718c86e1f80a84cecf9e` |
| Terminal | `4f886e723f459670aec0078dd0b076de3b994bb6bbcefcc76e50aeacaf2164d3` |
| Lifecycle | `d20e7788e8df2c782db50c4ea0daf8d970e57e3837f9743df3b8df3ea1c4db39` |
| Commit validation | `948346f0a9b9d3a217de75b7b39ee403e747730e4b89e07845e4a90740fb18db` |
| Executed-source archive | `cc23e3ab9e95307453e97193d71f84040a832b16b08fb4e9d231f661ecb1f5a5` |

The preregistration, review, verification, and implementation review predate root consumption.
The aggregate and terminal were published exclusively, and lifecycle was exchanged last. The
commit validator confirms the public and retained recovery identities at audit time. Its own hash
is not embedded back into lifecycle, so this last current-state observation has weaker long-term
tamper evidence than the three-file commit.

The run was dirty but included a complete 50-file start/end source map with zero errors. Before
any audited repair, those exact files were archived append-only. Independent extraction found
exactly 50 ordinary files and every SHA-256 matched `source_observation_end.hashes`. This resolves
the durability concern without pretending the archive existed before the run.

## Recomputed scientific decision

Independent parsing reproduced all per-arm aggregate means and the complete gate object from the
stored arm summaries:

- Gate 1: `PASS`; all 378 checkpoint observations and all observed construction/rank checks pass.
- Gate 2: `FAIL`; every fiber-conic root fails association, track, and center-p90 thresholds while
  passing exact source projection.
- Gate 3: `FAIL`; every fiber root is far from its paired oracle.
- Gate 4: `FAIL`; fiber is almost indistinguishable from shuffled in center p90.
- Gate 5: `UNINTERPRETABLE`; the free-control attribution prerequisite is false.
- Gate 6: `FAIL`; free source drift exceeds both thresholds in every root.

The oracle arm reaches 100% train, held-out, and complete-track accuracy from the same initial
geometry. This supports the narrow statement that correspondence, rather than the inverse-
projection fiber or optimizer alone, is the failed mechanism.

## Findings and corrections

1. The official result stores per-arm metric summaries and float32 PLYs, but not exact
   per-hypothesis evaluation arrays or float64 final tensors. Aggregate and gate arithmetic is
   independently reproducible; the base metrics are not exactly re-derivable without regenerating
   consumed roots. Future experiments must save the exact final state and evaluation arrays.
2. The executed combiner at `benchmarks/inverse_projection_fiber_protocol.py` omitted
   `construction.pass`, `finite_difference.pass`, and `duplicate_hard_min.pass`. All three are true
   in this run, so the negative result is unchanged. The post-result repair now combines every
   family, and its two focused tests passed.
3. The free-source coefficient 25 was too weak for the intended attribution control. Do not infer
   that a hard source fiber helps or hurts relative to a free Gaussian from this run.
4. The post-hoc residual/prune/cluster diagnostic is explicitly hypothesis-generating. Its
   thresholds must be frozen and tested on fresh roots before any topology claim.

## Commands and skipped work

The audit used read-only `jq`, `sha256sum`, canonical-JSON/hash checks, exact aggregate/gate
recomputation, artifact-tree inspection, and hash-by-hash extraction of the source archive. The
post-result regression command was:

```bash
env -i HOME=/tmp PATH=/usr/bin:/bin \
  PYTHONPATH=/home/alex/Documents/realtime-gs/src:/home/alex/Documents/realtime-gs \
  PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -c /dev/null -o addopts= \
  --noconftest \
  tests/test_inverse_projection_fiber.py::test_combined_sentinel_requires_literal_true \
  tests/test_inverse_projection_fiber.py::test_combined_sentinel_includes_every_frozen_family
```

Result: `2 passed`. No official root was rerun. No CUDA/GPU claim or work was performed. A
post-result, non-decision-bearing CPU viewer smoke returned HTTP 200; its exact command is in the
result note.
