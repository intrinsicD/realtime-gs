# Task: Accelerate full compact placement with exact CSR queries

Status: flattened CSR implementation landed; acceptance evidence incomplete
Priority: P0 runtime blocker
Primary path: `GaussianObservationIndex` → `score_world_points` →
`CompactCarveInitializer.initialize`

## Outcome

Reduce the exact, image-free placement stage for the full compact benchmark from
**4:02:28 to at most 5 minutes on CPU**, with a target of **3 minutes or less**, while
preserving the selected rays, selected depth samples, and downstream reconstruction quality.

Phase 1 is a CPU-first replacement of the current Python tile-group query loop by one flattened
exact CSR index and a bounded vectorized point/component pair stream. It is the required fix.
CUDA is a later, pluggable backend, not a prerequisite. A quadtree or multilevel grid is only
considered if a new profile shows that index memory or candidate cardinality remains a material
bottleneck after the CSR fix.

This task must not reintroduce RGB images. It operates on the compact 2D Gaussian fields already
stored in the datasets.

## 2026-07-21 status update

The production CPU query now uses flattened CSR storage and bounded canonical pair streaming, and
the grouped implementation remains as a private parity oracle. The full-frame beam experiment's
independently audited top-K diagnostic exercised all 26 views, 130,000 component-center rays, 32
depth samples, and 5,000 selected Gaussians in **91.665 seconds**, clearing both numeric wall-time
targets once. It also saved the missing initialization-only artifact and measured 11.8629 dB
all-fitted-view foreground PSNR, confirming quantitatively that runtime and initialization quality
are separate questions.

That diagnostic does **not** close this task. The frozen grouped-reference candidate audit and
full discrete-winner/geometry identity comparison were not run; retained CSR payload, peak RSS,
integer path, warm repetitions, tracked `benchmarks/run.py` table, control/control downstream
envelope, and complete task-specific verification bundle remain missing. Treat the observed
91.665 seconds as a successful production-scale timing diagnostic, not the Phase-1 confirmatory
acceptance result. The all-initializer convergence suite is a separate quality study and cannot
substitute for these exact-parity gates. That suite is now complete and independently audited:
top-K initialized 5,000, grew to 43,288, and reached 37.2992 dB fitted-view foreground PSNR at the
70k selection. Dense+merge led fitted PSNR while beam led objective, so the quality study found no
materially superior converged initializer. See
`benchmarks/results/20260721_all_initializers_frame00008_{RESULT,AUDIT}.md`; none of those quality
numbers supplies the missing grouped-reference parity, CSR-memory, or repeated CPU benchmark
evidence required here.

The user's qualitative 2026-07-19 viewer inspection judged the retained 5,000-Gaussian
initialization visibly weaker than the optimized 36,816-Gaussian result, despite all 130,000
compact components participating in its scoring. No initialization-only metrics were saved, so
this is not yet a quantitative finding. Phase 1 must expose and preserve that baseline honestly;
accelerating the current top-K placer does not solve its sparse initial-state quality. Record
initialization-only metrics and viewer artifacts so a later fiber/correspondence initializer can
be compared directly, without attributing quality recovered by photometric densification to
placement.

The 2026-07-22 post-result handoff now preserves a strict seven-method initial/final viewer manifest
at `benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json`. It loads the exact top-K,
beam, dense+merge, easy-only, splat-SfM, field, and random endpoints into one unchanged orbit camera.
That closes the visualization handoff for the separate quality suite only; it does not supply this
runtime task's missing grouped-reference parity, repeated timing, memory, or downstream-control
acceptance evidence.

## Why this is the right bottleneck

The 2026-07-19 exploratory profile used the production-sized compact placement:

- 26 calibrated views;
- 130,000 input 2D Gaussian components;
- 32 depth samples per component-center ray;
- 4,160,000 sampled 3D points;
- 108,160,000 point/view projections;
- 1,016 scoring batches of at most 4,096 points; and
- exact 16-pixel AABB support indexing.

The measured wall time was 14,547.6 seconds, or 4:02:28. The session-local unpreserved profile
suggested that projection arithmetic was not dominant. In a representative 4,096-point all-view
scoring batch, the current query path evaluated about 2.55 million point/component pairs through
about 47,127 small
`GaussianObservationField._cross_values` calls. Those calls consumed 83.04% of the batch time
(about 3:21 of the full run when extrapolated). The implementation groups points by tile in
Python, converts unique tile IDs to a Python list, and invokes a small Cartesian-product tensor
operation separately for every occupied tile.

The total time, view/component/sample counts, projection count, batch count, and 13,942,595
component/tile entries are recoverable from the saved development-run receipts. The representative
pair count, call count, function-level percentage, tile population/build/load timings, and the
prototype timings below were session-local exploratory measurements: their raw profiler artifact,
exact command, CPU model, thread count, and load record were not preserved. Treat them as inputs
to this implementation task, not as tracked benchmark results; Phase 0 must remeasure them.

The current index itself is exact but fragmented:

- 13,942,595 component/tile entries;
- about 107.25 tile entries per input component;
- 409,445 non-empty tiles;
- 34.05 candidates per non-empty tile on average;
- 172 candidates in the largest observed tile; and
- 6.648 seconds to build the indexes, versus 0.398 seconds to load the compact dataset.

A session-local exploratory flattened-CSR prototype kept the same exact 16-pixel support
candidates and changed one representative batch from 15.528 seconds to 0.129–0.144 seconds. The
corresponding 108–120× ratio, maximum observed differences (`7.75e-7` in aggregate score and
`1.79e-7` in consensus color), and extrapolated 2.5–3 minute full CPU placement are unverified
hypotheses until the tracked production benchmark below reproduces them.

## Required behavior

The optimization changes data layout, batching, and reduction mechanics only. It must preserve:

- StructSplat-compatible support centers, integer AABB radii, support fade, filter variance,
  mean residuals, affine color gradients, fit-window validity, and normalized/additive blending;
- all-view score equations in `_score_world_points_batch`;
- component-center proposal order and RNG behavior;
- ray/depth sampling, source lineage, peak-width statistics, and covariance lifting;
- stable `_balanced_topk` behavior, including original candidate order as the tie-breaker;
- CPU-only import and execution; and
- `ObservationQueryBackend` substitutability.

Do not change tile size, support cutoff, coverage thresholds, score equations, candidate budget,
or the number of views/depth samples to obtain the speedup.

## Phase 0: freeze the reference

- [ ] Preserve the current grouped implementation as a private test/benchmark reference until
      Phase 1 passes all gates. It must not remain the production default after acceptance.
- [ ] Add a tracked benchmark case to `benchmarks/run.py`; write its JSON through the existing
      benchmark machinery and update the generated `docs/BENCHMARKS.md` block rather than
      hand-editing it.
- [ ] Record CPU model, PyTorch version, thread counts, git revision, tile size, pair budget,
      view/component/sample counts, and input artifact digests.
- [ ] Separate timings for dataset load, index construction, world-to-image projection, CSR pair
      construction, Gaussian evaluation, reductions, winner selection, and total placement.
- [ ] Save the reference candidate audit needed to compare source view/component IDs, winning
      depth indices, scores, consensus colors, and final selected candidate indices. Hash the
      audit and bind it to the compact inputs and configuration.
- [ ] Evaluate and save the unoptimized 5,000-Gaussian initialization separately from the final
      fitted model, including compact-view metrics, Gaussian count, and a viewer-ready PLY. State
      explicitly that this is a diagnostic baseline, not acceptable initializer quality.

The production reference run is expensive. Run it once, keep its audit, and use small deterministic
fixtures plus the saved audit for subsequent iterations.

## Phase 1: flattened exact CPU CSR query

### 1. CSR storage

Refactor `GaussianObservationIndex` in `src/rtgs/core/observation2d.py` so each field retains three
contiguous arrays instead of a dictionary containing one tensor per non-empty tile:

```text
tile_keys       int64 [T]       sorted linear tile IDs for non-empty tiles
tile_offsets    int64 [T + 1]   CSR offsets into component_ids
component_ids   int32|int64 [E] component IDs, ascending within each tile
```

Here `T` is the number of non-empty tiles and `E` is the exact component/tile overlap count.
Keep `tiles_x`, `tiles_y`, the preallocation entry cap, the maximum-candidates cap, and immutable
index statistics. Build in canonical component-ID order, and make any sort stable so each CSR row
has the same ascending component order as the current tile lists.

The builder must avoid retaining both the Python dictionary/list representation and CSR tensors.
Prefer a two-pass direct build:

1. count exact clipped AABB overlaps per tile and validate caps;
2. prefix-sum counts into offsets;
3. fill component IDs using per-tile cursors in ascending component order; and
4. retain only non-empty keys, offsets, and flattened IDs.

An equivalent bounded builder is acceptable if its measured peak memory is no worse and canonical
ordering is tested.

Use `int32` for retained component IDs only when preflight proves that every component ID and
entry-addressing conversion is safe. Keep tile keys and offsets as `int64` unless a separate proof
and overflow tests justify narrowing them. Cast component IDs to PyTorch indexing `long` only for
the current streamed pair chunk. On the measured bundle, target at most 80 MiB of retained CSR
payload. If an input requires the `int64` fallback, report it explicitly; the retained payload
must not exceed the session-local, unverified approximately 144.4 MiB int64 prototype figure
without a documented reason. Phase 0 must remeasure that payload.

### 2. Vectorized query

Replace `_groups` and its per-tile `_cross_values` calls with a vectorized CSR query:

1. validate `xy` and retain original point indices;
2. compute valid points' linear tile keys;
3. use `torch.searchsorted` against `tile_keys` and reject missing rows;
4. obtain each point's CSR row length from `tile_offsets`;
5. create a pair stream of `(point_index, component_id)` in original point order and ascending
   component order within each point;
6. evaluate `GaussianObservationField._paired_values`, not a padded or repeated
   `_cross_values` Cartesian product; and
7. reduce paired weights and weighted colors back to the original points with deterministic
   segment/scatter reductions.

The exact support test still runs inside paired evaluation. The tile index supplies an exact
candidate superset; it never changes field semantics.

`query_weight_sum` must share the same pair construction and use paired weight evaluation without
computing color. Do not implement two divergent indexing paths.

### 3. Bound transient pairs

Never materialize all pairs for an entire placement or all views. Stream CSR pairs in canonical
order. Add an explicit `max_query_pairs` limit to the index/backend configuration, and pass
`CompactCarveConfig.max_query_pairs` when the initializer constructs indexes.

For backward compatibility, retain the `component_chunk` query argument. The effective pair
chunk must be no larger than both the backend's hard pair cap and the caller-implied
`max(1, number_of_query_points * component_chunk)` budget. Splitting a point across chunks must
not change pair order or the deterministic reduction result.

For the production default, a 1,048,576-pair cap should keep the streamed int64 point and
component indices to roughly 16 MiB before value tensors, rather than materializing the
representative batch's full approximately 2.55 million pairs. Record actual peak pair count and
peak resident memory.

### 4. Integration and progress

Keep `ObservationQueryBackend.query` and `query_weight_sum` as the shared production contract.
`score_world_points` and `CompactCarveInitializer` must not branch on a concrete fast backend
except for the existing identity/cap validation.

Add a silent-by-default progress callback rather than printing from library code. A typed progress
record should cover:

- index build completion and bytes per view;
- completed/total ray batches and sampled points;
- completed point/view queries and evaluated point/component pairs;
- elapsed time, throughput, and estimated remaining time; and
- current and peak pair-chunk size.

The benchmark and long-running CLI/harness should print this record at index completion and at a
bounded interval (for example, every 10 batches or 30 seconds). Persist final counters in
initialization diagnostics so a four-hour silent regression cannot recur.

### 5. Determinism and ties

The accelerated backend must be deterministic for fixed inputs, seed, dtype, pair budget, and
thread configuration. Preserve original point order, component order, candidate order, and the
stable candidate-index tie-break in `_balanced_topk`.

Do not add score rounding or a new approximate tie threshold. If vectorized accumulation changes
a winning depth or selected candidate, treat it as a correctness failure first. Diagnose whether
canonical accumulation can restore identity. Any proposed tolerance-based exception requires a
separate audited experiment and is outside this task.

## Phase 1 tests

All correctness tests are CPU-only and deterministic.

- [ ] CSR construction matches `estimate_entries`, exact tile membership, non-empty tile count,
      maximum candidates, and canonical component ordering for tile sizes 1, 2, and 16.
- [ ] Entry/candidate caps fail before unsafe allocation. Add signed-int32 boundary tests and
      force the int64 component-ID fallback without allocating a giant real field.
- [ ] CSR `query` and `query_weight_sum` match the frozen field reference on randomized points,
      support edges, fit-window edges, empty tiles, out-of-window points, and repeated tile keys.
- [ ] Cover normalized and additive blending, nonzero support fade, antialias/filter variance,
      odd crop offsets with mean residuals, zero amplitudes, affine color gradients, and mixed
      component radii.
- [ ] Values and coordinate gradients match the reference within the existing float32 numerical
      contract. Start with `atol=1e-6, rtol=2e-6`; use tighter bounds where already established.
- [ ] Monkeypatch paired evaluation to prove every call respects `max_query_pairs`, including a
      single heavily populated tile and a point whose CSR row crosses a stream boundary.
- [ ] Repeated runs and different legal pair budgets produce identical discrete winners and
      selected lineage; score/color tensors remain within the numerical contract.
- [ ] `score_world_points` matches the reference for every returned field: score, consensus
      color, variance, coverage, `n_seen`, and `n_covered`.
- [ ] `CompactCarveInitializer` preserves candidate order, winning depth index, selected candidate
      indices, source view/component lineage, and lifted geometry on deterministic fixtures.
- [ ] Existing third-party/mock `ObservationQueryBackend` implementations continue to work.
- [ ] The full CPU test suite remains below the repository's approximate three-minute CI budget.

## Phase 1 benchmark and acceptance gates

### Microbenchmark

Use the frozen representative 4,096-point, 26-view batch and report cold index build separately
from warmed query time. Run at least one warm-up and five measured repetitions; report median,
p10/p90, pair count, query chunks, RSS, CPU/thread configuration, and both old/new results.

Required:

- median accelerated scoring time at most 0.20 seconds for the frozen batch;
- at least 60× speedup over the 15.528-second reference on the same machine;
- maximum score error at most `1e-6`;
- maximum consensus-color error at most `2e-6`; and
- exact equality of `n_seen`, `n_covered`, hull decisions, winning depth indices, and selected
  candidate indices.

### Full production placement

Run the exact full protocol: all 26 views, all 130,000 compact 2D Gaussians, 32 depth samples,
4,160,000 sampled points, the same bounds/seed/thresholds, and 5,000 selected initial 3D
Gaussians. Do not use raw reference images and do not reduce resolution, views, components,
samples, or candidates.

Required:

- total CPU placement wall time at most 300 seconds on the baseline workstation;
- target total CPU placement wall time at most 180 seconds;
- exact equality of proposed lineage, winning depth indices, eligible mask, selected candidate
  indices, selected source lineage, and initialized mean/covariance geometry against the frozen
  reference;
- score max error at most `1e-6` and consensus/initialized-color max error at most `2e-6`;
- no index cap, pair cap, integer overflow, or unbounded-allocation violation; and
- retained CSR payload and peak RSS reported, with the int32/int64 path identified.

Run the existing fixed downstream fitting/evaluation protocol from both initializations. Because
the discrete placement and geometry are required to be identical, material metric drift is not
expected. As a final guard, the accelerated result must not lose more than 0.02 dB PSNR,
`2e-4` SSIM, or `2e-4` alpha IoU relative to the control; compare against a control/control
repeat envelope and use the tighter bound when that envelope is smaller. Any regression blocks
the change even if placement is faster.

Report initialization-only metrics beside the downstream metrics and preserve a side-by-side
viewer command. Exact baseline parity is required for this runtime task, but the weak absolute
initialization quality remains open work for the correspondence-aware fiber fitting method; do
not describe CSR acceleration as improving it.

After the confirmatory run:

- [ ] run the repository results-audit skill before making a README/default/capability claim;
- [ ] add the dated result to `docs/EXPERIMENTS.md`;
- [ ] update the generated benchmark table through `benchmarks/run.py --update-docs`;
- [ ] run `./scripts/verify.sh`; and
- [ ] preserve benchmark JSON, initialization audit, viewer-ready PLYs, previews, and the exact
      viewer command.

## Phase 2: optional pluggable CUDA scorer

Start only after Phase 1 is accepted. The CPU CSR implementation remains the correctness oracle
and default on CPU-only systems.

- [ ] Introduce a lift-local `CompactPointScoringBackend` seam at
      `_score_world_points_batch`, with a CPU implementation that owns the exact CSR observation
      backends. Select implementations through one backend object; do not fork placement logic.
- [ ] Lazily load the CUDA implementation. Importing `rtgs` and running CPU tests must not require
      CUDA, gsplat, Triton, or a compiled extension.
- [ ] Upload compact fields, camera matrices, and CSR arrays once. Avoid one host/device transfer
      or kernel launch per tile. Project and evaluate an entire bounded point/view/pair batch on
      device, then return `CompactPointScores` at the existing seam.
- [ ] Preserve bounded pair streaming and canonical ordering. Provide a deterministic reduction
      mode for parity tests; do not silently accept nondeterministic atomic accumulation.
- [ ] Add a CPU-reference counterpart test and CUDA-marked parity, memory-cap, determinism, and
      performance tests.

The expected CUDA runtime of tens of seconds to roughly one or two minutes is an unmeasured
hypothesis, not an acceptance claim. Measure transfers, launches, pair evaluation, reductions,
and end-to-end placement before choosing CUDA as a default. CUDA work is successful only if it
beats the accepted CPU CSR path materially without changing the Phase 1 correctness gates.

## Phase 3: evidence-gated hierarchy

Do not build a quadtree, BVH, or multilevel grid merely because components have large AABBs. The
measured tile candidate population is modest (34.05 average, 172 maximum); the dominant cost was
Python fragmentation, not projection or an excessive local candidate set.

Open Phase 3 only if a post-CSR profile shows at least one of:

- retained index memory is a material deployment blocker;
- index construction is at least 20% of placement wall time;
- candidate evaluation remains at least 50% of placement wall time and most evaluated pairs are
  rejected by the exact support check; or
- the production dataset family has substantially worse tile-entry or candidate distributions
  than the measured bundle.

If opened, compare an exact multilevel grid and an AABB quadtree/BVH behind the same query
contract. A hierarchy may return a candidate superset, but `_paired_values` must still enforce the
original exact support rectangle. Report traversal overhead, retained/peak memory, candidates per
point, rejected-pair fraction, and full placement time. Accept it only if it materially improves
the already-vectorized CSR baseline; never combine hierarchy and CUDA changes in the same
confirmatory experiment.

## Definition of done

Phase 1 is complete when the flattened CSR CPU path is the production default, all CPU
correctness/determinism tests pass, the full exact placement is confirmed at five minutes or
less, the three-minute target result is reported honestly as met or missed, selected placement
identity is unchanged, downstream model-quality guardrails pass, progress is visible and
persisted, benchmark/experiment documentation is updated through repository workflows, and the
saved result opens in `rtgs view`.

Phase 2 and Phase 3 are independent follow-up tasks. Their absence does not block closing the
massive runtime reduction once Phase 1 meets the definition above.
