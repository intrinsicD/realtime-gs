# Audit: geometric Stage-3 arena on Janelle `frame_00008`

## Referee verdict

**`INVALID_DYNAMIC_CONTROL_NONREPEATABILITY`.** The experiment completed and its artifacts replay, but the two supposedly identical dynamic controls did not reproduce one another. The frozen protocol therefore invalidates the arena comparison before a correctness or performance verdict.

The producer summary labels this `REJECT_CURRENT_ARENA_CORRECTNESS`; that is a reduction precedence error. Dynamic-A and dynamic-B already disagree, so their variation cannot be attributed to arena storage. The producer JSON remains untouched; this audit is the authoritative disposition.

## Claim disposition

| Claim | Evidence | Disposition |
|---|---|---|
| The geometric arena preserves the dynamic end-to-end trajectory. | Dynamic controls themselves end at 5,424 and 5,337; the arena ends at 5,395. | **NARROW — not testable under this invalid control bracket** |
| The arena materially reduces density-event latency. | 1.175× the faster dynamic event total; frozen ≤0.80 gate not met. | **RETIRE for this run** |
| The arena improves end-to-end 10k time. | 1.018× the faster dynamic total; frozen ≤0.98 gate not met. | **RETIRE for this run** |
| Arena memory is non-inferior at this scale. | Peak allocated ratio 1.049×; reserved ratio 0.971×. | **NARROW to this observed run** |
| The default may change from dynamic allocation. | Protocol forbids a default change; validity and speed gates fail. | **RETIRE — keep dynamic default** |

## Raw measurements

| arm | final N | native 10k s | density events ms | peak alloc MiB | held-out FG PSNR |
|---|---:|---:|---:|---:|---:|
| dynamic-a | 5424 | 39.887230 | 48.930 | 45.793 | 22.314209 |
| geometric | 5395 | 40.605433 | 57.516 | 48.032 | 22.344547 |
| dynamic-b | 5337 | 41.521786 | 55.348 | 45.688 | 22.134048 |

The arena/dynamic-A density-event ratio is **1.175×** against the faster dynamic control, and its native elapsed ratio is **1.018×**. These are descriptive only because dynamic control quality/counts drifted and the run has one arena observation.

## First divergence

All arms matched through the 300-step density event (422→815→1,378). At step 400, dynamic-A reached 2,098 rows, the arena 2,094, and dynamic-B 2,096. This symmetric control drift is consistent with end-to-end CUDA nondeterminism; it is not evidence that either storage policy is correct or incorrect. The mechanism-level transaction tests remain useful but cannot rescue the consumed end-to-end protocol.

## Required next evidence

Before another timing claim, freeze a real pre-event state and its accumulated selection tensors, then apply dynamic and arena topology transactions to that same payload for exact parity. For performance, use repeated fresh-process blocks on an idle named GPU, a warmup rule that excludes one-time kernel initialization, tolerant count/quality equivalence gates justified before access, and multiple scenes. Keep `dynamic` as the default.

## Audit checks

16/16 audit checks passed. An audit check passing means the referee detected and disposed of the invalid result correctly; it does not mean the arena experiment passed.

- Machine-readable audit: `benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.json`
- Results page: `runs/geometric_arena_frame00008_20260724/index.html`
- Viewer receipt: `benchmarks/results/20260724_geometric_arena_frame00008_VIEWER_RECEIPT.json`
