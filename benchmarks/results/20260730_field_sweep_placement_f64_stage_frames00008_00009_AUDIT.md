# Robust compact-field plane sweep (float64 successor) — independent results audit

Date: 2026-07-30 (Europe/Berlin)
Auditor: `Codex-results-auditor`
Verdict: **CONFIRMED FROZEN MIDPOINT RULE / ALL-VIEW CLAIM NARROWED**

## Referee disposition

The official successor run is audit-valid and the exact preregistered producer rule passes. Across
the two named frames and three paired measured seeds, source-excluded robust placement reduced
final held-out compact RGB MSE by `10.233620752156836%` relative to bounded midpoint in pooled
geometric mean, won all three paired seeds on each frame, retained at least `0.9921875` of tracks,
and kept the frozen source-projection invariant at or below `5.820766091346741e-11`.

That disposition is deliberately narrower than “robust beats both controls on both scenes.”
Robust beat all-view consensus on every frame-00008 seed, but on frame 00009 its geometric-mean
final compact RGB MSE was `3.10499997047535%` higher and it won only one of three paired seeds.
The pooled robust/all-view result favored robust, but the scene-level comparator behavior is
heterogeneous.

| Claim | Disposition | Independently checked evidence |
| --- | --- | --- |
| The run is bound to its prospectively approved task, data, source, and command. | **Confirm.** | The lock binds task SHA-256 `7fe68d…6570`, protocol `a45ee0…a1e0`, review SHA-256 `bea7ea…a352`, seal SHA-256 `571e71…feb7`, clean source commit `a69337…7d62`, and the exact frozen command. |
| The official schedule and artifact bundle are complete. | **Confirm.** | Six discarded warmups and all eighteen measured cells exist in frozen order, with seven required files per cell, nine root artifacts, no failed cells, and no abandoned worker directories. |
| The frozen robust-versus-midpoint rule passed. | **Confirm.** | Independent raw-cell recomputation gives ratio `0.8976637924784316`, robust wins `3/3` on each frame, minimum support `0.9921875`, and maximum measured projection invariant `5.820766091346741e-11`. |
| Robust improves over both placement controls on both scenes. | **Narrow.** | Robust beats midpoint on both scenes. It beats all-view on frame 00008, but is `1.0310499997047535×` all-view on frame 00009 and wins only `1/3` paired seeds there. |
| Float64 was the sole cause of predecessor recovery. | **Narrow.** | This successor completes and passes the same invariant, strengthening the precision diagnosis. The predecessor retained no raw replay tensors, so sole causality remains unproved. |
| The run supports RGB-image, physical-geometry, GPU, speed, topology, production-default, or cross-dataset claims. | **Retire for this attempt.** | This is CPU compact-field development/replication evidence on two outcome-exposed frames from one capture, with topology disabled and timings descriptive only. |

## Protocol, source, and chronology binding

- Locked source commit: `a69337346fbecd156c20211abd638f976e327d62`, committed
  `2026-07-29T23:05:53Z`.
- Run initialized at `2026-07-29T23:06:05.157086Z`, about twelve seconds after the source commit,
  with `source_dirty=false` and empty-diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- The canonical task, prospective review, and compact data seal at the locked commit reproduce the
  task-lock hashes exactly. The compact seal independently validates at 55 files and 8,373,380
  bytes.
- The first warmup published at approximately `23:07:13Z`; the last measured cell published at
  `23:28:01.293Z`. File modification order matches the six warmups followed by the frozen rotated
  measured-arm schedule.

The source checkpoint changes the common field computation to opt-in float64 and adds structured
worker-failure evidence. Arms, anchors, bounds, split, seeds, refit, topology, evaluation,
guardrails, and the decision rule remain matched to the prospectively reviewed successor
protocol.

## Independent outcome recomputation

All quality aggregates below were rebuilt from raw per-view validation rows and raw cell
summaries. Held-out rows are sample-weighted within cells; arm and scene summaries use the frozen
geometric mean. The recomputed producer metrics and predicate match exactly.

| Aggregate | Bounded midpoint | All-view consensus | Source-excluded robust |
| --- | ---: | ---: | ---: |
| Final held-out compact RGB MSE, both frames | `0.026755888660351994` | `0.02580366611690199` | `0.024017792485982237` |
| Median measured wall time, seconds | `52.37958948701271` | `52.65091423000558` | `52.75939034897601` |
| Median measured peak RSS, MiB | `3236.466796875` | `3234.86328125` | `3233.79296875` |

Additional robust-arm root metrics reproduce as:

- placement held-out compact RGB MSE geometric mean:
  `0.024161517300452903`;
- final held-out compact density MSE geometric mean:
  `4.91362609353891`;
- median wall time over all eighteen measured cells:
  `52.66802911751438`.

The resource values are descriptive. The run retained per-cell CPU/thread/CUDA scope and
accounting receipts, but not a replay-complete host and package fingerprint.

### Scene-level comparator behavior

| Scene and stage | Bounded midpoint | All-view consensus | Source-excluded robust |
| --- | ---: | ---: | ---: |
| frame 00008, placement RGB MSE | `0.031230997328780735` | `0.02846811301345287` | `0.023508500116422346` |
| frame 00008, final RGB MSE | `0.027857791252287118` | `0.02989098114351534` | `0.025116770249797894` |
| frame 00009, placement RGB MSE | `0.03150247098044282` | `0.02456729056497608` | `0.02483267394214887` |
| frame 00009, final RGB MSE | `0.025697571337296124` | `0.022275253591567134` | `0.02296690020900861` |

Robust/midpoint final ratios are `0.9016066644456535` on frame 00008 and
`0.8937381633289072` on frame 00009, with robust winning all three seeds on each. Robust/all-view
ratios are `0.8402792176410981` and `1.0310499997047535`, respectively; the corresponding paired
wins are `3/3` and `1/3`. The pooled robust/all-view ratio is `0.9307899264070091`, but that pooled
direction must not erase the frame-00009 reversal.

### Frozen producer rule

| Requirement | Frozen threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Robust/midpoint final RGB geometric-mean ratio | `≤ 0.95` | `0.8976637924784316` | Pass |
| Robust paired wins over midpoint, each scene | `≥ 2/3` | frame 00008 `3/3`; frame 00009 `3/3` | Pass |
| Minimum robust supported-track fraction | `≥ 0.95` | `0.9921875` | Pass |
| Maximum measured source-projection invariant | `≤ 0.0002` | `5.820766091346741e-11` | Pass |

The same guards also pass when discarded warmups are included: the maximum projection invariant
is `7.275957614183426e-11`, and the minimum robust support is `0.984375`.

## Isolation, parity, and artifact integrity

The audit checked every one of the 24 cell input-boundary and resource receipts.

- Every input receipt selects exactly 28 sealed records: shared calibration, one compact
  manifest, and 26 `.rtgsv` files. Current byte counts and SHA-256 values match.
- Every no-image guard passes: image-capable imports and image-suffix opens are denied, forbidden
  modules are absent at exit, all five negative controls pass, alpha loading is false, and the
  declared reconstruction modalities are only calibration and `gaussians2d`.
- Held-out views are exactly indices `[7, 15, 23]` (`C0014`, `C0028`, `C1001`). Optimized views are
  exactly `[0, 3, 6, 10, 14, 18, 21, 25]`, are a subset of the 23 training views, and are disjoint
  from held-out views. Placement and final sample hashes match, and samples match across arms
  within each scene/seed group.
- Anchor digest, source-lineage digest, AABB, and depth bounds match across all three arms in each
  of the six measured scene/seed groups. Normalized configurations differ only where the protocol
  permits seed and placement mode to differ. Float64 compute, 20-step refit, and zero topology
  rounds are preserved.
- All depths remain inside original bounds. There are no placement fallbacks, background tracks,
  sparse-depth anchors, topology proposals, or accepted topology changes.
- All 24 resource receipts record one CPU thread and no CUDA use. Worker fit time is bounded by
  scoped wall time, scoped wall time is bounded by process wall time, and output-byte counts match
  the actual files.

The audit independently recomputed all 48 placement/final validation reports. Each retains the
expected 26 view rows and 512 samples per view; aggregate rows reproduce from those samples.
`cell_results.json` is exactly the raw measured summaries with the scoped wall/RSS fields added,
and the root receipt/history arrays exactly consolidate the eighteen measured cells.

All 48 cell PLYs parse structurally with 128 finite vertices and nonzero quaternions. Initial PLYs
carry 17 float properties at SH degree 0; final PLYs carry 26 at SH degree 1. All 24 histories
contain 21 finite entries and their endpoints and accepted-step counts bind to cell diagnostics.
One history increases exactly at a frozen visibility/gain refresh, where adjacent entries describe
different refreshed objective states; no global-monotonicity claim is made.

Summary semantic hashes bind in-memory tensors and cannot be regenerated byte-for-byte from the
lossy float32 PLY export. This audit instead binds the exact PLY bytes and validates their parsed
structure. Decisive metrics are retained independently of PLY round-tripping.

## Artifact inventory and hashes

Before audit publication, the canonical run root contains 177 producer files totaling 2,269,149
bytes. Its canonical inventory digest is
`404cdeac3453af78f7c2b16c0e603617fb39ddbb0e6841f689c2681c76de325e`. The digest covers sorted
path, byte-count, and SHA-256 records for every file under `cells/` and `warmups/`, plus the nine
producer files at the run root. This selector intentionally excludes any downstream report/viewer
files that may later be added.

| Artifact | SHA-256 |
| --- | --- |
| `task.lock.json` | `f933e481b3e95b6a59c470b3f50dd84b27667a8450e4ff8b0f476d601a3cbc4b` |
| `cell_results.json` | `19d3a3e3df071bb3a6e2c35b024f71e8ffcd01b03189fed77cffa9cbf33733e2` |
| `metrics.json` | `d4d56e387224b04fed2b7b3dc7663bd30b50b8c0200d7f8a5568472b7cbebbe0` |
| root `gaussians_init.ply` | `f520126ff9dfbbff622d33c1ccbe6daa20f6b010157183dcfc9e2e001dcbe3a8` |
| root `gaussians.ply` | `9b673b0ca562e343ccb15d1a50725f74291030f4a6d90ac9ded2d815a14e39b9` |
| root `gaussians.config.json` | `b5c89fc0846a47d1b11bf7156f4a493ef7fada2599a2818e200c372be5c9b3c6` |
| root `training_history.json` | `c93bd8add7c18d92749a9efaa88cff5197938093c56a82487187274034da2833` |
| root `input_boundary_receipt.json` | `c48959040b8863f6af5cd46984221bcdead343ee58cb6160e12cd53d913c9df2` |
| root `resource_receipt.json` | `64ea467b2c7c70196db8cd1b09932a1f9e66339f897ae81997ca01520d1b196d` |
| producer result JSON | `e573cb8c6f00cfc48e18fe3c8fad0f091323faf66f4a98fadb0b835e2b5b565b` |
| producer result Markdown | `c606f304690422bc18200815aa00b4a39c09d0f9f378d4faa5911d8c042a4a7b` |

## Evidence boundary

This is outcome-exposed development and same-capture replication evidence from two named compact
frames with three measured seeds per arm. It supports only the audited three-arm CPU compact-field
placement disposition. It does not establish rendered RGB quality, physical surface accuracy,
GPU behavior, a speed advantage, topology utility, cross-dataset generalization, or a
production-default change.

The field named `source_projection_max_error` combines mean-coordinate discrepancies in pixels
with covariance-entry discrepancies in squared-pixel units. Its frozen numerical gate passes by a
large margin, but the task's single “pixels” unit is dimensionally imprecise; the value is treated
here as a mixed-unit implementation invariant, not as a physical pixel-error claim.

The successful float64 successor is consistent with the predecessor's precision postmortem and
makes that diagnosis more credible. It does not prove that float64 was the sole necessary cause of
recovery because the predecessor did not retain replay-complete tensors or environment evidence.

## Checks performed

The audit independently:

- recomputed task, review, protocol, seal, lock, producer, run-inventory, root-artifact, and locked
  source hashes;
- validated every compact seal and per-cell selected input record;
- recomputed every decisive aggregate, paired win, extremum, resource median, and frozen predicate
  from raw cells;
- verified schedule/cardinality, receipt consolidation, no-image and held-out isolation, parity,
  PLY/config/history integrity, projection/support guards, and root representative artifacts;
- passed 66 focused tests covering observation loading, field inputs, lifter, refit, validation,
  and experiment-driver behavior;
- passed task/data/digest/workflow checks, whitespace validation, and the complete
  `./scripts/verify.sh` suite, including tests, docs-sync, ARA, script-layout, agent-workflow, and
  experiment-contract gates.

The audit did not execute or replay the experiment, mutate any producer artifact, edit source or
the frozen task/protocol, update `docs/EXPERIMENTS.md` or the ARA, render a report, launch a viewer,
interpret visuals, delete a branch, merge, commit, or push.
