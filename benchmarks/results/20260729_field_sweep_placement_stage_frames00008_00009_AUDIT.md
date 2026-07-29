# Fixed-anchor compact-field sweep — independent failure audit

Date: 2026-07-30 (Europe/Berlin)  
Auditor: `Codex-results-auditor`  
Verdict: **CONFIRMED FAILED BEFORE MEASURED OUTCOMES / POSTMORTEM NARROWED**

## Referee disposition

The official task is consumed and inconclusive. Its first discarded warmup failed during refit
step 0, before a warmup cell could publish and before the driver could reach either placement or
final held-out validation. No measured arm ran, so the preregistered comparison, quality,
resource, geometry, generalization, GPU, and default questions remain unanswered.

| Claim | Disposition | Independently checked evidence |
| --- | --- | --- |
| The attempt was bound to the approved protocol and clean source. | **Confirm.** | The lock binds task SHA-256 `a08e71…eefe`, protocol `9ff705…2844`, review SHA-256 `3e6a0c…2edb`, data-seal SHA-256 `571e71…feb7`, clean commit `ec5735…1400`, and the exact frozen command. The same task, review, and seal bytes exist in that commit. |
| The first official cell was the frame-00008 bounded-midpoint warmup at seed 290900. | **Confirm.** | The locked driver schedules all warmups before measured jobs and orders frame 00008 / bounded midpoint first. The failure receipt names that cell. |
| The run failed before measured or held-out outcomes. | **Confirm.** | The run has no `cells/`, completed warmup directory, metrics, PLY, history, config, boundary/resource receipt, page, or preview. Both retained temp directories contain only the generic 71-byte failure marker. Locked control flow calls held-out validation only after `fit_field_fibers` returns; the exception arose inside that function. |
| Native-scale float32 round-off caused the absolute invariant failure. | **Narrow to strongly consistent, not replay-complete.** | The recorded covariance differences are 1, 0, 1, and 3 float32 ULPs, with maximum absolute error `0.0234375` and maximum relative error `2.80298e-7`. The frozen `0.0002` gate is below one ULP at each non-small recorded covariance scale. However, neither replay preserved raw tensors, stderr, or a diagnostic receipt, so this audit cannot prove round-off was the sole cause. |
| Source-excluded robust sweep improves compact-field reconstruction. | **Unresolved.** | No measured cell or treatment arm completed. |
| The run supports CPU time, memory, RGB, geometry, GPU, or default claims. | **Retire for this attempt.** | No scoped resource receipt or outcome bundle exists. |
| The same official task may be resumed or rerun. | **Reject.** | Its canonical run root and attempt are consumed. A repair requires a new task id, source state, protocol digest, and prospective review. |

## Protocol, source, and chronology binding

- Locked source commit: `ec5735d5a549147f64490e57832578e72ae51400`, committed
  `2026-07-29T22:26:42Z`.
- Run initialized: `2026-07-29T22:26:49.546653Z`, seven seconds after the source commit, with
  `source_dirty=false` and the empty-diff digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- First official failure marker: `2026-07-29T22:26:59.997596Z`.
- Outcome-free diagnostic replay marker: `2026-07-29T22:29:12.586355Z`.
- Failure receipt's declared `failed_at`: `2026-07-29T22:29:37Z`; the receipt and producer
  result pair were written at `2026-07-29T22:30:18.749344Z`.

The chronology is consistent with one official failure followed by the producer's diagnostic
replay. The `failed_at` field describes the post-diagnostic disposition rather than the original
crash instant. Both marker files have identical content and SHA-256
`dfb230e63f5a307cec55c6e67163855bc54c172ffd92b53f11b9deef6146e183`, so the failure receipt's
marker hash does not uniquely identify one directory.

The compact seal independently validates at 55 files and 8,373,380 bytes. The task lock itself has
SHA-256 `2a26e533e76084a53447e8e7171e59ece4829afa9007e38cf623f5046f45a113`,
matching the failure receipt.

## Outcome and artifact inventory

The canonical run contains four files only:

1. `task.lock.json`;
2. the first warmup's `WORKER_FAILED.txt`;
3. the diagnostic replay's `WORKER_FAILED.txt`;
4. `attempts/attempt-001/failure.json`.

There are zero completed warmups, zero measured cells, and zero held-out metric records. The
worker failed before it could write even its temporary summary or input-boundary receipt. The
locked driver uses `subprocess.run(..., check=True)`, so the first failed warmup aborts orchestration
before later warmups or any measured job.

Held-out compact teachers were loaded as part of the all-view compact dataset, but the locked
`FieldLifter.fit` invokes both held-out semantic validations only after refit/topology. Because
`fit_field_fibers` raised before returning, no held-out metric was computed or published. No
run-level guard receipt survived, so this failure package is not an artifact-level proof of the
complete no-image boundary; that limitation does not create an outcome claim.

## Independent float32 assessment

The locked invariant computes the largest absolute source-mean or source-covariance round-trip
error and applies `2e-4` for float32. Casting the producer-recorded matrices to float32 gives:

```text
absolute covariance difference
[[0.00048828125, 0.0],
 [0.001953125,   0.0234375]]
```

At target values `5159.26171875`, `19926.5390625`, and `83616.453125`, float32 spacings are
respectively `0.00048828125`, `0.001953125`, and `0.0078125`. The differences are therefore
one, one, and three ULPs. The maximum discrepancy is `117.1875×` the absolute gate but only
`2.802977e-7` relative to the corresponding covariance value. Float32 epsilon times the largest
term is about `0.00996786`, also showing that a scale-independent `0.0002` gate is not meaningful
at this covariance magnitude.

This independently supports the producer's characterization as a native-scale float32
round-trip/gating failure, not evidence about the bounded-midpoint arm or the robust treatment.
It does **not** prove that round-off was the sole implementation defect: the debugger preserved
only rounded displayed values and a generic marker, not the exact tensors, traceback, command,
environment, or diagnostic script/output.

## Evidence hashes

- Failure receipt:
  `f0bf70eef553f3cbde15591a0e382ef8621ff36634a6a432bc69db4049105435`
- Producer result Markdown:
  `0e953c01301038bcc2e56ba513993fd6cde9d5a4c72ce65b6422739088b25cf8`
- Producer result JSON:
  `cc425a241a961a81de6057d95ce59e368dd9a30ddbc3dbac983db61190bf6fb8`
- Locked driver:
  `baab94f1eb34a40c19a8df0f20c20d9704084a607293834b42328e5a41976107`
- Locked field refit:
  `32330b8c1483c58e220a0b91e0204efc6ce4ea3bc09c5d67f210f8ed224faa84`
- Locked field lifter:
  `8164ca12485b53bca2025d1e8bb06597ff86126dec8f084bed319610bbcb0ca7`
- Locked inverse-projection fiber:
  `4e9bd0c62954b2361d2cb79491d97cdfbad29c7c2dd23be0e5f5bff7e75cdd8b`

## Checks performed

The audit read the exact task, prospective review, lock, failure receipt, both failed temp
directories, producer result pair, and locked source at commit `ec5735…1400`. It independently:

- recomputed protocol, task, review, seal, lock, result, failure, marker, and locked-source hashes;
- validated the compact data seal and its file/byte totals;
- inspected locked job ordering, fail-closed subprocess behavior, refit exception location, and
  held-out-validation call order;
- enumerated the complete canonical run tree and absence of measured/result artifacts;
- recomputed float32 absolute differences, relative error, ULP spacing, and gate ratio from the
  recorded covariance.

The audit did not execute this experiment, access a successor experiment, inspect successor
outcomes, open RGB/mask files, or edit the immutable task, review, run, result, source,
`docs/EXPERIMENTS.md`, ARA, or the active task.

