# Image-backed masked and unmasked 3D Gaussian refinement — independent results audit

Date: 2026-08-07 (Europe/Berlin)  
Auditor: `Volta-protocol_review`  
Verdict: **ACCEPTED FOR BOUNDED PER-FOLDER DEVELOPMENT INTERPRETATION AND REPORT RENDERING**

## Referee disposition

The immutable scientific cells are internally consistent and satisfy the frozen task boundary. All
36 measured cells completed, the held-out-free warmup is separate, every task-specific cell receipt
and transitive artifact hash replays, and independent aggregation reproduces the producer's
per-folder results. The result may therefore be rendered and cited only as six separate,
development-only comparisons on one Janelle frame.

Within every folder, the joint masked arm has favorable held-out foreground PSNR, crop SSIM, alpha
IoU, exterior-alpha leakage, and validation foreground-PSNR AUC in all `3/3` paired seeds. The same
arm has lower full-canvas PSNR in all `3/3` seeds, confirming the prospectively expected background
tradeoff. These are end-to-end arm effects: this factorial does not distinguish the contribution of
mask-aware field placement from masked RGB refinement.

No folders are pooled or ranked. The proposed fixed-threshold/capacity/boundary convergence claim
is retired because the protocol froze AUC, not a threshold-crossing statistic, and expressly makes
each folder an independent result unit. Timing, RSS, and CUDA values are descriptive only: there is
no idle-host/load receipt, execution order was fixed, and the arms end with different Gaussian
counts.

## Claim inventory and disposition

| Claim | Disposition | Independently checked evidence |
| --- | --- | --- |
| The result is bound to the approved task, review, data, source, command, warmup, and complete 36-cell matrix. | **Confirm, development scope.** | Exact task/review/data/source/lock hashes match; one held-out-free warmup and 36/36 unique measured cells pass strict semantic receipt and transitive artifact-hash replay. |
| The joint masked arm improves held-out foreground PSNR, crop SSIM, alpha IoU, and exterior leakage. | **Confirm, per folder and end-to-end only.** | The favorable direction holds in `3/3` paired seeds separately for every one of the six folders. The design cannot attribute the effect to either masking substage alone. |
| Masking trades full-canvas background fidelity for foreground/silhouette quality. | **Confirm, per folder.** | Full-canvas PSNR is lower for masked in `3/3` paired seeds of every folder; median paired differences are -7.974 to -9.496 dB. |
| Masking improves validation convergence. | **Narrow to frozen AUC.** | Time-normalized validation foreground-PSNR AUC is favorable in `3/3` seeds of every folder, and the observer-excluded clocks/AUC replay. This is not threshold-crossing evidence. |
| Higher-capacity or boundary-aware folders reach a fixed quality threshold sooner. | **Retire for this protocol.** | No threshold/crossing metric was frozen or published; AUC is not equivalent, and cross-folder pooling/ranking is forbidden. |
| The masked arm is generally faster or more memory efficient. | **Narrow to descriptive run diagnostics.** | Resource mappings replay, but host idleness/load is absent, arm order is fixed, endpoint capacities differ, and evidence is one host/frame. No speedup, real-time, or general performance claim follows. |
| The carrier preserves every source component or complete field fidelity. | **Retire.** | Receipted bounded carrier reduction is part of the protocol; image metrics do not establish complete-field preservation. |
| Viewer failure/retry and visual usability are established. | **Narrow to launch recovery.** | Final receipt has six exact `reused=true`, live, HTTP-ready processes after all endpoints; audit probes returned six HTTP 200s. The overwritten first incomplete receipt prevents exact replay of its failure, and WebGL/orbit smoke is still pending. |
| The run establishes SOTA, GPS-Gaussian reproduction, cross-scene generality, a production default, or real-time behavior. | **Retire.** | There is one development frame, no external method comparator, no confirmatory scene, and no idle-host throughput protocol. |

## Binding, chronology, and artifact integrity

- Task SHA-256: `b738b7957fc09ca3ca22d2a2295a65311e43b18e0b1d8a4ae260b3e53e63e252`.
- Protocol SHA-256: `61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b`.
- Prospective review SHA-256: `b243f8eef02c3c660b44a5f952dd37f824a7fa00d45d47422f6b211321ed37f1`; outcome access was `none`.
- Data-seal SHA-256: `1199a410a7070e23126d51c55f5f5039cd0f505ff3f2a8a9b0d8e503b4ac5a63`, covering 215 files and 490,153,435 bytes; `validate-data` passed.
- Result-producing source binding: 103 files, SHA-256 `b10eb15c38bd44da97ad42464870fee64eb5a158f722e3b2cf3a6a1d77f4445a`.
- Task-lock SHA-256: `4e0d9ee39682c20446e74e4ecc383775a9545fb3888a55b0cc3f43d3568e1933`.
- Producer RESULT JSON/Markdown SHA-256: `9c273c1f2acef1f322b0a71191b0fc6a488805baaf31c62490a3a0a64fc2f938` / `d39fb7b11227011adcb69ab412350340708e5fd1e3527dd2f072b3d6d21c2a16`.
- Exact command: `.venv/bin/python scripts/experiments/20260806_gaussian2d_image_refinement_janelle_frame00008.py --task experiments/tasks/20260806_gaussian2d_image_refinement_janelle_frame00008.json --run runs/20260806_gaussian2d_image_refinement_janelle_frame00008`.

The lock was created at `2026-08-06T22:12:00.993018+00:00`. The held-out-forbidden warmup ran
from `22:12:15.140643` to `22:13:15.162765`; measured cells ran from the first start at
`22:13:16.705916` through the last finish at `22:53:33.133714`. The completed viewer receipt and
run receipt followed at `22:57:24.468087` and `22:57:24.469284`, respectively.

The final same-root retry revalidated source and the full data seal at `22:55:46`/`22:55:47`, but
that retry overwrote the coordinator's initial-attempt entry/exit receipt. Therefore the initial
coordinator-entry timestamp is not durable evidence. This does not make cell inputs ambiguous:
every worker's frozen input digests, effective configuration, summary identity, and required
artifacts are sealed in its cell receipt, and all 37 cell bundles replay against the locked source,
data, partitions, and current bytes.

Before audit publication the run contained 1,678 files and 356,841,676 bytes; the canonical
inventory digest was `e372bb768270b76ba64fe8015247fd8f233676d301ee5e6489ea81ceb6b1da4f`.
Excluding the six live viewer logs, which can append during later presentation access, the 1,672-file
scientific snapshot digest was `f8bd5b2a2e27d7339f524b218b64306c1d4d86b25eea4c7a2027584b3be7dc82`.

## Independent recomputation

Held-out aggregates were recomputed as means of the three fixed per-view records. Validation AUC
was independently trapezoid-integrated over the observer-excluded native optimizer clock. All
per-folder arm medians and paired seed differences/wins were then recomputed directly from the 36
cell summaries. Aggregate, median, clock, and resource replay is exact; maximum AUC roundoff is
`7.105427357601002e-15`. NPZ mean counts and PLY vertex headers agree with summary counts for every
initial and final field.

The table reports the median of the three within-seed `masked - unmasked` differences. Positive is
favorable for foreground PSNR, crop SSIM, alpha IoU, and AUC; negative is favorable for exterior
alpha. Full-canvas PSNR is intentionally shown as the adverse tradeoff. Every displayed direction
holds in `3/3` seeds for that folder.

| Folder | FG PSNR Δ dB | Crop SSIM Δ | Alpha IoU Δ | Exterior alpha Δ | Full PSNR Δ dB | Validation AUC Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussians2d` | +11.952 | +0.0975 | +0.8043 | -0.9569 | -8.623 | +8.043 |
| `gaussians2d_additive` | +10.490 | +0.0927 | +0.7803 | -0.9521 | -7.974 | +5.367 |
| `gaussians2d_gaussianimage_fullres` | +10.195 | +0.0745 | +0.7872 | -0.9008 | -9.012 | +8.067 |
| `gaussians2d_native_fullres` | +6.110 | +0.0402 | +0.7409 | -0.8902 | -9.496 | +5.144 |
| `gaussians2d_structsplat_mask_contained_fullres` | +11.263 | +0.0947 | +0.8055 | -0.9275 | -8.903 | +7.580 |
| `gaussians2d_structsplat_no_boundary_fullres` | +11.693 | +0.0941 | +0.7695 | -0.9195 | -8.253 | +7.869 |

The retained raw artifacts contain per-view metric values, not held-out render pixel arrays. The
audit therefore validates aggregation and accounting but does not claim an independent pixel-level
rerender or replay of the PSNR/SSIM/alpha kernels.

## Isolation and frozen endpoint

All six units use the same disjoint 20 optimizer / 3 validation / 3 held-out cameras. Across all 36
cells, field lifting opens exactly the optimizer compact fields; validation and held-out compact
fields remain unopened, and the lifter opens no RGB or mask file. Every one of 1,500 sampled RGB
training indices maps to the 20 optimizer cameras. Validation records occur every 100 optimizer
steps plus step zero and do not enter the native clock. Every frozen RGB endpoint, PLY/NPZ save, and
resource snapshot precedes held-out image access and presentation.

## Resource and timing accounting

The run used an NVIDIA GeForce RTX 3050 (8,192 MiB, driver 590.48.01), CUDA 12.8, Torch 2.9.0,
gsplat 1.5.3, and Python 3.12.9. Each cell is a fresh process; CUDA peaks reset before compact input
access and freeze at the synchronized endpoint. Native optimizer time excludes initial/checkpoint
validation observers, and observer sums, history clocks, stage clocks, CUDA peaks, RSS, and root
resource records all replay exactly.

For transparency, these are median paired differences only—not performance claims:

| Folder | Field lift Δ s | Native optimizer Δ s | Endpoint Δ s | CUDA allocated Δ MiB | Final Gaussians Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gaussians2d` | -0.335 | -4.029 | -4.622 | -32.17 | -23688 |
| `gaussians2d_additive` | -14.877 | -3.379 | -18.708 | -26.50 | -16060 |
| `gaussians2d_gaussianimage_fullres` | -7.484 | -3.905 | -11.457 | -21.31 | -14254 |
| `gaussians2d_native_fullres` | -9.057 | -3.066 | -12.230 | -19.70 | -11885 |
| `gaussians2d_structsplat_mask_contained_fullres` | -1.266 | -2.004 | -3.199 | -6.44 | -1288 |
| `gaussians2d_structsplat_no_boundary_fullres` | -37.120 | -3.722 | -40.825 | -29.67 | -20190 |

There is no host-load or idle-state receipt, masked always precedes unmasked within a seed, final
capacities are not matched, and all evidence comes from one host and frame. These values may be
plotted as run diagnostics but may not be called speedups, real-time throughput, or general memory
superiority.

## Viewer recovery and remaining presentation evidence

The completed launch receipt has the exact six dataset commands and ports 8400–8405, states that
launch followed all measurement endpoints, and records every process as `reused=true`, alive, and
HTTP-ready. An audit-time probe returned HTTP 200 on all six ports. The focused retry test also
passes and proves the bound code writes an incomplete receipt and raises before a later same-root
retry reuses valid viewers.

The actual first incomplete launch receipt was overwritten by the successful retry, so its exact
per-view failure fields and timestamp cannot be independently recovered. The operational claim is
therefore limited to final same-root reuse plus code-path evidence. No root/child report has yet
been rendered, and no browser receipt yet proves visible WebGL content or orbit-camera motion.

## Evidence boundary and downstream work

This audit supports six separate development-only end-to-end masked/unmasked comparisons, their
foreground/full-canvas tradeoff, their frozen validation-AUC directions, and structural operability
of the exact run. It does not support pooling, folder ranking, fixed-threshold convergence,
substage attribution, complete source-field fidelity, cross-scene generality, SOTA/GPS-Gaussian
comparison, production defaults, or real-time/performance claims.

Checks performed: complete data validation; exact task/review/lock/source/result binding; strict
semantic and artifact-hash replay of 37 cell bundles; independent metric/AUC/median/pair/resource/
count/isolation replay; all 25 focused Janelle protocol tests; `git diff --check`; and six HTTP
probes. No GPU rerender, report render, manifest check, visual browser smoke, or full repository
verification was performed in the audit session.

The Driver may now render the frozen root and six child reports, generate/check the manifest, run
structured visible-content/orbit browser smoke, update append-only docs with only these bounded
dispositions, and run repository verification. Those presentation and closeout steps must not
rewrite the producer RESULT or this AUDIT evidence.
