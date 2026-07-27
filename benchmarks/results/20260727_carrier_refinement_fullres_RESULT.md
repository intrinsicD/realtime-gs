# ADR-002 carrier refinement on full-resolution masked Janelle

Date: 2026-07-27
Base revision: `dd84c28deb33`
Evidence class: outcome-exposed, single-scene, single-seed development and mechanism evidence

Disposition: **reject the frozen short 30/40/30/60 schedule on frame 00008; the ADR's
until-convergence maturation process remains untested. Retain the implementation as an opt-in
research path; change no default and promote no paper claim.**

Scope correction after implementation review: the official matrix is a 160-update mechanism
screen, followed by a uniform 40-update recovery. It does not run fixed topology, clone
maturation, or standard 3DGS to a train-selected convergence criterion. References below to the
“complete schedule” mean only that frozen short instantiation, not the full process described in
ADR-002.

The result bundle is
[`runs/carrier_refinement_fullres_frame00008_20260727`](../../runs/carrier_refinement_fullres_frame00008_20260727/)
and its browser overview is
[`index.html`](../../runs/carrier_refinement_fullres_frame00008_20260727/index.html).
The frozen governing protocol is
[`20260727_carrier_refinement_fullres_PREREG_V4.md`](20260727_carrier_refinement_fullres_PREREG_V4.md).
The independent scientist pass is
[`20260727_carrier_refinement_fullres_AUDIT.md`](20260727_carrier_refinement_fullres_AUDIT.md).

## What was implemented

- `rtgs.lift.carrier_refinement` expands Beam Fusion's immutable CSR contributor lineage into a
  fixed association table. It separately fits a bounded Cholesky SPD covariance, an
  optical-density opacity against the explicitly experimental fitted-2D amplitude proxy, and a
  robust amplitude-weighted SH0 color. Means, associations, and topology are frozen during repair.
- `rtgs.optim.carrier_schedule` implements fixed-topology SH0 warm-up, protected-parent
  clone-only or low-opacity particle growth, higher-order SH expansion with geometry/opacity
  frozen, and ordinary 3DGS handover. `CarrierLineageTracker` reports survival, descendants,
  generations, mean displacement, covariance drift, and opacity drift.
- The classic density controller gained default-off clone-only, covariance-frame jitter,
  child-opacity scaling, and protected-prefix controls. The trainer gained an optional immutable
  density-surgery callback; all incumbent defaults remain unchanged.
- `run_carrier_pipeline` composes Beam Fusion, fixed-track repair, and carrier maturation behind
  typed configs/results. The calibrated loader can load exact named views in caller order so a
  run never has to decode sealed cameras.
- `benchmarks/carrier_refinement_fullres.py` runs the paper-plan matrix and emits bound JSON,
  PLY, metrics, previews, an orbit-viewer manifest, and an HTML overview.

## Frozen setup

- Dataset: masked Janelle
  `dataset/2025_03_07_stage_with_fabric/frame_00008`.
- Native undistorted resolution: **5328 x 4608**; no downsampling.
- Training cameras: `C0001`, `C0014`, `C0028`.
- Validation cameras: `C0031`, `C1000`, `C1002`.
- Sealed report-only cameras: `C1001`, `C1004`; the audit verifies that they were never loaded.
- Seed: `27027`; CUDA gsplat packed rasterization on an NVIDIA GeForce RTX 4090.
- Thirteen arms: Beam only, immediate standard optimization, the schedule with and without clone,
  six requested ablations/variants, random RGB/JPEG controls, and a means-only control.
- Optimized parent arms receive 160 native-resolution updates. Beam-only receives zero.

The paper plan requests Original 3DGS and Original 3DGS with compressed RGB. This capture contains
no COLMAP sparse model, and no COLMAP executable was available, so those baselines could not be
constructed without inventing geometry. The run therefore uses honestly named `random-rgb` and
`random-jpeg-q50` controls; neither is presented as Original 3DGS.

## Endpoint correction discovered by the scientist pass

Nine parent arms performed a non-empty density operation at update 160 and were evaluated
immediately. Their newborn rows had inherited parameters but zero recovery updates, so those
endpoints are not mature quality comparisons. The parent result is preserved unchanged.

The post-outcome recovery protocol
[`20260727_carrier_refinement_recovery_PREREG.md`](20260727_carrier_refinement_recovery_PREREG.md)
loads every optimized parent PLY by hash and gives all twelve optimized arms exactly 40 additional
native-resolution updates with topology fixed. Adam is restarted because optimizer state was not
serialized. The parent PLY round-trip changes foreground PSNR by exactly zero. The recovered
bundle and overview are
[`runs/carrier_refinement_fullres_frame00008_20260727_recovery`](../../runs/carrier_refinement_fullres_frame00008_20260727_recovery/)
and
[`index.html`](../../runs/carrier_refinement_fullres_frame00008_20260727_recovery/index.html).

## Mature development endpoints

Values are medians over the three validation cameras. LPIPS is AlexNet LPIPS on the masked
foreground crop resized to at most 512 pixels. `Beam only` is the unoptimized compact endpoint;
all other rows are the uniform 200-update recovered endpoints.

| Arm | FG PSNR (dB) | Crop SSIM | LPIPS down | Final N |
| --- | ---: | ---: | ---: | ---: |
| Beam only | 11.541 | 0.8774 | 0.3261 | 2,400 |
| Beam -> standard | **18.512** | **0.9529** | **0.1295** | 5,020 |
| Carrier schedule, no clone | 16.429 | 0.9471 | 0.1518 | 7,632 |
| Carrier schedule + clone | 16.939 | 0.9482 | 0.1537 | 8,453 |
| No covariance repair | 18.207 | 0.9514 | 0.1365 | 5,404 |
| No opacity repair | 16.718 | 0.9482 | 0.1498 | 8,719 |
| No warm-up | 17.303 | 0.9492 | 0.1413 | 9,600 |
| Split immediately | 17.274 | 0.9496 | 0.1444 | 6,588 |
| Clone-only | 17.390 | 0.9493 | 0.1463 | 6,000 |
| Particle | 17.552 | 0.9497 | 0.1474 | 6,000 |
| Random RGB | 14.574 | 0.9231 | 0.2435 | 4,792 |
| Random JPEG q50 | 14.576 | 0.9229 | 0.2405 | 4,831 |
| Means only | 17.328 | 0.9451 | 0.2233 | 7,004 |

## Mechanism tests and gates

| Question | Measured contrast | Frozen gate | Disposition |
| --- | ---: | ---: | --- |
| Covariance repair lowers its own whitened reprojection residual | 50.21% reduction | at least 25% | **pass locally** |
| Opacity repair lowers its fitted-amplitude proxy residual | 15.93% reduction | at least 25% | **fail** |
| Frozen 30/40/30/60 schedule beats immediate Beam -> standard | -1.573 dB | at least +0.25 dB | **fail** |
| Clone phase adds value within the schedule | +0.510 dB | at least +0.25 dB | **pass, narrow** |
| Beam means beat matched random means | +2.753 dB | at least +0.25 dB | **pass, exposed scene** |
| JPEG q50 materially changes the random control | +0.0018 dB | absolute 0.25 dB | **tie** |

The complete schedule preserves **94.625%** of original carriers, so its central failure cannot be
explained as carrier deletion. More importantly, removing covariance repair improves the mature
endpoint by **1.268 dB** even though covariance repair passes its local residual gate. This is a
clean local-objective/downstream-quality dissociation. Removing opacity repair costs only
0.221 dB, below the materiality band; removing warm-up gains 0.365 dB. Particle generation is
0.162 dB above clone-only, also inside the band.

## Storage, runtime, and replay limits

The three compact training captures occupy 481,310 bytes. The matched raw RGB+mask inputs occupy
43,608,053 bytes (90.60x compact), while second-generation JPEG-q50+mask inputs occupy 2,584,658
bytes (5.37x compact). These are valid storage counts.

The measured ingestion rates are **not** a controlled comparison: raw timing includes six selected
files plus decoding and undistortion, while compact timing loads all 26 strict containers. Peak
allocated CUDA memory is approximately 6.23--6.25 GiB for optimized arms, but the run has no
idle-GPU baseline or timing repetitions. No speed, bandwidth, or VRAM advantage is claimed.

An exact parent replay has topology-sensitive CUDA variation: maximum foreground-PSNR delta
0.1361 dB and maximum final-count delta 85. The fixed-topology recovery replay holds every count
exactly and has maximum foreground-PSNR delta 0.0000324 dB. This supports the recovery arithmetic
but does not turn a restarted optimizer segment into exact continuation.

## Conclusion

The requested mechanisms are operational and all runnable paper-plan arms completed at native
resolution. The frozen short schedule is not competitive on this development scene: immediate
standard optimization is best, and covariance repair is causally harmful within that short
budget despite solving its own objective. The run does not test the intended converged maturation
pipeline. Visual inspection agrees with the short-run metrics: even the leading models remain
blurry, oversized splat silhouettes rather than paper-ready reconstructions.

The main paper claim remains **unverified**. Every optimized carrier arm consumes raw RGB, Beam
only is poor, and the original SfM baselines are unavailable. No default changes and no claim-ledger
promotion of the paper claim are authorized by this run. The protocol-scoped negative is recorded
as `ara/logic/claims.md` C28 with its single-scene, short-schedule boundary intact.
