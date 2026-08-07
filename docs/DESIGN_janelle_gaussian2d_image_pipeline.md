# Janelle Gaussian2D → 3D Gaussian image-backed pipeline

Status: RTGS-013 executed development pipeline. This is the corrected end-to-end pipeline for the
six owner-selected Gaussian2D folders in `frame_00008`. It supersedes interpreting the RTGS-012
image-free field-proxy run as the requested Janelle image experiment. The immutable run, six child
reports, independent audit, and browser receipts are linked in the observed-outcome section.

## Experiment units

The following folders are six independent experiments, not six arms of one input:

1. `gaussians2d`
2. `gaussians2d_additive`
3. `gaussians2d_gaussianimage_fullres`
4. `gaussians2d_native_fullres`
5. `gaussians2d_structsplat_mask_contained_fullres`
6. `gaussians2d_structsplat_no_boundary_fullres`

Every folder contains all 26 calibrated camera ids. Each experiment uses that folder's own 2D
Gaussian fields, but the same Janelle RGB images, lossless masks, camera partition, pipeline,
optimizer, three seeds, metric definitions, and resource accounting. Results are not pooled to
hide folder-specific behavior; the report creates one child `index.html` for every folder.

## Pipeline

```text
one Gaussian2D folder                     Janelle frame_00008
26 integrity-bound .rtgsv views           26 JPG + 26 PNG masks + calibration
              |                                           |
              +--------------------+----------------------+
                                   |
                                   v
0. byte/hash/view/camera alignment and fixed T/V/H split
                                   |
                         optimizer cameras only
                                   |
                                   v
1. deterministic structural carrier from the complete 2D fields
   (only when a source field exceeds the frozen CPU carrier budget)
                                   |
                                   v
2. support-aware inverse-projection field lift
   masked: source-backed packed alpha | unmasked: alpha ignored
                                   |
                            initial Gaussians3D
                                   |
                                   v
3. standard image-supervised 3DGS refinement on Janelle JPGs
   masked: PNG matte loss            | unmasked: full RGB-canvas loss
   Trainer object: optimizer cameras only; built-in evaluation disabled
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
4. fixed validation curves                    5. final held-out evaluation
   no selection or stopping                       opened only after fitting
             |                                           |
             +---------------------+---------------------+
                                   |
                                   v
6. PLYs, calibrated previews, metric/resource curves,
   per-folder index.html, and masked/unmasked orbit viewer
   (all viewers launch only after every measured endpoint is closed)
```

## What to expect from each part

| Part | What it does | What a good outcome looks like | Cost | What can go wrong |
|---|---|---|---|---|
| 0. Alignment | Binds the compact manifest, every listed `.rtgsv`, calibration, JPG, PNG, view id, and split. Verifies that compact source hashes name the same Janelle image/mask bytes. | All six inputs report exactly 26 matching ids and no held-out camera is present in a fitting object. | Hashing is I/O-bound and paid before the measured cells. | A stale hard link, mismatched camera, incomplete folder, or changed byte aborts before optimization. |
| 1. Structural carrier | Loads every source component and records the original counts. Fields above the frozen budget use the existing deterministic 8×8 spatially stratified, mass-area selection for the CPU lift only. | Selection is identical for masked/unmasked arms and all seeds, with original/used counts and an index digest in every receipt. | The lift scales with the carrier instead of the 640–100k raw range. | Low-mass fine detail can be omitted. This approximation is explicit and prevents complete-field-fidelity claims. |
| 2. Field lift | Places source-exact inverse-projection fibers, enforces multi-view support, and performs a short analytic field refit. Mechanisms that failed RTGS-012's synthetic gates—transport association, candidate topology, and progressive scheduling—remain off. | A finite, orbitable initialization in calibrated world coordinates with valid covariance, opacity, and train-only provenance. | CPU float64; expected to be small compared with 3DGS but sensitive to source count and view count. | Sparse overlap, invalid source rays, or inconsistent decomposition semantics can make placement fail. There is no silent random substitute or outcome-driven fallback. |
| 2a. Masked lift | Uses the lossless packed alpha already bound to the Janelle PNG and its calibrated undistortion/crop. | Fewer obvious background tracks and better initial silhouette support. | Extra mask sampling is small. | Thin fabric boundaries can be suppressed; hard support is not uncertainty-aware. |
| 2b. Unmasked lift | Does not decode or consume alpha for placement. | A clean control showing how much geometry the field/camera evidence carries alone. | Similar lift cost. | Background/clutter fields can spend the limited 3D track budget. |
| 3. RGB refinement | Runs the repository's standard CUDA gsplat `Trainer` from the lifted PLY, with the same density schedule, SH schedule, iteration budget, and seed pairing for both arms. The object passed to `Trainer` contains the 20 optimizer cameras only. Its built-in checkpoint evaluation and CUDA-peak reset are disabled; one external observer owns the validation views. | Validation foreground PSNR rises, topology grows within the hard cap, and final held-out renders resemble the actual Janelle photographs. | Dominant GPU stage. Native optimizer elapsed excludes initial validation, checkpoint snapshot construction, every validation render, final serialization, held-out evaluation, previews, reports, and viewers. | A weak 3D start may converge slowly or densify the wrong support. Full-canvas and masked objectives optimize different targets, so both foreground and background metrics must be shown. |
| 3a. Masked RGB | Uses Janelle PNGs for foreground-weighted RGB, alpha supervision, and exterior-alpha penalty. | Better foreground/crop quality, alpha IoU, and lower exterior leakage. | Same iteration count; masked SSIM crops and alpha losses add work. | Black-matte supervision can reduce fidelity to the photographed stage background and may make full-canvas PSNR look worse. |
| 3b. Unmasked RGB | Trains against the complete Janelle JPG without a matte. | Better reproduction of the photographed full canvas if capacity is sufficient. | Same fixed iteration budget. | Background consumes Gaussian capacity; foreground metrics and silhouette may suffer. |
| 4. Validation curves | Three cameras removed from the task's training set are evaluated at fixed checkpoints. They never select a checkpoint, stop a run, or alter hyperparameters. | Comparable quality-versus-step, native-optimizer-time, and actual worker-cell-wall-time curves, plus a time-normalized PSNR AUC. | Observer time is measured separately and excluded from native optimizer elapsed time. | Calling this test evidence would be leakage; it is validation only. Mixing the observer-excluded optimizer clock with actual stage-wall boundaries would also misstate convergence. |
| 5. Held-out evaluation | Loads `C0014`, `C0028`, and `C1001` only after the endpoint is saved. Reports foreground/full/crop image metrics and alpha behavior per view and in aggregate. | Consistent multi-seed gains rather than one favorable camera or seed. | Three final renders per model. | Test outcomes cannot trigger repair or representative-model selection. Any such use invalidates the experiment. |
| 6. Presentation | Saves every cell, representative calibrated comparisons, seed curves, resource diagrams, and a synchronized masked/unmasked orbit viewer. All six viewer servers and browser windows launch only after all 36 resource endpoints and previews are closed. | Six canonical child pages and six inspectable viewers whose labels disclose folder, arm, seed, and counts; a six-entry browser receipt proves each child loaded and each WebGL viewer rendered and orbited. | Preview rendering, GIF encoding, report rendering, and viewer activity occur after measurement. | Orbit views are qualitative; they do not replace calibrated held-out metrics. Deferring all viewers means the six folders become inspectable together at the end, preventing interactive load from contaminating later timings. |

## Frozen camera roles

- Optimizer/lift cameras (20): `C0001`, `C0004`, `C0005`, `C0006`, `C0008`, `C0012`,
  `C0018`, `C0019`, `C0020`, `C0021`, `C0025`, `C0026`, `C0029`, `C0030`, `C0031`,
  `C0034`, `C0039`, `C1000`, `C1002`, `C1004`.
- Validation cameras (3): `C0009`, `C0022`, `C0037`.
- Final held-out cameras (3): `C0014`, `C0028`, `C1001`.

The task contract's `train` partition is the union of optimizer and validation views because the
repository schema has two top-level roles. The executable boundary separates the two before either
the field lifter or RGB trainer is constructed.

## Metrics and interpretation

Each child page shows every measured seed for masked and unmasked arms:

- held-out foreground and full-canvas PSNR;
- held-out foreground-crop SSIM;
- held-out alpha IoU and exterior-alpha leakage;
- validation foreground PSNR versus step and elapsed optimizer time;
- time-normalized validation PSNR AUC;
- training loss and Gaussian count versus step/time;
- field-lift, RGB-refinement, validation-observer, and total wall time;
- peak CUDA allocated/reserved bytes, peak process RSS, worker-start/input-open endpoint clocks,
  and final Gaussian count.

Foreground PSNR/crop SSIM answer object reconstruction quality. Full-canvas PSNR answers a
different question because the masked arm is not asked to reproduce the stage background.
Silhouette metrics diagnose support. Convergence claims require matched time-to-quality or AUC;
final quality alone is not convergence speed. GPU timings are development measurements on the
named local GPU, not real-time or cross-hardware claims.

## Execution integrity and exact timing boundaries

The protected coordinator is the only surface allowed to create a measured worker. For every
cell it writes an HMAC-authenticated ticket whose dataset, arm, seed, mode, iteration count,
canonical output path, task-lock digest, protocol digest, prospective-review artifact digest, and
data-seal digest are immutable. Coordinator and ticket-worker entry both require the approved
canonical review to remain a regular file with exactly the bytes hashed by `init-run`. The secret
exists only in the parent process environment. The public scratch surface always takes the warmup
branch and therefore cannot reach held-out loading; the former free-form `--worker` CLI is not
accepted.

Every completed worker writes a content-addressed `cell_receipt.json`. It binds exact identity,
split, effective configuration, compact/image/camera input digests, and hashes of all required
cell artifacts. Resume and aggregation both replay this validation. The root
`cell_bundle_receipt.json` then binds the warmup and all 36 measured receipts transitively, so a
copied seed, copied arm, stale folder, edited summary, or modified PLY fails closed. The canonical
bundle checker repeats the exact schema, strict JSON types, frozen iteration/split/effective-config
digests, input bindings, mode-specific artifact inventories, summary endpoint semantics, and every
artifact hash; it does not stop at transitive byte presence.

The behavior-bearing source binding covers all `src/rtgs/**/*.py`, the experiment driver, the
experiment contract, and the result-bundle checker. It is verified by the coordinator, every
official worker, and final bundle validation. The complete 215-file data seal is rehashed at
coordinator entry and exit and again by bundle validation; every worker also hashes each compact,
JPG, PNG, manifest, and calibration file it actually opens. Compact/image camera agreement binds
rotation, translation, dimensions, focal lengths, and principal point after exact downscaling.

CUDA peak counters reset once, immediately before compact-field access. `Trainer` is forbidden
from resetting them. Peak allocated/reserved bytes, peak RSS, and total cell wall time freeze
after the synchronized final PLY/NPZ save and before any held-out image is hashed or loaded.
Validation native-time curves use `Trainer.history["elapsed"]`, which contains optimizer work but
no internal evaluation. Worker-cell-wall curves use separately captured callback-cumulative and
cell timestamps so they align with actual stage boundaries. The separately receipted observer
interval includes the initial validation, checkpoint snapshot construction, synchronization, and
external validation renders. Both clocks are named explicitly on every child page.

All post-matrix work—exit integrity verification, cell replay, previews, aggregate sources,
viewer launch, canonical RESULT publication, and the success receipt—shares one terminal failure
boundary. A late exception rewrites the root into a schema-valid failed report while retaining the
validated 36 cells. Canonical RESULT publication is exact-content idempotent, and a retry reuses
healthy canonical viewer processes while relaunching only failed entries. The success receipt is
written last. After report rendering, `viewer_smoke.json` schema v2 must contain six ordered
entries, each bound to its child `index.html`, exact viewer argv, HTTP 200/local-link pass, WebGL2
renderer, visible non-background pixels, clean client errors, and an exercised orbit camera.

## Fail-closed rules

- No held-out compact field, JPG, or mask enters lifting or gradient computation.
- No validation or held-out value changes an endpoint or setting.
- No failed cell receives imputed metrics or a model borrowed from another folder/arm/seed.
- No mask failure silently retries unmasked under a masked label.
- No random initializer silently replaces a failed field lift.
- No official worker accepts free-form identity, iteration, or output arguments.
- No existing cell resumes unless its exact identity/config/input/artifact receipt revalidates.
- No source or input byte drift is tolerated between prospective review, entry, exit, and bundle
  validation.
- No missing or changed prospective-review artifact can authorize coordinator or worker entry.
- No partial RESULT publication, preview exception, or viewer-start failure can leave a canonical
  success report; the same locked run root remains retryable from validated cells.
- No completed bundle passes until all six child pages and all six viewers have browser-side smoke
  attestations.
- A complete root requires all six folders × two arms × three seeds.
- The earlier RTGS-012 field numerator/density errors remain mechanism diagnostics only and are
  never relabeled PSNR, SSIM, RGB reconstruction quality, or Janelle image evidence.

## Observed outcome

The exact approved 36-cell matrix completed without a failed measured cell and passed independent
semantic, metric, accounting, split-isolation, and transitive-hash replay. For each folder
separately, the joint masked arm had favorable held-out foreground PSNR, crop SSIM, alpha IoU,
exterior-alpha leakage, and validation foreground-PSNR AUC in all three paired seeds. It had lower
full-canvas PSNR in all three seeds for every folder, matching the expected photographed-background
tradeoff. Median paired masked-minus-unmasked effects were:

| Folder | FG PSNR (dB) | Crop SSIM | Alpha IoU | Exterior alpha | Full PSNR (dB) | Validation AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussians2d` | +11.952 | +0.0975 | +0.8043 | -0.9569 | -8.623 | +8.043 |
| `gaussians2d_additive` | +10.490 | +0.0927 | +0.7803 | -0.9521 | -7.974 | +5.367 |
| `gaussians2d_gaussianimage_fullres` | +10.195 | +0.0745 | +0.7872 | -0.9008 | -9.012 | +8.067 |
| `gaussians2d_native_fullres` | +6.110 | +0.0402 | +0.7409 | -0.8902 | -9.496 | +5.144 |
| `gaussians2d_structsplat_mask_contained_fullres` | +11.263 | +0.0947 | +0.8055 | -0.9275 | -8.903 | +7.580 |
| `gaussians2d_structsplat_no_boundary_fullres` | +11.693 | +0.0941 | +0.7695 | -0.9195 | -8.253 | +7.869 |

These are end-to-end arm effects, not evidence that separately identifies the contribution of the
lift mask or the RGB-loss mask. The six folders are not ranked or pooled. The frozen protocol did
not define a threshold-crossing metric, so the proposed fixed-quality convergence comparison is
retired for this run; timings and memory remain host-local descriptions only.

- Root report: `runs/20260806_gaussian2d_image_refinement_janelle_frame00008/index.html`
- Per-folder reports: `runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/<folder>/index.html`
- Independent audit:
  `benchmarks/results/20260806_gaussian2d_image_refinement_janelle_frame00008_AUDIT.md`
- Browser receipts:
  `runs/20260806_gaussian2d_image_refinement_janelle_frame00008/{viewer_smoke,report_browser_smoke}.json`
