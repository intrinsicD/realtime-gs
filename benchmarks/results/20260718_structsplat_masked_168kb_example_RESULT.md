# Result: masked StructSplat example under 168,000 bytes

Date: 2026-07-18 (Europe/Berlin)

Status: **PASS for single-view archive integrity and byte-cap feasibility; insufficient for
mask-free exact silhouette playback.** Foreground quality is measured, but user acceptance remains
pending because no quality floor was preregistered. This is exploratory real-image evidence from
one training view, not a dataset-wide conversion result or a default-selection experiment.

## Command and bindings

The preregistration was written before outcome access:
`benchmarks/results/20260718_structsplat_masked_168kb_example_PREREG.md` (SHA-256
`ba2fe2b2d8e6f3cee94c67986d9a58509d5d3154d9e34d36a157d200897f5824`). The harness present at
launch and still current is
`benchmarks/structsplat_masked_168kb_example.py` (SHA-256
`14362db0e7d775f74b13bd910cab860119e233b870735ae97d86a4dcb6f19741`). The result does not
self-bind the realtime-gs revision/diff, so the run is artifact-verifiable rather than fully
replay-bound.

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/structsplat_masked_168kb_example.py \
  --out runs/structsplat_masked_168kb_example_20260718
```

The run used StructSplat `0.1.0`, provider source digest
`f468ff325ce8f1d587482c453a522cad9e9a5eea98f53d6b406cc30cb72f77a8`, and the loaded CUDA
extension SHA-256 `7310edec295aabf013232ed5b3a62ed146b153f1df5c57ec9671db2f653fd540`.
The provider checkout was dirty but its full relevant source tree and status are bound in
`result.json`; the timing is therefore diagnostic, not benchmark evidence.

## Result

The exact archive is
`runs/structsplat_masked_168kb_example_20260718/C0014.teacher.npz`, SHA-256
`f07f6b777aa4d3c9c0d2cda7970d0f368dc6a84fa8bf3b664c1dcb85f1318ead`.

- Size: **150,492 bytes**, leaving **17,508 bytes** below the decimal 168,000-byte cap.
- Representation: 5,000 exact StructSplat Gaussians; archive members are only means,
  log-scales, rotations, colors, amplitudes, crop-local mean residuals, and metadata. No RGB or
  mask payload is serialized.
- Source reduction: the 14,817,975-byte JPEG alone is **98.46x** larger; JPEG plus the
  101,222-byte PNG mask is **99.14x** larger.
- Mask constraint: zero rounded centers were outside the foreground initially or finally. The
  optimizer projected 518 unique boundary rows back to the foreground over 23,221 update events.
- Foreground playback: **36.8788 dB** clamped PSNR and **0.901959** foreground-weighted SSIM.
- Exact playback: live terminal and strict-reloaded native `cuda_tiled` renders matched at zero
  maximum and mean absolute error, with render tensor SHA-256
  `27c07828779d2619740c6408cbf3d0698a83a4b570eef7d4cc48035799a29f52`.

The raw no-mask playback exposes the missing alpha/silhouette contract:

- raw render versus the masked crop: **17.8756 dB / 0.729800 SSIM**;
- **31.18%** of outside-mask crop pixels exceed one 8-bit RGB code value;
- finite-support coverage IoU against the source mask is **0.6032**, despite foreground coverage
  recall of **0.99978**;
- 180/5,000 rows have less than half their unnormalized support activity on foreground, although
  none has less than 10% foreground activity.

Therefore mask-gated initialization and fitting successfully prevent background *centers* from
becoming tokens and give strong foreground RGB quality, but a normalized RGB field alone does
not reproduce a hard cutout after the mask is discarded. The comparison panel deliberately
shows both the raw playback and a source-mask-composited foreground diagnostic:
`runs/structsplat_masked_168kb_example_20260718/C0014_comparison.png` (SHA-256
`6d7531915f427ff8c5bc1c77ddd3b5ba714e6a5beae6dc576352ab17b1dd88ea`).
The panel clamps render values to `[0,1]` for display; “raw” means no source-mask compositing.

## Post-run storage diagnostic

This was not part of the preregistered fit. Bit-packing the 4,643,496-pixel crop mask takes
580,437 bytes before compression and **7,226 bytes** with zlib level 9 (5,597 with bzip2; 6,336
with LZMA). The raw sum of the teacher plus zlib mask is 157,718 bytes, leaving 10,282 bytes for
a small manifest/container under the same cap. This suggests an exact lossless 1-bit alpha sidecar
inside one per-view package is feasible for this example, but no production package or all-view
result has been established.

## Artifacts

- Machine-readable record:
  `runs/structsplat_masked_168kb_example_20260718/result.json` (SHA-256
  `fc0f42165a7b4e3ebf170755b1dfbea101876693b1cd9730061edb0f06474a4a`)
- Archive: `runs/structsplat_masked_168kb_example_20260718/C0014.teacher.npz`
- Panel: `runs/structsplat_masked_168kb_example_20260718/C0014_comparison.png`
- Independent audit: `benchmarks/results/20260718_structsplat_masked_168kb_example_AUDIT.md`

## Decision

Proceed to a dataset converter only after choosing the playback contract:

1. **lifting-only foreground observations**: current mask-gated RGB fields are sufficient, and
   the raw silhouette is not a target;
2. **exact cutout playback**: package a compressed 1-bit alpha/mask or add an explicit alpha
   representation; do not claim that foreground RGB splats alone preserve the mask.

Every converted view must be staged, strict-reloaded, natively rerendered, and checked against
168,000 bytes independently. A fixed 5,000-component count is not itself a byte guarantee.
