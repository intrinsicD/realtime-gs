# Independent audit: masked StructSplat example under 168,000 bytes

Date: 2026-07-18 (Europe/Berlin)

Disposition: **PASS for a single-view archive-integrity and byte-cap feasibility example only.**
Foreground quality is measured but has no preregistered acceptance threshold. Mask-free silhouette
reconstruction, whole-dataset compliance, and subjective visual sufficiency are not established.

## Claim audit

| Claim | Disposition | Independently checked evidence |
|---|---|---|
| One 5,000-Gaussian archive fits under 168,000 bytes | Confirm for C0014 only | 150,492 bytes; 17,508-byte margin; SHA-256 `f07f6b777aa4d3c9c0d2cda7970d0f368dc6a84fa8bf3b664c1dcb85f1318ead` |
| Archive is materially smaller than its source | Confirm narrowly | 98.46x smaller than the 14,817,975-byte JPEG; 99.14x versus JPEG plus mask. The comparison partly reflects intentional background removal. |
| Strict archive has no RGB or mask arrays | Confirm | Strict reload passed; members are `means`, `mean_residuals`, `log_scales`, `rotations`, `colors`, `amplitudes`, and `metadata_utf8`. |
| Native archived playback is reproducible | Confirm for archived playback | Independent rerender reproduced tensor SHA-256 `27c07828779d2619740c6408cbf3d0698a83a4b570eef7d4cc48035799a29f52`. The recorded terminal live field is no longer independently available. |
| Foreground reconstruction metrics | Confirm for this view | Clamped PSNR 36.878836 dB; foreground-weighted SSIM 0.9019588. |
| Mask gating keeps centers on foreground | Confirm only for the rounded-center invariant | Independent check found zero rounded centers outside the mask. |
| Applying the mask eliminates the need for mask information | Retire for silhouette/background separation | 31.18% of outside-mask crop pixels exceed one 8-bit RGB code; full masked-crop playback is 17.875642 dB / 0.7297999 SSIM. |
| Every dataset view can be converted below the cap | Not established | One selected training view only; no batch conversion or adaptive count backoff was run. |
| Quality is good enough | User decision pending | No quality threshold was preregistered, and raw silhouette leakage is decision-relevant. |

## Rechecks

The audit independently:

- checked preregistration/harness/output chronology (11:03, 11:06, and 11:10 local time);
- hashed the archive, panel, record, current harness, and preregistration;
- strict-loaded the archive and recomputed its member allowlist, component count, cap margin, and
  source-size ratios;
- rerendered the strict archive with current StructSplat `cuda_tiled` CUDA playback;
- recomputed foreground and full-crop PSNR/SSIM from the calibrated source and PNG mask; and
- checked every rounded crop-local center against the undistorted foreground mask.

The independent rerender returned foreground PSNR/SSIM `36.878836478167074 / 0.9019588232040405`,
full masked-crop PSNR/SSIM `17.875642205588598 / 0.7297998666763306`, zero outside rounded centers,
and the recorded archive-render hash.

## Findings and required wording

1. **Mask-free playback does not preserve the silhouette.** Center membership is not support
   containment. Gaussian support crosses the boundary substantially even though every center is
   in foreground.
2. **`PASS` is a protocol-gate result, not visual acceptance.** Public wording must say that
   archive-integrity feasibility passed and user visual sufficiency is pending.
3. **Provenance is artifact-verifiable, not replay-complete.** Inputs, provider source digest,
   dirty provider status, and loaded extension are bound. The result does not self-bind the exact
   executed realtime-gs revision/diff, harness SHA, full environment, or GPU model. The current
   harness SHA is `14362db0e7d775f74b13bd910cab860119e233b870735ae97d86a4dcb6f19741`.
4. **Panel semantics must stay explicit.** Native render values extend outside `[0,1]`; the panel
   clamps for display. Its “raw” cell means no source-mask compositing, not an unclamped numeric
   visualization.
5. **No visual referee claim.** The image-view helper failed in its filesystem sandbox. The audit
   verified panel SHA-256 `6d7531915f427ff8c5bc1c77ddd3b5ba714e6a5beae6dc576352ab17b1dd88ea`
   and generation path, but subjective appearance remains for the user to inspect.

No default, roadmap item, 3D capability, novel-view claim, dataset-wide conversion claim, or GPU
performance claim is supported by this run.
