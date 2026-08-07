# Image-backed masked and unmasked 3D Gaussian refinement for six Janelle Gaussian2D fields

Status: **pending independent audit**.

## Boundary

Development-only, single-frame evidence over six pre-existing Gaussian2D decompositions of the same 26-camera Janelle capture. Every folder is an independent experiment unit with identical split, algorithm, seed schedule, and RGB source. The run may compare masked versus unmasked end-to-end behavior and descriptive differences among those six inputs. It cannot establish cross-scene generality, state-of-the-art quality, GPS-Gaussian reproduction, real-time performance, production-default suitability, or that a deterministic bounded field carrier preserves every low-mass source component. Held-out test views are final-reporting only.

## Raw result units

| Folder | Masked held-out FG PSNR | Unmasked held-out FG PSNR | Raw |
|---|---:|---:|---|
| `gaussians2d` | 24.844512 | 13.186609 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d/result.json) |
| `gaussians2d_additive` | 23.923395 | 13.502346 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d_additive/result.json) |
| `gaussians2d_gaussianimage_fullres` | 24.766662 | 14.519087 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d_gaussianimage_fullres/result.json) |
| `gaussians2d_native_fullres` | 23.531640 | 18.341404 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d_native_fullres/result.json) |
| `gaussians2d_structsplat_mask_contained_fullres` | 24.975620 | 13.750930 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d_structsplat_mask_contained_fullres/result.json) |
| `gaussians2d_structsplat_no_boundary_fullres` | 24.039747 | 12.432309 | [JSON](../../runs/20260806_gaussian2d_image_refinement_janelle_frame00008/datasets/gaussians2d_structsplat_no_boundary_fullres/result.json) |

## Interpretation

The producer records the frozen outputs only. The canonical AUDIT files must recompute the metrics and dispose of every claim before these values are cited.
