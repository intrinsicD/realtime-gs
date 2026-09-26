# RTGS-025 unregistered Gaussian-only diagnostics (2026-09-26)

**Boundary.** These are exploratory, single-seed, unaudited checks run interactively during the
RTGS-016 review and the RTGS-025 direction decision. They have no experiment contract, no
prospective review, no v2 bundle and no viewer receipt. They are not results-bearing runs under
Hard Rule 7 and support only staging observation O178. Do not cite them as claims.

Source revision: `25dc83d4db3e6e620d2d24bb29d74364cbb654c2` (clean `main`), `.venv`, one RTX 3050
(8 GB), gsplat classic unpacked rasterizer. Scripts are preserved verbatim as `*.py.txt` and were
run from the repository root with `.venv/bin/python`.

| File | What it checks |
|---|---|
| `cam_roundtrip.py.txt`, `haelyn_cam_roundtrip.txt` | Rerenders the RTGS-016 Haelyn reference PLY through `load_calibrated_scene` and the sealed calibration; PSNR against the stored JPEGs and alpha≥0.5 IoU against the stored masks, 16 views. |
| `photometric_control.py.txt` | Plain photometric 3DGS on the 12 RTGS-016 Haelyn training renders; random init in the camera-only bound, fixed count, no densification, SH degree 1, L1/L2 loss. Held-out C0004/C0008/C0012/C0016. |
| `gaussian_only_control.py.txt` | Same trainer, targets decoded only from the `gaussians2d_unmasked` compact views (2048/view) and their cameras. Held-out real renders are evaluation-only. |
| `janelle_demo.py.txt`, `janelle_frame00008_results.jsonl` | Stage `frame_00008`, downscale 8 (666×576), 22 training views, held-out C0009/C0022/C0031/C1004. Targets: premultiplied photographs (control), or decoded compact fields (`gaussians2d_native_fullres` 100k/view, `gaussians2d_structsplat_mask_contained_fullres` 11k/view) multiplied by the compact view's packed alpha. 30,000 Gaussians, random init inside 0.35× mean camera distance of the rig centre, 15,000 updates, seed 0. |

Evaluation caveats: the Janelle held-out foreground PSNR multiplies the prediction by the
held-out ground-truth mask, so excess opacity/halos outside the silhouette (the open C47 issue)
are not measured. The Haelyn and Janelle PSNR means are means of per-view PSNR, not the PSNR of a
mean MSE, and differ in definition from RTGS-016/RTGS-021 report metrics. The field-trained arms
use the compact views' packed alpha, which is mask-derived; a coverage-derived silhouette was
not tested. Previews and PLYs derived from the private dome capture were kept outside the
repository.
