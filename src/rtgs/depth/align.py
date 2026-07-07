"""Align non-metric depth predictions to the scene's world scale.

Standard recipe (3DGS depth regularization, Chung et al. 2023, MoGe's ROE solver — see
docs/RESEARCH.md §4): solve ``min_{s,b} sum_i w_i (s * d_pred_i + b - d_ref_i)^2`` over
pixels where a sparse reference depth exists (projected SfM/known 3D points), with one
robust re-weighting pass to reject outliers.
"""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera


def scale_shift_align(
    pred: torch.Tensor,
    ref: torch.Tensor,
    weights: torch.Tensor | None = None,
    robust_iters: int = 1,
) -> tuple[float, float]:
    """Solve for (scale, shift) mapping ``pred`` onto ``ref`` by weighted least squares.

    Both inputs are 1D tensors of matched samples. Returns (s, b) with s > 0 enforced by
    falling back to scale-only when the joint solve degenerates.
    """
    if pred.numel() < 2:
        raise ValueError("need at least 2 samples to align depth")
    w = torch.ones_like(pred) if weights is None else weights.clone()
    s, b = 1.0, 0.0
    for it in range(robust_iters + 1):
        ws = w.sum().clamp_min(1e-8)
        mp = (w * pred).sum() / ws
        mr = (w * ref).sum() / ws
        var = (w * (pred - mp) ** 2).sum() / ws
        cov = (w * (pred - mp) * (ref - mr)).sum() / ws
        if var < 1e-12 or cov <= 0:
            # Degenerate (flat prediction or negative correlation): scale-only fallback.
            s = float((ref.mean() / pred.mean().clamp_min(1e-8)).clamp_min(1e-6))
            b = 0.0
        else:
            s = float(cov / var)
            b = float(mr - s * mp)
        if it < robust_iters:
            resid = (s * pred + b - ref).abs()
            thresh = 2.5 * resid.median().clamp_min(1e-8)
            w = (resid < thresh).float()
    return s, b


def align_depth_to_points(
    depth: torch.Tensor,
    camera: Camera,
    points_world: torch.Tensor,
    robust_iters: int = 1,
) -> torch.Tensor:
    """Align an (H, W) depth map to known world points visible in this camera.

    Projects ``points_world`` (M,3); samples the predicted depth at those pixels; solves
    scale/shift; returns the aligned depth map (clamped positive).
    """
    uv, z = camera.project(points_world)
    valid = (z > 0.01) & camera.in_image(uv, margin=-0.5)
    if int(valid.sum()) < 2:
        raise ValueError("not enough points project into the view for depth alignment")
    uv = uv[valid]
    z = z[valid]
    xi = uv[:, 0].long().clamp(0, camera.width - 1)
    yi = uv[:, 1].long().clamp(0, camera.height - 1)
    pred_samples = depth[yi, xi]
    s, b = scale_shift_align(pred_samples, z, robust_iters=robust_iters)
    return (s * depth + b).clamp_min(1e-4)
