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
    pred_samples = _bilinear_sample(depth, uv)
    s, b = scale_shift_align(pred_samples, z, robust_iters=robust_iters)
    return (s * depth + b).clamp_min(1e-4)


def align_inverse_depth_to_points(
    inverse_depth: torch.Tensor,
    camera: Camera,
    points_world: torch.Tensor,
    robust_iters: int = 2,
) -> torch.Tensor:
    """Align a relative inverse-depth/disparity map, then convert it to metric depth."""
    uv, z = camera.project(points_world)
    valid = (z > 0.01) & camera.in_image(uv)
    if int(valid.sum()) < 2:
        raise ValueError("not enough observed points to align inverse depth")
    pred_samples = _bilinear_sample(inverse_depth, uv[valid])
    ref_inverse = z[valid].reciprocal()
    scale, shift = scale_shift_align(pred_samples, ref_inverse, robust_iters=robust_iters)
    aligned_inverse = (scale * inverse_depth + shift).clamp_min(1e-6)
    return aligned_inverse.reciprocal()


def align_inverse_depth_to_bounds(
    inverse_depth: torch.Tensor,
    camera: Camera,
    center_world: torch.Tensor,
    extent: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale inverse depth from known object bounds when no sparse points exist.

    This is intentionally a prior, not claimed metric depth: foreground 5/95 percentiles are
    mapped to the far/near faces of the calibrated object volume. Subsequent multi-view
    optimization can move the resulting splats, but initialization is finite and scene-scaled.
    """
    center_depth = camera.project(center_world.to(inverse_depth)[None])[1][0]
    near = (center_depth - 0.5 * extent).clamp_min(0.05)
    far = (center_depth + 0.5 * extent).clamp_min(near + 1e-3)
    values = _valid_values(inverse_depth, mask)
    lo, hi = torch.quantile(values, values.new_tensor([0.05, 0.95]))
    if hi - lo < 1e-8:
        return torch.full_like(inverse_depth, center_depth)
    scale = (near.reciprocal() - far.reciprocal()) / (hi - lo)
    shift = far.reciprocal() - scale * lo
    return (scale * inverse_depth + shift).clamp_min(1e-6).reciprocal()


def align_depth_to_bounds(
    depth: torch.Tensor,
    camera: Camera,
    center_world: torch.Tensor,
    extent: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Affine-align monotonically increasing relative depth to calibrated object bounds."""
    center_depth = camera.project(center_world.to(depth)[None])[1][0]
    near = (center_depth - 0.5 * extent).clamp_min(0.05)
    far = (center_depth + 0.5 * extent).clamp_min(near + 1e-3)
    values = _valid_values(depth, mask)
    lo, hi = torch.quantile(values, values.new_tensor([0.05, 0.95]))
    if hi - lo < 1e-8:
        return torch.full_like(depth, center_depth)
    return (near + (depth - lo) * (far - near) / (hi - lo)).clamp(near, far)


def _valid_values(depth: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    valid = torch.isfinite(depth)
    if mask is not None:
        valid &= mask.to(depth) > 0.5
    values = depth[valid]
    if values.numel() < 2:
        raise ValueError("not enough valid foreground pixels to align depth to scene bounds")
    return values


def _bilinear_sample(grid: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample an image with the repository's half-integer pixel-center convention."""
    h, w = grid.shape
    x = (uv[:, 0] - 0.5).clamp(0, w - 1)
    y = (uv[:, 1] - 0.5).clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(w - 1), (y0 + 1).clamp_max(h - 1)
    fx, fy = x - x0, y - y0
    return (
        grid[y0, x0] * (1 - fx) * (1 - fy)
        + grid[y0, x1] * fx * (1 - fy)
        + grid[y1, x0] * (1 - fx) * fy
        + grid[y1, x1] * fx * fy
    )
