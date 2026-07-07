"""Differentiable 2D gaussian splatting with accumulated (order-free) blending.

Follows GaussianImage (ECCV 2024): pixel color = sum_i weight_i * color_i * exp(-0.5 d^T
Sigma_i^-1 d). No depth sorting, no transmittance — order-independent and fast. Pure
PyTorch, chunked over pixel rows to bound memory; a CUDA kernel can replace this behind
the same signature later.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D

# Mahalanobis cutoff: exp(-0.5 * 12) ~ 2.5e-3, contributions beyond are zeroed (sparsity
# + slightly better-conditioned gradients).
_CUTOFF = 12.0


def render_gaussians_2d(
    g: Gaussians2D,
    height: int,
    width: int,
    background: torch.Tensor | None = None,
    row_chunk: int = 64,
) -> torch.Tensor:
    """Render 2D gaussians to an (H, W, 3) image, differentiably.

    Args:
        g: the gaussian set (pixel coordinates).
        height, width: output size.
        background: optional (3,) background color added everywhere (default zeros).
        row_chunk: pixel rows processed per chunk (memory/speed tradeoff).
    """
    device = g.xy.device
    inv_cov = g.inverse_covariance()  # (N,2,2)
    amplitude = g.weight[:, None] * g.color  # (N,3)

    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    rows_out = []
    for r0 in range(0, height, row_chunk):
        r1 = min(r0 + row_chunk, height)
        ys = torch.arange(r0, r1, device=device, dtype=torch.float32) + 0.5
        grid = torch.stack(
            [xs[None, :].expand(r1 - r0, width), ys[:, None].expand(r1 - r0, width)], dim=-1
        ).reshape(-1, 2)  # (P,2)
        d = grid[:, None, :] - g.xy[None, :, :]  # (P,N,2)
        # Mahalanobis distance via the symmetric inverse covariance.
        q = (
            d[..., 0] ** 2 * inv_cov[None, :, 0, 0]
            + 2.0 * d[..., 0] * d[..., 1] * inv_cov[None, :, 0, 1]
            + d[..., 1] ** 2 * inv_cov[None, :, 1, 1]
        )  # (P,N)
        w = torch.exp(-0.5 * q.clamp_max(_CUTOFF * 4)) * (q < _CUTOFF)
        rows_out.append(w @ amplitude)  # (P,3)
    img = torch.cat(rows_out, dim=0).reshape(height, width, 3)
    if background is not None:
        img = img + background[None, None, :]
    return img
