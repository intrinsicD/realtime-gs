"""Differentiable sparse 2D gaussian splatting with accumulated blending.

Follows GaussianImage (ECCV 2024): pixel color = sum_i weight_i * color_i * exp(-0.5 d^T
Sigma_i^-1 d). No depth sorting, no transmittance — order-independent and fast. Pure
PyTorch. Each Gaussian is evaluated only inside its detached cutoff bounding box; this keeps
the reference implementation practical for high-resolution images without changing the
mathematical support used by the previous dense ``pixels x gaussians`` implementation.
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
        row_chunk: controls the maximum number of support elements processed per chunk.
    """
    img, _ = _accumulate(g, height, width, row_chunk)
    if background is not None:
        img = img + background.to(img)[None, None, :]
    return img


def render_gaussian_coverage_2d(
    g: Gaussians2D,
    height: int,
    width: int,
    row_chunk: int = 64,
) -> torch.Tensor:
    """Return color-independent soft coverage in ``[0,1]``.

    Coverage is ``1-exp(-sum_i weight_i G_i)``. Unlike RGB luminance it remains valid for black
    objects and is therefore suitable for masks, carving and diagnostics.
    """
    _, density = _accumulate(g, height, width, row_chunk)
    return 1.0 - torch.exp(-density)


def _support_slices(counts: torch.Tensor, budget: int):
    """Yield Gaussian ranges with approximately ``budget`` support elements."""
    csum = counts.cumsum(0)
    start = 0
    base = 0
    while start < counts.numel():
        end = int(torch.searchsorted(csum, base + budget, right=True))
        end = max(start + 1, min(end, counts.numel()))
        yield start, end
        start = end
        base = int(csum[start - 1])


def _accumulate(
    g: Gaussians2D,
    height: int,
    width: int,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = g.xy.device, g.xy.dtype
    inv_cov = g.inverse_covariance()
    cov = g.covariance().detach()
    radius_x = (_CUTOFF * cov[:, 0, 0].clamp_min(0)).sqrt()
    radius_y = (_CUTOFF * cov[:, 1, 1].clamp_min(0)).sqrt()

    # Pixel j has center j+0.5. Bounds are detached just like tile culling in CUDA splatters.
    x0 = torch.ceil(g.xy[:, 0].detach() - radius_x - 0.5).long().clamp(0, width - 1)
    x1 = torch.floor(g.xy[:, 0].detach() + radius_x - 0.5).long().clamp(0, width - 1)
    y0 = torch.ceil(g.xy[:, 1].detach() - radius_y - 0.5).long().clamp(0, height - 1)
    y1 = torch.floor(g.xy[:, 1].detach() + radius_y - 0.5).long().clamp(0, height - 1)
    outside = (
        (g.xy[:, 0].detach() + radius_x < 0.5)
        | (g.xy[:, 0].detach() - radius_x > width - 0.5)
        | (g.xy[:, 1].detach() + radius_y < 0.5)
        | (g.xy[:, 1].detach() - radius_y > height - 0.5)
    )
    nx = (x1 - x0 + 1).clamp_min(0)
    ny = (y1 - y0 + 1).clamp_min(0)
    counts = torch.where(outside, torch.zeros_like(nx), nx * ny)

    color_sum = torch.zeros(height * width, 3, device=device, dtype=dtype)
    density = torch.zeros(height * width, device=device, dtype=dtype)
    budget = max(int(row_chunk) * width, 65_536)
    for start, end in _support_slices(counts, budget):
        local_counts = counts[start:end]
        total = int(local_counts.sum())
        if total == 0:
            continue
        local_ids = torch.repeat_interleave(torch.arange(end - start, device=device), local_counts)
        gids = local_ids + start
        ends = local_counts.cumsum(0)
        starts = ends - local_counts
        offset = torch.arange(total, device=device) - starts[local_ids]
        px = x0[gids] + offset % nx[gids]
        py = y0[gids] + offset // nx[gids]
        dx = px.to(dtype) + 0.5 - g.xy[gids, 0]
        dy = py.to(dtype) + 0.5 - g.xy[gids, 1]
        q = (
            dx.square() * inv_cov[gids, 0, 0]
            + 2.0 * dx * dy * inv_cov[gids, 0, 1]
            + dy.square() * inv_cov[gids, 1, 1]
        )
        support = q < _CUTOFF
        w = torch.exp(-0.5 * q.clamp_max(_CUTOFF * 4)) * support * g.weight[gids]
        flat = py * width + px
        color_sum = color_sum.index_add(0, flat, w[:, None] * g.color[gids])
        density = density.index_add(0, flat, w)
    return color_sum.reshape(height, width, 3), density.reshape(height, width)
