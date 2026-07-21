"""Differentiable sparse 2D gaussian splatting with accumulated blending.

Uses GaussianImage's accumulated-summation rendering, with this repository's additional
factorization of each accumulated RGB vector into ``weight_i * color_i``: pixel color is
``sum_i weight_i * color_i * exp(-0.5 d^T Sigma_i^-1 d)``. There is no depth sorting or
transmittance, so the result is order-independent. Each Gaussian is evaluated only inside its
detached cutoff bounding box; this keeps the pure-PyTorch reference practical without changing
the support of the previous dense ``pixels x gaussians`` implementation.

Two entry points share one flat scatter core: :func:`render_gaussians_2d` renders a single
image, :func:`render_gaussians_2d_batched` renders ``B`` same-sized views in one fused pass
(each gaussian scatters into its own view's canvas slice, so per-view results match the
single-image path). Both accept ``renderer=`` to select the pure-torch reference (default,
the correctness anchor) or the experimental CUDA extension in ``rtgs.image2gs.cuda_backend``.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D, chol_covariance, chol_inverse_covariance

# Mahalanobis cutoff: exp(-0.5 * 12) ~ 2.5e-3, contributions beyond are zeroed (sparsity
# + slightly better-conditioned gradients).
_CUTOFF = 12.0
_RENDERERS = ("torch", "cuda", "auto")


def _resolve_renderer(renderer: str, device: torch.device) -> str:
    if renderer not in _RENDERERS:
        raise ValueError(f"renderer must be one of {_RENDERERS}, got {renderer!r}")
    if renderer == "auto":
        return "cuda" if device.type == "cuda" and torch.cuda.is_available() else "torch"
    return renderer


def render_gaussians_2d(
    g: Gaussians2D,
    height: int,
    width: int,
    background: torch.Tensor | None = None,
    row_chunk: int = 64,
    renderer: str = "torch",
) -> torch.Tensor:
    """Render 2D gaussians to an (H, W, 3) image, differentiably.

    Args:
        g: the gaussian set (pixel coordinates).
        height, width: output size.
        background: optional (3,) background color added everywhere (default zeros).
        row_chunk: controls the maximum number of support elements processed per chunk.
        renderer: ``torch`` (reference), ``cuda`` (extension, CUDA tensors only), or ``auto``.
    """
    if _resolve_renderer(renderer, g.xy.device) == "cuda":
        return render_gaussians_2d_batched(
            g.xy[None],
            g.chol[None],
            g.color[None],
            g.weight[None],
            height,
            width,
            background=background,
            row_chunk=row_chunk,
            renderer="cuda",
        )[0]
    img, _ = _accumulate(g, height, width, row_chunk)
    if background is not None:
        img = img + background.to(img)[None, None, :]
    return img


def render_gaussians_2d_batched(
    xy: torch.Tensor,
    chol: torch.Tensor,
    color: torch.Tensor,
    weight: torch.Tensor,
    height: int,
    width: int,
    background: torch.Tensor | None = None,
    row_chunk: int = 64,
    renderer: str = "torch",
) -> torch.Tensor:
    """Render B stacked same-sized views to (B, H, W, 3) in one fused differentiable pass.

    Args:
        xy: (B, N, 2) centers in pixel coordinates of each view.
        chol: (B, N, 3) packed Cholesky factors (l11, l21, l22).
        color: (B, N, 3) RGB in [0, 1].
        weight: (B, N) amplitudes in [0, 1].
        height, width: shared output size of every view.
        background: optional (3,) background color added everywhere (default zeros).
        row_chunk: controls the maximum number of support elements processed per chunk.
        renderer: ``torch`` (reference), ``cuda`` (extension, CUDA tensors only), or ``auto``.
    """
    if xy.dim() != 3 or xy.shape[-1] != 2:
        raise ValueError("batched xy must be (B, N, 2)")
    n_views, n = xy.shape[:2]
    if chol.shape != (n_views, n, 3) or color.shape != (n_views, n, 3):
        raise ValueError("batched chol/color must be (B, N, 3)")
    if weight.shape != (n_views, n):
        raise ValueError("batched weight must be (B, N)")
    if _resolve_renderer(renderer, xy.device) == "cuda":
        from rtgs.image2gs.cuda_backend import render_batched_cuda

        img = render_batched_cuda(xy, chol, color, weight, height, width)
    else:
        inv_cov = chol_inverse_covariance(chol).reshape(n_views * n, 2, 2)
        cov = chol_covariance(chol).detach().reshape(n_views * n, 2, 2)
        view_index = torch.arange(n_views, device=xy.device).repeat_interleave(n)
        img_flat, _ = _accumulate_flat(
            xy.reshape(-1, 2),
            inv_cov,
            cov,
            color.reshape(-1, 3),
            weight.reshape(-1),
            view_index,
            n_views,
            height,
            width,
            row_chunk,
        )
        img = img_flat.reshape(n_views, height, width, 3)
    if background is not None:
        img = img + background.to(img)[None, None, None, :]
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


def _support_rects(
    xy: torch.Tensor, cov: torch.Tensor, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached per-gaussian support rectangles: (x0, x1, y0, y1, nx, counts).

    ``xy`` and ``cov`` must already be detached; bounds are cutoff-clipped like tile culling in
    CUDA splatters. Fully-outside gaussians get ``counts == 0``.
    """
    radius_x = (_CUTOFF * cov[:, 0, 0].clamp_min(0)).sqrt()
    radius_y = (_CUTOFF * cov[:, 1, 1].clamp_min(0)).sqrt()

    # Pixel j has center j+0.5.
    x0 = torch.ceil(xy[:, 0] - radius_x - 0.5).long().clamp(0, width - 1)
    x1 = torch.floor(xy[:, 0] + radius_x - 0.5).long().clamp(0, width - 1)
    y0 = torch.ceil(xy[:, 1] - radius_y - 0.5).long().clamp(0, height - 1)
    y1 = torch.floor(xy[:, 1] + radius_y - 0.5).long().clamp(0, height - 1)
    outside = (
        (xy[:, 0] + radius_x < 0.5)
        | (xy[:, 0] - radius_x > width - 0.5)
        | (xy[:, 1] + radius_y < 0.5)
        | (xy[:, 1] - radius_y > height - 0.5)
    )
    nx = (x1 - x0 + 1).clamp_min(0)
    ny = (y1 - y0 + 1).clamp_min(0)
    counts = torch.where(outside, torch.zeros_like(nx), nx * ny)
    return x0, x1, y0, y1, nx, counts


def _accumulate(
    g: Gaussians2D,
    height: int,
    width: int,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    img, density = _accumulate_flat(
        g.xy,
        g.inverse_covariance(),
        g.covariance().detach(),
        g.color,
        g.weight,
        None,
        1,
        height,
        width,
        row_chunk,
    )
    return img.reshape(height, width, 3), density.reshape(height, width)


def _accumulate_flat(
    xy: torch.Tensor,
    inv_cov: torch.Tensor,
    cov: torch.Tensor,
    color: torch.Tensor,
    weight: torch.Tensor,
    view_index: torch.Tensor | None,
    n_views: int,
    height: int,
    width: int,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter M gaussians into ``n_views`` stacked (H, W) canvases; returns flat sums.

    ``view_index`` maps each gaussian to its canvas (``None`` means a single canvas). Returns
    ``(n_views*H*W, 3)`` color sums and ``(n_views*H*W,)`` densities.
    """
    device, dtype = xy.device, xy.dtype
    x0, _, y0, _, nx, counts = _support_rects(xy.detach(), cov, height, width)

    color_sum = torch.zeros(n_views * height * width, 3, device=device, dtype=dtype)
    density = torch.zeros(n_views * height * width, device=device, dtype=dtype)
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
        dx = px.to(dtype) + 0.5 - xy[gids, 0]
        dy = py.to(dtype) + 0.5 - xy[gids, 1]
        q = (
            dx.square() * inv_cov[gids, 0, 0]
            + 2.0 * dx * dy * inv_cov[gids, 0, 1]
            + dy.square() * inv_cov[gids, 1, 1]
        )
        support = q < _CUTOFF
        w = torch.exp(-0.5 * q.clamp_max(_CUTOFF * 4)) * support * weight[gids]
        flat = py * width + px
        if view_index is not None:
            flat = flat + view_index[gids] * (height * width)
        color_sum = color_sum.index_add(0, flat, w[:, None] * color[gids])
        density = density.index_add(0, flat, w)
    return color_sum, density
