"""Image quality metrics: PSNR and SSIM, on (H, W, 3) float tensors in [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_DEFAULT_SSIM_TILE_ROWS = 256


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB (max value 1.0)."""
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Foreground-weighted PSNR without rewarding correctly black background pixels."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred and target must have matching (H, W, 3) shapes")
    if mask.shape != pred.shape[:2]:
        raise ValueError("mask must match the image height and width")
    weights = mask.to(device=pred.device, dtype=pred.dtype).clamp(0, 1)
    if not bool(weights.sum() > 0):
        raise ValueError("mask has no foreground pixels")
    denominator = weights.sum() * pred.shape[-1]
    mse = (((pred - target) ** 2) * weights[..., None]).sum() / denominator
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def masked_crop(
    image: torch.Tensor, mask: torch.Tensor, margin_fraction: float = 0.05
) -> torch.Tensor:
    """Mask an image and crop it to the foreground bounding box plus a small margin."""
    if mask.shape != image.shape[:2]:
        raise ValueError("mask must match the image height and width")
    foreground = mask > 0.5
    if not bool(foreground.any()):
        raise ValueError("mask has no foreground pixels")
    yy, xx = torch.where(foreground)
    height, width = image.shape[:2]
    margin = max(1, round(max(height, width) * margin_fraction))
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(height, int(yy.max()) + 1 + margin)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(width, int(xx.max()) + 1 + margin)
    masked = image * mask.to(image).clamp(0, 1)[..., None]
    return masked[y0:y1, x0:x1]


def image_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """Return explicit full-canvas and foreground-aware image quality metrics."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    if mask is None:
        return {"psnr": psnr(pred, target), "ssim": float(ssim(pred, target))}
    mask = mask.to(pred).clamp(0, 1)
    pred_crop = masked_crop(pred, mask)
    target_crop = masked_crop(target, mask)
    return {
        # Full is retained as a diagnostic, never as the masked-scene headline metric.
        "psnr_full": psnr(pred, target * mask[..., None]),
        "psnr_fg": masked_psnr(pred, target, mask),
        "psnr_crop": psnr(pred_crop, target_crop),
        "ssim_crop": float(ssim(pred_crop, target_crop)),
    }


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _separable_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply an exact separable Gaussian window channel-wise.

    This computes the same outer-product filter as a dense 11x11 convolution with roughly
    one fifth of the multiply-adds, which matters because SSIM runs every training step.
    """
    channels = image.shape[1]
    radius = kernel.numel() // 2
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    filtered = F.conv2d(image, vertical, padding=(radius, 0), groups=channels)
    return F.conv2d(filtered, horizontal, padding=(0, radius), groups=channels)


def _ssim_tile_sum(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    local_y0: int,
    local_y1: int,
) -> torch.Tensor:
    """Return the SSIM-map sum over the non-halo rows of an NCHW image tile."""
    mu_x = _separable_filter(x, kernel)
    mu_y = _separable_filter(y, kernel)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _separable_filter(x * x, kernel) - mu_x2
    sigma_y2 = _separable_filter(y * y, kernel) - mu_y2
    sigma_xy = _separable_filter(x * y, kernel) - mu_xy

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map[:, :, local_y0:local_y1].sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    *,
    tile_rows: int | None = _DEFAULT_SSIM_TILE_ROWS,
) -> torch.Tensor:
    """Structural similarity (mean over image), differentiable.

    Standard single-scale SSIM with an 11x11 gaussian window (sigma 1.5), matching the
    formulation used by 3DGS training losses. The image is evaluated at its exact resolution
    in row tiles by default. Each tile includes the gaussian-window halo, retaining the same
    zero-padding semantics as a full-frame convolution. During autograd, tile computations are
    checkpointed so only one tile's SSIM intermediates need to be resident at a time.

    Pass ``tile_rows=None`` to use one full-height tile.
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("expected matching (H, W, 3) images")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if tile_rows is not None and tile_rows <= 0:
        raise ValueError("tile_rows must be positive or None")

    x = pred.permute(2, 0, 1)[None]  # (1,3,H,W)
    y = target.permute(2, 0, 1)[None]
    kernel = _gaussian_window(window_size, 1.5, pred.device)
    radius = window_size // 2
    height = pred.shape[0]
    rows_per_tile = height if tile_rows is None else tile_rows
    total = pred.new_zeros(())

    for y0 in range(0, height, rows_per_tile):
        y1 = min(height, y0 + rows_per_tile)
        source_y0 = max(0, y0 - radius)
        source_y1 = min(height, y1 + radius)
        local_y0 = y0 - source_y0
        local_y1 = local_y0 + (y1 - y0)
        x_tile = x[:, :, source_y0:source_y1]
        y_tile = y[:, :, source_y0:source_y1]
        if torch.is_grad_enabled() and (x_tile.requires_grad or y_tile.requires_grad):
            tile_sum = checkpoint(
                _ssim_tile_sum,
                x_tile,
                y_tile,
                kernel,
                local_y0,
                local_y1,
                use_reentrant=False,
            )
        else:
            tile_sum = _ssim_tile_sum(x_tile, y_tile, kernel, local_y0, local_y1)
        total = total + tile_sum

    return total / pred.numel()
