"""Image quality metrics: PSNR and SSIM, on (H, W, 3) float tensors in [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB (max value 1.0)."""
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity (mean over image), differentiable.

    Standard single-scale SSIM with an 11x11 gaussian window (sigma 1.5), matching the
    formulation used by 3DGS training losses.
    """
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("expected (H, W, 3) images")
    x = pred.permute(2, 0, 1)[None]  # (1,3,H,W)
    y = target.permute(2, 0, 1)[None]
    win = _gaussian_window(window_size, 1.5, pred.device)
    win = win.expand(3, 1, window_size, window_size)
    pad = window_size // 2

    mu_x = F.conv2d(x, win, padding=pad, groups=3)
    mu_y = F.conv2d(y, win, padding=pad, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=3) - mu_xy

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean()
