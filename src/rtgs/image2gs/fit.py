"""Fit a set of 2D gaussians to an image by gradient descent (stage 1).

Recipe distilled from GaussianImage (ECCV 2024) and Image-GS (SIGGRAPH 2025), see
docs/RESEARCH.md §2: Cholesky covariance parametrization, accumulated-summation blending,
plain L2 loss (best PSNR per their ablations), positions initialized by sampling the image
gradient magnitude mixed with a uniform floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import psnr
from rtgs.image2gs.renderer2d import render_gaussians_2d

# Minimum Cholesky diagonal in pixels; keeps covariances well-conditioned.
_MIN_DIAG = 0.3


@dataclass
class FitConfig:
    """Hyperparameters for per-image 2D gaussian fitting."""

    n_gaussians: int = 512
    iterations: int = 300
    lr: float = 1e-2
    # Fraction of position-sampling probability taken from the gradient magnitude; the
    # rest is uniform (Image-GS uses a 0.3 uniform floor).
    grad_init_mix: float = 0.7
    row_chunk: int = 64
    log_every: int = 50
    # Convergence-based early stopping (step 1: "fit until convergence"). PSNR is checked
    # every ``convergence_check_every`` iterations; if the best PSNR has not improved by
    # ``convergence_tol`` dB for ``convergence_patience`` consecutive checks, fitting stops.
    # patience=0 disables early stopping (fixed ``iterations``).
    convergence_patience: int = 0
    convergence_tol: float = 0.05
    convergence_check_every: int = 25


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """(H,W) mean-channel finite-difference gradient magnitude."""
    gray = image.mean(dim=-1)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return torch.sqrt(gx**2 + gy**2)


def init_gaussians_2d(
    image: torch.Tensor,
    n: int,
    grad_mix: float = 0.7,
    generator: torch.Generator | None = None,
) -> Gaussians2D:
    """Initialize N gaussians: positions ~ gradient magnitude + uniform, colors sampled
    from the image, isotropic scale from the average area budget per gaussian."""
    h, w = image.shape[:2]
    device = image.device
    grad = _gradient_magnitude(image).reshape(-1)
    uniform = torch.ones_like(grad) / grad.numel()
    prob = grad_mix * grad / grad.sum().clamp_min(1e-8) + (1 - grad_mix) * uniform
    idx = torch.multinomial(prob, n, replacement=True, generator=generator)
    yy = (idx // w).float() + 0.5
    xx = (idx % w).float() + 0.5
    xy = torch.stack([xx, yy], dim=-1)
    # Jitter within the pixel to break ties from replacement sampling.
    xy = xy + (torch.rand(n, 2, generator=generator, device=device) - 0.5)
    xy[:, 0] = xy[:, 0].clamp(0.0, w - 1e-3)
    xy[:, 1] = xy[:, 1].clamp(0.0, h - 1e-3)

    scale0 = max((h * w / n) ** 0.5 * 0.6, _MIN_DIAG + 0.1)
    chol = torch.zeros(n, 3, device=device, dtype=image.dtype)
    chol[:, 0] = scale0
    chol[:, 2] = scale0

    color = image.reshape(-1, 3)[idx].clone()
    weight = torch.full((n,), 0.5, device=device, dtype=image.dtype)
    return Gaussians2D(xy=xy, chol=chol, color=color, weight=weight)


def fit_image(
    image: torch.Tensor,
    config: FitConfig | None = None,
    seed: int | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[Gaussians2D, dict]:
    """Fit 2D gaussians to an (H, W, 3) image in [0,1].

    Returns the fitted (detached) gaussians and a history dict with 'psnr' (list of
    (iteration, value)) and 'final_psnr'.
    """
    config = config or FitConfig()
    h, w = image.shape[:2]
    target = image
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size does not match image")
        target = image * mask.to(image).clamp(0, 1)[..., None]
    gen = None
    if seed is not None:
        gen = torch.Generator(device=image.device).manual_seed(seed)

    g0 = init_gaussians_2d(target, config.n_gaussians, config.grad_init_mix, gen)

    # Raw (unconstrained) parameters.
    wh = image.new_tensor([w, h])
    xy_raw = torch.logit((g0.xy / wh).clamp(1e-4, 1 - 1e-4))
    diag_raw = _softplus_inv(g0.chol[:, [0, 2]] - _MIN_DIAG)
    off_raw = g0.chol[:, 1].clone()
    color_raw = torch.logit(g0.color.clamp(1e-3, 1 - 1e-3))
    weight_raw = torch.logit(g0.weight.clamp(1e-3, 1 - 1e-3))
    params = [xy_raw, diag_raw, off_raw, color_raw, weight_raw]
    for p in params:
        p.requires_grad_(True)

    def build() -> Gaussians2D:
        diag = torch.nn.functional.softplus(diag_raw) + _MIN_DIAG
        chol = torch.stack([diag[:, 0], off_raw, diag[:, 1]], dim=-1)
        return Gaussians2D(
            xy=torch.sigmoid(xy_raw) * wh,
            chol=chol,
            color=torch.sigmoid(color_raw),
            weight=torch.sigmoid(weight_raw),
        )

    opt = torch.optim.Adam(params, lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.iterations, eta_min=config.lr * 0.1
    )
    history: dict = {"psnr": [], "stopped_iter": config.iterations - 1}
    best_psnr = -float("inf")
    stale_checks = 0
    for it in range(config.iterations):
        opt.zero_grad()
        rendered = render_gaussians_2d(build(), h, w, row_chunk=config.row_chunk)
        loss = torch.nn.functional.mse_loss(rendered, target)
        loss.backward()
        opt.step()
        sched.step()
        if it % config.log_every == 0 or it == config.iterations - 1:
            with torch.no_grad():
                history["psnr"].append((it, psnr(rendered.clamp(0, 1), target)))
        # Convergence check: stop once quality plateaus (step 1 "until convergence").
        if config.convergence_patience and (it + 1) % config.convergence_check_every == 0:
            with torch.no_grad():
                cur = psnr(rendered.clamp(0, 1), target)
            if cur > best_psnr + config.convergence_tol:
                best_psnr = cur
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.convergence_patience:
                    history["stopped_iter"] = it
                    break

    result = build().detach()
    with torch.no_grad():
        final = render_gaussians_2d(result, h, w, row_chunk=config.row_chunk)
        history["final_psnr"] = psnr(final.clamp(0, 1), target)
    return result, history


def fit_views(
    images: list[torch.Tensor],
    config: FitConfig | None = None,
    seed: int = 0,
    masks: list[torch.Tensor] | None = None,
) -> tuple[list[Gaussians2D], list[dict]]:
    """Fit every view of a scene independently (embarrassingly parallel across images)."""
    results, histories = [], []
    for i, image in enumerate(images):
        mask = None if masks is None else masks[i]
        g, hist = fit_image(image, config, seed=seed + i, mask=mask)
        results.append(g)
        histories.append(hist)
    return results, histories
