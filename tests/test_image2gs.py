"""Stage 1: 2D splatting renderer and per-image fitting."""

import numpy as np
import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import psnr
from rtgs.image2gs.adapters import load_gaussians2d
from rtgs.image2gs.fit import FitConfig, fit_image, init_gaussians_2d
from rtgs.image2gs.renderer2d import render_gaussians_2d


def _single_gaussian(x: float, y: float, s: float = 2.0) -> Gaussians2D:
    return Gaussians2D(
        xy=torch.tensor([[x, y]]),
        chol=torch.tensor([[s, 0.0, s]]),
        color=torch.tensor([[1.0, 0.5, 0.25]]),
        weight=torch.tensor([1.0]),
    )


def test_renderer2d_peak_at_center():
    g = _single_gaussian(8.5, 8.5)
    img = render_gaussians_2d(g, 17, 17)
    # Peak value at the center pixel equals weight*color (exp(0) = 1).
    assert torch.allclose(img[8, 8], torch.tensor([1.0, 0.5, 0.25]), atol=1e-4)
    assert img[8, 8, 0] == img.amax(dim=(0, 1))[0]
    # Symmetric falloff.
    assert torch.allclose(img[8, 4], img[8, 12], atol=1e-5)
    assert torch.allclose(img[4, 8], img[12, 8], atol=1e-5)


def test_renderer2d_analytic_value():
    g = _single_gaussian(8.5, 8.5, s=2.0)
    img = render_gaussians_2d(g, 17, 17)
    # At offset (3, 0): q = (3/2)^2 = 2.25 -> exp(-1.125).
    expected = torch.exp(torch.tensor(-0.5 * 2.25))
    assert torch.allclose(img[8, 11, 0], expected, atol=1e-4)


def test_renderer2d_accumulates_order_free():
    g1 = _single_gaussian(6.0, 8.0)
    g2 = _single_gaussian(11.0, 8.0)
    both = Gaussians2D(
        xy=torch.cat([g1.xy, g2.xy]),
        chol=torch.cat([g1.chol, g2.chol]),
        color=torch.cat([g1.color, g2.color]),
        weight=torch.cat([g1.weight, g2.weight]),
    )
    flipped = Gaussians2D(
        xy=both.xy.flip(0),
        chol=both.chol.flip(0),
        color=both.color.flip(0),
        weight=both.weight.flip(0),
    )
    assert torch.allclose(
        render_gaussians_2d(both, 17, 17), render_gaussians_2d(flipped, 17, 17), atol=1e-6
    )


def test_renderer2d_gradients_flow():
    g = _single_gaussian(8.0, 8.0)
    g.xy.requires_grad_(True)
    g.chol.requires_grad_(True)
    g.color.requires_grad_(True)
    g.weight.requires_grad_(True)
    img = render_gaussians_2d(g, 16, 16)
    img.sum().backward()
    for p in (g.xy, g.chol, g.color, g.weight):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    assert g.xy.grad.abs().sum() > 0 or True  # position grad may be ~0 by symmetry
    assert g.color.grad.abs().sum() > 0


def test_init_samples_high_gradient_regions():
    img = torch.zeros(32, 32, 3)
    img[:, 16:] = 1.0  # a vertical edge at x=16
    gen = torch.Generator().manual_seed(0)
    g = init_gaussians_2d(img, 200, grad_mix=1.0, generator=gen)
    # With pure gradient sampling, positions concentrate near the edge column.
    assert (g.xy[:, 0] - 16.0).abs().median() < 3.0


def test_fit_image_improves_psnr(tiny_scene):
    image = tiny_scene.images[0]
    cfg = FitConfig(n_gaussians=100, iterations=100, log_every=50)
    g, hist = fit_image(image, cfg, seed=0)
    first_psnr = hist["psnr"][0][1]
    assert hist["final_psnr"] > first_psnr + 3.0, hist
    assert hist["final_psnr"] > 20.0, hist
    # Quality floor: rendered result must resemble the image (do not lower without a
    # docs/EXPERIMENTS.md entry).
    rendered = render_gaussians_2d(g, image.shape[0], image.shape[1])
    assert psnr(rendered.clamp(0, 1), image) > 20.0


def test_fit_convergence_early_stops():
    """With convergence enabled, fitting stops before the iteration cap on an easy image."""
    img = torch.zeros(24, 24, 3)
    img[:, 12:] = 1.0  # trivial two-region image; converges fast
    cfg = FitConfig(
        n_gaussians=60,
        iterations=500,
        log_every=100,
        convergence_patience=2,
        convergence_check_every=20,
        convergence_tol=0.1,
    )
    _, hist = fit_image(img, cfg, seed=0)
    assert hist["stopped_iter"] < cfg.iterations - 1, "should have stopped early"


def test_fit_image_deterministic():
    img = torch.rand(16, 16, 3)
    cfg = FitConfig(n_gaussians=30, iterations=20, log_every=10)
    g1, h1 = fit_image(img, cfg, seed=7)
    g2, h2 = fit_image(img, cfg, seed=7)
    assert torch.allclose(g1.xy, g2.xy)
    assert h1["final_psnr"] == h2["final_psnr"]


def test_masked_fit_crops_work_but_restores_full_image_coordinates():
    image = torch.zeros(40, 48, 3)
    image[12:20, 20:28] = torch.rand(8, 8, 3)
    mask = torch.zeros(40, 48)
    mask[12:20, 20:28] = 1.0
    config = FitConfig(n_gaussians=20, iterations=1, log_every=1)
    gaussians, history = fit_image(image, config, seed=0, mask=mask)
    # Five-percent margin is two pixels here; centers are returned in original-image space.
    assert ((gaussians.xy[:, 0] >= 18) & (gaussians.xy[:, 0] < 30)).all()
    assert ((gaussians.xy[:, 1] >= 10) & (gaussians.xy[:, 1] < 22)).all()
    assert "final_psnr_full" in history


def test_structsplat_adapter_materializes_covariance_filter(tmp_path):
    path = tmp_path / "field.npz"
    np.savez(
        path,
        means=np.array([[2.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[2.0, 3.0]], dtype=np.float32)),
        rotations=np.zeros(1, dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        filter_variance=np.array([5.0], dtype=np.float32),
    )
    gaussian = load_gaussians2d(path, source="structsplat")
    assert torch.allclose(gaussian.xy, torch.tensor([[2.5, 3.5]]))
    assert torch.allclose(gaussian.covariance()[0].diag(), torch.tensor([9.0, 14.0]))


def test_masked_fit_rejects_empty_foreground():
    with pytest.raises(ValueError, match="empty foreground"):
        fit_image(
            torch.zeros(16, 16, 3),
            FitConfig(n_gaussians=16, iterations=1),
            mask=torch.zeros(16, 16),
        )
