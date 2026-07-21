"""Batched stage-1 fitting: fused multi-view renderer and fit_views parity."""

from dataclasses import replace

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.image2gs.renderer2d import render_gaussians_2d, render_gaussians_2d_batched


def _random_views(
    n_views: int, n: int, height: int, width: int, gen: torch.Generator
) -> list[Gaussians2D]:
    """Random per-view gaussian sets including off-image and edge-straddling centers."""
    views = []
    span = torch.tensor([width + 8.0, height + 8.0])
    for _ in range(n_views):
        xy = torch.rand(n, 2, generator=gen) * span - 4.0
        chol = torch.zeros(n, 3)
        chol[:, 0] = 0.5 + 3.0 * torch.rand(n, generator=gen)
        chol[:, 2] = 0.5 + 3.0 * torch.rand(n, generator=gen)
        chol[:, 1] = 0.5 * torch.randn(n, generator=gen)
        color = torch.rand(n, 3, generator=gen)
        weight = torch.rand(n, generator=gen)
        views.append(Gaussians2D(xy=xy, chol=chol, color=color, weight=weight))
    return views


def _stack(views: list[Gaussians2D]) -> tuple[torch.Tensor, ...]:
    return (
        torch.stack([v.xy for v in views]),
        torch.stack([v.chol for v in views]),
        torch.stack([v.color for v in views]),
        torch.stack([v.weight for v in views]),
    )


def test_batched_render_matches_per_view():
    gen = torch.Generator().manual_seed(0)
    views = _random_views(3, 40, 24, 20, gen)
    batched = render_gaussians_2d_batched(*_stack(views), 24, 20)
    assert batched.shape == (3, 24, 20, 3)
    for b, view in enumerate(views):
        single = render_gaussians_2d(view, 24, 20)
        assert torch.allclose(batched[b], single, atol=1e-6), f"view {b} diverged"


def test_batched_render_background_broadcasts():
    gen = torch.Generator().manual_seed(1)
    views = _random_views(2, 10, 12, 12, gen)
    background = torch.tensor([0.1, 0.2, 0.3])
    batched = render_gaussians_2d_batched(*_stack(views), 12, 12, background=background)
    single = render_gaussians_2d(views[1], 12, 12, background=background)
    assert torch.allclose(batched[1], single, atol=1e-6)


def test_batched_render_gradients_match_per_view():
    gen = torch.Generator().manual_seed(2)
    views = _random_views(2, 12, 16, 16, gen)
    stacked = [t.clone().requires_grad_(True) for t in _stack(views)]
    render_gaussians_2d_batched(*stacked, 16, 16).square().sum().backward()
    for b, view in enumerate(views):
        leaves = (
            view.xy.clone().requires_grad_(True),
            view.chol.clone().requires_grad_(True),
            view.color.clone().requires_grad_(True),
            view.weight.clone().requires_grad_(True),
        )
        g = Gaussians2D(*leaves)
        render_gaussians_2d(g, 16, 16).square().sum().backward()
        for stacked_leaf, single_leaf in zip(stacked, leaves):
            assert torch.allclose(stacked_leaf.grad[b], single_leaf.grad, atol=1e-5, rtol=1e-4), (
                f"view {b} gradient diverged"
            )


def test_batched_render_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(B, N, 2\)"):
        render_gaussians_2d_batched(
            torch.zeros(3, 2), torch.zeros(3, 3), torch.zeros(3, 3), torch.zeros(3), 8, 8
        )


def test_fit_views_batched_matches_serial(tiny_scene):
    images = [tiny_scene.images[0], tiny_scene.images[1]]
    serial_cfg = FitConfig(n_gaussians=80, iterations=60, log_every=30)
    batched_cfg = replace(serial_cfg, batch_views=True)
    gs_serial, hist_serial = fit_views(images, serial_cfg, seed=0)
    gs_batched, hist_batched = fit_views(images, batched_cfg, seed=0)
    assert len(gs_batched) == len(gs_serial) == 2
    for b in range(2):
        assert gs_batched[b].n == gs_serial[b].n
        assert set(hist_batched[b]) >= {"psnr", "stopped_iter", "final_psnr", "final_psnr_full"}
        # Identical seeds give identical initializations, so the step-0 log must agree.
        it0, psnr0_batched = hist_batched[b]["psnr"][0]
        it0_serial, psnr0_serial = hist_serial[b]["psnr"][0]
        assert it0 == it0_serial == 0
        assert abs(psnr0_batched - psnr0_serial) < 1e-3
        # Trajectories only differ by float summation order; quality floors stay loose
        # (rule 3) but the two paths must land in the same place.
        assert abs(hist_batched[b]["final_psnr"] - hist_serial[b]["final_psnr"]) < 1.0
        assert hist_batched[b]["final_psnr"] > 17.0


def test_fit_views_batched_is_deterministic(tiny_scene):
    images = [tiny_scene.images[0], tiny_scene.images[1]]
    cfg = FitConfig(n_gaussians=40, iterations=15, log_every=10, batch_views=True)
    first, hist_first = fit_views(images, cfg, seed=3)
    second, hist_second = fit_views(images, cfg, seed=3)
    for b in range(2):
        assert torch.allclose(first[b].xy, second[b].xy)
        assert hist_first[b]["final_psnr"] == hist_second[b]["final_psnr"]


def test_fit_views_batched_rejects_unsupported_modes(tiny_scene):
    images = [tiny_scene.images[0], tiny_scene.images[1]]
    base = FitConfig(n_gaussians=16, iterations=1, batch_views=True)
    with pytest.raises(ValueError, match="masks"):
        fit_views(images, base, masks=[torch.ones(32, 32), torch.ones(32, 32)])
    with pytest.raises(ValueError, match="native backend"):
        fit_views(images, replace(base, backend="structsplat"))
    with pytest.raises(ValueError, match="early stopping"):
        fit_views(images, replace(base, convergence_patience=2))
    with pytest.raises(ValueError, match="equally sized"):
        fit_views([images[0], images[1][:16]], base)


def test_fit_views_batched_empty_and_freeze_geometry(tiny_scene):
    assert fit_views([], FitConfig(batch_views=True)) == ([], [])
    cfg = FitConfig(n_gaussians=24, iterations=5, batch_views=True, freeze_geometry=True)
    fitted, _ = fit_views([tiny_scene.images[0]], cfg, seed=0)
    reference = fit_views([tiny_scene.images[0]], replace(cfg, batch_views=False), seed=0)[0]
    # Frozen geometry must stay at the (identically seeded) initialization on both paths.
    assert torch.allclose(fitted[0].xy, reference[0].xy, atol=1e-6)
    assert torch.allclose(fitted[0].chol, reference[0].chol, atol=1e-6)
