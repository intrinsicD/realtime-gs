"""Frozen CPU reference semantics for the kernel-support taper experiment."""

import math

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    _summarize_kernel_support_diagnostics,
)
from rtgs.render.base import KernelSupportDiagnostics, get_rasterizer
from rtgs.render.gsplat_backend import GsplatRasterizer
from rtgs.render.torch_ref import TorchRasterizer, kernel_support_weight


def test_hard_kernel_is_bit_exact_legacy_expression():
    q = torch.linspace(0.0, 50.0, 1001, dtype=torch.float64)
    legacy = torch.exp(-0.5 * q.clamp_max(48.0)) * (q < 12.0)
    assert torch.equal(kernel_support_weight(q), legacy)


def test_c1_taper_bounds_support_and_interior():
    q = torch.tensor([0.0, 4.0, 11.999, 12.0, 13.0, 15.0, 16.0, 50.0], dtype=torch.float64)
    hard = kernel_support_weight(q, "hard")
    smooth = kernel_support_weight(q, "c1_taper")
    assert torch.equal(smooth[q < 12.0], hard[q < 12.0])
    assert torch.equal(smooth[q >= 16.0], torch.zeros_like(smooth[q >= 16.0]))
    assert bool((smooth[(q >= 12.0) & (q < 16.0)] > 0.0).all())
    assert float((smooth - hard).abs().max()) <= math.exp(-6.0) + 1e-12

    boundaries = torch.tensor([12.0, 16.0], dtype=torch.float64, requires_grad=True)
    kernel_support_weight(boundaries, "c1_taper").sum().backward()
    assert torch.allclose(
        boundaries.grad,
        torch.tensor([-0.5 * math.exp(-6.0), 0.0], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-12,
    )

    def derivative(values):
        sample = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        kernel_support_weight(sample, "c1_taper").sum().backward()
        return sample.grad

    epsilon = 1e-6
    around_cutoff = derivative([12.0 - epsilon, 12.0 + epsilon])
    around_end = derivative([16.0 - epsilon, 16.0 + epsilon])
    assert torch.allclose(around_cutoff[0], around_cutoff[1], atol=1e-8, rtol=1e-5)
    assert torch.allclose(around_end, torch.zeros_like(around_end), atol=1e-9, rtol=0.0)


def test_hard_forward_control_has_hard_values_and_taper_gradients():
    values = torch.tensor([2.0, 11.0, 12.0, 13.0, 15.5, 16.0, 20.0], dtype=torch.float64)

    def value_and_gradient(mode):
        q = values.clone().requires_grad_(True)
        value = kernel_support_weight(q, mode)
        value.sum().backward()
        return value.detach(), q.grad

    hard, hard_grad = value_and_gradient("hard")
    smooth, smooth_grad = value_and_gradient("c1_taper")
    control, control_grad = value_and_gradient("hard_forward_c1_taper_gradient")
    assert torch.equal(control, hard)
    assert torch.equal(control_grad, smooth_grad)
    assert torch.equal(smooth_grad[values < 12.0], hard_grad[values < 12.0])
    assert bool((smooth_grad[(values >= 12.0) & (values < 16.0)] != 0.0).all())
    assert torch.equal(hard_grad[values >= 12.0], torch.zeros_like(hard_grad[values >= 12.0]))


def test_zero_kernel_alpha_clamp_propagates_pinned_upstream_gradient():
    kernel = torch.tensor(0.0, requires_grad=True)
    opacity = torch.tensor(0.1)
    (opacity * kernel).clamp(0.0, 0.999).backward()
    assert kernel.grad == opacity


def test_kernel_diagnostic_summary_is_additive_and_releases_chunks():
    q_leaf = torch.tensor([1.0, 9.0, 12.0, 13.0, 15.0, 16.0], requires_grad=True)
    q = q_leaf * 1.0
    kernel = kernel_support_weight(q, "hard")
    q.retain_grad()
    kernel.retain_grad()
    upstream = torch.tensor([1.0, 2.0, -3.0, 4.0, -5.0, -6.0])
    (kernel * upstream).sum().backward()
    diagnostics = KernelSupportDiagnostics([q], [kernel], torch.tensor([0]))
    summary = _summarize_kernel_support_diagnostics(
        diagnostics, iteration=1, view=2, active_sh_degree=0
    )
    assert summary["eligible_count"] == 5
    assert summary["interior_count"] == 2
    assert summary["boundary_count"] == 1
    assert summary["annulus_count"] == 3
    assert summary["recoverable_annulus_upstream_l1"] == 8.0
    assert summary["recoverable_annulus_candidate_qgrad_l1"] > 0.0
    assert summary["hard_outside_kernel_nonzero_count"] == 0
    assert summary["hard_outside_qgrad_nonzero_count"] == 0
    assert summary["hard_active_qgrad_violation_count"] == 0
    assert sum(row["count"] for row in summary["q_bins"]) == 5
    assert not diagnostics.q_chunks and not diagnostics.kernel_chunks


def test_trainer_collects_hard_kernel_diagnostics():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=3, image_size=12, seed=31)
    config = TrainConfig(
        iterations=2,
        rasterizer="torch",
        device="cpu",
        densify=False,
        target_sh_degree=0,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        collect_kernel_support_diagnostics=True,
        eval_every=2,
        seed=4,
    )
    _, history = Trainer(config).train(scene, scene.gt_gaussians.detach())
    records = history["kernel_support_diagnostics"]
    assert len(records) == 2
    assert all(record["observation_count"] > 0 for record in records)
    assert all(record["hard_outside_kernel_nonzero_count"] == 0 for record in records)
    assert all(record["hard_outside_qgrad_nonzero_count"] == 0 for record in records)
    assert all(record["hard_active_qgrad_violation_count"] == 0 for record in records)


def test_renderer_default_explicit_hard_and_diagnostics_forward_are_bit_exact():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=2, image_size=12, seed=32)
    gaussians = scene.gt_gaussians.detach()
    gaussians.means.requires_grad_(True)
    camera = scene.cameras[0]
    default = TorchRasterizer().render(gaussians, camera)
    explicit = TorchRasterizer(kernel_support_mode="hard").render(gaussians, camera)
    diagnostic = TorchRasterizer(collect_kernel_support_diagnostics=True).render(gaussians, camera)
    for field in ("color", "alpha", "depth"):
        assert torch.equal(getattr(default, field), getattr(explicit, field))
        assert torch.equal(getattr(default, field), getattr(diagnostic, field))
    assert diagnostic.kernel_support_diagnostics is not None


def test_nonhard_modes_and_diagnostics_are_torch_reference_only():
    with pytest.raises(NotImplementedError, match="torch reference"):
        GsplatRasterizer(kernel_support_mode="c1_taper")
    with pytest.raises(NotImplementedError, match="torch reference"):
        GsplatRasterizer(collect_kernel_support_diagnostics=True)
    renderer = get_rasterizer("torch", kernel_support_mode="c1_taper")
    assert isinstance(renderer, TorchRasterizer)
    with pytest.raises(ValueError, match="require the hard"):
        TorchRasterizer(kernel_support_mode="c1_taper", collect_kernel_support_diagnostics=True)
