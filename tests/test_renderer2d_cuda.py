"""Stage-1 CUDA renderer extension: CPU-safe guards plus GPU parity (self-skipping).

The parity tests mirror StructSplat's exact-renderer methodology: the pure-torch renderer is
the oracle, tolerances absorb atomic summation order (forward ~2e-5, gradients ~3e-4).
"""

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.image2gs.renderer2d import render_gaussians_2d, render_gaussians_2d_batched

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def _cpu_gaussians(n: int = 4) -> Gaussians2D:
    gen = torch.Generator().manual_seed(0)
    return Gaussians2D(
        xy=torch.rand(n, 2, generator=gen) * 8.0,
        chol=torch.tensor([[1.5, 0.2, 1.0]]).repeat(n, 1),
        color=torch.rand(n, 3, generator=gen),
        weight=torch.rand(n, generator=gen),
    )


def test_cuda_backend_imports_without_cuda():
    import rtgs.image2gs.cuda_backend as cuda_backend

    assert hasattr(cuda_backend, "render_batched_cuda")


def test_cuda_renderer_requires_cuda_tensors():
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        render_gaussians_2d(_cpu_gaussians(), 8, 8, renderer="cuda")


def test_unknown_renderer_rejected():
    with pytest.raises(ValueError, match="renderer"):
        render_gaussians_2d(_cpu_gaussians(), 8, 8, renderer="bogus")


def test_auto_renderer_falls_back_to_torch_on_cpu():
    g = _cpu_gaussians()
    auto = render_gaussians_2d(g, 8, 8, renderer="auto")
    torch_ref = render_gaussians_2d(g, 8, 8)
    assert torch.equal(auto, torch_ref)


def _random_stacked_cuda(n_views: int, n: int, height: int, width: int):
    gen = torch.Generator().manual_seed(7)
    span = torch.tensor([width + 8.0, height + 8.0])
    xy = (torch.rand(n_views, n, 2, generator=gen) * span - 4.0).cuda()
    chol = torch.zeros(n_views, n, 3)
    chol[..., 0] = 0.5 + 3.0 * torch.rand(n_views, n, generator=gen)
    chol[..., 2] = 0.5 + 3.0 * torch.rand(n_views, n, generator=gen)
    chol[..., 1] = 0.5 * torch.randn(n_views, n, generator=gen)
    return (
        xy,
        chol.cuda(),
        torch.rand(n_views, n, 3, generator=gen).cuda(),
        torch.rand(n_views, n, generator=gen).cuda(),
    )


@pytest.mark.cuda
@requires_cuda
def test_cuda_forward_matches_torch_reference():
    xy, chol, color, weight = _random_stacked_cuda(2, 64, 24, 20)
    reference = render_gaussians_2d_batched(xy, chol, color, weight, 24, 20, renderer="torch")
    fast = render_gaussians_2d_batched(xy, chol, color, weight, 24, 20, renderer="cuda")
    assert torch.allclose(fast, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
@requires_cuda
def test_cuda_backward_matches_torch_reference():
    tensors = _random_stacked_cuda(2, 48, 20, 16)
    grads = {}
    for renderer in ("torch", "cuda"):
        leaves = [t.clone().requires_grad_(True) for t in tensors]
        image = render_gaussians_2d_batched(*leaves, 20, 16, renderer=renderer)
        (
            image * torch.linspace(0.5, 1.5, image.numel(), device="cuda").view_as(image)
        ).sum().backward()
        grads[renderer] = [leaf.grad for leaf in leaves]
    for name, ref, fast in zip(("xy", "chol", "color", "weight"), grads["torch"], grads["cuda"]):
        assert torch.allclose(fast, ref, atol=3e-4, rtol=3e-4), f"{name} gradient diverged"


@pytest.mark.cuda
@requires_cuda
def test_cuda_empty_field_renders_zeros():
    empty = render_gaussians_2d_batched(
        torch.zeros(1, 0, 2).cuda(),
        torch.zeros(1, 0, 3).cuda(),
        torch.zeros(1, 0, 3).cuda(),
        torch.zeros(1, 0).cuda(),
        8,
        8,
        renderer="cuda",
    )
    assert torch.equal(empty, torch.zeros(1, 8, 8, 3).cuda())


@pytest.mark.cuda
@requires_cuda
def test_cuda_nonfinite_mean_is_dropped_without_crash():
    xy, chol, color, weight = _random_stacked_cuda(1, 8, 12, 12)
    xy = xy.clone()
    xy[0, 0] = torch.tensor([float("nan"), 1e12], device="cuda")
    image = render_gaussians_2d_batched(xy, chol, color, weight, 12, 12, renderer="cuda")
    assert torch.isfinite(image).all()
