"""CPU-safe guards and CUDA parity for the packed gsplat point rasterizer."""

from __future__ import annotations

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.gsplat_points import GsplatPointRasterizer
from rtgs.render.torch_points import TorchPointRasterizer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _scene(device: str = "cpu") -> tuple[Gaussians3D, Camera, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7721)
    count = 96
    means = torch.randn(count, 3, generator=generator) * 0.32
    means[:, 2] += 3.0
    quats = torch.randn(count, 4, generator=generator)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.rand(count, 3, generator=generator) * 0.05 + 0.025
    opacity = torch.rand(count, generator=generator) * 0.55 + 0.15
    sh = torch.zeros(count, 1, 3)
    sh[:, 0] = torch.rand(count, 3, generator=generator) - 0.5
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 3.0]),
        width=96,
        height=80,
    )
    xy = torch.rand(24, 2, generator=generator)
    xy[:, 0] = xy[:, 0] * 95.0 + 0.5
    xy[:, 1] = xy[:, 1] * 79.0 + 0.5
    return (
        Gaussians3D(means, quats, scales.log(), opacity, sh).to(device),
        camera.to(device),
        xy.to(device),
    )


def test_module_and_constructor_are_cpu_safe() -> None:
    renderer = GsplatPointRasterizer()
    assert renderer.absgrad is False
    assert renderer.antialiased is False


def test_rejects_non_hard_support() -> None:
    with pytest.raises(NotImplementedError, match="non-hard"):
        GsplatPointRasterizer(kernel_support_mode="c1_taper")


@pytest.mark.cuda
@requires_cuda
def test_cuda_point_render_matches_reference_and_backpropagates() -> None:
    source, camera, xy = _scene("cuda")
    reference = TorchPointRasterizer(point_chunk=24, gaussian_chunk=128).render_points(
        source,
        camera,
        xy,
        sh_degree=0,
    )
    trainable = source.detach()
    trainable.means.requires_grad_(True)
    trainable.quats.requires_grad_(True)
    trainable.log_scales.requires_grad_(True)
    trainable.opacity.requires_grad_(True)
    trainable.sh.requires_grad_(True)
    accelerated = GsplatPointRasterizer().render_points(
        trainable,
        camera,
        xy,
        sh_degree=0,
    )

    assert torch.equal(accelerated.visible, reference.visible)
    assert accelerated.density_gaussian_ids is not None
    assert accelerated.means2d is not None
    assert accelerated.means2d.shape[0] == accelerated.density_gaussian_ids.shape[0]
    assert torch.allclose(accelerated.color, reference.color, atol=2e-2, rtol=2e-2)
    assert torch.allclose(accelerated.alpha, reference.alpha, atol=2e-2, rtol=2e-2)

    accelerated.color.square().mean().backward()
    for tensor in (
        trainable.means,
        trainable.quats,
        trainable.log_scales,
        trainable.opacity,
        trainable.sh,
    ):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())
    assert accelerated.means2d.grad is not None
    assert bool(torch.isfinite(accelerated.means2d.grad).all())
