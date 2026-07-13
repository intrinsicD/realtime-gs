"""Reference rasterizer semantics + gsplat parity (parity requires CUDA)."""

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.base import get_rasterizer
from rtgs.render.gsplat_backend import _visible_indices
from rtgs.render.torch_ref import TorchRasterizer


def _one_gaussian(color=(1.0, 0.0, 0.0), opacity=0.9, scale=0.1, mean=(0.0, 0.0, 0.0)):
    return Gaussians3D.from_means_covs(
        means=torch.tensor([mean]),
        covs=(torch.eye(3) * scale**2)[None],
        colors=torch.tensor([color]),
        opacity=torch.tensor([opacity]),
    )


def _front_camera(size=33, dist=2.0):
    return Camera.look_at(torch.tensor([0.0, 0.0, -dist]), torch.zeros(3), width=size, height=size)


def test_center_gaussian_hits_center_pixel():
    cam = _front_camera()
    out = TorchRasterizer().render(_one_gaussian(), cam)
    c = 33 // 2
    assert out.color[c, c, 0] == out.color[..., 0].max()
    # alpha at center ~ opacity (dilation shrinks the peak slightly)
    assert 0.6 < out.alpha[c, c] <= 0.91
    # red channel only
    assert out.color[..., 1].max() < 1e-4


def test_depth_ordering_front_occludes_back():
    front = _one_gaussian(color=(1.0, 0.0, 0.0), mean=(0.0, 0.0, -0.5), opacity=0.95)
    back = _one_gaussian(color=(0.0, 1.0, 0.0), mean=(0.0, 0.0, 0.5), opacity=0.95)
    both = Gaussians3D.cat([back, front])  # order in container must not matter
    cam = _front_camera()
    out = TorchRasterizer().render(both, cam)
    c = 33 // 2
    assert out.color[c, c, 0] > out.color[c, c, 1], "front (red) must dominate"
    # Expected depth at the center is near the front gaussian.
    z_expected = (out.depth[c, c] / out.alpha[c, c].clamp_min(1e-6)).item()
    z_front = 2.0 - 0.5
    assert abs(z_expected - z_front) < 0.3


def test_alpha_composition_bounded():
    g = Gaussians3D.cat([_one_gaussian(opacity=0.9) for _ in range(5)])
    out = TorchRasterizer().render(g, _front_camera())
    assert (out.alpha <= 1.0 + 1e-5).all()
    assert (out.alpha >= 0.0).all()


def test_background_color():
    cam = _front_camera()
    bg = torch.tensor([0.2, 0.4, 0.6])
    out = TorchRasterizer().render(_one_gaussian(opacity=0.9), cam, background=bg)
    # Far corner is pure background.
    assert torch.allclose(out.color[0, 0], bg, atol=1e-3)


def test_gradients_flow_through_render():
    g = _one_gaussian()
    g.means.requires_grad_(True)
    g.log_scales.requires_grad_(True)
    g.opacity.requires_grad_(True)
    g.sh.requires_grad_(True)
    cam = _front_camera(size=17)
    out = TorchRasterizer().render(g, cam)
    target = torch.zeros(17, 17, 3)
    loss = (out.color - target).abs().mean()
    loss.backward()
    for p in (g.means, g.log_scales, g.opacity, g.sh):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # Screen-space gradients for densification are exposed.
    assert out.means2d is not None and out.means2d.grad is not None


def test_offcenter_gaussian_projects_correctly():
    cam = _front_camera(size=33)
    g = _one_gaussian(mean=(0.3, 0.2, 0.0), scale=0.05)
    out = TorchRasterizer().render(g, cam)
    uv, _ = cam.project(g.means)
    px = int(uv[0, 0]), int(uv[0, 1])
    peak = (out.color[..., 0] == out.color[..., 0].max()).nonzero()[0]
    assert abs(int(peak[1]) - px[0]) <= 1 and abs(int(peak[0]) - px[1]) <= 1


def test_empty_and_behind_camera():
    cam = _front_camera()
    g = _one_gaussian(mean=(0.0, 0.0, -5.0))  # behind the camera
    out = TorchRasterizer().render(g, cam)
    assert out.alpha.max() == 0.0
    assert out.color.abs().max() == 0.0


def test_registry():
    assert isinstance(get_rasterizer("torch"), TorchRasterizer)
    with pytest.raises(ValueError):
        get_rasterizer("nonsense")


def test_gsplat_radius_layouts_map_visibility():
    scalar = torch.tensor([[1.0, 0.0, 2.0]])
    axes = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [2.0, 3.0]]])
    expected = torch.tensor([0, 2])
    assert torch.equal(_visible_indices(scalar), expected)
    assert torch.equal(_visible_indices(axes), expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gsplat_parity(tiny_scene):
    """gsplat must match the reference renderer on a real scene render."""
    pytest.importorskip("gsplat")
    gs = tiny_scene.gt_gaussians
    cam = tiny_scene.cameras[0]
    ref = TorchRasterizer().render(gs, cam)
    fast = get_rasterizer("gsplat").render(
        Gaussians3D(
            means=gs.means.cuda(),
            quats=gs.quats.cuda(),
            log_scales=gs.log_scales.cuda(),
            opacity=gs.opacity.cuda(),
            sh=gs.sh.cuda(),
        ),
        cam,
    )
    diff = (ref.color - fast.color.cpu()).abs().mean()
    assert diff < 0.02, f"mean abs difference {diff}"
