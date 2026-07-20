"""Reference rasterizer semantics + gsplat parity (parity requires CUDA)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.base import get_rasterizer
from rtgs.render.gsplat_backend import (
    _remove_shadowed_gsplat_editable_finders,
    _visible_indices,
)
from rtgs.render.torch_ref import TorchRasterizer, TorchRenderProgress


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


def test_torch_rasterizer_emits_row_chunk_progress():
    records: list[TorchRenderProgress] = []
    camera = _front_camera(size=33)

    TorchRasterizer(row_chunk=7, progress_callback=records.append).render(_one_gaussian(), camera)

    assert [record.completed_rows for record in records] == [7, 14, 21, 28, 33]
    assert all(record.total_rows == 33 for record in records)
    assert all(record.visible_gaussians == 1 for record in records)
    assert all(record.elapsed_seconds >= 0 for record in records)


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


def test_sh_color_diagnostics_expose_hard_floor_gradient_suppression():
    g = _one_gaussian(color=(-0.2, 0.4, 0.6))
    g.sh.requires_grad_(True)
    renderer = TorchRasterizer(collect_sh_color_diagnostics=True)
    out = renderer.render(g, _front_camera(size=17))
    assert out.sh_color_diagnostics is not None
    diagnostics = out.sh_color_diagnostics
    loss = (out.color - 1.0).square().mean()
    loss.backward()
    assert diagnostics.preactivation.grad is not None
    assert diagnostics.activated.grad is not None
    negative = diagnostics.preactivation.detach() < 0.0
    assert bool(negative.any())
    assert torch.equal(
        diagnostics.preactivation.grad[negative],
        torch.zeros_like(diagnostics.preactivation.grad[negative]),
    )
    assert bool((diagnostics.activated.grad[negative] < 0.0).all())


def test_opt_in_smu1_and_straight_through_renderer_semantics():
    g = _one_gaussian(color=(-0.2, 0.4, 0.6))
    cam = _front_camera(size=17)
    hard = TorchRasterizer().render(g, cam)
    smooth = TorchRasterizer(sh_color_activation="smu1").render(g, cam)
    straight_through = TorchRasterizer(
        sh_color_activation="hard_forward_smu1_negative_gradient"
    ).render(g, cam)
    assert torch.equal(straight_through.color, hard.color)
    assert bool((smooth.color[..., 0] > hard.color[..., 0]).any())
    assert float((smooth.color - hard.color).abs().max()) <= 1.0 / 255.0 + 1e-6


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
    assert isinstance(get_rasterizer("auto", device="cpu"), TorchRasterizer)
    smooth = get_rasterizer("torch", sh_color_activation="smu1")
    assert isinstance(smooth, TorchRasterizer)
    assert smooth.sh_color_activation == "smu1"
    with pytest.raises(ValueError):
        get_rasterizer("nonsense")


def test_gsplat_radius_layouts_map_visibility():
    scalar = torch.tensor([[1.0, 0.0, 2.0]])
    axes = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [2.0, 3.0]]])
    expected = torch.tensor([0, 2])
    assert torch.equal(_visible_indices(scalar), expected)
    assert torch.equal(_visible_indices(axes), expected)


def test_shadowed_gsplat_editable_submodule_is_removed(monkeypatch, tmp_path):
    package_root = tmp_path / "current" / "gsplat"
    package_root.mkdir(parents=True)
    shadow = tmp_path / "stale" / "gsplat" / "csrc.so"
    matching = package_root / "csrc.so"

    class ShadowFinder:
        def find_spec(self, fullname, path):
            del fullname, path
            return SimpleNamespace(origin=str(shadow))

    class MatchingFinder:
        def find_spec(self, fullname, path):
            del fullname, path
            return SimpleNamespace(origin=str(matching))

    ShadowFinder.__module__ = "__editable___gsplat_1_1_3_finder"
    MatchingFinder.__module__ = "__editable___gsplat_current_finder"
    shadow_finder = ShadowFinder()
    matching_finder = MatchingFinder()
    monkeypatch.setattr(sys, "meta_path", [shadow_finder, matching_finder])
    module = SimpleNamespace(
        __file__=str(package_root / "__init__.py"),
        __path__=[str(package_root)],
    )

    assert _remove_shadowed_gsplat_editable_finders(module) == (str(shadow.resolve()),)
    assert sys.meta_path == [matching_finder]


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gsplat_smu1_precomputed_color_parity_and_gradient(tiny_scene):
    """The opt-in RGB path must match SMU-1 reference semantics and backpropagate."""
    pytest.importorskip("gsplat")
    source = tiny_scene.gt_gaussians.with_sh_degree(1)
    source.sh[:, 3] = 0.4
    cam = tiny_scene.cameras[0]
    reference = TorchRasterizer(sh_color_activation="smu1").render(source, cam)
    cuda_source = source.to("cuda")
    cuda_source.sh.requires_grad_(True)
    fast = get_rasterizer("gsplat", sh_color_activation="smu1").render(cuda_source, cam)
    difference = (reference.color - fast.color.cpu()).abs().mean()
    assert difference < 0.02, f"mean abs difference {difference}"
    fast.color.mean().backward()
    assert cuda_source.sh.grad is not None
    assert torch.isfinite(cuda_source.sh.grad).all()
