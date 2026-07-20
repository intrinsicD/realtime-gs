"""Torch coarse-visibility margin semantics and configuration plumbing."""

import math

import pytest
import torch

import rtgs.optim.trainer as trainer_module
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import DEFAULT_VISIBILITY_MARGIN_SIGMA, get_rasterizer
from rtgs.render.torch_ref import TorchRasterizer


def _camera() -> Camera:
    return Camera(
        fx=16.0,
        fy=16.0,
        cx=8.5,
        cy=8.5,
        width=17,
        height=17,
        R=torch.eye(3),
        t=torch.zeros(3),
    )


def _gaussians_at_projected_u(projected_u: list[float]) -> Gaussians3D:
    z = 2.0
    means = torch.tensor(
        [[(u - 8.5) * z / 16.0, 0.0, z] for u in projected_u],
        dtype=torch.float32,
    )
    covariance = torch.diag(torch.tensor([0.1**2, 0.1**2, 1e-8]))
    return Gaussians3D.from_means_covs(
        means=means,
        covs=covariance[None].expand(len(projected_u), -1, -1).clone(),
        colors=torch.ones(len(projected_u), 3),
        opacity=torch.full((len(projected_u),), 0.9),
    )


def test_default_visibility_margin_matches_explicit_three_exactly():
    camera = _camera()
    gaussians = _gaussians_at_projected_u([8.5, 13.0])

    default = TorchRasterizer().render(gaussians, camera)
    explicit = TorchRasterizer(visibility_margin_sigma=DEFAULT_VISIBILITY_MARGIN_SIGMA).render(
        gaussians, camera
    )

    assert torch.equal(default.color, explicit.color)
    assert torch.equal(default.alpha, explicit.alpha)
    assert torch.equal(default.depth, explicit.depth)
    assert torch.equal(default.visible, explicit.visible)


def test_sqrt_twelve_margin_admits_edge_support_excluded_by_three_sigma():
    camera = _camera()
    # u=19.65 lies outside the current 3-sigma image margin, but its nearest
    # right-edge pixel remains inside the established q<12 kernel support.
    gaussians = _gaussians_at_projected_u([8.5, 19.65])

    current = TorchRasterizer().render(gaussians, camera)
    support_safe = TorchRasterizer(visibility_margin_sigma=math.sqrt(12.0)).render(
        gaussians, camera
    )

    assert torch.equal(current.visible, torch.tensor([0]))
    assert torch.equal(support_safe.visible, torch.tensor([0, 1]))
    assert float(current.alpha[:, -1].max()) == 0.0
    assert float(support_safe.alpha[:, -1].max()) > 0.0


def test_expanded_margin_preserves_established_order_across_exact_depth_tie():
    camera = _camera()
    # On CPU, torch.argsort's unspecified tie order changes deterministically
    # between 16 and 17 equal values.  The final Gaussian is admitted only by
    # the expanded margin, reproducing the attribution defect behind the retry.
    gaussians = _gaussians_at_projected_u([8.5] * 16 + [19.65])

    current = TorchRasterizer().render(gaussians, camera)
    support_safe = TorchRasterizer(visibility_margin_sigma=math.sqrt(12.0)).render(
        gaussians, camera
    )

    old_expanded_order = torch.argsort(torch.full((17,), 2.0))
    old_filtered_order = old_expanded_order[old_expanded_order < 16]
    assert not torch.equal(old_filtered_order, current.visible)

    expanded_current_order = support_safe.visible[support_safe.visible < 16]
    assert torch.equal(expanded_current_order, current.visible)
    assert support_safe.visible[-1].item() == 16


def test_expanded_margin_is_bit_exact_when_it_admits_nothing_new():
    camera = _camera()
    gaussians = _gaussians_at_projected_u([4.0, 8.5, 13.0])

    current = TorchRasterizer().render(gaussians, camera)
    support_safe = TorchRasterizer(visibility_margin_sigma=math.sqrt(12.0)).render(
        gaussians, camera
    )

    assert torch.equal(support_safe.visible, current.visible)
    assert torch.equal(support_safe.color, current.color)
    assert torch.equal(support_safe.alpha, current.alpha)
    assert torch.equal(support_safe.depth, current.depth)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_visibility_margin_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="finite and positive"):
        TorchRasterizer(visibility_margin_sigma=value)
    with pytest.raises(ValueError, match="finite and positive"):
        get_rasterizer("torch", visibility_margin_sigma=value)


def test_gsplat_rejects_nondefault_visibility_margin_before_backend_import():
    with pytest.raises(NotImplementedError, match="visibility margins"):
        get_rasterizer("gsplat", visibility_margin_sigma=math.sqrt(12.0))


def test_trainer_passes_visibility_margin_to_renderer(monkeypatch, tiny_scene):
    captured: dict[str, float] = {}
    real_get_rasterizer = trainer_module.get_rasterizer

    def capture_get_rasterizer(*args, **kwargs):
        captured["visibility_margin_sigma"] = kwargs["visibility_margin_sigma"]
        return real_get_rasterizer(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "get_rasterizer", capture_get_rasterizer)
    configured = math.sqrt(12.0)
    Trainer(
        TrainConfig(
            iterations=0,
            rasterizer="torch",
            device="cpu",
            densify=False,
            visibility_margin_sigma=configured,
        )
    ).train(tiny_scene, tiny_scene.gt_gaussians.detach())

    assert captured["visibility_margin_sigma"] == configured
