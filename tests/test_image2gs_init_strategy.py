"""Stage 1: opt-in feature-aware oriented init strategy (``FitConfig.init_strategy``).

Unit coverage for the structure-tensor / WSE math lives in ``tests/test_structure_init.py``; this
file exercises the integration into the native fit.
"""

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.image2gs.fit import FitConfig, fit_image


def _smooth_image(h: int = 24, w: int = 24) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return torch.stack([xx, yy, 0.5 * (xx + yy)], dim=-1).clamp(0, 1)


def test_init_strategy_defaults_to_gradient():
    assert FitConfig().init_strategy == "gradient"


def test_structure_tensor_init_runs_and_is_valid():
    image = _smooth_image()
    h, w = image.shape[:2]
    cfg = FitConfig(init_strategy="structure_tensor", n_gaussians=40, iterations=80, log_every=80)
    gaussians, history = fit_image(image, cfg, seed=0)

    assert isinstance(gaussians, Gaussians2D)
    assert gaussians.n == 40
    for field in (gaussians.xy, gaussians.chol, gaussians.color, gaussians.weight):
        assert torch.isfinite(field).all()
    assert (gaussians.xy[:, 0] >= 0).all() and (gaussians.xy[:, 0] < w).all()
    assert (gaussians.xy[:, 1] >= 0).all() and (gaussians.xy[:, 1] < h).all()
    assert (gaussians.chol[:, 0] > 0).all() and (gaussians.chol[:, 2] > 0).all()
    assert history["final_psnr"] > 10.0  # loose floor: even coverage fits a smooth image


def test_structure_tensor_init_is_deterministic():
    image = _smooth_image()
    cfg = FitConfig(init_strategy="structure_tensor", n_gaussians=32, iterations=25, log_every=25)
    g1, _ = fit_image(image, cfg, seed=0)
    g2, _ = fit_image(image, cfg, seed=0)
    assert torch.allclose(g1.xy, g2.xy)
    assert torch.allclose(g1.chol, g2.chol)


def test_structure_density_sampling_runs_with_pool():
    image = _smooth_image()
    cfg = FitConfig(
        init_strategy="structure_tensor",
        structure_sampling="density",
        pool=True,
        pool_capacity=64,
        n_gaussians=32,
        iterations=25,
        log_every=25,
    )
    gaussians, history = fit_image(image, cfg, seed=0)
    assert gaussians.n == 32
    assert history["pool_capacity"] == 64
    assert history["live_count"] == 32
    assert all(
        torch.isfinite(field).all()
        for field in (gaussians.xy, gaussians.chol, gaussians.color, gaussians.weight)
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"init_strategy": "not-a-strategy"}, "init_strategy"),
        ({"init_strategy": "structure_tensor", "backend": "structsplat"}, "native backend"),
        ({"structure_sampling": "density"}, "structure_tensor"),
        (
            {"init_strategy": "structure_tensor", "structure_sampling": "not-a-mode"},
            "structure_sampling",
        ),
    ],
)
def test_init_strategy_validation(overrides, match):
    image = _smooth_image()
    with pytest.raises(ValueError, match=match):
        fit_image(image, FitConfig(iterations=3, **overrides), seed=0)
