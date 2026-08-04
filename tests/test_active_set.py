"""Priority-ranked active-set gradient masking (protocol 20260731_active_set_updates):
selection is deterministic and score-driven, masked rows with fresh optimizer state stay
bit-identical across steps, and the default dense configuration changes nothing."""

from __future__ import annotations

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.active_set import ActiveSetConfig, ActiveSetSelector
from rtgs.optim.trainer import TrainConfig, Trainer


def _params(n: int = 8) -> dict[str, torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(7)
    return {
        "means": torch.nn.Parameter(torch.randn(n, 3, generator=gen)),
        "opacities": torch.nn.Parameter(torch.randn(n, generator=gen)),
    }


def test_config_validation():
    with pytest.raises(ValueError):
        ActiveSetConfig(fraction=0.0)
    with pytest.raises(ValueError):
        ActiveSetConfig(fraction=1.5)
    with pytest.raises(ValueError):
        ActiveSetConfig(refresh_every=0)
    with pytest.raises(ValueError):
        ActiveSetConfig(selection="bogus")
    with pytest.raises(ValueError):
        ActiveSetConfig(weight_count=-1.0)
    assert not ActiveSetConfig().enabled
    assert ActiveSetConfig(fraction=0.5).enabled


def test_priority_selection_prefers_high_gradient_rows():
    config = ActiveSetConfig(fraction=0.25, weight_count=0.0, weight_variance=0.0, weight_age=0.0)
    params = _params(8)
    selector = ActiveSetSelector(config, 8, torch.device("cpu"))
    grad2d = torch.tensor([0.0, 9.0, 0.1, 0.2, 8.0, 0.3, 0.0, 0.05])
    selector.maybe_refresh(1, params, grad2d=grad2d, count=torch.zeros(8))
    assert selector.mask.sum() == 2
    assert bool(selector.mask[1]) and bool(selector.mask[4])


def test_random_selection_is_generator_deterministic():
    config = ActiveSetConfig(fraction=0.5, selection="random")
    params = _params(8)
    masks = []
    for _ in range(2):
        selector = ActiveSetSelector(config, 8, torch.device("cpu"))
        gen = torch.Generator().manual_seed(13)
        selector.maybe_refresh(1, params, generator=gen)
        masks.append(selector.mask.clone())
    assert torch.equal(masks[0], masks[1])
    assert masks[0].sum() == 4


def test_masked_rows_with_fresh_state_stay_bit_identical():
    params = _params(8)
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-2) for name, parameter in params.items()
    }
    config = ActiveSetConfig(fraction=0.5, weight_count=0.0, weight_variance=0.0, weight_age=0.0)
    selector = ActiveSetSelector(config, 8, torch.device("cpu"))
    grad2d = torch.arange(8.0)
    initial = {name: parameter.detach().clone() for name, parameter in params.items()}
    for step in (1, 2, 3):
        for parameter in params.values():
            parameter.grad = torch.randn(
                parameter.shape, generator=torch.Generator().manual_seed(step)
            )
        selector.observe(params, None, step)
        selector.maybe_refresh(step, params, grad2d=grad2d, count=torch.zeros(8))
        active = selector.mask_gradients(params)
        assert active == 4
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    inactive = ~selector.mask
    for name, parameter in params.items():
        assert torch.equal(parameter.detach()[inactive], initial[name][inactive])
        assert not torch.equal(parameter.detach()[selector.mask], initial[name][selector.mask])
    assert selector.cumulative_row_updates == 12
    assert selector.cumulative_steps == 3


def test_resize_preserves_prefix_and_stamps_birth():
    config = ActiveSetConfig(fraction=0.5)
    selector = ActiveSetSelector(config, 4, torch.device("cpu"))
    selector.grad_ema.copy_(torch.arange(4.0))
    params = _params(6)
    selector.observe(params, None, step=9)
    assert selector.n == 6
    assert torch.equal(selector.birth[4:], torch.full((2,), 9, dtype=torch.long))
    params_small = _params(3)
    selector.observe(params_small, None, step=10)
    assert selector.n == 3


def test_visibility_counts_accumulate_only_valid_rows():
    config = ActiveSetConfig(fraction=0.5)
    selector = ActiveSetSelector(config, 4, torch.device("cpu"))
    params = _params(4)
    selector.observe(params, torch.tensor([0, 2, 2, 9]), step=1)
    assert torch.equal(selector.visibility, torch.tensor([1.0, 0.0, 2.0, 0.0]))


def test_trainer_masks_updates_and_records_diagnostics():
    scene = make_synthetic_scene(n_gaussians=8, n_cameras=4, image_size=16, seed=5)
    init = scene.gt_gaussians.detach()
    init.means += 0.05
    cfg = TrainConfig(
        iterations=6,
        rasterizer="torch",
        densify=False,
        eval_every=6,
        ssim_lambda=0.0,
        active_fraction=0.5,
        active_refresh_every=100,
    )
    refined, history = Trainer(cfg).train(scene, init)
    diagnostics = history["active_set"]
    assert diagnostics is not None
    assert diagnostics["cumulative_steps"] == 6
    assert diagnostics["cumulative_row_updates"] == 6 * 4
    assert diagnostics["refresh_count"] == 1
    assert refined.n == 8


def test_trainer_default_records_no_active_set():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=2, image_size=16, seed=6)
    cfg = TrainConfig(
        iterations=2, rasterizer="torch", densify=False, eval_every=2, ssim_lambda=0.0
    )
    _, history = Trainer(cfg).train(scene, scene.gt_gaussians.detach())
    assert history["active_set"] is None
