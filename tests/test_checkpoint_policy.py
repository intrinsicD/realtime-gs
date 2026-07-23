"""Opt-in train-only best-checkpoint selection (never selects on held-out views)."""

from __future__ import annotations

import pytest
import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.trainer import TrainConfig, Trainer


def _config(**overrides) -> TrainConfig:
    values = {
        "iterations": 6,
        "rasterizer": "torch",
        "device": "cpu",
        "densify": False,
        "target_sh_degree": 0,
        "ssim_lambda": 0.0,
        "use_masks": False,
        "random_background": False,
        "eval_every": 2,
        "seed": 0,
    }
    values.update(overrides)
    return TrainConfig(**values)


def _split_scene():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=3, image_size=12, seed=1)
    scene.test_indices = [2]  # hold out the last view
    return scene


def _init(scene):
    init = scene.gt_gaussians.detach()
    init.means += 0.03
    return init


def test_checkpoint_policy_defaults_to_final():
    assert TrainConfig().checkpoint_policy == "final"


def test_final_policy_leaves_history_and_return_unchanged():
    scene = _split_scene()
    _, history = Trainer(_config()).train(scene, _init(scene))
    assert "selected_step" not in history
    assert "train_psnr" not in history


def test_best_train_psnr_selects_on_training_views_only():
    scene = _split_scene()
    assert scene.testing_views == [2]
    assert scene.training_views == [0, 1]
    refined, history = Trainer(_config(checkpoint_policy="best_train_psnr")).train(
        scene, _init(scene)
    )

    assert isinstance(refined, Gaussians3D)
    assert history["train_psnr"], "train-view PSNR is recorded at eval steps"
    steps = [step for step, _ in history["train_psnr"]]
    values = [value for _, value in history["train_psnr"]]
    assert history["selected_step"] in steps
    assert history["selected_train_psnr"] == pytest.approx(max(values))
    # The discipline: selection never touches held-out views.
    assert history["checkpoint_selection_views"] == scene.training_views
    assert 2 not in history["checkpoint_selection_views"]
    for field in (refined.means, refined.opacity, refined.sh):
        assert torch.isfinite(field).all()


def test_best_train_psnr_is_deterministic():
    scene_a = _split_scene()
    refined_a, _ = Trainer(_config(checkpoint_policy="best_train_psnr")).train(
        scene_a, _init(scene_a)
    )
    scene_b = _split_scene()
    refined_b, _ = Trainer(_config(checkpoint_policy="best_train_psnr")).train(
        scene_b, _init(scene_b)
    )
    assert torch.allclose(refined_a.means, refined_b.means)


def test_invalid_checkpoint_policy_is_rejected():
    scene = _split_scene()
    with pytest.raises(ValueError, match="checkpoint_policy"):
        Trainer(_config(checkpoint_policy="best_test_psnr")).train(scene, _init(scene))
