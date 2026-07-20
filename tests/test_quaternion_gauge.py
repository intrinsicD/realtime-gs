"""Focused CPU contracts for the opt-in quaternion update research seam."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    _apply_quaternion_update_policy_,
)


def _config(**overrides) -> TrainConfig:
    values = {
        "iterations": 2,
        "rasterizer": "torch",
        "device": "cpu",
        "densify": False,
        "target_sh_degree": 0,
        "ssim_lambda": 0.0,
        "use_masks": False,
        "random_background": False,
        "eval_every": 2,
        "seed": 3,
    }
    values.update(overrides)
    return TrainConfig(**values)


def _scene_and_scaled_init():
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=2, image_size=10, seed=41)
    init = scene.gt_gaussians.detach()
    init.quats *= 4.0
    init.means += 0.02
    return scene, init


@pytest.mark.parametrize(
    "policy",
    ["unit_retraction", "tangent_displacement_retraction"],
)
def test_quaternion_policy_equations_and_adam_state_nonmutation(policy):
    q_old = F.normalize(
        torch.tensor([[1.0, 0.2, -0.1, 0.3], [0.7, -0.4, 0.5, 0.1]], dtype=torch.float32),
        dim=-1,
    )
    parameter = torch.nn.Parameter(q_old.clone())
    optimizer = torch.optim.Adam([parameter], lr=1e-3, eps=1e-15)
    (parameter * torch.tensor([0.3, -0.2, 0.7, -0.5])).sum().backward()
    optimizer.step()
    q_star = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() for key, value in optimizer.state[parameter].items()
    }
    unrelated = torch.tensor([3.0, 4.0])
    if policy == "unit_retraction":
        expected = F.normalize(q_star, dim=-1)
    else:
        delta = q_star - q_old
        tangent = delta - q_old * (q_old * delta).sum(dim=-1, keepdim=True)
        expected = F.normalize(q_old + tangent, dim=-1)

    _apply_quaternion_update_policy_(parameter, q_old, q_star, policy)

    assert torch.equal(parameter, expected)
    assert torch.equal(unrelated, torch.tensor([3.0, 4.0]))
    for key, value in optimizer.state[parameter].items():
        assert torch.equal(value, state_before[key])


def test_candidate_entry_occurs_once_and_captures_actual_stored_tensor():
    scene, init = _scene_and_scaled_init()
    expected = F.normalize(init.quats, dim=-1)
    captures = []

    def capture_and_mutate(snapshot):
        assert not torch.is_grad_enabled()
        captures.append(snapshot.detach())
        snapshot.quats.fill_(float("nan"))
        snapshot.means.zero_()

    refined, _ = Trainer(_config(iterations=0, quaternion_update_policy="unit_retraction")).train(
        scene, init, initialization_callback=capture_and_mutate
    )

    assert len(captures) == 1
    assert torch.equal(captures[0].quats, expected)
    assert torch.equal(refined.quats, expected)
    assert torch.equal(refined.means, init.means)


def test_candidate_modes_preserve_identical_non_quaternion_entry_fields():
    scene, init = _scene_and_scaled_init()
    captures = {}
    for policy in (
        "current",
        "unit_retraction",
        "tangent_displacement_retraction",
    ):
        Trainer(_config(iterations=0, quaternion_update_policy=policy)).train(
            scene,
            init.detach(),
            initialization_callback=lambda snapshot, name=policy: captures.setdefault(
                name, snapshot.detach()
            ),
        )
    for field in ("means", "log_scales", "opacity", "sh"):
        reference = getattr(captures["current"], field)
        for policy in ("unit_retraction", "tangent_displacement_retraction"):
            assert torch.equal(getattr(captures[policy], field), reference)


def test_quaternion_step_observer_is_isolated_and_current_path_is_exact():
    scene, init = _scene_and_scaled_init()
    config = _config(quaternion_update_policy="current")
    reference_final, reference_history = Trainer(config).train(scene, init.detach())
    records = []

    def observe_and_mutate(q_old, q_star, q_new, step):
        assert not torch.is_grad_enabled()
        records.append((q_old.clone(), q_star.clone(), q_new.clone(), step))
        q_old.fill_(float("nan"))
        q_star.zero_()
        q_new.fill_(9.0)

    observed_final, observed_history = Trainer(config).train(
        scene,
        init.detach(),
        quaternion_step_callback=observe_and_mutate,
    )

    assert [record[3] for record in records] == [1, 2]
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(observed_final, field), getattr(reference_final, field))
    assert {key: value for key, value in observed_history.items() if key != "elapsed"} == {
        key: value for key, value in reference_history.items() if key != "elapsed"
    }


def test_default_and_explicit_current_are_exact_in_fields_history_and_rng():
    scene, init = _scene_and_scaled_init()
    default_config = _config()
    explicit_config = copy.deepcopy(default_config)
    explicit_config.quaternion_update_policy = "current"
    default_final, default_history = Trainer(default_config).train(scene, init.detach())
    explicit_final, explicit_history = Trainer(explicit_config).train(scene, init.detach())
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(default_final, field), getattr(explicit_final, field))
    assert {key: value for key, value in default_history.items() if key != "elapsed"} == {
        key: value for key, value in explicit_history.items() if key != "elapsed"
    }
    assert default_history["sampled_train_views"] == explicit_history["sampled_train_views"]


def test_candidate_rejects_density_unknown_policy_and_invalid_norm_before_callback():
    scene, init = _scene_and_scaled_init()
    with pytest.raises(ValueError, match="densify=False"):
        Trainer(_config(densify=True, quaternion_update_policy="unit_retraction")).train(
            scene, init.detach()
        )
    with pytest.raises(ValueError, match="must be one of"):
        Trainer(_config(quaternion_update_policy="not-a-policy")).train(scene, init.detach())

    invalid = init.detach()
    invalid.quats[0].zero_()
    called = False

    def callback(_snapshot):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="norm at or below"):
        Trainer(_config(quaternion_update_policy="unit_retraction")).train(
            scene, invalid, initialization_callback=callback
        )
    assert not called


def test_candidate_step_callback_reports_frozen_update_equations():
    scene, init = _scene_and_scaled_init()
    for policy in ("unit_retraction", "tangent_displacement_retraction"):
        records = []
        Trainer(_config(iterations=1, eval_every=1, quaternion_update_policy=policy)).train(
            scene,
            init.detach(),
            quaternion_step_callback=lambda q_old, q_star, q_new, step, sink=records: sink.append(
                (q_old, q_star, q_new, step)
            ),
        )
        assert len(records) == 1
        q_old, q_star, q_new, step = records[0]
        if policy == "unit_retraction":
            expected = F.normalize(q_star, dim=-1)
        else:
            delta = q_star - q_old
            tangent = delta - q_old * (q_old * delta).sum(dim=-1, keepdim=True)
            expected = F.normalize(q_old + tangent, dim=-1)
        assert step == 1
        assert torch.equal(q_new, expected)
        assert torch.max((torch.linalg.vector_norm(q_new, dim=-1) - 1.0).abs()) <= 2e-5
