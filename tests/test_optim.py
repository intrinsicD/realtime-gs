"""Stage 3: trainer and density control."""

import hashlib
import json

import pytest
import torch

from rtgs.core.gaussians3d import quat_to_rotmat
from rtgs.data.synthetic import make_gt_gaussians, make_synthetic_scene
from rtgs.optim.density import (
    DensityConfig,
    DensityController,
    apply_selected_birth_surgery,
)
from rtgs.optim.strategies import enforce_budget, validate_strategy_name
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    _resolve_means_lr_final_factor,
    _resolve_opacity_logit_epsilon,
    _resolve_schedule_iterations,
    _resolve_sh_interval,
)


def test_trainer_improves_perturbed_gt():
    """Perturbing GT and retraining must recover a good chunk of the lost PSNR."""
    scene = make_synthetic_scene(n_gaussians=15, n_cameras=6, image_size=24, seed=1)
    gt = scene.gt_gaussians
    gen = torch.Generator().manual_seed(3)
    perturbed = gt.detach()
    perturbed.means += 0.05 * torch.randn(gt.n, 3, generator=gen)
    perturbed.sh += 0.15 * torch.randn(gt.sh.shape, generator=gen)

    psnr_before = Trainer.evaluate(scene, perturbed)
    cfg = TrainConfig(
        iterations=60, rasterizer="torch", densify=False, eval_every=30, ssim_lambda=0.0
    )
    refined, history = Trainer(cfg).train(scene, perturbed)
    psnr_after = Trainer.evaluate(scene, refined)
    assert psnr_after > psnr_before + 1.0, (psnr_before, psnr_after)
    assert history["loss"][-1] < history["loss"][0]


def test_short_training_activates_and_optimizes_all_sh_bands():
    scene = make_synthetic_scene(n_gaussians=8, n_cameras=4, image_size=16, seed=7)
    init = scene.gt_gaussians.detach()
    init.means += 0.08
    config = TrainConfig(
        iterations=8,
        rasterizer="torch",
        densify=False,
        target_sh_degree=3,
        ssim_lambda=0.0,
        eval_every=8,
    )
    refined, history = Trainer(config).train(scene, init)
    assert history["resolved_sh_degree_interval"] == 2
    assert history["active_sh_degree"][-1] == (8, 3)
    assert bool((refined.sh[:, 1:].abs() > 0).any())


def test_trainer_records_reproducible_view_and_hard_floor_diagnostics():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=4, image_size=16, seed=21)
    init = scene.gt_gaussians.detach()
    init.sh[:, 0] -= 4.0
    config = TrainConfig(
        iterations=4,
        rasterizer="torch",
        device="cpu",
        densify=False,
        target_sh_degree=1,
        sh_degree_interval=2,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        collect_sh_color_diagnostics=True,
        eval_every=4,
        seed=3,
    )
    _, history = Trainer(config).train(scene, init)
    assert len(history["sampled_train_views"]) == config.iterations
    assert len(history["sh_color_diagnostics"]) == config.iterations
    populated = [
        entry for entry in history["sh_color_diagnostics"] if entry["observation_count"] > 0
    ]
    assert populated
    assert sum(entry["negative_count"] for entry in populated) > 0
    assert all(entry["negative_raw_gradient_nonzero_count"] == 0 for entry in populated)
    assert all(entry["negative_raw_gradient_max_abs"] == 0.0 for entry in populated)
    assert all(entry["positive_raw_upstream_max_abs_error"] == 0.0 for entry in populated)
    assert all(entry["upstream_l1"] > 0.0 for entry in populated)


def test_checkpoint_callback_is_isolated_and_none_preserves_default_exactly():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=4, image_size=12, seed=22)
    init = scene.gt_gaussians.detach()
    init.means += 0.03
    config = TrainConfig(
        iterations=4,
        rasterizer="torch",
        device="cpu",
        densify=False,
        target_sh_degree=1,
        sh_degree_interval=2,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        eval_every=2,
        seed=5,
    )

    default_final, default_history = Trainer(config).train(scene, init.detach())
    none_final, none_history = Trainer(config).train(scene, init.detach(), checkpoint_callback=None)
    callback_steps = []

    def mutate_snapshot(snapshot, step):
        assert not torch.is_grad_enabled()
        assert all(
            not getattr(snapshot, field).requires_grad
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        )
        callback_steps.append(step)
        snapshot.means.fill_(float("nan"))
        snapshot.sh.zero_()

    observed_final, observed_history = Trainer(config).train(
        scene, init.detach(), checkpoint_callback=mutate_snapshot
    )

    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(default_final, field), getattr(none_final, field))
        assert torch.equal(getattr(default_final, field), getattr(observed_final, field))
    ignored_history_keys = {"elapsed", "checkpoint_callback_seconds"}
    for history in (none_history, observed_history):
        assert {
            key: value for key, value in history.items() if key not in ignored_history_keys
        } == {
            key: value for key, value in default_history.items() if key not in ignored_history_keys
        }
    assert default_history["checkpoint_callback_seconds"] == 0.0
    assert none_history["checkpoint_callback_seconds"] == 0.0
    assert observed_history["checkpoint_callback_seconds"] >= 0.0
    assert callback_steps == [2, 4]


def test_segmented_training_uses_global_callbacks_sh_and_means_lr_schedule():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=3, image_size=12, seed=23)
    init = scene.gt_gaussians.detach()
    init.means += 0.03
    config = TrainConfig(
        iterations=4,
        iteration_offset=4,
        schedule_iterations=8,
        rasterizer="torch",
        device="cpu",
        densify=False,
        target_sh_degree=3,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        eval_every=2,
        seed=5,
    )
    callback_steps = []

    _, history = Trainer(config).train(
        scene,
        init,
        checkpoint_callback=lambda _snapshot, step: callback_steps.append(step),
    )

    gamma = 0.01 ** (1.0 / 8)
    assert callback_steps == [6, 8]
    assert history["active_sh_degree"] == [(6, 2), (8, 3)]
    assert history["iteration_offset"] == 4
    assert history["segment_iterations"] == 4
    assert history["schedule_iterations"] == 8
    assert history["means_lr_final"] == pytest.approx(
        history["means_lr_initial"] * gamma**4,
        rel=1e-12,
    )
    assert history["means_lr_final_factor"] == 0.01
    assert history["means_lr_gamma"] == pytest.approx(gamma, rel=1e-12)


def test_constant_means_lr_polish_factor_is_cpu_testable():
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=2, image_size=8, seed=25)
    config = TrainConfig(
        iterations=2,
        iteration_offset=4,
        schedule_iterations=6,
        means_lr_final_factor=1.0,
        rasterizer="torch",
        device="cpu",
        densify=False,
        target_sh_degree=3,
        sh_degree_interval=1,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        eval_every=1,
        seed=7,
    )

    _, history = Trainer(config).train(scene, scene.gt_gaussians.detach())

    assert history["active_sh_degree"] == [(5, 3), (6, 3)]
    assert history["means_lr_gamma"] == 1.0
    assert history["means_lr_initial"] == history["means_lr_final"]


def test_means_lr_final_factor_default_and_validation():
    assert _resolve_means_lr_final_factor(TrainConfig()) == 0.01
    assert _resolve_means_lr_final_factor(TrainConfig(means_lr_final_factor=1)) == 1.0
    with pytest.raises(TypeError, match="finite positive"):
        _resolve_means_lr_final_factor(TrainConfig(means_lr_final_factor=True))
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite positive"):
            _resolve_means_lr_final_factor(TrainConfig(means_lr_final_factor=value))


def test_opacity_logit_epsilon_preserves_polish_entry_values_on_cpu():
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=2, image_size=8, seed=26)
    init = scene.gt_gaussians.detach()
    init.opacity = torch.tensor([0.99995, 0.9999995, 0.25, 0.0000005])
    entries = {}

    for label, epsilon in (("default", 1e-4), ("polish", 1e-6)):
        Trainer(
            TrainConfig(
                iterations=0,
                rasterizer="torch",
                device="cpu",
                densify=False,
                target_sh_degree=0,
                opacity_logit_epsilon=epsilon,
            )
        ).train(
            scene,
            init.detach(),
            initialization_callback=lambda snapshot, key=label: entries.__setitem__(
                key, snapshot.opacity.clone()
            ),
        )

    default_expected = init.opacity.clamp(1e-4, 1.0 - 1e-4)
    polish_expected = init.opacity.clamp(1e-6, 1.0 - 1e-6)
    torch.testing.assert_close(entries["default"], default_expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(entries["polish"], polish_expected, rtol=1e-6, atol=1e-7)
    assert entries["polish"][0] > entries["default"][0]
    assert entries["polish"][2] == entries["default"][2]


def test_opacity_logit_epsilon_validation():
    assert _resolve_opacity_logit_epsilon(TrainConfig()) == 1e-4
    assert _resolve_opacity_logit_epsilon(TrainConfig(opacity_logit_epsilon=1e-6)) == 1e-6
    with pytest.raises(TypeError, match="finite number"):
        _resolve_opacity_logit_epsilon(TrainConfig(opacity_logit_epsilon=True))
    for value in (0.0, -1.0, 0.5, 1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite number"):
            _resolve_opacity_logit_epsilon(TrainConfig(opacity_logit_epsilon=value))


def test_segmented_training_uses_global_classic_density_schedule():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=3, image_size=12, seed=24)
    config = TrainConfig(
        iterations=2,
        iteration_offset=6,
        schedule_iterations=8,
        rasterizer="torch",
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            start_iter=7,
            stop_iter=7,
            every=1,
            grad_threshold=float("inf"),
            split_scale_frac=100.0,
            prune_scale_frac=100.0,
            max_gaussians=32,
        ),
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        eval_every=2,
        seed=6,
    )

    _, history = Trainer(config).train(scene, scene.gt_gaussians.detach())

    assert [record["iteration"] for record in history["density_stats"]] == [7]


def test_sh_schedule_validation_and_long_run_default():
    assert _resolve_sh_interval(TrainConfig(iterations=30_000, target_sh_degree=3)) == 1_000
    assert _resolve_sh_interval(TrainConfig(iterations=1_000, target_sh_degree=3)) == 250
    assert (
        _resolve_sh_interval(
            TrainConfig(
                iterations=4,
                iteration_offset=4,
                schedule_iterations=8,
                target_sh_degree=3,
            )
        )
        == 2
    )
    assert _resolve_schedule_iterations(TrainConfig(iterations=0)) == 1
    with pytest.raises(ValueError, match="positive"):
        _resolve_sh_interval(TrainConfig(sh_degree_interval=0))
    with pytest.raises(ValueError, match="nonnegative"):
        _resolve_schedule_iterations(TrainConfig(iterations=-1))
    with pytest.raises(ValueError, match="nonnegative"):
        _resolve_schedule_iterations(TrainConfig(iteration_offset=-1))
    with pytest.raises(ValueError, match="segmented-run iterations must be positive"):
        _resolve_schedule_iterations(TrainConfig(iterations=0, schedule_iterations=1))
    with pytest.raises(ValueError, match="must not exceed"):
        _resolve_schedule_iterations(
            TrainConfig(iterations=4, iteration_offset=5, schedule_iterations=8)
        )


def test_segmented_fields_preserve_train_config_positional_compatibility():
    config = TrainConfig(7, 3.2e-4)

    assert config.iterations == 7
    assert config.lr_means == 3.2e-4
    assert config.iteration_offset == 0
    assert config.schedule_iterations is None
    assert config.means_lr_final_factor == 0.01
    assert config.opacity_logit_epsilon == 1e-4


def test_strategy_names_validate_without_importing_gsplat():
    for name in ("classic", "gsplat-default", "gsplat-mcmc"):
        validate_strategy_name(name)
    with pytest.raises(ValueError, match="unknown density strategy"):
        validate_strategy_name("magic")


def test_density_controller_clones_and_prunes():
    scene = make_synthetic_scene(n_gaussians=10, n_cameras=6, image_size=24, seed=2)
    init = make_gt_gaussians(n=30, seed=5)
    # Force some to be nearly transparent -> should get pruned.
    init.opacity[:8] = 0.001
    cfg = TrainConfig(
        iterations=45,
        rasterizer="torch",
        densify=True,
        ssim_lambda=0.0,
        density=DensityConfig(start_iter=10, every=10, grad_threshold=1e-5, max_gaussians=500),
        eval_every=45,
    )
    refined, history = Trainer(cfg).train(scene, init)
    stats = history["density_stats"]
    assert stats, "density control never ran"
    total_pruned = sum(s["pruned"] for s in stats)
    total_grown = sum(s["cloned"] + s["split"] for s in stats)
    assert total_pruned >= 8, stats
    assert total_grown > 0, stats
    assert refined.n == stats[-1]["n_after"]


def test_density_surgery_preserves_optimizer_state():
    """After clone/prune the optimizer must keep exactly one state per new param."""
    controller = DensityController(
        DensityConfig(start_iter=1, every=1, grad_threshold=0.0), 4, scene_extent=1.0
    )
    params = {
        "means": torch.randn(4, 3, requires_grad=True),
        "quats": torch.nn.functional.normalize(torch.randn(4, 4), dim=-1).requires_grad_(True),
        "log_scales": (torch.zeros(4, 3) - 3).requires_grad_(True),
        "opacity_logit": torch.zeros(4, requires_grad=True),
        "sh": torch.zeros(4, 1, 3, requires_grad=True),
    }
    opt = torch.optim.Adam(
        [{"params": [p], "lr": 1e-2, "name": n} for n, p in params.items()], eps=1e-15
    )
    # One step so optimizer state exists.
    loss = sum(p.sum() for p in params.values())
    loss.backward()
    opt.step()
    controller.grad_accum = torch.tensor([1.0, 0.0, 1.0, 0.0])
    controller.count = torch.ones(4)
    new_params = controller.step(1, params, opt, generator=torch.Generator().manual_seed(0))
    n_new = new_params["means"].shape[0]
    assert n_new == 6  # 4 kept + 2 clones (small scales -> clone path)
    for group in opt.param_groups:
        p = group["params"][0]
        assert p.shape[0] == n_new
        assert opt.state[p]["exp_avg"].shape == p.shape
    # Training continues without error.
    loss = sum(p.sum() for p in new_params.values())
    loss.backward()
    opt.step()


def test_density_controller_enforces_budget_on_dense_init():
    controller = DensityController(
        DensityConfig(start_iter=1, every=1, grad_threshold=1.0, max_gaussians=3),
        5,
        scene_extent=1.0,
    )
    params = {
        "means": torch.randn(5, 3, requires_grad=True),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5, requires_grad=True),
        "log_scales": torch.full((5, 3), -3.0, requires_grad=True),
        "opacity_logit": torch.arange(5.0, requires_grad=True),
        "sh": torch.zeros(5, 1, 3, requires_grad=True),
    }
    optimizer = torch.optim.Adam(
        [{"params": [param], "lr": 1e-2, "name": name} for name, param in params.items()]
    )
    new_params = controller.step(1, params, optimizer)
    assert new_params["means"].shape[0] == 3


def test_trainer_enforces_initial_budget_before_scheduled_density():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=3, image_size=16, seed=0)
    init = make_gt_gaussians(n=12, seed=1)
    config = TrainConfig(
        iterations=0,
        rasterizer="torch",
        density=DensityConfig(start_iter=60, every=40, max_gaussians=5),
    )
    refined, history = Trainer(config).train(scene, init)
    assert refined.n == 5
    assert history["density_stats"][0]["iteration"] == 0


def test_density_pruned_rows_are_never_split_past_budget():
    controller = DensityController(
        DensityConfig(
            start_iter=1,
            every=1,
            grad_threshold=0.0,
            split_scale_frac=0.001,
            prune_opacity=0.1,
            max_gaussians=4,
        ),
        4,
        scene_extent=1.0,
    )
    params = {
        "means": torch.randn(4, 3, requires_grad=True),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, requires_grad=True),
        "log_scales": torch.full((4, 3), -3.0, requires_grad=True),
        "opacity_logit": torch.tensor([-10.0, -10.0, 1.0, 1.0], requires_grad=True),
        "sh": torch.zeros(4, 1, 3, requires_grad=True),
    }
    optimizer = torch.optim.Adam(
        [{"params": [param], "lr": 1e-2, "name": name} for name, param in params.items()]
    )
    controller.grad_accum = torch.ones(4)
    controller.count = torch.ones(4)
    new_params = controller.step(1, params, optimizer)
    assert new_params["means"].shape[0] <= 4
    assert controller.stats[-1]["pruned"] == 2
    assert controller.stats[-1]["split"] == 2


def _selected_birth_fixture():
    n = 40
    generator = torch.Generator().manual_seed(8401)
    raw_quats = torch.randn(n, 4, generator=generator)
    params = {
        "means": torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 100.0),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(raw_quats, dim=-1)),
        "scales": torch.nn.Parameter(
            torch.log(
                torch.cat(
                    (
                        torch.full((20, 3), 0.005),
                        torch.full((20, 3), 0.02),
                    )
                )
            )
        ),
        "opacities": torch.nn.Parameter(torch.linspace(-2.0, 2.0, n)),
        "sh0": torch.nn.Parameter(torch.randn(n, 1, 3, generator=generator)),
        # Degree-zero compact training has this exact zero-width optimizer group.
        "shN": torch.nn.Parameter(torch.empty(n, 0, 3)),
    }
    optimizers = {
        name: torch.optim.Adam(
            [
                {
                    "params": [parameter],
                    "lr": 1e-3 * (index + 1),
                    "name": name,
                    "research_marker": ("selected-birth", index),
                }
            ],
            betas=(0.8, 0.95),
            eps=1e-12,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False,
            foreach=False,
            fused=False,
        )
        for index, (name, parameter) in enumerate(params.items())
    }
    sum(
        (index + 1) * parameter.square().sum() for index, parameter in enumerate(params.values())
    ).backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # Match the preregistered surgery boundary without spending 35 fixture updates.
    for name, parameter in params.items():
        optimizers[name].state[parameter]["step"].fill_(35)
    return params, optimizers


def _test_tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def test_selected_birth_surgery_exact_mixed_16_16_order_draws_and_adam_state():
    params, optimizers = _selected_birth_fixture()
    small = (15, 0, 19, 3, 18, 4, 17, 5, 16, 6, 14, 7, 13, 8, 12, 9)
    large = (39, 20, 35, 24, 38, 21, 34, 25, 37, 22, 33, 26, 36, 23, 32, 27)
    selected = tuple(row for pair in zip(small, large, strict=True) for row in pair)
    old_params = {name: parameter.detach().clone() for name, parameter in params.items()}
    old_parameters = dict(params)
    old_states = {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizers[name].state[params[name]].items()
        }
        for name in params
    }
    old_group_fields = {
        name: {
            key: value for key, value in optimizers[name].param_groups[0].items() if key != "params"
        }
        for name in params
    }

    # Development-only: official experiment roots must never reach a generator in tests.
    split_seed = 63892
    expected_generator = torch.Generator().manual_seed(split_seed)
    expected_state_before = expected_generator.get_state().clone()
    expected_child0 = torch.randn((16, 3), generator=expected_generator)
    expected_child1 = torch.randn((16, 3), generator=expected_generator)
    expected_state_after = expected_generator.get_state().clone()
    split_generator = torch.Generator().manual_seed(split_seed)
    torch.manual_seed(123456)
    global_state_before = torch.random.get_rng_state().clone()

    new_params, receipt = apply_selected_birth_surgery(
        params,
        optimizers,
        selected,
        scene_extent=1.0,
        generator=split_generator,
        max_gaussians=72,
    )

    assert torch.equal(torch.random.get_rng_state(), global_state_before)
    assert receipt.n_before == 40
    assert receipt.n_after == 72
    assert receipt.net_growth == len(selected) == 32
    assert receipt.selected_parent_rows == selected
    assert receipt.clone_parent_rows == small
    assert receipt.split_parent_rows == large
    survivors = tuple(row for row in range(40) if row not in set(large))
    assert receipt.survivor_old_rows == survivors
    assert receipt.removed_parent_rows == large
    assert receipt.new_row_to_old_row == survivors + small + large + large
    assert receipt.clone_new_rows == tuple(range(24, 40))
    assert receipt.split_child0_new_rows == tuple(range(40, 56))
    assert receipt.split_child1_new_rows == tuple(range(56, 72))
    assert tuple(item.new_row for item in receipt.newborns) == tuple(range(24, 72))
    assert tuple(item.parent_row for item in receipt.newborns) == small + large + large
    assert tuple(item.operator for item in receipt.newborns) == (("clone",) * 16 + ("split",) * 32)
    assert tuple(item.child_ordinal for item in receipt.newborns) == ((0,) * 32 + (1,) * 16)
    assert all(
        receipt.old_row_to_new_row[old_row] == new_row for new_row, old_row in enumerate(survivors)
    )
    assert all(receipt.old_row_to_new_row[row] == -1 for row in large)

    receipt_child0 = torch.tensor(receipt.raw_split_child0_standard_normals)
    receipt_child1 = torch.tensor(receipt.raw_split_child1_standard_normals)
    assert torch.equal(receipt_child0, expected_child0)
    assert torch.equal(receipt_child1, expected_child1)
    assert receipt.raw_split_child0_sha256 == _test_tensor_sha256(expected_child0)
    assert receipt.raw_split_child1_sha256 == _test_tensor_sha256(expected_child1)
    assert receipt.generator_state_before_sha256 == _test_tensor_sha256(expected_state_before)
    assert receipt.generator_state_after_sha256 == _test_tensor_sha256(expected_state_after)
    assert torch.equal(split_generator.get_state(), expected_state_after)
    with pytest.raises(AttributeError):
        receipt.n_after = 0

    survivor_index = torch.tensor(survivors)
    small_index = torch.tensor(small)
    large_index = torch.tensor(large)
    for name in params:
        assert torch.equal(new_params[name][:24], old_params[name][survivor_index])
        assert torch.equal(new_params[name][24:40], old_params[name][small_index])
    split_rotation = quat_to_rotmat(old_params["quats"][large_index])
    native_scale = old_params["scales"][large_index].exp()
    expected_means0 = (
        old_params["means"][large_index]
        + (split_rotation @ (expected_child0 * native_scale)[..., None])[..., 0]
    )
    expected_means1 = (
        old_params["means"][large_index]
        + (split_rotation @ (expected_child1 * native_scale)[..., None])[..., 0]
    )
    assert torch.equal(new_params["means"][40:56], expected_means0)
    assert torch.equal(new_params["means"][56:72], expected_means1)
    expected_split_scales = old_params["scales"][large_index] - torch.log(
        old_params["scales"].new_tensor(1.6)
    )
    assert torch.equal(new_params["scales"][40:56], expected_split_scales)
    assert torch.equal(new_params["scales"][56:72], expected_split_scales)
    source_opacity = torch.sigmoid(old_params["opacities"][large_index])
    child_opacity = 1.0 - torch.sqrt(1.0 - source_opacity.clamp(max=1.0 - 1e-6))
    expected_opacity_logits = torch.logit(child_opacity.clamp_min(1e-6))
    assert torch.equal(new_params["opacities"][40:56], expected_opacity_logits)
    assert torch.equal(new_params["opacities"][56:72], expected_opacity_logits)
    for name in ("quats", "sh0", "shN"):
        assert torch.equal(new_params[name][40:56], old_params[name][large_index])
        assert torch.equal(new_params[name][56:72], old_params[name][large_index])
    assert new_params["shN"].shape == (72, 0, 3)

    for name, optimizer in optimizers.items():
        new_parameter = new_params[name]
        state = optimizer.state[new_parameter]
        assert optimizer.param_groups[0]["params"] == [new_parameter]
        assert old_parameters[name] not in optimizer.state
        assert {
            key: value for key, value in optimizer.param_groups[0].items() if key != "params"
        } == old_group_fields[name]
        assert torch.equal(state["step"], old_states[name]["step"])
        for moment_name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                state[moment_name][:24],
                old_states[name][moment_name][survivor_index],
            )
            assert torch.equal(
                state[moment_name][24:],
                torch.zeros_like(state[moment_name][24:]),
            )
            assert state[moment_name].shape == new_parameter.shape


@pytest.mark.parametrize(
    ("parent_rows", "error"),
    [
        ([], ValueError),
        ([1, 1], ValueError),
        ([40], IndexError),
        ([-1], IndexError),
        ([True], TypeError),
        ([1.0], TypeError),
        (torch.tensor([[1]], dtype=torch.long), ValueError),
        (torch.tensor([1.0]), TypeError),
    ],
)
def test_selected_birth_surgery_rejects_invalid_parent_rows_without_mutation(parent_rows, error):
    params, optimizers = _selected_birth_fixture()
    old_parameters = dict(params)
    split_generator = torch.Generator().manual_seed(7)
    generator_state = split_generator.get_state().clone()
    with pytest.raises(error):
        apply_selected_birth_surgery(
            params,
            optimizers,
            parent_rows,
            scene_extent=1.0,
            generator=split_generator,
            max_gaussians=100,
        )
    assert torch.equal(split_generator.get_state(), generator_state)
    assert all(
        optimizers[name].param_groups[0]["params"][0] is old_parameters[name] for name in params
    )


def test_selected_birth_surgery_rejects_cap_before_rng_or_optimizer_mutation():
    params, optimizers = _selected_birth_fixture()
    old_parameters = dict(params)
    old_state_keys = {name: tuple(optimizers[name].state.keys()) for name in params}
    split_generator = torch.Generator().manual_seed(8)
    generator_state = split_generator.get_state().clone()
    with pytest.raises(ValueError, match="exceeds max_gaussians"):
        apply_selected_birth_surgery(
            params,
            optimizers,
            [0],
            scene_extent=1.0,
            generator=split_generator,
            max_gaussians=40,
        )
    assert torch.equal(split_generator.get_state(), generator_state)
    assert all(
        optimizers[name].param_groups[0]["params"][0] is old_parameters[name] for name in params
    )
    assert all(tuple(optimizers[name].state.keys()) == old_state_keys[name] for name in params)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_selected_birth_surgery_uses_matching_cuda_generator_without_gsplat():
    device = torch.device("cuda:0")
    params = {
        "means": torch.nn.Parameter(torch.zeros(2, 3, device=device)),
        "quats": torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                device=device,
            )
        ),
        "scales": torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [[0.005, 0.005, 0.005], [0.02, 0.02, 0.02]],
                    device=device,
                )
            )
        ),
        "opacities": torch.nn.Parameter(torch.zeros(2, device=device)),
        "sh0": torch.nn.Parameter(torch.zeros(2, 1, 3, device=device)),
        "shN": torch.nn.Parameter(torch.empty(2, 0, 3, device=device)),
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [parameter], "name": name}], lr=1e-3)
        for name, parameter in params.items()
    }
    sum(parameter.square().sum() for parameter in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()

    with pytest.raises(ValueError, match="generator device"):
        apply_selected_birth_surgery(
            params,
            optimizers,
            [0, 1],
            scene_extent=1.0,
            generator=torch.Generator().manual_seed(9),
            max_gaussians=4,
        )
    new_params, receipt = apply_selected_birth_surgery(
        params,
        optimizers,
        [0, 1],
        scene_extent=1.0,
        generator=torch.Generator(device=device).manual_seed(9),
        max_gaussians=4,
    )
    assert new_params["means"].device == device
    assert receipt.clone_parent_rows == (0,)
    assert receipt.split_parent_rows == (1,)
    assert receipt.raw_split_shape == (1, 3)
    assert receipt.n_after == 4


def test_budget_enforcement_preserves_per_parameter_adam_state():
    params = {
        "means": torch.nn.Parameter(torch.randn(6, 3)),
        "quats": torch.nn.Parameter(torch.randn(6, 4)),
        "scales": torch.nn.Parameter(torch.full((6, 3), -3.0)),
        "opacities": torch.nn.Parameter(torch.arange(6.0)),
        "sh0": torch.nn.Parameter(torch.randn(6, 1, 3)),
        "shN": torch.nn.Parameter(torch.randn(6, 15, 3)),
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [value], "name": name}], lr=1e-2)
        for name, value in params.items()
    }
    sum(value.sum() for value in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
    removed = enforce_budget(params, optimizers, max_gaussians=4)
    assert removed == 2
    for name, parameter in params.items():
        assert parameter.shape[0] == 4
        state = optimizers[name].state[parameter]
        assert state["exp_avg"].shape == parameter.shape


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("strategy", ["gsplat-default", "gsplat-mcmc"])
def test_gsplat_strategies_train_and_grow(strategy):
    pytest.importorskip("gsplat")
    scene = make_synthetic_scene(n_gaussians=12, n_cameras=4, image_size=16, seed=9)
    init = make_gt_gaussians(n=40, seed=10)
    config = TrainConfig(
        iterations=18,
        rasterizer="gsplat",
        device="cuda",
        density_strategy=strategy,
        density=DensityConfig(
            start_iter=1,
            stop_iter=16,
            every=3,
            grad_threshold=0.0,
            absgrad=strategy == "gsplat-default",
            prune_scale_frac=10.0,
            max_gaussians=64,
            opacity_reset_every=0,
        ),
        eval_every=18,
        ssim_lambda=0.0,
    )
    refined, history = Trainer(config).train(scene, init)
    assert 40 < refined.n <= 64
    assert history["density_stats"]
    assert history["active_sh_degree"][-1][1] == 3
    assert torch.isfinite(refined.means).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_geometric_arena_gsplat_default_trains_and_grows():
    pytest.importorskip("gsplat")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    scene = make_synthetic_scene(n_gaussians=12, n_cameras=4, image_size=16, seed=9)
    init = make_gt_gaussians(n=40, seed=10)
    config = TrainConfig(
        iterations=18,
        rasterizer="gsplat",
        device="cuda",
        density_strategy="gsplat-default",
        gaussian_storage_policy="geometric",
        profile_density_events=True,
        density=DensityConfig(
            start_iter=1,
            stop_iter=16,
            every=3,
            grad_threshold=0.0,
            absgrad=True,
            prune_scale_frac=10.0,
            max_gaussians=64,
            opacity_reset_every=0,
        ),
        eval_every=18,
        ssim_lambda=0.0,
    )
    refined, history = Trainer(config).train(scene, init)
    assert 40 < refined.n <= 64
    assert history["gaussian_storage_policy"] == "geometric"
    assert history["storage_diagnostics"]["capacity"] >= refined.n
    assert history["density_stats"]
    assert all(item["event_seconds"] >= 0.0 for item in history["density_stats"])
    assert torch.isfinite(refined.means).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("strategy", ["gsplat-default", "gsplat-mcmc"])
def test_gsplat_strategies_enforce_budget_before_initializing_state(strategy):
    pytest.importorskip("gsplat")
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=3, image_size=16, seed=11)
    init = make_gt_gaussians(n=40, seed=12)
    config = TrainConfig(
        iterations=0,
        rasterizer="gsplat",
        device="cuda",
        density_strategy=strategy,
        density=DensityConfig(max_gaussians=20),
    )
    refined, _ = Trainer(config).train(scene, init)
    assert refined.n == 20
