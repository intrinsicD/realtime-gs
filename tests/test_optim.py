"""Stage 3: trainer and density control."""

import pytest
import torch

from rtgs.data.synthetic import make_gt_gaussians, make_synthetic_scene
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.strategies import enforce_budget, validate_strategy_name
from rtgs.optim.trainer import TrainConfig, Trainer, _resolve_sh_interval


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


def test_sh_schedule_validation_and_long_run_default():
    assert _resolve_sh_interval(TrainConfig(iterations=30_000, target_sh_degree=3)) == 1_000
    assert _resolve_sh_interval(TrainConfig(iterations=1_000, target_sh_degree=3)) == 250
    with pytest.raises(ValueError, match="positive"):
        _resolve_sh_interval(TrainConfig(sh_degree_interval=0))


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
