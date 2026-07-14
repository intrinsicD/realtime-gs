"""Stage 3: trainer and density control."""

import torch

from rtgs.data.synthetic import make_gt_gaussians, make_synthetic_scene
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer


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
