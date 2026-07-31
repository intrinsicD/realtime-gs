"""Step-conditional density gating (protocol 20260731_coarse_to_fine_density): the
default keeps the established rejection of non-unit step controls with density control,
the opt-in runs density hooks only at unit-scale steps, and screen-gradient statistics
reset at the coarse-to-full transition so low-resolution statistics never drive
full-resolution topology decisions."""

from __future__ import annotations

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer, TrainStepControl


def _controls(coarse: int, total: int) -> list[TrainStepControl]:
    return [
        TrainStepControl(render_downscale=2, loss_downscale=2)
        if index < coarse
        else TrainStepControl()
        for index in range(total)
    ]


def test_classic_reset_statistics_zeroes_accumulators():
    controller = DensityController(DensityConfig(), 4, 1.0, device=torch.device("cpu"))
    controller.grad_accum += 3.0
    controller.count += 2.0
    controller.reset_statistics()
    assert torch.equal(controller.grad_accum, torch.zeros(4))
    assert torch.equal(controller.count, torch.zeros(4))


def test_default_still_rejects_non_unit_controls_with_density():
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=2, image_size=16, seed=2)
    cfg = TrainConfig(iterations=4, rasterizer="torch", densify=True)
    with pytest.raises(ValueError, match="density control to be disabled"):
        Trainer(cfg).train(scene, scene.gt_gaussians.detach(), step_controls=_controls(2, 4))


def test_conditional_density_defers_topology_to_full_resolution():
    scene = make_synthetic_scene(n_gaussians=15, n_cameras=6, image_size=24, seed=1)
    init = scene.gt_gaussians.detach()
    init.means += 0.03
    coarse = 10
    total = 20
    cfg = TrainConfig(
        iterations=total,
        rasterizer="torch",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            every=4,
            start_iter=1,
            stop_iter=1000,
            grad_threshold=1e-9,
            max_gaussians=64,
        ),
        conditional_density=True,
        eval_every=total,
        ssim_lambda=0.0,
    )
    refined, history = Trainer(cfg).train(scene, init, step_controls=_controls(coarse, total))
    assert history["conditional_density"] is True
    stats = history["density_stats"]
    assert stats, "full-resolution phase must run density events"
    assert all(record["iteration"] > coarse for record in stats)
    counts = dict(history["n_gaussians"])
    assert counts[total] > 15, "densification must grow rows after the transition"


def test_conditional_density_unit_controls_match_plain_run():
    scene = make_synthetic_scene(n_gaussians=8, n_cameras=3, image_size=16, seed=4)
    init = scene.gt_gaussians.detach()
    init.means += 0.02
    base = dict(
        iterations=6,
        rasterizer="torch",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            every=3, start_iter=1, stop_iter=1000, grad_threshold=1e-9, max_gaussians=32
        ),
        eval_every=6,
        ssim_lambda=0.0,
    )
    plain, plain_history = Trainer(TrainConfig(**base)).train(scene, init)
    gated, gated_history = Trainer(TrainConfig(**base, conditional_density=True)).train(
        scene, init, step_controls=_controls(0, 6)
    )
    assert plain_history["loss"] == gated_history["loss"]
    assert torch.equal(plain.means, gated.means)


def test_transition_resets_statistics_and_coarse_never_accumulates(monkeypatch):
    from rtgs.optim import density as density_module

    calls = {"reset": 0, "accumulate": 0}
    original_reset = density_module.DensityController.reset_statistics
    original_accumulate = density_module.DensityController.accumulate

    def counting_reset(self):
        calls["reset"] += 1
        return original_reset(self)

    def counting_accumulate(self, out, width, height):
        calls["accumulate"] += 1
        return original_accumulate(self, out, width, height)

    monkeypatch.setattr(density_module.DensityController, "reset_statistics", counting_reset)
    monkeypatch.setattr(density_module.DensityController, "accumulate", counting_accumulate)

    scene = make_synthetic_scene(n_gaussians=8, n_cameras=3, image_size=16, seed=8)
    coarse = 3
    total = 5
    cfg = TrainConfig(
        iterations=total,
        rasterizer="torch",
        densify=True,
        density_strategy="classic",
        # start_iter beyond the run: no event fires, isolating accumulation and reset.
        density=DensityConfig(every=100, start_iter=100, stop_iter=1000, max_gaussians=32),
        conditional_density=True,
        eval_every=total,
        ssim_lambda=0.0,
    )
    Trainer(cfg).train(scene, scene.gt_gaussians.detach(), step_controls=_controls(coarse, total))
    assert calls["reset"] == 1, "exactly one coarse-to-full transition reset"
    assert calls["accumulate"] == total - coarse, "coarse steps must never accumulate"
