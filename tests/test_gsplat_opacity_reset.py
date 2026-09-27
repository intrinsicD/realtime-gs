"""RTGS-027: gsplat 1.5.3 DefaultStrategy never resets opacity; opt-in intended reset."""

from __future__ import annotations

import inspect

import pytest
import torch

from rtgs.optim.density import DensityConfig
from rtgs.optim.strategies import (
    IntendedOpacityReset,
    chain_parameter_callbacks,
    upstream_default_reset_fires,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_upstream_reset_condition_never_fires() -> None:
    assert not any(upstream_default_reset_fires(step, 3000) for step in range(0, 30001))
    assert not any(upstream_default_reset_fires(step, 7) for step in range(0, 200))


def test_installed_gsplat_carries_the_defective_expression() -> None:
    gsplat = pytest.importorskip("gsplat")
    default = pytest.importorskip("gsplat.strategy.default")
    source = inspect.getsource(default.DefaultStrategy.step_post_backward)
    assert "step % self.reset_every == 0 & step > 0" in source, (
        f"gsplat {gsplat.__version__} no longer carries the inert reset expression; "
        "re-evaluate RTGS-027 before relying on IntendedOpacityReset"
    )


def _params(n: int = 4):
    params = {"opacities": torch.nn.Parameter(torch.linspace(-3.0, 3.0, n))}
    optimizers = {"opacities": torch.optim.Adam([params["opacities"]], lr=0.1)}
    params["opacities"].grad = torch.ones(n)
    optimizers["opacities"].step()
    return params, optimizers


def test_intended_reset_clamps_and_zeroes_moments_only_when_due() -> None:
    reset = IntendedOpacityReset(reset_every=10, stop_iter=25, value=0.01)
    # Completed steps 11 and 21 are gsplat iterations 10 and 20.
    assert [s for s in range(0, 40) if reset.due(s)] == [11, 21]
    params, optimizers = _params()
    before = params["opacities"].detach().clone()
    reset(params, optimizers, 5)
    assert torch.equal(params["opacities"].detach(), before) and reset.events == []
    reset(params, optimizers, 11)
    cap = torch.logit(torch.tensor(0.01))
    assert float(params["opacities"].detach().max()) <= float(cap) + 1e-7
    below = before <= cap
    assert torch.equal(params["opacities"].detach()[below], before[below])
    state = optimizers["opacities"].state[params["opacities"]]
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0
    assert float(state["step"]) == 1.0  # Adam's scalar step survives, as in gsplat reset_opa
    assert reset.events == [
        {"step": 11, "gsplat_iteration": 10, "clamped": int((before > cap).sum()), "n": 4}
    ]


def test_from_density_uses_twice_the_prune_opacity() -> None:
    config = DensityConfig(opacity_reset_every=3000, stop_iter=6000, prune_opacity=0.005)
    reset = IntendedOpacityReset.from_density(config)
    assert (reset.reset_every, reset.stop_iter, reset.value) == (3000, 6000, 0.01)
    assert IntendedOpacityReset.from_density(DensityConfig(opacity_reset_every=0)) is None
    with pytest.raises(ValueError):
        IntendedOpacityReset(0, 10, 0.01)


def test_chain_runs_callbacks_in_order_and_skips_none() -> None:
    calls = []
    assert chain_parameter_callbacks(None, None) is None
    chained = chain_parameter_callbacks(
        lambda p, o, s: calls.append(("a", s)), None, lambda p, o, s: calls.append(("b", s))
    )
    chained({}, {}, 3)
    assert calls == [("a", 3), ("b", 3)]


@requires_cuda
def test_intended_reset_runs_inside_gsplat_default_training() -> None:
    pytest.importorskip("gsplat")
    from rtgs.data.synthetic import make_synthetic_scene
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = make_synthetic_scene(n_gaussians=12, n_cameras=3, image_size=32, seed=4)
    density = DensityConfig(
        start_iter=5, stop_iter=35, every=5, opacity_reset_every=20, max_gaussians=500
    )
    config = TrainConfig(
        iterations=40,
        rasterizer="gsplat",
        device="cuda:0",
        densify=True,
        density_strategy="gsplat-default",
        density=density,
        eval_every=40,
        ssim_lambda=0.0,
        use_masks=False,
    )
    reset = IntendedOpacityReset.from_density(density)
    seen = {}

    def observe(params, optimizers, step):
        if step == 21:
            seen["max"] = float(torch.sigmoid(params["opacities"]).max())

    init = scene.gt_gaussians.detach()
    Trainer(config).train(
        scene, init, parameter_step_callback=chain_parameter_callbacks(reset, observe)
    )
    assert [event["gsplat_iteration"] for event in reset.events] == [20]
    assert seen["max"] <= 2.0 * density.prune_opacity + 1e-6


def test_intended_reset_through_the_cpu_trainer_seam() -> None:
    from rtgs.data.synthetic import make_synthetic_scene
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = make_synthetic_scene(n_gaussians=8, n_cameras=3, image_size=16, seed=4)
    config = TrainConfig(
        iterations=12, rasterizer="torch", densify=False, eval_every=12, ssim_lambda=0.0
    )
    reset = IntendedOpacityReset(reset_every=5, stop_iter=11, value=0.05)
    refined, _ = Trainer(config).train(
        scene, scene.gt_gaussians.detach(), parameter_step_callback=reset
    )
    assert [event["gsplat_iteration"] for event in reset.events] == [5, 10]
    assert all(event["n"] == scene.gt_gaussians.n for event in reset.events)
    assert refined.n == scene.gt_gaussians.n


def test_no_reset_on_the_final_iteration_of_a_reset_multiple_run() -> None:
    from rtgs.data.synthetic import make_synthetic_scene
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = make_synthetic_scene(n_gaussians=8, n_cameras=3, image_size=16, seed=4)
    config = TrainConfig(
        iterations=10, rasterizer="torch", densify=False, eval_every=10, ssim_lambda=0.0
    )
    reset = IntendedOpacityReset(reset_every=5, stop_iter=10_000_000, value=0.05)
    refined, _ = Trainer(config).train(
        scene, scene.gt_gaussians.detach(), parameter_step_callback=reset
    )
    assert [event["gsplat_iteration"] for event in reset.events] == [5]
    assert float(refined.opacity.max()) > 0.05
