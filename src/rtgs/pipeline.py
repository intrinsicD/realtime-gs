"""End-to-end orchestration: fit 2D gaussians -> lift to 3D -> refine (stages 1-3).

`run_pipeline` executes the full chain with timings and per-stage metrics;
`compare_lifters` runs several lifting variants on the same 2D fits (the core
experiment of this repository).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift import get_lifter
from rtgs.optim.trainer import TrainConfig, Trainer


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    fit: FitConfig = field(default_factory=FitConfig)
    lifter: str = "depth"
    lifter_kwargs: dict = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    refine: bool = True
    seed: int = 0


@dataclass
class PipelineResult:
    """Everything produced by one pipeline run."""

    gaussians_init: Gaussians3D
    gaussians: Gaussians3D  # refined (== init when refine=False)
    metrics: dict
    timings: dict
    fit_histories: list[dict] = field(default_factory=list)
    train_history: dict = field(default_factory=dict)


def run_pipeline(
    scene: SceneData,
    config: PipelineConfig | None = None,
    gaussians2d: list[Gaussians2D] | None = None,
) -> PipelineResult:
    """Run stages 1-3 on a scene. Pass precomputed ``gaussians2d`` to skip stage 1."""
    config = config or PipelineConfig()
    scene.validate()
    timings: dict = {}
    metrics: dict = {}

    t0 = time.perf_counter()
    fit_histories: list[dict] = []
    if gaussians2d is None:
        gaussians2d, fit_histories = fit_views(scene.images, config.fit, seed=config.seed)
        metrics["fit_psnr_mean"] = sum(h["final_psnr"] for h in fit_histories) / len(fit_histories)
    timings["fit"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    lifter = get_lifter(config.lifter, **config.lifter_kwargs)
    init = lifter.lift(gaussians2d, scene)
    timings["lift"] = time.perf_counter() - t1
    metrics["init_n_gaussians"] = init.n
    metrics["init_psnr"] = Trainer.evaluate(scene, init)

    refined = init
    train_history: dict = {}
    if config.refine:
        t2 = time.perf_counter()
        refined, train_history = Trainer(config.train).train(scene, init)
        timings["refine"] = time.perf_counter() - t2
        metrics["final_n_gaussians"] = refined.n
        metrics["final_psnr"] = Trainer.evaluate(scene, refined)

    timings["total"] = time.perf_counter() - t0
    return PipelineResult(
        gaussians_init=init,
        gaussians=refined,
        metrics=metrics,
        timings=timings,
        fit_histories=fit_histories,
        train_history=train_history,
    )


def compare_lifters(
    scene: SceneData,
    lifters: dict[str, dict] | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, PipelineResult]:
    """Run several lifting variants against the same stage-1 fits.

    ``lifters`` maps variant name -> lifter kwargs (default: all registered variants
    with defaults). Stage 1 runs once and is shared.
    """
    from rtgs.lift import lifter_names

    config = config or PipelineConfig()
    if lifters is None:
        lifters = {name: {} for name in lifter_names()}

    gaussians2d, _ = fit_views(scene.images, config.fit, seed=config.seed)
    results: dict[str, PipelineResult] = {}
    for name, kwargs in lifters.items():
        cfg = PipelineConfig(
            fit=config.fit,
            lifter=name,
            lifter_kwargs=kwargs,
            train=config.train,
            refine=config.refine,
            seed=config.seed,
        )
        results[name] = run_pipeline(scene, cfg, gaussians2d=gaussians2d)
    return results
