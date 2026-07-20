"""End-to-end orchestration: fit 2D gaussians -> lift to 3D -> refine (stages 1-3).

`run_pipeline` executes the full chain with timings and per-stage metrics;
`compare_lifters` runs several lifting variants on the same 2D fits (the core
experiment of this repository).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.field_inputs import SceneFits
from rtgs.data.scene import SceneData
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift import get_lifter
from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter, FieldLiftResult
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    fit: FitConfig = field(default_factory=FitConfig)
    lifter: str = "depth"
    lifter_kwargs: dict = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    refine: bool = True
    device: str = "auto"
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


def run_field_pipeline(
    fits: SceneFits,
    config: FieldLiftConfig | None = None,
) -> FieldLiftResult:
    """Run image-free field lifting from frozen compact observations and cameras."""

    return FieldLifter(config).fit(fits)


def run_pipeline(
    scene: SceneData,
    config: PipelineConfig | None = None,
    gaussians2d: list[Gaussians2D] | None = None,
) -> PipelineResult:
    """Run stages 1-3 on a scene. Pass precomputed ``gaussians2d`` to skip stage 1."""
    config = config or PipelineConfig()
    scene.validate()
    device = _resolve_device(config.device)
    scene = scene.to(device)
    train_indices = scene.training_views
    train_scene = scene.subset(train_indices)
    if gaussians2d is not None:
        if len(gaussians2d) == scene.n_views:
            gaussians2d = [gaussians2d[i] for i in train_indices]
        elif len(gaussians2d) != train_scene.n_views:
            raise ValueError(
                "precomputed 2D fits must cover either every scene view or only training views"
            )
        gaussians2d = [gaussian.to(device) for gaussian in gaussians2d]
    timings: dict = {}
    metrics: dict = {}
    evaluation_renderer = get_rasterizer(
        config.train.rasterizer,
        device=device,
        packed=config.train.packed,
        antialiased=config.train.antialiased,
        sh_color_activation=config.train.sh_color_activation,
        sh_smu1_mu=config.train.sh_smu1_mu,
        kernel_support_mode=config.train.kernel_support_mode,
        visibility_margin_sigma=config.train.visibility_margin_sigma,
    )

    t0 = time.perf_counter()
    fit_histories: list[dict] = []
    if gaussians2d is None:
        gaussians2d, fit_histories = fit_views(
            train_scene.images,
            config.fit,
            seed=config.seed,
            masks=train_scene.masks,
        )
        metrics["fit_psnr_mean"] = sum(h["final_psnr"] for h in fit_histories) / len(fit_histories)
    _sync(device)
    timings["fit"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    lifter = get_lifter(config.lifter, **config.lifter_kwargs)
    init = lifter.lift(gaussians2d, train_scene)
    _sync(device)
    timings["lift"] = time.perf_counter() - t1
    metrics["init_n_gaussians"] = init.n
    _record_evaluation_metrics(metrics, "init", scene, init, evaluation_renderer)

    refined = init
    train_history: dict = {}
    if config.refine:
        t2 = time.perf_counter()
        train_config = replace(config.train, device=str(device))
        refined, train_history = Trainer(train_config).train(scene, init)
        _sync(device)
        timings["refine"] = time.perf_counter() - t2
        metrics["final_n_gaussians"] = refined.n
        _record_evaluation_metrics(metrics, "final", scene, refined, evaluation_renderer)

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

    device = _resolve_device(config.device)
    scene = scene.to(device)
    train_scene = scene.subset(scene.training_views)
    fit_started = time.perf_counter()
    gaussians2d, fit_histories = fit_views(
        train_scene.images, config.fit, seed=config.seed, masks=train_scene.masks
    )
    _sync(device)
    shared_fit_seconds = time.perf_counter() - fit_started
    mean_fit_psnr = sum(history["final_psnr"] for history in fit_histories) / len(fit_histories)
    results: dict[str, PipelineResult] = {}
    for name, kwargs in lifters.items():
        cfg = PipelineConfig(
            fit=config.fit,
            lifter=name,
            lifter_kwargs=kwargs,
            train=config.train,
            refine=config.refine,
            device=str(device),
            seed=config.seed,
        )
        results[name] = run_pipeline(scene, cfg, gaussians2d=gaussians2d)
        results[name].fit_histories = fit_histories
        results[name].timings["fit"] = shared_fit_seconds
        results[name].timings["total"] += shared_fit_seconds
        results[name].metrics["fit_psnr_mean"] = mean_fit_psnr
    return results


def _sync(device: torch.device) -> None:
    """Synchronize accelerators before recording wall-clock timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _record_evaluation_metrics(
    output: dict, prefix: str, scene: SceneData, gaussians: Gaussians3D, renderer=None
) -> None:
    """Store strict test metrics plus diagnostic train metrics with unambiguous names."""
    test_indices = scene.testing_views
    primary_indices = test_indices or scene.training_views
    split = "test" if test_indices else "train"
    primary = Trainer.evaluate_metrics(scene, gaussians, renderer, indices=primary_indices)
    for name, value in primary.items():
        output[f"{prefix}_{name}_{split}"] = value
    headline = primary["psnr_fg"] if "psnr_fg" in primary else primary["psnr"]
    output[f"{prefix}_psnr"] = headline
    if test_indices:
        train = Trainer.evaluate_metrics(scene, gaussians, renderer, indices=scene.training_views)
        for name, value in train.items():
            output[f"{prefix}_{name}_train"] = value


def _resolve_device(requested: str) -> torch.device:
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else ("cpu" if requested == "auto" else requested)
    )
