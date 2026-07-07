"""End-to-end pipeline: all lifting variants must beat the random baseline and refine."""

import pytest
import torch

from rtgs.image2gs.fit import FitConfig
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig
from rtgs.pipeline import PipelineConfig, compare_lifters, run_pipeline


def _fast_config(lifter: str, refine: bool = True) -> PipelineConfig:
    return PipelineConfig(
        fit=FitConfig(n_gaussians=120, iterations=100, log_every=50),
        lifter=lifter,
        lifter_kwargs={"iterations": 40, "rasterizer": "torch"} if lifter == "gradient" else {},
        train=TrainConfig(
            iterations=40,
            rasterizer="torch",
            ssim_lambda=0.0,
            density=DensityConfig(start_iter=15, every=15),
            eval_every=40,
        ),
        refine=refine,
        seed=0,
    )


@pytest.mark.parametrize("lifter", ["depth", "gradient", "carve"])
def test_pipeline_end_to_end(tiny_scene, tiny_fits, lifter):
    g2ds, _ = tiny_fits
    result = run_pipeline(tiny_scene, _fast_config(lifter), gaussians2d=g2ds)
    assert result.metrics["init_n_gaussians"] > 50
    assert result.metrics["init_psnr"] > 10.0, result.metrics
    # Refinement must not make things worse (allowing small jitter from short runs).
    assert result.metrics["final_psnr"] > result.metrics["init_psnr"] - 0.5, result.metrics
    assert result.timings["total"] > 0


def test_variants_beat_random_baseline(tiny_scene, tiny_fits):
    """The core research claim at sanity scale: every variant inits better than random."""
    g2ds, _ = tiny_fits
    cfg = _fast_config("depth", refine=False)
    results = compare_lifters(
        tiny_scene,
        lifters={
            "depth": {},
            "gradient": {"iterations": 40, "rasterizer": "torch"},
            "carve": {},
            "random": {"n": 800},
        },
        config=cfg,
    )
    random_psnr = results["random"].metrics["init_psnr"]
    for name in ("depth", "gradient", "carve"):
        assert results[name].metrics["init_psnr"] > random_psnr, (
            name,
            results[name].metrics["init_psnr"],
            random_psnr,
        )


def test_pipeline_runs_stage1_when_no_fits(tiny_scene):
    cfg = _fast_config("depth")
    cfg.fit = FitConfig(n_gaussians=60, iterations=30, log_every=15)
    cfg.train.iterations = 10
    result = run_pipeline(tiny_scene, cfg)
    assert "fit_psnr_mean" in result.metrics
    assert result.timings["fit"] > 0
    assert len(result.fit_histories) == tiny_scene.n_views


def test_pipeline_determinism(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    cfg = _fast_config("depth", refine=False)
    r1 = run_pipeline(tiny_scene, cfg, gaussians2d=g2ds)
    r2 = run_pipeline(tiny_scene, cfg, gaussians2d=g2ds)
    assert torch.allclose(r1.gaussians_init.means, r2.gaussians_init.means)
    assert r1.metrics["init_psnr"] == r2.metrics["init_psnr"]
