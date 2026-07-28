"""End-to-end pipeline: all lifting variants must beat the random baseline and refine."""

import inspect
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from rtgs.image2gs.fit import FitConfig
from rtgs.lift.beam_fusion import BeamFusionConfig
from rtgs.lift.carrier_refinement import CarrierRepairConfig
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig
from rtgs.pipeline import (
    CarrierPipelineConfig,
    PipelineConfig,
    compare_lifters,
    run_carrier_pipeline,
    run_pipeline,
)


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


def test_held_out_views_cannot_leak_into_initialization(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    split = replace(tiny_scene, train_indices=list(range(6)), test_indices=[6, 7])
    changed_images = list(split.images)
    changed_images[6] = torch.rand_like(changed_images[6])
    changed_images[7] = torch.rand_like(changed_images[7])
    changed = replace(split, images=changed_images)
    cfg = _fast_config("depth", refine=False)
    original_result = run_pipeline(split, cfg, gaussians2d=g2ds)
    changed_result = run_pipeline(changed, cfg, gaussians2d=g2ds)
    assert torch.equal(original_result.gaussians_init.means, changed_result.gaussians_init.means)
    assert "init_psnr_test" in original_result.metrics
    assert original_result.metrics["init_psnr_test"] != changed_result.metrics["init_psnr_test"]


def test_stage1_fits_training_views_only(tiny_scene):
    split = replace(tiny_scene, train_indices=[0, 1], test_indices=[2])
    cfg = _fast_config("depth", refine=False)
    cfg.fit = FitConfig(n_gaussians=20, iterations=1, log_every=1)
    result = run_pipeline(split, cfg)
    assert len(result.fit_histories) == 2


def test_carrier_pipeline_api_has_no_scene_or_image_input() -> None:
    assert list(inspect.signature(run_carrier_pipeline).parameters) == [
        "inputs",
        "config",
    ]


def test_carrier_pipeline_rejects_unbounded_or_legacy_policy() -> None:
    with pytest.raises(ValueError, match="at least three"):
        CarrierPipelineConfig(beam=BeamFusionConfig(min_views=2))
    with pytest.raises(ValueError, match="bounded"):
        CarrierPipelineConfig(beam=BeamFusionConfig(max_components=None))
    with pytest.raises(ValueError, match="covariance repair only"):
        CarrierPipelineConfig(
            repair=CarrierRepairConfig(
                repair_opacity=True,
            )
        )


def test_carrier_pipeline_import_keeps_dense_image_modules_out() -> None:
    code = """
import importlib
import sys
module = importlib.import_module('rtgs.carrier_pipeline')
pipeline = importlib.import_module('rtgs.pipeline')
assert module.run_carrier_pipeline is pipeline.run_carrier_pipeline
assert 'rtgs.data.scene' not in sys.modules
assert 'rtgs.data.calibrated' not in sys.modules
assert 'rtgs.image2gs.fit' not in sys.modules
assert 'rtgs.optim.trainer' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
