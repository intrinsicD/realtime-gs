"""End-to-end pipeline: all lifting variants must beat the random baseline and refine."""

from dataclasses import replace

import pytest
import torch

from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.image2gs.fit import FitConfig
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig
from rtgs.pipeline import (
    PipelineConfig,
    _validate_carrier_input_binding,
    compare_lifters,
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


def _carrier_inputs(scene, indices: list[int]) -> ReconstructionInputs:
    names = [f"view-{index:04d}" for index in indices]
    observations = [
        GaussianObservationField(
            width=scene.cameras[index].width,
            height=scene.cameras[index].height,
            means=torch.tensor([[16.0, 16.0]]),
            log_scales=torch.zeros(1, 2),
            rotations=torch.zeros(1),
            colors=torch.full((1, 3), 0.5),
            amplitudes=torch.full((1,), 0.5),
            view_id=name,
            provider="synthetic_fixture",
        )
        for index, name in zip(indices, names, strict=True)
    ]
    return ReconstructionInputs(
        observations=observations,
        cameras=[scene.cameras[index] for index in indices],
        view_names=names,
    )


def test_carrier_pipeline_binding_requires_exact_ordered_training_cameras(tiny_scene):
    split = replace(tiny_scene, train_indices=[0, 2], test_indices=[1, 3])
    valid = _carrier_inputs(split, [0, 2])
    _validate_carrier_input_binding(valid, split)

    wrong_view = _carrier_inputs(split, [0, 1])
    with pytest.raises(ValueError, match="exactly the scene training views"):
        _validate_carrier_input_binding(wrong_view, split)

    wrong_camera = _carrier_inputs(split, [0, 2])
    wrong_camera.cameras[1] = split.cameras[1]
    with pytest.raises(ValueError, match="camera does not match"):
        _validate_carrier_input_binding(wrong_camera, split)
