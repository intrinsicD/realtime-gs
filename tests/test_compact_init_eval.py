"""CPU tests for init-only compact-view evaluation and the dense-vs-top-K harness."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer
from rtgs.lift.compact_init_eval import (
    InitEvaluation,
    InitEvaluationProgress,
    crop_camera_to_fit_window,
    dense_merged_initialization,
    evaluate_initialization,
    prepare_evaluation_targets,
    render_teacher_image,
)
from rtgs.render.torch_ref import TorchRasterizer

_TARGETS = torch.tensor(
    [
        [-0.18, -0.16, 0.0],
        [0.18, -0.16, 0.0],
        [-0.18, 0.16, 0.05],
        [0.18, 0.16, -0.05],
    ]
)
_COLORS = torch.tensor(
    [
        [0.85, 0.15, 0.15],
        [0.15, 0.80, 0.20],
        [0.15, 0.25, 0.90],
        [0.80, 0.75, 0.15],
    ]
)


def _camera(x: float, size: int = 48) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, 0.0, -3.0]),
        target=torch.zeros(3),
        width=size,
        height=size,
        fov_x_deg=55.0,
    )


def _field(camera: Camera, view_id: str) -> GaussianObservationField:
    means, depth = camera.project(_TARGETS)
    assert bool((depth > 0).all())
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=means,
        log_scales=torch.log(torch.full((len(_TARGETS), 2), 1.7)),
        rotations=torch.tensor([0.0, 0.2, -0.15, 0.35]),
        colors=_COLORS,
        amplitudes=torch.tensor([0.9, 0.8, 0.7, 1.0]),
        view_id=view_id,
        n_init=len(_TARGETS),
    )


def _inputs() -> ReconstructionInputs:
    cameras = [_camera(x) for x in (-0.75, 0.0, 0.75)]
    names = [f"v{i}" for i in range(3)]
    return ReconstructionInputs(
        observations=[_field(c, n) for c, n in zip(cameras, names, strict=True)],
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.zeros(3), 1.2),
        name="init-eval-fixture",
    )


def _config(**overrides) -> CompactCarveConfig:
    base = dict(
        n_init_3d=4,
        candidate_multiplier=8,
        anchor_mode="component_centers",
        samples_per_ray=48,
        seed=17,
        min_views=2,
        hull_fraction=1.0,
        coverage_scale=4.0,
        coverage_threshold=0.2,
        color_std_sigma=0.25,
        min_score=0.01,
        init_opacity=0.6,
    )
    base.update(overrides)
    return CompactCarveConfig(**base)


def test_render_teacher_image_shape_coverage_and_determinism():
    field = _inputs().observations[0]
    color, coverage = render_teacher_image(field)
    fit_x, fit_y, fit_w, fit_h = field.fit_window
    assert color.shape == (fit_h, fit_w, 3)
    assert coverage.shape == (fit_h, fit_w)
    assert bool((color >= 0).all()) and bool((color <= 1).all())  # clamped to displayable range
    assert float((coverage > 0).float().mean()) > 0.0  # the object is visible

    # Row tiling must not change the rendered image.
    tiled = render_teacher_image(field, row_batch=7)[0]
    assert torch.equal(color, tiled)

    # A shared CSR backend renders identically to a freshly built one.
    shared = render_teacher_image(field, backend=GaussianObservationIndex(field))[0]
    assert torch.equal(color, shared)


def test_cropped_camera_matches_full_render_slice():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    camera = inputs.cameras[0]
    fit_window = (5, 7, 31, 29)
    cropped_camera = crop_camera_to_fit_window(camera, fit_window)
    rasterizer = TorchRasterizer(row_chunk=7)

    full = rasterizer.render(result.gaussians, camera).color
    cropped = rasterizer.render(result.gaussians, cropped_camera).color
    fit_x, fit_y, fit_width, fit_height = fit_window
    expected = full[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]

    assert cropped.shape == expected.shape
    assert torch.allclose(cropped, expected, atol=2e-6, rtol=0.0)
    assert cropped_camera.cx == camera.cx - fit_x
    assert cropped_camera.cy == camera.cy - fit_y


def test_crop_camera_rejects_invalid_fit_window():
    camera = _camera(0.0)
    with pytest.raises(ValueError, match="positive rectangle"):
        crop_camera_to_fit_window(camera, (0, 0, 0, 10))
    with pytest.raises(ValueError, match="inside the camera"):
        crop_camera_to_fit_window(camera, (40, 40, 16, 16))


def test_evaluate_initialization_silent_path_does_not_call_perf_counter(monkeypatch):
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)

    def fail_if_called():
        raise AssertionError("silent evaluation must not collect progress timing")

    monkeypatch.setattr("rtgs.lift.compact_init_eval.perf_counter", fail_if_called)

    evaluation = evaluate_initialization(inputs, result.gaussians)

    assert evaluation.n_gaussians == result.gaussians.n


def test_evaluate_initialization_emits_view_progress():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    records: list[InitEvaluationProgress] = []

    evaluate_initialization(inputs, result.gaussians, progress_callback=records.append)

    assert [record.phase for record in records] == [
        phase for _ in range(inputs.n_views) for phase in ("view_start", "view_complete")
    ]
    complete = records[1::2]
    assert [record.completed_views for record in complete] == list(range(1, inputs.n_views + 1))
    assert all(record.view_seconds is not None and record.view_seconds >= 0 for record in complete)
    assert all(record.visible_gaussians is not None for record in complete)


def test_evaluate_initialization_reports_finite_per_view_metrics():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    evaluation = evaluate_initialization(inputs, result.gaussians)

    assert isinstance(evaluation, InitEvaluation)
    assert evaluation.n_gaussians == result.gaussians.n
    assert len(evaluation.per_view) == inputs.n_views
    for view in evaluation.per_view:
        assert math.isfinite(view.psnr)
        assert math.isfinite(view.foreground_psnr)
        assert 0.0 <= view.teacher_coverage_fraction <= 1.0
    assert math.isfinite(evaluation.mean_foreground_psnr)

    # Deterministic and JSON-serializable.
    repeated = evaluate_initialization(inputs, result.gaussians)
    assert repeated.as_dict() == evaluation.as_dict()
    assert json.loads(json.dumps(evaluation.as_dict()))["n_gaussians"] == result.gaussians.n


def test_prepared_evaluation_targets_preserve_exact_metrics():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    direct = evaluate_initialization(inputs, result.gaussians)
    targets = prepare_evaluation_targets(inputs)

    prepared = evaluate_initialization(inputs, result.gaussians, targets=targets)

    assert prepared.as_dict() == direct.as_dict()


def test_evaluate_initialization_rejects_mismatched_prepared_target():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    targets = list(prepare_evaluation_targets(inputs))
    targets[0] = replace(targets[0], view_name="wrong-view")

    with pytest.raises(ValueError, match="does not match"):
        evaluate_initialization(inputs, result.gaussians, targets=targets)


def test_evaluate_initialization_validates_backend_count():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    with pytest.raises(ValueError, match="one query backend per view"):
        evaluate_initialization(
            inputs, result.gaussians, backends=[GaussianObservationIndex(inputs.observations[0])]
        )


def test_dense_merged_initialization_is_denser_than_topk_and_runs_eval():
    inputs = _inputs()
    topk = CompactCarveInitializer(_config()).initialize(inputs)
    topk_eval = evaluate_initialization(inputs, topk.gaussians)

    merged, dense, group = dense_merged_initialization(
        inputs,
        _config(n_init_3d=1, select_all_eligible=True),
        merge_voxel_size=0.06,
    )
    dense_eval = evaluate_initialization(inputs, merged)

    # The dense placement lifts every supported 2D Gaussian, then merges to a still-denser set.
    assert dense.gaussians.n > topk.gaussians.n
    assert merged.n <= dense.gaussians.n
    assert dense_eval.n_gaussians == merged.n
    assert group.shape == (dense.gaussians.n,)
    assert math.isfinite(dense_eval.mean_foreground_psnr)
    assert math.isfinite(topk_eval.mean_foreground_psnr)


def test_dense_merged_initialization_requires_select_all_eligible():
    inputs = _inputs()
    with pytest.raises(ValueError, match="select_all_eligible=True"):
        dense_merged_initialization(inputs, _config(), merge_voxel_size=0.06)


def test_harness_run_writes_metrics_and_viewer_plys(tmp_path):
    from benchmarks.compact_init_eval import _synthetic_inputs, run

    report = run(
        _synthetic_inputs(),
        n_init_3d=5,
        merge_voxel_size=0.06,
        out_dir=tmp_path,
        init_opacity=0.5,
    )
    assert report["dense_merged"]["n_gaussians"] >= 1
    assert report["dense_merged"]["lifted"] > report["topk"]["n_gaussians"]
    histogram = report["dense_merged"]["cluster_view_multiplicity_histogram"]
    assert sum(histogram.values()) == report["dense_merged"]["n_gaussians"]
    assert (
        sum(count for multiplicity, count in histogram.items() if int(multiplicity) > 1)
        == report["dense_merged"]["cross_view_correspondence_clusters"]
    )
    assert math.isfinite(report["delta_mean_foreground_psnr"])
    assert len(report["per_view_foreground_psnr_delta"]) == 5
    assert set(report["preregistered_e1_decision"]) == {
        "mean_gain_at_least_0_5_db",
        "no_view_regresses_more_than_0_25_db",
        "gaussian_count_within_2x",
        "worst_view_delta_db",
        "gaussian_count_ratio",
        "dense_is_better_init",
    }
    assert report["evaluation_backend"] == "TorchRasterizer"
    assert report["render_device"] == "cpu"
    assert report["fit_window_rendering"] is True
    assert set(report["stage_seconds"]) >= {
        "topk_placement",
        "topk_evaluation",
        "dense_placement",
        "dense_merge",
        "dense_evaluation",
    }
    assert all(seconds >= 0 for seconds in report["stage_seconds"].values())
    assert report["dense_refined_merged"] is None  # refine is off by default
    assert (tmp_path / "init_topk.ply").exists()
    assert (tmp_path / "init_dense_merged.ply").exists()
    assert (tmp_path / "init_eval.json").exists()
    assert report["viewer_command"].startswith(".venv/bin/rtgs view --gaussians")


def test_harness_refine_stage_reports_objective_and_saves_ply(tmp_path):
    from benchmarks.compact_init_eval import _synthetic_inputs, run

    report = run(
        _synthetic_inputs(),
        n_init_3d=5,
        merge_voxel_size=0.06,
        out_dir=tmp_path,
        init_opacity=0.5,
        refine=True,
    )
    refined = report["dense_refined_merged"]
    assert refined is not None
    assert math.isfinite(refined["consensus_objective_initial"])
    assert refined["consensus_objective_refined"] >= refined["consensus_objective_initial"] - 1e-6
    assert refined["n_gaussians"] >= 1
    assert (tmp_path / "init_dense_refined_merged.ply").exists()
