"""CPU tests for init-only compact-view evaluation and the dense-vs-top-K harness."""

from __future__ import annotations

import json
import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer
from rtgs.lift.compact_init_eval import (
    InitEvaluation,
    dense_merged_initialization,
    evaluate_initialization,
    render_teacher_image,
)

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
    assert math.isfinite(report["delta_mean_foreground_psnr"])
    assert (tmp_path / "init_topk.ply").exists()
    assert (tmp_path / "init_dense_merged.ply").exists()
    assert (tmp_path / "init_eval.json").exists()
    assert report["viewer_command"].startswith("rtgs view --gaussians")
