"""CPU tests for calibrated structure-from-splats (SfM-style RGB-free initialization).

The forward oracle is :func:`rtgs.render.projection.project_covariances_ewa` with zero dilation:
ground-truth 3D Gaussians are projected into each view to build exact 2D observation fields, and
splat-SfM must invert that construction — matching, tracks, DLT centers, and the linear covariance
triangulation — without access to the ground truth.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.splat_sfm import (
    SplatSfMConfig,
    SplatSfMResult,
    structure_from_splats,
)
from rtgs.render.projection import project_covariances_ewa

_GT_MEANS = torch.tensor(
    [
        [-0.20, -0.16, 0.12],
        [0.20, -0.16, -0.12],
        [-0.18, 0.18, 0.06],
        [0.20, 0.16, -0.10],
        [0.00, 0.00, 0.16],
        [0.05, -0.22, 0.02],
    ],
    dtype=torch.float64,
)
_GT_COLORS = torch.tensor(
    [
        [0.85, 0.15, 0.15],
        [0.15, 0.80, 0.20],
        [0.15, 0.25, 0.90],
        [0.80, 0.75, 0.15],
        [0.60, 0.20, 0.70],
        [0.20, 0.65, 0.65],
    ],
    dtype=torch.float64,
)


def _gt_covariances(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(_GT_MEANS.shape[0], 3, 3, generator=generator, dtype=torch.float64) * 0.4
    return raw @ raw.transpose(-1, -2) * 1e-4 + torch.eye(3, dtype=torch.float64) * 4e-4


def _camera(x: float, y: float, *, size: int = 96) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, y, -3.0]),
        target=torch.zeros(3),
        width=size,
        height=size,
        fov_x_deg=50.0,
    )


def _default_cameras() -> list[Camera]:
    poses = [(-1.0, 0.2), (-0.5, -0.2), (0.0, 0.25), (0.5, -0.15), (1.0, 0.1), (0.2, 0.6)]
    return [_camera(x, y) for x, y in poses]


def _cov2d_to_rs(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    theta = torch.atan2(eigenvectors[:, 1, 0], eigenvectors[:, 0, 0])
    return torch.log(eigenvalues.clamp_min(1e-12).sqrt()), theta


def _build_inputs(
    gt_means: torch.Tensor,
    gt_cov: torch.Tensor,
    gt_colors: torch.Tensor,
    cameras: list[Camera],
    *,
    extra: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    name: str = "sfm-fixture",
) -> ReconstructionInputs:
    """Project ground truth into every camera; ``extra`` adds per-view decoy splats."""
    fields = []
    for index, camera in enumerate(cameras):
        projection = project_covariances_ewa(gt_means, gt_cov, camera, dilation=0.0)
        means2d = projection.means2d
        cov2d = projection.covariances2d
        colors = gt_colors.clone()
        amplitudes = torch.full((gt_means.shape[0],), 0.8, dtype=torch.float64)
        if extra is not None and index in extra:
            extra_xy, extra_cov, extra_color = extra[index]
            means2d = torch.cat([means2d, extra_xy])
            cov2d = torch.cat([cov2d, extra_cov])
            colors = torch.cat([colors, extra_color])
            amplitudes = torch.cat(
                [amplitudes, torch.full((extra_xy.shape[0],), 0.8, dtype=torch.float64)]
            )
        log_scales, rotations = _cov2d_to_rs(cov2d)
        fields.append(
            GaussianObservationField(
                width=camera.width,
                height=camera.height,
                means=means2d.to(torch.float32),
                log_scales=log_scales.to(torch.float32),
                rotations=rotations.to(torch.float32),
                colors=colors.to(torch.float32),
                amplitudes=amplitudes.to(torch.float32),
                view_id=f"v{index}",
                n_init=int(means2d.shape[0]),
            )
        )
    return ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=[f"v{index}" for index in range(len(cameras))],
        bounds_hint=(torch.zeros(3), 1.2),
        name=name,
    )


def test_exact_recovery_of_projected_ground_truth():
    cameras = _default_cameras()
    gt_cov = _gt_covariances()
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = structure_from_splats(inputs)

    assert isinstance(result, SplatSfMResult)
    assert result.n_tracks == _GT_MEANS.shape[0]
    # Every splat joined a full-length track: nothing unmatched, all views in each track.
    assert result.unmatched_per_view == (0,) * len(cameras)
    assert result.diagnostics["track_length_histogram"] == {len(cameras): _GT_MEANS.shape[0]}

    distances, assignment = torch.cdist(result.gaussians.means.double(), _GT_MEANS).min(dim=1)
    assert sorted(assignment.tolist()) == list(range(_GT_MEANS.shape[0]))  # bijective
    assert float(distances.max()) < 1e-5

    recovered = result.gaussians.covariance().double()
    for track in range(result.n_tracks):
        target = gt_cov[assignment[track]]
        relative = torch.linalg.norm(recovered[track] - target) / torch.linalg.norm(target)
        assert float(relative) < 1e-4

    from rtgs.core.sh import sh_to_rgb

    colors = sh_to_rgb(result.gaussians.sh[:, 0]).double()
    assert float((colors - _GT_COLORS[assignment]).abs().max()) < 1e-5

    assert float(result.track_reprojection_error.max()) < 1e-3
    assert float(result.track_triangulation_angle_deg.min()) > 10.0
    assert float(result.track_covariance_residual.max()) < 1e-5


def test_single_view_decoy_is_excluded_and_counted():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=1)
    decoy = {
        2: (
            torch.tensor([[15.0, 80.0]], dtype=torch.float64),
            (torch.eye(2, dtype=torch.float64) * 2.0)[None],
            torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64),
        )
    }
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras, extra=decoy)
    result = structure_from_splats(inputs)

    assert result.n_tracks == _GT_MEANS.shape[0]
    assert result.unmatched_per_view == (0, 0, 1, 0, 0, 0)
    # The decoy component (index G) never appears in any track member list.
    in_view_two = result.member_component_indices[result.member_view_indices == 2]
    assert int(in_view_two.max()) < _GT_MEANS.shape[0]


def test_identical_color_twins_are_disambiguated_by_geometry():
    cameras = _default_cameras()[:5]
    twin_means = torch.tensor([[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=torch.float64)
    twin_cov = (torch.eye(3, dtype=torch.float64) * 4e-4).expand(2, 3, 3).clone()
    twin_colors = torch.full((2, 3), 0.5, dtype=torch.float64)
    inputs = _build_inputs(twin_means, twin_cov, twin_colors, cameras, name="twins")
    result = structure_from_splats(inputs)

    assert result.n_tracks == 2
    distances = torch.cdist(result.gaussians.means.double(), twin_means).min(dim=1).values
    assert float(distances.max()) < 1e-5
    assert float(result.track_reprojection_error.max()) < 1e-3


def test_per_view_uniqueness_never_places_two_same_view_splats_in_one_track():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=2)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = structure_from_splats(inputs)
    for track in range(result.n_tracks):
        start, stop = int(result.track_offsets[track]), int(result.track_offsets[track + 1])
        views = result.member_view_indices[start:stop].tolist()
        assert len(views) == len(set(views))


def test_lineage_offsets_are_consistent_and_deterministic():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=3)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    first = structure_from_splats(inputs)
    second = structure_from_splats(inputs)

    assert int(first.track_offsets[0]) == 0
    assert int(first.track_offsets[-1]) == first.member_view_indices.numel()
    assert first.member_view_indices.shape == first.member_component_indices.shape
    assert torch.equal(first.gaussians.means, second.gaussians.means)
    assert torch.equal(first.track_offsets, second.track_offsets)
    assert torch.equal(first.member_view_indices, second.member_view_indices)
    assert torch.equal(first.member_component_indices, second.member_component_indices)


def test_covariance_eigenvalues_respect_configured_bounds():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=4)
    config = SplatSfMConfig(min_sigma_world=5e-3, max_sigma_world=2e-2)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = structure_from_splats(inputs, config)
    eigenvalues = torch.linalg.eigvalsh(result.gaussians.covariance().double())
    assert float(eigenvalues.min()) >= (5e-3) ** 2 * (1.0 - 1e-4)
    assert float(eigenvalues.max()) <= (2e-2) ** 2 * (1.0 + 1e-4)


def test_tiny_baseline_fails_the_triangulation_angle_gate():
    cameras = [_camera(0.0, 0.0), _camera(0.003, 0.0)]
    gt_cov = _gt_covariances(seed=5)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras, name="tiny-baseline")
    with pytest.raises(ValueError, match="triangulation gates|no multi-view tracks"):
        structure_from_splats(inputs)


def test_min_views_exceeding_cameras_fails_closed():
    cameras = _default_cameras()[:2]
    gt_cov = _gt_covariances(seed=6)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    with pytest.raises(ValueError, match="min_views exceeds"):
        structure_from_splats(inputs, SplatSfMConfig(min_views=3))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_views", 1, "min_views must be at least two"),
        ("min_views", 2.0, "min_views must be an integer"),
        ("ratio_test", 0.0, "ratio_test must be in"),
        ("near", 0.0, "near must be finite and positive"),
        ("min_triangulation_angle_deg", 95.0, "min_triangulation_angle_deg"),
        ("max_sigma_world", 0.0, "max_sigma_world"),
        ("init_opacity", 1.0, "init_opacity"),
        ("source_chunk", 0, "source_chunk must be positive"),
    ],
)
def test_config_validation(field, value, message):
    kwargs = {field: value}
    with pytest.raises((ValueError, TypeError), match=message):
        SplatSfMConfig(**kwargs)


def test_result_feeds_downstream_merge_and_eval():
    from rtgs.lift.compact_init_eval import evaluate_initialization
    from rtgs.lift.merge import merge_by_voxel

    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=7)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = structure_from_splats(inputs)

    merged = merge_by_voxel(result.gaussians, voxel_size=0.05)
    assert merged.n <= result.gaussians.n
    evaluation = evaluate_initialization(inputs, result.gaussians)
    assert math.isfinite(evaluation.mean_foreground_psnr)
    assert evaluation.n_gaussians == result.n_tracks


def test_screen_harness_writes_metrics_and_plys(tmp_path):
    from benchmarks.compact_init_eval import _synthetic_inputs
    from benchmarks.splat_sfm_screen import run

    report = run(
        _synthetic_inputs(),
        n_init_3d=5,
        merge_voxel_size=0.06,
        out_dir=tmp_path,
    )
    assert set(report["arms"]) == {"topk", "dense_merged", "splat_sfm"}
    for name, arm in report["arms"].items():
        assert math.isfinite(arm["mean_foreground_psnr"])
        assert (tmp_path / f"init_{name}.ply").exists()
    assert report["splat_sfm_diagnostics"]["n_tracks"] >= 1
    assert (tmp_path / "splat_sfm_screen.json").exists()
    assert report["viewer_command"].startswith("rtgs view --gaussians")
