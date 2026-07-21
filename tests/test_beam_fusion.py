"""CPU tests for tomographic Gaussian beam fusion (density-based RGB-free initialization).

The forward oracle is :func:`rtgs.render.projection.project_covariances_ewa` with zero dilation:
ground-truth 3D Gaussians are projected into each view and beam fusion must localize them blind.
The covariance contract is deliberately different from exact triangulation: covariance
intersection is *exact on directions every view observes and conservative elsewhere* — it may
inflate weakly-shared directions but must never be overconfident. One test pins that property
against the rejected naive Gaussian product, which is provably overconfident on shared axes.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift import beam_fusion
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionResult,
    fuse_gaussian_beams,
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


def _camera(x: float, y: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, y, -3.0]),
        target=torch.zeros(3),
        width=96,
        height=96,
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
    decoy_view: int | None = None,
    name: str = "beam-fixture",
) -> ReconstructionInputs:
    fields = []
    count = gt_means.shape[0]
    for index, camera in enumerate(cameras):
        projection = project_covariances_ewa(gt_means, gt_cov, camera, dilation=0.0)
        means2d = projection.means2d
        cov2d = projection.covariances2d
        colors = gt_colors.clone()
        amplitudes = torch.full((count,), 0.8, dtype=torch.float64)
        if decoy_view == index:
            means2d = torch.cat([means2d, torch.tensor([[15.0, 80.0]], dtype=torch.float64)])
            cov2d = torch.cat([cov2d, (torch.eye(2, dtype=torch.float64) * 2.0)[None]])
            colors = torch.cat([colors, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)])
            amplitudes = torch.cat([amplitudes, torch.tensor([0.8], dtype=torch.float64)])
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


def test_fusion_localizes_projected_ground_truth_exactly():
    cameras = _default_cameras()
    gt_cov = _gt_covariances()
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))

    assert isinstance(result, BeamFusionResult)
    assert result.n_components == _GT_MEANS.shape[0]
    assert result.unmatched_per_view == (0,) * len(cameras)
    assert result.diagnostics["component_view_histogram"] == {len(cameras): _GT_MEANS.shape[0]}

    distances, assignment = torch.cdist(result.gaussians.means.double(), _GT_MEANS).min(dim=1)
    assert sorted(assignment.tolist()) == list(range(_GT_MEANS.shape[0]))
    assert float(distances.max()) < 1e-5

    from rtgs.core.sh import sh_to_rgb

    colors = sh_to_rgb(result.gaussians.sh[:, 0]).double()
    assert float((colors - _GT_COLORS[assignment]).abs().max()) < 1e-5


def test_covariance_intersection_is_never_overconfident_but_naive_product_is(monkeypatch):
    cameras = _default_cameras()[:5]
    gt_means = torch.zeros(1, 3, dtype=torch.float64)
    gt_cov = (torch.eye(3, dtype=torch.float64) * 4e-4)[None]
    gt_colors = torch.tensor([[0.6, 0.3, 0.2]], dtype=torch.float64)
    inputs = _build_inputs(gt_means, gt_cov, gt_colors, cameras)
    config = BeamFusionConfig(min_views=5)
    gt_eigen = torch.linalg.eigvalsh(gt_cov[0])

    ci = fuse_gaussian_beams(inputs, config)
    ci_ratio = torch.linalg.eigvalsh(ci.gaussians.covariance().double()[0]) / gt_eigen
    # CI: exact on well-observed directions, conservative elsewhere — never overconfident.
    assert float(ci_ratio.min()) > 0.9
    assert float(ci_ratio.max()) < 100.0

    def naive_product(precisions, means):
        fused = precisions.sum(dim=0)
        information = (precisions @ means[:, :, None]).sum(dim=0)
        return fused, torch.linalg.solve(fused, information)[:, 0]

    monkeypatch.setattr(beam_fusion, "_ci_fuse", naive_product)
    product = fuse_gaussian_beams(inputs, config)
    product_ratio = torch.linalg.eigvalsh(product.gaussians.covariance().double()[0]) / gt_eigen
    # The rejected product rule shrinks shared axes by ~1/K — measurably overconfident.
    assert float(product_ratio.min()) < 0.5


def test_identical_color_twins_have_no_surviving_ghosts_at_min_views_three():
    cameras = _default_cameras()[:5]
    twin_means = torch.tensor([[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=torch.float64)
    twin_cov = (torch.eye(3, dtype=torch.float64) * 4e-4).expand(2, 3, 3).clone()
    twin_colors = torch.full((2, 3), 0.5, dtype=torch.float64)
    inputs = _build_inputs(twin_means, twin_cov, twin_colors, cameras, name="twins")
    result = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))

    assert result.n_components == 2
    distances = torch.cdist(result.gaussians.means.double(), twin_means).min(dim=1).values
    assert float(distances.max()) < 1e-5


def test_single_view_decoy_is_excluded_and_counted():
    cameras = _default_cameras()[:5]
    gt_means = torch.tensor([[-0.2, -0.1, 0.1], [0.2, 0.1, -0.1]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(11)
    raw = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64) * 0.3
    gt_cov = raw @ raw.transpose(-1, -2) * 1e-4 + torch.eye(3, dtype=torch.float64) * 4e-4
    gt_colors = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]], dtype=torch.float64)
    inputs = _build_inputs(gt_means, gt_cov, gt_colors, cameras, decoy_view=2)
    result = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))

    assert result.n_components == 2
    assert result.unmatched_per_view == (0, 0, 1, 0, 0)
    in_view_two = result.contributor_component_indices[result.contributor_view_indices == 2]
    assert int(in_view_two.max()) < 2


def test_lineage_is_consistent_per_view_unique_and_deterministic():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=3)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    first = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))
    second = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))

    assert int(first.component_offsets[0]) == 0
    assert int(first.component_offsets[-1]) == first.contributor_view_indices.numel()
    for component in range(first.n_components):
        start = int(first.component_offsets[component])
        stop = int(first.component_offsets[component + 1])
        views = first.contributor_view_indices[start:stop].tolist()
        assert len(views) == len(set(views))  # at most one contributor per view
    assert torch.equal(first.gaussians.means, second.gaussians.means)
    assert torch.equal(first.component_offsets, second.component_offsets)
    assert torch.equal(first.contributor_view_indices, second.contributor_view_indices)
    assert torch.equal(first.component_weights, second.component_weights)


def test_covariance_eigenvalues_respect_configured_bounds():
    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=4)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    config = BeamFusionConfig(min_views=3, min_sigma_world=5e-3, max_sigma_world=2e-2)
    result = fuse_gaussian_beams(inputs, config)
    eigenvalues = torch.linalg.eigvalsh(result.gaussians.covariance().double())
    assert float(eigenvalues.min()) >= (5e-3) ** 2 * (1.0 - 1e-4)
    assert float(eigenvalues.max()) <= (2e-2) ** 2 * (1.0 + 1e-4)


def test_min_views_exceeding_cameras_fails_closed():
    cameras = _default_cameras()[:2]
    gt_cov = _gt_covariances(seed=5)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    with pytest.raises(ValueError, match="min_views exceeds"):
        fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))


def test_degenerate_identical_cameras_fail_closed_with_actionable_error():
    # Two identical cameras make every cross-view ray pair parallel, so the closed-form
    # intersection is undefined and the parallel guard must reject every candidate seed.
    cameras = [_camera(0.0, 0.0), _camera(0.0, 0.0)]
    gt_cov = _gt_covariances(seed=6)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras, name="degenerate")
    with pytest.raises(ValueError, match="no gated pair intersections"):
        fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=2))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_views", 1, "min_views must be at least two"),
        ("min_views", 2.0, "min_views must be an integer"),
        ("near", 0.0, "near must be finite and positive"),
        ("transverse_gate_sigma", 0.0, "transverse_gate_sigma"),
        ("weight_floor", -1.0, "weight_floor"),
        ("nms_voxel_size", 0.0, "nms_voxel_size"),
        ("max_sigma_world", 0.0, "max_sigma_world"),
        ("init_opacity", 1.0, "init_opacity"),
        ("source_chunk", 0, "source_chunk must be positive"),
        ("pair_limit", 0, "pair_limit"),
    ],
)
def test_config_validation(field, value, message):
    with pytest.raises((ValueError, TypeError), match=message):
        BeamFusionConfig(**{field: value})


def test_result_feeds_downstream_merge_and_eval():
    from rtgs.lift.compact_init_eval import evaluate_initialization
    from rtgs.lift.merge import merge_by_voxel

    cameras = _default_cameras()
    gt_cov = _gt_covariances(seed=7)
    inputs = _build_inputs(_GT_MEANS, gt_cov, _GT_COLORS, cameras)
    result = fuse_gaussian_beams(inputs, BeamFusionConfig(min_views=3))

    merged = merge_by_voxel(result.gaussians, voxel_size=0.05)
    assert merged.n <= result.gaussians.n
    evaluation = evaluate_initialization(inputs, result.gaussians)
    assert math.isfinite(evaluation.mean_foreground_psnr)
    assert evaluation.n_gaussians == result.n_components
