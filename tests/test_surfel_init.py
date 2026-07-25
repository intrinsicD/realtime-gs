"""CPU tests for cover-consistent surfel reconciliation of localization covariances.

The oracles are analytic rather than empirical: a planar lattice has a known normal, known
spacing, and a known hexagonal ripple; the partition-of-unity condition and the layered-alpha
inversion are closed-form and are checked against direct numerical evaluation, not against
current behavior.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    beam_contributor_footprints,
    estimate_local_surface_frames,
    hexagonal_cover_ripple,
    reconcile_covariances,
)


def _plane_grid(n: int = 9, spacing: float = 0.1) -> torch.Tensor:
    """A flat z=0 lattice in the xy-plane: normal is exactly +/-z, spacing is exact."""
    coords = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2.0) * spacing
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(n * n, dtype=torch.float64)], dim=-1
    )


def _needles(points: torch.Tensor, sigma: float = 1e-3, seed: int = 0) -> Gaussians3D:
    """Tiny, arbitrarily oriented, strongly anisotropic gaussians — the failure mode."""
    generator = torch.Generator().manual_seed(seed)
    n = points.shape[0]
    raw = torch.randn(n, 3, 3, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw)
    scales = torch.tensor([sigma, 2 * sigma, 4 * sigma], dtype=torch.float64).expand(n, 3)
    covs = q @ torch.diag_embed(scales.square()) @ q.transpose(-1, -2)
    return Gaussians3D.from_means_covs(
        means=points.clone(),
        covs=covs,
        colors=torch.full((n, 3), 0.5, dtype=torch.float64),
        opacity=torch.full((n,), 0.1, dtype=torch.float64),
        sh_degree=0,
    )


def test_local_frame_recovers_plane_normal_and_spacing():
    points = _plane_grid(spacing=0.1)
    frames = estimate_local_surface_frames(points, SurfelInitConfig(frame_neighbors=8))
    interior = (points[:, :2].abs().amax(dim=-1) < 0.35).nonzero(as_tuple=True)[0]
    normals = frames.normals[interior]
    assert torch.allclose(normals[:, 2].abs(), torch.ones_like(normals[:, 2]), atol=1e-8)
    # Every lattice point has four neighbours at exactly the lattice spacing.
    assert torch.allclose(
        frames.spacing[interior], torch.full_like(frames.spacing[interior], 0.1), atol=1e-9
    )
    assert float(frames.planarity[interior].min()) > 0.999
    assert float(frames.out_of_plane_sigma[interior].max()) < 1e-12
    tangents = frames.tangents[interior]
    assert torch.allclose(
        torch.einsum("nij,nj->ni", tangents.transpose(1, 2), normals),
        torch.zeros(interior.numel(), 2, dtype=torch.float64),
        atol=1e-8,
    )


def test_frames_are_chunk_invariant():
    points = _plane_grid(n=7, spacing=0.07)
    whole = estimate_local_surface_frames(points, SurfelInitConfig(neighbor_chunk=4096))
    chunked = estimate_local_surface_frames(points, SurfelInitConfig(neighbor_chunk=5))
    assert torch.equal(whole.spacing, chunked.spacing)
    assert torch.equal(whole.planarity, chunked.planarity)
    assert torch.allclose(whole.normals.abs(), chunked.normals.abs(), atol=0)


def test_reconciliation_produces_a_tiling_cover_from_needles():
    points = _plane_grid(spacing=0.1)
    gaussians = _needles(points)
    prior = torch.linalg.eigvalsh(gaussians.covariance()).clamp_min(0).sqrt()
    result = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8))

    # The failure mode: the largest prior axis is far below the neighbour spacing.
    assert float((prior[:, 2] / result.frames.spacing).median()) < 0.1
    # The repair: the tangential sigma is the derived fraction of the spacing.
    interior = (points[:, :2].abs().amax(dim=-1) < 0.35).nonzero(as_tuple=True)[0]
    assert torch.allclose(
        result.sigma_tangent[interior] / result.frames.spacing[interior],
        torch.full((interior.numel(),), 0.5, dtype=torch.float64),
        atol=1e-9,
    )
    # Frozen non-covariance fields are untouched.
    assert torch.equal(result.gaussians.means, gaussians.means)
    assert torch.equal(result.gaussians.sh, gaussians.sh)
    assert result.gaussians.n == gaussians.n


def test_reconciled_surfels_are_flat_along_the_measured_normal():
    points = _plane_grid(spacing=0.1)
    result = reconcile_covariances(_needles(points), SurfelInitConfig(frame_neighbors=8))
    rotations = quat_to_rotmat(result.gaussians.quats)
    scales = result.gaussians.scales
    shortest = rotations[torch.arange(points.shape[0]), :, scales.argmin(dim=-1)]
    interior = (points[:, :2].abs().amax(dim=-1) < 0.35).nonzero(as_tuple=True)[0]
    alignment = (shortest[interior] * result.frames.normals[interior]).sum(dim=-1).abs()
    assert float(alignment.min()) > 0.999
    assert (
        float((result.sigma_normal[interior] / result.sigma_tangent[interior]).max()) <= 0.5 + 1e-9
    )


def test_isotropic_mode_keeps_scale_but_asserts_no_orientation():
    points = _plane_grid(spacing=0.1)
    gaussians = _needles(points)
    oriented = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8))
    isotropic = reconcile_covariances(
        gaussians, SurfelInitConfig(frame_neighbors=8, isotropic=True)
    )
    assert torch.equal(oriented.sigma_tangent, isotropic.sigma_tangent)
    assert torch.allclose(
        isotropic.gaussians.scales,
        isotropic.sigma_tangent[:, None].expand(-1, 3).to(isotropic.gaussians.scales.dtype),
    )
    identity = torch.zeros_like(isotropic.gaussians.quats)
    identity[:, 0] = 1.0
    assert torch.equal(isotropic.gaussians.quats, identity)


def test_cover_ratio_is_derived_from_the_hexagonal_ripple_not_tuned():
    """The default 0.5 must sit at the knee where the lattice ripple collapses."""
    ripples = {ratio: hexagonal_cover_ripple(ratio) for ratio in (0.3, 0.4, 0.5, 0.6)}
    assert ripples[0.3] > 0.5  # a cover this fine is full of holes
    assert ripples[0.4] > 0.10
    assert ripples[0.5] < 0.02  # the default: smooth to better than two percent
    assert ripples[0.6] < ripples[0.5]
    assert SurfelInitConfig().coverage_sigma_ratio == 0.5

    # Independent re-derivation of the ripple by direct summation over a different window.
    spacing, sigma = 1.0, 0.5
    basis = torch.tensor(
        [[spacing, 0.0], [spacing / 2.0, spacing * math.sqrt(3.0) / 2.0]], dtype=torch.float64
    )
    span = torch.arange(-20, 21, dtype=torch.float64)
    grid_a, grid_b = torch.meshgrid(span, span, indexing="ij")
    centers = torch.stack([grid_a.reshape(-1), grid_b.reshape(-1)], dim=-1) @ basis

    def field(query: torch.Tensor) -> float:
        offset = query[None, :] - centers
        return float(torch.exp(-offset.square().sum(dim=-1) / (2 * sigma**2)).sum())

    values = [
        field(torch.zeros(2, dtype=torch.float64)),
        field(basis[0] / 2.0),
        field((basis[0] + basis[1]) / 3.0),
    ]
    independent = (max(values) - min(values)) / (sum(values) / len(values))
    assert independent == pytest.approx(ripples[0.5], rel=1e-6)
    # The cover's mean equals the analytic areal density 2*pi*sigma^2 / cell area, which is
    # exactly the overlap term the opacity rule inverts.
    expected_mean = 2 * math.pi * sigma**2 / (math.sqrt(3.0) / 2.0 * spacing**2)
    assert sum(values) / len(values) == pytest.approx(expected_mean, rel=1e-3)


def test_coverage_opacity_inverts_the_layered_alpha_equation():
    points = _plane_grid(spacing=0.1)
    config = SurfelInitConfig(frame_neighbors=8, target_alpha=0.9)
    result = reconcile_covariances(_needles(points), config)
    interior = (points[:, :2].abs().amax(dim=-1) < 0.35).nonzero(as_tuple=True)[0]
    opacity = result.gaussians.opacity[interior].to(torch.float64)
    overlap = result.overlap[interior]
    accumulated = 1.0 - (1.0 - opacity) ** overlap
    assert torch.allclose(accumulated, torch.full_like(accumulated, config.target_alpha), atol=1e-9)
    # A fixed 0.10 opacity cannot reach the target at this overlap; the derived one must be larger.
    assert float(opacity.min()) > 0.1


def test_fixed_opacity_mode_is_an_attribution_control():
    points = _plane_grid(spacing=0.1)
    result = reconcile_covariances(
        _needles(points),
        SurfelInitConfig(frame_neighbors=8, use_coverage_opacity=False, fixed_opacity=0.1),
    )
    assert torch.allclose(result.gaussians.opacity, torch.full_like(result.gaussians.opacity, 0.1))


def test_resolution_floor_raises_only_where_the_cover_is_finer_than_the_images():
    points = _plane_grid(spacing=0.1)
    gaussians = _needles(points)
    n = points.shape[0]
    floor = torch.full((n,), 0.02, dtype=torch.float64)
    floor[: n // 2] = 0.2  # coarser than the 0.05 cover sigma
    config = SurfelInitConfig(frame_neighbors=8)
    floored = reconcile_covariances(gaussians, config, resolution_floor=floor)
    plain = reconcile_covariances(gaussians, config)
    assert bool(floored.resolution_floor_active[: n // 2].all())
    assert not bool(floored.resolution_floor_active[n // 2 :].any())
    assert torch.allclose(floored.sigma_tangent[: n // 2], floor[: n // 2])
    assert torch.equal(floored.sigma_tangent[n // 2 :], plain.sigma_tangent[n // 2 :])
    # Disabling the floor must exactly restore the unfloored result.
    disabled = reconcile_covariances(
        gaussians,
        SurfelInitConfig(frame_neighbors=8, use_resolution_floor=False),
        resolution_floor=floor,
    )
    assert torch.equal(disabled.sigma_tangent, plain.sigma_tangent)


def test_normal_thickness_rises_to_localization_uncertainty_within_the_flatness_cap():
    points = _plane_grid(spacing=0.1)
    n = points.shape[0]
    # Wide, isotropic localization uncertainty: the surface may not be claimed thinner.
    covs = torch.eye(3, dtype=torch.float64).expand(n, 3, 3) * (0.02**2)
    gaussians = Gaussians3D.from_means_covs(
        means=points.clone(),
        covs=covs,
        colors=torch.full((n, 3), 0.5, dtype=torch.float64),
        opacity=torch.full((n,), 0.1, dtype=torch.float64),
        sh_degree=0,
    )
    result = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8, flatness=0.5))
    interior = (points[:, :2].abs().amax(dim=-1) < 0.35).nonzero(as_tuple=True)[0]
    # flatness*sigma_tangent = 0.5*0.05 = 0.025 exceeds the 0.02 localization sigma, so the
    # localization bound is the binding constraint and must be reproduced exactly.
    assert torch.allclose(
        result.sigma_normal[interior], torch.full((interior.numel(),), 0.02, dtype=torch.float64)
    )
    # With a tighter flatness the cap is applied last and overrides the localization bound.
    capped = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8, flatness=0.2))
    assert torch.allclose(
        capped.sigma_normal[interior],
        0.2 * capped.sigma_tangent[interior],
    )
    assert float(capped.sigma_normal[interior].max()) < 0.02


def test_reconciliation_is_deterministic_and_covariance_is_spd():
    points = (
        _plane_grid(spacing=0.1)
        + torch.randn(81, 3, generator=torch.Generator().manual_seed(3)) * 0.005
    )
    gaussians = _needles(points, seed=1)
    first = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8))
    second = reconcile_covariances(gaussians, SurfelInitConfig(frame_neighbors=8))
    assert torch.equal(first.gaussians.log_scales, second.gaussians.log_scales)
    assert torch.equal(first.gaussians.quats, second.gaussians.quats)
    eigenvalues = torch.linalg.eigvalsh(first.gaussians.covariance().to(torch.float64))
    assert float(eigenvalues.min()) > 0.0
    assert bool(torch.isfinite(first.gaussians.log_scales).all())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_neighbors": 2},
        {"spacing_neighbors": 20, "frame_neighbors": 8},
        {"target_alpha": 1.0},
        {"min_opacity": 0.9, "max_opacity": 0.1},
        {"coverage_sigma_ratio": 0.0},
        {"min_planarity": 1.0},
    ],
)
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SurfelInitConfig(**kwargs)


def test_reconciliation_rejects_mismatched_resolution_floor():
    points = _plane_grid(n=5, spacing=0.1)
    gaussians = _needles(points)
    with pytest.raises(ValueError):
        reconcile_covariances(
            gaussians,
            SurfelInitConfig(frame_neighbors=4),
            resolution_floor=torch.ones(3, dtype=torch.float64),
        )


def test_beam_contributor_footprints_matches_the_projection_scaling():
    """sigma_px * depth / f, reduced per component, from beam's own CSR lineage."""
    from rtgs.core.camera import Camera
    from rtgs.core.observation2d import GaussianObservationField
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    class _Beam:
        component_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
        contributor_view_indices = torch.tensor([0, 1, 0], dtype=torch.long)
        contributor_component_indices = torch.tensor([0, 1, 1], dtype=torch.long)
        contributor_depths = torch.tensor([2.0, 4.0, 3.0], dtype=torch.float64)

    def field(view_id: str) -> GaussianObservationField:
        return GaussianObservationField(
            width=64,
            height=64,
            means=torch.tensor([[10.0, 10.0], [20.0, 20.0]]),
            log_scales=torch.log(torch.tensor([[2.0, 2.0], [4.0, 4.0]])),
            rotations=torch.zeros(2),
            colors=torch.full((2, 3), 0.5),
            amplitudes=torch.ones(2),
            view_id=view_id,
        )

    camera = Camera(
        fx=100.0,
        fy=100.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
        R=torch.eye(3),
        t=torch.zeros(3),
    )
    inputs = ReconstructionInputs(
        observations=[field("a"), field("b")],
        cameras=[camera, camera],
        view_names=["a", "b"],
    )
    footprints = beam_contributor_footprints(inputs, _Beam())
    # Component 0 links sigma_px 2 at depth 2 (0.04) and sigma_px 4 at depth 4 (0.16); the
    # documented lower-median convention selects 0.04.
    assert footprints[0] == pytest.approx(0.04)
    assert footprints[1] == pytest.approx(4.0 * 3.0 / 100.0)
    assert beam_contributor_footprints(inputs, _Beam(), reducer="min")[0] == pytest.approx(0.04)
    assert beam_contributor_footprints(inputs, _Beam(), reducer="max")[0] == pytest.approx(0.16)
    with pytest.raises(ValueError):
        beam_contributor_footprints(inputs, _Beam(), reducer="mean")
