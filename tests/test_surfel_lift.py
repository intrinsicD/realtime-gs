"""ADR-XXXX acceptance criteria: synthetic ground truth, round trip, determinism, no autograd.

The synthetic fixture is a closed loop: a ground-truth planar surfel patch is *projected* with
the reference EWA equations into per-view 2D captures, and the lift must recover the surfel
parameters it was built from.  That is the only way to separate "the construction is right"
from "the construction agrees with itself".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.surfel_lift import (
    AlphaTarget,
    SurfelLiftConfig,
    build_surfel_lift,
    estimate_surfel_colors,
    estimate_surfel_rotations,
    estimate_surfel_scales,
    solve_transmittance_opacity,
)
from rtgs.render.projection import project_covariances_ewa

SIGMA_T1 = 0.030
SIGMA_T2 = 0.018
SIGMA_N = 0.002
GRID = 9
SPACING = 0.055


@dataclass(frozen=True)
class _Beam:
    """The minimal beam-fusion lineage surface the lift consumes."""

    gaussians: Gaussians3D
    component_offsets: torch.Tensor
    contributor_view_indices: torch.Tensor
    contributor_component_indices: torch.Tensor
    contributor_depths: torch.Tensor


def _look_at(eye: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward = target - eye
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.95:
        up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])  # world -> camera
    return rotation, -rotation @ eye


def _plane_scene(n_views: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Camera]]:
    """A planar patch of ground-truth surfels in the z = 0 plane, seen by ``n_views`` cameras."""
    axis = (torch.arange(GRID, dtype=torch.float32) - (GRID - 1) / 2.0) * SPACING
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(GRID * GRID)], dim=-1)
    normal = torch.tensor([0.0, 0.0, 1.0])
    tangents = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cameras = []
    for index in range(n_views):
        angle = 2.0 * math.pi * index / n_views
        eye = torch.tensor(
            [1.1 * math.cos(angle), 1.1 * math.sin(angle), 1.4 + 0.15 * index], dtype=torch.float32
        )
        rotation, translation = _look_at(eye, torch.zeros(3))
        cameras.append(
            Camera(
                fx=600.0,
                fy=600.0,
                cx=160.0,
                cy=120.0,
                width=320,
                height=240,
                R=rotation,
                t=translation,
            )
        )
    return means, normal, tangents, cameras


def _ground_truth_gaussians(means: torch.Tensor) -> Gaussians3D:
    n = means.shape[0]
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0  # identity: columns are exactly (x, y, z) = (t1, t2, n)
    log_scales = torch.tensor([SIGMA_T1, SIGMA_T2, SIGMA_N]).log().expand(n, 3).clone()
    return Gaussians3D(
        means=means.clone(),
        quats=quats,
        log_scales=log_scales,
        opacity=torch.full((n,), 0.5),
        sh=torch.zeros(n, 1, 3),
    )


def _observation_from_projection(
    camera: Camera, truth: Gaussians3D, colors: torch.Tensor
) -> tuple[GaussianObservationField, torch.Tensor, torch.Tensor]:
    """Project the ground-truth surfels and re-encode them as a 2D capture."""
    projection = project_covariances_ewa(
        truth.means, truth.covariance(), camera, dilation=0.0, near=0.05
    )
    covariance = projection.covariances2d.to(torch.float64)
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(1e-10)
    log_scales = (values.sqrt()).log().flip(-1).to(torch.float32)  # (major, minor)
    major = vectors[:, :, 1]
    rotations = torch.atan2(major[:, 1], major[:, 0]).to(torch.float32)
    field = GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=projection.means2d.to(torch.float32),
        log_scales=log_scales,
        rotations=rotations,
        colors=colors,
        amplitudes=torch.ones(truth.n),
        provider="synthetic_fixture",
    )
    return field, projection.means2d, projection.depth


def _synthetic_inputs(
    n_views: int = 4, color_offset: float = 0.0
) -> tuple[ReconstructionInputs, _Beam, Gaussians3D]:
    means, _, _, cameras = _plane_scene(n_views)
    truth = _ground_truth_gaussians(means)
    n = means.shape[0]
    observations, depths = [], []
    for index, camera in enumerate(cameras):
        colors = torch.full((n, 3), 0.4 + color_offset * index)
        field, _, depth = _observation_from_projection(camera, truth, colors)
        observations.append(field)
        depths.append(depth.to(torch.float64))
    inputs = ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=[f"V{index}" for index in range(n_views)],
        name="synthetic-plane",
    )
    offsets = torch.arange(n + 1, dtype=torch.int64) * n_views
    view_indices = torch.arange(n_views, dtype=torch.int64).repeat(n)
    component_indices = torch.arange(n, dtype=torch.int64).repeat_interleave(n_views)
    contributor_depths = torch.stack(depths, dim=-1).reshape(-1)
    beam = _Beam(
        gaussians=Gaussians3D(
            means=means.clone(),
            quats=truth.quats.clone(),
            log_scales=truth.log_scales.clone(),
            opacity=truth.opacity.clone(),
            sh=truth.sh.clone(),
        ),
        component_offsets=offsets,
        contributor_view_indices=view_indices,
        contributor_component_indices=component_indices,
        contributor_depths=contributor_depths,
    )
    return inputs, beam, truth


# -- A1: synthetic ground truth ---------------------------------------------------------------


def test_a1_normals_recover_the_ground_truth_plane():
    inputs, beam, _ = _synthetic_inputs()
    frames = estimate_surfel_rotations(beam.gaussians.means, inputs, beam)
    planar = frames.coherence > 0.5
    assert bool(planar.any()), "a flat grid must produce coherent primitives"
    cosine = frames.normals[planar][:, 2].abs().clamp(0.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cosine))
    assert float(angle.median()) < 15.0
    # The sign is disambiguated toward the cameras, which all sit at z > 0.
    assert float(frames.normals[:, 2].sign().mean()) > 0.9


def test_a1_tangential_scales_recover_the_projected_footprint():
    inputs, beam, _ = _synthetic_inputs()
    frames = estimate_surfel_rotations(beam.gaussians.means, inputs, beam)
    scales, _ = estimate_surfel_scales(beam.gaussians.means, frames, inputs, beam)
    # The PCA tangent frame is not axis-aligned with the ground-truth (t1, t2), so the
    # recoverable invariant is the geometric mean of the tangential sigmas.
    recovered = (scales.sigma_t1 * scales.sigma_t2).sqrt()
    truth = math.sqrt(SIGMA_T1 * SIGMA_T2)
    interior = frames.coherence > 0.5
    log_ratio = (recovered[interior] / truth).log().abs()
    assert float(log_ratio.median()) < 0.35, float(log_ratio.median())


def test_normal_axis_is_the_declared_prior_not_an_estimate():
    inputs, beam, _ = _synthetic_inputs()
    frames = estimate_surfel_rotations(beam.gaussians.means, inputs, beam)
    for rho in (0.05, 0.25):
        scales, flags = estimate_surfel_scales(
            beam.gaussians.means, frames, inputs, beam, SurfelLiftConfig(rho=rho)
        )
        geometric = (scales.sigma_t1 * scales.sigma_t2).sqrt()
        ratio = (scales.sigma_normal / geometric)[~flags["flag_isolated"]]
        assert torch.allclose(ratio, torch.full_like(ratio, rho), atol=1e-5)
        # §5: an isolated primitive is a ball, not a surfel, whatever rho says.
        isolated = (scales.sigma_normal / geometric)[flags["flag_isolated"]]
        assert torch.allclose(isolated, torch.ones_like(isolated), atol=1e-5)


def test_incidence_gate_rejects_grazing_observations():
    """The gate must reject views, and a primitive left with none must still be scaled."""
    inputs, beam, _ = _synthetic_inputs()
    frames = estimate_surfel_rotations(beam.gaussians.means, inputs, beam)
    permissive, _ = estimate_surfel_scales(
        beam.gaussians.means, frames, inputs, beam, SurfelLiftConfig(max_incidence_deg=89.0)
    )
    partial, _ = estimate_surfel_scales(
        beam.gaussians.means, frames, inputs, beam, SurfelLiftConfig(max_incidence_deg=34.0)
    )
    assert int(partial.used_views.sum()) < int(permissive.used_views.sum())
    # Rejecting *every* view would leave a primitive with no scale evidence at all, which is
    # worse than a grazing estimate; the rescue restores the candidates rather than emitting a
    # degenerate surfel.  This behaviour is an addition to the ADR and is asserted, not assumed.
    everything = estimate_surfel_scales(
        beam.gaussians.means, frames, inputs, beam, SurfelLiftConfig(max_incidence_deg=1.0)
    )[0]
    assert int(everything.used_views.min()) >= 1
    assert bool(torch.isfinite(everything.sigma_t1).all())


def test_colors_are_a_weighted_mean_and_higher_bands_are_zero():
    inputs, beam, _ = _synthetic_inputs()
    frames = estimate_surfel_rotations(beam.gaussians.means, inputs, beam)
    sh = estimate_surfel_colors(beam.gaussians.means, frames, inputs, beam)
    assert sh.shape == (beam.gaussians.n, 16, 3)
    assert torch.count_nonzero(sh[:, 1:]) == 0
    from rtgs.core.sh import sh_to_rgb

    recovered = sh_to_rgb(sh[:, 0])
    assert torch.allclose(recovered, torch.full_like(recovered, 0.4), atol=1e-4)


# -- A2: round trip ---------------------------------------------------------------------------


def test_a2_opacity_solve_reduces_the_transmittance_residual():
    inputs, beam, truth = _synthetic_inputs()
    targets = []
    for index, camera in enumerate(inputs.cameras):
        from rtgs.render.torch_ref import TorchRasterizer

        out = TorchRasterizer().render(truth, camera)
        targets.append(
            AlphaTarget(view_index=index, alpha=out.alpha.detach().clamp(0, 1), origin=(0, 0))
        )
    config = SurfelLiftConfig(pixel_stride=2, nnls_iterations=120)
    result = build_surfel_lift(inputs, beam, config, alpha_targets=targets)
    solve = result.opacity_solve
    assert solve is not None
    assert solve.residual_after < solve.residual_before
    assert result.diagnostics["opacity_source"] == "transmittance_nnls"
    assert bool((result.gaussians.opacity > 0).all())


def test_opacity_falls_back_when_no_alpha_is_supplied():
    inputs, beam, _ = _synthetic_inputs()
    result = build_surfel_lift(inputs, beam, SurfelLiftConfig(), alpha_targets=None)
    assert result.opacity_solve is None
    assert result.diagnostics["opacity_source"] == "fallback_constant"
    assert torch.allclose(result.gaussians.opacity, torch.full_like(result.gaussians.opacity, 0.1))


# -- A3: determinism, A4: no hidden optimization ----------------------------------------------


def test_a3_determinism_bit_identical_output():
    inputs, beam, _ = _synthetic_inputs()
    first = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    second = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(first.gaussians, name), getattr(second.gaussians, name))


def test_a3_chunk_invariance():
    inputs, beam, _ = _synthetic_inputs()
    small = build_surfel_lift(inputs, beam, SurfelLiftConfig(neighbor_chunk=7))
    large = build_surfel_lift(inputs, beam, SurfelLiftConfig(neighbor_chunk=4096))
    assert torch.equal(small.gaussians.log_scales, large.gaussians.log_scales)


def test_a4_no_autograd_graph_is_produced():
    inputs, beam, _ = _synthetic_inputs()
    result = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        tensor = getattr(result.gaussians, name)
        assert tensor.grad_fn is None
        assert not tensor.requires_grad


def test_spd_by_construction():
    inputs, beam, _ = _synthetic_inputs()
    result = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    eigenvalues = torch.linalg.eigvalsh(result.gaussians.covariance().to(torch.float64))
    assert bool((eigenvalues > 0).all())
    assert torch.allclose(
        result.gaussians.quats.norm(dim=-1), torch.ones(beam.gaussians.n), atol=1e-5
    )


# -- §5 degenerate cases -----------------------------------------------------------------------


def test_flags_are_reported_for_every_declared_degenerate_case():
    inputs, beam, _ = _synthetic_inputs()
    result = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    for name in (
        "flag_single_view",
        "flag_isolated",
        "flag_line_like",
        "flag_scale_disagree",
        "flag_normal_disagree",
        "merge_candidate",
    ):
        assert name in result.flags
        assert result.flags[name].shape == (beam.gaussians.n,)
        assert name in result.diagnostics["flag_fractions"]


def test_config_rejects_out_of_range_levels():
    with pytest.raises(ValueError):
        SurfelLiftConfig(rho=0.0)
    with pytest.raises(ValueError):
        SurfelLiftConfig(max_incidence_deg=90.0)
    with pytest.raises(ValueError):
        SurfelLiftConfig(alpha_clip=1.0)
    with pytest.raises(ValueError):
        SurfelLiftConfig(aspect_cap=0.5)


def test_solver_rejects_an_empty_system():
    inputs, beam, _ = _synthetic_inputs()
    result = build_surfel_lift(inputs, beam, SurfelLiftConfig())
    far = Camera(
        fx=600.0,
        fy=600.0,
        cx=160.0,
        cy=120.0,
        width=320,
        height=240,
        R=torch.eye(3),
        t=torch.tensor([0.0, 0.0, 500.0]),
    )
    target = AlphaTarget(view_index=0, alpha=torch.zeros(240, 320), origin=(0, 0))
    with pytest.raises(ValueError):
        solve_transmittance_opacity(result.gaussians, [far], [target], SurfelLiftConfig())
