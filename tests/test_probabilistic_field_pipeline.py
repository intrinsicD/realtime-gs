"""CPU contracts for the opt-in probabilistic compact-field pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import rtgs.lift.field_refit as field_refit_module
from rtgs.cli import _field_lift_config
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.field_inputs import SceneFits
from rtgs.lift.fiber_correspondence import FiberFitConfig
from rtgs.lift.field_lifter import (
    FieldAssociationConfig,
    FieldLiftConfig,
    FieldLifter,
    _place,
)
from rtgs.lift.field_refit import FieldRefitConfig
from rtgs.lift.probabilistic_pipeline import (
    ProbabilisticFieldPipelineConfig,
    _scene_with_training_half,
    run_probabilistic_field_pipeline,
)
from rtgs.pipeline import run_probabilistic_field_pipeline as run_public_pipeline
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


def _observation(
    gaussians: Gaussians3D,
    camera: Camera,
    view_id: str,
    colors: torch.Tensor,
) -> GaussianObservationField:
    projection = project_gaussians_ewa(gaussians, camera)
    intrinsic = projection.covariances2d - EWA_DILATION * torch.eye(
        2,
        dtype=projection.covariances2d.dtype,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(intrinsic)
    axis = eigenvectors[:, :, 0]
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=projection.means2d.float(),
        log_scales=0.5 * eigenvalues.float().log(),
        rotations=torch.atan2(axis[:, 1], axis[:, 0]).float(),
        colors=colors.float(),
        amplitudes=gaussians.opacity.float(),
        blend_mode="normalized",
        aa_dilation=EWA_DILATION,
        sigma_cutoff=3.0,
        support_fade_alpha=1.0,
        view_id=view_id,
        n_init=gaussians.n,
        provider="synthetic_fixture",
    )


def _fits(*, probability_masks: bool = False) -> SceneFits:
    dtype = torch.float64
    means = torch.tensor([[-0.30, 0.0, 0.0], [0.35, 0.05, 0.05]], dtype=dtype)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([0.012, 0.010, 0.009], dtype=dtype)),
            torch.diag(torch.tensor([0.010, 0.014, 0.011], dtype=dtype)),
        ]
    )
    colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=dtype)
    opacity = torch.tensor([0.8, 0.7], dtype=dtype)
    gaussians = Gaussians3D.from_means_covs(means, covariances, colors, opacity)
    camera_positions = (
        (1.4, 0.2, 1.5),
        (-1.4, 0.3, 1.6),
        (0.1, -1.5, 1.5),
        (0.2, 1.5, 1.4),
        (1.0, -1.0, 1.7),
    )
    cameras = tuple(
        Camera.look_at(
            torch.tensor(position),
            torch.zeros(3),
            width=40,
            height=40,
        )
        for position in camera_positions
    )
    names = tuple(f"v{index}" for index in range(len(cameras)))
    observations = tuple(
        _observation(gaussians, camera, name, colors)
        for camera, name in zip(cameras, names, strict=True)
    )
    alphas: list[torch.Tensor | None] = [None] * len(cameras)
    if probability_masks:
        alphas = []
        for observation in observations:
            alpha = torch.zeros((observation.height, observation.width), dtype=torch.float32)
            xy = observation.native_means()
            alpha[int(xy[0, 1].floor()), int(xy[0, 0].floor())] = 0.25
            alpha[int(xy[1, 1].floor()), int(xy[1, 0].floor())] = 0.75
            alphas.append(alpha)
    return SceneFits(
        observations=observations,
        cameras=cameras,
        view_names=names,
        alphas=tuple(alphas),
        train_view_indices=(0, 1, 2, 3),
        heldout_view_indices=(4,),
        bounds_hint=(torch.zeros(3), 2.0),
        geometry_is_train_only=False,
        name="probabilistic-field-tiny",
    )


def _config() -> FieldLiftConfig:
    return FieldLiftConfig(
        placement_mode="fixed_bounded_midpoint",
        compute_dtype="float64",
        max_tracks=2,
        max_train_views=4,
        depth_samples=2,
        candidate_multiplier=1,
        sweep_anchor_pool_multiplier=1,
        sweep_rounds=1,
        min_views=1,
        background_fraction=0.0,
        topology_rounds=0,
        validation_sample_cap=8,
        refit=FieldRefitConfig(iterations=0, appearance_start=0, chunk_size=8),
    )


def test_probability_masks_scale_support_mass_but_not_render_opacity() -> None:
    fits = _fits(probability_masks=True)
    probability = _place(fits, fits.train_view_indices, replace(_config(), mask_mode="probability"))
    hard = _place(fits, fits.train_view_indices, replace(_config(), mask_mode="hard"))
    unmasked = _place(fits, fits.train_view_indices, replace(_config(), mask_mode="none"))

    for row, (view, component) in enumerate(
        zip(
            probability.source_global_view_indices.tolist(),
            probability.fiber.source_component_indices.tolist(),
            strict=True,
        )
    ):
        amplitude = fits.observations[view].amplitudes[component].to(probability.field_masses.dtype)
        assert torch.allclose(
            probability.field_masses[row],
            amplitude * probability.source_support[row],
        )
    assert bool((probability.source_support < 1).all())
    assert torch.equal(probability.render_opacity, hard.render_opacity)
    assert bool((hard.source_support == 1).all())
    assert bool((unmasked.source_support == 1).all())
    assert torch.equal(unmasked.field_masses, hard.field_masses)


def test_target_component_cap_is_opt_in_deterministic_and_audited() -> None:
    fits = _fits()
    uncapped = FieldLifter(replace(_config(), max_tracks=1)).fit(fits)
    config = replace(_config(), max_tracks=1, target_component_cap=1)
    first = FieldLifter(config).fit(fits)
    second = FieldLifter(config).fit(fits)

    assert uncapped.diagnostics["target_component_cap"] is None
    assert uncapped.diagnostics["target_component_counts_used"] == [2] * fits.n_views
    assert first.diagnostics["target_component_cap"] == 1
    assert first.diagnostics["target_component_counts_original"] == [2] * fits.n_views
    assert first.diagnostics["target_component_counts_used"] == [1] * fits.n_views
    assert first.diagnostics["target_component_selection_rule"] == (
        "8x8-stratified-then-global-mass-area-v1"
    )
    assert (
        first.diagnostics["target_component_selection_sha256"]
        == second.diagnostics["target_component_selection_sha256"]
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_target_component_cap_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="target_component_cap"):
        replace(_config(), target_component_cap=value)


def test_transport_failure_rolls_back_the_whole_optional_stage() -> None:
    association = FieldAssociationConfig(
        failure_policy="rollback",
        fit=FiberFitConfig(
            temperatures=(1.0,),
            residual_variances=(0.0,),
            geometry_steps=1,
            assignment="unbalanced_sinkhorn",
            max_pair_cost=1e-12,
            min_real_mass=1e-10,
            sinkhorn_iterations=4,
        ),
    )
    result = FieldLifter(replace(_config(), association=association)).fit(_fits())

    assert result.association is None
    assert result.association_failure is not None
    assert result.diagnostics["association_status"] == "rolled_back"
    assert result.refit.source_projection_max_error <= 2e-9


def test_field_transport_requires_an_explicit_finite_projection_gate() -> None:
    with pytest.raises(ValueError, match="max_pair_cost"):
        FieldAssociationConfig(fit=FiberFitConfig())


def test_projection_nonlinearity_selects_a_depth_split_candidate() -> None:
    config = replace(
        _config(),
        topology_rounds=1,
        topology_split_mode="projection_nonlinearity",
        topology_split_min_score=0.0,
    )
    result = FieldLifter(config).fit(_fits())

    assert any(
        receipt.proposal.tag == "projection-nonlinearity-ray-depth"
        for receipt in result.topology_receipts
    )
    assert result.diagnostics["topology_split_mode"] == "projection_nonlinearity"


def test_progressive_schedule_ends_with_full_view_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gain_views: list[Camera] = []
    original_gain = field_refit_module._gain_for_view

    def recorded_gain(**kwargs):
        gain_views.append(kwargs["camera"])
        return original_gain(**kwargs)

    monkeypatch.setattr(field_refit_module, "_gain_for_view", recorded_gain)
    refit = FieldRefitConfig(
        iterations=4,
        appearance_start=4,
        learning_rate=0.005,
        visibility_refresh=1,
        chunk_size=8,
        view_schedule="progressive",
        progressive_start_views=2,
        full_view_cleanup_iterations=2,
    )
    result = FieldLifter(replace(_config(), refit=refit)).fit(_fits())

    assert result.refit.active_view_counts[0] == 2
    assert result.refit.active_view_counts[-2:] == (4, 4)
    assert sorted(result.refit.view_order) == [0, 1, 2, 3]
    assert len(result.refit.elapsed_seconds) == len(result.refit.objective_history)
    assert result.refit.elapsed_seconds[0] == 0.0
    assert all(
        right >= left
        for left, right in zip(
            result.refit.elapsed_seconds,
            result.refit.elapsed_seconds[1:],
        )
    )
    assert len(gain_views) == sum(result.refit.active_view_counts[1:])


def test_public_pipeline_keeps_independent_training_halves_disjoint() -> None:
    config = ProbabilisticFieldPipelineConfig(
        lift=_config(),
        independent_half_validation=True,
        minimum_half_views=2,
        match_radius_fraction=0.5,
    )
    result = run_public_pipeline(_fits(), config)

    assert result.half_reconstructions is not None
    assert result.stability is not None
    first, second = result.half_reconstructions
    assert first.optimized_view_indices == (0, 2)
    assert second.optimized_view_indices == (1, 3)
    assert set(first.optimized_view_indices).isdisjoint(second.optimized_view_indices)
    assert 4 not in first.optimized_view_indices
    assert 4 not in second.optimized_view_indices
    assert result.stability.first_train_views == (0, 2)
    assert result.stability.second_train_views == (1, 3)


def test_independent_half_pipeline_fails_closed_on_realized_half_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            SimpleNamespace(
                optimized_view_indices=(0, 1, 2, 3),
                heldout_view_indices=(4,),
            ),
            SimpleNamespace(
                optimized_view_indices=(0, 2),
                heldout_view_indices=(1, 3, 4),
            ),
            SimpleNamespace(
                optimized_view_indices=(0, 1, 3),
                heldout_view_indices=(0, 2, 4),
            ),
        )
    )
    monkeypatch.setattr(FieldLifter, "fit", lambda _self, _fits: next(results))
    config = ProbabilisticFieldPipelineConfig(
        lift=_config(),
        independent_half_validation=True,
        minimum_half_views=2,
    )

    with pytest.raises(RuntimeError, match="violated its camera partition"):
        run_probabilistic_field_pipeline(_fits(), config)


def test_independent_half_drops_geometry_without_half_local_provenance() -> None:
    fits = _fits()
    priors = tuple(torch.ones(field.n) for field in fits.observations)
    confidences = tuple(torch.ones(field.n) for field in fits.observations)
    with_priors = replace(
        fits,
        depth_priors=priors,
        depth_confidences=confidences,
    )

    half = _scene_with_training_half(with_priors, (0, 2))

    assert half.depth_priors is None
    assert half.depth_confidences is None
    assert half.bounds_hint is None
    assert half.points is None
    assert half.neighbors is None


def test_nested_cli_config_parses_transport_and_progressive_controls() -> None:
    config = _field_lift_config(
        json.dumps(
            {
                "mask_mode": "probability",
                "association": {
                    "failure_policy": "rollback",
                    "fit": {"max_pair_cost": 4.0, "geometry_steps": 1},
                },
                "refit": {
                    "iterations": 2,
                    "appearance_start": 2,
                    "view_schedule": "progressive",
                    "full_view_cleanup_iterations": 1,
                },
            }
        )
    )

    assert config.mask_mode == "probability"
    assert config.association is not None
    assert config.association.failure_policy == "rollback"
    assert config.association.fit.max_pair_cost == 4.0
    assert config.refit.view_schedule == "progressive"


def test_direct_wrapper_matches_public_entry_without_half_validation() -> None:
    config = ProbabilisticFieldPipelineConfig(lift=_config())
    direct = run_probabilistic_field_pipeline(_fits(), config)
    public = run_public_pipeline(_fits(), config)

    assert torch.equal(
        direct.reconstruction.gaussians.means,
        public.reconstruction.gaussians.means,
    )
    assert direct.stability is None
    assert public.stability is None
