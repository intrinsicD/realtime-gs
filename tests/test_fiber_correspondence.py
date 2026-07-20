"""CPU tests for soft correspondence over exact inverse-projection fibers."""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.lift import fiber_correspondence as correspondence
from rtgs.lift.fiber_correspondence import (
    FiberFitConfig,
    ObservationGaussians,
    exponential_schedule,
    fit_fiber_correspondence,
    pairwise_bhattacharyya_cost,
    row_softmax_plan,
    unbalanced_sinkhorn_plan,
    validate_fiber_state,
)
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber


def _field(*, amplitude: float = 17.0) -> GaussianObservationField:
    return GaussianObservationField(
        width=128,
        height=96,
        means=torch.tensor([[41.25, 32.75]], dtype=torch.float32),
        log_scales=torch.tensor([[math.log(2.0), math.log(3.0)]], dtype=torch.float32),
        rotations=torch.tensor([math.pi / 4], dtype=torch.float32),
        colors=torch.tensor([[1.4, -0.2, 0.7]], dtype=torch.float32),
        amplitudes=torch.tensor([amplitude], dtype=torch.float32),
        mean_residuals=torch.tensor([[0.125, -0.25]], dtype=torch.float32),
        filter_variance=torch.tensor([0.5], dtype=torch.float32),
        aa_dilation=0.25,
        fit_window=(16, 8, 64, 64),
        provider="synthetic_fixture",
    )


def _camera(*, translation: tuple[float, float, float]) -> Camera:
    return Camera(
        fx=52.0,
        fy=49.0,
        cx=16.0,
        cy=16.0,
        width=32,
        height=32,
        R=torch.eye(3),
        t=torch.tensor(translation),
    )


def _fiber_pair(
    *,
    initial_depth: float,
) -> tuple[InverseProjectionFiber, tuple[Camera, Camera]]:
    cameras = (
        _camera(translation=(0.0, 0.0, 0.0)),
        _camera(translation=(-0.2, 0.0, 0.0)),
    )
    fiber = InverseProjectionFiber(
        cameras=cameras,
        source_view_indices=torch.tensor([0], dtype=torch.long),
        source_component_indices=torch.tensor([0], dtype=torch.long),
        source_means2d=torch.tensor([[16.5, 15.25]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[4.2, 0.45], [0.45, 2.8]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([initial_depth], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=4.5,
        dilation=0.3,
    )
    return fiber, cameras


def _observation(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    capacity: float = 1.0,
    dilation: float = 0.3,
) -> ObservationGaussians:
    return ObservationGaussians(
        means=means.detach(),
        covariances=covariances.detach(),
        capacities=means.new_full((means.shape[0],), capacity),
        dilation=dilation,
    )


def _source_errors(fiber: InverseProjectionFiber) -> tuple[float, float]:
    means, covariances, _ = fiber.source_projection()
    center = (means - fiber.source_means2d).norm(dim=-1).amax()
    covariance = (
        (covariances - fiber.source_covariances2d)
        .flatten(1)
        .norm(dim=-1)
        .div(fiber.source_covariances2d.flatten(1).norm(dim=-1))
        .amax()
    )
    return float(center.detach()), float(covariance.detach())


def test_observation_adapter_uses_native_means_effective_covariance_and_explicit_capacity():
    field = _field(amplitude=17.0)
    area = ObservationGaussians.from_field(
        field,
        dtype=torch.float64,
        capacity_mode="footprint_area",
    )
    different_amplitude = ObservationGaussians.from_field(
        _field(amplitude=0.01),
        dtype=torch.float64,
        capacity_mode="footprint_area",
    )

    expected_means = field.means.double() + field.mean_residuals.double()
    variances = torch.tensor([4.75, 9.75], dtype=torch.float64)
    rotation = torch.tensor(
        [
            [math.sqrt(0.5), -math.sqrt(0.5)],
            [math.sqrt(0.5), math.sqrt(0.5)],
        ],
        dtype=torch.float64,
    )
    expected_covariance = rotation @ torch.diag(variances) @ rotation.T
    assert torch.equal(area.means, expected_means)
    assert torch.allclose(area.covariances[0], expected_covariance, atol=2e-6, rtol=0)
    assert area.capacities.item() == pytest.approx(math.sqrt(4.75 * 9.75), rel=1e-6)
    assert torch.equal(area.means, different_amplitude.means)
    assert torch.equal(area.covariances, different_amplitude.covariances)
    assert torch.equal(area.capacities, different_amplitude.capacities)
    assert area.dilation == 0.25

    explicit = ObservationGaussians.from_field(
        field,
        dtype=torch.float64,
        capacity_mode="footprint_area",
        capacities=torch.tensor([0.375]),
    )
    assert explicit.capacity_mode == "explicit"
    assert explicit.capacities.item() == pytest.approx(0.375)


def test_bhattacharyya_cost_is_symmetric_exact_at_equality_and_has_finite_gradients():
    first_means = torch.tensor([[0.0, 0.0], [1.0, -0.5]], dtype=torch.float64)
    first_covariances = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.3], [0.3, 0.7]]],
        dtype=torch.float64,
    )
    second_means = torch.tensor([[0.0, 0.0], [-0.25, 0.75]], dtype=torch.float64)
    second_covariances = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.8, -0.1], [-0.1, 1.4]]],
        dtype=torch.float64,
    )
    forward = pairwise_bhattacharyya_cost(
        first_means,
        first_covariances,
        second_means,
        second_covariances,
    )
    reverse = pairwise_bhattacharyya_cost(
        second_means,
        second_covariances,
        first_means,
        first_covariances,
    )
    assert torch.allclose(forward, reverse.T, atol=1e-12, rtol=1e-12)
    assert forward[0, 0].item() == pytest.approx(0, abs=1e-14)
    assert forward[0, 1].item() >= 0

    mean = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    covariance = torch.eye(2, dtype=torch.float64)[None].requires_grad_()
    equality = pairwise_bhattacharyya_cost(
        mean,
        covariance,
        mean.detach().clone(),
        covariance.detach().clone(),
    ).sum()
    equality.backward()
    assert torch.allclose(mean.grad, torch.zeros_like(mean), atol=1e-12, rtol=0)
    assert torch.allclose(covariance.grad, torch.zeros_like(covariance), atol=1e-12, rtol=0)

    moved_mean = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)
    moved_covariance = torch.tensor(
        [[[0.7, 0.08], [0.08, 1.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = pairwise_bhattacharyya_cost(
        moved_mean,
        moved_covariance,
        torch.zeros((1, 2), dtype=torch.float64),
        torch.eye(2, dtype=torch.float64)[None],
        residual_variance=0.2,
    ).sum()
    loss.backward()
    assert torch.isfinite(moved_mean.grad).all()
    assert torch.isfinite(moved_covariance.grad).all()
    assert moved_mean.grad.norm() > 0
    assert moved_covariance.grad.norm() > 0


def test_row_softmax_dustbin_masks_exactly_and_is_permutation_equivariant():
    cost = torch.tensor(
        [[0.1, 2.0, 0.8], [1.0, 0.2, 3.0], [0.4, 0.7, 0.3]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, False, True], [True, True, False], [False, False, False]])
    capacities = torch.tensor([0.5, 2.0, 1.25], dtype=torch.float64)
    plan = row_softmax_plan(
        cost,
        temperature=0.7,
        dustbin_cost=1.1,
        track_capacities=capacities,
        candidate_mask=mask,
    )
    assert torch.equal(plan.real_mass[~mask], torch.zeros_like(plan.real_mass[~mask]))
    assert torch.allclose(plan.track_row_mass, capacities, atol=1e-12, rtol=1e-12)
    assert torch.equal(plan.real_mass[2], torch.zeros_like(plan.real_mass[2]))
    assert plan.track_dustbin_probability[2].item() == 1.0
    assert torch.isfinite(plan.track_entropy).all()

    row_permutation = torch.tensor([2, 0, 1])
    column_permutation = torch.tensor([1, 2, 0])
    permuted = row_softmax_plan(
        cost[row_permutation][:, column_permutation],
        temperature=0.7,
        dustbin_cost=1.1,
        track_capacities=capacities[row_permutation],
        candidate_mask=mask[row_permutation][:, column_permutation],
    )
    inverse_rows = torch.argsort(row_permutation)
    inverse_columns = torch.argsort(column_permutation)
    assert torch.allclose(
        permuted.real_mass[inverse_rows][:, inverse_columns],
        plan.real_mass,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        permuted.track_dustbin_mass[inverse_rows],
        plan.track_dustbin_mass,
        atol=1e-12,
        rtol=1e-12,
    )


def test_augmented_sinkhorn_handles_unequal_counts_masks_and_capacity_scaling():
    cost = torch.tensor(
        [[0.1, 4.0], [0.3, 0.2], [3.0, 0.4]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[False, False], [True, False], [True, False]])
    track_capacity = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    observation_capacity = torch.tensor([1.25, 0.75], dtype=torch.float64)
    kwargs = {
        "temperature": 0.4,
        "marginal_penalty": math.inf,
        "dustbin_cost": 1.0,
        "iterations": 400,
        "tolerance": 1e-12,
        "candidate_mask": mask,
    }
    balanced = unbalanced_sinkhorn_plan(
        cost,
        track_capacities=track_capacity,
        observation_capacities=observation_capacity,
        **kwargs,
    )
    repeated = unbalanced_sinkhorn_plan(
        cost,
        track_capacities=track_capacity,
        observation_capacities=observation_capacity,
        **kwargs,
    )
    scaled = unbalanced_sinkhorn_plan(
        cost,
        track_capacities=7 * track_capacity,
        observation_capacities=7 * observation_capacity,
        **kwargs,
    )
    assert torch.equal(balanced.real_mass[~mask], torch.zeros_like(balanced.real_mass[~mask]))
    assert balanced.track_dustbin_mass[0] > 0
    assert balanced.observation_dustbin_mass[1] > 0
    assert torch.equal(balanced.augmented_mass, repeated.augmented_mass)
    assert torch.allclose(balanced.augmented_mass, scaled.augmented_mass, atol=1e-12, rtol=1e-12)

    total = track_capacity.sum() + observation_capacity.sum()
    expected_rows = torch.cat([track_capacity, observation_capacity.sum().reshape(1)]) / total
    expected_columns = torch.cat([observation_capacity, track_capacity.sum().reshape(1)]) / total
    assert torch.allclose(
        balanced.augmented_mass.sum(dim=1),
        expected_rows,
        atol=2e-8,
        rtol=0,
    )
    assert torch.allclose(
        balanced.augmented_mass.sum(dim=0),
        expected_columns,
        atol=2e-8,
        rtol=0,
    )
    assert torch.isfinite(balanced.track_entropy).all()
    assert torch.isfinite(balanced.observation_support).all()

    relaxed = unbalanced_sinkhorn_plan(
        cost,
        track_capacities=track_capacity,
        observation_capacities=observation_capacity,
        temperature=0.4,
        marginal_penalty=0.08,
        dustbin_cost=1.0,
        iterations=400,
        tolerance=1e-12,
        candidate_mask=mask,
    )
    balanced_error = (balanced.augmented_mass.sum(dim=1) - expected_rows).abs().amax()
    relaxed_error = (relaxed.augmented_mass.sum(dim=1) - expected_rows).abs().amax()
    assert balanced_error < 2e-8
    assert relaxed_error > balanced_error + 1e-4


def test_augmented_sinkhorn_has_one_consistent_dustbin_cost_crossover():
    """A 1x1 real match is indifferent exactly at the declared dustbin cost.

    Both unmatched routes and the dust-to-dust completion must carry the same bin cost. If the
    bottom-right completion were free, the real match would instead remain preferred until twice
    the declared cost, confounding comparisons with the row-softmax dustbin.
    """

    def support(real_cost: float) -> float:
        plan = unbalanced_sinkhorn_plan(
            torch.tensor([[real_cost]], dtype=torch.float64),
            track_capacities=torch.ones(1, dtype=torch.float64),
            observation_capacities=torch.ones(1, dtype=torch.float64),
            temperature=0.4,
            marginal_penalty=math.inf,
            dustbin_cost=4.0,
            iterations=400,
            tolerance=1e-13,
        )
        return float(plan.track_support)

    assert support(3.5) > 0.5
    assert support(4.0) == pytest.approx(0.5, abs=1e-12)
    assert support(4.5) < 0.5


def test_correspondence_plan_rejects_nonfinite_or_negative_final_mass():
    valid = dict(
        real_mass=torch.ones((1, 1), dtype=torch.float64),
        track_dustbin_mass=torch.zeros(1, dtype=torch.float64),
        observation_dustbin_mass=None,
        dustbin_dustbin_mass=None,
        track_capacities=torch.ones(1, dtype=torch.float64),
        observation_capacities=None,
        method="sentinel",
        iterations=1,
    )
    correspondence.CorrespondencePlan(**valid)
    with pytest.raises(ValueError, match="finite and non-negative"):
        correspondence.CorrespondencePlan(
            **{**valid, "real_mass": torch.tensor([[math.nan]], dtype=torch.float64)}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        correspondence.CorrespondencePlan(
            **{**valid, "track_dustbin_mass": torch.tensor([-1.0], dtype=torch.float64)}
        )


def test_official_uot_schedule_reports_small_finite_fixed_point_residual():
    plan = unbalanced_sinkhorn_plan(
        torch.tensor([[0.1, 3.0], [2.0, 0.2], [1.2, 0.7]], dtype=torch.float64),
        track_capacities=torch.tensor([1.0, 0.8, 1.2], dtype=torch.float64),
        observation_capacities=torch.tensor([0.75, 1.25], dtype=torch.float64),
        temperature=0.1,
        marginal_penalty=1.0,
        dustbin_cost=4.0,
        iterations=50,
        tolerance=0.0,
    )
    assert plan.fixed_point_residual is not None
    assert math.isfinite(plan.fixed_point_residual)
    # The frozen 50-iteration schedule is intentionally fixed. This prospective sentinel records
    # its measured convergence scale rather than pretending that it reached machine precision.
    assert plan.fixed_point_residual < 1e-3
    targets = plan.augmented_target_marginals
    assert targets is not None
    assert targets[0].sum().item() == pytest.approx(1.0)
    assert targets[1].sum().item() == pytest.approx(1.0)


def test_exponential_schedule_is_deterministic_and_endpoint_exact():
    first = exponential_schedule(8.0, 0.5, 5)
    second = exponential_schedule(8.0, 0.5, 5)
    assert first == second
    assert first[0] == 8.0
    assert first[-1] == 0.5
    assert all(left > right for left, right in zip(first, first[1:]))
    assert exponential_schedule(3.0, 2.0, 1) == (3.0,)


@pytest.mark.parametrize("geometry_steps", [1, 4])
def test_frozen_plan_fit_improves_cross_view_geometry_and_preserves_source_invariant(
    geometry_steps: int,
):
    truth, cameras = _fiber_pair(initial_depth=3.15)
    with torch.no_grad():
        truth.cross.copy_(torch.tensor([[0.12, -0.08]], dtype=torch.float64))
        truth.log_ray_scale.add_(0.15)
    truth_source = truth.source_projection()
    truth_other = truth.project(cameras[1])
    observations = (
        _observation(truth_source[0], truth_source[1]),
        _observation(truth_other.means2d, truth_other.covariances2d),
    )

    fitted, _ = _fiber_pair(initial_depth=1.65)
    initial_projection = fitted.project(cameras[1])
    initial_cost = pairwise_bhattacharyya_cost(
        initial_projection.means2d,
        initial_projection.covariances2d,
        observations[1].means,
        observations[1].covariances,
    ).item()
    result = fit_fiber_correspondence(
        fitted,
        observations,
        config=FiberFitConfig(
            temperatures=(2.0, 1.0),
            residual_variances=(0.5, 0.1),
            geometry_steps=geometry_steps,
            learning_rate=0.08,
            assignment="row_softmax",
            dustbin_cost=100.0,
            track_batch_size=1,
            max_grad_norm=20.0,
        ),
    )
    final_projection = fitted.project(cameras[1])
    final_cost = pairwise_bhattacharyya_cost(
        final_projection.means2d,
        final_projection.covariances2d,
        observations[1].means,
        observations[1].covariances,
    ).item()
    center_error, covariance_error = _source_errors(fitted)

    assert final_cost < initial_cost
    assert len(result.history) == 2
    assert result.plans[0].real_mass.item() == 0
    assert result.plans[0].track_row_mass.item() == 0
    assert result.plans[0].track_support.item() == 0
    assert result.plans[1].real_mass.item() > 0
    assert all(not plan.real_mass.requires_grad for plan in result.plans)
    assert center_error <= 1e-8
    assert covariance_error <= 1e-8
    assert result.source_center_error == pytest.approx(center_error)
    assert result.source_covariance_relative_error == pytest.approx(covariance_error)


def test_uot_fit_improves_cross_view_geometry_and_preserves_source_invariant():
    truth, cameras = _fiber_pair(initial_depth=3.15)
    with torch.no_grad():
        truth.cross.copy_(torch.tensor([[0.12, -0.08]], dtype=torch.float64))
        truth.log_ray_scale.add_(0.15)
    truth_source = truth.source_projection()
    truth_other = truth.project(cameras[1])
    observations = (
        _observation(truth_source[0], truth_source[1]),
        _observation(truth_other.means2d, truth_other.covariances2d),
    )

    fitted, _ = _fiber_pair(initial_depth=1.65)
    initial_projection = fitted.project(cameras[1])
    initial_cost = pairwise_bhattacharyya_cost(
        initial_projection.means2d,
        initial_projection.covariances2d,
        observations[1].means,
        observations[1].covariances,
    ).item()
    result = fit_fiber_correspondence(
        fitted,
        observations,
        config=FiberFitConfig(
            temperatures=(2.0, 1.0),
            residual_variances=(0.5, 0.1),
            geometry_steps=4,
            learning_rate=0.08,
            assignment="unbalanced_sinkhorn",
            dustbin_cost=100.0,
            marginal_penalty=1.0,
            sinkhorn_iterations=200,
            sinkhorn_tolerance=1e-12,
            track_batch_size=1,
            max_grad_norm=20.0,
        ),
    )
    final_projection = fitted.project(cameras[1])
    final_cost = pairwise_bhattacharyya_cost(
        final_projection.means2d,
        final_projection.covariances2d,
        observations[1].means,
        observations[1].covariances,
    ).item()
    center_error, covariance_error = _source_errors(fitted)

    assert final_cost < initial_cost
    assert result.plans[0].track_row_mass.item() == 0
    assert result.plans[1].real_mass.item() > 0
    assert result.plans[1].fixed_point_residual is not None
    assert math.isfinite(result.plans[1].fixed_point_residual)
    assert center_error <= 1e-8
    assert covariance_error <= 1e-8


def test_bhattacharyya_autograd_matches_finite_differences_for_all_four_fiber_coordinates():
    truth, cameras = _fiber_pair(initial_depth=3.35)
    with torch.no_grad():
        truth.cross.copy_(torch.tensor([[0.41, -0.29]], dtype=torch.float64))
        truth.log_ray_scale.add_(0.47)
    target = truth.project(cameras[1])

    fitted, _ = _fiber_pair(initial_depth=1.72)
    with torch.no_grad():
        fitted.cross.copy_(torch.tensor([[-0.18, 0.13]], dtype=torch.float64))
        fitted.log_ray_scale.add_(-0.21)

    def objective() -> torch.Tensor:
        projected = fitted.project(cameras[1])
        return pairwise_bhattacharyya_cost(
            projected.means2d,
            projected.covariances2d,
            target.means2d.detach(),
            target.covariances2d.detach(),
            residual_variance=0.17,
        ).sum()

    objective().backward()
    coordinates = (
        (fitted.depth_logits, (0,)),
        (fitted.cross, (0, 0)),
        (fitted.cross, (0, 1)),
        (fitted.log_ray_scale, (0,)),
    )
    epsilon = 1e-6
    for parameter, index in coordinates:
        analytic = float(parameter.grad[index])
        with torch.no_grad():
            original = float(parameter[index])
            parameter[index] = original + epsilon
            plus = float(objective())
            parameter[index] = original - epsilon
            minus = float(objective())
            parameter[index] = original
        finite_difference = (plus - minus) / (2.0 * epsilon)
        scale = max(abs(analytic), abs(finite_difference), 1e-12)
        assert abs(analytic) > 1e-8
        assert abs(analytic - finite_difference) / scale <= 3e-4


def test_m_step_weights_camera_views_equally_independent_of_raw_plan_mass():
    cameras = (
        _camera(translation=(0.0, 0.0, 0.0)),
        _camera(translation=(-0.2, 0.0, 0.0)),
        _camera(translation=(0.0, -0.25, 0.0)),
    )

    def make_fiber(depth: float) -> InverseProjectionFiber:
        return InverseProjectionFiber(
            cameras=cameras,
            source_view_indices=torch.tensor([0], dtype=torch.long),
            source_component_indices=torch.tensor([0], dtype=torch.long),
            source_means2d=torch.tensor([[16.5, 15.25]], dtype=torch.float64),
            source_covariances2d=torch.tensor([[[4.2, 0.45], [0.45, 2.8]]], dtype=torch.float64),
            initial_depths=torch.tensor([depth], dtype=torch.float64),
            depth_lower=1.0,
            depth_upper=4.5,
            dilation=0.3,
        )

    truth = make_fiber(3.2)
    with torch.no_grad():
        truth.cross.copy_(torch.tensor([[0.2, -0.1]], dtype=torch.float64))
    observations = tuple(
        _observation(
            truth.project(camera).means2d.detach(),
            truth.project(camera).covariances2d.detach(),
        )
        for camera in cameras
    )

    def run(first_active_mass: float) -> tuple[float, tuple[torch.Tensor, ...]]:
        fitted = make_fiber(1.8)
        plans = (
            correspondence.CorrespondencePlan(
                real_mass=torch.zeros((1, 1), dtype=torch.float64),
                track_dustbin_mass=torch.zeros(1, dtype=torch.float64),
                observation_dustbin_mass=None,
                dustbin_dustbin_mass=None,
                track_capacities=torch.zeros(1, dtype=torch.float64),
                observation_capacities=None,
                method="sentinel",
                iterations=1,
            ),
            correspondence.CorrespondencePlan(
                real_mass=torch.tensor([[first_active_mass]], dtype=torch.float64),
                track_dustbin_mass=torch.zeros(1, dtype=torch.float64),
                observation_dustbin_mass=None,
                dustbin_dustbin_mass=None,
                track_capacities=torch.tensor([first_active_mass], dtype=torch.float64),
                observation_capacities=None,
                method="sentinel",
                iterations=1,
            ),
            correspondence.CorrespondencePlan(
                real_mass=torch.ones((1, 1), dtype=torch.float64),
                track_dustbin_mass=torch.zeros(1, dtype=torch.float64),
                observation_dustbin_mass=None,
                dustbin_dustbin_mass=None,
                track_capacities=torch.ones(1, dtype=torch.float64),
                observation_capacities=None,
                method="sentinel",
                iterations=1,
            ),
        )
        optimizer = torch.optim.SGD(fitted.parameters(), lr=0.0)
        loss, _mass = correspondence._m_step(
            fitted,
            observations,
            plans,
            optimizer,
            residual_variance=0.1,
            config=FiberFitConfig(
                temperatures=(1.0,),
                residual_variances=(0.1,),
                geometry_steps=1,
                track_batch_size=1,
            ),
        )
        gradients = tuple(parameter.grad.detach().clone() for parameter in fitted.parameters())
        return loss, gradients

    unit_loss, unit_gradients = run(1.0)
    scaled_loss, scaled_gradients = run(37.0)
    assert scaled_loss == pytest.approx(unit_loss, rel=0, abs=1e-12)
    for scaled, unit in zip(scaled_gradients, unit_gradients, strict=True):
        torch.testing.assert_close(scaled, unit, rtol=0, atol=1e-12)


def test_fit_fails_closed_when_source_exclusion_leaves_no_real_mass():
    camera = _camera(translation=(0.0, 0.0, 0.0))
    fiber = InverseProjectionFiber(
        cameras=(camera,),
        source_view_indices=torch.tensor([0], dtype=torch.long),
        source_component_indices=torch.tensor([0], dtype=torch.long),
        source_means2d=torch.tensor([[16.5, 15.25]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[4.2, 0.45], [0.45, 2.8]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([2.0], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=4.5,
        dilation=0.3,
    )
    source = fiber.source_projection()
    observation = _observation(source[0], source[1])
    with pytest.raises(RuntimeError, match="transported real mass"):
        fit_fiber_correspondence(
            fiber,
            (observation,),
            config=FiberFitConfig(
                temperatures=(1.0,),
                residual_variances=(0.0,),
                geometry_steps=1,
                assignment="row_softmax",
            ),
        )


def test_excluded_source_capacity_cannot_change_another_tracks_uot_plan():
    cameras = (
        _camera(translation=(0.0, 0.0, 0.0)),
        _camera(translation=(-0.2, 0.0, 0.0)),
    )
    fiber = InverseProjectionFiber(
        cameras=cameras,
        source_view_indices=torch.tensor([0, 1], dtype=torch.long),
        source_component_indices=torch.tensor([0, 0], dtype=torch.long),
        source_means2d=torch.tensor([[16.5, 15.25], [17.25, 15.75]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[4.2, 0.45], [0.45, 2.8]], [[3.8, -0.2], [-0.2, 3.1]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([2.0, 2.4], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=4.5,
        dilation=0.3,
    )
    observations = tuple(
        _observation(
            fiber.project(camera).means2d.detach(),
            fiber.project(camera).covariances2d.detach(),
        )
        for camera in cameras
    )
    config = FiberFitConfig(
        temperatures=(0.7,),
        residual_variances=(0.1,),
        geometry_steps=1,
        assignment="unbalanced_sinkhorn",
        dustbin_cost=4.0,
        marginal_penalty=1.0,
        sinkhorn_iterations=100,
        sinkhorn_tolerance=0.0,
    )

    baseline = correspondence._e_step(
        fiber,
        observations,
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        temperature=0.7,
        residual_variance=0.1,
        config=config,
    )
    changed = correspondence._e_step(
        fiber,
        observations,
        torch.tensor([1000.0, 1.0], dtype=torch.float64),
        temperature=0.7,
        residual_variance=0.1,
        config=config,
    )

    # Track 0 is sourced in view 0, so its capacity is absent—not forced through that view's
    # dustbin—and cannot alter track 1 or either observation marginal there.
    assert baseline[0].track_row_mass[0].item() == 0
    assert changed[0].track_row_mass[0].item() == 0
    torch.testing.assert_close(changed[0].real_mass, baseline[0].real_mass, rtol=0, atol=0)
    torch.testing.assert_close(
        changed[0].track_dustbin_mass,
        baseline[0].track_dustbin_mass,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        changed[0].observation_dustbin_mass,
        baseline[0].observation_dustbin_mass,
        rtol=0,
        atol=0,
    )


def test_fit_routes_behind_camera_cross_view_projection_to_dustbin():
    source_camera = _camera(translation=(0.0, 0.0, 0.0))
    behind_camera = _camera(translation=(0.0, 0.0, -3.0))
    fiber = InverseProjectionFiber(
        cameras=(source_camera, behind_camera),
        source_view_indices=torch.tensor([0], dtype=torch.long),
        source_component_indices=torch.tensor([0], dtype=torch.long),
        source_means2d=torch.tensor([[16.5, 15.25]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[4.2, 0.45], [0.45, 2.8]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([2.0], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=2.5,
        dilation=0.3,
    )
    source = fiber.source_projection()
    observations = (
        _observation(source[0], source[1]),
        _observation(
            torch.tensor([[16.0, 16.0]], dtype=torch.float64),
            torch.eye(2, dtype=torch.float64)[None],
        ),
    )
    with pytest.raises(RuntimeError, match="transported real mass"):
        fit_fiber_correspondence(
            fiber,
            observations,
            config=FiberFitConfig(
                temperatures=(1.0,),
                residual_variances=(0.0,),
                geometry_steps=1,
                assignment="row_softmax",
            ),
        )


def test_fiber_state_validation_rejects_depth_saturation_and_ray_variance_underflow():
    saturated, _ = _fiber_pair(initial_depth=2.0)
    with torch.no_grad():
        saturated.depth_logits.fill_(1e6)
    with pytest.raises(RuntimeError, match="depth reached a bound"):
        validate_fiber_state(saturated)

    collapsed, _ = _fiber_pair(initial_depth=2.0)
    with torch.no_grad():
        collapsed.log_ray_scale.fill_(-1e6)
    with pytest.raises(RuntimeError, match="ray variance underflowed"):
        validate_fiber_state(collapsed)
