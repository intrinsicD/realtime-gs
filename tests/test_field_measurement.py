"""CPU controls for labeled, unequal multi-view Gaussian field decompositions."""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift.field_measurement import (
    association_metrics,
    normalized_overlap_plan,
    oracle_parent_aggregate,
    project_labeled_fields,
)
from rtgs.lift.field_observability import (
    solve_projected_covariance,
    triangulate_projected_mean,
)
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


def _fixture() -> tuple[Gaussians3D, list[Camera]]:
    dtype = torch.float64
    means = torch.tensor(
        [[-0.60, 0.05, 0.1], [0.65, -0.08, -0.05]],
        dtype=dtype,
    )
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([0.018, 0.012, 0.010], dtype=dtype)),
            torch.diag(torch.tensor([0.013, 0.020, 0.009], dtype=dtype)),
        ]
    )
    colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]], dtype=dtype)
    opacity = torch.tensor([0.7, 0.5], dtype=dtype)
    gaussians = Gaussians3D.from_means_covs(means, covariances, colors, opacity)
    cameras = [
        Camera.look_at(
            torch.tensor([1.7, 0.2, 1.6]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
        Camera.look_at(
            torch.tensor([-1.5, 0.4, 1.8]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
        Camera.look_at(
            torch.tensor([0.2, -1.6, 1.7]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
    ]
    return gaussians, cameras


def test_unequal_decompositions_recover_oracle_projected_moments() -> None:
    gaussians, cameras = _fixture()
    views = project_labeled_fields(
        gaussians,
        cameras,
        split_counts=(1, 2, 3),
    )
    assert [view.field.n for view in views] == [2, 4, 6]
    assert [torch.bincount(view.parent_ids).tolist() for view in views] == [
        [1, 1],
        [2, 2],
        [3, 3],
    ]

    for camera, view in zip(cameras, views, strict=True):
        aggregate = oracle_parent_aggregate(view)
        projection = project_gaussians_ewa(gaussians, camera)
        assert torch.allclose(aggregate.means, projection.means2d, atol=2e-6, rtol=2e-6)
        assert torch.allclose(
            aggregate.covariances,
            projection.covariances2d,
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(aggregate.density_amplitudes, gaussians.opacity)


def test_oracle_split_aggregate_directly_lifts_mean_and_covariance() -> None:
    gaussians, cameras = _fixture()
    views = project_labeled_fields(
        gaussians,
        cameras,
        split_counts=(1, 2, 3),
    )
    aggregates = [oracle_parent_aggregate(view) for view in views]

    recovered_means: list[torch.Tensor] = []
    recovered_covariances: list[torch.Tensor] = []
    for parent in range(gaussians.n):
        projected_means = torch.stack([aggregate.means[parent] for aggregate in aggregates])
        mean_result = triangulate_projected_mean(
            projected_means,
            cameras,
            rcond=1e-10,
        )
        projected_covariances = torch.stack(
            [aggregate.covariances[parent] for aggregate in aggregates]
        )
        covariance_result = solve_projected_covariance(
            mean_result.mean,
            cameras,
            projected_covariances,
            dilation=EWA_DILATION,
            minimum_eigenvalue=1e-12,
            rcond=1e-10,
        )
        recovered_means.append(mean_result.mean)
        recovered_covariances.append(covariance_result.covariance)

    assert torch.allclose(
        torch.stack(recovered_means),
        gaussians.means,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        torch.stack(recovered_covariances),
        gaussians.covariance(),
        atol=2e-6,
        rtol=2e-6,
    )


def test_overlap_plan_has_perfect_parent_purity_for_separated_fixture() -> None:
    gaussians, cameras = _fixture()
    views = project_labeled_fields(
        gaussians,
        cameras,
        split_counts=(1, 3, 2),
        split_fraction=0.05,
    )
    predicted = oracle_parent_aggregate(views[1])
    predicted_ids = torch.arange(predicted.n, dtype=torch.long)
    plan = normalized_overlap_plan(predicted, views[1].field)
    metrics = association_metrics(plan, predicted_ids, views[1].parent_ids)
    assert metrics.purity > 0.99
    assert metrics.completeness > 0.99


def test_labeled_harness_is_deterministic() -> None:
    gaussians, cameras = _fixture()
    left = project_labeled_fields(gaussians, cameras, split_counts=(3, 1, 2))
    right = project_labeled_fields(gaussians, cameras, split_counts=(3, 1, 2))
    for first, second in zip(left, right, strict=True):
        assert torch.equal(first.parent_ids, second.parent_ids)
        assert torch.equal(first.field.means, second.field.means)
        assert torch.equal(first.field.covariances, second.field.covariances)
        assert torch.equal(
            first.field.density_amplitudes,
            second.field.density_amplitudes,
        )
