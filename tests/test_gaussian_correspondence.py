"""CPU tests for network-independent G²SR-style correspondence geometry."""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.lift.gaussian_correspondence import (
    five_sigma_points,
    forward_backward_consistency,
    sample_flow_bilinear,
    solve_local_affine,
    track_gaussians_2d,
    triangulate_centers_dlt,
    warp_gaussians_2d,
)


def _affine_flow(
    height: int,
    width: int,
    linear: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Return flow whose warp is ``linear @ uv + offset`` on the pixel-center lattice."""
    ys = torch.arange(height, dtype=linear.dtype) + 0.5
    xs = torch.arange(width, dtype=linear.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx, yy], dim=-1)
    warped = torch.einsum("ij,hwj->hwi", linear, points) + offset
    return warped - points


def _parallel_camera(center_x: float, *, width: int = 96, height: int = 72) -> Camera:
    return Camera(
        fx=80.0,
        fy=78.0,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        R=torch.eye(3),
        t=torch.tensor([-center_x, 0.0, 0.0]),
    )


def test_five_sigma_points_follow_principal_covariance_axes_in_batches():
    means = torch.tensor([[10.0, 20.0], [4.0, 7.0]], dtype=torch.float64)
    angle = torch.tensor(0.4, dtype=torch.float64)
    rotation = torch.tensor(
        [[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]],
        dtype=torch.float64,
    )
    scales = torch.tensor([3.0, 1.25], dtype=torch.float64)
    covariance = rotation @ torch.diag(scales.square()) @ rotation.T
    covariances = torch.stack([covariance, torch.diag(torch.tensor([4.0, 1.0]))])

    result = five_sigma_points(means, covariances)

    assert result.points.shape == (2, 5, 2)
    assert bool(result.valid.all())
    assert torch.equal(result.points[:, 0], means)
    offsets = result.points[:, 1:] - means[:, None]
    assert torch.allclose(offsets[:, 0], -offsets[:, 1], atol=1e-12, rtol=0)
    assert torch.allclose(offsets[:, 2], -offsets[:, 3], atol=1e-12, rtol=0)
    recovered = result.factor @ result.factor.transpose(-1, -2)
    assert torch.allclose(recovered, covariances, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        offsets[:, 0].norm(dim=-1),
        math.sqrt(2.0) * result.eigenvalues[:, 0].sqrt(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_sigma_points_fail_closed_for_nonfinite_asymmetric_or_singular_covariance():
    means = torch.tensor([[2.0, 2.0]] * 4)
    covariances = torch.tensor(
        [
            [[2.0, 0.2], [0.2, 1.0]],
            [[2.0, 0.5], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[float("nan"), 0.0], [0.0, 1.0]],
        ]
    )

    result = five_sigma_points(means, covariances)

    assert torch.equal(result.valid, torch.tensor([True, False, False, False]))
    assert torch.isfinite(result.points[0]).all()
    assert torch.isnan(result.points[1:]).all()


def test_bilinear_flow_sampling_uses_half_pixel_centers_and_rejects_bad_samples():
    flow = torch.zeros(3, 4, 2, dtype=torch.float64)
    ys = torch.arange(3, dtype=torch.float64) + 0.5
    xs = torch.arange(4, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    flow[..., 0] = 2.0 * xx + 3.0 * yy
    flow[..., 1] = -xx + 0.5 * yy
    points = torch.tensor(
        [[0.5, 0.5], [2.0, 1.25], [3.5, 2.5], [0.49, 1.5], [float("nan"), 1.0]],
        dtype=torch.float64,
    )

    sampled = sample_flow_bilinear(flow, points)

    expected = torch.stack(
        [2.0 * points[:3, 0] + 3.0 * points[:3, 1], -points[:3, 0] + 0.5 * points[:3, 1]],
        dim=-1,
    )
    assert torch.allclose(sampled.values[:3], expected, atol=1e-12, rtol=0)
    assert torch.equal(sampled.valid, torch.tensor([True, True, True, False, False]))
    assert torch.equal(sampled.values[3:], torch.zeros(2, 2, dtype=torch.float64))

    flow[1, 1] = torch.nan
    poisoned = sample_flow_bilinear(flow, torch.tensor([[1.5, 1.5]], dtype=torch.float64))
    assert not bool(poisoned.valid[0])
    assert torch.equal(poisoned.values, torch.zeros_like(poisoned.values))


def test_affine_estimate_and_gaussian_warp_match_exact_known_transform():
    means = torch.tensor([[10.0, 12.0], [20.0, 16.0]], dtype=torch.float64)
    covariances = torch.tensor(
        [[[4.0, 0.5], [0.5, 2.0]], [[2.0, -0.2], [-0.2, 3.0]]],
        dtype=torch.float64,
    )
    sigma = five_sigma_points(means, covariances)
    linear = torch.tensor(
        [[[1.1, 0.2], [-0.1, 0.9]], [[0.85, -0.15], [0.25, 1.05]]],
        dtype=torch.float64,
    )
    offsets = torch.tensor([[3.0, -2.0], [-1.0, 4.0]], dtype=torch.float64)
    targets = torch.einsum("bij,bpj->bpi", linear, sigma.points) + offsets[:, None]

    estimate = solve_local_affine(sigma.points, targets)
    expected_translation = torch.einsum("bij,bj->bi", linear, means) + offsets - means
    warped_means, warped_covariances = warp_gaussians_2d(
        means,
        covariances,
        estimate.linear,
        estimate.translation,
    )

    assert bool(estimate.valid.all())
    assert torch.equal(estimate.rank, torch.tensor([2, 2]))
    assert torch.allclose(estimate.linear, linear, atol=1e-12, rtol=1e-12)
    assert torch.allclose(estimate.translation, expected_translation, atol=1e-12, rtol=1e-12)
    assert torch.all(estimate.rms_residual < 1e-12)
    assert torch.allclose(
        warped_means,
        torch.einsum("bij,bj->bi", linear, means) + offsets,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_covariances = linear @ covariances @ linear.transpose(-1, -2)
    assert torch.allclose(warped_covariances, expected_covariances, atol=1e-12, rtol=1e-12)


def test_affine_solver_rejects_collinear_or_insufficient_valid_support():
    reference = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
        ]
    )
    target = reference + 1.0
    valid_samples = torch.tensor([[True] * 4, [True, True, False, False]])

    estimate = solve_local_affine(reference, target, sample_valid=valid_samples)

    assert torch.equal(estimate.valid, torch.tensor([False, False]))
    assert torch.equal(estimate.rank, torch.tensor([1, 1]))
    assert torch.equal(estimate.linear, torch.zeros_like(estimate.linear))


def test_forward_backward_consistency_accepts_roundtrip_and_rejects_error_or_oob():
    identity = torch.eye(2)
    forward = _affine_flow(24, 24, identity, torch.tensor([2.0, -1.0]))
    backward = _affine_flow(24, 24, identity, torch.tensor([-2.0, 1.0]))
    points = torch.tensor([[8.5, 8.5], [12.5, 10.5], [23.5, 12.5]])

    accepted = forward_backward_consistency(points, forward, backward, max_error=0.01)
    assert torch.equal(accepted.valid, torch.tensor([True, True, False]))
    assert torch.allclose(accepted.roundtrip_error[:2], torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.isinf(accepted.roundtrip_error[2])

    inconsistent_backward = backward.clone()
    inconsistent_backward[..., 0] += 0.25
    rejected = forward_backward_consistency(
        points[:2],
        forward,
        inconsistent_backward,
        max_error=0.1,
    )
    assert not bool(rejected.valid.any())
    assert torch.allclose(rejected.roundtrip_error, torch.full((2,), 0.25), atol=1e-6, rtol=0)


def test_end_to_end_flow_tracking_recovers_affine_mean_and_covariance():
    linear = torch.tensor([[1.02, 0.04], [-0.03, 0.97]], dtype=torch.float64)
    offset = torch.tensor([1.5, 0.75], dtype=torch.float64)
    inverse = torch.linalg.inv(linear)
    inverse_offset = -(inverse @ offset)
    forward = _affine_flow(48, 56, linear, offset)
    backward = _affine_flow(48, 56, inverse, inverse_offset)
    means = torch.tensor([[18.0, 16.0], [34.0, 28.0]], dtype=torch.float64)
    covariances = torch.tensor(
        [[[3.0, 0.3], [0.3, 1.4]], [[2.2, -0.4], [-0.4, 2.8]]],
        dtype=torch.float64,
    )

    tracked = track_gaussians_2d(
        means,
        covariances,
        forward,
        backward,
        max_roundtrip_error=1e-9,
    )

    assert bool(tracked.valid.all())
    assert torch.allclose(tracked.affine.linear, linear.expand(2, -1, -1), atol=1e-12, rtol=1e-12)
    expected_means = torch.einsum("ij,bj->bi", linear, means) + offset
    expected_covariances = linear @ covariances @ linear.T
    assert torch.allclose(tracked.means, expected_means, atol=1e-12, rtol=1e-12)
    assert torch.allclose(tracked.covariances, expected_covariances, atol=1e-12, rtol=1e-12)


def test_end_to_end_identity_flow_preserves_gaussians_exactly():
    flow = torch.zeros(32, 32, 2, dtype=torch.float64)
    means = torch.tensor([[8.5, 10.5], [20.5, 18.5]], dtype=torch.float64)
    covariances = torch.tensor(
        [[[2.0, 0.25], [0.25, 1.0]], [[1.5, -0.1], [-0.1, 2.5]]],
        dtype=torch.float64,
    )

    tracked = track_gaussians_2d(means, covariances, flow, flow, max_roundtrip_error=0.0)

    assert bool(tracked.valid.all())
    assert torch.allclose(
        tracked.affine.linear,
        torch.eye(2, dtype=torch.float64).expand(2, -1, -1),
        atol=1e-12,
        rtol=0,
    )
    assert torch.equal(tracked.means, means)
    assert torch.allclose(tracked.covariances, covariances, atol=1e-12, rtol=0)


def test_dlt_exact_batch_matches_camera_world_to_camera_convention():
    cameras = [_parallel_camera(-0.6), _parallel_camera(0.4), _parallel_camera(1.1)]
    points_world = torch.tensor(
        [[-0.4, -0.2, 3.0], [0.25, 0.35, 4.5], [0.7, -0.1, 6.0]],
        dtype=torch.float64,
    )
    image_points = torch.stack([camera.project(points_world)[0] for camera in cameras], dim=1)

    result = triangulate_centers_dlt(cameras, image_points)

    assert result.points_world.shape == (3, 3)
    assert bool(result.valid.all())
    assert torch.all(result.observation_count == 3)
    assert torch.allclose(result.points_world, points_world, atol=2e-6, rtol=2e-6)
    assert torch.all(result.depths > 0)
    assert bool(result.cheiral.all())
    assert torch.all(result.max_observed_reprojection_error < 1e-5)
    assert torch.all(result.rank >= 3)
    assert torch.all(result.condition_number < 1e8)
    assert torch.all(result.nullspace_gap > 1e4)

    expected_depth = torch.stack([camera.world_to_cam(points_world)[:, 2] for camera in cameras], 1)
    assert torch.allclose(result.depths, expected_depth, atol=2e-6, rtol=2e-6)


def test_dlt_preserves_arbitrary_leading_batch_dimensions():
    cameras = [_parallel_camera(-0.7), _parallel_camera(0.6)]
    points_world = torch.tensor(
        [
            [[-0.4, -0.2, 3.0], [0.25, 0.35, 4.5], [0.7, -0.1, 6.0]],
            [[0.1, 0.15, 2.5], [-0.6, 0.2, 5.0], [0.4, -0.3, 3.5]],
        ],
        dtype=torch.float64,
    )
    image_points = torch.stack(
        [camera.project(points_world.reshape(-1, 3))[0].reshape(2, 3, 2) for camera in cameras],
        dim=-2,
    )

    result = triangulate_centers_dlt(cameras, image_points)

    assert result.points_world.shape == (2, 3, 3)
    assert result.valid.shape == (2, 3)
    assert result.depths.shape == (2, 3, 2)
    assert bool(result.valid.all())
    assert torch.allclose(result.points_world, points_world, atol=2e-6, rtol=2e-6)


def test_dlt_noisy_multiview_is_stable_and_reports_reprojection_error():
    cameras = [
        Camera.look_at(
            torch.tensor([2.0, 0.1, 0.0]),
            torch.zeros(3),
            width=96,
            height=72,
            fov_x_deg=55.0,
        ),
        Camera.look_at(
            torch.tensor([-1.0, 0.2, 1.8]),
            torch.zeros(3),
            width=96,
            height=72,
            fov_x_deg=55.0,
        ),
        Camera.look_at(
            torch.tensor([-0.8, -0.2, -1.7]),
            torch.zeros(3),
            width=96,
            height=72,
            fov_x_deg=55.0,
        ),
    ]
    points_world = torch.tensor([[0.1, -0.1, 0.2], [-0.25, 0.2, -0.15]], dtype=torch.float64)
    image_points = torch.stack([camera.project(points_world)[0] for camera in cameras], dim=1)
    noise = torch.tensor(
        [
            [[0.10, -0.08], [-0.06, 0.04], [0.02, 0.07]],
            [[-0.05, 0.09], [0.08, -0.03], [-0.04, -0.06]],
        ],
        dtype=torch.float64,
    )

    result = triangulate_centers_dlt(
        cameras,
        image_points + noise,
        max_reprojection_error=0.25,
    )

    assert bool(result.valid.all())
    assert torch.all((result.points_world - points_world).norm(dim=-1) < 0.01)
    assert torch.all(result.max_observed_reprojection_error > 0)
    assert torch.all(result.max_observed_reprojection_error < 0.25)
    assert torch.all(torch.isfinite(result.nullspace_gap))


def test_dlt_masks_invalid_views_and_rejects_oob_degenerate_or_behind_camera():
    cameras = [_parallel_camera(-0.5), _parallel_camera(0.5), _parallel_camera(1.0)]
    point = torch.tensor([[0.0, 0.0, 4.0]])
    image_points = torch.stack([camera.project(point)[0] for camera in cameras], dim=1)
    with_bad_view = image_points.clone()
    with_bad_view[:, 2] = torch.tensor([float("nan"), 0.0])

    recovered = triangulate_centers_dlt(cameras, with_bad_view)
    assert bool(recovered.valid[0])
    assert recovered.observation_count[0] == 2
    assert not bool(recovered.observation_valid[0, 2])
    assert torch.allclose(recovered.points_world, point, atol=1e-5, rtol=1e-5)

    oob = image_points[:, :2].clone()
    oob[:, 1, 0] = -10.0
    oob_result = triangulate_centers_dlt(cameras[:2], oob)
    assert not bool(oob_result.valid[0])
    assert oob_result.observation_count[0] == 1

    same_camera = [cameras[0], cameras[0]]
    repeated = image_points[:, :1].expand(-1, 2, -1).clone()
    degenerate = triangulate_centers_dlt(same_camera, repeated)
    assert not bool(degenerate.valid[0])
    assert degenerate.rank[0] < 3 or torch.isinf(degenerate.condition_number[0])

    behind = torch.tensor([[0.0, 0.0, -3.0]])
    behind_pixels = torch.stack([camera.project(behind)[0] for camera in cameras[:2]], dim=1)
    behind_result = triangulate_centers_dlt(cameras[:2], behind_pixels)
    assert not bool(behind_result.valid[0])
    assert not bool(behind_result.cheiral[0].any())


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: sample_flow_bilinear(torch.zeros(4, 4, 2), torch.zeros(3, 3)),
            "points must end",
        ),
        (
            lambda: solve_local_affine(torch.zeros(2, 2), torch.zeros(2, 2)),
            "at least three",
        ),
        (
            lambda: triangulate_centers_dlt(
                [_parallel_camera(-0.5), _parallel_camera(0.5)],
                torch.zeros(2, 2),
                min_views=3,
            ),
            "min_views",
        ),
    ],
)
def test_correspondence_api_rejects_malformed_inputs(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()
