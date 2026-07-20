"""Lifting variants: shared geometry, depth, gradient, carve, baselines, merging."""

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift import get_lifter, lifter_names
from rtgs.lift.base import (
    bilinear_sample,
    depth_map_gradients,
    footprint_sigma_ray,
    lift_covariance,
    lift_view_at_depth,
    minor_axis_sigma_ray,
    ray_basis,
)
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.gradient import (
    _build_photometric_keep_masks,
    _resolve_anchor_weights,
    _validate_position_pairs,
    bounded_ray_anchor_loss,
    world_position_consistency_loss,
)
from rtgs.lift.merge import merge_by_voxel
from rtgs.lift.surface import OrientedPointTargets
from rtgs.optim.trainer import Trainer


def test_lifter_registry():
    assert set(lifter_names()) == {
        "gradient",
        "depth",
        "hybrid",
        "carve",
        "field",
        "sfm",
        "random",
    }


def test_bilinear_sample_exact_at_centers():
    grid = torch.arange(12.0).reshape(3, 4)
    uv = torch.tensor([[1.5, 0.5], [2.5, 2.5]])  # centers of (0,1) and (2,2)
    vals = bilinear_sample(grid, uv)
    assert torch.allclose(vals, torch.tensor([1.0, 10.0]))


def test_ray_basis_orthonormal(tiny_scene):
    cam = tiny_scene.cameras[0]
    uv = torch.rand(40, 2) * 32
    basis = ray_basis(cam, uv)
    eye = torch.eye(3).expand(40, 3, 3)
    assert torch.allclose(basis @ basis.transpose(-1, -2), eye, atol=1e-5)
    # Third column is the pixel ray (in camera coords); rotated to world it must match
    # the normalized pixel_rays direction.
    _, d = cam.pixel_rays(uv)
    d_unit = torch.nn.functional.normalize(d, dim=-1)
    ray_world = basis[:, :, 2] @ cam.R  # row-vector form of R^T @ ray
    assert torch.allclose(ray_world, d_unit, atol=1e-5)


def test_lift_covariance_scales_with_depth(tiny_scene):
    cam = tiny_scene.cameras[0]
    uv = torch.tensor([[16.0, 16.0]])
    cov2d = torch.tensor([[[4.0, 0.5], [0.5, 2.0]]])
    c1 = lift_covariance(cam, uv, cov2d, torch.tensor([1.0]), torch.tensor([0.01]))
    c2 = lift_covariance(cam, uv, cov2d, torch.tensor([2.0]), torch.tensor([0.02]))
    # Doubling depth (and sigma_ray) quadruples the covariance.
    assert torch.allclose(c2, 4.0 * c1, rtol=1e-4, atol=1e-8)
    evals = torch.linalg.eigvalsh(c1)
    assert (evals > 0).all()


def test_lift_covariance_reprojects_off_axis(tiny_scene):
    cam = tiny_scene.cameras[0]
    uv = torch.tensor([[4.5, 27.5]])
    cov2d = torch.tensor([[[5.0, 1.2], [1.2, 2.0]]])
    depth = torch.tensor([2.0])
    cov_world = lift_covariance(cam, uv, cov2d, depth, torch.tensor([0.03]))
    point = cam.unproject(uv, depth)
    point_cam = cam.world_to_cam(point)
    x, y, z = point_cam[0]
    jac = torch.tensor(
        [[[cam.fx / z, 0.0, -cam.fx * x / z**2], [0.0, cam.fy / z, -cam.fy * y / z**2]]]
    )
    cov_cam = cam.R @ cov_world @ cam.R.T
    projected = jac @ cov_cam @ jac.transpose(-1, -2)
    assert torch.allclose(projected, cov2d, atol=1e-4, rtol=1e-4)


def test_lift_view_at_depth_positions(tiny_scene):
    """Lifted means must land exactly on unproject(xy, depth)."""
    cam = tiny_scene.cameras[0]
    n = 15
    g2d = Gaussians2D(
        xy=torch.rand(n, 2) * 30 + 1,
        chol=torch.stack([torch.rand(n) + 0.8, torch.randn(n) * 0.2, torch.rand(n) + 0.8], -1),
        color=torch.rand(n, 3),
        weight=torch.rand(n) * 0.9 + 0.05,
    )
    depth = torch.rand(n) + 1.5
    g3d = lift_view_at_depth(cam, g2d, depth, torch.full((n,), 0.05))
    expected = cam.unproject(g2d.xy, depth)
    assert torch.allclose(g3d.means, expected, atol=1e-5)
    uv, z = cam.project(g3d.means)
    assert torch.allclose(uv, g2d.xy, atol=1e-3)
    assert torch.allclose(z, depth, atol=1e-4)


def test_depth_lifter_recovers_geometry(tiny_scene, tiny_fits):
    """With GT depth, lifted gaussians must sit near the true surfaces."""
    g2ds, _ = tiny_fits
    lifter = get_lifter("depth")
    g3d = lifter.lift(g2ds, tiny_scene)
    assert g3d.n > 100
    # Distance from each lifted gaussian to the nearest GT gaussian center must be small
    # for the majority (GT scales are 0.04-0.12; the scene radius is ~0.7).
    d = torch.cdist(g3d.means, tiny_scene.gt_gaussians.means).min(dim=1).values
    assert d.median() < 0.15, f"median distance to GT {d.median()}"
    # Init must render substantially better than random.
    psnr_depth = Trainer.evaluate(tiny_scene, g3d)
    psnr_random = Trainer.evaluate(tiny_scene, get_lifter("random", n=g3d.n).lift(g2ds, tiny_scene))
    assert psnr_depth > psnr_random + 2.0, (psnr_depth, psnr_random)


def test_depth_covariance_modes_are_finite_and_share_means(tiny_scene, tiny_fits):
    """Covariance ablations must isolate shape while preserving sampled geometry."""
    g2ds, _ = tiny_fits
    lifted = {}
    for mode in ("surface", "footprint", "isotropic"):
        kwargs = {"isotropic_sigma": 0.01} if mode == "isotropic" else {}
        lifted[mode] = get_lifter("depth", covariance_mode=mode, merge=False, **kwargs).lift(
            g2ds, tiny_scene
        )
    reference = lifted["surface"]
    for result in lifted.values():
        assert result.n == reference.n
        assert torch.allclose(result.means, reference.means)
        assert torch.equal(result.opacity, reference.opacity)
        assert torch.equal(result.sh, reference.sh)
        assert torch.isfinite(result.log_scales).all()
        assert (result.scales > 0).all()
    assert not torch.allclose(lifted["surface"].covariance(), lifted["footprint"].covariance())


def test_footprint_sigma_matches_minor_axis_floor_on_constant_depth(tiny_scene):
    cam = tiny_scene.cameras[0]
    g2d = Gaussians2D(
        xy=torch.tensor([[8.5, 8.5], [20.5, 18.5]]),
        chol=torch.tensor([[1.2, 0.1, 0.8], [0.7, -0.2, 1.4]]),
        color=torch.rand(2, 3),
        weight=torch.ones(2),
    )
    depth_map = torch.full((cam.height, cam.width), 2.0)
    depth = torch.full((2,), 2.0)
    assert torch.allclose(
        footprint_sigma_ray(cam, g2d, depth_map, depth),
        minor_axis_sigma_ray(cam, g2d, depth),
    )


def test_validity_aware_depth_gradients_do_not_cross_zero_background():
    depth = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
        ]
    )
    raw_x, _ = depth_map_gradients(depth, validity_aware=False)
    robust_x, robust_y = depth_map_gradients(depth)
    assert raw_x[:, 1:3].abs().max() == 1.0
    assert torch.count_nonzero(robust_x) == 0
    assert torch.count_nonzero(robust_y) == 0


def test_footprint_sigma_responds_to_planar_depth_slope(tiny_scene):
    cam = tiny_scene.cameras[0]
    g2d = Gaussians2D(
        xy=torch.tensor([[16.5, 16.5]]),
        chol=torch.tensor([[1.0, 0.0, 1.0]]),
        color=torch.rand(1, 3),
        weight=torch.ones(1),
    )
    x = torch.arange(cam.width).float()[None]
    depth_map = 2.0 + 0.02 * x.expand(cam.height, -1)
    depth = bilinear_sample(depth_map, g2d.xy)
    assert footprint_sigma_ray(cam, g2d, depth_map, depth) > minor_axis_sigma_ray(cam, g2d, depth)


def test_constant_sigma_gives_global_ray_variance(tiny_scene):
    cam = tiny_scene.cameras[0]
    g2d = Gaussians2D(
        xy=torch.tensor([[8.5, 8.5], [23.5, 20.5]]),
        chol=torch.tensor([[0.8, 0.1, 1.2], [1.4, -0.2, 0.7]]),
        color=torch.rand(2, 3),
        weight=torch.ones(2),
    )
    depth = torch.tensor([1.5, 2.5])
    sigma = torch.full((2,), 0.03)
    lifted = lift_view_at_depth(cam, g2d, depth, sigma)
    _, directions = cam.pixel_rays(g2d.xy)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    variance = torch.einsum("ni,nij,nj->n", directions, lifted.covariance(), directions)
    assert torch.allclose(variance, sigma.square(), atol=1e-6, rtol=1e-4)


def test_depth_lifter_rejects_unknown_covariance_mode():
    with pytest.raises(ValueError):
        get_lifter("depth", covariance_mode="unknown")


def test_isotropic_covariance_mode_requires_positive_sigma():
    with pytest.raises(ValueError):
        get_lifter("depth", covariance_mode="isotropic")
    with pytest.raises(ValueError):
        get_lifter("depth", covariance_mode="isotropic", isotropic_sigma=0.0)
    with pytest.raises(ValueError):
        get_lifter("depth", covariance_mode="isotropic", isotropic_sigma=float("nan"))
    with pytest.raises(ValueError):
        get_lifter("depth", covariance_mode="isotropic", isotropic_sigma=float("inf"))


def test_depth_lifter_default_is_robust_surface(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    default = get_lifter("depth", merge=False).lift(g2ds, tiny_scene)
    explicit = get_lifter(
        "depth", covariance_mode="surface", robust_depth_gradients=True, merge=False
    ).lift(g2ds, tiny_scene)
    assert torch.allclose(default.means, explicit.means)
    assert torch.allclose(default.covariance(), explicit.covariance())


def test_gradient_lifter_improves_over_iterations(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    lifter = get_lifter("gradient", iterations=60, rasterizer="torch", seed=0)
    g3d = lifter.lift(g2ds, tiny_scene)
    assert g3d.n > 100
    hist = lifter.history
    first = sum(hist[:10]) / 10
    last = sum(hist[-10:]) / 10
    assert last < first * 0.9, f"photometric loss did not decrease: {first} -> {last}"


def test_hybrid_lifter_uses_depth_prior_and_refines(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    lifter = get_lifter("hybrid", iterations=10, rasterizer="torch", seed=0)
    g3d = lifter.lift(g2ds, tiny_scene)
    assert g3d.n > 100
    distance = torch.cdist(g3d.means, tiny_scene.gt_gaussians.means).min(dim=1).values
    assert distance.median() < 0.25
    assert len(lifter.history) == 10


def test_confidence_anchor_loss_blocks_zero_weight_gradients():
    raw_t = torch.tensor([0.0, 0.0], requires_grad=True)
    raw_t_init = torch.tensor([0.0, 0.0])
    target = torch.tensor([0.25, 0.75])
    weights = torch.tensor([1.0, 0.0])
    loss = bounded_ray_anchor_loss(raw_t, raw_t_init, target, weights, "confidence", beta=0.05)
    loss.backward()
    assert raw_t.grad is not None
    assert raw_t.grad[0].abs() > 0
    assert raw_t.grad[1] == 0

    zero_weight_raw = torch.zeros(2, requires_grad=True)
    zero_loss = bounded_ray_anchor_loss(
        zero_weight_raw,
        raw_t_init,
        target,
        torch.zeros(2),
        "confidence",
        beta=0.05,
    )
    zero_loss.backward()
    assert torch.isfinite(zero_loss)
    assert torch.count_nonzero(zero_weight_raw.grad) == 0


def test_world_position_consistency_loss_matches_normalized_l1_huber_and_gradients():
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.02, -0.01, 0.0],
            [0.20, -0.10, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.4, 0.4],
        ],
        requires_grad=True,
    )
    pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    loss = world_position_consistency_loss(means, pairs, scene_extent=1.0, beta=0.05)
    expected_quadratic = 0.5 * 0.03**2
    expected_linear = 0.05 * (0.30 - 0.5 * 0.05)
    assert torch.allclose(loss, torch.tensor(0.5 * (expected_quadratic + expected_linear)))

    translated_scaled = means.detach() * 7.0 + torch.tensor([3.0, -2.0, 1.0])
    scaled_loss = world_position_consistency_loss(
        translated_scaled, pairs, scene_extent=7.0, beta=0.05
    )
    assert torch.equal(loss.detach(), scaled_loss)

    loss.backward()
    assert means.grad is not None
    assert torch.allclose(means.grad[0], -means.grad[1])
    assert torch.allclose(means.grad[2], -means.grad[3])
    assert torch.count_nonzero(means.grad[4]) == 0
    assert torch.isfinite(means.grad).all()


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        (torch.tensor([[0.0, 2.0]]), "int64"),
        (torch.empty((0, 2), dtype=torch.long), "non-empty shape"),
        (torch.tensor([0, 2], dtype=torch.long), "non-empty shape"),
        (torch.tensor([[2, 0]], dtype=torch.long), "canonical"),
        (torch.tensor([[0, 1]], dtype=torch.long), "different source views"),
        (torch.tensor([[0, 2], [0, 2]], dtype=torch.long), "unique"),
        (torch.tensor([[0, 4]], dtype=torch.long), "out-of-range"),
    ],
)
def test_position_pairs_are_strictly_validated(pairs, message):
    source_view_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=message):
        _validate_position_pairs(pairs, source_view_ids)


def test_position_pair_validation_freezes_a_detached_clone():
    source_view_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    frozen = _validate_position_pairs(pairs, source_view_ids)
    assert torch.equal(frozen, pairs)
    assert frozen.data_ptr() != pairs.data_ptr()
    assert not frozen.requires_grad


def test_sampled_confidence_shuffle_preserves_exact_valid_multiset():
    valid = torch.tensor([True, True, False, True, True])
    confidence = torch.tensor([0.0, 0.2, 0.7, 0.6, 1.0])
    generator = torch.Generator().manual_seed(1_000_003)
    shuffled, diagnostics = _resolve_anchor_weights(
        valid, confidence, "confidence_shuffled", generator
    )
    repeated, _ = _resolve_anchor_weights(
        valid,
        confidence,
        "confidence_shuffled",
        torch.Generator().manual_seed(1_000_003),
    )
    assert torch.equal(shuffled, repeated)
    assert shuffled[~valid].count_nonzero() == 0
    assert torch.equal(shuffled[valid].sort().values, confidence[valid].sort().values)
    assert not torch.equal(shuffled[valid], confidence[valid])
    assert diagnostics["shuffle_multiset_exact"] is True
    assert diagnostics["shuffle_location_changed"] is True
    assert diagnostics["confidence_sum"] == diagnostics["resolved_sum"]
    assert diagnostics["confidence_square_sum"] == diagnostics["resolved_square_sum"]

    uniform, _ = _resolve_anchor_weights(
        valid, confidence, "valid_uniform", torch.Generator().manual_seed(0)
    )
    assert torch.equal(uniform, valid.float())
    assert uniform[0] == 1  # A valid zero-confidence prior remains valid in this control.


def test_thresholded_anchor_loss_hard_gates_low_confidence():
    raw_t = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = bounded_ray_anchor_loss(
        raw_t,
        torch.zeros(2),
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.9, 0.1]),
        "thresholded",
        beta=0.05,
        confidence_threshold=0.5,
    )
    loss.backward()
    assert raw_t.grad is not None
    assert raw_t.grad[0].abs() > 0
    assert raw_t.grad[1] == 0


def test_thresholded_anchor_never_includes_invalid_zero_weight():
    raw_t = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = bounded_ray_anchor_loss(
        raw_t,
        torch.zeros(2),
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.0, 1.0]),
        "thresholded",
        beta=0.05,
        confidence_threshold=0.0,
    )
    loss.backward()
    assert raw_t.grad is not None
    assert raw_t.grad[0] == 0
    assert raw_t.grad[1].abs() > 0


def test_normalized_anchor_targets_unjittered_fraction_with_finite_edge_gradients():
    target = torch.tensor([0.05, 0.95])
    at_target = torch.logit(target).requires_grad_(True)
    zero = bounded_ray_anchor_loss(
        at_target,
        torch.zeros(2),
        target,
        torch.ones(2),
        "normalized",
        beta=0.05,
    )
    assert zero == 0

    displaced = torch.logit(torch.tensor([0.01, 0.99])).requires_grad_(True)
    loss = bounded_ray_anchor_loss(
        displaced,
        torch.zeros(2),
        target,
        torch.ones(2),
        "normalized",
        beta=0.05,
    )
    loss.backward()
    assert displaced.grad is not None
    assert torch.isfinite(displaced.grad).all()
    assert displaced.grad[0] < 0
    assert displaced.grad[1] > 0


def test_gradient_lifter_detaches_differentiable_priors_and_sanitizes_confidence(
    tiny_scene, tiny_fits
):
    g2ds, _ = tiny_fits
    priors = [
        AlignedDepthPrior(
            depth=depth.clone().requires_grad_(True),
            confidence=torch.full_like(depth, float("nan")),
        )
        for depth in tiny_scene.gt_depths
    ]
    lifter = get_lifter(
        "gradient",
        iterations=1,
        rasterizer="torch",
        depth_anchor_mode="confidence",
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    result = lifter.lift_with_priors(g2ds, tiny_scene, priors)
    assert torch.isfinite(result.means).all()
    assert torch.isfinite(torch.tensor(lifter.history)).all()
    assert torch.isfinite(torch.tensor(lifter.anchor_history)).all()


def test_gradient_lifter_default_matches_explicit_legacy(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    kwargs = {
        "iterations": 2,
        "rasterizer": "torch",
        "optimize_rotation": False,
        "optimize_scale": False,
        "merge": False,
        "seed": 0,
    }
    default = get_lifter("gradient", **kwargs)
    explicit = get_lifter(
        "gradient",
        depth_anchor_mode="legacy",
        photometric_supervision_mode="all",
        **kwargs,
    )
    default_result = default.lift(g2ds, tiny_scene)
    explicit_result = explicit.lift(g2ds, tiny_scene)
    assert torch.equal(default_result.means, explicit_result.means)
    assert torch.equal(default_result.log_scales, explicit_result.log_scales)
    assert default.history == explicit.history


def _first_cross_source_pair(lifter) -> torch.Tensor:
    starts = [start for start, end in lifter.source_view_ranges_before_merge if end > start]
    assert len(starts) >= 2
    return torch.tensor([[starts[0], starts[1]]], dtype=torch.long)


def test_position_pairs_at_zero_lambda_preserve_default_and_optimizer_rng(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    kwargs = {
        "iterations": 2,
        "rasterizer": "torch",
        "optimize_rotation": False,
        "optimize_scale": False,
        "merge": False,
        "seed": 0,
    }
    layout = get_lifter("gradient", iterations=0, rasterizer="torch", merge=False, seed=0)
    layout.lift(g2ds, tiny_scene)
    pairs = _first_cross_source_pair(layout)

    default = get_lifter("gradient", **kwargs)
    paired = get_lifter("gradient", position_consistency_lambda=0.0, **kwargs)
    default_result = default.lift(g2ds, tiny_scene)
    paired_result = paired.lift_with_position_pairs(g2ds, tiny_scene, pairs)

    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(default_result, field), getattr(paired_result, field))
    assert default.history == paired.history
    assert default.anchor_history == paired.anchor_history
    assert default.target_view_history == paired.target_view_history
    assert default.rendered_count_history == paired.rendered_count_history
    assert default.position_history == []
    assert len(paired.position_history) == 2
    assert torch.equal(default.source_xy_before_merge, paired.source_xy_before_merge)
    assert torch.equal(paired.position_pairs_before_merge, pairs)
    assert torch.equal(
        default.initial_ray_fractions_before_merge,
        paired.initial_ray_fractions_before_merge,
    )
    assert torch.equal(
        default.final_ray_fractions_before_merge,
        paired.final_ray_fractions_before_merge,
    )


def test_oriented_targets_at_zero_lambdas_preserve_default_and_optimizer_rng(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    layout = get_lifter(
        "gradient",
        iterations=0,
        rasterizer="torch",
        ray_thickness=0.15,
        optimize_rotation=True,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    initial = layout.lift(g2ds, tiny_scene)
    targets = OrientedPointTargets(
        indices=torch.tensor([0, 1], dtype=torch.long),
        points=initial.means[:2].clone(),
        plane_normals=torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.3, 0.2], [0.2, 1.0, 0.4]]), dim=-1
        ),
    )
    kwargs = {
        "iterations": 2,
        "rasterizer": "torch",
        "ray_thickness": 0.15,
        "optimize_rotation": True,
        "optimize_scale": False,
        "merge": False,
        "seed": 0,
    }
    default = get_lifter("gradient", **kwargs)
    targeted = get_lifter(
        "gradient",
        plane_consistency_lambda=0.0,
        normal_consistency_lambda=0.0,
        **kwargs,
    )
    default_result = default.lift(g2ds, tiny_scene)
    targeted_result = targeted.lift_with_oriented_points(g2ds, tiny_scene, targets)

    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(default_result, field), getattr(targeted_result, field))
    assert default.history == targeted.history
    assert default.anchor_history == targeted.anchor_history
    assert default.target_view_history == targeted.target_view_history
    assert default.rendered_count_history == targeted.rendered_count_history
    assert default.plane_history == []
    assert default.normal_history == []
    assert len(targeted.plane_history) == 2
    assert len(targeted.normal_history) == 2
    assert torch.equal(targeted.oriented_targets_before_merge.indices, targets.indices)
    assert targeted.oriented_axis_indices_before_merge.shape == targets.indices.shape
    assert torch.equal(
        targeted.oriented_axis_indices_before_merge,
        initial.log_scales[targets.indices].argmin(dim=-1),
    )
    assert torch.equal(
        default.initial_ray_fractions_before_merge,
        targeted.initial_ray_fractions_before_merge,
    )
    assert torch.equal(
        default.final_ray_fractions_before_merge,
        targeted.final_ray_fractions_before_merge,
    )


def test_plane_target_moves_only_along_retained_ray(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    layout = get_lifter(
        "gradient",
        iterations=0,
        rasterizer="torch",
        ray_thickness=0.15,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    initial = layout.lift(g2ds, tiny_scene)
    view = int(layout.source_view_ids_before_merge[0])
    _, ray = tiny_scene.cameras[view].pixel_rays(layout.source_xy_before_merge[:1])
    ray_unit = torch.nn.functional.normalize(ray, dim=-1)
    targets = OrientedPointTargets(
        indices=torch.tensor([0], dtype=torch.long),
        points=initial.means[:1] + 0.1 * ray_unit,
        plane_normals=ray_unit,
    )
    lifter = get_lifter(
        "gradient",
        iterations=4,
        lr=0.1,
        rasterizer="torch",
        ray_thickness=0.15,
        plane_consistency_lambda=1.0,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    result = lifter.lift_with_oriented_points(g2ds, tiny_scene, targets)
    assert lifter.plane_history[-1] < lifter.plane_history[0]
    assert not torch.equal(result.means[0], initial.means[0])
    projected_xy, _ = tiny_scene.cameras[view].project(result.means[:1])
    assert torch.allclose(projected_xy, layout.source_xy_before_merge[:1], atol=1e-4, rtol=0)


def test_position_pair_path_records_loss_layout_and_bounded_ray_fractions(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    layout = get_lifter("gradient", iterations=0, rasterizer="torch", merge=False, seed=0)
    baseline = layout.lift(g2ds, tiny_scene)
    pairs = _first_cross_source_pair(layout)
    lifter = get_lifter(
        "gradient",
        iterations=2,
        rasterizer="torch",
        position_consistency_lambda=0.25,
        position_consistency_beta=0.05,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    result = lifter.lift_with_position_pairs(g2ds, tiny_scene, pairs)

    assert not torch.equal(result.means, baseline.means)
    assert len(lifter.position_history) == 2
    assert torch.isfinite(torch.tensor(lifter.position_history)).all()
    assert lifter.source_xy_before_merge.shape == (result.n, 2)
    assert torch.equal(lifter.position_pairs_before_merge, pairs)
    for fractions in (
        lifter.initial_ray_fractions_before_merge,
        lifter.final_ray_fractions_before_merge,
    ):
        assert fractions.shape == (result.n,)
        assert bool(((fractions > 0) & (fractions < 1)).all())
    assert lifter.rendered_count_history == [result.n, result.n]

    for view_index, (start, end) in enumerate(lifter.source_view_ranges_before_merge):
        if end == start:
            continue
        projected_xy, _ = tiny_scene.cameras[view_index].project(result.means[start:end])
        assert torch.allclose(
            projected_xy,
            lifter.source_xy_before_merge[start:end],
            atol=1e-4,
            rtol=0,
        )


def test_positive_position_lambda_requires_pairs(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    lifter = get_lifter("gradient", iterations=0, position_consistency_lambda=0.25)
    with pytest.raises(ValueError, match="requires position_pairs"):
        lifter.lift(g2ds, tiny_scene)


def test_photometric_keep_masks_isolate_own_source_from_matched_dropout():
    source_view_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    target_views = [0, 1, 2]
    all_masks, all_diagnostics = _build_photometric_keep_masks(
        source_view_ids, target_views, "all", seed=7
    )
    loso_masks, loso_diagnostics = _build_photometric_keep_masks(
        source_view_ids, target_views, "leave_one_source_out", seed=7
    )
    matched_masks, matched_diagnostics = _build_photometric_keep_masks(
        source_view_ids, target_views, "matched_nonself_dropout", seed=7
    )
    repeated_masks, repeated_diagnostics = _build_photometric_keep_masks(
        source_view_ids, target_views, "matched_nonself_dropout", seed=7
    )

    for target_view in target_views:
        own = source_view_ids == target_view
        assert all_masks[target_view].all()
        assert torch.equal(loso_masks[target_view], ~own)
        assert matched_masks[target_view][own].all()
        assert int((~matched_masks[target_view]).sum()) == int(own.sum())
        assert torch.equal(matched_masks[target_view], repeated_masks[target_view])
    matched_exposure = torch.stack(
        [(~matched_masks[target_view]).long() for target_view in target_views]
    ).sum(dim=0)
    assert torch.equal(matched_exposure, torch.ones_like(matched_exposure))
    for all_item, loso_item, matched_item, repeated_item in zip(
        all_diagnostics, loso_diagnostics, matched_diagnostics, repeated_diagnostics
    ):
        assert all_item["excluded_count"] == 0
        assert loso_item["excluded_count"] == matched_item["excluded_count"]
        assert loso_item["excluded_own_count"] == loso_item["own_source_count"]
        assert matched_item["excluded_own_count"] == 0
        assert matched_item == repeated_item


def test_photometric_keep_masks_reject_empty_or_unmatched_controls():
    with pytest.raises(ValueError, match="no rendered primitives"):
        _build_photometric_keep_masks(
            torch.tensor([0, 0], dtype=torch.long), [0], "leave_one_source_out", seed=0
        )
    with pytest.raises(ValueError, match="at most half"):
        _build_photometric_keep_masks(
            torch.tensor([0, 0, 0, 1], dtype=torch.long),
            [0, 1],
            "matched_nonself_dropout",
            seed=0,
        )


def test_photometric_modes_preserve_initialization_and_main_rng_at_zero_lr(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    modes = ("all", "leave_one_source_out", "matched_nonself_dropout")
    results = []
    lifters = []
    for mode in modes:
        lifter = get_lifter(
            "gradient",
            iterations=3,
            lr=0.0,
            rasterizer="torch",
            photometric_supervision_mode=mode,
            optimize_rotation=False,
            optimize_scale=False,
            merge=False,
            seed=0,
        )
        results.append(lifter.lift(g2ds, tiny_scene))
        lifters.append(lifter)

    reference = results[0]
    for result in results[1:]:
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            assert torch.equal(getattr(reference, field), getattr(result, field))
    assert lifters[0].target_view_history == lifters[1].target_view_history
    assert lifters[0].target_view_history == lifters[2].target_view_history
    assert torch.equal(
        lifters[0].source_view_ids_before_merge,
        lifters[1].source_view_ids_before_merge,
    )
    assert torch.equal(
        lifters[0].source_view_ids_before_merge,
        lifters[2].source_view_ids_before_merge,
    )

    loso = {item["target_view"]: item for item in lifters[1].photometric_supervision_diagnostics}
    matched = {item["target_view"]: item for item in lifters[2].photometric_supervision_diagnostics}
    for target_view in tiny_scene.training_views:
        assert loso[target_view]["excluded_count"] == matched[target_view]["excluded_count"]
        assert loso[target_view]["excluded_own_count"] == loso[target_view]["own_source_count"]
        assert matched[target_view]["excluded_own_count"] == 0
        assert (
            loso[target_view]["excluded_opacity_sum"]
            == matched[target_view]["excluded_opacity_sum"]
        )
    for lifter in lifters:
        rendered_by_target = {
            item["target_view"]: item["rendered_count"]
            for item in lifter.photometric_supervision_diagnostics
        }
        assert lifter.rendered_count_history == [
            rendered_by_target[target_view] for target_view in lifter.target_view_history
        ]


def test_anchor_modes_do_not_change_main_rng_when_lambda_is_zero(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    priors = []
    for depth in tiny_scene.gt_depths:
        confidence = torch.linspace(0.0, 1.0, depth.numel()).reshape_as(depth)
        priors.append(AlignedDepthPrior(depth=depth, confidence=confidence))
    modes = (
        "legacy",
        "normalized",
        "valid_uniform",
        "confidence",
        "confidence_shuffled",
        "thresholded",
    )
    results = []
    histories = []
    lifters = []
    for mode in modes:
        lifter = get_lifter(
            "gradient",
            iterations=2,
            rasterizer="torch",
            depth_prior_lambda=0.0,
            depth_anchor_mode=mode,
            optimize_rotation=False,
            optimize_scale=False,
            merge=False,
            seed=0,
        )
        results.append(lifter.lift_with_priors(g2ds, tiny_scene, priors))
        histories.append(lifter.history)
        lifters.append(lifter)
    reference = results[0]
    for result, history in zip(results[1:], histories[1:]):
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            assert torch.equal(getattr(reference, field), getattr(result, field))
        assert histories[0] == history
    shuffled_diagnostics = lifters[modes.index("confidence_shuffled")].anchor_weight_diagnostics
    assert shuffled_diagnostics
    assert all(item["shuffle_multiset_exact"] is True for item in shuffled_diagnostics)


@pytest.mark.parametrize(
    "mode",
    [
        "legacy",
        "normalized",
        "valid_uniform",
        "confidence",
        "confidence_shuffled",
        "thresholded",
    ],
)
def test_hybrid_lifter_forwards_anchor_mode(mode):
    lifter = get_lifter("hybrid", iterations=0, depth_anchor_mode=mode)
    assert lifter.gradient.depth_anchor_mode == mode


@pytest.mark.parametrize("mode", ["all", "leave_one_source_out", "matched_nonself_dropout"])
def test_hybrid_lifter_forwards_photometric_supervision_mode(mode):
    lifter = get_lifter("hybrid", iterations=0, photometric_supervision_mode=mode)
    assert lifter.gradient.photometric_supervision_mode == mode


def test_hybrid_lifter_forwards_and_uses_position_pairs(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    layout = get_lifter("gradient", iterations=0, rasterizer="torch", merge=False, seed=0)
    layout.lift(g2ds, tiny_scene)
    pairs = _first_cross_source_pair(layout)
    lifter = get_lifter(
        "hybrid",
        iterations=1,
        rasterizer="torch",
        position_consistency_lambda=0.25,
        position_consistency_beta=0.05,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    result = lifter.lift_with_position_pairs(g2ds, tiny_scene, pairs)
    assert lifter.gradient.position_consistency_lambda == 0.25
    assert lifter.gradient.position_consistency_beta == 0.05
    assert torch.equal(lifter.gradient.position_pairs_before_merge, pairs)
    assert torch.equal(lifter.gradient.source_xy_before_merge, layout.source_xy_before_merge)
    assert len(lifter.position_history) == 1
    assert result.n == layout.n_before_merge


def test_hybrid_lifter_forwards_and_uses_oriented_targets(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    layout = get_lifter(
        "hybrid",
        iterations=0,
        rasterizer="torch",
        ray_thickness=0.15,
        optimize_rotation=True,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    initial = layout.lift(g2ds, tiny_scene)
    targets = OrientedPointTargets(
        indices=torch.tensor([0, 1], dtype=torch.long),
        points=initial.means[:2].clone(),
        plane_normals=torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.4, 0.2], [0.3, 1.0, 0.2]]), dim=-1
        ),
    )
    lifter = get_lifter(
        "hybrid",
        iterations=1,
        rasterizer="torch",
        ray_thickness=0.15,
        plane_consistency_lambda=0.05,
        normal_consistency_lambda=0.2,
        optimize_rotation=True,
        optimize_scale=False,
        merge=False,
        seed=0,
    )
    result = lifter.lift_with_oriented_points(g2ds, tiny_scene, targets)
    assert lifter.gradient.plane_consistency_lambda == 0.05
    assert lifter.gradient.normal_consistency_lambda == 0.2
    assert torch.equal(lifter.gradient.oriented_targets_before_merge.indices, targets.indices)
    assert len(lifter.plane_history) == 1
    assert len(lifter.normal_history) == 1
    assert result.n == layout.gradient.n_before_merge


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"depth_anchor_mode": "unknown"}, "depth_anchor_mode"),
        ({"depth_anchor_beta": 0.0}, "depth_anchor_beta"),
        ({"depth_anchor_beta": float("nan")}, "depth_anchor_beta"),
        ({"depth_confidence_threshold": -0.1}, "depth_confidence_threshold"),
        ({"depth_confidence_threshold": float("inf")}, "depth_confidence_threshold"),
        ({"photometric_supervision_mode": "unknown"}, "photometric_supervision_mode"),
        ({"position_consistency_lambda": -0.1}, "position_consistency_lambda"),
        ({"position_consistency_lambda": float("inf")}, "position_consistency_lambda"),
        ({"position_consistency_beta": 0.0}, "position_consistency_beta"),
        ({"position_consistency_beta": float("nan")}, "position_consistency_beta"),
        ({"plane_consistency_lambda": -0.1}, "plane_consistency_lambda"),
        ({"plane_consistency_lambda": float("inf")}, "plane_consistency_lambda"),
        ({"normal_consistency_lambda": -0.1}, "normal_consistency_lambda"),
        ({"normal_consistency_lambda": float("nan")}, "normal_consistency_lambda"),
    ],
)
def test_gradient_lifter_rejects_invalid_anchor_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_lifter("gradient", **kwargs)


def test_positive_oriented_lambdas_require_targets_and_rotation(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    plane = get_lifter("gradient", iterations=0, plane_consistency_lambda=0.05)
    with pytest.raises(ValueError, match="requires oriented_targets"):
        plane.lift(g2ds, tiny_scene)
    with pytest.raises(ValueError, match="requires optimize_rotation"):
        get_lifter(
            "gradient",
            normal_consistency_lambda=0.2,
            optimize_rotation=False,
        )


def test_carve_lifter_places_in_occupied_space(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    lifter = get_lifter("carve", grid_res=32)
    g3d = lifter.lift(g2ds, tiny_scene)
    assert g3d.n > 50
    # Placed gaussians concentrate near true geometry.
    d = torch.cdist(g3d.means, tiny_scene.gt_gaussians.means).min(dim=1).values
    assert d.median() < 0.25, f"median distance to GT {d.median()}"


def test_baselines(tiny_scene, tiny_fits):
    g2ds, _ = tiny_fits
    sfm = get_lifter("sfm").lift(g2ds, tiny_scene)
    assert sfm.n == tiny_scene.points.shape[0]
    rnd = get_lifter("random", n=500).lift(g2ds, tiny_scene)
    assert rnd.n == 500


def test_merge_by_voxel_moment_matching():
    """Merging two gaussians in one cell preserves the mixture's mean and covariance."""
    means = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    covs = torch.eye(3).expand(2, 3, 3) * 1e-4
    g = Gaussians3D.from_means_covs(
        means=means,
        covs=covs.clone(),
        colors=torch.rand(2, 3),
        opacity=torch.tensor([0.5, 0.5]),
    )
    merged = merge_by_voxel(g, voxel_size=1.0)
    assert merged.n == 1
    # Equal weights: merged mean is the midpoint.
    assert torch.allclose(merged.means[0], torch.tensor([0.01, 0.0, 0.0]), atol=1e-5)
    # Merged covariance = within + between: 1e-4 + 0.01^2 on the x axis.
    cov_m = merged.covariance()[0]
    assert abs(cov_m[0, 0].item() - (1e-4 + 0.01**2)) < 1e-6
    assert abs(cov_m[1, 1].item() - 1e-4) < 1e-6
    # Opacity composes as 1 - (1-a)(1-b).
    assert abs(merged.opacity[0].item() - 0.75) < 1e-3


def test_merge_keeps_separate_cells():
    means = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    g = Gaussians3D.from_means_covs(
        means=means,
        covs=torch.eye(3).expand(2, 3, 3) * 1e-4,
        colors=torch.rand(2, 3),
        opacity=torch.tensor([0.5, 0.5]),
    )
    merged = merge_by_voxel(g, voxel_size=1.0)
    assert merged.n == 2


def test_merge_by_voxel_returns_group_correspondence_map():
    # Two views of one surface patch (near 0) plus a distinct patch (near 5).
    means = torch.tensor(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [5.0, 5.0, 5.0]],
    )
    g = Gaussians3D.from_means_covs(
        means=means,
        covs=torch.eye(3).expand(3, 3, 3) * 1e-4,
        colors=torch.rand(3, 3),
        opacity=torch.full((3,), 0.5),
    )
    # Default return type is unchanged (backward compatible).
    assert isinstance(merge_by_voxel(g, voxel_size=1.0), Gaussians3D)

    merged, group = merge_by_voxel(g, voxel_size=1.0, return_group=True)
    assert merged.n == 2
    assert group.shape == (3,)
    # The first two inputs (same patch, different views) fuse; the third stays separate.
    assert int(group[0]) == int(group[1])
    assert int(group[2]) != int(group[0])

    # Empty input stays empty and still returns a group vector.
    empty = Gaussians3D.from_means_covs(
        means=torch.zeros(0, 3),
        covs=torch.zeros(0, 3, 3),
        colors=torch.zeros(0, 3),
        opacity=torch.zeros(0),
    )
    empty_merged, empty_group = merge_by_voxel(empty, voxel_size=1.0, return_group=True)
    assert empty_merged.n == 0
    assert empty_group.shape == (0,)
