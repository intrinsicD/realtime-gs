"""Lifting variants: shared geometry, depth, gradient, carve, baselines, merging."""

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift import get_lifter, lifter_names
from rtgs.lift.base import bilinear_sample, lift_covariance, lift_view_at_depth, ray_basis
from rtgs.lift.merge import merge_by_voxel
from rtgs.optim.trainer import Trainer


def test_lifter_registry():
    assert set(lifter_names()) == {"gradient", "depth", "hybrid", "carve", "sfm", "random"}


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
