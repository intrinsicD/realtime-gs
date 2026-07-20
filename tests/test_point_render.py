"""CPU development tests for sparse point-rasterizer semantics."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import rtgs.render.torch_points as point_module
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import activate_sh_color, eval_sh_preactivation, rgb_to_sh
from rtgs.render import PointRasterizer, TorchPointRasterizer
from rtgs.render.torch_ref import TorchRasterizer

_DEVELOPMENT_SEEDS = (424201, 424202, 424203)


def _development_fixture(seed: int) -> tuple[Gaussians3D, Camera]:
    assert seed in _DEVELOPMENT_SEEDS
    generator = torch.Generator(device="cpu").manual_seed(seed)
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, -2.6]),
        torch.zeros(3),
        fov_x_deg=51.0,
        width=13,
        height=11,
    )
    means = torch.empty(8, 3)
    means[0] = torch.tensor([0.0, 0.0, -0.45])
    means[1] = torch.tensor([0.0, 0.0, 0.30])
    means[2] = torch.tensor([-0.24, 0.13, 0.05])
    means[3] = torch.tensor([0.29, -0.19, 0.05])
    means[4:6, 0] = 0.8 * torch.rand(2, generator=generator) - 0.4
    means[4:6, 1] = 0.6 * torch.rand(2, generator=generator) - 0.3
    means[4:6, 2] = 0.8 * torch.rand(2, generator=generator) - 0.4
    means[6] = torch.tensor([2.3, 0.0, 0.0])
    means[7] = torch.tensor([0.0, 0.0, -3.0])
    quats = torch.randn(8, 4, generator=generator)
    log_scales = torch.log(0.05 + 0.13 * torch.rand(8, 3, generator=generator))
    opacity = 0.12 + 0.73 * torch.rand(8, generator=generator)
    sh = 0.05 * torch.randn(8, 9, 3, generator=generator)
    sh[:, 0] = rgb_to_sh(0.15 + 0.7 * torch.rand(8, 3, generator=generator))
    return Gaussians3D(means, quats, log_scales, opacity, sh), camera


def _pixel_centers(camera: Camera) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(camera.height, dtype=torch.float32) + 0.5,
        torch.arange(camera.width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1).reshape(-1, 2)


def _leaf_clone(gaussians: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(
        means=gaussians.means.detach().clone().requires_grad_(True),
        quats=gaussians.quats.detach().clone().requires_grad_(True),
        log_scales=gaussians.log_scales.detach().clone().requires_grad_(True),
        opacity=gaussians.opacity.detach().clone().requires_grad_(True),
        sh=gaussians.sh.detach().clone().requires_grad_(True),
    )


def _six_parameter_clone(
    gaussians: Gaussians3D,
) -> tuple[Gaussians3D, dict[str, torch.Tensor]]:
    params = {
        "means": gaussians.means.detach().clone().requires_grad_(True),
        "quats": gaussians.quats.detach().clone().requires_grad_(True),
        "scales": gaussians.log_scales.detach().clone().requires_grad_(True),
        "opacities": torch.logit(gaussians.opacity.detach()).clone().requires_grad_(True),
        "sh0": gaussians.sh[:, :1].detach().clone().requires_grad_(True),
        "shN": gaussians.sh[:, 1:].detach().clone().requires_grad_(True),
    }
    return (
        Gaussians3D(
            means=params["means"],
            quats=params["quats"],
            log_scales=params["scales"],
            opacity=torch.sigmoid(params["opacities"]),
            sh=torch.cat([params["sh0"], params["shN"]], dim=1),
        ),
        params,
    )


@torch.no_grad()
def _literal_compositing_weights(
    gaussians: Gaussians3D,
    camera: Camera,
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow, independent expression of the preregistered point compositor."""
    means_cam = camera.world_to_cam(gaussians.means)
    z = means_cam[:, 2]
    uv, _ = camera.project(gaussians.means)
    covariance_cam = (
        camera.R.to(gaussians.means) @ gaussians.covariance() @ camera.R.to(gaussians.means).T
    )
    safe_z = z.clamp_min(0.05)
    jacobian = torch.zeros(gaussians.n, 2, 3, dtype=xy.dtype, device=xy.device)
    jacobian[:, 0, 0] = camera.fx / safe_z
    jacobian[:, 0, 2] = -camera.fx * means_cam[:, 0] / safe_z.square()
    jacobian[:, 1, 1] = camera.fy / safe_z
    jacobian[:, 1, 2] = -camera.fy * means_cam[:, 1] / safe_z.square()
    covariance_2d = jacobian @ covariance_cam @ jacobian.transpose(-1, -2)
    covariance_2d = covariance_2d + 0.3 * torch.eye(2, dtype=xy.dtype, device=xy.device)
    max_eigenvalue = (
        0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
        + (
            0.25 * (covariance_2d[:, 0, 0] - covariance_2d[:, 1, 1]).square()
            + covariance_2d[:, 0, 1].square()
        ).sqrt()
    )
    radius = 3.0 * max_eigenvalue.clamp_min(1e-8).sqrt()
    visible = (z > 0.05) & camera.in_image(uv, margin=radius)
    visible_indices = visible.nonzero(as_tuple=True)[0]
    visible_indices = visible_indices[torch.argsort(z[visible_indices])]

    inverse = torch.linalg.inv(covariance_2d[visible_indices])
    weights = xy.new_zeros((xy.shape[0], visible_indices.numel()))
    for point_index in range(xy.shape[0]):
        transmittance = xy.new_tensor(1.0)
        for visible_row in range(visible_indices.numel()):
            delta = xy[point_index] - uv[visible_indices[visible_row]]
            mahalanobis = delta @ inverse[visible_row] @ delta
            gaussian_weight = torch.where(
                mahalanobis < 12.0,
                torch.exp(-0.5 * mahalanobis),
                mahalanobis.new_zeros(()),
            )
            alpha = (gaussians.opacity[visible_indices[visible_row]] * gaussian_weight).clamp(
                0.0, 0.999
            )
            weights[point_index, visible_row] = alpha * transmittance
            transmittance = transmittance * (1.0 - alpha + 1e-10)
    return visible_indices, weights


def _compositing_vjp(
    output,
    active: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    basis = output.compositing_color_basis
    if basis is None:
        visible_count = 0 if output.visible is None else output.visible.numel()
        assert visible_count == 0 or output.color.shape[0] == 0
        return output.color.new_zeros((visible_count, 3))
    grad_outputs = torch.zeros_like(output.color)
    grad_outputs[:, 0] = active.to(output.color) * residual.to(output.color)
    grad_outputs[:, 1] = active.to(output.color)
    (vjp,) = torch.autograd.grad(
        output.color,
        basis,
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=False,
    )
    return vjp.detach()


@pytest.mark.parametrize("seed", _DEVELOPMENT_SEEDS)
@pytest.mark.parametrize(("point_chunk", "gaussian_chunk"), ((1, 1), (7, 3), (4096, 4096)))
def test_point_centers_match_dense_forward(seed, point_chunk, gaussian_chunk):
    gaussians, camera = _development_fixture(seed)
    background = torch.tensor([0.13, 0.29, 0.47])
    dense = TorchRasterizer(row_chunk=4).render(
        gaussians, camera, background=background, sh_degree=2
    )
    points = TorchPointRasterizer(
        point_chunk=point_chunk, gaussian_chunk=gaussian_chunk
    ).render_points(gaussians, camera, _pixel_centers(camera), background, sh_degree=2)

    assert torch.equal(points.visible, dense.visible)
    torch.testing.assert_close(points.color, dense.color.reshape(-1, 3), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(points.alpha, dense.alpha.reshape(-1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(points.depth, dense.depth.reshape(-1), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    ("activation", "kernel"),
    (
        ("smu1", "c1_taper"),
        ("hard_forward_smu1_negative_gradient", "hard_forward_c1_taper_gradient"),
    ),
)
def test_declared_nondeault_modes_match_dense_forward(activation, kernel):
    gaussians, camera = _development_fixture(424201)
    xy = _pixel_centers(camera)[::5]
    dense = TorchRasterizer(
        row_chunk=3, sh_color_activation=activation, kernel_support_mode=kernel
    ).render(gaussians, camera, sh_degree=2)
    points = TorchPointRasterizer(
        point_chunk=4,
        gaussian_chunk=2,
        sh_color_activation=activation,
        kernel_support_mode=kernel,
    ).render_points(gaussians, camera, xy, sh_degree=2)
    flat_indices = (xy[:, 1] - 0.5).long() * camera.width + (xy[:, 0] - 0.5).long()
    torch.testing.assert_close(
        points.color, dense.color.reshape(-1, 3)[flat_indices], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        points.alpha, dense.alpha.reshape(-1)[flat_indices], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        points.depth, dense.depth.reshape(-1)[flat_indices], atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("seed", _DEVELOPMENT_SEEDS)
@pytest.mark.parametrize(("point_chunk", "gaussian_chunk"), ((1, 1), (7, 3), (4096, 4096)))
def test_point_gradients_match_gathered_dense(seed, point_chunk, gaussian_chunk):
    source, camera = _development_fixture(seed)
    dense_gaussians = _leaf_clone(source)
    point_gaussians = _leaf_clone(source)
    xy = _pixel_centers(camera)
    sample_count = xy.shape[0]
    color_coefficients = torch.linspace(-0.7, 0.9, sample_count * 3).reshape(sample_count, 3)
    alpha_coefficients = torch.linspace(0.8, -0.3, sample_count)
    depth_coefficients = torch.linspace(-0.2, 0.6, sample_count)

    dense = TorchRasterizer(row_chunk=4).render(dense_gaussians, camera, sh_degree=2)
    points = TorchPointRasterizer(
        point_chunk=point_chunk, gaussian_chunk=gaussian_chunk
    ).render_points(point_gaussians, camera, xy, sh_degree=2)
    dense_loss = (
        (dense.color.reshape(-1, 3) * color_coefficients).sum() / (3 * sample_count)
        + 0.17 * (dense.alpha.reshape(-1) * alpha_coefficients).sum() / sample_count
        + 0.03 * (dense.depth.reshape(-1) * depth_coefficients).sum() / sample_count
    )
    point_loss = (
        (points.color * color_coefficients).sum() / (3 * sample_count)
        + 0.17 * (points.alpha * alpha_coefficients).sum() / sample_count
        + 0.03 * (points.depth * depth_coefficients).sum() / sample_count
    )
    dense_loss.backward()
    point_loss.backward()

    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        dense_gradient = getattr(dense_gaussians, name).grad
        point_gradient = getattr(point_gaussians, name).grad
        assert dense_gradient is not None and point_gradient is not None
        assert bool(torch.isfinite(dense_gradient).all())
        assert bool(torch.isfinite(point_gradient).all())
        torch.testing.assert_close(point_gradient, dense_gradient, atol=4e-6, rtol=5e-5)
    assert dense.means2d is not None and points.means2d is not None
    assert dense.means2d.grad is not None and points.means2d.grad is not None
    torch.testing.assert_close(points.means2d.grad, dense.means2d.grad, atol=4e-6, rtol=5e-5)


def test_opt_in_compositing_basis_vjp_matches_literal_weights_and_global_mapping():
    gaussians, camera = _development_fixture(424201)
    gaussians = _leaf_clone(gaussians)
    xy = _pixel_centers(camera)[::13]
    output = TorchPointRasterizer(point_chunk=4, gaussian_chunk=3).render_points(
        gaussians,
        camera,
        xy,
        sh_degree=2,
        collect_compositing_color_basis=True,
    )
    assert output.visible is not None
    assert output.compositing_color_basis is not None
    assert output.compositing_color_basis.shape == (output.visible.numel(), 3)
    assert output.compositing_color_basis.requires_grad
    assert not output.compositing_color_basis.is_leaf

    directions = torch.nn.functional.normalize(
        gaussians.means[output.visible] - camera.position.to(gaussians.means), dim=-1
    )
    expected_basis = activate_sh_color(
        eval_sh_preactivation(2, gaussians.sh[output.visible], directions), "hard"
    )
    assert torch.equal(output.compositing_color_basis, expected_basis)

    literal_visible, weights = _literal_compositing_weights(gaussians, camera, xy)
    assert torch.equal(output.visible, literal_visible)
    active = torch.tensor([True, False, True, True, False, True, True, False, True, True, False])
    teacher = torch.linspace(0.05, 0.95, xy.shape[0] * 3).reshape(xy.shape[0], 3)
    residual = (output.color.detach() - teacher).square().mean(dim=-1)
    vjp = _compositing_vjp(output, active, residual)

    expected_residual = (weights * (active * residual)[:, None]).sum(dim=0)
    expected_support = (weights * active[:, None]).sum(dim=0)
    torch.testing.assert_close(vjp[:, 0], expected_residual, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(vjp[:, 1], expected_support, atol=2e-6, rtol=2e-5)
    assert torch.equal(vjp[:, 2], torch.zeros_like(vjp[:, 2]))
    torch.testing.assert_close(
        vjp[:, 0].sum(),
        (active * residual * output.alpha.detach()).sum(),
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        vjp[:, 1].sum(),
        (active * output.alpha.detach()).sum(),
        atol=2e-6,
        rtol=2e-5,
    )

    global_residual = torch.zeros(gaussians.n)
    global_residual.index_copy_(0, output.visible, vjp[:, 0])
    assert torch.equal(global_residual[output.visible], vjp[:, 0])
    nonvisible = torch.ones(gaussians.n, dtype=torch.bool)
    nonvisible[output.visible] = False
    assert torch.equal(global_residual[nonvisible], torch.zeros_like(global_residual[nonvisible]))


@pytest.mark.parametrize(
    ("point_chunk", "gaussian_chunk"),
    ((1, 1), (2, 3), (7, 2), (4096, 4096)),
)
def test_compositing_basis_vjp_is_chunk_invariant(point_chunk, gaussian_chunk):
    source, camera = _development_fixture(424202)
    xy = _pixel_centers(camera)[::9]
    active = torch.arange(xy.shape[0]) % 3 != 1
    teacher = torch.linspace(0.9, 0.1, xy.shape[0] * 3).reshape(xy.shape[0], 3)

    output = TorchPointRasterizer(
        point_chunk=point_chunk, gaussian_chunk=gaussian_chunk
    ).render_points(
        _leaf_clone(source),
        camera,
        xy,
        sh_degree=2,
        collect_compositing_color_basis=True,
    )
    residual = (output.color.detach() - teacher).square().mean(dim=-1)
    actual = _compositing_vjp(output, active, residual)

    reference_output = TorchPointRasterizer(point_chunk=4096, gaussian_chunk=4096).render_points(
        _leaf_clone(source),
        camera,
        xy,
        sh_degree=2,
        collect_compositing_color_basis=True,
    )
    reference_residual = (reference_output.color.detach() - teacher).square().mean(dim=-1)
    reference = _compositing_vjp(reference_output, active, reference_residual)
    assert torch.equal(output.visible, reference_output.visible)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_compositing_basis_inactive_and_empty_contracts():
    source, camera = _development_fixture(424203)
    xy = _pixel_centers(camera)[::17]
    output = TorchPointRasterizer(point_chunk=2, gaussian_chunk=2).render_points(
        _leaf_clone(source),
        camera,
        xy,
        collect_compositing_color_basis=True,
    )
    active = torch.arange(xy.shape[0]) % 2 == 0
    first_residual = torch.arange(xy.shape[0], dtype=torch.float32)
    second_residual = first_residual.clone()
    second_residual[~active] += 1_000_000.0
    first = _compositing_vjp(output, active, first_residual)
    second = _compositing_vjp(output, active, second_residual)
    assert torch.equal(second, first)
    all_inactive = _compositing_vjp(output, torch.zeros_like(active), second_residual)
    assert torch.equal(all_inactive, torch.zeros_like(all_inactive))

    behind = source.detach()
    behind.means.fill_(0.0)
    behind.means[:, 2] = -3.1
    empty_visible = TorchPointRasterizer().render_points(
        behind,
        camera,
        xy,
        collect_compositing_color_basis=True,
    )
    assert empty_visible.visible is not None and empty_visible.visible.numel() == 0
    assert empty_visible.compositing_color_basis is None
    empty_reductions = _compositing_vjp(
        empty_visible, torch.zeros(xy.shape[0], dtype=torch.bool), torch.zeros(xy.shape[0])
    )
    assert empty_reductions.shape == (0, 3)
    empty_global_residual = torch.zeros(source.n).index_add(
        0, empty_visible.visible, empty_reductions[:, 0]
    )
    empty_global_support = torch.zeros(source.n).index_add(
        0, empty_visible.visible, empty_reductions[:, 1]
    )
    assert torch.equal(empty_global_residual, torch.zeros_like(empty_global_residual))
    assert torch.equal(empty_global_support, torch.zeros_like(empty_global_support))

    empty_query = TorchPointRasterizer().render_points(
        source,
        camera,
        torch.empty(0, 2),
        collect_compositing_color_basis=True,
    )
    assert empty_query.visible is not None and empty_query.visible.numel() > 0
    assert empty_query.compositing_color_basis is None
    assert empty_query.color.shape == (0, 3)
    empty_query_reductions = _compositing_vjp(
        empty_query, torch.zeros(0, dtype=torch.bool), torch.zeros(0)
    )
    assert torch.equal(
        empty_query_reductions,
        torch.zeros((empty_query.visible.numel(), 3)),
    )


def test_compositing_basis_on_off_preserves_exact_forward_backward_and_rng():
    source, camera = _development_fixture(424201)
    off_gaussians, off_params = _six_parameter_clone(source)
    on_gaussians, on_params = _six_parameter_clone(source)
    xy = _pixel_centers(camera)[::7]
    background = torch.tensor([0.17, 0.31, 0.43])
    target = torch.linspace(0.0, 1.0, xy.shape[0] * 3).reshape(xy.shape[0], 3)

    initial_rng = torch.get_rng_state().clone()
    off = TorchPointRasterizer(point_chunk=4, gaussian_chunk=3).render_points(
        off_gaussians, camera, xy, background, sh_degree=2
    )
    off_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    on = TorchPointRasterizer(point_chunk=4, gaussian_chunk=3).render_points(
        on_gaussians,
        camera,
        xy,
        background,
        sh_degree=2,
        collect_compositing_color_basis=True,
    )
    assert torch.equal(torch.get_rng_state(), off_rng)
    assert off.compositing_color_basis is None
    assert on.compositing_color_basis is not None
    assert torch.equal(off.color, on.color)
    assert torch.equal(off.alpha, on.alpha)
    assert torch.equal(off.depth, on.depth)
    assert torch.equal(off.visible, on.visible)
    assert off.means2d is not None and on.means2d is not None
    assert torch.equal(off.means2d, on.means2d)

    off_loss = (off.color - target).square().mean() + 0.13 * off.alpha.mean()
    on_loss = (on.color - target).square().mean() + 0.13 * on.alpha.mean()
    assert torch.equal(off_loss, on_loss)
    active = torch.arange(xy.shape[0]) % 4 != 0
    point_residual = (on.color.detach() - target).square().mean(dim=-1)
    vjp = _compositing_vjp(on, active, point_residual)
    assert not vjp.requires_grad
    assert bool(torch.isfinite(vjp).all())
    assert all(parameter.grad is None for parameter in on_params.values())
    assert all(parameter.grad is None for parameter in off_params.values())
    assert on.means2d.grad is None
    assert off.means2d.grad is None

    off_loss.backward()
    on_loss.backward()
    for name in ("means", "quats", "scales", "opacities", "sh0", "shN"):
        assert off_params[name].grad is not None
        assert on_params[name].grad is not None
        assert torch.equal(off_params[name].grad, on_params[name].grad)
    assert off.means2d.grad is not None and on.means2d.grad is not None
    assert torch.equal(off.means2d.grad, on.means2d.grad)
    assert torch.equal(torch.get_rng_state(), off_rng)


def test_continuous_coordinates_have_finite_coordinate_gradients():
    gaussians, camera = _development_fixture(424202)
    xy = torch.tensor(
        [[0.8, 0.7], [3.125, 2.875], [11.8, 9.9], [5.2, 6.4]],
        requires_grad=True,
    )
    output = TorchPointRasterizer(point_chunk=3, gaussian_chunk=2).render_points(
        gaussians, camera, xy, background=torch.tensor([0.2, 0.3, 0.4])
    )
    assert bool(torch.isfinite(output.color).all())
    assert bool(torch.isfinite(output.alpha).all())
    assert bool(torch.isfinite(output.depth).all())
    (output.color.sum() + output.alpha.sum() + output.depth.sum()).backward()
    assert xy.grad is not None and bool(torch.isfinite(xy.grad).all())


def test_global_compositor_uses_every_visible_gaussian_and_depth_order():
    camera = Camera.look_at(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), width=9, height=9)
    means = torch.tensor([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]])
    covariances = torch.eye(3)[None].repeat(2, 1, 1) * 0.1**2
    gaussians = Gaussians3D.from_means_covs(
        means,
        covariances,
        colors=torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        opacity=torch.tensor([0.7, 0.8]),
    )
    xy = torch.tensor([[4.5, 4.5]])
    renderer: PointRasterizer = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1)
    baseline = renderer.render_points(gaussians, camera, xy)

    changed_sh = gaussians.sh.clone()
    changed_sh[0, 0] = rgb_to_sh(torch.tensor([0.1, 0.1, 0.9]))
    changed_opacity = gaussians.opacity.clone()
    changed_opacity[0] = 0.3
    changed = Gaussians3D(
        gaussians.means,
        gaussians.quats,
        gaussians.log_scales,
        changed_opacity,
        changed_sh,
    )
    intervention = renderer.render_points(changed, camera, xy)
    assert float((intervention.color - baseline.color).abs().max()) > 1e-4
    assert "proposal" not in inspect.signature(renderer.render_points).parameters
    assert "lineage" not in inspect.signature(renderer.render_points).parameters

    reversed_output = renderer.render_points(gaussians.subset(torch.tensor([1, 0])), camera, xy)
    torch.testing.assert_close(reversed_output.color, baseline.color, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(reversed_output.alpha, baseline.alpha, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(reversed_output.depth, baseline.depth, atol=1e-7, rtol=1e-6)


def test_streamed_terminal_background_matches_dense_last_factor_exactly():
    camera = Camera.look_at(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), width=9, height=9)
    gaussians = Gaussians3D(
        means=torch.tensor([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.full((2, 3), math.log(0.1)),
        opacity=torch.tensor([0.5, 0.999]),
        sh=torch.full((2, 1, 3), -10.0),
    )
    xy = torch.tensor([[4.5, 4.5]])
    background = torch.tensor([0.13, 0.29, 0.47])
    dense = TorchRasterizer().render(gaussians, camera, background)
    point = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1).render_points(
        gaussians, camera, xy, background
    )
    expected_weight = (1.0 - gaussians.opacity[0] + 1e-10) * (1.0 - gaussians.opacity[1])
    assert torch.equal(point.color[0], expected_weight * background)
    assert torch.equal(point.color[0], dense.color[4, 4])


def test_empty_visible_and_empty_query_contracts():
    source, camera = _development_fixture(424203)
    behind = Gaussians3D(
        means=torch.full_like(source.means, 0.0),
        quats=source.quats,
        log_scales=source.log_scales,
        opacity=source.opacity,
        sh=source.sh,
    )
    behind.means[:, 2] = -3.1
    background = torch.tensor([0.2, 0.4, 0.6])
    xy = torch.tensor([[0.5, 0.5], [6.5, 5.5]])
    empty_visible = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1).render_points(
        behind, camera, xy, background
    )
    assert empty_visible.means2d is None
    assert empty_visible.visible is not None and empty_visible.visible.numel() == 0
    assert torch.equal(empty_visible.color, background[None].expand(2, 3))
    assert torch.equal(empty_visible.alpha, torch.zeros(2))
    assert torch.equal(empty_visible.depth, torch.zeros(2))

    empty_query = TorchPointRasterizer().render_points(
        source, camera, torch.empty(0, 2), background
    )
    assert empty_query.color.shape == (0, 3)
    assert empty_query.alpha.shape == (0,)
    assert empty_query.depth.shape == (0,)
    assert empty_query.means2d is not None
    assert empty_query.visible is not None and empty_query.visible.numel() > 0


def test_pair_temporaries_respect_both_chunk_caps(monkeypatch):
    gaussians, camera = _development_fixture(424201)
    observed_shapes: list[tuple[int, int]] = []
    original = point_module.kernel_support_weight

    def capture(q, mode="hard"):
        observed_shapes.append(tuple(q.shape))
        return original(q, mode)

    monkeypatch.setattr(point_module, "kernel_support_weight", capture)
    TorchPointRasterizer(point_chunk=4, gaussian_chunk=2).render_points(
        gaussians, camera, _pixel_centers(camera)
    )
    assert observed_shapes
    assert max(rows for rows, _ in observed_shapes) <= 4
    assert max(columns for _, columns in observed_shapes) <= 2


@pytest.mark.parametrize("name", ("point_chunk", "gaussian_chunk"))
@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_chunk_controls_must_be_positive_integers(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match="positive integer"):
        TorchPointRasterizer(**kwargs)


def test_nondefault_visibility_and_invalid_coordinates_fail_explicitly():
    with pytest.raises(NotImplementedError, match="visibility_margin_sigma=3.0"):
        TorchPointRasterizer(visibility_margin_sigma=math.sqrt(12.0))
    gaussians, camera = _development_fixture(424201)
    renderer = TorchPointRasterizer()
    with pytest.raises(TypeError, match="floating point"):
        renderer.render_points(gaussians, camera, torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="shape"):
        renderer.render_points(gaussians, camera, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        renderer.render_points(gaussians, camera, torch.tensor([[float("nan"), 2.0]]))
