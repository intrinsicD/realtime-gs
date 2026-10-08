"""Jet-consistency priors: v1 kNN prior and v2 collapse-proof field prior."""

import math

import pytest
import torch

from rtgs.core.gaussians3d import rotmat_to_quat
from rtgs.optim.jet_prior import (
    FieldPrior,
    JetPrior,
    field_estimate,
    field_residuals,
    jet_residuals,
    knn_indices,
    mean_neighbour_distance,
    radius_pairs,
)


def _sphere_splats(n: int, radius: float, normals: torch.Tensor | None = None):
    golden = math.pi * (3 - math.sqrt(5))
    i = torch.arange(n, dtype=torch.float64)
    z = 1 - (2 * i + 1) / n
    r = (1 - z * z).sqrt()
    points = torch.stack([r * torch.cos(golden * i), r * torch.sin(golden * i), z], 1)
    nu = points.clone() if normals is None else normals.double()
    helper = torch.where(
        nu[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0])
    )
    t1 = torch.nn.functional.normalize(torch.linalg.cross(nu, helper.double()), dim=1)
    t2 = torch.linalg.cross(nu, t1)
    rot = torch.stack([t1, t2, nu], dim=2).float()  # columns: scale axes
    quats = rotmat_to_quat(rot)
    log_scales = torch.log(torch.tensor([0.02, 0.015, 0.001])).expand(n, 3).clone()
    return (radius * points).float(), quats, log_scales


def test_sphere_residual_small_and_shape_operator():
    radius = 2.0
    means, quats, log_scales = _sphere_splats(2000, radius)
    neighbours = knn_indices(means, 16)
    r_nu, r_c, shape = jet_residuals(means, quats, log_scales, neighbours)
    h = (means[neighbours[:, :6]] - means[:, None]).norm(dim=-1).mean()
    assert float(r_nu.mean()) < float((h / radius) ** 2) * 1e-2
    assert float(r_c.mean()) < float((h / radius) ** 2) * 1e-2
    eye = torch.eye(2).expand_as(shape) / radius
    assert float((shape - eye).abs().max()) < 0.05 / radius
    # The first-order control cannot represent curvature: O((h/R)^2), not ~0.
    r1_nu, _, _ = jet_residuals(means, quats, log_scales, neighbours, order=1)
    assert float(r1_nu.mean()) > 10 * float(r_nu.mean())


def test_random_normals_are_order_one():
    normals = torch.nn.functional.normalize(torch.randn(2000, 3), dim=1)
    means, quats, log_scales = _sphere_splats(2000, 2.0, normals)
    neighbours = knn_indices(means, 16)
    r_nu, _, _ = jet_residuals(means, quats, log_scales, neighbours)
    assert float(r_nu.mean()) > 0.2


def test_knn_prior_collapse_minimum_from_clones():
    # Negative control for the v1 failure: once clones fill the 6-neighbour bandwidth, h -> 0
    # and only zero-offset copies keep weight, so a random (wrong) normal field scores ~0.
    # One clone per splat is not enough (measured 2026-10-08: r_nu 0.61 -> 0.45).
    normals = torch.nn.functional.normalize(torch.randn(2000, 3), dim=1)
    means, quats, log_scales = _sphere_splats(2000, 2.0, normals)
    clean = jet_residuals(means, quats, log_scales, knn_indices(means, 16))[0].mean()
    cloned = [torch.cat([x] * 8) for x in (means, quats, log_scales)]
    collapsed = jet_residuals(*cloned, knn_indices(cloned[0], 16))[0].mean()
    assert float(collapsed) < 1e-6 * float(clean)


def test_prior_gradients_reach_quats_and_means():
    normals = torch.nn.functional.normalize(torch.randn(300, 3), dim=1)
    means, quats, log_scales = _sphere_splats(300, 1.0, normals)
    means.requires_grad_(True)
    quats.requires_grad_(True)
    prior = JetPrior(1.0, order=2)
    loss = prior(means, quats, log_scales, step=0)
    loss.backward()
    assert torch.isfinite(loss) and float(quats.grad.abs().sum()) > 0
    assert float(means.grad.abs().sum()) > 0
    prior(means, quats, log_scales, step=1)
    assert prior.refreshes == 1


def test_trainer_jet_term_is_opt_in():
    from rtgs.data.synthetic import make_synthetic_scene
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = make_synthetic_scene(n_gaussians=24, n_cameras=3, image_size=16, seed=4)
    init = scene.gt_gaussians.detach()
    base = dict(
        iterations=4,
        rasterizer="torch",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(every=2, start_iter=1, stop_iter=1000, grad_threshold=1e-9),
        eval_every=4,
        ssim_lambda=0.0,
    )
    _, off = Trainer(TrainConfig(**base)).train(scene, init)
    _, on = Trainer(TrainConfig(**base)).train(scene, init, jet_prior=JetPrior(0.1))
    assert all(term["jet_regularization"] == 0.0 for term in off["loss_terms"])
    assert all(term["jet_regularization"] > 0.0 for term in on["loss_terms"])
    assert off["loss"] != on["loss"]


def test_cli_flags_build_the_prior():
    from rtgs.cli import _jet_prior

    class Args:
        jet_lambda, jet_order, jet_prior, jet_start = 0.0, 2, "jet", 7500

    assert _jet_prior(Args) is None
    Args.jet_lambda, Args.jet_order = 0.1, 1
    prior = _jet_prior(Args)
    assert prior.weight == 0.1 and prior.order == 1
    Args.jet_prior, Args.jet_start, Args.jet_terms = "field", 300, "normal+centre"
    field = _jet_prior(Args)
    assert isinstance(field, FieldPrior) and field.start == 300 and field.weight == 0.1
    assert field.terms == "normal+centre"


def _field(means, quats, log_scales, opacities=None):
    h0 = mean_neighbour_distance(means)
    pairs = radius_pairs(means, 3 * h0)
    opacities = torch.ones(means.shape[0]) if opacities is None else opacities
    return field_residuals(means, quats, log_scales, opacities, h0, pairs), h0


def _surfels(n, radius, normals=None):
    means, quats, log_scales = _sphere_splats(n, radius, normals)
    log_scales[:, 2] = -math.inf  # 2DGS: flat by construction
    return means, quats, log_scales


def _plane(k, rotation=None, offset=0.0):
    """k x k unit-spaced surfels in the plane z = offset (rotated), exact normals."""
    rotation = torch.eye(3) if rotation is None else rotation
    u = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    points = torch.stack([uu.flatten(), vv.flatten(), torch.full((k * k,), offset)], 1)
    quats = rotmat_to_quat(rotation.expand(k * k, 3, 3).contiguous())
    log_scales = torch.tensor([math.log(0.4), math.log(0.4), -math.inf]).expand(k * k, 3).clone()
    return points @ rotation.T, quats, log_scales


def test_field_prior_sphere_exact_normals_small_and_centres_order_h2():
    radius = 2.0
    ratios = []
    for n in (1000, 4000):  # h halves
        means, quats, log_scales = _surfels(n, radius)
        log_scales[:, :2] = torch.log(0.3 * mean_neighbour_distance(means))[:, None]
        (r_nu, r_c, field), h0 = _field(means, quats, log_scales)
        h2 = float((h0.mean() / radius) ** 2)
        assert bool(field["ok"].all())  # a sphere is well posed everywhere
        assert float(r_nu.mean()) < 1e-2 * h2  # r_nu ~ O(h^4)
        # sagitta of the 1.5 h window: (c - m).n ~ 1.6 h^2 / R, so r_c ~ 2.5 (h/R)^2
        assert float(r_c.mean()) < 5 * h2
        assert float((field["normal"] * means / radius).sum(-1).abs().min()) > 0.999
        ratios.append(float(r_c.mean()))
    assert 2.0 < ratios[0] / ratios[1] < 8.0  # quarter the residual when h halves


def _sphere_gradients(n):
    means, quats, log_scales = _surfels(n, 2.0)
    log_scales[:, :2] = torch.log(0.3 * mean_neighbour_distance(means))[:, None]
    means.requires_grad_(True)
    quats.requires_grad_(True)
    (r_nu, r_c, _), _ = _field(means, quats, log_scales)
    exact = torch.autograd.grad(r_nu.sum(), quats, retain_graph=True)[0].norm(dim=-1)
    descent = -torch.autograd.grad(r_c.sum(), means)[0]
    radial = (descent * means.detach()).sum(-1) / 2.0
    tilted = _surfels(n, 2.0, torch.nn.functional.normalize(torch.randn(n, 3), dim=1))
    tilted[1].requires_grad_(True)
    tilted[2][:, :2] = log_scales[:, :2]
    (t_nu, _, _), _ = _field(*tilted)
    random = torch.autograd.grad(t_nu.sum(), tilted[1])[0].norm(dim=-1)
    return float(exact.median()), float(random.median()), float((radial < 0).float().mean())


def test_field_prior_sphere_gradients_normal_term_zero_centre_term_inward():
    """Why the centre term is only an ablation: on exact normals the normal term's torque is only
    the field's own O(h^2) discretization error (vanishing with h), but the centre term pushes
    every correctly placed centre inward (shrinkage, O(H))."""
    torch.manual_seed(0)
    coarse, random, inward = _sphere_gradients(1000)
    fine, _, inward_fine = _sphere_gradients(4000)
    assert coarse < 0.05 * random and fine < coarse / 2
    assert inward > 0.95 and inward_fine > 0.95  # gradient descent moves centres inward


def test_field_prior_offset_plane_pulls_back_and_untilts():
    means, quats, log_scales = _plane(15)
    centre = 7 * 15 + 7
    means[centre, 2] = 0.3
    tilt = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(0.5), -math.sin(0.5)], [0.0, 0.0, 0.0]])
    tilt[2] = torch.linalg.cross(tilt[0], tilt[1])
    quats[centre] = rotmat_to_quat(tilt.T[None])[0]
    means.requires_grad_(True)
    quats.requires_grad_(True)
    (r_nu, r_c, field), _ = _field(means, quats, log_scales)
    assert bool(field["ok"][centre]) and float(r_c[centre]) > 0 and float(r_nu[centre]) > 0.1
    g_means, g_quats = torch.autograd.grad((r_nu + r_c).sum(), [means, quats])
    assert float(g_means[centre, 2]) > 0  # descent moves the offset centre back to z = 0
    with torch.no_grad():
        stepped = quats - 0.05 * g_quats
    (after, _, _), _ = _field(means.detach(), stepped, log_scales)
    assert float(after[centre]) < float(r_nu[centre])


def test_field_prior_two_parallel_sheets_and_crossing_sheets():
    """Guard behaviour, stated: opposing parallel sheets are *not* masked (their field normal is
    still right, so r_nu = 0, but the centroid lies between them and r_c pulls them together);
    crossing sheets near the intersection line are masked."""
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    lower, upper = _plane(20), _plane(20, flip, offset=-1.5)
    sheets = [torch.cat([a, b]) for a, b in zip(lower, upper, strict=True)]
    sheets[0].requires_grad_(True)
    (r_nu, r_c, field), _ = _field(*sheets)
    inner = (sheets[0][:, :2].detach().abs() < 6).all(-1)
    assert float(1 - field["ok"][inner].float().mean()) < 0.05
    assert float(r_nu[inner].max()) < 1e-4  # float32 round-off
    pull = -torch.autograd.grad(r_c.sum(), sheets[0])[0][:, 2]
    upper_rows = torch.arange(sheets[0].shape[0]) >= 400
    assert float(pull[inner & ~upper_rows].mean()) > 0 > float(pull[inner & upper_rows].mean())
    turn = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    crossing = [torch.cat([a, b]) for a, b in zip(_plane(24), _plane(24, turn), strict=True)]
    (_, _, field), _ = _field(*crossing)
    line = (crossing[0][:, [0, 2]].abs() < 2).all(-1) & (crossing[0][:, 1].abs() < 8)
    assert float(1 - field["ok"][line].float().mean()) > 0.3


def test_field_prior_invariant_under_uniform_replication_at_fixed_bandwidth():
    """Duplicating every splat (h0 copied) leaves n, m, the eigengap and the residuals unchanged;
    the support ratio sum a / max a doubles (copies count), which only loosens the guard."""
    torch.manual_seed(0)
    normals = torch.nn.functional.normalize(
        torch.randn(600, 3) * 0.3 + _sphere_splats(600, 1.0)[0], dim=1
    )
    means, quats, log_scales = _surfels(600, 1.0, normals)
    log_scales[:, :2] = math.log(0.02)
    opacities = torch.rand(600) * 0.9 + 0.1
    h0 = mean_neighbour_distance(means)
    once = field_estimate(means, quats, log_scales, opacities, h0, radius_pairs(means, 3 * h0))
    twice = [torch.cat([t, t]) for t in (means, quats, log_scales, opacities, h0)]
    both = field_estimate(*twice, radius_pairs(twice[0], 3 * twice[4]))
    for key in ("centroid", "gap"):
        assert torch.allclose(both[key][:600], once[key], atol=1e-5)
        assert torch.allclose(both[key][600:], once[key], atol=1e-5)
    assert torch.allclose((both["normal"][:600] * once["normal"]).sum(-1).abs(), torch.ones(600))
    assert torch.allclose(both["support"][:600], 2 * once["support"], rtol=1e-5)
    pairs = radius_pairs(means, 3 * h0)
    r_nu, r_c, _ = field_residuals(means, quats, log_scales, opacities, h0, pairs, min_support=0)
    d_nu, d_c, _ = field_residuals(*twice, radius_pairs(twice[0], 3 * twice[4]), min_support=0)
    assert abs(float((d_nu + d_c).mean() - (r_nu + r_c).mean())) < 1e-5


def test_field_prior_non_uniform_duplication_is_reported_not_invariant():
    """Duplicating a random half changes rho non-uniformly; the change is reported, not bounded."""
    torch.manual_seed(1)
    normals = torch.nn.functional.normalize(
        torch.randn(800, 3) * 0.3 + _sphere_splats(800, 1.0)[0], dim=1
    )
    means, quats, log_scales = _surfels(800, 1.0, normals)
    log_scales[:, :2] = math.log(0.02)
    (r_nu, r_c, _), h0 = _field(means, quats, log_scales)
    pick = torch.randperm(800)[:400]
    grown = [torch.cat([t, t[pick]]) for t in (means, quats, log_scales, torch.ones(800), h0)]
    g_nu, g_c, _ = field_residuals(*grown, radius_pairs(grown[0], 3 * grown[4]))
    change = float(g_nu[:800].mean() - r_nu.mean()), float(g_c[:800].mean() - r_c.mean())
    before, after = float(r_nu.mean()), float(g_nu[:800].mean())
    print(f"non-uniform duplication: mean r_nu {before:.4f} -> {after:.4f}, ", end="")
    print(f"mean r_c {float(r_c.mean()):.4f} -> {float(g_c[:800].mean()):.4f}")
    assert all(math.isfinite(x) for x in change)


def test_field_prior_random_normals_are_order_one():
    normals = torch.nn.functional.normalize(torch.randn(2000, 3), dim=1)
    means, quats, log_scales = _surfels(2000, 2.0, normals)
    log_scales[:, :2] = math.log(0.01)
    (r_nu, _, _), _ = _field(means, quats, log_scales)
    assert float(r_nu.mean()) > 0.4  # E[1 - (nu.n)^2] = 2/3 for random nu


def test_field_prior_terms_gradients_start_freeze_and_topology():
    normals = torch.nn.functional.normalize(torch.randn(300, 3), dim=1)
    means, quats, log_scales = _surfels(300, 1.0, normals)
    means.requires_grad_(True)
    quats.requires_grad_(True)
    opacities = torch.full((300,), 0.5, requires_grad=True)
    prior = FieldPrior(1.0, start=5, refresh_every=2, log_every=2)
    assert float(prior(means, quats, log_scales, 4, opacities=opacities)) == 0.0
    assert prior.h0 is None
    loss = prior(means, quats, log_scales, 5, opacities=opacities)
    loss.backward()
    assert torch.isfinite(loss) and float(quats.grad.abs().sum()) > 0
    assert means.grad is None and opacities.grad is None  # normal term: quats only
    assert 0.0 <= prior.stats["guarded_fraction"] <= 1.0 and prior.log[0]["step"] == 5
    centre = FieldPrior(1.0, start=5, terms="normal+centre")
    centre(means, quats, log_scales, 5, opacities=opacities).backward()
    assert float(means.grad.abs().sum()) > 0
    h0 = prior.h0.clone()
    prior(means.detach() * 2, quats, log_scales, 6, opacities=opacities)
    assert torch.equal(prior.h0, h0) and prior.refreshes == 1  # frozen; refresh at 7
    prior(means, quats, log_scales, 7, opacities=opacities)
    assert prior.refreshes == 2
    with pytest.raises(RuntimeError, match="count changed"):
        prior(means[:-1], quats[:-1], log_scales[:-1], 8, opacities=opacities[:-1])
    with pytest.raises(ValueError):
        FieldPrior(1.0, terms="centre")
