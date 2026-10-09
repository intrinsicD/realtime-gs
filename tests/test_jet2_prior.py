"""v3 2-jet prior: analytic checks only (plan docs/TASK_jet2_synthetic_sphere_ellipsoid.md P1)."""

import math

import pytest
import torch

from rtgs.core.gaussians3d import rotmat_to_quat
from rtgs.optim.jet2_prior import Jet2Prior, exact_radius_pairs, jet2_residuals


def _surfels(points: torch.Tensor, normals: torch.Tensor, sigma: float, aniso: float = 1.0):
    """Surfels at ``points`` with frame normal ``normals``, tangent scales (sigma, sigma/aniso)."""
    nu = normals.double()
    helper = torch.where(
        nu[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0])
    ).double()
    t1 = torch.nn.functional.normalize(torch.linalg.cross(nu, helper), dim=1)
    t2 = torch.linalg.cross(nu, t1)
    quats = rotmat_to_quat(torch.stack([t1, t2, nu], dim=2)).double()
    scales = torch.tensor([sigma, sigma / aniso, 0.0]).double().log().expand(len(points), 3)
    return points.double().clone(), quats, scales.clone(), torch.ones(len(points)).double()


def _fibonacci(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64)
    z = 1 - (2 * i + 1) / n
    r = (1 - z * z).sqrt()
    phi = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], 1)


def _sphere(n: int, aniso: float = 1.0):
    p = _fibonacci(n)
    return _surfels(p, p, 0.75 * math.sqrt(4 * math.pi / n), aniso)  # sigma = 0.75 spacing


def _residuals(splats, order=2, **kw):
    means, quats, scales, opac = splats
    sig = scales.max(1).values.exp()
    return jet2_residuals(
        means, quats, scales, opac, sig, exact_radius_pairs(means, 3 * sig), order=order, **kw
    )


def test_exact_sphere_jet2_fits_and_beats_jet1():
    splats = _sphere(2000)
    r_nu2, r_c2, info = _residuals(splats, 2)
    r_nu1, r_c1, _ = _residuals(splats, 1)
    assert bool(info["ok"].all())
    eye = torch.eye(2, dtype=torch.float64).expand_as(info["S"])
    assert float((info["S"] - eye).abs().max()) < 0.05  # S = +I/R, R = 1
    assert float(r_nu2.mean()) < 1e-2 * float(r_nu1.mean())
    assert float(r_c2.mean()) < 1e-2 * float(r_c1.mean())


def test_exact_sphere_jet2_residual_converges_at_sixth_order():
    # On the unit sphere nu_j = c_j and e_i is orthogonal to c_i, so e_i . nu_j = s_ij exactly:
    # r_nu vanishes identically. The height is cos(t) - 1 + sin(t)^2/2 = -t^4/8 + ..., so with
    # sigma proportional to spacing r_c scales as sigma^8/sigma^2 = sigma^6: /64 per 4x points.
    (nu_c, c_c, _), (nu_f, c_f, _) = _residuals(_sphere(1000)), _residuals(_sphere(4000))
    assert float(nu_c.max()) < 1e-12 and float(nu_f.max()) < 1e-12
    assert float(c_c.mean()) / float(c_f.mean()) > 48  # 64 at sixth order, 32 at fifth


def _cap_gradient_z(order: int) -> float:
    """Mean dz of interior centres under the prior's descent direction on an exact 40 deg cap."""
    p = _fibonacci(4000)
    p = p[p[:, 2] > math.cos(math.radians(40))]
    means, quats, scales, opac = _surfels(p, p, 0.75 * math.sqrt(4 * math.pi / 4000))
    means.requires_grad_(True)
    sig = scales.max(1).values.exp()
    r_nu, r_c, _ = jet2_residuals(
        means, quats, scales, opac, sig, exact_radius_pairs(means, 3 * sig), order=order
    )
    (r_nu + r_c).mean().backward()
    interior = p[:, 2] > math.cos(math.radians(30))  # boundary band held fixed
    return float(-means.grad[interior, 2].mean())


def test_cap_jet1_flattens_jet2_keeps_curvature():
    jet1, jet2 = _cap_gradient_z(1), _cap_gradient_z(2)
    assert jet1 < 0  # descent lowers the apex region toward the rim plane: flattening
    assert abs(jet2) < 1e-2 * abs(jet1)


def test_flipping_half_the_normals_changes_nothing():
    means, quats, scales, opac = _sphere(1500)
    flip = torch.arange(len(means)) % 2 == 0
    p = means.clone()
    nu = torch.where(flip[:, None], -p, p)
    flipped = _surfels(p, nu, float(scales[0, 0].exp()))
    for order in (1, 2):
        a = sum(r.mean() for r in _residuals((means, quats, scales, opac), order)[:2])
        b = sum(r.mean() for r in _residuals(flipped, order)[:2])
        assert float(abs(a - b)) <= 1e-9 * float(a)


def test_uniform_clones_change_nothing():
    rng = torch.Generator().manual_seed(3)
    means, quats, scales, opac = _sphere(300)
    means = means + 0.02 * torch.randn(means.shape, generator=rng, dtype=means.dtype)
    base = (means, quats, scales, opac)
    cloned = tuple(torch.cat([x] * 8) for x in base)
    for order in (1, 2):
        a = sum(r.mean() for r in _residuals(base, order)[:2])
        b = sum(r.mean() for r in _residuals(cloned, order)[:2])
        assert float(abs(a - b)) <= 1e-6 * float(a)
    # the gate quantity itself must not count copies (a neighbour count would grow 8x)
    mass, mass8 = _residuals(base)[2]["mass"], _residuals(cloned)[2]["mass"][: len(means)]
    assert float((mass - mass8).abs().max()) <= 1e-9


def test_needle_splats_keep_full_shape_operator():
    _, _, info = _residuals(_sphere(2000, aniso=10.0))
    eye = torch.eye(2, dtype=torch.float64).expand_as(info["S"])
    assert bool(info["ok"].all())
    assert float((info["S"] - eye).abs().max()) < 0.05


def test_isolated_splat_is_masked_and_finite():
    means, quats, scales, opac = _sphere(1000)
    far = _surfels(torch.tensor([[5.0, 0, 0]]), torch.tensor([[1.0, 0, 0]]), 0.01)
    splats = tuple(
        torch.cat([x, y]) for x, y in zip((means, quats, scales, opac), far, strict=True)
    )
    splats[0].requires_grad_(True)
    r_nu, r_c, info = _residuals(splats)
    assert not bool(info["ok"][-1]) and bool(info["ok"][:-1].all())
    assert float(info["mass"][-1]) == 0.0  # its own weight is not neighbour mass
    loss = (r_nu + r_c).mean()
    loss.backward()
    assert torch.isfinite(loss) and float(splats[0].grad[-1].abs().sum()) == 0.0


def test_prior_schedule_gradients_and_topology_guard():
    means, quats, scales, opac = _sphere(800)
    for t in (means, quats, scales, opac):
        t.requires_grad_(True)
    prior = Jet2Prior(1.0, order=2, start=10)
    off = prior(means, quats, scales, step=9, opacities=opac)
    assert float(off) == 0.0 and not off.requires_grad and prior.sig is None
    prior(means, quats, scales, step=10, opacities=opac).backward()
    assert float(means.grad.abs().sum()) > 0 and float(quats.grad.abs().sum()) > 0
    assert scales.grad is None and opac.grad is None
    with pytest.raises(RuntimeError, match="count changed"):
        prior(means[:-1], quats[:-1], scales[:-1], step=11, opacities=opac[:-1])


def test_sparse_well_conditioned_neighbourhood_is_masked_by_mass():
    # three neighbours on a triangle at 2.5 sigma: the 3x3 design is well conditioned, but the
    # neighbours hold ~0.13 of the kernel mass, so only the mass gate can reject the fit.
    sigma = 0.1
    ang = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=torch.float64)
    ring = torch.stack([2.5 * sigma * ang.cos(), 2.5 * sigma * ang.sin(), torch.zeros(3)], 1)
    points = torch.cat([torch.zeros(1, 3, dtype=torch.float64), ring])
    normals = torch.tensor([[0, 0, 1.0]]).double().expand(4, 3)
    _, _, info = _residuals(_surfels(points, normals, sigma), min_conditioning=0.0)
    assert float(info["mass"][0]) < 0.5 and not bool(info["ok"][0])


def test_collinear_massive_neighbourhood_is_masked_by_conditioning():
    # nine neighbours on a line hold plenty of mass, but S22 is unobservable from them
    sigma = 0.1
    line = torch.zeros(10, 3, dtype=torch.float64)
    line[:, 0] = torch.linspace(-0.5, 0.5, 10, dtype=torch.float64) * sigma * 4
    normals = torch.tensor([[0, 0, 1.0]]).double().expand(10, 3)
    _, _, info = _residuals(_surfels(line, normals, sigma), min_mass=0.0)
    assert float(info["mass"][5]) > 0.5 and not bool(info["ok"][5])


def test_float32_tiny_masked_splat_has_finite_gradients():
    means, quats, scales, opac = (x.float() for x in _sphere(100))
    tiny = _surfels(torch.tensor([[3.0, 0, 0]]), torch.tensor([[1.0, 0, 0]]), 1e-8)
    splats = [
        torch.cat([x, y.float()]) for x, y in zip((means, quats, scales, opac), tiny, strict=True)
    ]
    scales = splats[2]
    scales[:, 2] = float("-inf")  # 2DGS surfels
    splats[0].requires_grad_(True)
    splats[1].requires_grad_(True)
    sig = scales.max(1).values.exp()
    r_nu, r_c, info = jet2_residuals(*splats, sig, exact_radius_pairs(splats[0], 3 * sig))
    loss = (r_nu + r_c).mean()
    loss.backward()
    assert not bool(info["ok"][-1]) and torch.isfinite(loss)
    assert bool(torch.isfinite(splats[0].grad).all() and torch.isfinite(splats[1].grad).all())


def test_exact_pairs_keep_self_for_tiny_float32_splats():
    # the matmul cdist path reports self distances up to ~5e-4 for 14 of these 100 points
    points = _fibonacci(100).float()
    rows, cols = exact_radius_pairs(points, torch.full((100,), 3e-5))
    assert bool((rows == cols).all()) and len(rows) == 100


@pytest.mark.cuda
def test_cuda_order2_runs_and_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    splats = [x.float() for x in _sphere(500)]
    sig = splats[2].max(1).values.exp()
    cpu = jet2_residuals(*splats, sig, exact_radius_pairs(splats[0], 3 * sig))
    gpu_in = [x.cuda() for x in splats]
    gpu = jet2_residuals(*gpu_in, sig.cuda(), exact_radius_pairs(gpu_in[0], 3 * sig.cuda()))
    for a, b in zip(cpu[:2], gpu[:2], strict=True):
        assert torch.allclose(a, b.cpu(), rtol=1e-3, atol=1e-9)
