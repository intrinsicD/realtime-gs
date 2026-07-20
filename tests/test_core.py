"""Core math: SH, cameras, gaussian containers, metrics."""

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat, rotmat_to_quat
from rtgs.core.metrics import image_metrics, masked_psnr, psnr, ssim
from rtgs.core.sh import (
    DEFAULT_SMU1_MU,
    activate_sh_color,
    eval_sh,
    eval_sh_preactivation,
    num_sh_bases,
    rgb_to_sh,
    sh_to_rgb,
)


def test_sh_roundtrip_degree0():
    rgb = torch.rand(10, 3)
    dirs = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    sh = torch.zeros(10, 1, 3)
    sh[:, 0] = rgb_to_sh(rgb)
    out = eval_sh(0, sh, dirs)
    assert torch.allclose(out, rgb, atol=1e-5)
    assert torch.allclose(sh_to_rgb(sh[:, 0]), rgb, atol=1e-5)


def test_sh_higher_degree_view_dependence():
    sh = torch.zeros(1, num_sh_bases(2), 3)
    sh[:, 0] = 0.5
    sh[:, 3] = 0.8  # an x-direction band
    d1 = torch.tensor([[1.0, 0.0, 0.0]])
    d2 = torch.tensor([[-1.0, 0.0, 0.0]])
    assert not torch.allclose(eval_sh(2, sh, d1), eval_sh(2, sh, d2))


def test_default_sh_activation_matches_explicit_hard_floor():
    sh = torch.randn(7, num_sh_bases(3), 3)
    dirs = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    preactivation = eval_sh_preactivation(3, sh, dirs)
    assert torch.equal(eval_sh(3, sh, dirs), preactivation.clamp_min(0.0))
    assert torch.equal(eval_sh(3, sh, dirs, activation="hard"), preactivation.clamp_min(0.0))


def test_smu1_is_nonnegative_smooth_and_one_level_bounded():
    x = torch.linspace(-2.0, 2.0, 1001, dtype=torch.float64, requires_grad=True)
    hard = activate_sh_color(x, "hard")
    smooth = activate_sh_color(x, "smu1")
    difference = smooth - hard
    assert bool((smooth > 0.0).all())
    assert float(difference.detach().min()) >= 0.0
    assert float(difference.detach().max()) <= DEFAULT_SMU1_MU / 2.0 + 1e-12
    smooth.sum().backward()
    assert x.grad is not None
    assert bool((x.grad > 0.0).all())
    assert bool((x.grad < 1.0).all())


def test_smu1_negative_only_straight_through_control():
    x = torch.tensor([-0.2, -0.01, 0.01, 0.2], dtype=torch.float64, requires_grad=True)
    output = activate_sh_color(x, "hard_forward_smu1_negative_gradient")
    hard = x.detach().clamp_min(0.0)
    assert torch.equal(output.detach(), hard)
    output.sum().backward()
    expected_negative = 0.5 * (
        1.0 + x.detach()[:2] / torch.sqrt(x.detach()[:2].square() + DEFAULT_SMU1_MU**2)
    )
    assert torch.allclose(x.grad[:2], expected_negative, atol=1e-12, rtol=0.0)
    assert torch.equal(x.grad[2:], torch.ones_like(x.grad[2:]))


def test_unknown_or_invalid_sh_activation_is_rejected():
    x = torch.zeros(1)
    with pytest.raises(ValueError, match="unknown SH color activation"):
        activate_sh_color(x, "mystery")
    with pytest.raises(ValueError, match="finite and positive"):
        activate_sh_color(x, "smu1", smu1_mu=0.0)


def test_camera_project_unproject_roundtrip():
    cam = Camera.look_at(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), width=64, height=48)
    pts = torch.randn(50, 3) * 0.4
    uv, z = cam.project(pts)
    assert (z > 0).all()
    back = cam.unproject(uv, z)
    assert torch.allclose(back, pts, atol=1e-4)


def test_camera_position_and_rays():
    eye = torch.tensor([1.0, -2.0, 3.0])
    cam = Camera.look_at(eye, torch.zeros(3), width=32, height=32)
    assert torch.allclose(cam.position, eye, atol=1e-5)
    uv = torch.tensor([[16.0, 16.0], [3.0, 28.0]])
    o, d = cam.pixel_rays(uv)
    depth = torch.tensor([2.0, 1.5])
    pts_ray = o[None, :] + depth[:, None] * d
    pts_up = cam.unproject(uv, depth)
    assert torch.allclose(pts_ray, pts_up, atol=1e-5)


def test_gaussians2d_covariance_psd_and_inverse():
    g = Gaussians2D(
        xy=torch.rand(20, 2) * 30,
        chol=torch.stack([torch.rand(20) + 0.5, torch.randn(20), torch.rand(20) + 0.5], -1),
        color=torch.rand(20, 3),
        weight=torch.rand(20),
    )
    cov = g.covariance()
    eye = torch.eye(2).expand(20, 2, 2)
    assert torch.allclose(cov @ g.inverse_covariance(), eye, atol=1e-4)
    evals = torch.linalg.eigvalsh(cov)
    assert (evals > 0).all()


def test_quat_rotmat_roundtrip():
    q = torch.nn.functional.normalize(torch.randn(100, 4), dim=-1)
    r = quat_to_rotmat(q)
    assert torch.allclose(torch.linalg.det(r), torch.ones(100), atol=1e-5)
    q2 = rotmat_to_quat(r)
    # q and -q are the same rotation.
    sign = torch.sign((q * q2).sum(-1, keepdim=True))
    assert torch.allclose(q, q2 * sign, atol=1e-4)


def test_gaussians3d_from_covs_roundtrip():
    n = 30
    a = torch.randn(n, 3, 3)
    covs = a @ a.transpose(-1, -2) + 0.01 * torch.eye(3)
    g = Gaussians3D.from_means_covs(
        means=torch.randn(n, 3),
        covs=covs,
        colors=torch.rand(n, 3),
        opacity=torch.rand(n).clamp(0.1, 0.9),
    )
    assert torch.allclose(g.covariance(), covs, atol=1e-4, rtol=1e-3)


def test_gaussians3d_ply_roundtrip(tmp_path):
    n = 12
    g = Gaussians3D.from_means_covs(
        means=torch.randn(n, 3),
        covs=torch.eye(3).expand(n, 3, 3) * 0.04,
        colors=torch.rand(n, 3),
        opacity=torch.rand(n).clamp(0.05, 0.95),
        sh_degree=1,
    )
    path = tmp_path / "test.ply"
    g.save_ply(path)
    g2 = Gaussians3D.load_ply(path)
    assert torch.allclose(g.means, g2.means, atol=1e-5)
    assert torch.allclose(g.sh, g2.sh, atol=1e-5)
    assert torch.allclose(g.opacity, g2.opacity, atol=1e-4)
    assert torch.allclose(g.log_scales, g2.log_scales, atol=1e-5)


def test_gaussians3d_empty_degree_zero_ply_roundtrip(tmp_path):
    g = Gaussians3D(
        means=torch.empty(0, 3),
        quats=torch.empty(0, 4),
        log_scales=torch.empty(0, 3),
        opacity=torch.empty(0),
        sh=torch.empty(0, 1, 3),
    )
    path = tmp_path / "empty.ply"
    g.save_ply(path)

    loaded = Gaussians3D.load_ply(path)

    assert loaded.n == 0
    assert loaded.sh.shape == (0, 1, 3)


def test_gaussians3d_npz_roundtrip(tmp_path):
    g = Gaussians3D.from_means_covs(
        means=torch.randn(5, 3),
        covs=torch.eye(3).expand(5, 3, 3) * 0.01,
        colors=torch.rand(5, 3),
        opacity=torch.rand(5).clamp(0.1, 0.9),
    )
    g.save_npz(tmp_path / "g.npz")
    g2 = Gaussians3D.load_npz(tmp_path / "g.npz")
    assert torch.allclose(g.means, g2.means)
    assert torch.allclose(g.opacity, g2.opacity)


def test_metrics_sanity():
    img = torch.rand(24, 24, 3)
    assert psnr(img, img) > 60
    assert ssim(img, img) > 0.999
    noisy = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    assert psnr(noisy, img) < 25
    assert ssim(noisy, img) < 0.99
    assert math.isfinite(psnr(torch.zeros(8, 8, 3), torch.ones(8, 8, 3)))


def test_ssim_differentiable():
    a = torch.rand(16, 16, 3, requires_grad=True)
    b = torch.rand(16, 16, 3)
    (1 - ssim(a, b)).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_foreground_metrics_do_not_reward_black_canvas():
    target = torch.zeros(32, 32, 3)
    target[12:20, 12:20] = 1.0
    mask = torch.zeros(32, 32)
    mask[12:20, 12:20] = 1.0
    pred = torch.zeros_like(target)
    values = image_metrics(pred, target, mask)
    assert values["psnr_full"] > values["psnr_fg"] + 10.0
    assert values["psnr_fg"] == masked_psnr(pred, target, mask)
    with pytest.raises(ValueError, match="no foreground"):
        image_metrics(pred, target, torch.zeros_like(mask))


def test_shape_validation():
    with pytest.raises(ValueError):
        Gaussians2D(torch.zeros(3, 2), torch.zeros(2, 3), torch.zeros(3, 3), torch.zeros(3))
    with pytest.raises(ValueError):
        Gaussians3D(
            torch.zeros(3, 3),
            torch.zeros(3, 4),
            torch.zeros(3, 3),
            torch.zeros(3),
            torch.zeros(3, 2, 3),  # 2 is not a valid SH basis count
        )
