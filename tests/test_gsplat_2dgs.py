"""gsplat 2DGS surfel backend: CPU import/guards, CUDA conventions and an end-to-end train."""

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat, rotmat_to_quat
from rtgs.render.base import RenderOutput, get_rasterizer
from rtgs.render.gsplat_2dgs_backend import SurfelRegularization, surfel_terms


def test_module_is_cpu_importable_and_terms_need_2dgs_maps():
    reg = SurfelRegularization(100.0, 0.05)
    assert reg.weights(3000) == (0.0, 0.0)
    assert reg.weights(3001) == (100.0, 0.0)
    assert reg.weights(7001) == (100.0, 0.05)
    plain = RenderOutput(torch.zeros(4, 4, 3), torch.zeros(4, 4), torch.zeros(4, 4))
    with pytest.raises(RuntimeError, match="gsplat-2dgs"):
        surfel_terms(plain)


def test_trainer_rejects_surfel_terms_without_2dgs(tiny_scene):
    from rtgs.optim.trainer import TrainConfig, Trainer

    config = TrainConfig(iterations=1, rasterizer="torch", densify=False, eval_every=1)
    with pytest.raises(ValueError, match="gsplat-2dgs"):
        Trainer(config).train(
            tiny_scene,
            tiny_scene.gt_gaussians,
            surfel_regularization=SurfelRegularization(1.0, 1.0),
        )


def _facing_surfel(device, third_log_scale):
    # One flat surfel at z = 3 in a camera at the origin looking down +z; normal = +-z.
    log_scales = torch.tensor([[math.log(0.4), math.log(0.3), third_log_scale]], device=device)
    return Gaussians3D(
        means=torch.tensor([[0.05, -0.05, 3.0]], device=device),
        quats=torch.tensor([[0.96, 0.0, 0.28, 0.0]], device=device),  # tilted about y
        log_scales=log_scales,
        opacity=torch.tensor([0.9], device=device),
        sh=torch.full((1, 1, 3), 0.8, device=device),
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_2dgs_conventions_match_3dgs_flat_limit():
    pytest.importorskip("gsplat")
    camera = Camera(60.0, 60.0, 32.0, 32.0, 64, 64, torch.eye(3), torch.zeros(3)).to("cuda")
    surfel = _facing_surfel("cuda", -math.inf)
    surfel.means.requires_grad_(True)
    out = get_rasterizer("gsplat-2dgs").render(surfel, camera)
    flat3d = get_rasterizer("gsplat").render(_facing_surfel("cuda", math.log(1e-4)), camera)
    assert float((out.alpha - flat3d.alpha).abs().mean().detach()) < 0.02
    covered = out.alpha > 0.5
    assert int(covered.sum()) > 50
    # Rendered normals are world-frame and camera-facing; depth normals agree with them.
    rendered = torch.nn.functional.normalize(out.normals[covered], dim=-1)
    normal = quat_to_rotmat(surfel.quats.detach())[0, :, 2]
    assert float((rendered @ normal).abs().min()) > 0.99
    assert float(rendered[:, 2].max()) < 0  # faces the camera (camera looks down +z)
    # gsplat 1.5.3 composites each surfel's *centre* depth (not the per-pixel ray-surfel
    # intersection), so one isolated tilted surfel has a fronto-parallel depth normal ...
    depth_normal = out.normals_from_depth[covered]
    assert float(depth_normal[:, 2].median()) < -0.99
    # ... while a sheet tiled by many small surfels recovers the tilt (the regime that matters).
    angle = 0.6
    rot = torch.tensor(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )
    grid = torch.linspace(-1.2, 1.2, 30)
    u, v = torch.meshgrid(grid, grid, indexing="ij")
    centres = torch.stack([u.flatten(), v.flatten(), torch.zeros(900)], 1) @ rot.T
    spacing = math.log(2.4 / 29)
    sheet = Gaussians3D(
        means=centres + torch.tensor([0.0, 0.0, 3.0]),
        quats=rotmat_to_quat(rot.expand(900, 3, 3).contiguous()),
        log_scales=torch.tensor([spacing, spacing, -math.inf]).expand(900, 3).clone(),
        opacity=torch.full((900,), 0.9),
        sh=torch.full((900, 1, 3), 0.5),
    ).to("cuda")
    with torch.no_grad():
        tiled = get_rasterizer("gsplat-2dgs").render(sheet, camera)
    inside = tiled.alpha > 0.5
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    unit = torch.nn.functional.normalize(tiled.normals[inside], dim=-1)
    assert float((unit @ rot[:, 2].cuda()).max()) < -0.99  # camera-facing sheet normal
    agree = (unit * tiled.normals_from_depth[inside]).sum(-1)
    assert float(agree.median()) > 0.95
    assert torch.isfinite(out.distortion).all() and float(out.distortion.min()) >= 0
    depth = out.depth[covered] / out.alpha[covered]
    assert float((depth - 3.0).abs().median()) < 0.2
    distortion, normal_error = surfel_terms(out)
    (out.color.mean() + distortion + normal_error).backward()
    assert out.means2d.grad is not None and torch.isfinite(surfel.means.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_2dgs_training_with_regularizers_and_field_prior(tiny_scene):
    pytest.importorskip("gsplat")
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.jet_prior import FieldPrior
    from rtgs.optim.trainer import TrainConfig, Trainer

    config = TrainConfig(
        iterations=40,
        rasterizer="gsplat-2dgs",
        device="cuda",
        densify=True,
        density_strategy="gsplat-default",
        density=DensityConfig(start_iter=5, stop_iter=20, every=5, grad_threshold=1e-9),
        eval_every=10,
        ssim_lambda=0.0,
    )
    prior = FieldPrior(0.1, start=20, refresh_every=5)
    model, history = Trainer(config).train(
        tiny_scene,
        tiny_scene.gt_gaussians,
        jet_prior=prior,
        surfel_regularization=SurfelRegularization(1.0, 0.05, 2, 4),
    )
    assert torch.isneginf(model.log_scales[:, 2]).all()
    assert torch.isfinite(model.log_scales[:, :2]).all() and torch.isfinite(model.means).all()
    terms = history["loss_terms"]
    assert terms[-1]["depth_distortion_lambda"] == 1.0 and terms[-1]["normal_consistency"] > 0
    assert terms[19]["jet_regularization"] == 0.0 and terms[25]["jet_regularization"] > 0
    assert prior.h0.shape[0] == model.n
