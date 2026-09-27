"""CPU tests for active silhouette-hull floater relocation (RTGS-026)."""

from __future__ import annotations

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.silhouette_relocation import (
    SilhouetteHull,
    SilhouetteRelocationConfig,
    SilhouetteRelocator,
)
from rtgs.optim.trainer import TrainConfig, Trainer


def _disc_views(radius_px: float = 6.0, size: int = 32, count: int = 6):
    """Cameras on a ring around the origin, each seeing a centred disc silhouette."""
    cameras, masks = [], []
    yy, xx = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    disc = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= radius_px**2).float()
    for angle in torch.linspace(0, 2 * torch.pi, count + 1)[:-1]:
        eye = torch.tensor([3 * torch.cos(angle), 0.4, 3 * torch.sin(angle)])
        cameras.append(Camera.look_at(eye=eye, target=torch.zeros(3), width=size, height=size))
        masks.append(disc.clone())
    return cameras, masks


def test_single_rejecting_view_marks_a_floater() -> None:
    cameras, masks = _disc_views()
    hull = SilhouetteHull(cameras, masks, dilation_px=0)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.5, 0.0]])
    assert hull.rejected(points).tolist() == [False, True, True]
    assert hull.supported(points).tolist() == [True, False, False]
    # One rejecting view suffices: only the first camera keeps its disc, all others accept all.
    single = SilhouetteHull(cameras, [masks[0]] + [torch.ones_like(m) for m in masks[1:]])
    sideways = torch.tensor([[0.0, 0.0, 1.4]])  # off-centre for camera 0, which looks along -x
    assert single.rejected(sideways).item() is True
    accepting = SilhouetteHull(cameras, [torch.ones_like(m) for m in masks])
    assert accepting.rejected(sideways).item() is False


def test_points_behind_or_outside_a_camera_are_not_rejected_by_it() -> None:
    camera = Camera.look_at(
        eye=torch.tensor([0.0, 0.0, 3.0]), target=torch.zeros(3), width=16, height=16
    )
    hull = SilhouetteHull([camera], [torch.zeros(16, 16)], dilation_px=0)
    behind = torch.tensor([[0.0, 0.0, 5.0]])
    off_frame = torch.tensor([[50.0, 0.0, 0.0]])
    in_front = torch.tensor([[0.0, 0.0, 0.0]])
    assert hull.rejected(torch.cat([behind, off_frame, in_front])).tolist() == [False, False, True]


def test_dilation_accepts_points_just_outside_the_silhouette() -> None:
    cameras, masks = _disc_views(radius_px=6.0)
    strict = SilhouetteHull(cameras, masks, dilation_px=0)
    loose = SilhouetteHull(cameras, masks, dilation_px=3)
    edge = torch.tensor([[0.0, 0.0, 0.0]])
    for radius in torch.linspace(0.3, 1.2, 40):
        edge = torch.tensor([[float(radius), 0.0, 0.0]])
        if strict.rejected(edge).item():
            break
    assert strict.rejected(edge).item() and not loose.rejected(edge).item()


def test_occupied_voxels_lie_inside_the_hull() -> None:
    cameras, masks = _disc_views()
    hull = SilhouetteHull(cameras, masks, dilation_px=0)
    voxels, step = hull.occupied_voxels(torch.zeros(3), 2.4, 24)
    assert voxels.shape[0] > 0 and step == pytest.approx(0.1)
    assert bool(hull.supported(voxels).all())
    assert float(voxels.norm(dim=1).max()) < 1.2


def _params_with_state(means: torch.Tensor):
    params = {
        "means": torch.nn.Parameter(means.clone()),
        "scales": torch.nn.Parameter(torch.zeros(len(means), 3)),
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [p], "lr": 0.01, "name": name}])
        for name, p in params.items()
    }
    for parameter in params.values():
        parameter.grad = torch.ones_like(parameter)
    for optimizer in optimizers.values():
        optimizer.step()
    return params, optimizers


def test_relocator_moves_only_floaters_and_resets_their_moments() -> None:
    cameras, masks = _disc_views()
    hull = SilhouetteHull(cameras, masks, dilation_px=0)
    voxels, step = hull.occupied_voxels(torch.zeros(3), 2.4, 24)
    config = SilhouetteRelocationConfig(every=10, start=10, stop=30, dilation_px=0, seed=3)
    relocator = SilhouetteRelocator(hull, voxels, step, config)
    means = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.05, 0.02, 0.0], [0.0, 1.5, 0.0]])
    params, optimizers = _params_with_state(means)
    before = params["means"].detach().clone()
    relocator(params, optimizers, 5)  # inactive step: nothing happens
    assert torch.equal(params["means"].detach(), before) and relocator.events == []
    relocator(params, optimizers, 10)
    after = params["means"].detach()
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    assert bool(hull.supported(after).all())
    assert relocator.events == [{"step": 10, "n_gaussians": 4, "relocated": 2}]
    for name, optimizer in optimizers.items():
        state = optimizer.state[params[name]]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.count_nonzero(state[key][[1, 3]]) == 0
            assert torch.count_nonzero(state[key][[0, 2]]) > 0
    relocator(params, optimizers, 20)
    assert relocator.events[-1]["relocated"] == 0


def test_relocation_is_seeded() -> None:
    cameras, masks = _disc_views()
    hull = SilhouetteHull(cameras, masks)
    voxels, step = hull.occupied_voxels(torch.zeros(3), 2.4, 24)
    far = torch.tensor([[1.4, 0.0, 0.0], [0.0, 1.5, 0.3]])
    config = SilhouetteRelocationConfig(seed=11)
    first = SilhouetteRelocator(hull, voxels, step, config).relocate(far)[1]
    second = SilhouetteRelocator(hull, voxels, step, config).relocate(far)[1]
    assert torch.equal(first, second)


def test_schedule_validation() -> None:
    with pytest.raises(ValueError):
        SilhouetteRelocationConfig(every=0)
    with pytest.raises(ValueError):
        SilhouetteRelocationConfig(start=10, stop=5)
    config = SilhouetteRelocationConfig(every=100, start=500, stop=700)
    assert [s for s in range(0, 900) if config.active(s)] == [500, 600, 700]


def _tiny_run(callback):
    scene = make_synthetic_scene(n_gaussians=8, n_cameras=3, image_size=16, seed=4)
    init = scene.gt_gaussians.detach()
    config = TrainConfig(
        iterations=5, rasterizer="torch", densify=False, eval_every=5, ssim_lambda=0.0
    )
    return Trainer(config).train(scene, init, parameter_step_callback=callback)


def test_trainer_calls_parameter_step_callback_every_iteration() -> None:
    steps = []
    _tiny_run(lambda params, optimizers, step: steps.append((step, sorted(params))))
    assert [s for s, _ in steps] == [1, 2, 3, 4, 5]
    assert all("means" in names for _, names in steps)
    baseline, _ = _tiny_run(None)
    noop, _ = _tiny_run(lambda *_: None)
    assert torch.equal(baseline.means, noop.means)


def test_trainer_rejects_count_changing_callback() -> None:
    def shrink(params, optimizers, step):
        params["means"] = params["means"][:-1]

    with pytest.raises(RuntimeError, match="must not change the Gaussian count"):
        _tiny_run(shrink)
