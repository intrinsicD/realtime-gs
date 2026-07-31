"""CPU tests for classic clone/split/prune control under compact supervision."""

from __future__ import annotations

import math

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import rgb_to_sh
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim.compact_density import ClassicCompactDensityController
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.point_base import PointRenderOutput


def _inputs() -> ReconstructionInputs:
    cameras = [
        Camera.look_at(
            eye=torch.tensor([x, 0.0, -2.4]),
            target=torch.zeros(3),
            width=9,
            height=7,
            fov_x_deg=52.0,
        )
        for x in (-0.25, 0.30)
    ]
    world = torch.tensor([[-0.13, -0.08, 0.0], [0.16, 0.10, 0.08]])
    colors = torch.tensor([[0.82, 0.18, 0.22], [0.12, 0.72, 0.88]])
    observations = []
    for index, camera in enumerate(cameras):
        means, depth = camera.project(world)
        assert bool((depth > 0).all())
        observations.append(
            GaussianObservationField(
                width=camera.width,
                height=camera.height,
                means=means,
                log_scales=torch.log(torch.tensor([[1.1, 0.8], [0.9, 1.2]])),
                rotations=torch.tensor([0.25, -0.35]),
                colors=colors,
                amplitudes=torch.tensor([0.8, 0.65]),
                blend_mode="additive",
                epsilon=1e-8,
                sigma_cutoff=math.sqrt(10.0),
                fit_window=(1, 1, 7, 5),
                view_id=f"density-{index}",
                n_init=2,
            )
        )
    return ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=["density-0", "density-1"],
        bounds_hint=(torch.zeros(3), 1.0),
        name="compact-density-test",
    )


def _init(*, low_opacity: bool = False) -> Gaussians3D:
    means = torch.tensor([[-0.09, -0.05, -0.03], [0.12, 0.07, 0.12]])
    quats = torch.tensor([[1.0, 0.08, -0.03, 0.02], [0.96, -0.05, 0.15, 0.08]])
    log_scales = torch.stack(
        [
            torch.full((3,), math.log(0.005)),
            torch.full((3,), math.log(0.08)),
        ]
    )
    opacity = torch.tensor([0.0001 if low_opacity else 0.55, 0.48])
    sh = rgb_to_sh(torch.tensor([[0.70, 0.25, 0.20], [0.16, 0.63, 0.76]]))[:, None]
    return Gaussians3D(means, quats, log_scales, opacity, sh)


def _train_config(*, opacity_lr: float = 5e-2) -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=2,
        attempts_per_step=12,
        proposal_mode="pixel_gaussian",
        seed=63821,
        extent=1.0,
        device="cpu",
        lr_opacity=opacity_lr,
        point_chunk=3,
        gaussian_chunk=1,
        outer_microbatch=12,
        query_component_chunk=1,
        teacher_tile_size=4,
        evaluation_chunk=6,
        checkpoints=(0, 1, 2),
        evaluate_checkpoint_risks=False,
    )


def test_classic_compact_density_runs_clone_and_split_with_complete_lineage() -> None:
    controller = ClassicCompactDensityController(
        DensityConfig(
            start_iter=1,
            stop_iter=1,
            every=1,
            grad_threshold=0.0,
            split_scale_frac=0.01,
            prune_opacity=0.0,
            prune_scale_frac=10.0,
            max_gaussians=4,
            opacity_reset_every=0,
        ),
        seed=917,
    )
    final, history = CompactTrainer(_train_config()).train(
        _inputs(),
        _init(),
        topology_controller=controller,
    )

    assert final.n == history["n_opt_3d"] == 4
    topology = history["topology_control"]
    assert topology["schema"] == "rtgs.classic_compact_density.v1"
    assert topology["events"] == [
        {
            "step": 1,
            "rows_before": 2,
            "rows_after": 4,
            "cloned": 1,
            "split": 1,
            "pruned": 0,
            "opacity_reset": False,
        }
    ]
    assert history["persistent_ids"] == [0, 2, 3, 4]
    assert history["surviving_original_ids"] == [0]
    assert history["removed_original_ids"] == [1]
    assert [
        (item["birth_id"], item["parent_id"], item["operator"], item["child_ordinal"])
        for item in topology["lineage"]
    ] == [
        (2, 0, "clone", 0),
        (3, 1, "split", 0),
        (4, 1, "split", 1),
    ]
    boundary = history["steps"][0]["topology_optimizer_boundary"]
    assert boundary["survivor_moments_bit_preserved"] is True
    assert boundary["newborn_moments_exact_zero"] is True
    assert boundary["scalar_clocks_unchanged"] is True


def test_classic_compact_density_prunes_low_opacity_without_inventing_lineage() -> None:
    controller = ClassicCompactDensityController(
        DensityConfig(
            start_iter=1,
            stop_iter=1,
            every=1,
            grad_threshold=1e9,
            prune_opacity=0.005,
            prune_scale_frac=10.0,
            max_gaussians=4,
            opacity_reset_every=0,
        ),
        seed=918,
    )
    final, history = CompactTrainer(_train_config(opacity_lr=0.0)).train(
        _inputs(),
        _init(low_opacity=True),
        topology_controller=controller,
    )

    assert final.n == 1
    assert history["persistent_ids"] == [1]
    assert history["topology_control"]["lineage"] == []
    assert history["topology_control"]["events"][0]["pruned"] == 1
    boundary = history["steps"][0]["topology_optimizer_boundary"]
    assert boundary["rows_before"] == 2
    assert boundary["rows_after"] == 1
    assert boundary["survivor_moments_bit_preserved"] is True


def test_classic_compact_density_persists_multigeneration_ancestry() -> None:
    controller = ClassicCompactDensityController(
        DensityConfig(
            start_iter=1,
            stop_iter=2,
            every=1,
            grad_threshold=0.0,
            split_scale_frac=0.01,
            prune_opacity=0.0,
            prune_scale_frac=10.0,
            max_gaussians=8,
            opacity_reset_every=0,
        ),
        seed=920,
    )
    initial = _init()
    params = {
        "means": initial.means.detach().clone().requires_grad_(True),
        "quats": initial.quats.detach().clone().requires_grad_(True),
        "scales": initial.log_scales.detach().clone().requires_grad_(True),
        "opacities": torch.logit(initial.opacity).detach().clone().requires_grad_(True),
        "sh0": initial.sh[:, :1].detach().clone().requires_grad_(True),
        "shN": initial.sh[:, 1:].detach().clone().requires_grad_(True),
    }
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-3) for name, parameter in params.items()
    }
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=1,
        attempts_per_step=1,
    )

    for step in (1, 2):
        count = params["means"].shape[0]
        means2d = torch.zeros(count, 2, requires_grad=True)
        means2d.sum().backward()
        controller.observe_post_backward(
            step=step,
            view_index=0,
            output=PointRenderOutput(
                color=torch.zeros(1, 3),
                alpha=torch.zeros(1),
                depth=torch.zeros(1),
                visible=torch.arange(count),
                means2d=means2d,
            ),
            width=10,
            height=8,
        )
        params = controller.after_step(
            step=step,
            params=params,
            optimizers=optimizers,
            snapshot=initial,
        )

    assert params["means"].shape[0] == 8
    lineage = controller.history_record()["lineage"]
    births = {item["birth_id"]: item for item in lineage}
    assert len(births) == 9
    assert {item["parent_id"] for item in lineage if item["parent_id"] >= 2} <= births.keys()
    assert {3, 4} <= births.keys()
    assert any(item["parent_id"] in {3, 4} for item in lineage)
    assert births[3]["survives_final"] is False
    assert births[4]["survives_final"] is False


def test_unscheduled_compact_density_controller_is_numerically_noop() -> None:
    config = _train_config()
    baseline, baseline_history = CompactTrainer(config).train(_inputs(), _init())
    controller = ClassicCompactDensityController(
        DensityConfig(start_iter=10, stop_iter=20, every=5),
        seed=919,
    )
    controlled, controlled_history = CompactTrainer(config).train(
        _inputs(),
        _init(),
        topology_controller=controller,
    )

    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(controlled, name), getattr(baseline, name))
    assert controlled_history["topology_control"]["events"] == []
    assert controlled_history["topology_control"]["lineage"] == []


def test_density_accumulation_reduces_packed_query_rows_to_physical_gaussians() -> None:
    controller = DensityController(
        DensityConfig(),
        n_gaussians=3,
        scene_extent=1.0,
    )
    means2d = torch.zeros(3, 2, requires_grad=True)
    coefficients = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]])
    (means2d * coefficients).sum().backward()
    output = PointRenderOutput(
        color=torch.zeros(1, 3),
        alpha=torch.zeros(1),
        depth=torch.zeros(1),
        visible=torch.tensor([0, 1, 2]),
        means2d=means2d,
        density_gaussian_ids=torch.tensor([0, 0, 2]),
    )

    controller.accumulate(output, width=10, height=8)

    expected = torch.tensor(
        [
            (torch.tensor([4.0, 1.0]).norm() * 5.0).item(),
            0.0,
            (torch.tensor([0.0, 4.0]).norm() * 5.0).item(),
        ]
    )
    assert torch.allclose(controller.grad_accum, expected)
    assert torch.equal(controller.count, torch.ones(3))
