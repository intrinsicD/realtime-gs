"""CPU mechanism tests for RGB-free fixed-attempt compact training."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from dataclasses import asdict, replace

import pytest
import torch

import rtgs.optim.compact_trainer as compact_trainer_module
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    GaussianPixelProposal,
    GaussianPointProposal,
    ObservationSamples,
    fixed_attempt_mean,
)
from rtgs.core.sh import rgb_to_sh
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim.compact_trainer import (
    PROPOSAL_MODES,
    CompactTrainConfig,
    CompactTrainer,
    build_view_schedule,
    observation_digest,
    preflight_observations,
    step_sample_seed,
)
from rtgs.optim.density import apply_selected_birth_surgery
from rtgs.render.torch_points import TorchPointRasterizer

_DEV_SEED = 63821


def _camera(x: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, 0.0, -2.4]),
        target=torch.zeros(3),
        width=9,
        height=7,
        fov_x_deg=52.0,
    )


def _inputs() -> ReconstructionInputs:
    cameras = [_camera(-0.25), _camera(0.30)]
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
                epsilon=1e-8,
                sigma_cutoff=math.sqrt(10.0),
                fit_window=(1, 1, 7, 5),
                view_id=f"dev-{index}",
                n_init=3 + index,
            )
        )
    return ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=["dev-0", "dev-1"],
        bounds_hint=None,
        name="compact-trainer-dev",
    )


def _init() -> Gaussians3D:
    means = torch.tensor([[-0.09, -0.05, -0.03], [0.12, 0.07, 0.12]])
    quats = torch.tensor([[1.0, 0.08, -0.03, 0.02], [0.96, -0.05, 0.15, 0.08]])
    log_scales = torch.log(torch.tensor([[0.10, 0.07, 0.08], [0.08, 0.11, 0.07]]))
    opacity = torch.tensor([0.55, 0.48])
    sh = rgb_to_sh(torch.tensor([[0.70, 0.25, 0.20], [0.16, 0.63, 0.76]]))[:, None]
    return Gaussians3D(means, quats, log_scales, opacity, sh)


def _proposal_field(
    teacher: GaussianObservationField,
    occupancy: torch.Tensor,
    *,
    colors: torch.Tensor | None = None,
) -> GaussianObservationField:
    occupancy = torch.as_tensor(occupancy, dtype=teacher.dtype, device=teacher.device)
    if occupancy.shape != (teacher.n,):
        raise ValueError("occupancy must contain one scalar per teacher component")
    return GaussianObservationField(
        width=teacher.width,
        height=teacher.height,
        means=teacher.means,
        log_scales=teacher.log_scales,
        rotations=teacher.rotations,
        colors=(torch.ones_like(teacher.colors) if colors is None else colors.to(teacher.colors)),
        amplitudes=teacher.amplitudes * occupancy,
        color_grads=None,
        filter_variance=teacher.filter_variance,
        blend_mode="normalized",
        epsilon=teacher.epsilon,
        sigma_cutoff=teacher.sigma_cutoff,
        support_fade_alpha=teacher.support_fade_alpha,
        aa_dilation=teacher.aa_dilation,
        view_id=teacher.view_id,
        fit_window=teacher.fit_window,
        n_init=teacher.n_init,
        provider="synthetic_fixture",
        producer_version="center-gated-proposal-test-v1",
    )


def _proposal_fields(
    inputs: ReconstructionInputs,
    *,
    poison_colors: bool = False,
) -> list[GaussianObservationField]:
    result = []
    for view_index, teacher in enumerate(inputs.observations):
        occupancy = torch.linspace(0.35, 0.85, teacher.n, dtype=teacher.dtype)
        colors = (
            torch.full_like(teacher.colors, 10_000.0 + view_index)
            if poison_colors
            else torch.ones_like(teacher.colors)
        )
        result.append(_proposal_field(teacher, occupancy, colors=colors))
    return result


def _unequal_component_inputs() -> ReconstructionInputs:
    source = _inputs()
    field = source.observations[1]
    ids = torch.tensor([0, 0, 1])
    split = GaussianObservationField(
        width=field.width,
        height=field.height,
        means=field.means[ids],
        log_scales=field.log_scales[ids],
        rotations=field.rotations[ids],
        colors=field.colors[ids],
        amplitudes=torch.stack(
            [0.3 * field.amplitudes[0], 0.7 * field.amplitudes[0], field.amplitudes[1]]
        ),
        color_grads=None,
        filter_variance=None,
        blend_mode=field.blend_mode,
        epsilon=field.epsilon,
        sigma_cutoff=field.sigma_cutoff,
        support_fade_alpha=field.support_fade_alpha,
        aa_dilation=field.aa_dilation,
        view_id=field.view_id,
        fit_window=field.fit_window,
        n_init=field.n_init,
        provider=field.provider,
    )
    return ReconstructionInputs(
        observations=[source.observations[0], split],
        cameras=source.cameras,
        view_names=source.view_names,
        name=f"{source.name}-unequal",
    )


def _inputs_with_optional_geometry() -> ReconstructionInputs:
    source = _inputs()
    points = torch.tensor(
        [[-0.3, 0.0, 0.1], [0.2, -0.1, 0.0], [0.1, 0.3, 0.2]],
        dtype=torch.float32,
    )
    return ReconstructionInputs(
        observations=source.observations,
        cameras=source.cameras,
        view_names=source.view_names,
        points=points,
        point_visibility=[torch.tensor([0, 2]), torch.tensor([1, 2])],
        bounds_hint=(torch.tensor([0.1, -0.2, 0.3]), 3.25),
        name=source.name,
        archive_stats=source.archive_stats,
    )


def _config(mode: str, *, outer: int = 4) -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=2,
        attempts_per_step=12,
        proposal_mode=mode,
        seed=_DEV_SEED,
        extent=1.0,
        point_chunk=3,
        gaussian_chunk=1,
        outer_microbatch=outer,
        query_component_chunk=1,
        teacher_tile_size=4,
        evaluation_chunk=6,
        checkpoints=(0, 2),
    )


class _NoOpTopologyController:
    def __init__(self) -> None:
        self._persistent_ids = torch.empty(0, dtype=torch.long)

    @property
    def persistent_ids(self) -> torch.Tensor:
        return self._persistent_ids

    def bind(self, params, optimizers, *, extent, n_views, attempts_per_step):
        del optimizers, extent, n_views, attempts_per_step
        self._persistent_ids = torch.arange(
            params["means"].shape[0], device=params["means"].device, dtype=torch.long
        )

    def needs_compositing_color_basis(self, step):
        del step
        return False

    def observe_pre_backward(self, **kwargs):
        del kwargs

    def observe_post_backward(self, **kwargs):
        del kwargs

    def after_step(self, *, step, params, optimizers, snapshot):
        del step, optimizers, snapshot
        return params

    def history_record(self):
        return {"kind": "noop"}


class _OneWaveTopologyController(_NoOpTopologyController):
    def __init__(self) -> None:
        super().__init__()
        self.receipt = None
        self._next_id = 0
        self._lineage = []

    def bind(self, params, optimizers, *, extent, n_views, attempts_per_step):
        super().bind(
            params,
            optimizers,
            extent=extent,
            n_views=n_views,
            attempts_per_step=attempts_per_step,
        )
        self._next_id = int(self._persistent_ids.numel())

    def after_step(self, *, step, params, optimizers, snapshot):
        del snapshot
        if step != 1:
            return params
        new_params, receipt = apply_selected_birth_surgery(
            params,
            optimizers,
            (0, 1),
            scene_extent=1.0,
            generator=torch.Generator(device=params["means"].device).manual_seed(63891),
            max_gaussians=4,
        )
        survivor_rows = torch.tensor(
            receipt.survivor_old_rows,
            device=self._persistent_ids.device,
            dtype=torch.long,
        )
        survivor_ids = self._persistent_ids[survivor_rows]
        newborn_ids = torch.arange(
            self._next_id,
            self._next_id + len(receipt.newborns),
            device=self._persistent_ids.device,
            dtype=torch.long,
        )
        self._next_id += len(receipt.newborns)
        self._persistent_ids = torch.cat([survivor_ids, newborn_ids])
        self._lineage = [
            {
                "birth_id": int(birth_id),
                "parent_id": int(receipt.new_row_to_old_row[item.new_row]),
                "operator": item.operator,
                "child_ordinal": item.child_ordinal,
            }
            for birth_id, item in zip(
                newborn_ids.detach().cpu().tolist(),
                receipt.newborns,
                strict=True,
            )
        ]
        self.receipt = receipt
        return new_params

    def history_record(self):
        return {
            "kind": "one_wave",
            "receipt": None if self.receipt is None else asdict(self.receipt),
            "lineage": self._lineage,
        }


class _BasisParityController(_NoOpTopologyController):
    """Capture one real compositor VJP and all quantities it must not perturb."""

    def __init__(self, *, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.params = {}
        self.forward = []
        self.losses = []
        self.gradients = []
        self.means2d_gradients = []
        self.updated = []
        self.contractions = []

    def bind(self, params, optimizers, *, extent, n_views, attempts_per_step):
        super().bind(
            params,
            optimizers,
            extent=extent,
            n_views=n_views,
            attempts_per_step=attempts_per_step,
        )
        self.params = params

    def needs_compositing_color_basis(self, step):
        del step
        return self.enabled

    def observe_pre_backward(
        self,
        *,
        step,
        view_index,
        output,
        point_loss,
        active,
        attempts,
    ):
        del step, view_index, attempts
        self.forward.append(
            {
                "color": output.color.detach().clone(),
                "alpha": output.alpha.detach().clone(),
                "depth": output.depth.detach().clone(),
                "visible": (None if output.visible is None else output.visible.detach().clone()),
            }
        )
        self.losses.append(point_loss.detach().clone())
        basis = output.compositing_color_basis
        if self.enabled:
            assert basis is not None
            native_active = active.to(device=output.color.device, dtype=output.color.dtype)
            native_error = point_loss.to(device=output.color.device, dtype=output.color.dtype)
            grad_outputs = torch.zeros_like(output.color)
            grad_outputs[:, 0] = native_active * native_error
            grad_outputs[:, 1] = native_active
            contracted = torch.autograd.grad(
                output.color,
                basis,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach()
            assert bool(torch.isfinite(contracted).all())
            self.contractions.append(contracted.clone())
            output.compositing_color_basis = None
        else:
            assert basis is None

    def observe_post_backward(
        self,
        *,
        step,
        view_index,
        output,
        width,
        height,
    ):
        del step, view_index, width, height
        assert output.means2d is not None
        assert output.means2d.grad is not None
        self.means2d_gradients.append(output.means2d.grad.detach().clone())
        self.gradients.append(
            {name: parameter.grad.detach().clone() for name, parameter in self.params.items()}
        )

    def after_step(self, *, step, params, optimizers, snapshot):
        del step, optimizers, snapshot
        self.updated.append(
            {name: parameter.detach().clone() for name, parameter in params.items()}
        )
        return params

    def history_record(self):
        return {
            "kind": "basis_parity",
            "diagnostic_enabled": self.enabled,
            "contraction_count": len(self.contractions),
            "lineage": [],
        }


class _CorruptingTopologyController(_OneWaveTopologyController):
    def __init__(self, corruption: str) -> None:
        super().__init__()
        self.corruption = corruption

    def after_step(self, *, step, params, optimizers, snapshot):
        new_params = super().after_step(
            step=step,
            params=params,
            optimizers=optimizers,
            snapshot=snapshot,
        )
        if step != 1:
            return new_params
        optimizer = optimizers["means"]
        state = optimizer.state[new_params["means"]]
        if self.corruption == "metadata":
            optimizer.param_groups[0]["lr"] *= 2.0
        elif self.corruption == "moment_shape":
            state["exp_avg"] = state["exp_avg"][:-1]
        elif self.corruption == "moment_dtype":
            state["exp_avg"] = state["exp_avg"].double()
        elif self.corruption == "moment_nonfinite":
            state["exp_avg"][0, 0] = torch.nan
        elif self.corruption == "survivor_moment":
            state["exp_avg"][0, 0] += 1.0
        elif self.corruption == "newborn_moment":
            state["exp_avg"][-1, 0] = 1.0
        elif self.corruption == "newborn_negative_zero":
            state["exp_avg"][-1, 0] = -0.0
        elif self.corruption == "clock":
            state["step"].add_(1)
        else:
            raise AssertionError(f"unknown corruption {self.corruption}")
        return new_params


def test_schedule_and_per_step_seeds_are_mode_independent_and_stable():
    expected = build_view_schedule(2, 12, _DEV_SEED)
    assert expected == (1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0)
    assert expected == build_view_schedule(2, 12, _DEV_SEED, mode="iid")
    assert expected == build_view_schedule(2, 12, _DEV_SEED)
    assert len(expected) == 12 and set(expected) <= {0, 1}
    seeds = [step_sample_seed(_DEV_SEED, step) for step in range(12)]
    assert seeds == [step_sample_seed(_DEV_SEED, step) for step in range(12)]
    assert len(set(seeds)) == len(seeds)
    for mode in PROPOSAL_MODES:
        config = CompactTrainConfig(
            iterations=12,
            proposal_mode=mode,
            checkpoints=(0, 12),
            seed=_DEV_SEED,
        )
        assert config.schedule_mode == "iid"
        assert config.target_mode == "uniform"
        assert (
            build_view_schedule(
                2,
                config.iterations,
                config.seed,
                mode=config.schedule_mode,
            )
            == expected
        )


def test_balanced_cycle_schedule_is_deterministic_prefix_stable_and_count_balanced():
    schedule = build_view_schedule(5, 13, _DEV_SEED, mode="balanced_cycle")
    assert schedule == build_view_schedule(5, 13, _DEV_SEED, mode="balanced_cycle")
    assert schedule == build_view_schedule(5, 22, _DEV_SEED, mode="balanced_cycle")[:13]
    assert set(schedule[:5]) == set(range(5))
    assert set(schedule[5:10]) == set(range(5))
    counts = [schedule.count(view) for view in range(5)]
    assert max(counts) - min(counts) <= 1
    assert schedule != build_view_schedule(5, 13, _DEV_SEED + 1, mode="balanced_cycle")


def test_proposal_attempt_target_rejects_noncontinuous_gaussian_modes():
    with pytest.raises(ValueError, match="requires proposal_mode='area_gaussian'"):
        CompactTrainConfig(
            proposal_mode="pixel_gaussian",
            target_mode="proposal_attempt",
        )


def test_later_step_stream_is_independent_of_earlier_rng_consumption(monkeypatch):
    config = CompactTrainConfig(
        **{
            **_config("pixel_gaussian").__dict__,
            "evaluate_checkpoint_risks": False,
        }
    )
    _, baseline = CompactTrainer(config).train(_inputs(), _init())
    original = GaussianPixelProposal.sample
    invocations = 0

    def consume_only_on_first_call(self, count, *, uniform_fraction, generator):
        nonlocal invocations
        if invocations == 0:
            torch.rand(97, generator=generator, device=self.field.device)
        invocations += 1
        return original(
            self,
            count,
            uniform_fraction=uniform_fraction,
            generator=generator,
        )

    monkeypatch.setattr(GaussianPixelProposal, "sample", consume_only_on_first_call)
    _, intervened = CompactTrainer(config).train(_inputs(), _init())

    assert baseline["view_schedule"] == intervened["view_schedule"]
    assert baseline["steps"][0]["xy_sha256"] != intervened["steps"][0]["xy_sha256"]
    for key in (
        "sample_seed",
        "xy_sha256",
        "active_sha256",
        "importance_sha256",
        "proposal_component_ids_sha256",
    ):
        assert baseline["steps"][1][key] == intervened["steps"][1][key]


def test_all_null_fixed_attempt_loss_is_differentiable_zero():
    count = 5
    values = torch.linspace(0.2, 0.8, count, requires_grad=True)
    samples = ObservationSamples(
        xy=torch.zeros(count, 2),
        proposal_component_ids=torch.arange(count),
        proposal_density=torch.zeros(count),
        joint_density=torch.zeros(count),
        target_density=torch.zeros(count),
        importance=torch.zeros(count),
        inside_fit_window=torch.zeros(count, dtype=torch.bool),
        active=torch.zeros(count, dtype=torch.bool),
        risk_measure="continuous_area",
    )
    loss = fixed_attempt_mean(values, samples)
    assert torch.equal(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_proposal_attempt_retarget_has_unit_active_weight_and_keeps_null_denominator():
    field = _inputs().observations[0]
    proposal = GaussianPointProposal(field)
    samples = proposal.sample(
        256,
        uniform_fraction=0.25,
        generator=torch.Generator().manual_seed(_DEV_SEED + 17),
    )
    retargeted = compact_trainer_module._retarget_samples(samples, "proposal_attempt")

    assert torch.equal(retargeted.xy, samples.xy)
    assert torch.equal(retargeted.active, samples.active)
    assert torch.equal(
        retargeted.importance,
        retargeted.active.to(retargeted.importance.dtype),
    )
    assert torch.equal(
        retargeted.target_density[retargeted.active],
        retargeted.proposal_density[retargeted.active],
    )
    assert torch.equal(
        retargeted.target_density[~retargeted.active],
        torch.zeros_like(retargeted.target_density[~retargeted.active]),
    )
    losses = torch.linspace(0.1, 1.1, samples.xy.shape[0])
    expected = losses[retargeted.active].sum() / samples.xy.shape[0]
    torch.testing.assert_close(fixed_attempt_mean(losses, retargeted), expected)


@pytest.mark.parametrize("mode", PROPOSAL_MODES)
def test_all_modes_keep_fixed_attempts_topology_teachers_and_adam_clocks(mode):
    inputs = _inputs()
    digest = observation_digest(inputs)
    final, history = CompactTrainer(_config(mode)).train(inputs, _init())

    assert final.n == 2
    assert history["n_init_3d"] == history["n_opt_3d"] == 2
    assert history["teacher_digest_before"] == history["teacher_digest_after"] == digest
    assert history["optimizer_steps"] == {
        "means": 2,
        "quats": 2,
        "scales": 2,
        "opacities": 2,
        "sh0": 2,
        "shN": 2,
    }
    assert set(history["parameter_motion"]) == {
        "means",
        "quaternions",
        "log_scales",
        "opacity_logits",
        "sh0",
    }
    assert set(history["optimizer_group_motion"]) == {
        "means",
        "quats",
        "scales",
        "opacities",
        "sh0",
        "shN",
    }
    for step in history["steps"]:
        assert step["attempts"] == 12
        assert step["active_count"] + step["null_count"] == 12
        assert step["uniform_attempt_count"] + step["gaussian_attempt_count"] == 12
        assert (
            step["gaussian_accepted_count"] + step["gaussian_rejected_count"]
            == step["gaussian_attempt_count"]
        )
        assert step["teacher_query_attempts"] == step["student_query_attempts"] == 12
        assert step["teacher_query_calls"] == step["student_query_calls"] == 3
        assert step["cardinality"] == 2
        assert step["importance_max"] <= 4.00001
        assert len(step["xy_sha256"]) == len(step["active_sha256"]) == 64
        for value in step["gradient_max"].values():
            assert math.isfinite(value)
    assert len(history["checkpoints"]) == 2
    assert [item["step"] for item in history["checkpoints"]] == [0, 2]


def test_observation_digest_covers_mean_residual_values_and_absence():
    inputs = _inputs()
    base_digest = observation_digest(inputs)
    zeros = torch.zeros(
        inputs.observations[0].n,
        2,
        dtype=torch.float32,
    )
    zero_residual_inputs = replace(
        inputs,
        observations=[
            replace(inputs.observations[0], mean_residuals=zeros),
            inputs.observations[1],
        ],
    )
    changed_residual_inputs = replace(
        zero_residual_inputs,
        observations=[
            replace(
                zero_residual_inputs.observations[0],
                mean_residuals=zeros + torch.tensor([0.0, 2.0**-16]),
            ),
            zero_residual_inputs.observations[1],
        ],
    )

    zero_digest = observation_digest(zero_residual_inputs)
    changed_digest = observation_digest(changed_residual_inputs)
    assert len({base_digest, zero_digest, changed_digest}) == 3


def test_noop_topology_controller_preserves_default_numerics_exactly():
    config = CompactTrainConfig(
        **{
            **_config("pixel_gaussian", outer=12).__dict__,
            "iterations": 1,
            "checkpoints": (0, 1),
            "evaluate_checkpoint_risks": False,
        }
    )
    baseline, baseline_history = CompactTrainer(config).train(_inputs(), _init())
    controlled, controlled_history = CompactTrainer(config).train(
        _inputs(),
        _init(),
        topology_controller=_NoOpTopologyController(),
    )

    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        torch.testing.assert_close(
            getattr(controlled, name),
            getattr(baseline, name),
            atol=0.0,
            rtol=0.0,
        )
    ignored = {
        "cardinality_after_topology",
        "topology_changed",
        "topology_optimizer_boundary",
        "elapsed_seconds",
    }
    assert {
        key: value for key, value in controlled_history["steps"][0].items() if key not in ignored
    } == {key: value for key, value in baseline_history["steps"][0].items() if key not in ignored}
    assert controlled_history["optimizer_steps"] == baseline_history["optimizer_steps"]
    assert controlled_history["checkpoints"] == baseline_history["checkpoints"]


def test_topology_controller_runs_one_matched_birth_wave_and_keeps_adam_aligned():
    source = _init()
    source = Gaussians3D(
        means=source.means,
        quats=source.quats,
        log_scales=torch.stack([torch.full((3,), math.log(0.005)), source.log_scales[1]], dim=0),
        opacity=source.opacity,
        sh=source.sh,
    )
    controller = _OneWaveTopologyController()
    config = CompactTrainConfig(
        iterations=2,
        attempts_per_step=12,
        proposal_mode="pixel_gaussian",
        seed=_DEV_SEED,
        extent=1.0,
        point_chunk=3,
        gaussian_chunk=1,
        outer_microbatch=12,
        query_component_chunk=1,
        teacher_tile_size=4,
        evaluation_chunk=6,
        checkpoints=(0, 1, 2),
        evaluate_checkpoint_risks=False,
    )
    final, history = CompactTrainer(config).train(
        _inputs(),
        source,
        topology_controller=controller,
    )

    assert controller.receipt is not None
    assert controller.receipt.clone_parent_rows == (0,)
    assert controller.receipt.split_parent_rows == (1,)
    assert final.n == history["n_opt_3d"] == 4
    assert history["steps"][0]["cardinality"] == 2
    assert history["steps"][0]["cardinality_after_topology"] == 4
    assert history["steps"][0]["topology_changed"] is True
    assert history["steps"][1]["cardinality"] == 4
    assert history["steps"][1]["cardinality_after_topology"] == 4
    assert history["optimizer_steps"] == {name: 2 for name in history["optimizer_steps"]}
    assert history["persistent_ids"] == [0, 2, 3, 4]
    assert history["surviving_original_ids"] == [0]
    assert history["removed_original_ids"] == [1]
    assert history["newborn_parameter_summary"]["means"]["rows"] == 3
    family_summary = history["newborn_parameter_summary_by_lineage"]
    assert family_summary["clone"]["means"]["rows"] == 1
    assert family_summary["split_child_0"]["means"]["rows"] == 1
    assert family_summary["split_child_1"]["means"]["rows"] == 1
    boundary = history["steps"][0]["topology_optimizer_boundary"]
    assert boundary["rows_before"] == 2
    assert boundary["rows_after"] == 4
    assert boundary["survivor_rows"] == 1
    assert boundary["removed_rows"] == 1
    assert boundary["newborn_rows"] == 3
    assert boundary["group_metadata_preserved"] is True
    assert boundary["survivor_moments_bit_preserved"] is True
    assert boundary["newborn_moments_exact_zero"] is True
    assert boundary["scalar_clocks_unchanged"] is True
    assert [item["step"] for item in history["checkpoints"]] == [0, 1, 2]
    assert [item["snapshot_sha256"] for item in history["checkpoints"]]
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        assert bool(torch.isfinite(getattr(final, name)).all())


def test_enabled_compositor_vjp_is_an_exact_noop_through_one_adam_update():
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=12,
        proposal_mode="pixel_gaussian",
        seed=_DEV_SEED + 31,
        extent=1.0,
        point_chunk=3,
        gaussian_chunk=1,
        outer_microbatch=12,
        query_component_chunk=1,
        teacher_tile_size=4,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )

    initial_rng = torch.random.get_rng_state().clone()
    disabled = _BasisParityController(enabled=False)
    disabled_final, disabled_history = CompactTrainer(config).train(
        _inputs(),
        _init(),
        topology_controller=disabled,
    )
    disabled_rng = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(initial_rng)
    enabled = _BasisParityController(enabled=True)
    enabled_final, enabled_history = CompactTrainer(config).train(
        _inputs(),
        _init(),
        topology_controller=enabled,
    )
    enabled_rng = torch.random.get_rng_state().clone()

    assert len(enabled.contractions) == 1
    assert enabled.contractions[0].numel() > 0
    assert float(enabled.contractions[0].abs().max()) > 0.0
    assert torch.equal(enabled_rng, disabled_rng)
    assert len(enabled.forward) == len(disabled.forward) == 1
    for key in ("color", "alpha", "depth", "visible"):
        assert torch.equal(enabled.forward[0][key], disabled.forward[0][key])
    assert torch.equal(enabled.losses[0], disabled.losses[0])
    assert torch.equal(enabled.means2d_gradients[0], disabled.means2d_gradients[0])
    assert (
        tuple(enabled.gradients[0])
        == tuple(disabled.gradients[0])
        == (
            "means",
            "quats",
            "scales",
            "opacities",
            "sh0",
            "shN",
        )
    )
    for name in enabled.gradients[0]:
        assert torch.equal(enabled.gradients[0][name], disabled.gradients[0][name])
        assert torch.equal(enabled.updated[0][name], disabled.updated[0][name])
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(getattr(enabled_final, name), getattr(disabled_final, name))

    def without_diagnostics_or_timing(value):
        if isinstance(value, dict):
            return {
                key: without_diagnostics_or_timing(item)
                for key, item in value.items()
                if key
                not in {
                    "contraction_count",
                    "diagnostic_enabled",
                    "elapsed_seconds",
                    "peak_rss_bytes",
                }
            }
        if isinstance(value, list):
            return [without_diagnostics_or_timing(item) for item in value]
        return value

    assert without_diagnostics_or_timing(enabled_history) == without_diagnostics_or_timing(
        disabled_history
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("metadata", "changed optimizer metadata"),
        ("moment_shape", "wrong shape"),
        ("moment_dtype", "changed dtype"),
        ("moment_nonfinite", "non-finite"),
        ("survivor_moment", "changed surviving exp_avg moments"),
        ("newborn_moment", "non-exact-zero newborn exp_avg moments"),
        ("newborn_negative_zero", "non-exact-zero newborn exp_avg moments"),
        ("clock", "Adam group clocks are not aligned"),
    ],
)
def test_topology_optimizer_boundary_fails_closed_on_state_corruption(
    corruption,
    message,
):
    source = _init()
    source = Gaussians3D(
        means=source.means,
        quats=source.quats,
        log_scales=torch.stack(
            [torch.full((3,), math.log(0.005)), source.log_scales[1]],
            dim=0,
        ),
        opacity=source.opacity,
        sh=source.sh,
    )
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=6,
        proposal_mode="pixel_uniform",
        seed=_DEV_SEED + 32,
        extent=1.0,
        outer_microbatch=6,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    with pytest.raises(RuntimeError, match=message):
        CompactTrainer(config).train(
            _inputs(),
            source,
            topology_controller=_CorruptingTopologyController(corruption),
        )


def test_explicit_proposal_target_changes_only_weights_and_never_queries_proxy_colors():
    inputs = _inputs()
    poison = _proposal_fields(inputs, poison_colors=True)
    benign = _proposal_fields(inputs, poison_colors=False)

    class WeightOnlyBackend:
        def __init__(self, field):
            self.field = field
            self.index = GaussianObservationIndex(field, tile_size=4)

        def query(self, *_args, **_kwargs):
            raise AssertionError("proposal colors must never be queried")

        def query_weight_sum(self, xy, component_chunk=4096):
            return self.index.query_weight_sum(xy, component_chunk=component_chunk)

    def run(target_mode: str, fields: list[GaussianObservationField]):
        config = CompactTrainConfig(
            **{
                **_config("area_gaussian").__dict__,
                "target_mode": target_mode,
                "evaluate_checkpoint_risks": False,
            }
        )
        return CompactTrainer(config).train(
            inputs,
            _init(),
            proposal_fields=fields,
            proposal_query_backends=[WeightOnlyBackend(field) for field in fields],
        )

    uniform_final, uniform_history = run("uniform", poison)
    proposal_final, proposal_history = run("proposal_attempt", poison)
    benign_final, benign_history = run("proposal_attempt", benign)

    assert uniform_history["proposal_field_source"] == "explicit"
    assert proposal_history["target_mode"] == "proposal_attempt"
    assert proposal_history["proposal_digest_before"] == proposal_history["proposal_digest_after"]
    for uniform_step, proposal_step in zip(
        uniform_history["steps"], proposal_history["steps"], strict=True
    ):
        for key in (
            "view_index",
            "sample_seed",
            "xy_sha256",
            "active_sha256",
            "inside_fit_window_sha256",
            "proposal_density_sha256",
            "joint_density_sha256",
            "proposal_component_ids_sha256",
        ):
            assert uniform_step[key] == proposal_step[key]
        assert proposal_step["importance_max"] <= 1.0
    assert any(
        uniform_step["importance_sha256"] != proposal_step["importance_sha256"]
        for uniform_step, proposal_step in zip(
            uniform_history["steps"], proposal_history["steps"], strict=True
        )
    )
    assert proposal_history["proposal_normalizers"] != []
    assert proposal_history["proposal_view_diagnostics"] != []
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        torch.testing.assert_close(
            getattr(proposal_final, name),
            getattr(benign_final, name),
            atol=0.0,
            rtol=0.0,
        )
    assert proposal_history["steps"] != uniform_history["steps"]
    assert benign_history["proposal_digest_before"] != proposal_history["proposal_digest_before"]
    assert uniform_final.n == proposal_final.n == benign_final.n == _init().n


def test_iid_uniform_and_proposal_attempt_share_the_complete_attempt_stream():
    """A/C target arms differ only after sampling, never in the sampled attempts."""
    inputs = _inputs()
    proposals = _proposal_fields(inputs)

    def run(target_mode: str) -> dict:
        config = CompactTrainConfig(
            **{
                **_config("area_gaussian").__dict__,
                "schedule_mode": "iid",
                "target_mode": target_mode,
                "evaluate_checkpoint_risks": False,
            }
        )
        _, history = CompactTrainer(config).train(
            inputs,
            _init(),
            proposal_fields=proposals,
        )
        return history

    uniform = run("uniform")
    proposal_attempt = run("proposal_attempt")

    assert uniform["view_schedule"] == [1, 0]
    assert uniform["view_schedule"] == proposal_attempt["view_schedule"]
    invariant_keys = (
        "view_index",
        "sample_seed",
        "xy_sha256",
        "active_sha256",
        "inside_fit_window_sha256",
        "proposal_component_ids_sha256",
        "proposal_density_sha256",
        "joint_density_sha256",
    )
    for uniform_step, proposal_step in zip(
        uniform["steps"], proposal_attempt["steps"], strict=True
    ):
        for key in invariant_keys:
            assert uniform_step[key] == proposal_step[key]
        assert uniform_step["target_density_sha256"] != proposal_step["target_density_sha256"]
        assert uniform_step["importance_sha256"] != proposal_step["importance_sha256"]


def test_all_invisible_microbatches_still_advance_six_aligned_adam_groups():
    source = _init()
    behind = Gaussians3D(
        means=torch.tensor([[0.0, 0.0, -5.0], [0.1, 0.0, -5.0]]),
        quats=source.quats,
        log_scales=source.log_scales,
        opacity=source.opacity,
        sh=source.sh,
    )
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=6,
        proposal_mode="pixel_uniform",
        seed=_DEV_SEED,
        extent=1.0,
        outer_microbatch=2,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    final, history = CompactTrainer(config).train(_inputs(), behind)
    assert history["steps"][0]["visible_count"] == 0
    assert set(history["optimizer_steps"].values()) == {1}
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        torch.testing.assert_close(getattr(final, name), getattr(behind, name))


def test_outer_microbatch_matches_one_batch_update_and_sample_stream():
    inputs = _inputs()
    one_config = CompactTrainConfig(
        **{
            **_config("pixel_gaussian", outer=12).__dict__,
            "iterations": 1,
            "checkpoints": (0, 1),
        }
    )
    micro_config = CompactTrainConfig(
        **{
            **_config("pixel_gaussian", outer=3).__dict__,
            "iterations": 1,
            "checkpoints": (0, 1),
        }
    )
    one, one_history = CompactTrainer(one_config).train(inputs, _init())
    micro, micro_history = CompactTrainer(micro_config).train(inputs, _init())

    one_step = one_history["steps"][0]
    micro_step = micro_history["steps"][0]
    for key in ("xy_sha256", "active_sha256", "importance_sha256"):
        assert one_step[key] == micro_step[key]
    assert one_step["sampled_loss"] == pytest.approx(micro_step["sampled_loss"], abs=2e-7)
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        torch.testing.assert_close(getattr(one, name), getattr(micro, name), atol=5e-6, rtol=5e-5)


def test_streamed_pixel_and_area_evaluation_match_materialized_reference():
    inputs = _inputs()
    trainer = CompactTrainer(_config("area_uniform"))
    metrics = trainer.evaluate(inputs, _init())
    renderer = TorchPointRasterizer(point_chunk=3, gaussian_chunk=1)
    indexes = [GaussianObservationIndex(field, tile_size=4) for field in inputs.observations]

    expected = {}
    for measure, offsets in {
        "pixel": ((0.5, 0.5),),
        "area": ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)),
    }.items():
        view_mses = []
        for field, camera, backend in zip(
            inputs.observations, inputs.cameras, indexes, strict=True
        ):
            fit_x, fit_y, fit_width, fit_height = field.fit_window
            values = []
            for ox, oy in offsets:
                y, x = torch.meshgrid(
                    torch.arange(fit_y, fit_y + fit_height, dtype=torch.float32) + oy,
                    torch.arange(fit_x, fit_x + fit_width, dtype=torch.float32) + ox,
                    indexing="ij",
                )
                xy = torch.stack([x, y], dim=-1).reshape(-1, 2)
                teacher = backend.query(xy, component_chunk=1).color
                prediction = renderer.render_points(_init(), camera, xy, sh_degree=0).color
                values.append((prediction.double() - teacher.double()).square().reshape(-1))
            error = torch.cat(values)
            view_mses.append(float(error.sum(dtype=torch.float64) / error.numel()))
        expected[measure] = sum(view_mses) / len(view_mses)
    assert metrics["J_pixel"] == pytest.approx(expected["pixel"], abs=1e-14)
    assert metrics["J_area"] == pytest.approx(expected["area"], abs=1e-14)
    assert metrics["pixel"]["scalar_count"] == 2 * 7 * 5 * 3
    assert metrics["area"]["scalar_count"] == 4 * 2 * 7 * 5 * 3


def test_preflight_estimate_matches_constructed_index_and_fails_before_allocation(monkeypatch):
    inputs = _inputs()
    config = _config("pixel_uniform")
    report = preflight_observations(inputs, config)
    for field, view in zip(inputs.observations, report["views"], strict=True):
        index = GaussianObservationIndex(field, tile_size=config.teacher_tile_size)
        assert view["estimated_index_entries"] == index.n_entries
        assert (
            compact_trainer_module._estimate_device_index_entries(
                field,
                tile_size=config.teacher_tile_size,
                component_chunk=1,
            )
            == index.n_entries
        )

    called = False
    original = GaussianObservationIndex.__init__

    def tracked_init(self, *args, **kwargs):
        nonlocal called
        called = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GaussianObservationIndex, "__init__", tracked_init)
    blocked = CompactTrainConfig(
        **{
            **config.__dict__,
            "max_index_entries_per_view": 1,
        }
    )
    with pytest.raises(ValueError, match="index entries"):
        CompactTrainer(blocked).train(inputs, _init())
    assert not called


def test_explicit_proposal_geometry_mismatch_fails_before_index_allocation(monkeypatch):
    inputs = _inputs()
    proposals = _proposal_fields(inputs)
    source = proposals[0]
    proposals[0] = GaussianObservationField(
        width=source.width,
        height=source.height,
        means=source.means + torch.tensor([0.01, 0.0]),
        log_scales=source.log_scales,
        rotations=source.rotations,
        colors=source.colors,
        amplitudes=source.amplitudes,
        filter_variance=source.filter_variance,
        epsilon=source.epsilon,
        sigma_cutoff=source.sigma_cutoff,
        support_fade_alpha=source.support_fade_alpha,
        aa_dilation=source.aa_dilation,
        view_id=source.view_id,
        fit_window=source.fit_window,
        n_init=source.n_init,
        provider=source.provider,
    )
    allocated = False

    def forbidden_index(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("proposal validation must precede every index allocation")

    monkeypatch.setattr(GaussianObservationIndex, "__init__", forbidden_index)
    with pytest.raises(ValueError, match="means differs"):
        CompactTrainer(_config("area_gaussian")).train(
            inputs,
            _init(),
            proposal_fields=proposals,
        )
    assert allocated is False


def test_explicit_proposal_preflight_records_unequal_per_view_component_counts():
    inputs = _unequal_component_inputs()
    proposals = _proposal_fields(inputs)
    config = CompactTrainConfig(
        iterations=2,
        attempts_per_step=8,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        seed=_DEV_SEED,
        extent=1.0,
        point_chunk=2,
        gaussian_chunk=1,
        outer_microbatch=4,
        query_component_chunk=1,
        teacher_tile_size=4,
        checkpoints=(0, 2),
        evaluate_checkpoint_risks=False,
    )
    _, history = CompactTrainer(config).train(
        inputs,
        _init(),
        proposal_fields=proposals,
    )

    assert history["view_visit_counts"] == [1, 1]
    assert [view["components"] for view in history["preflight"]["views"]] == [2, 3]
    assert [view["components"] for view in history["proposal_preflight"]["views"]] == [2, 3]
    assert [item["components"] for item in history["proposal_normalizers"]] == [2, 3]
    assert history["n_init_3d"] == history["n_opt_3d"] == 2


def test_anchor_amplitude_times_center_scalar_is_exact_split_invariant():
    teacher = _inputs().observations[0]
    center_scalar = torch.tensor([0.6, 0.4])
    parent = _proposal_field(teacher, center_scalar)
    ids = torch.tensor([0, 0, 1])
    child_anchor = GaussianObservationField(
        width=teacher.width,
        height=teacher.height,
        means=teacher.means[ids],
        log_scales=teacher.log_scales[ids],
        rotations=teacher.rotations[ids],
        colors=teacher.colors[ids],
        amplitudes=torch.tensor([0.2, 0.6, float(teacher.amplitudes[1])]),
        epsilon=teacher.epsilon,
        sigma_cutoff=teacher.sigma_cutoff,
        support_fade_alpha=teacher.support_fade_alpha,
        aa_dilation=teacher.aa_dilation,
        fit_window=teacher.fit_window,
        provider="synthetic_fixture",
    )
    child_scalar = center_scalar[ids]
    children = _proposal_field(child_anchor, child_scalar)
    parent_proposal = GaussianPointProposal(parent)
    child_proposal = GaussianPointProposal(children)
    xy = parent.pixel_centers()

    torch.testing.assert_close(child_proposal.total_mass, parent_proposal.total_mass)
    torch.testing.assert_close(
        child_proposal.gaussian_density(xy),
        parent_proposal.gaussian_density(xy),
    )

    scalar_only_parent = GaussianObservationField(
        width=teacher.width,
        height=teacher.height,
        means=teacher.means,
        log_scales=teacher.log_scales,
        rotations=teacher.rotations,
        colors=torch.ones_like(teacher.colors),
        amplitudes=center_scalar,
        epsilon=teacher.epsilon,
        sigma_cutoff=teacher.sigma_cutoff,
        support_fade_alpha=teacher.support_fade_alpha,
        aa_dilation=teacher.aa_dilation,
        fit_window=teacher.fit_window,
        provider="synthetic_fixture",
    )
    scalar_only_children = GaussianObservationField(
        width=children.width,
        height=children.height,
        means=children.means,
        log_scales=children.log_scales,
        rotations=children.rotations,
        colors=children.colors,
        amplitudes=child_scalar,
        epsilon=children.epsilon,
        sigma_cutoff=children.sigma_cutoff,
        support_fade_alpha=children.support_fade_alpha,
        aa_dilation=children.aa_dilation,
        fit_window=children.fit_window,
        provider="synthetic_fixture",
    )
    assert not torch.allclose(
        GaussianPointProposal(scalar_only_children).gaussian_density(xy),
        GaussianPointProposal(scalar_only_parent).gaussian_density(xy),
    )


def test_non_cpu_preflight_estimate_does_not_bulk_copy_teacher(monkeypatch):
    inputs = _inputs()
    source = inputs.observations[0]

    class DeviceTaggedField:
        @property
        def device(self):
            return torch.device("cuda")

        def to(self, _device):
            raise AssertionError("preflight must not bulk-copy a device teacher")

        def __getattr__(self, name):
            return getattr(source, name)

    inputs.observations[0] = DeviceTaggedField()
    expected = GaussianObservationIndex.estimate_entries(
        source, tile_size=_config("pixel_uniform").teacher_tile_size
    )
    report = preflight_observations(inputs, _config("pixel_uniform"))
    assert report["views"][0]["estimated_index_entries"] == expected


def test_preflight_rejects_oversized_teacher_before_any_whole_input_device_transfer(
    monkeypatch,
):
    inputs = _inputs_with_optional_geometry()
    transferred = False

    def forbidden_transfer(self, device):
        del self, device
        nonlocal transferred
        transferred = True
        raise AssertionError("whole ReconstructionInputs moved before compact preflight")

    monkeypatch.setattr(ReconstructionInputs, "to", forbidden_transfer)
    blocked = CompactTrainConfig(
        **{
            **_config("pixel_uniform").__dict__,
            "device": "meta",
            "max_components_per_view": 1,
        }
    )
    with pytest.raises(ValueError, match="view 0 components"):
        CompactTrainer(blocked).train(inputs, _init())
    assert transferred is False


def test_compact_device_working_set_moves_only_teachers_and_cameras(monkeypatch):
    inputs = _inputs_with_optional_geometry()
    geometry_ids = {
        id(inputs.points),
        id(inputs.bounds_hint[0]),
        *(id(indices) for indices in inputs.point_visibility),
    }
    teacher_moves = []
    camera_moves = []
    original_tensor_to = torch.Tensor.to

    def guarded_tensor_to(self, *args, **kwargs):
        if id(self) in geometry_ids:
            raise AssertionError("unused optional geometry was transferred")
        return original_tensor_to(self, *args, **kwargs)

    def record_teacher_move(self, device):
        teacher_moves.append((self, torch.device(device)))
        return self

    def record_camera_move(self, device):
        camera_moves.append((self, torch.device(device)))
        return self

    monkeypatch.setattr(torch.Tensor, "to", guarded_tensor_to)
    monkeypatch.setattr(GaussianObservationField, "to", record_teacher_move)
    monkeypatch.setattr(Camera, "to", record_camera_move)
    working = compact_trainer_module._compact_working_inputs(inputs, torch.device("meta"))

    assert [field for field, _ in teacher_moves] == inputs.observations
    assert [camera for camera, _ in camera_moves] == inputs.cameras
    assert {device for _, device in (*teacher_moves, *camera_moves)} == {torch.device("meta")}
    assert all(
        working_field is source_field
        for working_field, source_field in zip(
            working.observations, inputs.observations, strict=True
        )
    )
    assert all(
        working_camera is source_camera
        for working_camera, source_camera in zip(working.cameras, inputs.cameras, strict=True)
    )
    assert working.points is None
    assert working.point_visibility is None
    assert working.bounds_hint is None
    assert working.view_names == inputs.view_names
    assert working.name == inputs.name
    assert working.archive_stats is inputs.archive_stats


def test_host_bounds_extent_semantics_survive_geometry_free_working_set():
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=4,
        proposal_mode="pixel_uniform",
        seed=_DEV_SEED,
        extent=None,
        outer_microbatch=4,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    _, history = CompactTrainer(config).train(_inputs_with_optional_geometry(), _init())
    assert history["extent"] == pytest.approx(3.25)
    assert history["extent_source"] == "bounds_hint"


def test_initial_cloud_extent_fallback_is_frozen_and_finite():
    inputs = _inputs()
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=4,
        proposal_mode="pixel_uniform",
        seed=_DEV_SEED,
        extent=None,
        outer_microbatch=4,
        evaluation_chunk=64,
        checkpoints=(0, 1),
    )
    _, history = CompactTrainer(config).train(inputs, _init())
    expected_center = 0.5 * (_init().means.amin(dim=0) + _init().means.amax(dim=0))
    expected_radius = torch.linalg.vector_norm(_init().means - expected_center, dim=1).amax()
    assert history["extent_source"] == "initial_mean_cloud"
    assert history["extent"] == pytest.approx(max(2.2 * float(expected_radius), 1e-3))


def test_checkpoint_evaluation_switch_preserves_snapshot_schedule_without_expansion():
    inputs = _inputs()

    class TrackingBackend:
        def __init__(self, field):
            self.field = field
            self.backend = GaussianObservationIndex(field, tile_size=4)
            self.color_query_points = 0

        def query(self, xy, component_chunk=4096):
            self.color_query_points += xy.shape[0]
            return self.backend.query(xy, component_chunk=component_chunk)

        def query_weight_sum(self, xy, component_chunk=4096):
            return self.backend.query_weight_sum(xy, component_chunk=component_chunk)

    backends = [TrackingBackend(field) for field in inputs.observations]
    callback_steps = []
    config = CompactTrainConfig(
        iterations=1,
        attempts_per_step=4,
        proposal_mode="pixel_uniform",
        seed=_DEV_SEED,
        extent=1.0,
        outer_microbatch=4,
        evaluation_chunk=1,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    _, history = CompactTrainer(config).train(
        inputs,
        _init(),
        query_backends=backends,
        checkpoint_callback=lambda _snapshot, step: callback_steps.append(step),
    )
    assert callback_steps == [0, 1]
    assert [item["step"] for item in history["checkpoints"]] == [0, 1]
    assert all(item["evaluation"] is None for item in history["checkpoints"])
    assert all(len(item["snapshot_sha256"]) == 64 for item in history["checkpoints"])
    assert history["checkpoint_risk_evaluation_enabled"] is False
    assert history["checkpoint_risk_evaluation_call_count"] == 0
    assert history["checkpoint_snapshot_count"] == 2
    assert history["checkpoint_callback_call_count"] == 2
    # Only the four training attempts are color-queried; evaluation_chunk=1 would otherwise
    # make checkpoint expansion immediately visible here.
    assert sum(backend.color_query_points for backend in backends) == 4


def test_offgrid_coordinate_mean_and_logscale_gradients_match_float64_central_difference():
    dtype = torch.float64
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, -2.0]),
        torch.zeros(3),
        width=15,
        height=13,
        fov_x_deg=48.0,
    )
    means = torch.tensor([[0.08, -0.04, 0.02]], dtype=dtype, requires_grad=True)
    quats = torch.tensor([[0.91, 0.16, -0.25, 0.29]], dtype=dtype)
    log_scales = torch.log(torch.tensor([[0.18, 0.105, 0.075]], dtype=dtype)).requires_grad_()
    opacity = torch.tensor([0.72], dtype=dtype)
    sh = rgb_to_sh(torch.tensor([[0.72, 0.31, 0.18]], dtype=dtype))[:, None]
    projected, _ = camera.project(means.detach())
    xy = (projected + torch.tensor([[0.37, -0.29]], dtype=dtype)).requires_grad_()
    target = torch.tensor([[0.19, 0.52, 0.41]], dtype=dtype)
    renderer = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1)

    def objective(
        current_means: torch.Tensor,
        current_log_scales: torch.Tensor,
        current_xy: torch.Tensor,
    ) -> torch.Tensor:
        gaussians = Gaussians3D(
            current_means,
            quats,
            current_log_scales,
            opacity,
            sh,
        )
        color = renderer.render_points(gaussians, camera, current_xy, sh_degree=0).color
        return (color - target).square().mean()

    objective(means, log_scales, xy).backward()
    autograd = {
        "coordinate": xy.grad.detach().clone(),
        "mean": means.grad.detach().clone(),
        "log_scale": log_scales.grad.detach().clone(),
    }
    bases = {
        "coordinate": xy.detach(),
        "mean": means.detach(),
        "log_scale": log_scales.detach(),
    }

    def central_difference(name: str) -> torch.Tensor:
        step = 1e-4
        result = torch.empty_like(bases[name])
        for flat_index in range(result.numel()):
            positive = bases[name].clone()
            negative = bases[name].clone()
            positive.reshape(-1)[flat_index] += step
            negative.reshape(-1)[flat_index] -= step
            arguments = {
                "current_means": bases["mean"],
                "current_log_scales": bases["log_scale"],
                "current_xy": bases["coordinate"],
            }
            argument_name = {
                "coordinate": "current_xy",
                "mean": "current_means",
                "log_scale": "current_log_scales",
            }[name]
            arguments[argument_name] = positive
            positive_loss = objective(**arguments)
            arguments[argument_name] = negative
            negative_loss = objective(**arguments)
            result.reshape(-1)[flat_index] = (positive_loss - negative_loss) / (2.0 * step)
        return result

    for name, gradient in autograd.items():
        assert float(gradient.abs().max()) > 1e-8
        finite_difference = central_difference(name)
        torch.testing.assert_close(gradient, finite_difference, atol=2e-6, rtol=2e-3)


def test_global_sample_loss_changes_when_unrelated_visible_gaussian_changes():
    camera = Camera.look_at(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), width=11, height=11)
    gaussians = Gaussians3D.from_means_covs(
        means=torch.tensor([[0.0, 0.0, -0.18], [0.03, -0.02, 0.22]]),
        covs=torch.eye(3)[None].repeat(2, 1, 1) * 0.12**2,
        colors=torch.tensor([[0.82, 0.12, 0.16], [0.08, 0.78, 0.21]]),
        opacity=torch.tensor([0.68, 0.74]),
    )
    xy = torch.tensor([[5.37, 5.64]])
    target = torch.tensor([[0.24, 0.18, 0.69]])
    renderer = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1)

    baseline = (renderer.render_points(gaussians, camera, xy).color - target).square().mean()
    changed_sh = gaussians.sh.clone()
    changed_sh[1, 0] = rgb_to_sh(torch.tensor([0.13, 0.16, 0.91]))
    intervention = Gaussians3D(
        gaussians.means,
        gaussians.quats,
        gaussians.log_scales,
        gaussians.opacity,
        changed_sh,
    )
    changed = (renderer.render_points(intervention, camera, xy).color - target).square().mean()
    assert float((changed - baseline).abs()) > 1e-5


def _sample_slice(samples: ObservationSamples, start: int, end: int) -> ObservationSamples:
    return ObservationSamples(
        xy=samples.xy[start:end],
        proposal_component_ids=samples.proposal_component_ids[start:end],
        proposal_density=samples.proposal_density[start:end],
        joint_density=samples.joint_density[start:end],
        target_density=samples.target_density[start:end],
        importance=samples.importance[start:end],
        inside_fit_window=samples.inside_fit_window[start:end],
        active=samples.active[start:end],
        risk_measure=samples.risk_measure,
    )


def test_presampled_loss_and_five_family_gradients_match_outer_microbatches_and_chunks():
    inputs = _inputs()
    init = _init()
    field = inputs.observations[0]
    backend = GaussianObservationIndex(field, tile_size=4)
    samples = GaussianPixelProposal(field, backend).sample(
        12,
        uniform_fraction=0.25,
        generator=torch.Generator(device="cpu").manual_seed(_DEV_SEED + 8),
    )

    def loss_and_gradients(
        outer_microbatch: int,
        point_chunk: int,
        gaussian_chunk: int,
    ) -> tuple[float, dict[str, torch.Tensor]]:
        parameters = {
            "means": torch.nn.Parameter(init.means.detach().clone()),
            "quaternions": torch.nn.Parameter(init.quats.detach().clone()),
            "log_scales": torch.nn.Parameter(init.log_scales.detach().clone()),
            "opacity_logits": torch.nn.Parameter(torch.logit(init.opacity).detach().clone()),
            "sh0": torch.nn.Parameter(init.sh.detach().clone()),
        }

        def build() -> Gaussians3D:
            return Gaussians3D(
                parameters["means"],
                parameters["quaternions"],
                parameters["log_scales"],
                torch.sigmoid(parameters["opacity_logits"]),
                parameters["sh0"],
            )

        renderer = TorchPointRasterizer(
            point_chunk=point_chunk,
            gaussian_chunk=gaussian_chunk,
        )
        detached_loss = 0.0
        for start in range(0, 12, outer_microbatch):
            end = min(start + outer_microbatch, 12)
            current = _sample_slice(samples, start, end)
            target = backend.query(current.xy, component_chunk=1).color
            prediction = renderer.render_points(
                build(), inputs.cameras[0], current.xy, sh_degree=0
            ).color
            point_loss = (prediction - target).square().mean(dim=-1)
            scaled = fixed_attempt_mean(point_loss, current) * ((end - start) / 12)
            detached_loss += float(scaled.detach())
            scaled = scaled + sum(parameter.sum() * 0.0 for parameter in parameters.values())
            scaled.backward()
        return detached_loss, {
            name: parameter.grad.detach().clone() for name, parameter in parameters.items()
        }

    reference_loss, reference_gradients = loss_and_gradients(12, 12, 8)
    streamed_loss, streamed_gradients = loss_and_gradients(3, 2, 1)
    assert reference_loss == pytest.approx(streamed_loss, abs=5e-6, rel=5e-5)
    for name, reference in reference_gradients.items():
        streamed = streamed_gradients[name]
        assert bool(torch.isfinite(reference).all()) and bool(torch.isfinite(streamed).all())
        assert float(reference.abs().max()) > 1e-10
        torch.testing.assert_close(streamed, reference, atol=5e-6, rtol=5e-5)


def test_fresh_process_strict_bundle_training_denies_rgb_scene_and_source_access(tmp_path):
    bundle = tmp_path / "compact-bundle"
    initialization = tmp_path / "initialization.npz"
    _inputs().save(bundle)
    _init().save_npz(initialization)
    script = textwrap.dedent(
        """
        import builtins
        import io
        import json
        import os
        import sys
        from pathlib import Path

        bundle_path, initialization_path = sys.argv[1:]
        source_denials = 0
        pil_denials = 0
        loader_denials = 0
        scene_denials = 0

        def forbidden(*_args, **_kwargs):
            raise AssertionError("RGB/SceneData/source-image access is forbidden")

        def source_like(path):
            try:
                candidate = Path(os.fspath(path))
            except TypeError:
                return False
            image_suffix = candidate.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"
            }
            return image_suffix or "dataset" in {part.lower() for part in candidate.parts}

        original_builtin_open = builtins.open
        original_io_open = io.open
        original_os_open = os.open

        def guarded_builtin_open(path, *args, **kwargs):
            global source_denials
            if source_like(path):
                source_denials += 1
                forbidden()
            return original_builtin_open(path, *args, **kwargs)

        def guarded_io_open(path, *args, **kwargs):
            global source_denials
            if source_like(path):
                source_denials += 1
                forbidden()
            return original_io_open(path, *args, **kwargs)

        def guarded_os_open(path, *args, **kwargs):
            global source_denials
            if source_like(path):
                source_denials += 1
                forbidden()
            return original_os_open(path, *args, **kwargs)

        # Install filesystem hooks before importing any trainer, lifter, or rtgs.data module.
        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        os.open = guarded_os_open

        # Negative control: Path.read_bytes uses io.open, not builtins.open.
        source_probe = Path(bundle_path).parent / "dataset" / "source-frame.bin"
        try:
            source_probe.read_bytes()
        except AssertionError:
            source_denial_proven = True
        else:
            raise AssertionError("dataset Path.read_bytes bypassed the denial hooks")
        try:
            os.open(source_probe, os.O_RDONLY)
        except AssertionError:
            os_source_denial_proven = True
        else:
            raise AssertionError("dataset os.open bypassed the denial hooks")

        from PIL import Image as PILImage

        import rtgs.data as data_package
        import rtgs.data.calibrated as calibrated_module
        import rtgs.data.scene as scene_module
        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.data.reconstruction_inputs import ReconstructionInputs

        def deny_pil(*_args, **_kwargs):
            global pil_denials
            pil_denials += 1
            forbidden()

        def deny_loader(*_args, **_kwargs):
            global loader_denials
            loader_denials += 1
            forbidden()

        PILImage.open = deny_pil

        class ForbiddenSceneData:
            def __init__(self, *_args, **_kwargs):
                global scene_denials
                scene_denials += 1
                forbidden()

        scene_module.SceneData = ForbiddenSceneData
        calibrated_module.SceneData = ForbiddenSceneData
        data_package.SceneData = ForbiddenSceneData
        calibrated_module.load_calibrated_scene = deny_loader
        calibrated_module._resize_image = deny_loader
        data_package.load_calibrated_scene = deny_loader

        try:
            PILImage.open(source_probe)
        except AssertionError:
            pil_denial_proven = True
        else:
            raise AssertionError("PIL alias bypassed denial")
        for loader in (
            calibrated_module.load_calibrated_scene,
            calibrated_module._resize_image,
            data_package.load_calibrated_scene,
        ):
            try:
                loader(source_probe)
            except AssertionError:
                pass
            else:
                raise AssertionError("calibrated loader alias bypassed denial")
        for scene_class in (
            scene_module.SceneData,
            calibrated_module.SceneData,
            data_package.SceneData,
        ):
            try:
                scene_class()
            except AssertionError:
                pass
            else:
                raise AssertionError("SceneData alias bypassed denial")

        from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer

        # Strict bundle/NPZ reads are outside dataset and must remain allowed.
        inputs = ReconstructionInputs.load(bundle_path, strict=True)
        init = Gaussians3D.load_npz(initialization_path)
        config = CompactTrainConfig(
            iterations=1,
            attempts_per_step=4,
            proposal_mode="pixel_uniform",
            seed=63831,
            extent=1.0,
            outer_microbatch=2,
            checkpoints=(0, 1),
            evaluate_checkpoint_risks=False,
        )
        final, history = CompactTrainer(config).train(
            inputs,
            init,
            bundle_path=bundle_path,
        )
        print(json.dumps({
            "n": final.n,
            "clocks": history["optimizer_steps"],
            "archive": history["preflight"]["archive"],
            "risk_calls": history["checkpoint_risk_evaluation_call_count"],
            "source_denial_proven": source_denial_proven,
            "os_source_denial_proven": os_source_denial_proven,
            "pil_denial_proven": pil_denial_proven,
            "source_denials": source_denials,
            "pil_denials": pil_denials,
            "loader_denials": loader_denials,
            "scene_denials": scene_denials,
        }, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(bundle), str(initialization)],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    record = json.loads(completed.stdout)
    assert record["n"] == 2
    assert set(record["clocks"].values()) == {1}
    assert record["archive"]["teacher_archive_count"] == 2
    assert record["risk_calls"] == 0
    assert record["source_denial_proven"] is True
    assert record["os_source_denial_proven"] is True
    assert record["pil_denial_proven"] is True
    assert record["source_denials"] == 2
    assert record["pil_denials"] == 1
    assert record["loader_denials"] == 3
    assert record["scene_denials"] == 3
