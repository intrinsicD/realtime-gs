"""Focused, nonofficial checks for responsibility-birth allocation iter2.

Every randomized experiment fixture in this module uses either an iter2 focused-only
root (``994xxx``) or an unrelated development seed.  In particular, the official
``781xx``--``784xx`` roots are inspected only as inert values by the fail-closed root
guard; they are never passed to a generator, schedule, sampler, bank, or trainer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from benchmarks import compact_responsibility_birth_allocation_iter2 as birth

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.render.point_base import PointRenderOutput
from rtgs.render.torch_points import TorchPointRasterizer

_DEV_SEED = 641_901
_BANK_FIELDS = frozenset(
    {
        "xy",
        "active",
        "inside_fit_window",
        "proposal_component_ids",
        "proposal_density",
        "joint_density",
        "target_density",
        "importance",
        "color",
    }
)


def _selection_fixture() -> dict[str, torch.Tensor | float | int]:
    n = 40
    support = torch.arange(1, n + 1, dtype=torch.float64)
    residual = torch.arange(n, 0, -1, dtype=torch.float64).square()
    return {
        "gradient_score": torch.roll(residual.to(torch.float32), 7),
        "residual_score": residual,
        "support_score": support,
        "support_by_view": support[None, :].repeat(3, 1) / 3.0,
        "visible_step_count": torch.ones(n, dtype=torch.int64),
        "scale_max": torch.cat((torch.full((20,), 0.005), torch.full((20,), 0.02))),
        "persistent_ids": torch.arange(n, dtype=torch.long),
        "extent": 1.0,
        "shuffle_root": birth.FOCUSED_SHUFFLE_ROOTS[0],
    }


def _params(n: int = 8) -> dict[str, torch.nn.Parameter]:
    scales = torch.empty(n, 3)
    scales[: n // 2] = math.log(0.005)
    scales[n // 2 :] = math.log(0.02)
    means = torch.linspace(-0.2, 0.2, n * 3).reshape(n, 3)
    means[:, 2] += 2.4
    return {
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)),
        "scales": torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(torch.linspace(-0.8, 0.8, n)),
        "sh0": torch.nn.Parameter(torch.linspace(-0.4, 0.4, n * 3).reshape(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.empty(n, 0, 3)),
    }


def _optimizers(
    params: dict[str, torch.nn.Parameter],
) -> dict[str, torch.optim.Optimizer]:
    return {
        name: torch.optim.Adam(
            [
                {
                    "params": [parameter],
                    "lr": 1e-3 * (index + 1),
                    "name": name,
                    "iter2_test_tag": index,
                }
            ],
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False,
            foreach=False,
            fused=False,
        )
        for index, (name, parameter) in enumerate(params.items())
    }


def _phase_a_optimizers(
    params: dict[str, torch.nn.Parameter],
    *,
    extent: float,
) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": 1.6e-4 * extent,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    return {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": lrs[name], "name": name}],
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False,
            foreach=False,
            fused=False,
        )
        for name, parameter in params.items()
    }


def _initialize_adam(
    params: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    clock: int,
) -> None:
    loss = sum(
        (index + 1) * parameter.square().sum() for index, parameter in enumerate(params.values())
    )
    loss.backward()
    for name, optimizer in optimizers.items():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = optimizer.state[params[name]]
        state["step"].fill_(clock)


def _artificial_score_step(
    controller: birth.ResponsibilityBirthController,
    params: dict[str, torch.nn.Parameter],
    *,
    step: int,
    view_index: int,
    n: int,
    empty_visible: bool = False,
    reverse_screen_gradient: bool = False,
) -> None:
    attempts = 6
    visible_n = 0 if empty_visible else n
    means2d = torch.nn.Parameter(
        torch.stack(
            (torch.arange(visible_n, dtype=torch.float32), torch.ones(visible_n)),
            dim=-1,
        )
    )
    basis = torch.stack(
        (
            torch.linspace(0.1, 0.8, visible_n),
            torch.linspace(0.2, 0.9, visible_n),
            torch.linspace(0.3, 1.0, visible_n),
        ),
        dim=-1,
    )
    basis = (basis + 0.0 * means2d[:, :1]).requires_grad_(True)
    raw = (
        torch.arange(attempts * visible_n, dtype=torch.float32).reshape(attempts, visible_n)
        + 1
        + view_index
    )
    weights = raw / (2.0 * raw.sum(dim=1, keepdim=True)) if visible_n else torch.empty(attempts, 0)
    position_term = (weights @ means2d[:, :1]).expand(-1, 3) * 1e-3
    color = weights @ basis + position_term
    output = PointRenderOutput(
        color=color,
        alpha=weights.sum(dim=1),
        depth=torch.zeros(attempts),
        visible=torch.arange(visible_n),
        means2d=means2d,
        compositing_color_basis=basis,
    )
    point_loss = (color.detach() - 0.25).square().mean(dim=-1)
    active = torch.tensor((True, False, True, True, False, True))
    controller.observe_pre_backward(
        step=step,
        view_index=view_index,
        output=output,
        point_loss=point_loss,
        active=active,
        attempts=attempts,
    )
    assert output.compositing_color_basis is None
    color.square().mean().backward()
    if reverse_screen_gradient:
        output.means2d.grad = torch.stack(
            (
                torch.linspace(2.0, 0.1, visible_n),
                torch.zeros(visible_n),
            ),
            dim=-1,
        )
    # The real trainer has populated all six model gradients by this boundary.
    # This isolated compositor fixture is independent of those parameters, so
    # provide deterministic finite stand-ins for the evidence recorder.
    for index, parameter in enumerate(params.values()):
        parameter.grad = torch.full_like(parameter, (index + 1) * step / 100.0)
    controller.observe_post_backward(
        step=step,
        view_index=view_index,
        output=output,
        width=64,
        height=48,
    )


def _run_score_window(
    controller: birth.ResponsibilityBirthController,
    params: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    visits: tuple[int, ...] = (0, 1) * 5,
) -> dict[str, torch.Tensor]:
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=2,
        attempts_per_step=6,
    )
    current: dict[str, torch.Tensor] = params
    for step, view_index in enumerate(visits, start=1):
        _artificial_score_step(
            controller,
            params,
            step=step,
            view_index=view_index,
            n=params["means"].shape[0],
        )
        current = controller.after_step(
            step=step,
            params=current,
            optimizers=optimizers,
            snapshot=birth._params_to_gaussians(current).detach(),
        )
    return current


def _tiny_inputs() -> tuple[ReconstructionInputs, list[GaussianObservationField]]:
    camera = Camera.look_at(
        eye=torch.tensor([0.0, 0.0, -2.4]),
        target=torch.zeros(3),
        width=9,
        height=7,
        fov_x_deg=52.0,
    )
    world = torch.tensor([[-0.13, -0.08, 0.0], [0.16, 0.10, 0.08]])
    means, depth = camera.project(world)
    assert bool((depth > 0).all())
    teacher = GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=means,
        log_scales=torch.log(torch.tensor([[1.1, 0.8], [0.9, 1.2]])),
        rotations=torch.tensor([0.25, -0.35]),
        colors=torch.tensor([[0.82, 0.18, 0.22], [0.12, 0.72, 0.88]]),
        amplitudes=torch.tensor([0.8, 0.65]),
        epsilon=1e-8,
        sigma_cutoff=math.sqrt(10.0),
        fit_window=(1, 1, 7, 5),
        view_id="iter2-focused-view",
        n_init=3,
        provider="synthetic_fixture",
    )
    proposal = GaussianObservationField(
        width=teacher.width,
        height=teacher.height,
        means=teacher.means,
        log_scales=teacher.log_scales,
        rotations=teacher.rotations,
        colors=torch.ones_like(teacher.colors),
        amplitudes=teacher.amplitudes * torch.tensor([0.6, 0.8]),
        epsilon=teacher.epsilon,
        sigma_cutoff=teacher.sigma_cutoff,
        fit_window=teacher.fit_window,
        view_id=teacher.view_id,
        n_init=teacher.n_init,
        provider="synthetic_fixture",
    )
    inputs = ReconstructionInputs(
        observations=[teacher],
        cameras=[camera],
        view_names=["iter2-focused-view"],
        name="iter2-focused-bank",
    )
    return inputs, [proposal]


def _npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _camera_for_update() -> Camera:
    return Camera.look_at(
        eye=torch.tensor([0.0, 0.0, -2.0]),
        target=torch.tensor([0.0, 0.0, 2.0]),
        width=12,
        height=10,
        fov_x_deg=50.0,
    )


def _full_update_evidence(*, collect_basis: bool) -> dict[str, Any]:
    params = _params(6)
    optimizers = _optimizers(params)
    camera = _camera_for_update()
    xy = torch.tensor(
        [[2.5, 2.5], [4.5, 3.5], [6.5, 4.5], [8.5, 6.5]],
        dtype=torch.float32,
    )
    target = torch.linspace(0.1, 0.9, xy.shape[0] * 3).reshape(-1, 3)
    rng_before = torch.get_rng_state().clone()
    output = TorchPointRasterizer(point_chunk=2, gaussian_chunk=3).render_points(
        birth._params_to_gaussians(params),
        camera,
        xy,
        sh_degree=0,
        collect_compositing_color_basis=collect_basis,
    )
    vjp_hash = None
    if collect_basis:
        assert output.compositing_color_basis is not None
        point_loss = (output.color.detach() - target).square().mean(dim=-1)
        grad_outputs = torch.zeros_like(output.color)
        grad_outputs[:, 0] = point_loss
        grad_outputs[:, 1] = 1.0
        (vjp,) = torch.autograd.grad(
            output.color,
            output.compositing_color_basis,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
        )
        assert all(parameter.grad is None for parameter in params.values())
        vjp_hash = birth.tensor_hash(vjp.detach())
        output.compositing_color_basis = None
    loss = (output.color - target).square().mean() + 0.13 * output.alpha.mean()
    forward = {
        "color": birth.tensor_hash(output.color),
        "alpha": birth.tensor_hash(output.alpha),
        "depth": birth.tensor_hash(output.depth),
        "visible": birth.tensor_hash(output.visible),
        "loss": birth.tensor_hash(loss.reshape(1)),
    }
    loss.backward()
    assert output.means2d is not None and output.means2d.grad is not None
    gradients = {name: birth.tensor_hash(parameter.grad) for name, parameter in params.items()}
    means2d_gradient = birth.tensor_hash(output.means2d.grad)
    for optimizer in optimizers.values():
        optimizer.step()
    updates = {name: birth.tensor_hash(value) for name, value in params.items()}
    optimizer_state = {
        name: {
            key: (birth.tensor_hash(value) if isinstance(value, torch.Tensor) else value)
            for key, value in optimizer.state[params[name]].items()
        }
        for name, optimizer in optimizers.items()
    }
    return {
        "forward": forward,
        "gradients": gradients,
        "means2d_gradient": means2d_gradient,
        "updates": updates,
        "optimizer_state": optimizer_state,
        "rng_before": birth.tensor_hash(rng_before),
        "rng_after": birth.tensor_hash(torch.get_rng_state()),
        "diagnostic_vjp": vjp_hash,
    }


def _focused_replay_samples(
    generator: torch.Generator,
    *,
    attempts: int = 6,
) -> SimpleNamespace:
    xy = torch.rand(attempts, 2, generator=generator)
    active = torch.tensor((True, False, True, True, False, True))
    inside = torch.ones(attempts, dtype=torch.bool)
    component_ids = torch.tensor((-1, 0, 1, -1, 2, 3), dtype=torch.long)
    proposal_density = torch.linspace(0.2, 0.7, attempts)
    joint_density = proposal_density + 0.125
    active_float = active.to(proposal_density.dtype)
    return SimpleNamespace(
        xy=xy,
        active=active,
        inside_fit_window=inside,
        proposal_component_ids=component_ids,
        proposal_density=proposal_density,
        joint_density=joint_density,
        target_density=proposal_density * active_float,
        importance=active_float,
    )


def _focused_sample_seed(training_root: int, step_index: int) -> int:
    payload = f"rtgs.iter2.focused-recompute.v1\0{training_root}\0{step_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _sample_replay_fields(
    *,
    training_root: int,
    step_index: int,
    view_index: int,
    view_name: str,
) -> dict[str, Any]:
    seed = _focused_sample_seed(training_root, step_index)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = _focused_replay_samples(generator)
    return {
        "step": step_index + 1,
        "view_index": view_index,
        "view_name": view_name,
        "sample_seed": seed,
        "xy_sha256": birth.tensor_hash(samples.xy),
        "active_sha256": birth.tensor_hash(samples.active),
        "inside_fit_window_sha256": birth.tensor_hash(samples.inside_fit_window),
        "proposal_component_ids_sha256": birth.tensor_hash(samples.proposal_component_ids),
        "proposal_density_sha256": birth.tensor_hash(samples.proposal_density),
        "joint_density_sha256": birth.tensor_hash(samples.joint_density),
        "target_density_sha256": birth.tensor_hash(samples.target_density),
        "importance_sha256": birth.tensor_hash(samples.importance),
        "attempts": 6,
        "active_count": int(samples.active.sum()),
        "null_count": int((~samples.active).sum()),
        "invalid_count": int((~samples.inside_fit_window).sum()),
        "uniform_attempt_count": int((samples.proposal_component_ids == -1).sum()),
        "gaussian_attempt_count": int((samples.proposal_component_ids != -1).sum()),
        "gaussian_accepted_count": int(
            ((samples.proposal_component_ids != -1) & samples.active).sum()
        ),
        "gaussian_rejected_count": int(
            ((samples.proposal_component_ids != -1) & ~samples.active).sum()
        ),
    }


def _phase_a_recompute_worker_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    training_root = birth.FOCUSED_TRAIN_ROOTS[0]
    shuffle_root = birth.FOCUSED_SHUFFLE_ROOTS[0]
    view_names = ("focused-recompute-a", "focused-recompute-b")
    schedule = (0, 1) * 5
    monkeypatch.setattr(birth, "SCORE_STEPS", 10)
    monkeypatch.setattr(birth, "TRAIN_ITERATIONS", 10)
    monkeypatch.setattr(birth, "TRAIN_ATTEMPTS", 6)
    monkeypatch.setattr(birth, "EXPECTED_VIEWS", view_names)
    monkeypatch.setattr(birth, "EXPLICIT_EXTENT", 1.0)

    params = _params(835)
    optimizers = _phase_a_optimizers(params, extent=1.0)
    controller = birth.ResponsibilityBirthController(
        arm=None,
        split_root=None,
        shuffle_root=shuffle_root,
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=835,
        expected_clone_count=16,
        expected_split_count=16,
        quota_per_stratum=8,
    )
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=2,
        attempts_per_step=6,
    )
    _initialize_adam(params, optimizers, clock=10)
    optimizers["means"].param_groups[0]["lr"] = birth._expected_phase_a_optimizer_group(
        "means",
        10,
    )["lr"]
    current: dict[str, torch.Tensor] = params
    for step, view_index in enumerate(schedule, start=1):
        _artificial_score_step(
            controller,
            params,
            step=step,
            view_index=view_index,
            n=835,
            empty_visible=step == 1,
            reverse_screen_gradient=True,
        )
        current = controller.after_step(
            step=step,
            params=current,
            optimizers=optimizers,
            snapshot=birth._params_to_gaussians(current).detach(),
        )
    assert current is params
    controller_record = controller.history_record()
    assert controller_record["selection"]["all_phase_a_gates_pass"], json.dumps(
        controller_record["selection"]["gates_without_assigned_fraction"],
        sort_keys=True,
    )
    assert controller_record["step_evidence"][0]["visible_to_global"] == []
    assert controller_record["step_evidence"][0]["means2d_gradient"] is None

    history_steps = [
        {
            **_sample_replay_fields(
                training_root=training_root,
                step_index=step_index,
                view_index=view_index,
                view_name=view_names[view_index],
            ),
            "visible_count": 0 if step_index == 0 else 835,
            "sampled_loss": 0.125,
            "importance_max": 1.0,
            "importance_ess": 4.0,
            "importance_ess_per_attempt": 4.0 / 6.0,
            "rendered_point_gaussian_pairs": 0 if step_index == 0 else 835 * 6,
            "teacher_query_attempts": 6,
            "student_query_attempts": 6,
            "teacher_query_calls": 1,
            "student_query_calls": 1,
            "group_lrs_used": {
                name: float(optimizer.param_groups[0]["lr"])
                for name, optimizer in optimizers.items()
            },
            "gradient_max": {
                name: float(index + 1) for index, name in enumerate(birth.GROUP_ORDER)
            },
            "cardinality": 835,
        }
        for step_index, view_index in enumerate(schedule)
    ]
    schedule_sha = hashlib.sha256(
        json.dumps(list(schedule), separators=(",", ":")).encode()
    ).hexdigest()
    history = {
        "schema": "rtgs.compact_train_history.v2",
        "proposal_mode": "area_gaussian",
        "schedule_mode": "balanced_cycle",
        "target_mode": "proposal_attempt",
        "seed": training_root,
        "iterations": 10,
        "attempts_per_step": 6,
        "uniform_fraction": 0.25,
        "extent": 1.0,
        "extent_source": "explicit",
        "n_init_3d": 835,
        "n_opt_3d": 835,
        "completed_iterations": 10,
        "stop_after_step": 10,
        "topology_control_enabled": True,
        "proposal_field_source": "explicit",
        "view_schedule": list(schedule),
        "view_schedule_sha256": schedule_sha,
        "planned_view_visit_counts": [5, 5],
        "view_visit_counts": [5, 5],
        "teacher_digest_before": "focused-teacher",
        "teacher_digest_after": "focused-teacher",
        "proposal_digest_before": "focused-proposal",
        "proposal_digest_after": "focused-proposal",
        "proposal_branch_counts": {
            "uniform": 20,
            "gaussian": 40,
            "gaussian_accepted": 20,
            "gaussian_rejected": 20,
        },
        "proposal_view_diagnostics": [
            {
                "view_index": view_index,
                "view_name": view_name,
                "steps": 5,
                "attempts": 30,
                "active_count": 20,
                "null_count": 10,
                "active_fraction": 20 / 30,
                "null_fraction": 10 / 30,
                "gaussian_attempt_count": 20,
                "gaussian_accepted_count": 10,
                "gaussian_acceptance_fraction": 0.5,
            }
            for view_index, view_name in enumerate(view_names)
        ],
        "steps": history_steps,
        "topology_control": controller_record,
    }
    monkeypatch.setattr(birth, "ROOT", tmp_path)
    run_dir = tmp_path / "focused_recompute_run"
    monkeypatch.setattr(birth, "RUN_DIR", run_dir)
    worker_dir = run_dir / f"seed_{training_root}" / "phase_a"
    worker_dir.mkdir(parents=True)
    history_path = worker_dir / "history.json"
    history_sha = birth.exclusive_json(history_path, history)
    archive_path = worker_dir / "states.npz"
    archive = birth.save_state_archive(archive_path, controller)
    snapshot = birth._save_npz_snapshot(
        worker_dir / "gaussians_35_pre.npz",
        birth._params_to_gaussians(params).detach(),
    )
    monkeypatch.setattr(birth, "TRAIN_ROOTS", (training_root,))
    monkeypatch.setattr(birth, "SHUFFLE_ROOTS", (shuffle_root,))
    monkeypatch.setattr(birth, "_AUTHORIZED_PHASES", {"phase-a"})
    binding = {"focused": "binding"}
    monkeypatch.setattr(birth, "_verified_binding_receipt", lambda: binding)
    original_array_receipt = birth._array_tensor_receipt
    monkeypatch.setattr(
        birth,
        "_array_tensor_receipt",
        lambda array, device="cpu", include_values=False: original_array_receipt(
            array,
            device="cpu",
            include_values=include_values,
        ),
    )
    replay = birth.prefix_replay_record(history, controller_record)
    worker = {
        "artifact_type": "compact_responsibility_birth_iter2_phase_a_worker_v1",
        "status": "PASS",
        "training_root": training_root,
        "shuffle_root": shuffle_root,
        "split_root": None,
        "n_init_3d": 835,
        "n_opt_3d": 835,
        "m_init_i_2d": [3, 3],
        "m_opt_i_2d": [4, 4],
        "sum_m_opt_i_2d": 8,
        "alignment": [{"view": name, "focused": True} for name in view_names],
        "rgb_denial": {
            "passed": True,
            "source_rgb_open_attempts": 0,
            "forbidden_import_attempts": 0,
            "negative_control_denials": 3,
            "forbidden_modules_at_entry": [],
            "forbidden_modules_at_exit": [],
            "boundary_active_at_receipt": True,
        },
        "binding_receipts": {
            "entry": binding,
            "exit": binding,
            "exact_match": True,
        },
        "selection": controller_record["selection"],
        "replay": replay,
        "snapshot_35_pre": snapshot,
        "history": {
            "path": history_path.relative_to(tmp_path).as_posix(),
            "sha256": history_sha,
            "view_schedule_sha256": schedule_sha,
            "steps": history_steps,
            "controller": controller_record,
        },
        "raw_state_archive": archive,
    }

    def focused_replay(
        root: int,
        replay_history: dict[str, Any],
    ) -> dict[str, Any]:
        assert root == training_root
        replay_fields = (
            "step",
            "view_index",
            "view_name",
            "sample_seed",
            "xy_sha256",
            "active_sha256",
            "inside_fit_window_sha256",
            "proposal_component_ids_sha256",
            "proposal_density_sha256",
            "joint_density_sha256",
            "target_density_sha256",
            "importance_sha256",
            "attempts",
            "active_count",
            "null_count",
            "invalid_count",
            "uniform_attempt_count",
            "gaussian_attempt_count",
            "gaussian_accepted_count",
            "gaussian_rejected_count",
        )
        identities: list[dict[str, Any]] = []
        for step_index, view_index in enumerate(schedule):
            expected = _sample_replay_fields(
                training_root=root,
                step_index=step_index,
                view_index=view_index,
                view_name=view_names[view_index],
            )
            actual = replay_history["steps"][step_index]
            if any(actual[field] != expected[field] for field in replay_fields):
                raise birth.ProtocolInvalid("focused Phase-A proposal/sample replay changed")
            identities.append({field: expected[field] for field in replay_fields})
        record = {
            "view_schedule_sha256": schedule_sha,
            "step_replay_sha256": birth.canonical_hash(identities),
            "steps_replayed": 10,
            "teacher_digest": "focused-teacher",
            "proposal_digest": "focused-proposal",
            "camera_dimensions": [
                {"width": 64, "height": 48},
                {"width": 64, "height": 48},
            ],
            "m_init_i_2d": [3, 3],
            "m_opt_i_2d": [4, 4],
            "sum_m_opt_i_2d": 8,
            "alignment": [{"view": name, "focused": True} for name in view_names],
        }
        record["semantic_sha256"] = birth.canonical_hash(record)
        return record

    monkeypatch.setattr(birth, "_replay_phase_a_samples", focused_replay)
    monkeypatch.setattr(
        birth,
        "step_sample_seed",
        _focused_sample_seed,
    )
    return worker


def _rewrite_state_archive(
    worker: dict[str, Any],
    path: Path,
    mutation: Any,
) -> dict[str, Any]:
    mutated = copy.deepcopy(worker)
    source = birth.ROOT / worker["raw_state_archive"]["path"]
    payload = _npz_payload(source)
    metadata = json.loads(np.asarray(payload["metadata_utf8"], dtype=np.uint8).tobytes())
    mutation(payload, metadata)
    state = metadata["states"]["35_pre"]
    state_without_digest = dict(state)
    state_without_digest.pop("semantic_sha256", None)
    state["semantic_sha256"] = birth.canonical_hash(state_without_digest)
    metadata_without_digest = dict(metadata)
    metadata_without_digest.pop("semantic_sha256", None)
    metadata["semantic_sha256"] = birth.canonical_hash(metadata_without_digest)
    payload["metadata_utf8"] = np.frombuffer(birth.canonical_bytes(metadata), dtype=np.uint8)
    _write_npz(path, payload)
    mutated["raw_state_archive"] = {
        "path": str(path),
        "sha256": birth.sha256_file(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }
    return mutated


def _rewrite_recompute_history(
    worker: dict[str, Any],
    path: Path,
    mutation: Any,
) -> dict[str, Any]:
    mutated = copy.deepcopy(worker)
    history = birth.strict_json(birth.ROOT / worker["history"]["path"])
    mutation(history)
    path.unlink(missing_ok=True)
    history_sha = birth.exclusive_json(path, history)
    mutated["history"]["path"] = (
        path.relative_to(birth.ROOT).as_posix() if path.is_relative_to(birth.ROOT) else str(path)
    )
    mutated["history"]["sha256"] = history_sha
    mutated["history"]["steps"] = history["steps"]
    mutated["history"]["view_schedule_sha256"] = history["view_schedule_sha256"]
    return mutated


def _rewrite_controller_evidence(
    worker: dict[str, Any],
    mutation: Any,
) -> dict[str, Any]:
    mutated = copy.deepcopy(worker)
    controller = mutated["history"]["controller"]
    mutation(controller)
    digest = dict(controller)
    digest.pop("semantic_sha256", None)
    controller["semantic_sha256"] = birth.canonical_hash(digest)
    history_path = birth.ROOT / worker["history"]["path"]
    history = birth.strict_json(history_path)
    history["topology_control"] = controller
    history_path.unlink()
    mutated["history"]["sha256"] = birth.exclusive_json(history_path, history)
    mutated["replay"] = birth.prefix_replay_record(history, controller)
    return mutated


def _terminal_records(
    *,
    r_q: float,
    g_q: float,
    u_q: float,
    r_u: float = 1.0,
    g_u: float = 1.0,
    u_u: float = 1.0,
) -> dict[int, dict[str, dict[str, Any]]]:
    def arm(q: float, uniform: float) -> dict[str, Any]:
        return {
            "checkpoint_metrics": {
                key: {"J_Q": q, "J_U": uniform} for key in birth.RECOVERY_CHECKPOINTS
            }
        }

    return {
        root: {
            "R": arm(r_q, r_u),
            "G": arm(g_q, g_u),
            "U": arm(u_q, u_u),
        }
        for root in (1, 2, 3)
    }


def _ply_fixture(n: int = 867) -> Gaussians3D:
    return Gaussians3D(
        means=torch.linspace(-0.3, 0.3, n * 3).reshape(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        log_scales=torch.full((n, 3), -2.0),
        opacity=torch.linspace(0.2, 0.8, n),
        sh=torch.linspace(-0.2, 0.2, n * 3).reshape(n, 1, 3),
    )


def test_iter2_domain_seed_exact_encoding_and_disjoint_roots() -> None:
    root = birth.FOCUSED_EVALUATION_ROOTS[0]
    payload = (
        b"rtgs.compact-responsibility-birth.iter2.v1\0"
        b"evaluation_bank\0" + str(root).encode("ascii") + b"\0iter2-focused-view\0uniform"
    )
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    expected &= (1 << 63) - 1
    assert birth.domain_seed("evaluation_bank", root, "iter2-focused-view", "uniform") == expected
    assert birth.evaluation_bank_seed(root, "iter2-focused-view", "uniform") == expected
    assert birth.encode_atom("µ") == "µ".encode()
    for value in (-1, True, 1.0, b"x", None):
        with pytest.raises(TypeError):
            birth.encode_atom(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        birth.evaluation_bank_seed(root, "iter2-focused-view", "Uniform")

    frozen_sets = (
        birth.TRAIN_ROOTS,
        birth.EVALUATION_ROOTS,
        birth.SPLIT_ROOTS,
        birth.SHUFFLE_ROOTS,
        birth.FOCUSED_TRAIN_ROOTS,
        birth.FOCUSED_EVALUATION_ROOTS,
        birth.FOCUSED_SPLIT_ROOTS,
        birth.FOCUSED_SHUFFLE_ROOTS,
    )
    flattened = tuple(value for group in frozen_sets for value in group)
    assert len(flattened) == 23
    assert len(set(flattened)) == len(flattened)
    assert birth.DOMAIN_PREFIX == (b"rtgs.compact-responsibility-birth.iter2.v1\0")
    assert birth.EVALUATION_SEED_DOMAIN == "rtgs.compact-responsibility-birth.iter2.eval.v1"


def test_mode_root_rejection_and_official_root_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(birth, "_AUTHORIZED_PHASES", set())
    official_training = birth.TRAIN_ROOTS[0]
    focused_training = birth.FOCUSED_TRAIN_ROOTS[0]
    with pytest.raises(birth.ProtocolInvalid):
        birth._require_root_authorized(official_training, domain="training", focused=True)
    with pytest.raises(birth.ProtocolInvalid):
        birth._require_root_authorized(focused_training, domain="training", focused=False)
    with pytest.raises(birth.ProtocolInvalid):
        birth._require_root_authorized(official_training, domain="training", focused=False)
    with pytest.raises((birth.ProtocolInvalid, FileNotFoundError)):
        birth._require_root_authorized(
            official_training,
            domain="training",
            focused=False,
            marker_path=tmp_path / "absent.json",
            marker_artifact_type="iter2_test_attempt",
        )
    # A focused root needs no official attempt marker and can be inspected without
    # constructing a schedule or generator.
    birth._require_root_authorized(focused_training, domain="training", focused=True)

    for historical_root in birth.FAILED_ROOTS:
        with pytest.raises(birth.ProtocolInvalid):
            birth._require_root_authorized(historical_root, domain="training", focused=False)


def test_static_and_instrumented_dynamic_premarker_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(birth, "_AUTHORIZED_PHASES", set())
    static = birth._static_root_use_proof()
    assert static["passed"]
    assert static["direct_official_literal_calls"] == []
    assert static["official_literal_occurrences_outside_root_declarations"] == []

    dynamic = birth._dynamic_pre_marker_root_proof()
    assert dynamic["passed"]
    assert dynamic["rejected_before_mechanism"] == dynamic["gateways"]
    assert dynamic["mechanism_spy_calls"] == {
        "generator": 0,
        "schedule": 0,
        "sampler": 0,
        "trainer": 0,
        "surgery": 0,
    }
    assert not (birth.ROOT / "never-created.npz").exists()


def test_matched_strata_u_mapping_and_state_hashes_are_deterministic() -> None:
    first = birth.build_matched_selections(**_selection_fixture())
    second = birth.build_matched_selections(**_selection_fixture())
    assert first == second
    assert first["semantic_sha256"] == second["semantic_sha256"]
    eligible = set(first["eligible_rows"])
    seen: set[int] = set()
    for stratum in birth.STRATA:
        record = first["strata"][stratum]
        members = set(record["members"])
        assert len(members) == 10
        assert not seen & members
        seen |= members
        shuffle = first["shuffle"][stratum]
        assert sorted(shuffle["permutation"]) == list(range(10))
        assert len(shuffle["assignments"]) == 10
        assert len(shuffle["generator_state_before_sha256"]) == 64
        assert len(shuffle["generator_state_after_sha256"]) == 64
        assert len(shuffle["permutation_sha256"]) == 64
        assert len(shuffle["draw_sha256"]) == 64
        assert len(shuffle["assignment_sha256"]) == 64
        assert shuffle["draw_sha256"] == birth.canonical_hash(shuffle["permutation"])
        assert shuffle["assignment_sha256"] == birth.canonical_hash(shuffle["assignments"])
        source_ranks = [item["source_rank"] for item in shuffle["assignments"]]
        assert sorted(source_ranks) == list(range(len(record["members"])))
        assert len(source_ranks) == len(set(source_ranks))
        expected_u = [
            item["recipient_row"]
            for item in sorted(shuffle["assignments"], key=lambda item: item["source_rank"])
            if item["source_rank"] < 8
        ]
        assert shuffle["selected_rows"] == expected_u
        assert record["selected"]["U"] == expected_u
        for arm in birth.ARMS:
            assert len(record["selected"][arm]) == 8
    assert seen == eligible
    for arm in birth.ARMS:
        selected = first["selected_rows"][arm]
        assert len(selected) == len(set(selected)) == 32
        assert set(selected) <= eligible
        assert selected == sorted(selected)

    tied_fixture = _selection_fixture()
    tied_fixture["residual_score"] = torch.ones(40, dtype=torch.float64)
    tied_fixture["persistent_ids"] = torch.arange(39, -1, -1, dtype=torch.long)
    tied = birth.build_matched_selections(**tied_fixture)
    for stratum in birth.STRATA:
        record = tied["strata"][stratum]
        residual_ids = [
            int(tied_fixture["persistent_ids"][row]) for row in record["residual_order"]
        ]
        assert residual_ids == sorted(residual_ids)
        assignments = tied["shuffle"][stratum]["assignments"]
        assert sorted(item["source_rank"] for item in assignments) == list(range(len(assignments)))
        expected_u = [
            item["recipient_row"]
            for item in sorted(assignments, key=lambda item: item["source_rank"])
            if item["source_rank"] < 8
        ]
        assert record["selected"]["U"] == expected_u


def test_controller_literal_vjp_evidence_exact_five_visits_and_phase_a_no_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _params()
    optimizers = _optimizers(params)
    controller = birth.ResponsibilityBirthController(
        arm=None,
        split_root=None,
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[0],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=2,
        expected_split_count=2,
        quota_per_stratum=1,
    )

    def forbidden_split_seed(_root: int) -> int:
        raise AssertionError("selection-only Phase A derived a split seed")

    monkeypatch.setattr(birth, "split_seed", forbidden_split_seed)
    unchanged = _run_score_window(controller, params, optimizers)
    assert unchanged is params
    history = controller.history_record()
    assert history["surgery"] is None
    assert history["split_root"] is None
    assert "split_seed" not in history
    selection = history["selection"]
    assert selection["scores"]["view_step_count"] == [5, 5]
    assert selection["operator_counts"] == {arm: {"clone": 2, "split": 2} for arm in birth.ARMS}
    assert len(history["step_evidence"]) == 10
    assigned_numerator = 0.0
    assigned_denominator = 0.0
    for evidence in history["step_evidence"]:
        visible = evidence["visible_to_global"]
        assert visible == list(range(8))
        assert len(evidence["native_residual_visible_float32"]) == len(visible)
        assert len(evidence["native_support_visible_float32"]) == len(visible)
        assert len(evidence["native_residual_visible_sha256"]) == 64
        assert len(evidence["native_support_visible_sha256"]) == 64
        assert len(evidence["residual_global_divided_float64"]) == 8
        assert len(evidence["support_global_divided_float64"]) == 8
        assert evidence["native_residual_parent_sum_float32"] == pytest.approx(
            evidence["alpha_residual_sum_float32"], rel=2e-5, abs=2e-6
        )
        assert evidence["native_support_parent_sum_float32"] == pytest.approx(
            evidence["alpha_support_sum_float32"], rel=2e-5, abs=2e-6
        )
        assert set(evidence["six_parameter_gradients"]) == set(birth.GROUP_ORDER)
        assert len(evidence["six_parameter_gradients_sha256"]) == 64
        assert evidence["means2d_gradient"] is not None
        active = torch.tensor(evidence["active_float32"], dtype=torch.float32)
        error = torch.tensor(evidence["error_float32"], dtype=torch.float32)
        alpha = torch.tensor(evidence["alpha_float32"], dtype=torch.float32)
        native_residual = torch.tensor(
            evidence["native_residual_visible_float32"], dtype=torch.float32
        )
        native_support = torch.tensor(
            evidence["native_support_visible_float32"], dtype=torch.float32
        )
        assert birth.tensor_hash(active) == evidence["active_float32_sha256"]
        assert birth.tensor_hash(error) == evidence["error_float32_sha256"]
        assert birth.tensor_hash(alpha) == evidence["alpha_float32_sha256"]
        alpha_residual = (active * error * alpha).sum(dtype=torch.float32)
        alpha_support = (active * alpha).sum(dtype=torch.float32)
        denominator = (active * error).sum(dtype=torch.float32)
        assert float(alpha_residual) == evidence["alpha_residual_sum_float32"]
        assert float(alpha_support) == evidence["alpha_support_sum_float32"]
        assert float(denominator) == evidence["active_error_sum_float32"]
        torch.testing.assert_close(
            native_residual.sum(dtype=torch.float32),
            alpha_residual,
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            native_support.sum(dtype=torch.float32),
            alpha_support,
            atol=2e-6,
            rtol=2e-5,
        )
        assigned_numerator += float(native_residual.sum(dtype=torch.float32))
        assigned_denominator += float(denominator)
        corrupted = active.clone()
        corrupted[0] = 1.0 - corrupted[0]
        assert birth.tensor_hash(corrupted) != evidence["active_float32_sha256"]
    assigned = selection["assigned_residual"]
    assert assigned_numerator == (assigned["native_float32_numerator_reduced_before_division"])
    assert assigned_denominator == assigned["native_float32_denominator"]
    assert assigned_numerator / assigned_denominator == assigned["fraction"]
    pre_state = history["selection"]["pre_surgery_state"]
    assert pre_state["persistent_ids"]["values"] == list(range(8))
    assert len(pre_state["semantic_sha256"]) == 64


def test_controller_rejects_nonexact_score_window_view_counts() -> None:
    params = _params()
    optimizers = _optimizers(params)
    controller = birth.ResponsibilityBirthController(
        arm=None,
        split_root=None,
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[1],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=2,
        expected_split_count=2,
        quota_per_stratum=1,
    )
    with pytest.raises(birth.ProtocolInvalid, match="five|visit|count"):
        _run_score_window(
            controller,
            params,
            optimizers,
            visits=(0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
        )


def test_controller_rejects_mismatched_clone_split_allocation() -> None:
    params = _params()
    optimizers = _optimizers(params)
    controller = birth.ResponsibilityBirthController(
        arm="R",
        split_root=birth.FOCUSED_SPLIT_ROOTS[2],
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[1],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=1,
        expected_split_count=3,
        quota_per_stratum=1,
    )
    with pytest.raises(birth.ProtocolInvalid, match="quota|operator"):
        _run_score_window(controller, params, optimizers)


def test_phase_a_raw_recomputation_and_corruption_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _phase_a_recompute_worker_fixture(tmp_path, monkeypatch)
    summary = birth.recompute_phase_a_worker(worker)
    assert summary["training_root"] == birth.FOCUSED_TRAIN_ROOTS[0]
    assert summary["view_counts"] == [5, 5]
    assert summary["all_phase_a_gates_pass"]
    assert len(summary["semantic_sha256"]) == 64

    def reject(mutated: dict[str, Any], pattern: str = "") -> None:
        with pytest.raises(birth.ProtocolInvalid, match=pattern):
            birth.recompute_phase_a_worker(mutated)

    archive_path = birth.ROOT / worker["raw_state_archive"]["path"]
    original_archive_bytes = archive_path.read_bytes()

    def reject_archive(mutation: Any, pattern: str) -> None:
        archive_path.write_bytes(original_archive_bytes)
        mutated = _rewrite_state_archive(worker, archive_path, mutation)
        reject(mutated, pattern)
        archive_path.write_bytes(original_archive_bytes)

    def scale_mutation(payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        descriptor = metadata["states"]["35_pre"]["groups"]["scales"]["tensors"]["parameter"]
        values = payload[descriptor["key"]].copy()
        values.reshape(-1)[0] += np.float32(0.25)
        payload[descriptor["key"]] = values
        descriptor["sha256"] = birth.array_hash(values)

    reject_archive(scale_mutation, "state|selection|pre|snapshot")

    def id_mutation(payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        descriptor = metadata["states"]["35_pre"]["persistent_ids"]
        values = payload[descriptor["key"]].copy()
        values[[0, 1]] = values[[1, 0]]
        payload[descriptor["key"]] = values
        descriptor["sha256"] = birth.array_hash(values)

    reject_archive(id_mutation, "state|persistent|pre")

    def group_mutation(_payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        metadata["states"]["35_pre"]["groups"]["means"]["group"]["lr"] *= 2.0

    reject_archive(group_mutation, "state|optimizer|pre")

    def moment_mutation(payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        descriptor = metadata["states"]["35_pre"]["groups"]["means"]["tensors"]["exp_avg"]
        values = payload[descriptor["key"]].copy()
        values.reshape(-1)[0] += np.float32(0.125)
        payload[descriptor["key"]] = values
        descriptor["sha256"] = birth.array_hash(values)

    reject_archive(moment_mutation, "state|optimizer|pre")

    bad_prestate = copy.deepcopy(worker)
    bad_prestate["selection"]["pre_surgery_state"]["semantic_sha256"] = "0" * 64
    reject(bad_prestate, "state|semantic|selection|pre")

    history_path = birth.ROOT / worker["history"]["path"]
    original_history_bytes = history_path.read_bytes()

    def reject_controller(mutation: Any, pattern: str) -> None:
        history_path.write_bytes(original_history_bytes)
        reject(_rewrite_controller_evidence(worker, mutation), pattern)
        history_path.write_bytes(original_history_bytes)

    def permute_visible(controller: dict[str, Any]) -> None:
        evidence = controller["step_evidence"][1]
        permutation = torch.arange(835).roll(17)
        visible = torch.tensor(evidence["visible_to_global"], dtype=torch.long)[permutation]
        native_r = torch.tensor(
            evidence["native_residual_visible_float32"],
            dtype=torch.float32,
        )[permutation]
        native_s = torch.tensor(
            evidence["native_support_visible_float32"],
            dtype=torch.float32,
        )[permutation]
        means = torch.tensor(
            evidence["means2d_gradient"]["values"],
            dtype=torch.float32,
        )[permutation]
        gradient = torch.tensor(
            evidence["gradient_visible_float32"],
            dtype=torch.float32,
        )[permutation]
        evidence["visible_to_global"] = visible.tolist()
        evidence["visible_to_global_sha256"] = birth.tensor_hash(visible)
        evidence["native_residual_visible_float32"] = native_r.tolist()
        evidence["native_residual_visible_sha256"] = birth.tensor_hash(native_r)
        evidence["native_support_visible_float32"] = native_s.tolist()
        evidence["native_support_visible_sha256"] = birth.tensor_hash(native_s)
        evidence["means2d_gradient"]["values"] = means.tolist()
        evidence["means2d_gradient"]["sha256"] = birth.tensor_hash(means)
        evidence["gradient_visible_float32"] = gradient.tolist()
        evidence["gradient_visible_sha256"] = birth.tensor_hash(gradient)

    reject_controller(permute_visible, "VJP|parent-sum|identity")

    def active_mutation(controller: dict[str, Any]) -> None:
        empty_evidence = controller["step_evidence"][0]
        active = torch.tensor(empty_evidence["active_float32"], dtype=torch.float32)
        error = torch.tensor(empty_evidence["error_float32"], dtype=torch.float32)
        active[1] = 0.5
        error[1] = 0.0
        empty_evidence["active_float32"] = active.tolist()
        empty_evidence["active_float32_sha256"] = birth.tensor_hash(active)
        empty_evidence["error_float32"] = error.tolist()
        empty_evidence["error_float32_sha256"] = birth.tensor_hash(error)
        empty_evidence["point_loss_float32_sha256"] = birth.tensor_hash(error)

    reject_controller(active_mutation, "active|binary|bool")

    def attempts_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][0]["attempts"] = 7

    reject_controller(attempts_mutation, "attempt")

    def view_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][0]["view_index"] = 2

    reject_controller(view_mutation, "view")

    def duplicate_mutation(controller: dict[str, Any]) -> None:
        evidence = controller["step_evidence"][1]
        visible = torch.tensor(evidence["visible_to_global"], dtype=torch.long)
        visible[1] = visible[0]
        evidence["visible_to_global"] = visible.tolist()
        evidence["visible_to_global_sha256"] = birth.tensor_hash(visible)

    reject_controller(duplicate_mutation, "visible|duplicate")

    def range_mutation(controller: dict[str, Any]) -> None:
        evidence = controller["step_evidence"][1]
        visible = torch.tensor(evidence["visible_to_global"], dtype=torch.long)
        visible[0] = 835
        evidence["visible_to_global"] = visible.tolist()
        evidence["visible_to_global_sha256"] = birth.tensor_hash(visible)

    reject_controller(range_mutation, "visible|range")

    def scale_score_mutation(controller: dict[str, Any]) -> None:
        controller["selection"]["scores"]["scale_max"][0] += 0.25

    reject_controller(scale_score_mutation, "scale|raw-state|score")

    def screen_scale_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][1]["screen_gradient_scale"] = {
            "width": 65,
            "height": 48,
            "factor": 32.5,
        }

    reject_controller(screen_scale_mutation, "screen-gradient|scale")

    def empty_means_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][0]["means2d_gradient"] = {
            "dtype": "torch.float32",
            "shape": [0, 2],
            "device": "cpu",
            "finite": True,
            "sha256": birth.tensor_hash(torch.empty(0, 2)),
            "values": [],
        }

    reject_controller(empty_means_mutation, "empty-visible|means2d")

    def empty_gradient_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][0]["gradient_visible_float32"] = [0.0]

    reject_controller(empty_gradient_mutation, "gradient|shape|native G")

    def point_loss_hash_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][1]["point_loss_float32_sha256"] = "0" * 64

    reject_controller(point_loss_hash_mutation, "active|error|native")

    def six_gradient_digest_mutation(controller: dict[str, Any]) -> None:
        controller["step_evidence"][1]["six_parameter_gradients_sha256"] = "0" * 64

    reject_controller(six_gradient_digest_mutation, "six-gradient|digest")

    for nonfinite in (float("nan"), float("inf")):
        bad_assigned = copy.deepcopy(worker)
        bad_assigned["selection"]["assigned_residual"]["native_float32_denominator"] = nonfinite
        with pytest.raises(
            (birth.ProtocolInvalid, ValueError),
            match="controller|duplicate|semantic|assigned|Out of range",
        ):
            birth.recompute_phase_a_worker(bad_assigned)

    def schedule_mutation(history: dict[str, Any]) -> None:
        history["view_schedule"][0] = 1

    def reject_history(mutation: Any, pattern: str) -> None:
        history_path.write_bytes(original_history_bytes)
        reject(_rewrite_recompute_history(worker, history_path, mutation), pattern)
        history_path.write_bytes(original_history_bytes)

    reject_history(schedule_mutation, "schedule")

    def schedule_hash_mutation(history: dict[str, Any]) -> None:
        history["view_schedule_sha256"] = "0" * 64

    reject_history(schedule_hash_mutation, "schedule|hash")

    def view_name_mutation(history: dict[str, Any]) -> None:
        history["steps"][0]["view_name"] = "wrong-view"

    reject_history(view_name_mutation, "sample|replay|view|identity")

    def sample_seed_mutation(history: dict[str, Any]) -> None:
        history["steps"][0]["sample_seed"] += 1

    reject_history(sample_seed_mutation, "sample|replay|seed|identity")

    def active_bool_hash_mutation(history: dict[str, Any]) -> None:
        history["steps"][0]["active_sha256"] = "0" * 64

    reject_history(active_bool_hash_mutation, "sample|replay|active|hash")


def test_research_vjp_full_optimizer_update_hashes_match_noop_path() -> None:
    torch.manual_seed(_DEV_SEED)
    off = _full_update_evidence(collect_basis=False)
    torch.manual_seed(_DEV_SEED)
    on = _full_update_evidence(collect_basis=True)
    assert on["diagnostic_vjp"] is not None
    for field in (
        "forward",
        "gradients",
        "means2d_gradient",
        "updates",
        "optimizer_state",
        "rng_before",
        "rng_after",
    ):
        assert on[field] == off[field]


def test_selected_surgery_lineage_optimizer_and_accounting() -> None:
    params = _params()
    optimizers = _optimizers(params)
    _initialize_adam(params, optimizers, clock=10)
    old_params = {name: value.detach().clone() for name, value in params.items()}
    old_states = {
        name: {
            key: value.detach().clone()
            for key, value in optimizers[name].state[params[name]].items()
            if isinstance(value, torch.Tensor)
        }
        for name in params
    }
    old_groups = {
        name: {
            key: value for key, value in optimizers[name].param_groups[0].items() if key != "params"
        }
        for name in params
    }
    controller = birth.ResponsibilityBirthController(
        arm="R",
        split_root=birth.FOCUSED_SPLIT_ROOTS[0],
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[2],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=2,
        expected_split_count=2,
        quota_per_stratum=1,
    )
    changed = _run_score_window(controller, params, optimizers)
    history = controller.history_record()
    receipt = history["surgery"]["receipt"]
    assert receipt["n_before"] == 8
    assert receipt["n_after"] == 12
    assert receipt["net_growth"] == 4
    assert len(receipt["clone_parent_rows"]) == 2
    assert len(receipt["split_parent_rows"]) == 2
    assert len(receipt["survivor_old_rows"]) == 6
    assert len(receipt["newborns"]) == 6
    assert tuple(receipt["raw_split_shape"]) == (2, 3)
    assert len(receipt["generator_state_before_sha256"]) == 64
    assert len(receipt["generator_state_after_sha256"]) == 64
    assert len(receipt["raw_split_child0_sha256"]) == 64
    assert len(receipt["raw_split_child1_sha256"]) == 64
    assert controller.persistent_ids.tolist()[:6] == [
        index for index in range(8) if index not in set(receipt["split_parent_rows"])
    ]
    assert controller.persistent_ids.tolist()[6:] == list(range(8, 14))
    assert len(history["lineage"]) == 6
    assert {item["operator"] for item in history["lineage"]} == {"clone", "split"}

    survivor_rows = torch.tensor(receipt["survivor_old_rows"])
    for name, parameter in changed.items():
        assert parameter.shape[0] == 12
        assert bool(torch.isfinite(parameter).all())
        assert torch.equal(parameter[:6], old_params[name][survivor_rows])
        optimizer = optimizers[name]
        assert len(optimizer.param_groups[0]["params"]) == 1
        assert optimizer.param_groups[0]["params"][0] is parameter
        assert {
            key: value for key, value in optimizer.param_groups[0].items() if key != "params"
        } == old_groups[name]
        state = optimizer.state[parameter]
        assert int(state["step"].item()) == 10
        assert torch.equal(state["exp_avg"][:6], old_states[name]["exp_avg"][survivor_rows])
        assert torch.equal(
            state["exp_avg_sq"][:6],
            old_states[name]["exp_avg_sq"][survivor_rows],
        )
        assert torch.equal(state["exp_avg"][6:], torch.zeros_like(state["exp_avg"][6:]))
        assert torch.equal(state["exp_avg_sq"][6:], torch.zeros_like(state["exp_avg_sq"][6:]))
    accounting = history["surgery"]["accounting"]
    assert accounting == {
        "n_before": 8,
        "removed_split_parents": 2,
        "survivors": 6,
        "clone_children": 2,
        "split_child_0": 2,
        "split_child_1": 2,
        "newborn_rows": 6,
        "net_growth": 4,
        "n_after": 12,
        "pruned": 0,
    }


def test_raw_state_archive_persists_values_and_strictly_recomputes_hashes(
    tmp_path: Path,
) -> None:
    params = _params()
    optimizers = _optimizers(params)
    _initialize_adam(params, optimizers, clock=10)
    controller = birth.ResponsibilityBirthController(
        arm="R",
        split_root=birth.FOCUSED_SPLIT_ROOTS[1],
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[0],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=2,
        expected_split_count=2,
        quota_per_stratum=1,
    )
    label0_means = params["means"].detach().clone()
    label0_means_moment = optimizers["means"].state[params["means"]]["exp_avg"].detach().clone()
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=2,
        attempts_per_step=6,
    )
    # Captured CPU arrays must own their bytes. Mutating the live tensors after
    # label 0 is captured must not retroactively rewrite the archived snapshot.
    with torch.no_grad():
        params["means"].add_(0.125)
        optimizers["means"].state[params["means"]]["exp_avg"].add_(0.25)
    changed: dict[str, torch.Tensor] = params
    for step, view_index in enumerate((0, 1) * 5, start=1):
        _artificial_score_step(
            controller,
            params,
            step=step,
            view_index=view_index,
            n=params["means"].shape[0],
        )
        changed = controller.after_step(
            step=step,
            params=changed,
            optimizers=optimizers,
            snapshot=birth._params_to_gaussians(changed).detach(),
        )
    archive_path = tmp_path / "raw_states.npz"
    receipt = birth.save_state_archive(archive_path, controller)
    arrays, metadata = birth.load_state_archive(archive_path)
    assert receipt["metadata"] == metadata
    assert metadata["schema"] == ("rtgs.compact_responsibility_birth_iter2.states.v1")
    assert metadata["labels"] == ["0", "35_pre", "35_post"]
    assert len(metadata["semantic_sha256"]) == 64

    initial = metadata["states"]["0"]["groups"]["means"]["tensors"]
    assert np.array_equal(
        arrays[initial["parameter"]["key"]],
        label0_means.cpu().contiguous().numpy(),
    )
    assert np.array_equal(
        arrays[initial["exp_avg"]["key"]],
        label0_means_moment.cpu().contiguous().numpy(),
    )
    assert not np.array_equal(
        arrays[initial["parameter"]["key"]],
        params["means"].detach().cpu().contiguous().numpy(),
    )

    post = metadata["states"]["35_post"]
    assert post["persistent_ids"]["shape"] == [12]
    assert arrays[post["persistent_ids"]["key"]].tolist() == (controller.persistent_ids.tolist())
    for name in birth.GROUP_ORDER:
        group = post["groups"][name]
        assert group["step"] == 10
        assert set(group["tensors"]) == {"parameter", "exp_avg", "exp_avg_sq"}
        parameter = arrays[group["tensors"]["parameter"]["key"]]
        exp_avg = arrays[group["tensors"]["exp_avg"]["key"]]
        exp_avg_sq = arrays[group["tensors"]["exp_avg_sq"]["key"]]
        assert np.array_equal(parameter, changed[name].detach().cpu().contiguous().numpy())
        state = optimizers[name].state[changed[name]]
        assert np.array_equal(exp_avg, state["exp_avg"].detach().cpu().contiguous().numpy())
        assert np.array_equal(exp_avg_sq, state["exp_avg_sq"].detach().cpu().contiguous().numpy())

    corrupted = _npz_payload(archive_path)
    tensor_key = post["groups"]["means"]["tensors"]["exp_avg"]["key"]
    corrupted[tensor_key] = corrupted[tensor_key].copy()
    corrupted[tensor_key].reshape(-1)[0] += 1.0
    corrupted_path = tmp_path / "corrupted_raw_states.npz"
    _write_npz(corrupted_path, corrupted)
    with pytest.raises(birth.ProtocolInvalid, match="changed"):
        birth.load_state_archive(corrupted_path)

    unexpected = _npz_payload(archive_path)
    unexpected["unexpected"] = np.zeros(1, dtype=np.float32)
    unexpected_path = tmp_path / "extra_raw_state.npz"
    _write_npz(unexpected_path, unexpected)
    with pytest.raises(birth.ProtocolInvalid, match="unexpected"):
        birth.load_state_archive(unexpected_path)


def test_complexity_categories_exactly_account_for_toy_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _params()
    optimizers = _optimizers(params)
    controller = birth.ResponsibilityBirthController(
        arm=None,
        split_root=None,
        shuffle_root=birth.FOCUSED_SHUFFLE_ROOTS[0],
        score_steps=10,
        expected_view_visits=5,
        expected_n_before=8,
        expected_clone_count=2,
        expected_split_count=2,
        quota_per_stratum=1,
    )
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=2,
        attempts_per_step=6,
    )
    current: dict[str, torch.Tensor] = params
    for step, view_index in enumerate((0, 1) * 5, start=1):
        _artificial_score_step(
            controller,
            params,
            step=step,
            view_index=view_index,
            n=8,
            empty_visible=step == 1,
        )
        current = controller.after_step(
            step=step,
            params=current,
            optimizers=optimizers,
            snapshot=birth._params_to_gaussians(current).detach(),
        )
    controller_record = controller.history_record()
    assert controller_record["step_evidence"][0]["means2d_gradient"] is None
    inputs, products = _tiny_inputs()
    steps = [
        {
            "visible_count": 0 if index == 0 else 8,
            "rendered_point_gaussian_pairs": 0 if index == 0 else 48,
            "teacher_query_attempts": 6,
            "student_query_attempts": 6,
            "peak_rss_bytes": 1000 + index,
        }
        for index in range(10)
    ]
    history = {
        "steps": steps,
        "preflight": {"views": [{"components": inputs.observations[0].n}]},
        "proposal_preflight": {"views": [{"components": products[0].n}]},
        "index_diagnostics": {"fixture": "teacher"},
        "proposal_index_diagnostics": {"fixture": "proposal"},
        "n_init_3d": 8,
        "n_opt_3d": 8,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = birth._complexity_accounting(
        history,
        controller_record,
        inputs=inputs,
        products=products,
    )

    def tensor_bytes(value: object) -> int:
        if not isinstance(value, torch.Tensor):
            return 0
        return int(value.numel() * value.element_size())

    field_names = (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "amplitudes",
        "color_grads",
        "filter_variance",
    )
    expected_teacher = sum(
        tensor_bytes(getattr(inputs.observations[0], name, None)) for name in field_names
    )
    expected_proposal = sum(tensor_bytes(getattr(products[0], name, None)) for name in field_names)
    expected_camera = sum(tensor_bytes(value) for value in vars(inputs.cameras[0]).values())

    dtype_bytes = {
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int64": 8,
        "torch.bool": 1,
    }

    def receipt_bytes(receipt: dict[str, Any]) -> int:
        return math.prod(receipt["shape"]) * dtype_bytes[receipt["dtype"]]

    checkpoints = controller_record["state_checkpoints"]
    final = checkpoints["35_pre"]
    expected_parameters = sum(
        receipt_bytes(group["parameter"]) for group in final["optimizers"]["groups"].values()
    )
    expected_moments = sum(
        receipt_bytes(moment)
        for group in final["optimizers"]["groups"].values()
        for moment in group["moments"].values()
    )
    expected_checkpoints = sum(
        receipt_bytes(state["persistent_ids"])
        + sum(
            receipt_bytes(group["parameter"])
            + sum(receipt_bytes(moment) for moment in group["moments"].values())
            for group in state["optimizers"]["groups"].values()
        )
        for state in checkpoints.values()
    )
    expected_nonempty_score = (
        8 * 8 + 8 * 4 + 8 * 4 + 6 * 4 + 6 * 4 + 6 * 4 + 8 * 8 + 8 * 8 + 8 * 4 + 8 * 2 * 4
    )
    expected_empty_score = 6 * 4 + 6 * 4 + 6 * 4 + 8 * 8 + 8 * 8
    expected_score = 9 * expected_nonempty_score + expected_empty_score
    expected_categories = {
        "teacher_tensor_bytes": expected_teacher,
        "proposal_tensor_bytes": expected_proposal,
        "camera_tensor_bytes": expected_camera,
        "current_parameter_bytes": expected_parameters,
        "current_moment_bytes": expected_moments,
        "score_evidence_bytes": expected_score,
        "raw_state_checkpoint_tensor_bytes": expected_checkpoints,
    }
    assert record["bytes_by_category"] == expected_categories
    assert record["sum_reported_category_bytes"] == sum(expected_categories.values())
    assert record["vjp_native_output_bytes"] == 9 * 8 * 2 * 4
    assert record["cuda"] == {
        "available": False,
        "max_memory_allocated": 0,
        "max_memory_reserved": 0,
    }


def test_evaluation_bank_exact_schema_and_strict_loader_rejection(
    tmp_path: Path,
) -> None:
    inputs, product = _tiny_inputs()
    bank_path = tmp_path / "focused_bank.npz"
    record = birth.generate_evaluation_bank(
        evaluation_root=birth.FOCUSED_EVALUATION_ROOTS[0],
        teachers=inputs,
        product_fields=product,
        path=bank_path,
        focused=True,
        attempts=32,
    )
    loaded, metadata = birth.load_evaluation_bank(
        bank_path,
        expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
        attempts=32,
        expected_views=("iter2-focused-view",),
    )
    validated, validated_metadata = birth.load_evaluation_bank(
        bank_path,
        expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
        attempts=32,
        expected_views=("iter2-focused-view",),
        product_fields=product,
    )
    assert record["metadata"] == metadata
    assert validated_metadata == metadata
    assert all(
        np.array_equal(validated[0][measure][name], loaded[0][measure][name])
        for measure in ("uniform", "proposal")
        for name in _BANK_FIELDS
    )
    assert len(loaded) == 1
    assert set(loaded[0]) == {"uniform", "proposal"}
    for measure in ("uniform", "proposal"):
        assert set(loaded[0][measure]) == _BANK_FIELDS
        descriptor_fields = set(metadata["views"][0]["banks"][measure]["tensors"])
        assert descriptor_fields == _BANK_FIELDS
        assert loaded[0][measure]["target_density"].shape == (32,)
        assert loaded[0][measure]["importance"].shape == (32,)

    payload = _npz_payload(bank_path)
    missing_path = tmp_path / "missing_target_density.npz"
    del payload["v0_uniform_target_density"]
    _write_npz(missing_path, payload)
    with pytest.raises(birth.ProtocolInvalid):
        birth.load_evaluation_bank(
            missing_path,
            expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
            attempts=32,
            expected_views=("iter2-focused-view",),
        )

    def rewrite_consistently(
        path: Path,
        mutation: Any,
    ) -> None:
        payload = _npz_payload(bank_path)
        rewritten_metadata = json.loads(
            np.asarray(payload["metadata_utf8"], dtype=np.uint8).tobytes()
        )
        mutation(payload, rewritten_metadata)
        for measure in ("uniform", "proposal"):
            measure_record = rewritten_metadata["views"][0]["banks"][measure]
            for name in birth.BANK_TENSOR_NAMES:
                value = payload[f"v0_{measure}_{name}"]
                measure_record["tensors"][name] = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": birth.array_hash(value),
                }
            measure_record["draw_sha256"] = birth.canonical_hash(
                {
                    name: measure_record["tensors"][name]["sha256"]
                    for name in birth.BANK_TENSOR_NAMES
                }
            )
            active_count = int(payload[f"v0_{measure}_active"].sum())
            measure_record["active_count"] = active_count
            measure_record["null_count"] = 32 - active_count
            measure_record["active_fraction"] = active_count / 32
        digest = dict(rewritten_metadata)
        digest.pop("semantic_sha256", None)
        rewritten_metadata["semantic_sha256"] = birth.canonical_hash(digest)
        payload["metadata_utf8"] = np.frombuffer(
            birth.canonical_bytes(rewritten_metadata),
            dtype=np.uint8,
        )
        _write_npz(path, payload)

    negative_density_path = tmp_path / "negative_proposal_density.npz"

    def make_density_negative(
        payload: dict[str, np.ndarray],
        _metadata: dict[str, Any],
    ) -> None:
        density = payload["v0_proposal_proposal_density"].copy()
        density[0] = np.float32(-1.0)
        payload["v0_proposal_proposal_density"] = density

    rewrite_consistently(negative_density_path, make_density_negative)
    with pytest.raises(birth.ProtocolInvalid, match="negative"):
        birth.load_evaluation_bank(
            negative_density_path,
            expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
            attempts=32,
            expected_views=("iter2-focused-view",),
            product_fields=product,
        )

    malformed_generator_path = tmp_path / "malformed_generator_state.npz"

    def malform_generator_state(
        _payload: dict[str, np.ndarray],
        rewritten_metadata: dict[str, Any],
    ) -> None:
        rewritten_metadata["views"][0]["banks"]["proposal"]["generator_state_before_sha256"] = (
            "not-a-sha256"
        )

    rewrite_consistently(malformed_generator_path, malform_generator_state)
    with pytest.raises(birth.ProtocolInvalid, match="seed|binding"):
        birth.load_evaluation_bank(
            malformed_generator_path,
            expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
            attempts=32,
            expected_views=("iter2-focused-view",),
            product_fields=product,
        )

    payload = _npz_payload(bank_path)
    payload["unexpected"] = np.zeros(1, dtype=np.float32)
    extra_path = tmp_path / "extra_array.npz"
    _write_npz(extra_path, payload)
    with pytest.raises(birth.ProtocolInvalid):
        birth.load_evaluation_bank(
            extra_path,
            expected_root=birth.FOCUSED_EVALUATION_ROOTS[0],
            attempts=32,
            expected_views=("iter2-focused-view",),
        )


def test_phase_b_checkpoint_metrics_are_exactly_recomputed_and_parent_wires_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = {
        "0": SimpleNamespace(semantic="initial"),
        "140": SimpleNamespace(semantic="final"),
    }
    expected_metric = {
        "J_U": 0.25,
        "J_Q": 0.5,
        "worst_view_J_Q": 0.5,
        "per_view": [
            {
                "view_index": 0,
                "view_name": "iter2-focused-view",
                "J_U": 0.25,
                "J_Q": 0.5,
            }
        ],
    }
    stored = {label: copy.deepcopy(expected_metric) for label in snapshots}
    evaluated: list[str] = []

    monkeypatch.setattr(
        birth.factorial,
        "gaussians_hash",
        lambda snapshot: snapshot.semantic,
    )

    def exact_evaluation(
        snapshot: SimpleNamespace,
        _inputs: object,
        _banks: object,
    ) -> dict[str, Any]:
        evaluated.append(snapshot.semantic)
        return copy.deepcopy(expected_metric)

    monkeypatch.setattr(birth, "evaluate_snapshot", exact_evaluation)
    receipts = birth._validate_checkpoint_metric_recomputation(
        snapshots,
        stored,
        object(),  # type: ignore[arg-type]
        [],  # type: ignore[arg-type]
    )
    assert evaluated == ["initial", "final"]
    assert receipts == {label: birth.canonical_hash(expected_metric) for label in snapshots}

    corrupted = copy.deepcopy(stored)
    corrupted["140"]["J_Q"] = 0.5000000000000001
    with pytest.raises(birth.ProtocolInvalid, match="metric recomputation"):
        birth._validate_checkpoint_metric_recomputation(
            snapshots,
            corrupted,
            object(),  # type: ignore[arg-type]
            [],  # type: ignore[arg-type]
        )

    source = Path(birth.__file__).read_text(encoding="utf-8")
    parent_validator = source[
        source.index("def _validate_phase_b_worker_artifacts(") : source.index(
            "def _phase_b_secondary_diagnostics("
        )
    ]
    assert "metric_receipts = _validate_checkpoint_metric_recomputation(" in parent_validator
    assert 'record["checkpoint_metrics"]' in parent_validator


def test_exclusive_marker_and_strict_reread_are_fail_closed(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "attempt.json"
    payload = {
        "artifact_type": "iter2_focused_test_attempt",
        "status": "PASS",
        "nonce": "focused-only",
    }
    digest = birth.exclusive_json(marker, payload)
    reread, reread_digest = birth._strict_reread_attempt(
        marker,
        artifact_type="iter2_focused_test_attempt",
        expected=payload,
    )
    assert reread == payload
    assert reread_digest == digest
    assert birth.strict_json(marker) == payload
    with pytest.raises(FileExistsError):
        birth.exclusive_json(marker, payload)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(birth.ProtocolInvalid):
        birth.strict_json(noncanonical)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"artifact_type":"x","artifact_type":"x"}\n', encoding="utf-8")
    with pytest.raises(birth.ProtocolInvalid):
        birth.strict_json(duplicate)
    with pytest.raises(birth.ProtocolInvalid):
        birth._strict_reread_attempt(
            marker,
            artifact_type="wrong",
            expected=payload,
        )


def test_worker_output_paths_reject_before_binding_or_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_sha = "marker"
    seal_sha = "seal"
    result_sha = "phase-a-result"
    audit_sha = "phase-a-audit"
    monkeypatch.setattr(
        birth,
        "_authorize_marker",
        lambda *_args, **_kwargs: ({}, marker_sha),
    )

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path)
        if resolved == birth.SEAL:
            return seal_sha
        if resolved == birth.PHASE_A_RESULT:
            return result_sha
        if resolved == birth.PHASE_A_AUDIT:
            return audit_sha
        raise AssertionError(f"unexpected pre-path SHA request: {resolved}")

    monkeypatch.setattr(birth, "sha256_file", fake_sha)
    binding_calls = 0

    def binding_must_not_run() -> dict[str, Any]:
        nonlocal binding_calls
        binding_calls += 1
        raise AssertionError("binding ran before worker output-path validation")

    monkeypatch.setattr(birth, "_verified_binding_receipt", binding_must_not_run)
    wrong = tmp_path / "wrong-worker-output.json"
    with pytest.raises(birth.ProtocolInvalid, match="output path"):
        birth._phase_a_worker_inside_guard(
            guard=object(),  # type: ignore[arg-type]
            training_root=_DEV_SEED,
            shuffle_root_value=birth.FOCUSED_SHUFFLE_ROOTS[0],
            marker_sha256=marker_sha,
            seal_sha256=seal_sha,
            output_path=wrong,
        )
    with pytest.raises(birth.ProtocolInvalid, match="output path"):
        birth._phase_b_worker_inside_guard(
            guard=object(),  # type: ignore[arg-type]
            training_root=_DEV_SEED,
            evaluation_root=birth.FOCUSED_EVALUATION_ROOTS[0],
            split_root_value=birth.FOCUSED_SPLIT_ROOTS[0],
            shuffle_root_value=birth.FOCUSED_SHUFFLE_ROOTS[0],
            marker_sha256=marker_sha,
            seal_sha256=seal_sha,
            phase_a_result_sha256=result_sha,
            phase_a_audit_sha256=audit_sha,
            output_path=wrong,
        )
    assert binding_calls == 0
    assert not wrong.exists()


def test_nonzero_worker_record_survives_in_actual_phase_a_terminal_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal = tmp_path / "seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(birth, "ROOT", tmp_path)
    monkeypatch.setattr(birth, "SEAL", seal)
    monkeypatch.setattr(birth, "PHASE_A_ATTEMPT", tmp_path / "phase_a_attempt.json")
    monkeypatch.setattr(birth, "PHASE_A_RESULT", tmp_path / "phase_a_result.json")
    monkeypatch.setattr(birth, "PHASE_A_AUDIT", tmp_path / "phase_a_audit.json")
    monkeypatch.setattr(birth, "PHASE_B_ATTEMPT", tmp_path / "phase_b_attempt.json")
    monkeypatch.setattr(birth, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(birth, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(birth, "TRAIN_ROOTS", (birth.FOCUSED_TRAIN_ROOTS[0],))
    monkeypatch.setattr(birth, "SHUFFLE_ROOTS", (birth.FOCUSED_SHUFFLE_ROOTS[0],))
    monkeypatch.setattr(birth, "_AUTHORIZED_PHASES", set())
    monkeypatch.setattr(
        birth,
        "verify_seal",
        lambda: {"bindings_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        birth,
        "_verified_binding_receipt",
        lambda: {"focused": "binding"},
    )
    monkeypatch.setattr(
        birth,
        "_live_guard_receipt",
        lambda _guard: {
            "passed": True,
            "boundary_active_at_receipt": True,
        },
    )

    def failed_worker(command: list[str]) -> dict[str, Any]:
        return {
            "command": list(command),
            "returncode": 17,
            "stdout_bytes": 7,
            "stdout_sha256": hashlib.sha256(b"partial").hexdigest(),
            "stdout_tail": "partial",
            "stderr_bytes": 4,
            "stderr_sha256": hashlib.sha256(b"boom").hexdigest(),
            "stderr_tail": "boom",
            "elapsed_seconds": 1.25,
        }

    monkeypatch.setattr(birth, "_run_worker", failed_worker)
    result = birth._run_phase_a_inside_guard(object())  # type: ignore[arg-type]
    assert result["status"] == "FAIL"
    assert result["phase_a_decision"] == "STOP_PHASE_B"
    assert result["commands"][0]["returncode"] == 17
    assert result["commands"][0]["stderr_tail"] == "boom"
    assert birth.strict_json(birth.PHASE_A_RESULT) == result


def test_phase_a_audit_must_equal_independent_raw_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration = tmp_path / "prereg.md"
    seal = tmp_path / "seal.json"
    result_path = tmp_path / "phase_a_result.json"
    archive_path = tmp_path / "states.npz"
    preregistration.write_text("focused preregistration fixture\n", encoding="utf-8")
    seal.write_text("{}\n", encoding="utf-8")
    archive_path.write_bytes(b"focused-state-fixture")
    marker_sha = "c" * 64
    monkeypatch.setattr(birth, "PREREGISTRATION", preregistration)
    monkeypatch.setattr(birth, "SEAL", seal)
    monkeypatch.setattr(birth, "PHASE_A_RESULT", result_path)
    seal_sha = birth.sha256_file(seal)
    bindings_sha = "d" * 64
    marker = {
        "artifact_type": "compact_responsibility_birth_iter2_phase_a_attempt_v1",
        "timestamp_utc": "2026-07-17T00:00:00Z",
        "seal_sha256": seal_sha,
        "bindings_sha256": bindings_sha,
    }
    monkeypatch.setattr(
        birth,
        "_authorize_marker",
        lambda *_args, **_kwargs: (marker, marker_sha),
    )
    monkeypatch.setattr(
        birth,
        "verify_seal",
        lambda: {"bindings_sha256": bindings_sha},
    )
    monkeypatch.setattr(
        birth,
        "_validate_phase_a_parent_receipts",
        lambda *_args, **_kwargs: None,
    )

    metadata = {
        "schema": "rtgs.compact-responsibility-birth.iter2.audit-test.v1",
        "labels": ["0", "35_pre"],
    }

    def fake_load_state_archive(
        path: Path,
        **_kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert path.resolve() == archive_path.resolve()
        return {}, metadata

    monkeypatch.setattr(birth, "load_state_archive", fake_load_state_archive)
    workers: list[dict[str, Any]] = []
    for training_root in birth.TRAIN_ROOTS:
        replay: dict[str, Any] = {"training_root": training_root}
        replay["semantic_sha256"] = birth.canonical_hash(replay)
        workers.append(
            {
                "status": "PASS",
                "training_root": training_root,
                "split_root": None,
                "selection": {"all_phase_a_gates_pass": True},
                "replay": replay,
                "raw_state_archive": {
                    "path": str(archive_path),
                    "sha256": birth.sha256_file(archive_path),
                    "metadata": metadata,
                },
            }
        )
    result = {
        "artifact_type": "compact_responsibility_birth_iter2_phase_a_result_v1",
        "timestamp_utc": "2026-07-17T00:01:00Z",
        "status": "PASS",
        "phase_a_decision": "AUTHORIZE_AUDIT",
        "seal_sha256": seal_sha,
        "phase_a_attempt_sha256": marker_sha,
        "all_phase_a_gates_pass": True,
        "workers": workers,
        "gates": {},
        "commands": [],
        "artifact_validation": {},
        "parent_binding_receipts": {},
        "parent_rgb_denial": {},
    }
    birth.exclusive_json(result_path, result)
    recomputed = {
        "schema": "rtgs.compact_responsibility_birth_iter2.phase_a_recompute.v1",
        "workers": [
            {
                "training_root": root,
                "all_phase_a_gates_pass": True,
                "raw_evidence_sha256": f"{root:064x}",
            }
            for root in birth.TRAIN_ROOTS
        ],
        "all_phase_a_gates_pass": True,
    }
    recomputed["semantic_sha256"] = birth.canonical_hash(recomputed)
    monkeypatch.setattr(birth, "recompute_phase_a_result", lambda _result: recomputed)
    bindings = {
        "preregistration_sha256": birth.sha256_file(preregistration),
        "seal_sha256": birth.sha256_file(seal),
        "phase_a_attempt_sha256": marker_sha,
        "phase_a_result_sha256": birth.sha256_file(result_path),
    }

    def write_audit(path: Path, recomputation: object) -> None:
        payload = {
            "artifact_type": "compact_responsibility_birth_phase_a_audit_v1",
            "verdict": "PASS",
            "unresolved_findings": [],
            "bindings": bindings,
            "auditor": {
                "identity": "focused independent fixture",
                "provenance": "test-only recomputation",
            },
        }
        if recomputation is not None:
            payload["recomputed"] = recomputation
        birth.exclusive_json(path, payload)

    missing_path = tmp_path / "audit_missing_recompute.json"
    write_audit(missing_path, None)
    monkeypatch.setattr(birth, "PHASE_A_AUDIT", missing_path)
    with pytest.raises(birth.ProtocolInvalid, match="audit"):
        birth.verify_phase_a_authorization(result_path, missing_path)

    superficial_path = tmp_path / "audit_superficial_recompute.json"
    write_audit(superficial_path, {"all_phase_a_gates_pass": True})
    monkeypatch.setattr(birth, "PHASE_A_AUDIT", superficial_path)
    with pytest.raises(birth.ProtocolInvalid, match="audit"):
        birth.verify_phase_a_authorization(result_path, superficial_path)

    exact_path = tmp_path / "audit_exact_recompute.json"
    write_audit(exact_path, recomputed)
    monkeypatch.setattr(birth, "PHASE_A_AUDIT", exact_path)
    verified_result, verified_audit = birth.verify_phase_a_authorization(
        result_path,
        exact_path,
    )
    assert verified_result == result
    assert verified_audit["recomputed"] == recomputed


def test_rgb_guard_and_verified_binding_receipt_are_live_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = birth.RGBAccessGuard()
    # Earlier focused fixtures may import a forbidden loader while constructing
    # synthetic inputs.  Remove those cached modules for this isolated process
    # boundary check; pytest restores them after the test.
    for module_name in guard._loaded_forbidden_modules():
        monkeypatch.delitem(sys.modules, module_name)
    with guard:
        denial = birth._live_guard_receipt(guard)
    assert denial == {
        "source_rgb_open_attempts": 0,
        "forbidden_import_attempts": 0,
        "negative_control_denials": 3,
        "forbidden_modules_at_entry": [],
        "forbidden_modules_at_exit": [],
        "passed": True,
        "boundary_active_at_receipt": True,
    }

    binding = {
        "source_aggregate_sha256": "1" * 64,
        "inputs": {"fixture": "inputs"},
        "runtime": {"fixture": "runtime"},
        "git": {"fixture": "git"},
    }
    sealed = {"bindings": binding}
    monkeypatch.setattr(birth, "verify_seal", lambda: sealed)
    monkeypatch.setattr(birth, "_binding_state", lambda: binding)
    monkeypatch.setattr(birth, "sha256_file", lambda path: "2" * 64)
    receipt = birth._verified_binding_receipt()
    assert receipt == {
        "seal_sha256": "2" * 64,
        "binding_sha256": birth.canonical_hash(binding),
        "source_aggregate_sha256": "1" * 64,
        "input_binding_sha256": birth.canonical_hash(binding["inputs"]),
        "runtime_binding_sha256": birth.canonical_hash(binding["runtime"]),
        "git_binding_sha256": birth.canonical_hash(binding["git"]),
    }

    drifted = dict(binding)
    drifted["runtime"] = {"fixture": "drifted"}
    monkeypatch.setattr(birth, "_binding_state", lambda: drifted)
    with pytest.raises(birth.ProtocolInvalid, match="binding differs"):
        birth._verified_binding_receipt()


def test_phase_b_callbacks_capture_arm_and_compare_full_common_state() -> None:
    module_source = Path(birth.__file__).read_text(encoding="utf-8")
    source = module_source.split("def _phase_b_worker_inside_guard", 1)[1].split(
        "\ndef _phase_b_worker(",
        1,
    )[0]
    assert source.count("current_arm_index: int = arm_index") == 2
    assert 'holder["controller"]._state_checkpoints["0"]["semantic_sha256"]' in source
    assert 'state_hash != common_state_hashes["0"]' in source
    assert (
        'selection["pre_surgery_state"]["semantic_sha256"]\n'
        '                        != common_state_hashes["35_pre"]'
    ) in source


def test_iter2_runtime_binding_has_no_unbound_local_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_origins = birth.expected_rtgs_module_origins()
    loaded_origins = {
        name: origin
        for name, origin in birth.loaded_rtgs_module_origins().items()
        if expected_origins.get(name) == origin
    }
    bound = {
        path.relative_to(birth.ROOT).as_posix()
        for path in birth._source_paths(include_implementation_review=False)
    }
    loaded_sources = tuple(path for path in birth.loaded_local_sources() if path in bound)
    monkeypatch.setattr(birth, "loaded_rtgs_module_origins", lambda: loaded_origins)
    monkeypatch.setattr(birth, "loaded_local_sources", lambda: loaded_sources)
    assert birth.module_origin_violations() == ()
    assert birth.unbound_loaded_local_sources() == ()
    assert set(birth.loaded_local_sources()) <= bound

    def old_runtime_must_not_run() -> dict[str, Any]:
        raise AssertionError("iter2 delegated its runtime receipt to the old harness")

    monkeypatch.setattr(birth.factorial, "runtime_binding", old_runtime_must_not_run)
    monkeypatch.setattr(
        birth.factorial,
        "_torch_runtime_import_path_binding",
        lambda: {
            "normalized_sys_path": ["/focused/runtime"],
            "torch_generated_path": "/focused/torch",
        },
    )
    monkeypatch.setenv("LD_PRELOAD", str(birth.PRELOAD))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA GeForce RTX 3050")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(
            total_memory=4_294_967_296,
            multi_processor_count=20,
            uuid="focused-gpu-uuid",
            pci_bus_id=1,
        ),
    )
    monkeypatch.setattr(birth.importlib.metadata, "version", lambda _package: "1.5.3")
    monkeypatch.setattr(
        birth.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="focused-driver, 00000000:01:00.0, focused-gpu-uuid\n",
        ),
    )
    frozen_runtime = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "gsplat": "1.5.3",
        "cuda_device": "NVIDIA GeForce RTX 3050",
        "cuda_capability": [8, 6],
        "cuda_total_memory": 4_294_967_296,
        "cuda_multiprocessor_count": 20,
        "cuda_uuid": "focused-gpu-uuid",
        "cuda_pci_bus_id": 1,
        "nvidia_smi_driver_device": ("focused-driver, 00000000:01:00.0, focused-gpu-uuid"),
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "module_origins": birth.expected_rtgs_module_origins(),
        "loaded_local_sources": list(birth.loaded_local_sources()),
        "unbound_loaded_local_sources": [],
        "sys_path": ["/focused/runtime"],
        "torch_generated_import_path": {"torch_generated_path": "/focused/torch"},
        "pythonpath": birth.os.environ.get("PYTHONPATH"),
        "preload": str(birth.PRELOAD),
        "preload_sha256": birth.factorial.EXPECTED_PRELOAD_SHA256,
        "effective_ld_preload": str(birth.PRELOAD),
    }
    monkeypatch.setattr(
        birth,
        "EXPECTED_FROZEN_RUNTIME_SHA256",
        birth.canonical_hash(frozen_runtime),
    )
    monkeypatch.setattr(
        birth,
        "_load_hash_frozen_json",
        lambda _path, **_kwargs: {"runtime": frozen_runtime},
    )
    runtime = birth.runtime_binding()
    assert runtime["unbound_loaded_local_sources"] == []
    assert runtime["module_origins"] == birth.expected_rtgs_module_origins()
    assert runtime["cuda_device"] == "NVIDIA GeForce RTX 3050"
    assert runtime["cuda_capability"] == [8, 6]
    assert runtime["sys_path"] == ["/focused/runtime"]

    drifted_runtime = dict(frozen_runtime)
    drifted_runtime["torch"] = "drifted-torch"
    monkeypatch.setattr(
        birth,
        "EXPECTED_FROZEN_RUNTIME_SHA256",
        birth.canonical_hash(drifted_runtime),
    )
    monkeypatch.setattr(
        birth,
        "_load_hash_frozen_json",
        lambda _path, **_kwargs: {"runtime": drifted_runtime},
    )
    with pytest.raises(birth.ProtocolInvalid, match="ABI|runtime"):
        birth.runtime_binding()


def test_binding_state_binds_exact_prerequisite_artifact_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_paths = {
        "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_PREREG.md",
        "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json",
        "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_AUDIT.md",
        "benchmarks/results/20260716_residual_responsibility_density_PREREG.md",
    }
    assert set(birth.EXPECTED_PREREQUISITE_SHA256) == expected_paths
    assert all(len(value) == 64 for value in birth.EXPECTED_PREREQUISITE_SHA256.values())
    frozen_inputs = {"focused": "inputs"}
    monkeypatch.setattr(birth, "_source_hashes", lambda: ({"focused.py": "a" * 64}, "b" * 64))
    monkeypatch.setattr(birth.factorial, "input_bindings", lambda: frozen_inputs)
    monkeypatch.setattr(birth, "runtime_binding", lambda: {"focused": "runtime"})
    monkeypatch.setattr(birth, "_git_binding", lambda: {"focused": "git"})
    monkeypatch.setattr(birth, "_config_record", lambda: {"focused": "config"})
    monkeypatch.setattr(
        birth,
        "EXPECTED_FROZEN_INPUT_BINDINGS_SHA256",
        birth.canonical_hash(frozen_inputs),
    )

    def prerequisite_hash(path: str | Path) -> str:
        relative = Path(path).relative_to(birth.ROOT).as_posix()
        assert relative in expected_paths
        return birth.EXPECTED_PREREQUISITE_SHA256[relative]

    monkeypatch.setattr(birth, "sha256_file", prerequisite_hash)
    monkeypatch.setattr(
        birth,
        "_load_hash_frozen_json",
        lambda _path, **_kwargs: {"inputs": frozen_inputs},
    )
    binding = birth._binding_state()
    assert binding["prerequisite_artifacts"] == birth.EXPECTED_PREREQUISITE_SHA256

    monkeypatch.setattr(birth.factorial, "input_bindings", lambda: {"focused": "drifted"})
    with pytest.raises(birth.ProtocolInvalid, match="current inputs"):
        birth._binding_state()
    monkeypatch.setattr(birth.factorial, "input_bindings", lambda: frozen_inputs)
    monkeypatch.setattr(birth, "EXPECTED_FROZEN_INPUT_BINDINGS_SHA256", "0" * 64)
    with pytest.raises(birth.ProtocolInvalid, match="input-binding"):
        birth._binding_state()


def test_unbound_and_shadowed_loaded_modules_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound_paths = set(birth._source_paths(include_implementation_review=False))
    unbound_path = next(
        path.resolve()
        for path in sorted((birth.ROOT / "benchmarks").glob("*.py"))
        if path.resolve() not in bound_paths and path.resolve() != birth.VISUALIZER
    )
    unbound_local = types.ModuleType("iter2_unbound_local_source_probe")
    unbound_local.__file__ = str(unbound_path)
    monkeypatch.setitem(sys.modules, "iter2_unbound_local_source_probe", unbound_local)
    relative_unbound = unbound_path.relative_to(birth.ROOT).as_posix()
    assert relative_unbound in birth.unbound_loaded_local_sources()
    with monkeypatch.context() as closure_patch:
        closure_patch.setattr(birth, "module_origin_violations", lambda: ())
        with pytest.raises(birth.ProtocolInvalid, match="source closure"):
            birth._source_hashes()
    monkeypatch.delitem(sys.modules, "iter2_unbound_local_source_probe")

    bound_source = Path(__file__).resolve()
    unbound_module = types.ModuleType("rtgs.iter2_unbound_probe")
    unbound_module.__file__ = str(bound_source)
    monkeypatch.setitem(sys.modules, "rtgs.iter2_unbound_probe", unbound_module)
    violations = birth.module_origin_violations()
    assert any(item.startswith("unbound:rtgs.iter2_unbound_probe=") for item in violations)
    with pytest.raises(birth.ProtocolInvalid, match="origin mismatch"):
        birth._source_hashes()
    monkeypatch.delitem(sys.modules, "rtgs.iter2_unbound_probe")

    shadow = types.ModuleType("rtgs.core.camera")
    shadow.__file__ = str(bound_source)
    monkeypatch.setitem(sys.modules, "rtgs.core.camera", shadow)
    violations = birth.module_origin_violations()
    assert any(item.startswith("shadowed:rtgs.core.camera=") for item in violations)
    with pytest.raises(birth.ProtocolInvalid, match="origin mismatch"):
        birth._source_hashes()


def test_implementation_review_requires_one_exact_reviewed_aggregate(
    tmp_path: Path,
) -> None:
    _, aggregate = birth.reviewed_source_hashes()

    def review_text(*aggregate_lines: str) -> str:
        lines = [
            "# Focused implementation review fixture",
            "",
            "Verdict: PASS",
            "",
            "Unresolved findings: none",
            "",
            *aggregate_lines,
            "",
        ]
        return "\n".join(lines)

    prefix = "Reviewed source aggregate SHA-256: "
    correct = tmp_path / "correct.md"
    correct.write_text(review_text(prefix + aggregate), encoding="utf-8")
    assert birth.implementation_review_passed(correct)

    wrong = tmp_path / "wrong.md"
    wrong_digest = "0" * 64 if aggregate != "0" * 64 else "1" * 64
    wrong.write_text(review_text(prefix + wrong_digest), encoding="utf-8")
    assert not birth.implementation_review_passed(wrong)

    duplicate = tmp_path / "duplicate.md"
    duplicate.write_text(
        review_text(prefix + aggregate, prefix + aggregate),
        encoding="utf-8",
    )
    assert not birth.implementation_review_passed(duplicate)

    missing = tmp_path / "missing.md"
    missing.write_text(review_text(), encoding="utf-8")
    assert not birth.implementation_review_passed(missing)


def test_visualizer_existence_cannot_change_decision_source_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = birth._source_paths(include_implementation_review=False)
    assert birth.VISUALIZER not in original
    probe = birth.ROOT / "benchmarks/_iter2_visualizer_existence_probe.py"
    monkeypatch.setattr(birth, "VISUALIZER", probe)
    absent = birth._source_paths(include_implementation_review=False)

    original_is_file = Path.is_file

    def pretend_visualizer_exists(path: Path) -> bool:
        if path == probe:
            return True
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", pretend_visualizer_exists)
    present = birth._source_paths(include_implementation_review=False)
    assert absent == present == original
    assert probe not in present


@pytest.mark.parametrize(
    ("records", "structural", "expected"),
    (
        (
            _terminal_records(r_q=0.8, g_q=1.0, u_q=1.0),
            True,
            "RESIDUAL_RESPONSIBILITY_ALLOCATION_PROMISING",
        ),
        (
            _terminal_records(r_q=0.8, g_q=1.0, u_q=1.0, r_u=1.2),
            True,
            "UNIFORM_RISK_TRADEOFF",
        ),
        (
            _terminal_records(r_q=0.8, g_q=1.0, u_q=0.8),
            True,
            "MAPPING_NOT_ISOLATED",
        ),
        (
            _terminal_records(r_q=0.8, g_q=0.8, u_q=1.0),
            True,
            "NOT_BETTER_THAN_GRADIENT",
        ),
        (
            _terminal_records(r_q=1.0, g_q=1.0, u_q=1.0),
            True,
            "NO_PARENT_ALLOCATION_WIN",
        ),
        (
            _terminal_records(r_q=0.8, g_q=1.0, u_q=1.0),
            False,
            "UNAVAILABLE",
        ),
    ),
)
def test_terminal_decision_exhaustive_precedence(
    records: dict[int, dict[str, dict[str, Any]]],
    structural: bool,
    expected: str,
) -> None:
    result = birth.compute_terminal_decision(
        records,
        [0.97] * 21,
        roots=(1, 2, 3),
        structural_passed=structural,
    )
    assert result["scientific_decision"] == expected


def test_final_ply_elementwise_tolerance_helper() -> None:
    source = _ply_fixture()
    exact = source.detach()
    record = birth._validate_final_ply_roundtrip(source, exact, expected_count=867)
    assert set(record) == {"means", "quats", "log_scales", "opacity", "sh"}
    assert all(value["max_abs_error"] == 0.0 for value in record.values())

    within = source.detach()
    within.means[0, 0] += 5e-7
    birth._validate_final_ply_roundtrip(source, within, expected_count=867)

    outside = source.detach()
    outside.means[0, 0] += 5e-6
    with pytest.raises(birth.ProtocolInvalid, match="tolerance"):
        birth._validate_final_ply_roundtrip(source, outside, expected_count=867)
    with pytest.raises(birth.ProtocolInvalid, match="count"):
        birth._validate_final_ply_roundtrip(source, _ply_fixture(866), expected_count=867)


def test_focused_smoke_lifecycle_cannot_create_official_artifacts(
    tmp_path: Path,
) -> None:
    official_paths = (
        birth.SEAL,
        birth.PHASE_A_ATTEMPT,
        birth.PHASE_A_RESULT,
        birth.PHASE_A_AUDIT,
        birth.PHASE_B_ATTEMPT,
        birth.RESULT,
        birth.EXECUTED_SOURCES,
        birth.RUN_DIR,
    )
    before = {path: path.exists() for path in official_paths}
    output = tmp_path / "focused_smoke.json"
    record = birth.run_focused_smoke(output)
    after = {path: path.exists() for path in official_paths}
    assert after == before
    assert record == birth.strict_json(output)
    assert record["artifact_type"] == ("compact_responsibility_birth_iter2_focused_smoke_v1")
    assert record["status"] == "PASS"
    roots_used = set(record["focused_roots"].values())
    assert roots_used <= set(birth.FOCUSED_ROOTS)
    assert not roots_used & set(birth.OFFICIAL_ROOTS)
    assert record["official_roots_consumed"] == []
    assert record["root_guard"]["passed"]
    assert record["root_guard"]["no_generator_or_schedule_constructed"]
    assert record["selection_counts"] == {"G": 32, "R": 32, "U": 32}
