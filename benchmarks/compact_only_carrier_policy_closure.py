#!/usr/bin/env python3
"""Compact-only closure of carrier mean, continuation, SH, clone, and containment policy."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

if __package__:
    from benchmarks import compact_only_carrier_stage_ablation as stage1
else:
    import compact_only_carrier_stage_ablation as stage1

from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionResult, fuse_gaussian_beams
from rtgs.lift.carrier_refinement import repair_beam_carriers
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.render.projection import EWA_NEAR
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_PREREG.md"
DEFAULT_OUT = ROOT / "runs/compact_only_carrier_policy_closure_20260728"
DATASET = stage1.DATASET

PHASE1_ROOTS = (282701, 282702, 282703)
LATER_ROOTS = (283701, 283702, 283703)
EVALUATION_ROOTS = (284701, 284702, 284703)
FIT_GLOBAL = (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 24, 25)
VALIDATION_GLOBAL = (4, 10, 16, 21)
HELDOUT_GLOBAL = (7, 15, 23)
DEVELOPMENT_GLOBAL = (*FIT_GLOBAL, *VALIDATION_GLOBAL)
LOCAL_SPLITS = {
    "fit": tuple(range(len(FIT_GLOBAL))),
    "validation": tuple(range(len(FIT_GLOBAL), len(DEVELOPMENT_GLOBAL))),
}
EVALUATION_ATTEMPTS = stage1.EVALUATION_ATTEMPTS

PHASE1_ARMS: dict[str, tuple[str, ...]] = {
    "corrected_C_all_380": ("means", "quats", "scales", "opacities", "sh0"),
    "corrected_C_means_fixed_380": ("quats", "scales", "opacities", "sh0"),
    "corrected_C_means_only_380": ("means",),
}
LATER_TRAINING_ARMS = (
    "continue_all_380",
    "continue_means_fixed_380",
    "higher_sh_only_380",
    "legacy_clone_all_recover_190",
    "mass_tangent_clone_all_recover_190",
    "strict_prune_recover_190",
)
STATIC_ARMS = ("stop_380", "strict_prune")
ALL_ARMS = (*PHASE1_ARMS, *STATIC_ARMS, *LATER_TRAINING_ARMS)

SOURCE_PATHS = (
    "benchmarks/compact_only_carrier_policy_closure.py",
    "benchmarks/compact_only_carrier_stage_ablation.py",
    "src/rtgs/core/camera.py",
    "src/rtgs/core/gaussians3d.py",
    "src/rtgs/core/observation2d.py",
    "src/rtgs/data/__init__.py",
    "src/rtgs/data/compact_views.py",
    "src/rtgs/data/reconstruction_inputs.py",
    "src/rtgs/lift/__init__.py",
    "src/rtgs/lift/base.py",
    "src/rtgs/lift/beam_fusion.py",
    "src/rtgs/lift/carrier_refinement.py",
    "src/rtgs/optim/__init__.py",
    "src/rtgs/optim/compact_trainer.py",
    "src/rtgs/render/projection.py",
    "src/rtgs/render/torch_points.py",
)


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_binding() -> dict[str, object]:
    paths = set(SOURCE_PATHS)
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "src/rtgs").rglob("*.py"))
    files = {
        path: {
            "sha256": stage1._file_sha256(ROOT / path),
            "bytes": (ROOT / path).stat().st_size,
        }
        for path in sorted(paths)
    }
    return {
        "git_revision": _command("git", "rev-parse", "HEAD").strip(),
        "git_status": _command("git", "status", "--short").splitlines(),
        "command": [sys.executable, *sys.argv],
        "files": files,
        "semantic_sha256": stage1._canonical_hash(files),
    }


def _learning_rates(trainable: Sequence[str], *, sh_degree: int) -> dict[str, float]:
    enabled = set(trainable)
    return {
        "lr_means": 1.6e-4 if "means" in enabled else 0.0,
        "lr_quats": 1e-3 if "quats" in enabled else 0.0,
        "lr_scales": 5e-3 if "scales" in enabled else 0.0,
        "lr_opacity": 5e-2 if "opacities" in enabled else 0.0,
        "lr_sh": 2.5e-3 if "sh0" in enabled else 0.0,
        "lr_sh_rest": 2.5e-3 / 20 if sh_degree > 0 and "shN" in enabled else 0.0,
    }


def _train_config(
    *,
    iterations: int,
    checkpoints: tuple[int, ...],
    seed: int,
    extent: float,
    trainable: Sequence[str],
    sh_degree: int = 0,
) -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=iterations,
        attempts_per_step=256,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        seed=seed,
        extent=extent,
        device="cuda:0",
        **_learning_rates(trainable, sh_degree=sh_degree),
        point_chunk=256,
        gaussian_chunk=512,
        outer_microbatch=128,
        query_component_chunk=512,
        teacher_tile_size=16,
        evaluation_chunk=8192,
        checkpoints=checkpoints,
        evaluate_checkpoint_risks=False,
        sh_degree=sh_degree,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        support_mode="none",
        support_loss_weight=0.0,
    )


def _evaluate_views(
    gaussians: Gaussians3D,
    inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
    indices: Sequence[int],
) -> dict[str, object]:
    device = gaussians.means.device
    renderer = TorchPointRasterizer(point_chunk=8192, gaussian_chunk=512)
    per_view: list[dict[str, object]] = []
    with torch.no_grad():
        for index in indices:
            field = inputs.observations[index]
            camera = inputs.cameras[index]
            name = inputs.view_names[index]
            values: dict[str, float | int | str] = {
                "view_index": index,
                "view_name": name,
            }
            for kind, metric in (("uniform", "J_U"), ("proposal", "J_Q")):
                xy = banks[name][f"{kind}_xy"].to(device)
                active = banks[name][f"{kind}_active"].to(device)
                target = field.query(xy, component_chunk=512).color
                prediction = renderer.render_points(
                    gaussians,
                    camera,
                    xy,
                    background=torch.zeros(3, device=device),
                    sh_degree=gaussians.sh_degree,
                ).color
                point_loss = (prediction - target).square().mean(dim=-1)
                if kind == "proposal":
                    point_loss = point_loss * active.to(point_loss)
                values[metric] = float(point_loss.double().mean())
                values[f"{kind}_active_count"] = int(active.sum())
            per_view.append(values)
    return {
        "J_U": sum(float(record["J_U"]) for record in per_view) / len(per_view),
        "J_Q": sum(float(record["J_Q"]) for record in per_view) / len(per_view),
        "per_view": per_view,
    }


def _checkpoint_metrics(
    snapshots: Mapping[int, Gaussians3D],
    inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for step, snapshot in sorted(snapshots.items()):
        model = snapshot.to("cuda:0")
        metrics[str(step)] = {
            split: _evaluate_views(model, inputs, banks, indices)
            for split, indices in LOCAL_SPLITS.items()
        }
        del model
    return metrics


def _run_training_arm(
    out: Path,
    *,
    seed: int,
    arm: str,
    initialization: Gaussians3D,
    fit_inputs: ReconstructionInputs,
    development_inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
    config: CompactTrainConfig,
    transformation: Mapping[str, object] | None = None,
) -> tuple[Gaussians3D, dict[str, object]]:
    directory = out / "arms" / f"seed_{seed}" / arm
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["status"] != "complete":
            raise RuntimeError(f"incomplete arm receipt exists: {result_path}")
        return Gaussians3D.load_npz(directory / "gaussians_final.npz"), result
    directory.mkdir(parents=True, exist_ok=False)
    snapshots: dict[int, Gaussians3D] = {}

    def capture(snapshot: Gaussians3D, step: int) -> None:
        if step in snapshots:
            raise RuntimeError("duplicate checkpoint")
        snapshots[step] = snapshot.to("cpu")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    final, history = CompactTrainer(config).train(
        fit_inputs,
        initialization,
        checkpoint_callback=capture,
    )
    torch.cuda.synchronize(0)
    train_seconds = time.perf_counter() - started
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    if tuple(sorted(snapshots)) != config.checkpoints:
        raise RuntimeError("checkpoint set changed")
    metrics = _checkpoint_metrics(snapshots, development_inputs, banks)
    support = stage1._support_diagnostics(final, development_inputs, LOCAL_SPLITS)
    final_cpu = final.to("cpu")
    final_cpu.save_npz(directory / "gaussians_final.npz")
    stage1._atomic_json(directory / "history.json", history)
    frozen_motion = {
        family: history["optimizer_group_motion"][family]["max_abs"]
        for family in history["frozen_parameter_groups"]
    }
    if any(value != 0.0 for value in frozen_motion.values()):
        raise RuntimeError("a frozen parameter family moved")
    result: dict[str, object] = {
        "schema": "rtgs.compact-carrier-policy.arm.v1",
        "status": "complete",
        "seed": seed,
        "arm": arm,
        "kind": "training",
        "config": asdict(config),
        "transformation": transformation,
        "initial_semantic_sha256": stage1._gaussians_sha256(initialization),
        "final_semantic_sha256": stage1._gaussians_sha256(final_cpu),
        "final_file_sha256": stage1._file_sha256(directory / "gaussians_final.npz"),
        "history_file_sha256": stage1._file_sha256(directory / "history.json"),
        "checkpoint_metrics": metrics,
        "support": support,
        "frozen_motion": frozen_motion,
        "resources": {
            "train_seconds": train_seconds,
            "train_peak_cuda_allocated_bytes": train_peak_allocated,
            "train_peak_cuda_reserved_bytes": train_peak_reserved,
            "arm_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "arm_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "fit_teacher_tensor_bytes": stage1._teacher_bytes(fit_inputs),
            "development_teacher_tensor_bytes": stage1._teacher_bytes(development_inputs),
            "initial_3d_parameter_bytes": stage1._gaussian_parameter_bytes(initialization),
            "final_3d_parameter_bytes": stage1._gaussian_parameter_bytes(final_cpu),
            "point_chunk": config.point_chunk,
            "gaussian_chunk": config.gaussian_chunk,
            "outer_microbatch": config.outer_microbatch,
            "teacher_query_component_chunk": config.query_component_chunk,
        },
    }
    stage1._atomic_json(result_path, result)
    final_validation = metrics[str(config.iterations)]["validation"]
    print(
        f"[arm:{arm}:{seed}] n={final_cpu.n} "
        f"Q={final_validation['J_Q']:.8f} U={final_validation['J_U']:.8f} "
        f"{train_seconds:.1f}s",
        flush=True,
    )
    return final_cpu, result


def _run_static_arm(
    out: Path,
    *,
    seed: int,
    arm: str,
    model: Gaussians3D,
    development_inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
    transformation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    directory = out / "arms" / f"seed_{seed}" / arm
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["status"] != "complete":
            raise RuntimeError(f"incomplete static receipt exists: {result_path}")
        return result
    directory.mkdir(parents=True, exist_ok=False)
    gpu = model.to("cuda:0")
    metrics = {
        split: _evaluate_views(gpu, development_inputs, banks, indices)
        for split, indices in LOCAL_SPLITS.items()
    }
    support = stage1._support_diagnostics(gpu, development_inputs, LOCAL_SPLITS)
    cpu = gpu.to("cpu")
    cpu.save_npz(directory / "gaussians_final.npz")
    result = {
        "schema": "rtgs.compact-carrier-policy.arm.v1",
        "status": "complete",
        "seed": seed,
        "arm": arm,
        "kind": "static",
        "transformation": transformation,
        "final_semantic_sha256": stage1._gaussians_sha256(cpu),
        "final_file_sha256": stage1._file_sha256(directory / "gaussians_final.npz"),
        "metrics": metrics,
        "support": support,
        "resources": {
            "development_teacher_tensor_bytes": stage1._teacher_bytes(development_inputs),
            "final_3d_parameter_bytes": stage1._gaussian_parameter_bytes(cpu),
        },
    }
    stage1._atomic_json(result_path, result)
    print(
        f"[arm:{arm}:{seed}] n={cpu.n} "
        f"Q={metrics['validation']['J_Q']:.8f} U={metrics['validation']['J_U']:.8f}",
        flush=True,
    )
    return result


CloneMode = Literal["legacy_copy", "mass_tangent"]


def clone_all(base: Gaussians3D, mode: CloneMode) -> tuple[Gaussians3D, dict[str, object]]:
    """Apply the preregistered matched deterministic clone-all transformation."""

    if mode not in {"legacy_copy", "mass_tangent"}:
        raise ValueError("unknown clone mode")
    source = base.detach()
    scales = source.scales
    axis = scales.argmax(dim=-1)
    rows = torch.arange(source.n, device=source.means.device)
    sigma = scales[rows, axis]
    delta = 0.5 * sigma
    rotation = quat_to_rotmat(source.quats)
    unit = rotation[rows, :, axis]
    child_means = source.means + delta[:, None] * unit

    if mode == "legacy_copy":
        parent = source
        child = Gaussians3D(
            means=child_means,
            quats=source.quats.clone(),
            log_scales=source.log_scales.clone(),
            opacity=source.opacity.clone(),
            sh=source.sh.clone(),
        )
        combined = 1.0 - (1.0 - source.opacity).square()
        opacity_error = combined - source.opacity
        moment_error = delta.new_full((source.n,), float("nan"))
    else:
        matched_opacity = 1.0 - torch.sqrt((1.0 - source.opacity).clamp_min(1e-9))
        shrunk_sigma = torch.sqrt((sigma.square() - (0.5 * delta).square()).clamp_min(1e-12))
        log_scales = source.log_scales.clone()
        log_scales[rows, axis] = shrunk_sigma.log()
        parent = Gaussians3D(
            means=source.means.clone(),
            quats=source.quats.clone(),
            log_scales=log_scales,
            opacity=matched_opacity,
            sh=source.sh.clone(),
        )
        child = Gaussians3D(
            means=child_means,
            quats=source.quats.clone(),
            log_scales=log_scales.clone(),
            opacity=matched_opacity.clone(),
            sh=source.sh.clone(),
        )
        combined = 1.0 - (1.0 - matched_opacity).square()
        opacity_error = combined - source.opacity
        moment_error = shrunk_sigma.square() + (0.5 * delta).square() - sigma.square()

    output = Gaussians3D.cat([parent, child])
    if output.n != 2 * source.n:
        raise RuntimeError("clone-all count invariant failed")
    receipt = {
        "schema": "rtgs.compact-carrier-policy.clone.v1",
        "mode": mode,
        "n_before": source.n,
        "n_after": output.n,
        "axis_sha256": stage1._tensor_sha256(axis),
        "parent_means_sha256": stage1._tensor_sha256(output.means[: source.n]),
        "child_means_sha256": stage1._tensor_sha256(output.means[source.n :]),
        "delta_fraction": 0.5,
        "combined_opacity_error_max_abs": float(opacity_error.abs().max()),
        "combined_opacity_error_mean": float(opacity_error.mean()),
        "axis_second_moment_error_max_abs": (
            None if mode == "legacy_copy" else float(moment_error.abs().max())
        ),
    }
    return output, receipt


def strict_prune(
    base: Gaussians3D,
    fit_inputs: ReconstructionInputs,
) -> tuple[Gaussians3D, dict[str, object]]:
    """Remove every row outside at least one fitting-view teacher ellipse union."""

    model = base.to(fit_inputs.observations[0].device)
    keep = torch.ones(model.n, dtype=torch.bool, device=model.means.device)
    violating_view_count = torch.zeros(model.n, dtype=torch.int64, device=model.means.device)
    for field, camera in zip(fit_inputs.observations, fit_inputs.cameras, strict=True):
        if field.sigma_cutoff != 3.0:
            raise RuntimeError("strict-prune teacher cutoff changed")
        parts: list[torch.Tensor] = []
        depth_parts: list[torch.Tensor] = []
        for start in range(0, model.n, 512):
            end = min(start + 512, model.n)
            xy, depth = camera.project(model.means[start:end])
            parts.append(field.minimum_mahalanobis_squared(xy, component_chunk=512))
            depth_parts.append(depth)
        squared = torch.cat(parts)
        depth = torch.cat(depth_parts)
        valid = (squared <= 9.0) & (depth > EWA_NEAR)
        keep &= valid
        violating_view_count += ~valid
    output = model.subset(keep).to("cpu")
    receipt = {
        "schema": "rtgs.compact-carrier-policy.strict-prune.v1",
        "n_before": model.n,
        "n_after": output.n,
        "removed": int((~keep).sum()),
        "removed_fraction": float((~keep).float().mean()),
        "keep_mask_sha256": stage1._tensor_sha256(keep),
        "removed_rows": (~keep).nonzero(as_tuple=True)[0].detach().cpu().tolist(),
        "violating_view_count_sha256": stage1._tensor_sha256(violating_view_count),
        "max_violating_views_before": int(violating_view_count.max()),
        "criterion": "positive depth and nearest positive-amplitude ellipse q<=9 in every fit view",
    }
    return output, receipt


def _metric(record: Mapping[str, object], metric: str, *, step: int | None = None) -> float:
    if record["kind"] == "static":
        return float(record["metrics"]["validation"][metric])
    resolved = record["config"]["iterations"] if step is None else step
    return float(record["checkpoint_metrics"][str(resolved)]["validation"][metric])


def _geometric(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _comparison(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    numerator: str,
    denominator: str,
    *,
    numerator_step: int | None = None,
    denominator_step: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in ("J_Q", "J_U"):
        values = [
            _metric(records[seed][numerator], metric, step=numerator_step)
            / _metric(records[seed][denominator], metric, step=denominator_step)
            for seed in PHASE1_ROOTS
        ]
        result[metric] = {
            "per_seed": values,
            "geometric": _geometric(values),
            "wins": sum(value < 1.0 for value in values),
        }
    return result


def _analysis(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    comparisons = {
        "phase1_means_fixed": _comparison(
            records,
            "corrected_C_means_fixed_380",
            "corrected_C_all_380",
        ),
        "phase1_means_only": _comparison(
            records,
            "corrected_C_means_only_380",
            "corrected_C_all_380",
        ),
        "continue_all": _comparison(records, "continue_all_380", "stop_380"),
        "continue_means_fixed_vs_all": _comparison(
            records,
            "continue_means_fixed_380",
            "continue_all_380",
        ),
        "higher_sh": _comparison(records, "higher_sh_only_380", "stop_380"),
        "legacy_clone_vs_stop": _comparison(
            records,
            "legacy_clone_all_recover_190",
            "stop_380",
        ),
        "mass_clone_vs_stop": _comparison(
            records,
            "mass_tangent_clone_all_recover_190",
            "stop_380",
        ),
        "legacy_clone_vs_half_continue": _comparison(
            records,
            "legacy_clone_all_recover_190",
            "continue_all_380",
            denominator_step=190,
        ),
        "mass_clone_vs_half_continue": _comparison(
            records,
            "mass_tangent_clone_all_recover_190",
            "continue_all_380",
            denominator_step=190,
        ),
        "mass_vs_legacy_clone": _comparison(
            records,
            "mass_tangent_clone_all_recover_190",
            "legacy_clone_all_recover_190",
        ),
        "strict_prune": _comparison(records, "strict_prune", "stop_380"),
        "strict_prune_recover": _comparison(
            records,
            "strict_prune_recover_190",
            "stop_380",
        ),
    }
    decisions = {
        "phase1_means_fixed_noninferior": all(
            comparisons["phase1_means_fixed"][metric]["geometric"] <= 1.02
            for metric in ("J_Q", "J_U")
        ),
        "phase1_means_only_noninferior": all(
            comparisons["phase1_means_only"][metric]["geometric"] <= 1.02
            for metric in ("J_Q", "J_U")
        ),
        "continue_all_necessary": (
            comparisons["continue_all"]["J_Q"]["geometric"] <= 0.95
            and comparisons["continue_all"]["J_Q"]["wins"] == 3
            and comparisons["continue_all"]["J_U"]["geometric"] <= 1.02
        ),
        "continue_means_fixed_noninferior": all(
            comparisons["continue_means_fixed_vs_all"][metric]["geometric"] <= 1.02
            for metric in ("J_Q", "J_U")
        ),
        "higher_sh_necessary": (
            comparisons["higher_sh"]["J_Q"]["geometric"] <= 0.95
            and comparisons["higher_sh"]["J_Q"]["wins"] == 3
            and comparisons["higher_sh"]["J_U"]["geometric"] <= 1.02
        ),
        "legacy_clone_necessary": (
            comparisons["legacy_clone_vs_stop"]["J_Q"]["geometric"] <= 0.95
            and comparisons["legacy_clone_vs_stop"]["J_Q"]["wins"] == 3
            and comparisons["legacy_clone_vs_stop"]["J_U"]["geometric"] <= 1.02
            and comparisons["legacy_clone_vs_half_continue"]["J_Q"]["geometric"] <= 0.95
            and comparisons["legacy_clone_vs_half_continue"]["J_Q"]["wins"] == 3
            and comparisons["legacy_clone_vs_half_continue"]["J_U"]["geometric"] <= 1.02
        ),
        "mass_clone_necessary": (
            comparisons["mass_clone_vs_stop"]["J_Q"]["geometric"] <= 0.95
            and comparisons["mass_clone_vs_stop"]["J_Q"]["wins"] == 3
            and comparisons["mass_clone_vs_stop"]["J_U"]["geometric"] <= 1.02
            and comparisons["mass_clone_vs_half_continue"]["J_Q"]["geometric"] <= 0.95
            and comparisons["mass_clone_vs_half_continue"]["J_Q"]["wins"] == 3
            and comparisons["mass_clone_vs_half_continue"]["J_U"]["geometric"] <= 1.02
        ),
        "mass_clone_replaces_legacy": (
            comparisons["mass_vs_legacy_clone"]["J_Q"]["geometric"] <= 0.98
            and comparisons["mass_vs_legacy_clone"]["J_U"]["geometric"] <= 1.02
        ),
    }
    strict: dict[str, object] = {}
    for arm, comparison_key in (
        ("strict_prune", "strict_prune"),
        ("strict_prune_recover_190", "strict_prune_recover"),
    ):
        fit_outside = [
            float(records[seed][arm]["support"]["splits"]["fit"]["outside_fraction"])
            for seed in PHASE1_ROOTS
        ]
        fit_behind = [
            float(records[seed][arm]["support"]["splits"]["fit"]["behind_near_fraction"])
            for seed in PHASE1_ROOTS
        ]
        removed = [
            float(records[seed][arm]["transformation"]["removed_fraction"]) for seed in PHASE1_ROOTS
        ]
        strict[arm] = {
            "fit_outside_fraction": fit_outside,
            "fit_behind_near_fraction": fit_behind,
            "removed_fraction": removed,
            "viable": (
                max(fit_outside) == 0.0
                and max(fit_behind) == 0.0
                and max(removed) <= 0.10
                and all(
                    comparisons[comparison_key][metric]["geometric"] <= 1.05
                    for metric in ("J_Q", "J_U")
                )
            ),
        }
    decisions["strict_prune_viable"] = strict["strict_prune"]["viable"]
    decisions["strict_prune_recover_viable"] = strict["strict_prune_recover_190"]["viable"]
    ranking = sorted(
        ALL_ARMS,
        key=lambda arm: _geometric([_metric(records[seed][arm], "J_Q") for seed in PHASE1_ROOTS]),
    )
    return {
        "comparisons": comparisons,
        "strict_containment": strict,
        "decisions": decisions,
        "validation_ranking": ranking,
        "validation_summaries": {
            arm: {
                metric: {
                    "per_seed": [_metric(records[seed][arm], metric) for seed in PHASE1_ROOTS],
                    "geometric": _geometric(
                        [_metric(records[seed][arm], metric) for seed in PHASE1_ROOTS]
                    ),
                }
                for metric in ("J_Q", "J_U")
            }
            for arm in ALL_ARMS
        },
    }


def _fresh_corrected_initialization(
    out: Path,
    fit_inputs: ReconstructionInputs,
    extent: float,
) -> tuple[BeamFusionResult, Gaussians3D]:
    beam_path = out / "beam_gaussians.npz"
    if beam_path.exists():
        beam = stage1._load_beam(out)
    else:
        config = stage1._beam_config(extent)
        started = time.perf_counter()
        beam = fuse_gaussian_beams(fit_inputs, config)
        if beam.n_components != stage1.N_CARRIERS:
            raise RuntimeError("Beam carrier count changed")
        stage1._save_beam(out, beam, time.perf_counter() - started, config)
    repaired_path = out / "corrected_C_initialization.npz"
    receipt_path = out / "corrected_C_initialization.json"
    if repaired_path.exists() and receipt_path.exists():
        return beam, Gaussians3D.load_npz(repaired_path)
    config = stage1._repair_config(stage1.REPAIR_SPECS["corrected_C"])
    started = time.perf_counter()
    repaired = repair_beam_carriers(beam, fit_inputs, config)
    seconds = time.perf_counter() - started
    repaired.gaussians.save_npz(repaired_path)
    stage1._atomic_json(
        receipt_path,
        {
            "config": asdict(config),
            "seconds": seconds,
            "diagnostics": repaired.diagnostics,
            "semantic_sha256": stage1._gaussians_sha256(repaired.gaussians),
            "file_sha256": stage1._file_sha256(repaired_path),
        },
    )
    return beam, repaired.gaussians


def run(out: Path) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen policy-closure run requires cuda:0")
    out.mkdir(parents=True, exist_ok=True)
    source_binding = _source_binding()
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["source_binding"]["semantic_sha256"] != source_binding["semantic_sha256"]:
            raise RuntimeError("refusing to resume under a different source binding")
        manifest = existing
    else:
        manifest = {
            "schema": "rtgs.compact-carrier-policy.run.v1",
            "status": "running",
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "sha256": stage1._file_sha256(PREREGISTRATION),
            },
            "source_binding": source_binding,
            "roots": {
                "phase1": list(PHASE1_ROOTS),
                "later": list(LATER_ROOTS),
                "evaluation": list(EVALUATION_ROOTS),
            },
            "global_splits": {
                "fit": list(FIT_GLOBAL),
                "validation": list(VALIDATION_GLOBAL),
                "heldout_not_evaluated": list(HELDOUT_GLOBAL),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        }
        stage1._atomic_json(manifest_path, manifest)

    guard = stage1.NoImageGuard()
    started = time.perf_counter()
    with guard:
        dataset = CompactDataset.load(DATASET, device="cpu", load_alpha=False)
        development_cpu = stage1._inputs(
            dataset,
            DEVELOPMENT_GLOBAL,
            name="frame00008-policy-development",
        )
        fit_cpu = stage1._subset(
            development_cpu,
            LOCAL_SPLITS["fit"],
            name="frame00008-policy-fit",
        )
        extent = float(dataset.bounds_hint[1])
        manifest["dataset"] = {
            "path": str(DATASET.relative_to(ROOT)),
            "manifest_sha256": stage1._file_sha256(DATASET / "manifest.json"),
            "container_sha256": {view.view_id: view.sha256 for view in dataset.views},
            "container_bytes": sum(view.bytes for view in dataset.views),
            "packed_alpha_accessed": False,
            "development_view_ids": list(development_cpu.view_names),
            "fit_view_ids": list(fit_cpu.view_names),
            "heldout_view_ids_not_evaluated": [
                dataset.views[index].view_id for index in HELDOUT_GLOBAL
            ],
        }
        stage1._atomic_json(manifest_path, manifest)

        print("[initialization] fresh fit-only Beam + corrected covariance", flush=True)
        beam, corrected = _fresh_corrected_initialization(out, fit_cpu, extent)
        development_gpu = development_cpu.to("cuda:0")
        fit_gpu = stage1._subset(
            development_gpu,
            LOCAL_SPLITS["fit"],
            name="frame00008-policy-fit-gpu",
        )
        banks_by_root: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        for root in EVALUATION_ROOTS:
            path, bank = stage1._prepare_bank(out, development_cpu, root)
            banks_by_root[root] = bank
            print(f"[bank:{root}] {path.relative_to(out)}", flush=True)

        records: dict[int, dict[str, dict[str, object]]] = {}
        for seed, later_seed, evaluation_root in zip(
            PHASE1_ROOTS,
            LATER_ROOTS,
            EVALUATION_ROOTS,
            strict=True,
        ):
            records[seed] = {}
            bank = banks_by_root[evaluation_root]
            phase1_models: dict[str, Gaussians3D] = {}
            for arm, trainable in PHASE1_ARMS.items():
                model, record = _run_training_arm(
                    out,
                    seed=seed,
                    arm=arm,
                    initialization=corrected,
                    fit_inputs=fit_gpu,
                    development_inputs=development_gpu,
                    banks=bank,
                    config=_train_config(
                        iterations=380,
                        checkpoints=(0, 95, 190, 380),
                        seed=seed,
                        extent=extent,
                        trainable=trainable,
                    ),
                )
                phase1_models[arm] = model
                records[seed][arm] = record

            base = phase1_models["corrected_C_all_380"]
            records[seed]["stop_380"] = _run_static_arm(
                out,
                seed=seed,
                arm="stop_380",
                model=base,
                development_inputs=development_gpu,
                banks=bank,
                transformation={"kind": "identity_phase_boundary"},
            )
            for arm, trainable in (
                (
                    "continue_all_380",
                    ("means", "quats", "scales", "opacities", "sh0"),
                ),
                (
                    "continue_means_fixed_380",
                    ("quats", "scales", "opacities", "sh0"),
                ),
            ):
                _model, record = _run_training_arm(
                    out,
                    seed=seed,
                    arm=arm,
                    initialization=base,
                    fit_inputs=fit_gpu,
                    development_inputs=development_gpu,
                    banks=bank,
                    config=_train_config(
                        iterations=380,
                        checkpoints=(0, 190, 380),
                        seed=later_seed,
                        extent=extent,
                        trainable=trainable,
                    ),
                    transformation={"kind": "fresh_adam_continuation"},
                )
                records[seed][arm] = record

            _model, record = _run_training_arm(
                out,
                seed=seed,
                arm="higher_sh_only_380",
                initialization=base.with_sh_degree(3),
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=_train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=later_seed,
                    extent=extent,
                    trainable=("sh0", "shN"),
                    sh_degree=3,
                ),
                transformation={"kind": "expand_sh", "from_degree": 0, "to_degree": 3},
            )
            records[seed]["higher_sh_only_380"] = record

            legacy, legacy_receipt = clone_all(base, "legacy_copy")
            mass, mass_receipt = clone_all(base, "mass_tangent")
            if not (
                torch.equal(legacy.means, mass.means)
                and torch.equal(legacy.quats, mass.quats)
                and torch.equal(legacy.sh, mass.sh)
            ):
                raise RuntimeError("matched clone arms changed positions/frame/appearance")
            for arm, initialization, receipt in (
                ("legacy_clone_all_recover_190", legacy, legacy_receipt),
                ("mass_tangent_clone_all_recover_190", mass, mass_receipt),
            ):
                _model, record = _run_training_arm(
                    out,
                    seed=seed,
                    arm=arm,
                    initialization=initialization,
                    fit_inputs=fit_gpu,
                    development_inputs=development_gpu,
                    banks=bank,
                    config=_train_config(
                        iterations=190,
                        checkpoints=(0, 95, 190),
                        seed=later_seed,
                        extent=extent,
                        trainable=("means", "quats", "scales", "opacities", "sh0"),
                    ),
                    transformation=receipt,
                )
                records[seed][arm] = record

            pruned, prune_receipt = strict_prune(base, fit_gpu)
            records[seed]["strict_prune"] = _run_static_arm(
                out,
                seed=seed,
                arm="strict_prune",
                model=pruned,
                development_inputs=development_gpu,
                banks=bank,
                transformation=prune_receipt,
            )
            _model, record = _run_training_arm(
                out,
                seed=seed,
                arm="strict_prune_recover_190",
                initialization=pruned,
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=_train_config(
                    iterations=190,
                    checkpoints=(0, 95, 190),
                    seed=later_seed,
                    extent=extent,
                    trainable=("quats", "scales", "opacities", "sh0"),
                ),
                transformation=prune_receipt,
            )
            records[seed]["strict_prune_recover_190"] = record
            for arm in ("strict_prune", "strict_prune_recover_190"):
                support = records[seed][arm]["support"]["splits"]["fit"]
                if support["outside_fraction"] != 0.0 or support["behind_near_fraction"] != 0.0:
                    raise RuntimeError("strict containment did not survive arm execution")

        analysis = _analysis(records)
        stage1._atomic_json(out / "analysis.json", analysis)
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"no-image boundary failed: {guard_record}")
        result = {
            "schema": "rtgs.compact-carrier-policy.result.v1",
            "status": "complete",
            "scope": "single-scene compact-only development",
            "beam_semantic_sha256": stage1._gaussians_sha256(beam.gaussians),
            "corrected_initialization_semantic_sha256": stage1._gaussians_sha256(corrected),
            "records": {
                str(seed): {
                    arm: {
                        "path": str(
                            (out / "arms" / f"seed_{seed}" / arm / "result.json").relative_to(out)
                        ),
                        "sha256": stage1._file_sha256(
                            out / "arms" / f"seed_{seed}" / arm / "result.json"
                        ),
                    }
                    for arm in ALL_ARMS
                }
                for seed in PHASE1_ROOTS
            },
            "analysis": analysis,
            "heldout_evaluated": False,
            "no_image_boundary": guard_record,
            "resources": {
                "total_seconds": time.perf_counter() - started,
                "peak_rss_bytes": stage1._peak_rss_bytes(),
                "compact_container_bytes": sum(view.bytes for view in dataset.views),
                "development_teacher_tensor_bytes": stage1._teacher_bytes(development_gpu),
            },
        }
        stage1._atomic_json(out / "result.json", result)
        manifest["status"] = "complete"
        manifest["result_sha256"] = stage1._file_sha256(out / "result.json")
        manifest["no_image_boundary"] = guard_record
        manifest["total_seconds"] = result["resources"]["total_seconds"]
        stage1._atomic_json(manifest_path, manifest)
    print(
        f"[complete] winner={analysis['validation_ranking'][0]} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    return run(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
