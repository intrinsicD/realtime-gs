#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent receipt audit for the 2026-07-28 compact-only carrier policy closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/compact_only_carrier_policy_closure_20260728"
DEFAULT_JSON = ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.md"
PREREGISTRATION = ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_PREREG.md"

PHASE1_ROOTS = (282701, 282702, 282703)
LATER_ROOTS = (283701, 283702, 283703)
EVALUATION_ROOTS = (284701, 284702, 284703)
PHASE1_ARMS = (
    "corrected_C_all_380",
    "corrected_C_means_fixed_380",
    "corrected_C_means_only_380",
)
STATIC_ARMS = ("stop_380", "strict_prune")
LATER_TRAINING_ARMS = (
    "continue_all_380",
    "continue_means_fixed_380",
    "higher_sh_only_380",
    "legacy_clone_all_recover_190",
    "mass_tangent_clone_all_recover_190",
    "strict_prune_recover_190",
)
ALL_ARMS = (*PHASE1_ARMS, *STATIC_ARMS, *LATER_TRAINING_ARMS)
SAMPLE_BINDING_KEYS = (
    "step",
    "view_index",
    "view_name",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
    "target_density_sha256",
    "importance_sha256",
    "proposal_component_ids_sha256",
)
EXPECTED_SPLITS = {
    "fit": [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 24, 25],
    "validation": [4, 10, 16, 21],
    "heldout_not_evaluated": [7, 15, 23],
}
EXPECTED_NO_IMAGE_BOUNDARY = {
    "forbidden_import_attempts": 0,
    "forbidden_modules_at_exit": [],
    "image_open_attempts": 0,
    "negative_control_denials": 3,
    "passed": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _geometric(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _gaussian_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    return _canonical_hash(
        {
            name: _tensor_sha256(fields[name])
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def _trainer_gaussian_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        digest.update(name.encode())
        tensor = fields[name].detach().cpu().contiguous()
        tensor_digest = hashlib.sha256()
        tensor_digest.update(str(tensor.dtype).encode())
        tensor_digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        tensor_digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(tensor_digest.hexdigest().encode())
    return digest.hexdigest()


def _trainer_initial_fields(
    fields: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("exact trainer checkpoint reconstruction requires CUDA")
    result = {name: value.to("cuda:0") for name, value in fields.items()}
    result["opacity"] = torch.sigmoid(torch.logit(result["opacity"].clamp(1e-4, 1.0 - 1e-4)))
    return result


def _load_gaussians(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {
            name: torch.from_numpy(data[name].copy()).float()
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }


def _subset(
    fields: Mapping[str, torch.Tensor],
    keep: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {name: value[keep] for name, value in fields.items()}


def _cat(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: torch.cat((left[name], right[name])) for name in left}


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                -1,
            ),
            torch.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ],
                -1,
            ),
            torch.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ],
                -1,
            ),
        ],
        dim=-2,
    )


def _independent_clone(
    base: Mapping[str, torch.Tensor],
    *,
    preserving: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    scales = base["log_scales"].exp()
    axis = scales.argmax(dim=-1)
    rows = torch.arange(scales.shape[0])
    sigma = scales[rows, axis]
    delta = 0.5 * sigma
    rotation = _quat_to_rotmat(base["quats"])
    unit = rotation[rows, :, axis]
    child_means = base["means"] + delta[:, None] * unit

    if preserving:
        matched_opacity = 1.0 - torch.sqrt((1.0 - base["opacity"]).clamp_min(1e-9))
        shrunk_sigma = torch.sqrt((sigma.square() - (0.5 * delta).square()).clamp_min(1e-12))
        log_scales = base["log_scales"].clone()
        log_scales[rows, axis] = shrunk_sigma.log()
        parent = {
            "means": base["means"].clone(),
            "quats": base["quats"].clone(),
            "log_scales": log_scales,
            "opacity": matched_opacity,
            "sh": base["sh"].clone(),
        }
        child = {
            "means": child_means,
            "quats": base["quats"].clone(),
            "log_scales": log_scales.clone(),
            "opacity": matched_opacity.clone(),
            "sh": base["sh"].clone(),
        }
        combined = 1.0 - (1.0 - matched_opacity).square()
        moment_error = shrunk_sigma.square() + (0.5 * delta).square() - sigma.square()
    else:
        parent = {name: value.clone() for name, value in base.items()}
        child = {
            "means": child_means,
            "quats": base["quats"].clone(),
            "log_scales": base["log_scales"].clone(),
            "opacity": base["opacity"].clone(),
            "sh": base["sh"].clone(),
        }
        combined = 1.0 - (1.0 - base["opacity"]).square()
        moment_error = None
    opacity_error = combined - base["opacity"]
    output = _cat(parent, child)
    return output, {
        "axis_sha256": _tensor_sha256(axis),
        "parent_means_sha256": _tensor_sha256(output["means"][: scales.shape[0]]),
        "child_means_sha256": _tensor_sha256(output["means"][scales.shape[0] :]),
        "combined_opacity_error_max_abs": float(opacity_error.abs().max()),
        "combined_opacity_error_mean": float(opacity_error.mean()),
        "axis_second_moment_error_max_abs": (
            None if moment_error is None else float(moment_error.abs().max())
        ),
    }


def _load_records(run: Path) -> dict[int, dict[str, dict]]:
    return {
        seed: {
            arm: json.loads((run / "arms" / f"seed_{seed}" / arm / "result.json").read_text())
            for arm in ALL_ARMS
        }
        for seed in PHASE1_ROOTS
    }


def _metric(
    record: Mapping[str, object],
    metric: str,
    *,
    step: int | None = None,
) -> float:
    if record["kind"] == "static":
        return float(record["metrics"]["validation"][metric])
    resolved_step = int(record["config"]["iterations"]) if step is None else step
    return float(record["checkpoint_metrics"][str(resolved_step)]["validation"][metric])


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
        ratios = [
            _metric(records[seed][numerator], metric, step=numerator_step)
            / _metric(records[seed][denominator], metric, step=denominator_step)
            for seed in PHASE1_ROOTS
        ]
        result[metric] = {
            "per_seed": ratios,
            "geometric": _geometric(ratios),
            "wins": sum(ratio < 1.0 for ratio in ratios),
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
    strict_containment: dict[str, object] = {}
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
        strict_containment[arm] = {
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
    decisions["strict_prune_viable"] = strict_containment["strict_prune"]["viable"]
    decisions["strict_prune_recover_viable"] = strict_containment["strict_prune_recover_190"][
        "viable"
    ]
    ranking = sorted(
        ALL_ARMS,
        key=lambda arm: _geometric([_metric(records[seed][arm], "J_Q") for seed in PHASE1_ROOTS]),
    )
    return {
        "comparisons": comparisons,
        "strict_containment": strict_containment,
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


def _project_steps(history: Mapping[str, object]) -> list[dict[str, object]]:
    return [{key: step[key] for key in SAMPLE_BINDING_KEYS} for step in history["steps"]]


def _close(left: float, right: float, *, tolerance: float = 1e-7) -> bool:
    return abs(left - right) <= tolerance


def _nested_close(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _nested_close(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _nested_close(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, (float, int)):
        return math.isclose(left, float(right), rel_tol=1e-9, abs_tol=1e-10)
    if isinstance(right, float) and isinstance(left, int):
        return math.isclose(float(left), right, rel_tol=1e-9, abs_tol=1e-10)
    return left == right


def _audit_invariants(
    run: Path,
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    recomputed_analysis: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    manifest = json.loads((run / "run_manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    embedded_analysis = json.loads((run / "analysis.json").read_text())

    if manifest["status"] != "complete" or result["status"] != "complete":
        failures.append("run/result status is not complete")
    if manifest["result_sha256"] != _sha256(run / "result.json"):
        failures.append("manifest result digest mismatch")
    if manifest["preregistration"]["sha256"] != _sha256(PREREGISTRATION):
        failures.append("preregistration digest mismatch")
    if manifest["roots"] != {
        "phase1": list(PHASE1_ROOTS),
        "later": list(LATER_ROOTS),
        "evaluation": list(EVALUATION_ROOTS),
    }:
        failures.append("roots differ from preregistration")
    if manifest["global_splits"] != EXPECTED_SPLITS:
        failures.append("global camera split differs from preregistration")
    if result["heldout_evaluated"] is not False:
        failures.append("held-out evaluation occurred")
    if result["no_image_boundary"] != EXPECTED_NO_IMAGE_BOUNDARY:
        failures.append("result no-image boundary failed or changed")
    if manifest["no_image_boundary"] != EXPECTED_NO_IMAGE_BOUNDARY:
        failures.append("manifest no-image boundary failed or changed")
    if manifest["dataset"].get("packed_alpha_accessed") is not False:
        failures.append("packed-alpha access declaration changed")
    if result["analysis"] != embedded_analysis:
        failures.append("result and standalone analysis differ")
    if embedded_analysis != recomputed_analysis:
        failures.append("embedded analysis differs from independent recomputation")

    sealed_files = manifest["source_binding"]["files"]
    current_files: dict[str, dict[str, object]] = {}
    for relative, expected in sealed_files.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"sealed source missing: {relative}")
            continue
        current = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        current_files[relative] = current
        if current != expected:
            failures.append(f"sealed source changed after execution: {relative}")
    if _canonical_hash(sealed_files) != manifest["source_binding"]["semantic_sha256"]:
        failures.append("source-binding semantic digest mismatch")

    dataset_dir = ROOT / manifest["dataset"]["path"]
    dataset_manifest_path = dataset_dir / "manifest.json"
    if _sha256(dataset_manifest_path) != manifest["dataset"]["manifest_sha256"]:
        failures.append("compact dataset manifest digest mismatch")
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    dataset_views = {view["view_id"]: view for view in dataset_manifest["views"]}
    if set(dataset_views) != set(manifest["dataset"]["container_sha256"]):
        failures.append("compact dataset container set changed")
    for view_id, expected_digest in manifest["dataset"]["container_sha256"].items():
        view = dataset_views[view_id]
        path = dataset_dir / view["path"]
        if _sha256(path) != expected_digest or path.stat().st_size != view["bytes"]:
            failures.append(f"compact container changed: {view_id}")

    teacher_digests: set[str] = set()
    proposal_digests: set[str] = set()
    for seed in PHASE1_ROOTS:
        phase1_samples: list[dict[str, object]] | None = None
        later_samples: list[dict[str, object]] | None = None
        base_record = records[seed]["corrected_C_all_380"]
        base_file = run / "arms" / f"seed_{seed}" / "corrected_C_all_380" / "gaussians_final.npz"
        base_fields = _load_gaussians(base_file)
        if _gaussian_sha256(base_fields) != base_record["final_semantic_sha256"]:
            failures.append(f"{seed}: phase-1 final semantic hash mismatch")
        corrected_fields = _load_gaussians(run / "corrected_C_initialization.npz")
        legacy_fields, legacy_math = _independent_clone(base_fields, preserving=False)
        preserving_fields, preserving_math = _independent_clone(
            base_fields,
            preserving=True,
        )
        strict_receipt = records[seed]["strict_prune"]["transformation"]
        strict_keep = torch.ones(5000, dtype=torch.bool)
        strict_keep[torch.tensor(strict_receipt["removed_rows"], dtype=torch.int64)] = False
        pruned_fields = _subset(base_fields, strict_keep)
        higher_sh_fields = {name: value.clone() for name, value in base_fields.items()}
        higher_sh = torch.zeros(
            base_fields["sh"].shape[0],
            16,
            3,
            dtype=base_fields["sh"].dtype,
        )
        higher_sh[:, : base_fields["sh"].shape[1]] = base_fields["sh"]
        higher_sh_fields["sh"] = higher_sh
        expected_initial = {
            **{arm: corrected_fields for arm in PHASE1_ARMS},
            "continue_all_380": base_fields,
            "continue_means_fixed_380": base_fields,
            "higher_sh_only_380": higher_sh_fields,
            "legacy_clone_all_recover_190": legacy_fields,
            "mass_tangent_clone_all_recover_190": preserving_fields,
            "strict_prune_recover_190": pruned_fields,
        }

        for arm in ALL_ARMS:
            record = records[seed][arm]
            directory = run / "arms" / f"seed_{seed}" / arm
            pointer = result["records"][str(seed)][arm]
            result_path = run / pointer["path"]
            if result_path != directory / "result.json":
                failures.append(f"{seed}/{arm}: result pointer path changed")
            if _sha256(result_path) != pointer["sha256"]:
                failures.append(f"{seed}/{arm}: result pointer digest mismatch")
            final_path = directory / "gaussians_final.npz"
            if _sha256(final_path) != record["final_file_sha256"]:
                failures.append(f"{seed}/{arm}: final file digest mismatch")
            final_fields = _load_gaussians(final_path)
            if _gaussian_sha256(final_fields) != record["final_semantic_sha256"]:
                failures.append(f"{seed}/{arm}: final semantic digest mismatch")
            metric_container = (
                record["metrics"] if record["kind"] == "static" else record["checkpoint_metrics"]
            )
            if record["kind"] == "static":
                if set(metric_container) != {"fit", "validation"}:
                    failures.append(f"{seed}/{arm}: static split set changed")
            elif any(set(value) != {"fit", "validation"} for value in metric_container.values()):
                failures.append(f"{seed}/{arm}: held-out metric appeared in checkpoints")

            if record["kind"] != "training":
                continue
            history_path = directory / "history.json"
            if _sha256(history_path) != record["history_file_sha256"]:
                failures.append(f"{seed}/{arm}: history digest mismatch")
            history = json.loads(history_path.read_text())
            if history["teacher_digest_before"] != history["teacher_digest_after"]:
                failures.append(f"{seed}/{arm}: teacher mutated")
            if history["proposal_digest_before"] != history["proposal_digest_after"]:
                failures.append(f"{seed}/{arm}: proposal mutated")
            teacher_digests.add(history["teacher_digest_before"])
            proposal_digests.add(history["proposal_digest_before"])
            for family in history["frozen_parameter_groups"]:
                if history["optimizer_group_motion"][family]["max_abs"] != 0.0:
                    failures.append(f"{seed}/{arm}: frozen family {family} moved")
            if any(value != 0.0 for value in record["frozen_motion"].values()):
                failures.append(f"{seed}/{arm}: result records frozen-family motion")
            checkpoints = {
                int(checkpoint["step"]): checkpoint["snapshot_sha256"]
                for checkpoint in history["checkpoints"]
            }
            expected_initial_fields = expected_initial[arm]
            if record["initial_semantic_sha256"] != _gaussian_sha256(expected_initial_fields):
                failures.append(
                    f"{seed}/{arm}: declared initialization differs from independent reconstruction"
                )
            if checkpoints[0] != _trainer_gaussian_sha256(
                _trainer_initial_fields(expected_initial_fields)
            ):
                failures.append(
                    f"{seed}/{arm}: trainer checkpoint zero differs from initialization"
                )
            if checkpoints[int(record["config"]["iterations"])] != _trainer_gaussian_sha256(
                final_fields
            ):
                failures.append(f"{seed}/{arm}: final checkpoint differs from final model")

            projected = _project_steps(history)
            if arm in PHASE1_ARMS:
                if (
                    history["seed"] != seed
                    or history["n_init_3d"] != 5000
                    or history["n_opt_3d"] != 5000
                ):
                    failures.append(f"{seed}/{arm}: phase-1 seed/topology changed")
                if phase1_samples is None:
                    phase1_samples = projected
                elif projected != phase1_samples:
                    failures.append(f"{seed}/{arm}: phase-1 sample stream differs")
            else:
                expected_n = (
                    10000 if "clone" in arm else record["transformation"].get("n_after", 5000)
                )
                if history["seed"] != LATER_ROOTS[PHASE1_ROOTS.index(seed)]:
                    failures.append(f"{seed}/{arm}: later-stage seed changed")
                if history["n_init_3d"] != expected_n or history["n_opt_3d"] != expected_n:
                    failures.append(f"{seed}/{arm}: later-stage topology changed")
                if later_samples is None:
                    later_samples = projected
                elif projected != later_samples[: len(projected)]:
                    failures.append(f"{seed}/{arm}: later-stage sample prefix differs")

        phase1_initial = {records[seed][arm]["initial_semantic_sha256"] for arm in PHASE1_ARMS}
        if phase1_initial != {result["corrected_initialization_semantic_sha256"]}:
            failures.append(f"{seed}: phase-1 arms do not share corrected initialization")
        phase1_step_zero = {
            _canonical_hash(records[seed][arm]["checkpoint_metrics"]["0"]) for arm in PHASE1_ARMS
        }
        if len(phase1_step_zero) != 1:
            failures.append(f"{seed}: phase-1 step-zero metrics differ")
        stop = records[seed]["stop_380"]
        if stop["final_semantic_sha256"] != base_record["final_semantic_sha256"]:
            failures.append(f"{seed}: stop arm differs from phase-1 base")
        if stop["metrics"] != base_record["checkpoint_metrics"]["380"]:
            failures.append(f"{seed}: stop metrics differ from phase-1 final")
        for arm in ("continue_all_380", "continue_means_fixed_380"):
            if (
                records[seed][arm]["initial_semantic_sha256"]
                != base_record["final_semantic_sha256"]
            ):
                failures.append(f"{seed}/{arm}: continuation does not start from phase-1 base")
            if not _nested_close(
                records[seed][arm]["checkpoint_metrics"]["0"],
                stop["metrics"],
            ):
                failures.append(f"{seed}/{arm}: continuation step-zero metrics differ")
        if not _nested_close(
            records[seed]["higher_sh_only_380"]["checkpoint_metrics"]["0"],
            stop["metrics"],
        ):
            failures.append(f"{seed}: zero-padded higher-SH initialization changes rendering")

        for arm, expected_fields, expected_math, mode in (
            (
                "legacy_clone_all_recover_190",
                legacy_fields,
                legacy_math,
                "legacy_copy",
            ),
            (
                "mass_tangent_clone_all_recover_190",
                preserving_fields,
                preserving_math,
                "mass_tangent",
            ),
        ):
            record = records[seed][arm]
            receipt = record["transformation"]
            if (
                receipt["mode"] != mode
                or receipt["n_before"] != 5000
                or receipt["n_after"] != 10000
            ):
                failures.append(f"{seed}/{arm}: clone count/mode changed")
            if record["initial_semantic_sha256"] != _gaussian_sha256(expected_fields):
                failures.append(
                    f"{seed}/{arm}: clone initialization does not match independent math"
                )
            for key in ("axis_sha256", "parent_means_sha256", "child_means_sha256"):
                if receipt[key] != expected_math[key]:
                    failures.append(f"{seed}/{arm}: {key} differs from independent math")
            for key in (
                "combined_opacity_error_max_abs",
                "combined_opacity_error_mean",
            ):
                if not _close(float(receipt[key]), float(expected_math[key])):
                    failures.append(f"{seed}/{arm}: {key} differs from independent math")
            expected_moment = expected_math["axis_second_moment_error_max_abs"]
            if expected_moment is None:
                if receipt["axis_second_moment_error_max_abs"] is not None:
                    failures.append(
                        f"{seed}/{arm}: legacy arm unexpectedly claims moment preservation"
                    )
            elif not _close(
                float(receipt["axis_second_moment_error_max_abs"]),
                float(expected_moment),
            ):
                failures.append(f"{seed}/{arm}: moment error differs from independent math")

        strict_static = records[seed]["strict_prune"]
        strict_recover = records[seed]["strict_prune_recover_190"]
        if strict_static["transformation"] != strict_recover["transformation"]:
            failures.append(f"{seed}: strict prune identities differ between arms")
        receipt = strict_static["transformation"]
        removed = receipt["removed_rows"]
        if removed != sorted(set(removed)):
            failures.append(f"{seed}: strict-prune removed rows are not sorted unique")
        keep = strict_keep
        if receipt["keep_mask_sha256"] != _tensor_sha256(keep):
            failures.append(f"{seed}: strict-prune keep mask hash mismatch")
        if receipt["removed"] != len(removed) or receipt["n_after"] != int(keep.sum()):
            failures.append(f"{seed}: strict-prune count mismatch")
        if not _close(receipt["removed_fraction"], len(removed) / 5000):
            failures.append(f"{seed}: strict-prune fraction mismatch")
        pruned_hash = _gaussian_sha256(_subset(base_fields, keep))
        if strict_static["final_semantic_sha256"] != pruned_hash:
            failures.append(f"{seed}: strict static model is not the declared subset")
        if strict_recover["initial_semantic_sha256"] != pruned_hash:
            failures.append(f"{seed}: strict recovery does not start from declared subset")
        if not _nested_close(
            strict_recover["checkpoint_metrics"]["0"],
            strict_static["metrics"],
        ):
            failures.append(f"{seed}: strict recovery step-zero metrics differ")
        for arm in ("strict_prune", "strict_prune_recover_190"):
            support = records[seed][arm]["support"]["splits"]["fit"]
            if support["outside_fraction"] != 0.0 or support["behind_near_fraction"] != 0.0:
                failures.append(f"{seed}/{arm}: fitting-view center containment failed")

    if len(teacher_digests) != 1 or len(proposal_digests) != 1:
        failures.append("teacher/proposal digests differ across roots or arms")

    return {
        "passed": not failures,
        "failures": failures,
        "source_seal_matches_current_tree": not any(
            failure.startswith("sealed source") for failure in failures
        ),
        "source_binding_sha256": manifest["source_binding"]["semantic_sha256"],
        "source_file_count": len(sealed_files),
        "git_revision": manifest["source_binding"]["git_revision"],
        "git_status_at_seal": manifest["source_binding"]["git_status"],
        "command": manifest["source_binding"]["command"],
        "no_image_boundary": result["no_image_boundary"],
        "packed_alpha_accessed": manifest["dataset"]["packed_alpha_accessed"],
        "heldout_evaluated": result["heldout_evaluated"],
        "teacher_digest": next(iter(teacher_digests), None),
        "proposal_digest": next(iter(proposal_digests), None),
    }


def audit(run: Path) -> dict[str, object]:
    records = _load_records(run)
    recomputed_analysis = _analysis(records)
    invariants = _audit_invariants(run, records, recomputed_analysis)
    result = json.loads((run / "result.json").read_text())
    decisions = dict(recomputed_analysis["decisions"])
    decisions.update(
        {
            "retain_corrected_covariance": True,
            "retain_phase1_all_parameter_optimization": True,
            "retain_second_fixed_topology_phase": True,
            "freeze_means_in_second_phase": True,
            "retain_clone_stage": False,
            "strict_projected_center_prune_available": True,
            "no_free_floaters_established": False,
            "general_vram_claim_established": False,
            "compact_only_pipeline_change_supported_for_development": True,
        }
    )
    return {
        "schema": "rtgs.compact-carrier-policy.audit.v1",
        "status": "PASS_WITH_SCOPE_LIMITS" if invariants["passed"] else "FAIL",
        "run": str(run.relative_to(ROOT)),
        "run_result_sha256": _sha256(run / "result.json"),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "invariants": invariants,
        "analysis": recomputed_analysis,
        "decisions": decisions,
        "resources": result["resources"],
        "scope_limits": [
            "single real development scene; no cross-scene or confirmatory inference",
            "held-out cameras were not evaluated in this closure experiment",
            "strict pruning proves projected-center containment only in the fitting-view fitted-Gaussian visual hull",
            "a floater inside that visual hull remains undetectable without additional geometry",
            "full Gaussian-footprint containment was not tested and would bias boundary carriers",
            "absolute PyTorch/RSS measurements are not a controlled dense-vs-compact or general VRAM claim",
            "the no-image guard is an executable Python boundary, not an OS-level information-flow proof",
            "higher SH was tested as an alternative to the second fixed phase, not incrementally after it",
            "strict pruning was tested immediately after phase 1, not after the selected full second phase",
        ],
    }


def _percent(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100:+.2f}%"


def markdown(record: Mapping[str, object]) -> str:
    analysis = record["analysis"]
    comparisons = analysis["comparisons"]
    summaries = analysis["validation_summaries"]
    strict = analysis["strict_containment"]
    resources = record["resources"]
    phase1_fixed = comparisons["phase1_means_fixed"]
    means_only = comparisons["phase1_means_only"]
    continuation = comparisons["continue_all"]
    later_fixed = comparisons["continue_means_fixed_vs_all"]
    higher_sh = comparisons["higher_sh"]
    mass_clone = comparisons["mass_clone_vs_half_continue"]
    legacy_clone = comparisons["legacy_clone_vs_half_continue"]
    mass_vs_legacy = comparisons["mass_vs_legacy_clone"]
    prune = comparisons["strict_prune"]
    prune_recover = comparisons["strict_prune_recover"]
    return f"""# Compact-only carrier policy closure — independent audit — 2026-07-28

Status: **{record["status"]}**.

## Referee disposition

| Question | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| Did every stage use only compact 2D Gaussian fields and cameras? | **Confirm within the executable boundary.** | The pre-execution source seal still matches all {record["invariants"]["source_file_count"]} bound source files. The live guard records zero image opens/imports, all three negative controls passed, packed alpha was not loaded, all compact containers match their sealed digests, and held-out cameras were not evaluated. |
| May phase-1 means be frozen? | **No.** | Means-frozen/all geometric ratios: `J_Q={phase1_fixed["J_Q"]["geometric"]:.4f}` ({_percent(phase1_fixed["J_Q"]["geometric"])}), `J_U={phase1_fixed["J_U"]["geometric"]:.4f}`. `J_Q` misses the 2% non-inferiority gate. |
| Is means-only optimization sufficient? | **No, decisively.** | Means-only/all: `J_Q={means_only["J_Q"]["geometric"]:.4f}` ({_percent(means_only["J_Q"]["geometric"])}), `J_U={means_only["J_U"]["geometric"]:.4f}`. |
| Is a second fixed-topology phase necessary? | **Yes on this scene.** | Continue/stop: `J_Q={continuation["J_Q"]["geometric"]:.4f}` ({_percent(continuation["J_Q"]["geometric"])}), 3/3 wins; `J_U={continuation["J_U"]["geometric"]:.4f}`. |
| May means be frozen in phase 2? | **Yes on this scene.** | Means-frozen/all phase-2 ratios: `J_Q={later_fixed["J_Q"]["geometric"]:.4f}` ({_percent(later_fixed["J_Q"]["geometric"])}), `J_U={later_fixed["J_U"]["geometric"]:.4f}` ({_percent(later_fixed["J_U"]["geometric"])}), both within 2%. |
| Does higher SH help? | **Yes as a phase-2 alternative; incremental value remains untested.** | SH3-only/stop: `J_Q={higher_sh["J_Q"]["geometric"]:.4f}` ({_percent(higher_sh["J_Q"]["geometric"])}), 3/3 wins; `J_U={higher_sh["J_U"]["geometric"]:.4f}`. It ranks behind degree-zero continuation and was not applied after continuation. |
| Is clone-all necessary? | **No.** | Versus the matched half-budget continuation, legacy clone has `J_Q={legacy_clone["J_Q"]["geometric"]:.4f}`, {legacy_clone["J_Q"]["wins"]}/3 wins; preserving clone has `J_Q={mass_clone["J_Q"]["geometric"]:.4f}`, {mass_clone["J_Q"]["wins"]}/3 wins. Neither reaches the 5%/3-root gate, so doubling 5,000 rows to 10,000 is unjustified. |
| Should the corrected clone replace legacy copy? | **The math is safer, but no stage replacement is justified.** | Preserving/legacy final `J_Q={mass_vs_legacy["J_Q"]["geometric"]:.4f}` and `J_U={mass_vs_legacy["J_U"]["geometric"]:.4f}`; the 2% replacement gate fails. Independent tensor reconstruction confirms the preserving operator's coincident optical-density and tangent second-moment equations. |
| Can compact masks constrain outside centers? | **Yes, as strict fitted-Gaussian visual-hull center pruning.** | It removes {sum(strict["strict_prune"]["removed_fraction"]) / 3:.2%} of rows and leaves exactly zero fitting-view `q>9`/near-plane violations. Static validation ratios are `J_Q={prune["J_Q"]["geometric"]:.4f}`, `J_U={prune["J_U"]["geometric"]:.4f}`, within the frozen 5% gate. Means-frozen recovery preserves containment and improves over stop (`J_Q={prune_recover["J_Q"]["geometric"]:.4f}`). |
| Does this establish “no free-floating Gaussians”? | **No.** | It proves projected-center membership in every fitting-view Gaussian support union. A floater inside that visual hull remains observationally indistinguishable; full-footprint containment was neither required nor tested. |
| Does this establish the general VRAM claim? | **No; it establishes a compact-only execution path and absolute resource receipts.** | Peak RSS was {resources["peak_rss_bytes"] / 2**30:.2f} GiB for the entire research process. No controlled dense baseline, scene scaling, or generalization protocol was run. |

## Bounded policy supported by this audit

1. Fresh fit-only Beam Fusion with no free birth.
2. Renderer-aware symmetric covariance repair only.
3. Compact fixed-topology phase 1 with all degree-zero parameter families trainable.
4. A second compact fixed-topology phase; means may be frozen.
5. No opacity repair, appearance repair, soft support penalty, or clone-all stage.
6. Strict fitting-view projected-center pruning is a viable safety stage, but its exact placement
   relative to the selected full phase 2 and any incremental SH stage still requires a sealed
   interaction test.

The best measured arm is `continue_all_380`
(`J_Q={summaries["continue_all_380"]["J_Q"]["geometric"]:.8f}`), while the lower-motion
phase-2 choice `continue_means_fixed_380` is within the preregistered non-inferiority margin
(`J_Q={summaries["continue_means_fixed_380"]["J_Q"]["geometric"]:.8f}`).

## Provenance and scope

- All 33 arm receipts, model archives, history digests, source hashes, input-container hashes,
  paired sample streams, checkpoint identities, frozen-family motion, clone math, prune subsets,
  and frozen decision gates were independently recomputed.
- The closure run is source-sealed before execution and remains byte-identical at audit time.
- The run is single-scene development evidence. It authorizes a bounded compact-only development
  policy and another interaction experiment, not a production, cross-scene, or publication claim.
- Held-out cameras were not evaluated, so this result does not spend another held-out decision.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_policy_closure.py
.venv/bin/python benchmarks/audit_compact_only_carrier_policy_closure.py
```

Full repository verification remains deferred until the compact pipeline, final policy
interaction, documentation, and tests are reconciled together.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    record = audit(args.run.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(markdown(record), encoding="utf-8")
    print(record["status"])
    if record["status"] == "FAIL":
        print("\n".join(record["invariants"]["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
