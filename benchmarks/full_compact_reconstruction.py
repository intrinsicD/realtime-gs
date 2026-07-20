#!/usr/bin/env python3
"""Full compact-view reconstruction with source RGB reserved for final evaluation.

The fit phase consumes only checked-in ``.rtgsv`` views. It:

1. enumerates every selected 2D component as a component-center placement candidate;
2. keeps a bounded latent 3D set on exact source-projection fibers;
3. replays each compact StructSplat teacher on its native fit-window crop; and
4. runs the ordinary CUDA gsplat trainer with dynamic density control.

The evaluation phase is the only code path allowed to open the provenance-matched original
capture. It is unreachable until ``gaussians_final.ply`` and the fit-complete receipt exist.

Default fitting intentionally uses all 26 views (130,000 compact components). The frozen T/V/H
partition remains available through ``--fit-mode protocol`` and is always reported separately;
when the default all-view fit reports V, V is a fitted validation diagnostic, not held-out
evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics, masked_crop, masked_psnr, ssim
from rtgs.data.compact_views import CompactDataset, CompactView
from rtgs.data.field_inputs import SceneFits
from rtgs.data.scene import SceneData
from rtgs.lift.field_lifter import FieldLiftConfig, _place
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    _resolve_schedule_iterations,
    _resolve_sh_interval,
)
from rtgs.render.projection import EWA_DILATION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
ORIGINAL_FRAME = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
ORIGINAL_CALIBRATION = Path(
    "/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/calibration_dome.json"
)
EXPECTED_CALIBRATION_SHA256 = "51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091"
EXPECTED_VIEW_NAMES = (
    "C0001",
    "C0004",
    "C0005",
    "C0006",
    "C0008",
    "C0009",
    "C0012",
    "C0014",
    "C0018",
    "C0019",
    "C0020",
    "C0021",
    "C0022",
    "C0025",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
    "C1004",
)
V_INDICES = (3, 11, 19)
H_INDICES = (7, 15, 23)
T_INDICES = tuple(
    index
    for index in range(len(EXPECTED_VIEW_NAMES))
    if index not in set(V_INDICES) | set(H_INDICES)
)
FINAL_FIT_ARTIFACTS = (
    "gaussians_final.ply",
    "training_history.json",
    "compact_metrics.json",
    "fit_complete.json",
)
LEGACY_NONDETERMINISTIC_TARGET_SCHEMA = "rtgs.full_compact_reconstruction.targets.v1"
DETERMINISTIC_TARGET_SCHEMA = "rtgs.full_compact_reconstruction.targets.v2"
TARGET_EXTREMA_ATOL = 1e-6
TARGET_METADATA_KEYS = (
    "global_index",
    "view_id",
    "fit_window",
    "renderer",
    "blend_mode",
    "components",
    "has_alpha",
    "alpha_applied",
)
TARGET_DETERMINISTIC_REPLAY_KEYS = (
    *TARGET_METADATA_KEYS,
    "crop_sha256",
    "unclamped_min",
    "unclamped_max",
)
PROVENANCE_IDENTITY_KEYS = (
    "schema",
    "written_before_source_rgb_access",
    "manifest",
    "manifest_sha256",
    "calibration_sha256",
    "bounds_hint",
    "fit_indices",
    "expected_component_center_candidates",
    "views",
    "environment",
)
POLISH_PARENT_RESUME_STEP = 4_000
POLISH_PARENT_ITERATIONS = 30_000
POLISH_ITERATIONS = 10_000
POLISH_SCHEDULE_ITERATIONS = 40_000
POLISH_OTHER_LR_FACTOR = 0.25
POLISH_SEED = 1
TAIL_PARENT_ITERATIONS = POLISH_SCHEDULE_ITERATIONS
TAIL_ITERATIONS = 10_000
TAIL_SCHEDULE_ITERATIONS = 50_000
TAIL_SEED = 2
COOLDOWN_PARENT_ITERATIONS = TAIL_SCHEDULE_ITERATIONS
COOLDOWN_ITERATIONS = 10_000
COOLDOWN_SCHEDULE_ITERATIONS = 60_000
COOLDOWN_LR_FACTOR = 0.25
COOLDOWN_SEED = 3
SETTLE_PARENT_ITERATIONS = COOLDOWN_SCHEDULE_ITERATIONS
SETTLE_ITERATIONS = 10_000
SETTLE_SCHEDULE_ITERATIONS = 70_000
SETTLE_LR_FACTOR = 0.25
SETTLE_SEED = 4
MODEL_SELECTION_SCHEMA = "rtgs.full_compact_reconstruction.model_selection.v1"
MODEL_SELECTION_RELATIVE_TIE = 1e-6
MATERIAL_OBJECTIVE_REDUCTION = 0.0025
MATERIAL_PSNR_GAIN_DB = 0.05
MATERIAL_PLATEAU_TRANSITIONS = 5
POLISH_PARENT_ARTIFACTS = (
    "config.json",
    "provenance.json",
    "training_config.json",
    "compact_targets.json",
    "training_history.json",
    "compact_metrics.json",
    "fit_complete.json",
    "gaussians_final.ply",
)
POLISH_COPIED_ARTIFACTS = (
    "config.json",
    "provenance.json",
)
TAIL_PARENT_ARTIFACTS = (
    "config.json",
    "provenance.json",
    "training_config.json",
    "compact_targets.json",
    "polish_start.json",
    "training_history.json",
    "compact_metrics.json",
    "model_selection.json",
    "fit_complete.json",
    "gaussians_final.ply",
)
COOLDOWN_PARENT_ARTIFACTS = (
    "config.json",
    "provenance.json",
    "training_config.json",
    "compact_targets.json",
    "tail_start.json",
    "training_history.json",
    "compact_metrics.json",
    "model_selection.json",
    "fit_complete.json",
    "gaussians_final.ply",
)
SETTLE_PARENT_ARTIFACTS = (
    "config.json",
    "provenance.json",
    "training_config.json",
    "compact_targets.json",
    "cooldown_start.json",
    "training_history.json",
    "compact_metrics.json",
    "model_selection.json",
    "fit_complete.json",
    "gaussians_final.ply",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _train_config_from_payload(payload: dict[str, Any], *, label: str) -> TrainConfig:
    density_payload = payload.get("density")
    if not isinstance(density_payload, dict):
        raise RuntimeError(f"{label} has no density object")
    values = dict(payload)
    values["density"] = DensityConfig(**density_payload)
    return TrainConfig(**values)


def _load_train_config(path: Path) -> TrainConfig:
    payload = _load_json_object(path, label="training config")
    return _train_config_from_payload(payload, label=f"training config: {path}")


def _record_identity(
    record: dict[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    missing = [key for key in keys if key not in record]
    if missing:
        raise RuntimeError(f"{label} is missing identity keys: {missing}")
    return {key: _json_safe(record[key]) for key in keys}


def _target_identity(record: dict[str, Any]) -> dict[str, Any]:
    return _record_identity(record, TARGET_METADATA_KEYS, label="compact target record")


def _provenance_identity(record: dict[str, Any]) -> dict[str, Any]:
    return _record_identity(record, PROVENANCE_IDENTITY_KEYS, label="compact provenance")


def _verify_provenance_identity(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    reference_identity = _provenance_identity(reference)
    current_identity = _provenance_identity(current)
    if reference_identity != current_identity:
        raise RuntimeError(
            "current compact provenance differs from the frozen fit "
            "(manifest, bundle, camera, source, view order, or environment)"
        )
    views = reference_identity["views"]
    if not isinstance(views, list):
        raise RuntimeError("compact provenance views must be a list")
    return {
        "identity_sha256": _json_sha256(reference_identity),
        "manifest_sha256": reference_identity["manifest_sha256"],
        "calibration_sha256": reference_identity["calibration_sha256"],
        "view_count": len(views),
        "all_fields_match": True,
    }


@contextmanager
def _deterministic_algorithms():
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )


def _verify_target_replay(
    reference_path: Path,
    replayed: Sequence[dict[str, Any]],
    independently_replayed: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_views = reference.get("views") if isinstance(reference, dict) else None
    if not isinstance(reference_views, list):
        raise RuntimeError("compact target receipt has no view list")
    reference_schema = reference.get("schema")
    if reference_schema not in (
        LEGACY_NONDETERMINISTIC_TARGET_SCHEMA,
        DETERMINISTIC_TARGET_SCHEMA,
    ):
        raise RuntimeError(f"unsupported compact target receipt schema: {reference_schema!r}")
    if reference_schema == DETERMINISTIC_TARGET_SCHEMA and (
        reference.get("deterministic_algorithms") is not True
    ):
        raise RuntimeError("deterministic compact target receipt lacks its determinism marker")

    reference_metadata = [_target_identity(record) for record in reference_views]
    replayed_metadata = [_target_identity(record) for record in replayed]
    independent_metadata = [_target_identity(record) for record in independently_replayed]
    if not (len(reference_metadata) == len(replayed_metadata) == len(independent_metadata)):
        raise RuntimeError(
            "compact target replay view count differs from the frozen target receipt"
        )
    if reference_metadata != replayed_metadata:
        raise RuntimeError("compact target replay metadata differs from the frozen fit")
    if replayed_metadata != independent_metadata:
        raise RuntimeError("independent deterministic replay metadata differs")

    replayed_deterministic = [
        _record_identity(
            record,
            TARGET_DETERMINISTIC_REPLAY_KEYS,
            label="deterministic compact target replay",
        )
        for record in replayed
    ]
    independent_deterministic = [
        _record_identity(
            record,
            TARGET_DETERMINISTIC_REPLAY_KEYS,
            label="independent deterministic compact target replay",
        )
        for record in independently_replayed
    ]
    if replayed_deterministic != independent_deterministic:
        raise RuntimeError("independent deterministic target replay differs bitwise")

    extrema_deltas: list[dict[str, Any]] = []
    frozen_hash_records: list[dict[str, Any]] = []
    for frozen, current in zip(reference_views, replayed, strict=True):
        view_id = str(current["view_id"])
        for key in ("unclamped_min", "unclamped_max"):
            frozen_value = frozen.get(key)
            current_value = current.get(key)
            if (
                isinstance(frozen_value, bool)
                or not isinstance(frozen_value, (int, float))
                or isinstance(current_value, bool)
                or not isinstance(current_value, (int, float))
                or not math.isfinite(float(frozen_value))
                or not math.isfinite(float(current_value))
            ):
                raise RuntimeError(f"{view_id} target {key} must be finite")
            delta = abs(float(frozen_value) - float(current_value))
            if delta > TARGET_EXTREMA_ATOL:
                raise RuntimeError(f"{view_id} target {key} differs from the frozen fit by {delta}")
            extrema_deltas.append({"view_id": view_id, "field": key, "delta": delta})
        frozen_hash = frozen.get("crop_sha256")
        current_hash = current.get("crop_sha256")
        if not isinstance(frozen_hash, str) or not isinstance(current_hash, str):
            raise RuntimeError(f"{view_id} target crop hash must be a string")
        frozen_hash_records.append(
            {
                "view_id": view_id,
                "frozen_crop_sha256": frozen_hash,
                "recovery_crop_sha256": current_hash,
                "match": frozen_hash == current_hash,
            }
        )

    frozen_hash_match_count = sum(record["match"] for record in frozen_hash_records)
    all_frozen_hashes_match = frozen_hash_match_count == len(reference_views)
    deterministic_origin = reference_schema == DETERMINISTIC_TARGET_SCHEMA
    if deterministic_origin and not all_frozen_hashes_match:
        raise RuntimeError("deterministic-origin frozen target crop hash differs")
    original_tensor_equivalence_verified = deterministic_origin and all_frozen_hashes_match
    original_tensor_equivalence_reason = (
        "The deterministic-origin frozen crop hashes match every deterministic recovery crop."
        if original_tensor_equivalence_verified
        else (
            "The legacy fit rendered CUDA reference targets without deterministic algorithms; "
            "its original tensors were not persisted, so raw crop-hash mismatches cannot be "
            "replaced by a claim of original tensor equality."
        )
    )

    reference_identity = {"schema": reference_schema, "metadata": reference_metadata}
    return {
        "reference": str(reference_path.resolve()),
        "reference_sha256": _sha256_file(reference_path),
        "reference_schema": reference_schema,
        "reference_deterministic_origin": deterministic_origin,
        "view_count": len(reference_metadata),
        "identity_sha256": _json_sha256(reference_identity),
        "all_metadata_match": True,
        "frozen_raw_crop_hash_match_count": frozen_hash_match_count,
        "frozen_raw_crop_hash_view_count": len(reference_views),
        "all_frozen_raw_crop_hashes_match": all_frozen_hashes_match,
        "frozen_raw_crop_hashes": frozen_hash_records,
        "original_tensor_equivalence_verified": original_tensor_equivalence_verified,
        "original_tensor_equivalence_reason": original_tensor_equivalence_reason,
        "deterministic_recovery_replay_verified": True,
        "deterministic_recovery_identity_sha256": _json_sha256(replayed_deterministic),
        "frozen_extrema_atol": TARGET_EXTREMA_ATOL,
        "max_frozen_extrema_delta": max(
            (record["delta"] for record in extrema_deltas), default=0.0
        ),
        "frozen_extrema_deltas": extrema_deltas,
    }


def _next_recovery_receipt(out: Path) -> Path:
    directory = out / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        candidate = directory / f"recovery_attempt_{attempt:03d}.json"
        if not candidate.exists():
            return candidate
        attempt += 1


def _resume_checkpoint(out: Path, requested: Path) -> tuple[Path, int]:
    checkpoint_directory = (out / "checkpoints").resolve(strict=True)
    checkpoint = requested.resolve(strict=True)
    try:
        checkpoint.relative_to(checkpoint_directory)
    except ValueError as exc:
        raise RuntimeError(
            "resume checkpoint must be inside the run checkpoints directory"
        ) from exc
    match = re.fullmatch(r"gaussians_step_(\d{6})\.ply", checkpoint.name)
    if match is None or not checkpoint.is_file():
        raise RuntimeError("resume checkpoint must be an existing gaussians_step_XXXXXX.ply")
    return checkpoint, int(match.group(1))


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _copy_file_new(source: Path, destination: Path) -> None:
    """Copy one immutable artifact without permitting destination replacement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(chunk)
    if _sha256_file(source) != _sha256_file(destination):
        raise RuntimeError(f"copied artifact hash differs: {destination}")


def _history_last_step(history: dict[str, Any], key: str) -> int:
    records = history.get(key)
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"parent training history has no {key} records")
    last = records[-1]
    if not isinstance(last, list) or not last:
        raise RuntimeError(f"parent training history has invalid {key} records")
    step = last[0]
    if isinstance(step, bool) or not isinstance(step, int):
        raise RuntimeError(f"parent training history has a non-integer {key} step")
    return step


def _require_selection_objective_contract(config: TrainConfig) -> None:
    opacity_reg = (
        0.01
        if config.density_strategy == "gsplat-mcmc" and config.opacity_reg is None
        else config.opacity_reg
    ) or 0.0
    scale_reg = (
        0.01
        if config.density_strategy == "gsplat-mcmc" and config.scale_reg is None
        else config.scale_reg
    ) or 0.0
    if not config.use_masks or config.random_background:
        raise RuntimeError("model selection requires masked training with a fixed black background")
    if opacity_reg != 0.0 or scale_reg != 0.0:
        raise RuntimeError("model selection does not support a hidden opacity/scale regularizer")
    if config.target_sh_degree != 3:
        raise RuntimeError("model selection requires the frozen degree-3 objective")


def _compact_training_objective_terms(
    color: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: TrainConfig,
) -> dict[str, float]:
    """Literal frozen Trainer loss for one view, with no random background or regularizer."""
    _require_selection_objective_contract(config)
    if color.shape != target.shape or color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError("color and target must be matching (H,W,3) tensors")
    if alpha.shape != color.shape[:2] or mask.shape != color.shape[:2]:
        raise ValueError("alpha and mask must match the image canvas")
    target = target.to(device=color.device, dtype=color.dtype)
    mask = mask.to(device=color.device, dtype=color.dtype).clamp(0, 1)
    alpha = alpha.to(device=color.device, dtype=color.dtype)
    target_for_loss = target * mask[..., None]
    weights = 0.1 + 0.9 * mask
    weighted_l1 = ((color - target_for_loss).abs() * weights[..., None]).mean()
    outside_alpha = (alpha * (1.0 - mask)).mean()
    l1_with_outside = weighted_l1 + config.outside_alpha_lambda * outside_alpha
    mask_alpha_l1 = (alpha - mask).abs().mean()
    crop_ssim = ssim(masked_crop(color, mask), masked_crop(target_for_loss, mask))
    objective = (
        (1.0 - config.ssim_lambda) * l1_with_outside
        + config.mask_alpha_lambda * mask_alpha_l1
        + config.ssim_lambda * (1.0 - crop_ssim)
    )
    result = {
        "weighted_rgb_l1": float(weighted_l1),
        "outside_alpha_mean": float(outside_alpha),
        "l1_with_outside_alpha": float(l1_with_outside),
        "mask_alpha_l1": float(mask_alpha_l1),
        "crop_ssim": float(crop_ssim),
        "objective": float(objective),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("model-selection objective contains a non-finite value")
    return result


def _choose_earliest_objective_tie(
    records: Sequence[dict[str, Any]],
    *,
    relative_tolerance: float = MODEL_SELECTION_RELATIVE_TIE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not records or not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("model selection requires candidates and a finite nonnegative tolerance")
    ordered = sorted(records, key=lambda record: int(record["global_step"]))
    steps = [int(record["global_step"]) for record in ordered]
    objectives = [float(record["objective"]) for record in ordered]
    if len(set(steps)) != len(steps) or not all(
        math.isfinite(value) and value >= 0.0 for value in objectives
    ):
        raise ValueError("candidate steps/objectives are invalid")
    minimum = min(objectives)
    maximum_eligible = minimum * (1.0 + relative_tolerance)
    eligible = [record for record in ordered if float(record["objective"]) <= maximum_eligible]
    selected = min(eligible, key=lambda record: int(record["global_step"]))
    return selected, {
        "primary": "equal_view_mean_compact_training_objective",
        "relative_tolerance": relative_tolerance,
        "minimum_objective": minimum,
        "maximum_eligible_objective": maximum_eligible,
        "eligible_global_steps": [int(record["global_step"]) for record in eligible],
        "tie_break": "earliest_global_step",
    }


def _material_plateau(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: int(record["global_step"]))
    transitions = []
    for previous, current in zip(ordered, ordered[1:]):
        previous_step = int(previous["global_step"])
        current_step = int(current["global_step"])
        if current_step - previous_step != 1_000:
            raise ValueError("material-plateau candidates must be spaced by 1,000 steps")
        previous_objective = float(previous["objective"])
        if not math.isfinite(previous_objective) or previous_objective <= 0.0:
            raise ValueError("material-plateau objectives must be finite and positive")
        objective_reduction = (
            previous_objective - float(current["objective"])
        ) / previous_objective
        psnr_gain = float(current["psnr_fg_db"]) - float(previous["psnr_fg_db"])
        material = (
            objective_reduction >= MATERIAL_OBJECTIVE_REDUCTION
            or psnr_gain >= MATERIAL_PSNR_GAIN_DB
        )
        transitions.append(
            {
                "from_step": previous_step,
                "to_step": current_step,
                "objective_reduction_fraction": objective_reduction,
                "psnr_fg_gain_db": psnr_gain,
                "material_improvement": material,
            }
        )
    trailing_nonmaterial = 0
    for transition in reversed(transitions):
        if transition["material_improvement"]:
            break
        trailing_nonmaterial += 1
    plateau = trailing_nonmaterial >= MATERIAL_PLATEAU_TRANSITIONS
    return {
        "status": "plateau" if plateau else "still_improving",
        "plateau": plateau,
        "required_consecutive_nonmaterial_transitions": MATERIAL_PLATEAU_TRANSITIONS,
        "objective_reduction_threshold_fraction": MATERIAL_OBJECTIVE_REDUCTION,
        "psnr_gain_threshold_db": MATERIAL_PSNR_GAIN_DB,
        "trailing_nonmaterial_transitions": trailing_nonmaterial,
        "transitions": transitions,
    }


def _theil_sen_psnr_slope(records: Sequence[dict[str, Any]]) -> float:
    slopes = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            delta_ksteps = (int(right["global_step"]) - int(left["global_step"])) / 1_000.0
            slopes.append((float(right["psnr_fg_db"]) - float(left["psnr_fg_db"])) / delta_ksteps)
    return float(statistics.median(slopes))


def _frozen_last_six_trend(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: int(record["global_step"]))
    if len(ordered) < 6:
        raise ValueError("the frozen trend rule requires six artifact checkpoints")
    window = ordered[-6:]
    if any(
        int(right["global_step"]) - int(left["global_step"]) != 1_000
        for left, right in zip(window, window[1:])
    ):
        raise ValueError("frozen trend candidates must be spaced by 1,000 steps")
    slope = _theil_sen_psnr_slope(window)
    psnr_values = [float(record["psnr_fg_db"]) for record in window]
    recent_psnr_gain = statistics.median(psnr_values[3:]) - statistics.median(psnr_values[:3])
    view_names = [str(record["view_name"]) for record in window[0]["per_view"]]
    if not view_names or len(set(view_names)) != len(view_names):
        raise ValueError("candidate per-view identities must be unique and non-empty")
    per_view_improvements = []
    for view_name in view_names:
        early_values = []
        late_values = []
        for position, record in enumerate(window):
            by_view = {str(item["view_name"]): item for item in record["per_view"]}
            if len(by_view) != len(view_names) or set(by_view) != set(view_names):
                raise ValueError("candidate per-view identities differ across the trend window")
            destination = early_values if position < 3 else late_values
            destination.append(float(by_view[view_name]["objective"]))
        early = float(statistics.median(early_values))
        late = float(statistics.median(late_values))
        if not math.isfinite(early) or early <= 0.0:
            raise ValueError("per-view trend objectives must be finite and positive")
        per_view_improvements.append(
            {
                "view_name": view_name,
                "relative_objective_improvement_percent": 100.0 * (early - late) / early,
            }
        )
    median_view_improvement = float(
        statistics.median(
            item["relative_objective_improvement_percent"] for item in per_view_improvements
        )
    )
    fraction_over_one = sum(
        item["relative_objective_improvement_percent"] > 1.0 for item in per_view_improvements
    ) / len(per_view_improvements)
    final_below_best = max(psnr_values) - psnr_values[-1]
    objective_values = [float(record["objective"]) for record in window]
    if not all(math.isfinite(value) and value > 0.0 for value in objective_values):
        raise ValueError("frozen trend objectives must be finite and positive")
    best_objective = min(objective_values)
    endpoint_objective_regression = (objective_values[-1] - best_objective) / best_objective
    plateau_conditions = {
        "abs_theil_sen_slope_at_most_0_01_db_per_1k": abs(slope) <= 0.01,
        "last3_minus_previous3_median_at_most_0_05_db": recent_psnr_gain <= 0.05,
        "median_per_view_objective_improvement_at_most_0_5_percent": (
            median_view_improvement <= 0.5
        ),
        "fraction_views_improving_over_1_percent_at_most_0_2": fraction_over_one <= 0.2,
    }
    plateau = all(plateau_conditions.values())
    regression = (
        final_below_best > 0.10
        or slope < -0.01
        or endpoint_objective_regression > MATERIAL_OBJECTIVE_REDUCTION
    )
    status = "regression" if regression else ("plateau" if plateau else "still_improving")
    return {
        "status": status,
        "window_global_steps": [int(record["global_step"]) for record in window],
        "theil_sen_psnr_slope_db_per_1k": slope,
        "last3_minus_previous3_median_psnr_db": recent_psnr_gain,
        "median_per_view_relative_objective_improvement_percent": median_view_improvement,
        "fraction_views_improving_over_one_percent": fraction_over_one,
        "final_below_best_last_six_psnr_db": final_below_best,
        "endpoint_objective_regression_fraction": endpoint_objective_regression,
        "endpoint_objective_regression_threshold_fraction": MATERIAL_OBJECTIVE_REDUCTION,
        "plateau_conditions": plateau_conditions,
        "plateau": plateau,
        "regression": regression,
        "per_view": per_view_improvements,
    }


@torch.no_grad()
def _score_compact_candidate(
    path: Path,
    *,
    global_step: int,
    kind: str,
    out: Path,
    scene: SceneData,
    trainer_config: TrainConfig,
    renderer: Any,
    expected_n: int,
) -> dict[str, Any]:
    sha256 = _sha256_file(path)
    device = torch.device(trainer_config.device)
    gaussians = Gaussians3D.load_ply(path).to(device)
    if gaussians.n != expected_n or gaussians.sh_degree != 3:
        raise RuntimeError(f"invalid fixed-topology candidate: {path}")
    if scene.masks is None or scene.view_names is None:
        raise RuntimeError("model selection requires named masked compact views")
    per_view = []
    term_names = None
    for index in scene.training_views:
        target = scene.images[index].to(device)
        mask = scene.masks[index].to(device)
        output = renderer.render(
            gaussians,
            scene.cameras[index].to(device),
            background=None,
            sh_degree=3,
        )
        if not all(
            bool(torch.isfinite(getattr(output, field)).all())
            for field in ("color", "alpha", "depth")
        ):
            raise RuntimeError(f"non-finite candidate render at step {global_step}")
        terms = _compact_training_objective_terms(
            output.color, output.alpha, target, mask, trainer_config
        )
        term_names = tuple(terms) if term_names is None else term_names
        per_view.append(
            {
                "view_index": int(index),
                "view_name": scene.view_names[index],
                **terms,
                "psnr_fg_db": masked_psnr(output.color.clamp(0, 1), target.clamp(0, 1), mask),
            }
        )
        del output, target, mask
    if _sha256_file(path) != sha256:
        raise RuntimeError(f"candidate artifact changed while it was evaluated: {path}")
    if term_names is None:
        raise RuntimeError("model selection found no compact training views")
    means = {
        name: sum(float(record[name]) for record in per_view) / len(per_view)
        for name in (*term_names, "psnr_fg_db")
    }
    return {
        "global_step": global_step,
        "kind": kind,
        "artifact": str(path.relative_to(out)),
        "sha256": sha256,
        "bytes": path.stat().st_size,
        "n_gaussians": gaussians.n,
        "sh_degree": gaussians.sh_degree,
        **means,
        "per_view": per_view,
    }


def _evaluate_polish_candidates(
    out: Path,
    scene: SceneData,
    trainer_config: TrainConfig,
    *,
    expected_n: int,
    baseline_step: int = POLISH_PARENT_ITERATIONS,
    final_step: int = POLISH_SCHEDULE_ITERATIONS,
) -> tuple[Path, dict[str, Any]]:
    _require_selection_objective_contract(trainer_config)
    if baseline_step < 0 or final_step <= baseline_step:
        raise ValueError("model selection requires an increasing nonnegative step interval")
    if baseline_step % 1_000 != 0 or final_step % 1_000 != 0:
        raise ValueError("model-selection endpoints must be spaced on 1,000-step boundaries")
    specs = [
        (baseline_step, f"parent_{baseline_step // 1_000}k_baseline", out / "gaussians_init.ply"),
        *[
            (
                step,
                "polish_checkpoint",
                out / "checkpoints" / f"gaussians_step_{step:06d}.ply",
            )
            for step in range(
                baseline_step + 1_000,
                final_step + 1,
                1_000,
            )
        ],
    ]
    missing = [str(path) for _, _, path in specs if not path.is_file()]
    if missing:
        raise RuntimeError(f"model selection is missing candidate artifacts: {missing}")
    from rtgs.render.base import get_rasterizer

    device = torch.device(trainer_config.device)
    renderer = get_rasterizer(
        trainer_config.rasterizer,
        device=device,
        packed=trainer_config.packed,
        antialiased=trainer_config.antialiased,
        sh_color_activation=trainer_config.sh_color_activation,
        sh_smu1_mu=trainer_config.sh_smu1_mu,
        kernel_support_mode=trainer_config.kernel_support_mode,
        visibility_margin_sigma=trainer_config.visibility_margin_sigma,
    )
    records = []
    for ordinal, (step, kind, path) in enumerate(specs, start=1):
        print(f"[select] scoring {ordinal:02d}/{len(specs):02d} step={step}", flush=True)
        records.append(
            _score_compact_candidate(
                path,
                global_step=step,
                kind=kind,
                out=out,
                scene=scene,
                trainer_config=trainer_config,
                renderer=renderer,
                expected_n=expected_n,
            )
        )
    selected, rule = _choose_earliest_objective_tie(records)
    material = _material_plateau(records)
    frozen = _frozen_last_six_trend(records)
    if frozen["status"] == "regression":
        joint_status = "regression"
    elif material["plateau"] and frozen["plateau"]:
        joint_status = "plateau"
    else:
        joint_status = "still_improving"
    receipt = {
        "schema": MODEL_SELECTION_SCHEMA,
        "created_utc": _utc_now(),
        "written_before_gaussians_final": True,
        "candidate_set": (
            f"child gaussians_init at {baseline_step // 1_000}k plus reloaded child "
            f"checkpoints {baseline_step // 1_000 + 1}k..{final_step // 1_000}k"
        ),
        "equal_view_weighting": True,
        "full_native_compact_targets": True,
        "active_sh_degree": 3,
        "objective": {
            "formula": (
                "(1-ssim_lambda)*(weighted_rgb_l1 + "
                "outside_alpha_lambda*outside_alpha_mean) + "
                "mask_alpha_lambda*mask_alpha_l1 + ssim_lambda*(1-crop_ssim)"
            ),
            "ssim_lambda": trainer_config.ssim_lambda,
            "outside_alpha_lambda": trainer_config.outside_alpha_lambda,
            "mask_alpha_lambda": trainer_config.mask_alpha_lambda,
            "random_background": False,
            "opacity_regularizer": 0.0,
            "scale_regularizer": 0.0,
        },
        "selection_rule": rule,
        "selected": {
            "global_step": int(selected["global_step"]),
            "artifact": selected["artifact"],
            "sha256": selected["sha256"],
            "objective": selected["objective"],
            "psnr_fg_db": selected["psnr_fg_db"],
        },
        "convergence": {
            "joint_status": joint_status,
            "joint_plateau_requires_both_rules": True,
            "material_five_transition_rule": material,
            "frozen_last_six_rule": frozen,
        },
        "finalization": {
            "method": "byte_copy_selected_candidate_to_gaussians_final.ply",
            "expected_final_sha256": selected["sha256"],
        },
        "candidates": records,
    }
    return out / selected["artifact"], receipt


def _polish_train_config(parent: TrainConfig) -> TrainConfig:
    """Return the frozen fixed-topology 30k-to-40k polish schedule."""

    return dataclasses.replace(
        parent,
        iterations=POLISH_ITERATIONS,
        lr_means=parent.lr_means * 0.01,
        lr_quats=parent.lr_quats * POLISH_OTHER_LR_FACTOR,
        lr_scales=parent.lr_scales * POLISH_OTHER_LR_FACTOR,
        lr_opacity=parent.lr_opacity * POLISH_OTHER_LR_FACTOR,
        lr_sh=parent.lr_sh * POLISH_OTHER_LR_FACTOR,
        lr_sh_rest=parent.lr_sh_rest * POLISH_OTHER_LR_FACTOR,
        densify=False,
        eval_every=1_000,
        target_sh_degree=3,
        sh_degree_interval=1_000,
        seed=POLISH_SEED,
        iteration_offset=POLISH_PARENT_ITERATIONS,
        schedule_iterations=POLISH_SCHEDULE_ITERATIONS,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )


def _tail_train_config(parent: TrainConfig) -> TrainConfig:
    """Continue the frozen fixed-topology schedule from 40k through 50k."""

    return dataclasses.replace(
        parent,
        iterations=TAIL_ITERATIONS,
        densify=False,
        eval_every=1_000,
        target_sh_degree=3,
        sh_degree_interval=1_000,
        seed=TAIL_SEED,
        iteration_offset=TAIL_PARENT_ITERATIONS,
        schedule_iterations=TAIL_SCHEDULE_ITERATIONS,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )


def _cooldown_train_config(parent: TrainConfig) -> TrainConfig:
    """Run a lower-rate fixed-topology segment from 50k through 60k."""

    return dataclasses.replace(
        parent,
        iterations=COOLDOWN_ITERATIONS,
        lr_means=parent.lr_means * COOLDOWN_LR_FACTOR,
        lr_quats=parent.lr_quats * COOLDOWN_LR_FACTOR,
        lr_scales=parent.lr_scales * COOLDOWN_LR_FACTOR,
        lr_opacity=parent.lr_opacity * COOLDOWN_LR_FACTOR,
        lr_sh=parent.lr_sh * COOLDOWN_LR_FACTOR,
        lr_sh_rest=parent.lr_sh_rest * COOLDOWN_LR_FACTOR,
        densify=False,
        eval_every=1_000,
        target_sh_degree=3,
        sh_degree_interval=1_000,
        seed=COOLDOWN_SEED,
        iteration_offset=COOLDOWN_PARENT_ITERATIONS,
        schedule_iterations=COOLDOWN_SCHEDULE_ITERATIONS,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )


def _settle_train_config(parent: TrainConfig) -> TrainConfig:
    """Run a second lower-rate fixed-topology segment from 60k through 70k."""

    return dataclasses.replace(
        parent,
        iterations=SETTLE_ITERATIONS,
        lr_means=parent.lr_means * SETTLE_LR_FACTOR,
        lr_quats=parent.lr_quats * SETTLE_LR_FACTOR,
        lr_scales=parent.lr_scales * SETTLE_LR_FACTOR,
        lr_opacity=parent.lr_opacity * SETTLE_LR_FACTOR,
        lr_sh=parent.lr_sh * SETTLE_LR_FACTOR,
        lr_sh_rest=parent.lr_sh_rest * SETTLE_LR_FACTOR,
        densify=False,
        eval_every=1_000,
        target_sh_degree=3,
        sh_degree_interval=1_000,
        seed=SETTLE_SEED,
        iteration_offset=SETTLE_PARENT_ITERATIONS,
        schedule_iterations=SETTLE_SCHEDULE_ITERATIONS,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )


def _opacity_entry_plan(init: Gaussians3D, epsilon: float) -> dict[str, Any]:
    source = init.opacity.detach().cpu().float().contiguous()
    if source.ndim != 1 or source.shape[0] != init.n:
        raise RuntimeError("parent PLY opacity shape differs from its Gaussian count")
    if not bool(torch.isfinite(source).all()) or bool(((source < 0.0) | (source > 1.0)).any()):
        raise RuntimeError("parent PLY opacities must be finite values in [0, 1]")
    expected = source.clamp(epsilon, 1.0 - epsilon)
    default_epsilon = 1e-4
    return {
        "source_rows": init.n,
        "source_sha256": _tensor_sha256(source),
        "source_min": float(source.min()),
        "source_max": float(source.max()),
        "historical_default_epsilon": default_epsilon,
        "polish_epsilon": epsilon,
        "above_historical_upper_count": int((source > 1.0 - default_epsilon).sum()),
        "below_historical_lower_count": int((source < default_epsilon).sum()),
        "historical_default_would_change_count": int(
            ((source < default_epsilon) | (source > 1.0 - default_epsilon)).sum()
        ),
        "polish_will_change_count": int(((source < epsilon) | (source > 1.0 - epsilon)).sum()),
        "polish_preserved_count": int(((source >= epsilon) & (source <= 1.0 - epsilon)).sum()),
        "expected_entry_sha256": _tensor_sha256(expected),
    }


def _opacity_entry_verifier(
    init: Gaussians3D,
    epsilon: float,
    record: dict[str, Any],
) -> Callable[[Gaussians3D], None]:
    source = init.opacity.detach().cpu().float().contiguous()
    expected = source.clamp(epsilon, 1.0 - epsilon)
    preserved = (source >= epsilon) & (source <= 1.0 - epsilon)

    def verify(snapshot: Gaussians3D) -> None:
        observed = snapshot.opacity.detach().cpu().float().contiguous()
        if observed.shape != expected.shape:
            raise RuntimeError("trainer-entry opacity shape differs from the parent PLY")
        if not torch.allclose(observed, expected, rtol=1e-6, atol=1e-7):
            error = float((observed - expected).abs().max())
            raise RuntimeError(f"trainer-entry opacity clamp differs (max error {error})")
        preserved_error = (
            float((observed[preserved] - source[preserved]).abs().max())
            if bool(preserved.any())
            else 0.0
        )
        observed_sha256 = _tensor_sha256(observed)
        expected_sha256 = _tensor_sha256(expected)
        record.update(
            {
                "verified_before_optimizer_construction": True,
                "epsilon": epsilon,
                "observed_sha256": observed_sha256,
                "expected_sha256": expected_sha256,
                "expected_hash_match": observed_sha256 == expected_sha256,
                "preserved_count": int(preserved.sum()),
                "preserved_max_abs_error": preserved_error,
            }
        )

    return verify


def _polish_parent_preflight(parent: Path, out: Path) -> dict[str, Any]:
    """Bind the completed non-exact 4k-to-30k recovery parent without source RGB."""

    parent = parent.resolve()
    out = out.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"polish requires an existing parent run: {parent}")
    if out == parent:
        raise RuntimeError("polish output must not be the parent run")
    if out.parent != parent.parent:
        raise RuntimeError("polish output must be a sibling of the parent run")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite polish output: {out}")
    missing = [name for name in POLISH_PARENT_ARTIFACTS if not (parent / name).is_file()]
    if missing:
        raise RuntimeError(f"polish parent is missing completed-fit artifacts: {missing}")

    config = _load_json_object(parent / "config.json", label="parent config")
    provenance = _load_json_object(parent / "provenance.json", label="parent provenance")
    targets = _load_json_object(parent / "compact_targets.json", label="parent targets")
    history = _load_json_object(parent / "training_history.json", label="parent history")
    fit_receipt = _load_json_object(parent / "fit_complete.json", label="parent fit receipt")
    trainer_config = _load_train_config(parent / "training_config.json")

    if config.get("schema") != "rtgs.full_compact_reconstruction.config.v1":
        raise RuntimeError("polish parent has an unsupported config schema")
    if config.get("fit_mode") != "all" or config.get("smoke") is not False:
        raise RuntimeError("polish requires the full all-view non-smoke parent")
    if config.get("iterations") != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("polish requires a frozen 30,000-step parent config")
    if config.get("seed") != 0:
        raise RuntimeError("polish requires the frozen parent seed 0")
    if tuple(config.get("view_names", ())) != EXPECTED_VIEW_NAMES:
        raise RuntimeError("polish parent view order differs from the frozen benchmark")
    if tuple(config.get("fit_indices", ())) != tuple(range(len(EXPECTED_VIEW_NAMES))):
        raise RuntimeError("polish parent did not fit every compact view")

    if trainer_config.iterations != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("parent training config is not a 30,000-step invocation")
    if trainer_config.iteration_offset != 0:
        raise RuntimeError("parent frozen training config must begin at global step zero")
    if _resolve_schedule_iterations(trainer_config) != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("parent frozen training schedule does not end at step 30,000")
    if trainer_config.target_sh_degree != 3 or _resolve_sh_interval(trainer_config) != 1_000:
        raise RuntimeError("parent training config does not use the frozen degree-3 SH schedule")
    if not math.isclose(
        float(trainer_config.means_lr_final_factor), 0.01, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("parent means LR does not have the frozen 1% terminal factor")
    if not math.isclose(
        float(trainer_config.opacity_logit_epsilon), 1e-4, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("parent opacity entry clamp differs from the historical default")
    for config_key, trainer_value in (
        ("iterations", trainer_config.iterations),
        ("eval_every", trainer_config.eval_every),
        ("device", trainer_config.device),
        ("densify", trainer_config.densify),
        ("density_strategy", trainer_config.density_strategy),
        ("seed", trainer_config.seed),
    ):
        if config.get(config_key) != trainer_value:
            raise RuntimeError(f"parent config/training config disagree on {config_key}")
    if trainer_config.eval_every != 1_000:
        raise RuntimeError("polish requires the frozen 1,000-step checkpoint interval")
    if int(config.get("max_gaussians", -1)) != trainer_config.density.max_gaussians:
        raise RuntimeError("parent config/training config disagree on max_gaussians")

    if provenance.get("schema") != "rtgs.full_compact_reconstruction.provenance.v1":
        raise RuntimeError("polish parent has an unsupported provenance schema")
    _provenance_identity(provenance)
    provenance_views = provenance.get("views")
    if (
        not isinstance(provenance_views, list)
        or tuple(record.get("view_id") for record in provenance_views if isinstance(record, dict))
        != EXPECTED_VIEW_NAMES
    ):
        raise RuntimeError("parent provenance view order differs from the frozen benchmark")
    if tuple(provenance.get("fit_indices", ())) != tuple(config["fit_indices"]):
        raise RuntimeError("parent provenance/config fit indices differ")

    if targets.get("schema") != LEGACY_NONDETERMINISTIC_TARGET_SCHEMA:
        raise RuntimeError("polish requires the exact legacy-v1 parent target lineage")
    target_views = targets.get("views")
    if not isinstance(target_views, list) or not all(
        isinstance(record, dict) for record in target_views
    ):
        raise RuntimeError("polish target receipt has no valid view list")
    target_order = [(record.get("global_index"), record.get("view_id")) for record in target_views]
    if target_order != list(enumerate(EXPECTED_VIEW_NAMES)):
        raise RuntimeError("polish target receipt differs from the frozen all-view order")

    final_path = parent / "gaussians_final.ply"
    final_sha256 = _sha256_file(final_path)
    if fit_receipt.get("schema") != "rtgs.full_compact_reconstruction.fit_complete.v1":
        raise RuntimeError("polish parent has an unsupported fit receipt schema")
    if fit_receipt.get("final_ply_sha256") != final_sha256:
        raise RuntimeError("parent final PLY differs from its completed-fit receipt")
    n_final = fit_receipt.get("n_final_gaussians")
    if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final <= 0:
        raise RuntimeError("parent fit receipt has an invalid final Gaussian count")
    if fit_receipt.get("resume_exact") is not False:
        raise RuntimeError("polish requires the frozen non-exact recovery parent lineage")

    fit_recovery = fit_receipt.get("recovery")
    if not isinstance(fit_recovery, dict):
        raise RuntimeError("polish parent fit receipt has no recovery lineage")
    recovery_relative = fit_recovery.get("receipt")
    if not isinstance(recovery_relative, str):
        raise RuntimeError("polish parent recovery receipt path is invalid")
    recovery_path = (parent / recovery_relative).resolve(strict=True)
    recovery_directory = (parent / "recovery").resolve(strict=True)
    try:
        recovery_path.relative_to(recovery_directory)
    except ValueError as exc:
        raise RuntimeError("polish parent recovery receipt escapes its recovery directory") from exc
    if not re.fullmatch(r"recovery_attempt_\d{3}\.json", recovery_path.name):
        raise RuntimeError("polish parent recovery receipt name is invalid")
    recovery_sha256 = _sha256_file(recovery_path)
    if fit_recovery.get("receipt_sha256") != recovery_sha256:
        raise RuntimeError("polish parent recovery receipt hash differs from fit completion")
    recovery = _load_json_object(recovery_path, label="parent recovery receipt")
    if recovery.get("schema") != "rtgs.full_compact_reconstruction.recovery_attempt.v2":
        raise RuntimeError("polish parent has an unsupported recovery receipt schema")
    if recovery.get("written_before_recovery_training") is not True:
        raise RuntimeError("parent recovery receipt was not frozen before recovery training")
    if recovery.get("resume_exact") is not False:
        raise RuntimeError("parent recovery receipt does not declare its non-exact lineage")
    expected_recovery_fields = {
        "resume_step": POLISH_PARENT_RESUME_STEP,
        "first_recovered_step": POLISH_PARENT_RESUME_STEP + 1,
        "last_recovered_step": POLISH_PARENT_ITERATIONS,
        "schedule_iterations": POLISH_PARENT_ITERATIONS,
        "remaining_iterations": POLISH_PARENT_ITERATIONS - POLISH_PARENT_RESUME_STEP,
    }
    for key, expected in expected_recovery_fields.items():
        if recovery.get(key) != expected:
            raise RuntimeError(f"parent recovery receipt has unexpected {key}")

    recovery_training_payload = recovery.get("recovery_training_config")
    if not isinstance(recovery_training_payload, dict):
        raise RuntimeError("parent recovery receipt has no recovery training config")
    recovery_trainer_config = _train_config_from_payload(
        recovery_training_payload,
        label="parent recovery training config",
    )
    if recovery_trainer_config.iterations != 26_000:
        raise RuntimeError("parent recovery training config is not a 26,000-step segment")
    if recovery_trainer_config.iteration_offset != POLISH_PARENT_RESUME_STEP:
        raise RuntimeError("parent recovery training config does not start at global step 4,000")
    if _resolve_schedule_iterations(recovery_trainer_config) != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("parent recovery training schedule does not terminate at step 30,000")
    for name in (
        "lr_means",
        "lr_quats",
        "lr_scales",
        "lr_opacity",
        "lr_sh",
        "lr_sh_rest",
        "target_sh_degree",
        "sh_degree_interval",
        "density_strategy",
        "seed",
        "means_lr_final_factor",
        "opacity_logit_epsilon",
    ):
        if getattr(recovery_trainer_config, name) != getattr(trainer_config, name):
            raise RuntimeError(f"parent recovery changed frozen trainer field {name}")

    if history.get("iteration_offset") != POLISH_PARENT_RESUME_STEP:
        raise RuntimeError("parent history does not start at recovery step 4,000")
    if history.get("segment_iterations") != 26_000:
        raise RuntimeError("parent history is not the complete 26,000-step recovery segment")
    if history.get("schedule_iterations") != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("parent history has a different global schedule length")
    losses = history.get("loss")
    if not isinstance(losses, list) or len(losses) != 26_000:
        raise RuntimeError("parent history does not contain all 26,000 recovery losses")
    for key in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        if _history_last_step(history, key) != POLISH_PARENT_ITERATIONS:
            raise RuntimeError(f"parent {key} history does not end at global step 30,000")
    if history["n_gaussians"][-1][1] != n_final:
        raise RuntimeError("parent history and fit receipt final Gaussian counts differ")

    frozen_artifacts = recovery.get("frozen_artifacts")
    for name in ("config.json", "provenance.json", "training_config.json", "compact_targets.json"):
        record = frozen_artifacts.get(name) if isinstance(frozen_artifacts, dict) else None
        if not isinstance(record, dict) or record.get("sha256") != _sha256_file(parent / name):
            raise RuntimeError(f"parent recovery receipt no longer binds {name}")
    recovery_target = recovery.get("target_replay")
    if not isinstance(recovery_target, dict):
        raise RuntimeError("parent recovery receipt has no target replay")
    if recovery_target.get("reference_sha256") != _sha256_file(parent / "compact_targets.json"):
        raise RuntimeError("parent recovery target replay no longer binds compact_targets.json")
    if recovery_target.get("deterministic_recovery_replay_verified") is not True:
        raise RuntimeError("parent recovery receipt did not verify deterministic target replay")

    return {
        "parent": parent,
        "out": out,
        "config": config,
        "provenance": provenance,
        "targets": targets,
        "history": history,
        "fit_receipt": fit_receipt,
        "trainer_config": trainer_config,
        "recovery_path": recovery_path,
        "recovery_sha256": recovery_sha256,
        "recovery": recovery,
        "recovery_trainer_config": recovery_trainer_config,
        "final_path": final_path,
        "final_sha256": final_sha256,
        "n_final": n_final,
        "artifacts": {
            name: {
                "path": str((parent / name).resolve()),
                "sha256": _sha256_file(parent / name),
                "bytes": (parent / name).stat().st_size,
            }
            for name in POLISH_PARENT_ARTIFACTS
        },
    }


def _tail_parent_preflight(parent: Path, out: Path) -> dict[str, Any]:
    """Bind an immutable selected 40k polish child for one more fixed-topology segment."""

    parent = parent.resolve()
    out = out.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"tail requires an existing 40k polish parent: {parent}")
    if out == parent:
        raise RuntimeError("tail output must not be the parent run")
    if out.parent != parent.parent:
        raise RuntimeError("tail output must be a sibling of the parent run")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite tail output: {out}")
    missing = [name for name in TAIL_PARENT_ARTIFACTS if not (parent / name).is_file()]
    checkpoint = parent / "checkpoints" / "gaussians_step_040000.ply"
    if missing or not checkpoint.is_file():
        missing_checkpoint = [] if checkpoint.is_file() else [str(checkpoint.relative_to(parent))]
        raise RuntimeError(
            f"tail parent is missing completed-polish artifacts: {[*missing, *missing_checkpoint]}"
        )

    config = _load_json_object(parent / "config.json", label="tail parent config")
    provenance = _load_json_object(parent / "provenance.json", label="tail parent provenance")
    targets = _load_json_object(parent / "compact_targets.json", label="tail parent targets")
    start = _load_json_object(parent / "polish_start.json", label="tail parent polish receipt")
    history = _load_json_object(parent / "training_history.json", label="tail parent history")
    selection = _load_json_object(parent / "model_selection.json", label="tail parent selection")
    fit_receipt = _load_json_object(parent / "fit_complete.json", label="tail parent fit receipt")
    trainer_config = _load_train_config(parent / "training_config.json")

    if config.get("schema") != "rtgs.full_compact_reconstruction.config.v1":
        raise RuntimeError("tail parent has an unsupported config schema")
    if config.get("fit_mode") != "all" or config.get("smoke") is not False:
        raise RuntimeError("tail requires the full all-view non-smoke parent")
    if config.get("iterations") != POLISH_PARENT_ITERATIONS or config.get("seed") != 0:
        raise RuntimeError("tail parent does not preserve the frozen 30k fit config")
    if tuple(config.get("view_names", ())) != EXPECTED_VIEW_NAMES or tuple(
        config.get("fit_indices", ())
    ) != tuple(range(len(EXPECTED_VIEW_NAMES))):
        raise RuntimeError("tail parent does not preserve the frozen all-view order")

    if provenance.get("schema") != "rtgs.full_compact_reconstruction.provenance.v1":
        raise RuntimeError("tail parent has an unsupported provenance schema")
    _provenance_identity(provenance)
    if (
        targets.get("schema") != DETERMINISTIC_TARGET_SCHEMA
        or targets.get("deterministic_algorithms") is not True
    ):
        raise RuntimeError("tail requires deterministic-v2 compact targets")
    target_views = targets.get("views")
    if not isinstance(target_views, list) or [
        (record.get("global_index"), record.get("view_id"))
        for record in target_views
        if isinstance(record, dict)
    ] != list(enumerate(EXPECTED_VIEW_NAMES)):
        raise RuntimeError("tail target receipt differs from the frozen all-view order")

    if trainer_config.iterations != POLISH_ITERATIONS:
        raise RuntimeError("tail parent training config is not a 10,000-step segment")
    if trainer_config.iteration_offset != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("tail parent training config does not start at 30,000")
    if _resolve_schedule_iterations(trainer_config) != TAIL_PARENT_ITERATIONS:
        raise RuntimeError("tail parent training schedule does not terminate at 40,000")
    if trainer_config.densify or trainer_config.eval_every != 1_000:
        raise RuntimeError("tail parent is not the frozen fixed-topology 1k-checkpoint segment")
    if trainer_config.target_sh_degree != 3 or _resolve_sh_interval(trainer_config) != 1_000:
        raise RuntimeError("tail parent does not use degree-3 SH throughout")
    if trainer_config.seed != POLISH_SEED:
        raise RuntimeError("tail parent does not use the frozen polish seed")
    if not math.isclose(trainer_config.means_lr_final_factor, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("tail parent means LR is not constant")
    if not math.isclose(trainer_config.opacity_logit_epsilon, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("tail parent opacity entry epsilon differs")

    if start.get("schema") != "rtgs.full_compact_reconstruction.polish_start.v1":
        raise RuntimeError("tail parent has an unsupported polish-start receipt")
    if start.get("written_before_polish_training") is not True:
        raise RuntimeError("tail parent polish receipt was not frozen before training")
    if start.get("continuation_exact") is not False:
        raise RuntimeError("tail parent polish receipt must declare non-exact continuation")
    expected_steps = {
        "parent_last": POLISH_PARENT_ITERATIONS,
        "first_polish": POLISH_PARENT_ITERATIONS + 1,
        "last_polish": TAIL_PARENT_ITERATIONS,
        "segment_iterations": POLISH_ITERATIONS,
        "schedule_iterations": TAIL_PARENT_ITERATIONS,
    }
    if start.get("global_steps") != expected_steps:
        raise RuntimeError("tail parent polish receipt has unexpected global steps")
    child_targets = start.get("child_target_receipt")
    if not isinstance(child_targets, dict) or child_targets.get("sha256") != _sha256_file(
        parent / "compact_targets.json"
    ):
        raise RuntimeError("tail parent polish receipt no longer binds compact targets")

    final_path = parent / "gaussians_final.ply"
    final_sha256 = _sha256_file(final_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if fit_receipt.get("schema") != "rtgs.full_compact_reconstruction.fit_complete.v1":
        raise RuntimeError("tail parent has an unsupported fit receipt schema")
    if fit_receipt.get("fit_kind") != "fixed_topology_polish":
        raise RuntimeError("tail requires a completed fixed-topology polish parent")
    if (
        fit_receipt.get("continuation_exact") is not False
        or fit_receipt.get("fixed_topology") is not True
    ):
        raise RuntimeError("tail parent fit receipt has invalid continuation semantics")
    n_final = fit_receipt.get("n_final_gaussians")
    if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final <= 0:
        raise RuntimeError("tail parent fit receipt has an invalid Gaussian count")
    if fit_receipt.get("source_n_gaussians") != n_final:
        raise RuntimeError("tail parent fixed topology changed the Gaussian count")
    if fit_receipt.get("final_ply_sha256") != final_sha256:
        raise RuntimeError("tail parent final PLY differs from its fit receipt")
    polish_binding = fit_receipt.get("polish")
    if not isinstance(polish_binding, dict) or polish_binding.get("receipt") != "polish_start.json":
        raise RuntimeError("tail parent fit receipt has no valid polish-start binding")
    if polish_binding.get("receipt_sha256") != _sha256_file(parent / "polish_start.json"):
        raise RuntimeError("tail parent polish-start receipt hash differs")

    if selection.get("schema") != MODEL_SELECTION_SCHEMA:
        raise RuntimeError("tail parent has an unsupported model-selection schema")
    if selection.get("written_before_gaussians_final") is not True:
        raise RuntimeError("tail parent selection was not frozen before finalization")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("tail parent selection has no selected candidate")
    if selected.get("global_step") != TAIL_PARENT_ITERATIONS:
        raise RuntimeError("tail requires the selected 40,000-step parent candidate")
    if selected.get("artifact") != "checkpoints/gaussians_step_040000.ply":
        raise RuntimeError("tail parent selected artifact is not the 40,000-step checkpoint")
    if selected.get("sha256") != final_sha256 or checkpoint_sha256 != final_sha256:
        raise RuntimeError("tail parent final PLY differs from the selected 40k checkpoint")
    convergence = selection.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("joint_status") != "still_improving":
        raise RuntimeError("tail is only valid for a 40k parent classified still_improving")
    selection_binding = fit_receipt.get("model_selection")
    if not isinstance(selection_binding, dict):
        raise RuntimeError("tail parent fit receipt has no model-selection binding")
    if selection_binding.get("receipt") != "model_selection.json" or selection_binding.get(
        "receipt_sha256"
    ) != _sha256_file(parent / "model_selection.json"):
        raise RuntimeError("tail parent model-selection receipt hash differs")
    if selection_binding.get("selected_global_step") != TAIL_PARENT_ITERATIONS or (
        selection_binding.get("selected_candidate_sha256") != final_sha256
    ):
        raise RuntimeError("tail parent fit receipt does not bind the selected 40k candidate")
    if selection_binding.get("joint_convergence_status") != "still_improving":
        raise RuntimeError("tail parent fit receipt does not bind its convergence status")

    if history.get("iteration_offset") != POLISH_PARENT_ITERATIONS:
        raise RuntimeError("tail parent history does not start at 30,000")
    if history.get("segment_iterations") != POLISH_ITERATIONS:
        raise RuntimeError("tail parent history is not the complete 10,000-step segment")
    if history.get("schedule_iterations") != TAIL_PARENT_ITERATIONS:
        raise RuntimeError("tail parent history does not terminate at 40,000")
    losses = history.get("loss")
    if not isinstance(losses, list) or len(losses) != POLISH_ITERATIONS:
        raise RuntimeError("tail parent history does not contain all 10,000 polish losses")
    for key in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        if _history_last_step(history, key) != TAIL_PARENT_ITERATIONS:
            raise RuntimeError(f"tail parent {key} history does not end at global step 40,000")
    if history["n_gaussians"][-1][1] != n_final:
        raise RuntimeError("tail parent history and fit receipt Gaussian counts differ")

    artifact_paths = {name: parent / name for name in TAIL_PARENT_ARTIFACTS}
    artifact_paths["checkpoints/gaussians_step_040000.ply"] = checkpoint
    return {
        "parent": parent,
        "out": out,
        "config": config,
        "provenance": provenance,
        "targets": targets,
        "polish_start": start,
        "history": history,
        "selection": selection,
        "fit_receipt": fit_receipt,
        "trainer_config": trainer_config,
        "final_path": final_path,
        "final_sha256": final_sha256,
        "n_final": n_final,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }


def _cooldown_parent_preflight(parent: Path, out: Path) -> dict[str, Any]:
    """Bind an immutable selected 50k tail child for a lower-rate cooldown segment."""

    parent = parent.resolve()
    out = out.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"cooldown requires an existing 50k tail parent: {parent}")
    if out == parent:
        raise RuntimeError("cooldown output must not be the parent run")
    if out.parent != parent.parent:
        raise RuntimeError("cooldown output must be a sibling of the parent run")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite cooldown output: {out}")
    missing = [name for name in COOLDOWN_PARENT_ARTIFACTS if not (parent / name).is_file()]
    checkpoint = parent / "checkpoints" / "gaussians_step_050000.ply"
    if missing or not checkpoint.is_file():
        missing_checkpoint = [] if checkpoint.is_file() else [str(checkpoint.relative_to(parent))]
        raise RuntimeError(
            "cooldown parent is missing completed-tail artifacts: "
            f"{[*missing, *missing_checkpoint]}"
        )

    config = _load_json_object(parent / "config.json", label="cooldown parent config")
    provenance = _load_json_object(parent / "provenance.json", label="cooldown provenance")
    targets = _load_json_object(parent / "compact_targets.json", label="cooldown targets")
    start = _load_json_object(parent / "tail_start.json", label="cooldown tail receipt")
    history = _load_json_object(parent / "training_history.json", label="cooldown history")
    selection = _load_json_object(parent / "model_selection.json", label="cooldown selection")
    fit_receipt = _load_json_object(parent / "fit_complete.json", label="cooldown fit receipt")
    trainer_config = _load_train_config(parent / "training_config.json")

    if config.get("schema") != "rtgs.full_compact_reconstruction.config.v1":
        raise RuntimeError("cooldown parent has an unsupported config schema")
    if config.get("fit_mode") != "all" or config.get("smoke") is not False:
        raise RuntimeError("cooldown requires the full all-view non-smoke parent")
    if config.get("iterations") != POLISH_PARENT_ITERATIONS or config.get("seed") != 0:
        raise RuntimeError("cooldown parent does not preserve the frozen 30k fit config")
    if tuple(config.get("view_names", ())) != EXPECTED_VIEW_NAMES or tuple(
        config.get("fit_indices", ())
    ) != tuple(range(len(EXPECTED_VIEW_NAMES))):
        raise RuntimeError("cooldown parent does not preserve the frozen all-view order")

    if provenance.get("schema") != "rtgs.full_compact_reconstruction.provenance.v1":
        raise RuntimeError("cooldown parent has an unsupported provenance schema")
    _provenance_identity(provenance)
    if (
        targets.get("schema") != DETERMINISTIC_TARGET_SCHEMA
        or targets.get("deterministic_algorithms") is not True
    ):
        raise RuntimeError("cooldown requires deterministic-v2 compact targets")
    target_views = targets.get("views")
    if not isinstance(target_views, list) or [
        (record.get("global_index"), record.get("view_id"))
        for record in target_views
        if isinstance(record, dict)
    ] != list(enumerate(EXPECTED_VIEW_NAMES)):
        raise RuntimeError("cooldown target receipt differs from the frozen all-view order")

    if trainer_config.iterations != TAIL_ITERATIONS:
        raise RuntimeError("cooldown parent training config is not a 10,000-step segment")
    if trainer_config.iteration_offset != TAIL_PARENT_ITERATIONS:
        raise RuntimeError("cooldown parent training config does not start at 40,000")
    if _resolve_schedule_iterations(trainer_config) != COOLDOWN_PARENT_ITERATIONS:
        raise RuntimeError("cooldown parent training schedule does not terminate at 50,000")
    if trainer_config.densify or trainer_config.eval_every != 1_000:
        raise RuntimeError("cooldown parent is not the frozen fixed-topology segment")
    if trainer_config.target_sh_degree != 3 or _resolve_sh_interval(trainer_config) != 1_000:
        raise RuntimeError("cooldown parent does not use degree-3 SH throughout")
    if trainer_config.seed != TAIL_SEED:
        raise RuntimeError("cooldown parent does not use the frozen tail seed")
    if not math.isclose(trainer_config.means_lr_final_factor, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("cooldown parent means LR is not constant")
    if not math.isclose(trainer_config.opacity_logit_epsilon, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("cooldown parent opacity entry epsilon differs")

    if start.get("schema") != "rtgs.full_compact_reconstruction.tail_start.v1":
        raise RuntimeError("cooldown parent has an unsupported tail-start receipt")
    if start.get("written_before_tail_training") is not True:
        raise RuntimeError("cooldown parent tail receipt was not frozen before training")
    if start.get("continuation_exact") is not False:
        raise RuntimeError("cooldown parent tail receipt must declare non-exact continuation")
    if start.get("global_steps") != {
        "parent_last": TAIL_PARENT_ITERATIONS,
        "first_tail": TAIL_PARENT_ITERATIONS + 1,
        "last_tail": COOLDOWN_PARENT_ITERATIONS,
        "segment_iterations": TAIL_ITERATIONS,
        "schedule_iterations": COOLDOWN_PARENT_ITERATIONS,
    }:
        raise RuntimeError("cooldown parent tail receipt has unexpected global steps")
    child_targets = start.get("child_target_receipt")
    if not isinstance(child_targets, dict) or child_targets.get("sha256") != _sha256_file(
        parent / "compact_targets.json"
    ):
        raise RuntimeError("cooldown parent tail receipt no longer binds compact targets")

    final_path = parent / "gaussians_final.ply"
    final_sha256 = _sha256_file(final_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if fit_receipt.get("schema") != "rtgs.full_compact_reconstruction.fit_complete.v1":
        raise RuntimeError("cooldown parent has an unsupported fit receipt schema")
    if fit_receipt.get("fit_kind") != "fixed_topology_tail":
        raise RuntimeError("cooldown requires a completed fixed-topology tail parent")
    if (
        fit_receipt.get("continuation_exact") is not False
        or fit_receipt.get("fixed_topology") is not True
    ):
        raise RuntimeError("cooldown parent fit receipt has invalid continuation semantics")
    n_final = fit_receipt.get("n_final_gaussians")
    if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final <= 0:
        raise RuntimeError("cooldown parent fit receipt has an invalid Gaussian count")
    if fit_receipt.get("source_n_gaussians") != n_final:
        raise RuntimeError("cooldown parent fixed topology changed the Gaussian count")
    if fit_receipt.get("final_ply_sha256") != final_sha256:
        raise RuntimeError("cooldown parent final PLY differs from its fit receipt")
    tail_binding = fit_receipt.get("tail")
    if not isinstance(tail_binding, dict) or tail_binding.get("receipt") != "tail_start.json":
        raise RuntimeError("cooldown parent fit receipt has no valid tail-start binding")
    if tail_binding.get("receipt_sha256") != _sha256_file(parent / "tail_start.json"):
        raise RuntimeError("cooldown parent tail-start receipt hash differs")

    if selection.get("schema") != MODEL_SELECTION_SCHEMA:
        raise RuntimeError("cooldown parent has an unsupported model-selection schema")
    if selection.get("written_before_gaussians_final") is not True:
        raise RuntimeError("cooldown parent selection was not frozen before finalization")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("cooldown parent selection has no selected candidate")
    if selected.get("global_step") != COOLDOWN_PARENT_ITERATIONS:
        raise RuntimeError("cooldown requires the selected 50,000-step parent candidate")
    if selected.get("artifact") != "checkpoints/gaussians_step_050000.ply":
        raise RuntimeError("cooldown selected artifact is not the 50,000-step checkpoint")
    if selected.get("sha256") != final_sha256 or checkpoint_sha256 != final_sha256:
        raise RuntimeError("cooldown final PLY differs from the selected 50k checkpoint")
    convergence = selection.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("joint_status") != "still_improving":
        raise RuntimeError("cooldown requires a 50k parent classified still_improving")
    selection_binding = fit_receipt.get("model_selection")
    if not isinstance(selection_binding, dict):
        raise RuntimeError("cooldown parent fit receipt has no model-selection binding")
    if selection_binding.get("receipt") != "model_selection.json" or selection_binding.get(
        "receipt_sha256"
    ) != _sha256_file(parent / "model_selection.json"):
        raise RuntimeError("cooldown parent model-selection receipt hash differs")
    if selection_binding.get("selected_global_step") != COOLDOWN_PARENT_ITERATIONS or (
        selection_binding.get("selected_candidate_sha256") != final_sha256
    ):
        raise RuntimeError("cooldown fit receipt does not bind the selected 50k candidate")
    if selection_binding.get("joint_convergence_status") != "still_improving":
        raise RuntimeError("cooldown fit receipt does not bind its convergence status")

    if history.get("iteration_offset") != TAIL_PARENT_ITERATIONS:
        raise RuntimeError("cooldown parent history does not start at 40,000")
    if history.get("segment_iterations") != TAIL_ITERATIONS:
        raise RuntimeError("cooldown parent history is not the complete 10,000-step segment")
    if history.get("schedule_iterations") != COOLDOWN_PARENT_ITERATIONS:
        raise RuntimeError("cooldown parent history does not terminate at 50,000")
    losses = history.get("loss")
    if not isinstance(losses, list) or len(losses) != TAIL_ITERATIONS:
        raise RuntimeError("cooldown parent history does not contain all 10,000 tail losses")
    for key in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        if _history_last_step(history, key) != COOLDOWN_PARENT_ITERATIONS:
            raise RuntimeError(f"cooldown parent {key} history does not end at step 50,000")
    if history["n_gaussians"][-1][1] != n_final:
        raise RuntimeError("cooldown history and fit receipt Gaussian counts differ")

    artifact_paths = {name: parent / name for name in COOLDOWN_PARENT_ARTIFACTS}
    artifact_paths["checkpoints/gaussians_step_050000.ply"] = checkpoint
    return {
        "parent": parent,
        "out": out,
        "config": config,
        "provenance": provenance,
        "targets": targets,
        "tail_start": start,
        "history": history,
        "selection": selection,
        "fit_receipt": fit_receipt,
        "trainer_config": trainer_config,
        "final_path": final_path,
        "final_sha256": final_sha256,
        "n_final": n_final,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }


def _settle_parent_preflight(parent: Path, out: Path) -> dict[str, Any]:
    """Bind an immutable selected 60k cooldown child for one settle segment."""

    parent = parent.resolve()
    out = out.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"settle requires an existing 60k cooldown parent: {parent}")
    if out == parent:
        raise RuntimeError("settle output must not be the parent run")
    if out.parent != parent.parent:
        raise RuntimeError("settle output must be a sibling of the parent run")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite settle output: {out}")
    missing = [name for name in SETTLE_PARENT_ARTIFACTS if not (parent / name).is_file()]
    checkpoint = parent / "checkpoints" / "gaussians_step_060000.ply"
    if missing or not checkpoint.is_file():
        missing_checkpoint = [] if checkpoint.is_file() else [str(checkpoint.relative_to(parent))]
        raise RuntimeError(
            "settle parent is missing completed-cooldown artifacts: "
            f"{[*missing, *missing_checkpoint]}"
        )

    config = _load_json_object(parent / "config.json", label="settle parent config")
    provenance = _load_json_object(parent / "provenance.json", label="settle provenance")
    targets = _load_json_object(parent / "compact_targets.json", label="settle targets")
    start = _load_json_object(parent / "cooldown_start.json", label="settle cooldown receipt")
    history = _load_json_object(parent / "training_history.json", label="settle history")
    selection = _load_json_object(parent / "model_selection.json", label="settle selection")
    fit_receipt = _load_json_object(parent / "fit_complete.json", label="settle fit receipt")
    trainer_config = _load_train_config(parent / "training_config.json")

    if config.get("schema") != "rtgs.full_compact_reconstruction.config.v1":
        raise RuntimeError("settle parent has an unsupported config schema")
    if config.get("fit_mode") != "all" or config.get("smoke") is not False:
        raise RuntimeError("settle requires the full all-view non-smoke parent")
    if config.get("iterations") != POLISH_PARENT_ITERATIONS or config.get("seed") != 0:
        raise RuntimeError("settle parent does not preserve the frozen 30k fit config")
    if tuple(config.get("view_names", ())) != EXPECTED_VIEW_NAMES or tuple(
        config.get("fit_indices", ())
    ) != tuple(range(len(EXPECTED_VIEW_NAMES))):
        raise RuntimeError("settle parent does not preserve the frozen all-view order")

    if provenance.get("schema") != "rtgs.full_compact_reconstruction.provenance.v1":
        raise RuntimeError("settle parent has an unsupported provenance schema")
    _provenance_identity(provenance)
    if (
        targets.get("schema") != DETERMINISTIC_TARGET_SCHEMA
        or targets.get("deterministic_algorithms") is not True
    ):
        raise RuntimeError("settle requires deterministic-v2 compact targets")
    target_views = targets.get("views")
    if not isinstance(target_views, list) or [
        (record.get("global_index"), record.get("view_id"))
        for record in target_views
        if isinstance(record, dict)
    ] != list(enumerate(EXPECTED_VIEW_NAMES)):
        raise RuntimeError("settle target receipt differs from the frozen all-view order")

    if trainer_config.iterations != COOLDOWN_ITERATIONS:
        raise RuntimeError("settle parent training config is not a 10,000-step segment")
    if trainer_config.iteration_offset != COOLDOWN_PARENT_ITERATIONS:
        raise RuntimeError("settle parent training config does not start at 50,000")
    if _resolve_schedule_iterations(trainer_config) != SETTLE_PARENT_ITERATIONS:
        raise RuntimeError("settle parent training schedule does not terminate at 60,000")
    if trainer_config.densify or trainer_config.eval_every != 1_000:
        raise RuntimeError("settle parent is not the frozen fixed-topology segment")
    if trainer_config.target_sh_degree != 3 or _resolve_sh_interval(trainer_config) != 1_000:
        raise RuntimeError("settle parent does not use degree-3 SH throughout")
    if trainer_config.seed != COOLDOWN_SEED:
        raise RuntimeError("settle parent does not use the frozen cooldown seed")
    if not math.isclose(trainer_config.means_lr_final_factor, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("settle parent means LR is not constant")
    if not math.isclose(trainer_config.opacity_logit_epsilon, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("settle parent opacity entry epsilon differs")

    if start.get("schema") != "rtgs.full_compact_reconstruction.cooldown_start.v1":
        raise RuntimeError("settle parent has an unsupported cooldown-start receipt")
    if start.get("written_before_cooldown_training") is not True:
        raise RuntimeError("settle parent cooldown receipt was not frozen before training")
    if start.get("continuation_exact") is not False:
        raise RuntimeError("settle parent cooldown must declare non-exact continuation")
    if start.get("global_steps") != {
        "parent_last": COOLDOWN_PARENT_ITERATIONS,
        "first_cooldown": COOLDOWN_PARENT_ITERATIONS + 1,
        "last_cooldown": SETTLE_PARENT_ITERATIONS,
        "segment_iterations": COOLDOWN_ITERATIONS,
        "schedule_iterations": SETTLE_PARENT_ITERATIONS,
    }:
        raise RuntimeError("settle parent cooldown receipt has unexpected global steps")
    child_targets = start.get("child_target_receipt")
    if not isinstance(child_targets, dict) or child_targets.get("sha256") != _sha256_file(
        parent / "compact_targets.json"
    ):
        raise RuntimeError("settle parent cooldown receipt no longer binds compact targets")

    final_path = parent / "gaussians_final.ply"
    final_sha256 = _sha256_file(final_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if fit_receipt.get("schema") != "rtgs.full_compact_reconstruction.fit_complete.v1":
        raise RuntimeError("settle parent has an unsupported fit receipt schema")
    if fit_receipt.get("fit_kind") != "fixed_topology_cooldown":
        raise RuntimeError("settle requires a completed fixed-topology cooldown parent")
    if (
        fit_receipt.get("continuation_exact") is not False
        or fit_receipt.get("fixed_topology") is not True
    ):
        raise RuntimeError("settle parent fit receipt has invalid continuation semantics")
    n_final = fit_receipt.get("n_final_gaussians")
    if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final <= 0:
        raise RuntimeError("settle parent fit receipt has an invalid Gaussian count")
    if fit_receipt.get("source_n_gaussians") != n_final:
        raise RuntimeError("settle parent fixed topology changed the Gaussian count")
    if fit_receipt.get("final_ply_sha256") != final_sha256:
        raise RuntimeError("settle parent final PLY differs from its fit receipt")
    cooldown_binding = fit_receipt.get("cooldown")
    if (
        not isinstance(cooldown_binding, dict)
        or cooldown_binding.get("receipt") != "cooldown_start.json"
    ):
        raise RuntimeError("settle parent fit receipt has no valid cooldown-start binding")
    if cooldown_binding.get("receipt_sha256") != _sha256_file(parent / "cooldown_start.json"):
        raise RuntimeError("settle parent cooldown-start receipt hash differs")

    if selection.get("schema") != MODEL_SELECTION_SCHEMA:
        raise RuntimeError("settle parent has an unsupported model-selection schema")
    if selection.get("written_before_gaussians_final") is not True:
        raise RuntimeError("settle parent selection was not frozen before finalization")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("settle parent selection has no selected candidate")
    if selected.get("global_step") != SETTLE_PARENT_ITERATIONS:
        raise RuntimeError("settle requires the selected 60,000-step parent candidate")
    if selected.get("artifact") != "checkpoints/gaussians_step_060000.ply":
        raise RuntimeError("settle selected artifact is not the 60,000-step checkpoint")
    if selected.get("sha256") != final_sha256 or checkpoint_sha256 != final_sha256:
        raise RuntimeError("settle final PLY differs from the selected 60k checkpoint")
    convergence = selection.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("joint_status") != "still_improving":
        raise RuntimeError("settle requires a 60k parent classified still_improving")
    selection_binding = fit_receipt.get("model_selection")
    if not isinstance(selection_binding, dict):
        raise RuntimeError("settle parent fit receipt has no model-selection binding")
    if selection_binding.get("receipt") != "model_selection.json" or selection_binding.get(
        "receipt_sha256"
    ) != _sha256_file(parent / "model_selection.json"):
        raise RuntimeError("settle parent model-selection receipt hash differs")
    if selection_binding.get("selected_global_step") != SETTLE_PARENT_ITERATIONS or (
        selection_binding.get("selected_candidate_sha256") != final_sha256
    ):
        raise RuntimeError("settle fit receipt does not bind the selected 60k candidate")
    if selection_binding.get("joint_convergence_status") != "still_improving":
        raise RuntimeError("settle fit receipt does not bind its convergence status")

    if history.get("iteration_offset") != COOLDOWN_PARENT_ITERATIONS:
        raise RuntimeError("settle parent history does not start at 50,000")
    if history.get("segment_iterations") != COOLDOWN_ITERATIONS:
        raise RuntimeError("settle parent history is not the complete 10,000-step segment")
    if history.get("schedule_iterations") != SETTLE_PARENT_ITERATIONS:
        raise RuntimeError("settle parent history does not terminate at 60,000")
    losses = history.get("loss")
    if not isinstance(losses, list) or len(losses) != COOLDOWN_ITERATIONS:
        raise RuntimeError("settle parent history does not contain all 10,000 cooldown losses")
    for key in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        if _history_last_step(history, key) != SETTLE_PARENT_ITERATIONS:
            raise RuntimeError(f"settle parent {key} history does not end at step 60,000")
    if history["n_gaussians"][-1][1] != n_final:
        raise RuntimeError("settle history and fit receipt Gaussian counts differ")

    artifact_paths = {name: parent / name for name in SETTLE_PARENT_ARTIFACTS}
    artifact_paths["checkpoints/gaussians_step_060000.ply"] = checkpoint
    return {
        "parent": parent,
        "out": out,
        "config": config,
        "provenance": provenance,
        "targets": targets,
        "cooldown_start": start,
        "history": history,
        "selection": selection,
        "fit_receipt": fit_receipt,
        "trainer_config": trainer_config,
        "final_path": final_path,
        "final_sha256": final_sha256,
        "n_final": n_final,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R": camera.R.detach().cpu().reshape(-1).tolist(),
        "t": camera.t.detach().cpu().tolist(),
    }


def _crop_camera(camera: Camera, fit_window: tuple[int, int, int, int]) -> Camera:
    x, y, width, height = fit_window
    return Camera(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx - x,
        cy=camera.cy - y,
        width=width,
        height=height,
        R=camera.R,
        t=camera.t,
    )


def _effective_config(args: argparse.Namespace, dataset: CompactDataset) -> dict[str, Any]:
    if tuple(view.view_id for view in dataset.views) != EXPECTED_VIEW_NAMES:
        raise RuntimeError("compact view order differs from the frozen frame_00008 order")
    if dataset.calibration_sha256 != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("compact calibration digest differs from the frozen calibration")
    if dataset.n_views != 26:
        raise RuntimeError("frame_00008 must contain exactly 26 compact views")

    fit_indices = tuple(range(dataset.n_views)) if args.fit_mode == "all" else T_INDICES
    validation_indices = V_INDICES
    excluded_indices = () if args.fit_mode == "all" else H_INDICES
    if args.smoke:
        max_tracks = 32
        depth_samples = 2
        iterations = 1
        eval_every = 1
        densify = False
        max_gaussians = 64
    else:
        max_tracks = args.max_tracks
        depth_samples = args.depth_samples
        iterations = args.iterations
        eval_every = args.eval_every
        densify = not args.no_densify
        max_gaussians = args.max_gaussians
    expected_candidates = sum(dataset.views[index].observation.n for index in fit_indices)
    return {
        "schema": "rtgs.full_compact_reconstruction.config.v1",
        "created_utc": _utc_now(),
        "argv": list(sys.argv),
        "git_revision": _git_revision(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "dataset": str(dataset.path),
        "dataset_name": dataset.name,
        "fit_mode": args.fit_mode,
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "excluded_indices": excluded_indices,
        "T_indices": T_INDICES,
        "V_indices": V_INDICES,
        "H_indices": H_INDICES,
        "view_names": tuple(view.view_id for view in dataset.views),
        "expected_component_center_candidates": expected_candidates,
        "max_tracks": max_tracks,
        "depth_samples": depth_samples,
        "min_views": min(args.min_views, len(fit_indices)),
        "robust_view_fraction": args.robust_view_fraction,
        "min_placement_score": args.min_placement_score,
        "init_opacity": args.init_opacity,
        "iterations": iterations,
        "eval_every": eval_every,
        "device": args.device,
        "structsplat_renderer": args.structsplat_renderer,
        "structsplat_chunk": args.structsplat_chunk,
        "rasterizer": "gsplat",
        "packed": True,
        "antialiased": True,
        "target_sh_degree": 3,
        "densify": densify,
        "density_strategy": args.density_strategy,
        "densify_start": args.densify_start,
        "densify_stop": args.densify_stop,
        "densify_every": args.densify_every,
        "max_gaussians": max_gaussians,
        "prune_opacity": args.prune_opacity,
        "prune_scale_frac": args.prune_scale_frac,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "original_frame": str(ORIGINAL_FRAME),
        "original_calibration": str(ORIGINAL_CALIBRATION),
        "expected_calibration_sha256": EXPECTED_CALIBRATION_SHA256,
        "interpretation": (
            "All selected compact components contribute placement candidates and all selected "
            "teacher crops supervise training. max_tracks bounds the latent 3D representation; "
            "it does not subsample compact evidence."
        ),
        "validation_interpretation": (
            "V is fitted-view validation, not held-out evidence."
            if args.fit_mode == "all"
            else "V is compact validation excluded from fitting; H is reporting-only."
        ),
    }


def _compact_provenance(dataset: CompactDataset, config: dict[str, Any]) -> dict[str, Any]:
    manifest = dataset.path / "manifest.json"
    return {
        "schema": "rtgs.full_compact_reconstruction.provenance.v1",
        "written_before_source_rgb_access": True,
        "created_utc": _utc_now(),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "calibration_sha256": dataset.calibration_sha256,
        "bounds_hint": dataset.bounds_hint,
        "fit_indices": config["fit_indices"],
        "expected_component_center_candidates": config["expected_component_center_candidates"],
        "views": [
            {
                "index": index,
                "view_id": view.view_id,
                "bundle": str(view.path),
                "bundle_sha256": view.sha256,
                "bundle_bytes": view.bytes,
                "n_components": view.observation.n,
                "fit_window": view.observation.fit_window,
                "has_alpha": view.alpha is not None,
                "source": view.source,
                "camera": _camera_record(view.camera),
            }
            for index, view in enumerate(dataset.views)
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gsplat": _package_version("gsplat"),
            "structsplat": _package_version("structsplat"),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"),
        },
    }


def _place_initial_gaussians(
    dataset: CompactDataset,
    config: dict[str, Any],
) -> tuple[Gaussians3D, dict[str, Any]]:
    fit_indices = tuple(config["fit_indices"])
    heldout = tuple(index for index in range(dataset.n_views) if index not in fit_indices)
    fits = SceneFits.from_compact_dataset(
        dataset,
        train_view_indices=fit_indices,
        heldout_view_indices=heldout,
    )
    placement_config = FieldLiftConfig(
        max_tracks=int(config["max_tracks"]),
        max_train_views=len(fit_indices),
        depth_samples=int(config["depth_samples"]),
        min_views=int(config["min_views"]),
        robust_view_fraction=float(config["robust_view_fraction"]),
        min_placement_score=float(config["min_placement_score"]),
        init_opacity=float(config["init_opacity"]),
        background_fraction=0.0,
        topology_rounds=0,
        seed=int(config["seed"]),
    )
    started = time.perf_counter()
    placement = _place(fits, fit_indices, placement_config)
    fallback = placement.diagnostics.get("placement_fallback_reason")
    if fallback is not None:
        raise RuntimeError(f"component-center placement fell back: {fallback}")
    placement_diagnostics = placement.diagnostics["placement"]
    actual_candidates = int(placement_diagnostics["candidate_count"])
    expected_candidates = int(config["expected_component_center_candidates"])
    if actual_candidates != expected_candidates:
        raise RuntimeError(
            f"component-center candidate count mismatch: {actual_candidates} != "
            f"{expected_candidates}"
        )
    if placement_diagnostics["anchor_mode"] != "component_centers":
        raise RuntimeError("field placement did not use component-center anchors")
    gaussians = placement.fiber.as_gaussians(
        colors=placement.source_colors.clamp(0, 1),
        opacity=placement.render_opacity,
    ).detach()
    gaussians = Gaussians3D(
        means=gaussians.means.float(),
        quats=gaussians.quats.float(),
        log_scales=gaussians.log_scales.float(),
        opacity=gaussians.opacity.float(),
        sh=gaussians.sh.float(),
    )
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.placement.v1",
        "elapsed_seconds": time.perf_counter() - started,
        "field_lift_config": dataclasses.asdict(placement_config),
        "diagnostics": placement.diagnostics,
        "n_gaussians": gaussians.n,
        "candidate_count": actual_candidates,
        "source_global_view_indices": placement.source_global_view_indices.tolist(),
        "source_component_indices": placement.fiber.source_component_indices.tolist(),
        "source_projection_dilation": placement.fiber.dilation,
        "standard_renderer_dilation": EWA_DILATION,
        "renderer_refits_source_dilation_mismatch": not math.isclose(
            placement.fiber.dilation,
            EWA_DILATION,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "gaussians_sha256": {
            "means": _tensor_sha256(gaussians.means),
            "quats": _tensor_sha256(gaussians.quats),
            "log_scales": _tensor_sha256(gaussians.log_scales),
            "opacity": _tensor_sha256(gaussians.opacity),
            "sh": _tensor_sha256(gaussians.sh),
        },
    }
    return gaussians, receipt


def _structsplat_mode(observation, requested: str) -> str:
    if requested == "reference":
        return observation.blend_mode
    if observation.color_grads is not None or observation.support_fade_alpha not in (0.0, 1.0):
        return observation.blend_mode
    if observation.blend_mode == "additive":
        return "cuda_tiled_additive"
    return "cuda_tiled"


@torch.no_grad()
def _render_compact_crop(
    view: CompactView,
    *,
    device: torch.device,
    renderer_name: str,
    chunk: int,
    apply_alpha: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    observation = view.observation.to(device)
    x, y, width, height = observation.fit_window
    field = GaussianField(
        observation.local_means(),
        observation.log_scales,
        observation.rotations,
        observation.colors,
        opacities=None,
        color_grads=observation.color_grads,
        filter_variance=observation.filter_variance,
    )
    mode = _structsplat_mode(observation, renderer_name)
    started = time.perf_counter()
    crop = render_field(
        field.means,
        field.conics(observation.aa_dilation),
        field.colors,
        field.radii(observation.sigma_cutoff, observation.aa_dilation),
        height,
        width,
        chunk=chunk,
        mode=mode,
        opacities=observation.amplitudes,
        scales=field.effective_scales(0.0),
        rotations=field.rotations,
        support_fade=observation.support_fade_alpha > 0.0,
        support_fade_alpha=observation.support_fade_alpha,
        sigma_cutoff=observation.sigma_cutoff,
        color_grads=field.color_grads,
    )
    if crop.shape != (height, width, 3) or not bool(torch.isfinite(crop).all()):
        raise RuntimeError(f"invalid StructSplat replay for {view.view_id}")
    alpha = None if view.alpha is None else view.alpha.crop_mask(device)
    if alpha is not None:
        if view.alpha.origin != (x, y) or alpha.shape != (height, width):
            raise RuntimeError(f"packed alpha is misaligned for {view.view_id}")
        if apply_alpha:
            crop = crop * alpha[..., None]
    unclamped_min = float(crop.min())
    unclamped_max = float(crop.max())
    crop = crop.clamp(0, 1).float()
    receipt = {
        "view_id": view.view_id,
        "fit_window": observation.fit_window,
        "renderer": mode,
        "blend_mode": observation.blend_mode,
        "components": observation.n,
        "has_alpha": alpha is not None,
        "alpha_applied": bool(alpha is not None and apply_alpha),
        "unclamped_min": unclamped_min,
        "unclamped_max": unclamped_max,
        "crop_sha256": _tensor_sha256(crop),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return crop, alpha, receipt


def _materialize_training_scene(
    dataset: CompactDataset,
    config: dict[str, Any],
) -> tuple[SceneData, list[dict[str, Any]]]:
    fit_indices = tuple(config["fit_indices"])
    validation_indices = tuple(config["validation_indices"])
    included = tuple(sorted(set(fit_indices) | set(validation_indices)))
    device = torch.device(config["device"])
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    cameras: list[Camera] = []
    names: list[str] = []
    receipts: list[dict[str, Any]] = []
    for global_index in included:
        view = dataset.views[global_index]
        target, alpha, receipt = _render_compact_crop(
            view,
            device=device,
            renderer_name=str(config["structsplat_renderer"]),
            chunk=int(config["structsplat_chunk"]),
            apply_alpha=True,
        )
        if alpha is None:
            raise RuntimeError(f"full compact reconstruction requires alpha for {view.view_id}")
        images.append(target.cpu())
        masks.append(alpha.cpu())
        cameras.append(_crop_camera(view.camera, view.observation.fit_window))
        names.append(view.view_id)
        receipts.append({"global_index": global_index, **receipt})
        print(
            f"[targets] {len(receipts):02d}/{len(included):02d} {view.view_id} "
            f"{view.observation.fit_window[2]}x{view.observation.fit_window[3]}",
            flush=True,
        )
        del target, alpha
    local_by_global = {global_index: local for local, global_index in enumerate(included)}
    local_train = [local_by_global[index] for index in fit_indices]
    local_validation = (
        []
        if config["fit_mode"] == "all"
        else [local_by_global[index] for index in validation_indices]
    )
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=names,
        masks=masks,
        train_indices=local_train,
        test_indices=local_validation,
        bounds_hint=dataset.bounds_hint,
        name=f"{dataset.name}-compact-native-crops",
    )
    scene.validate()
    return scene, receipts


def _independently_replay_training_targets(
    dataset: CompactDataset,
    config: dict[str, Any],
    scene: SceneData,
    replayed: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_indices = tuple(config["fit_indices"])
    validation_indices = tuple(config["validation_indices"])
    included = tuple(sorted(set(fit_indices) | set(validation_indices)))
    if len(included) != len(replayed) or len(scene.images) != len(replayed):
        raise RuntimeError("independent target replay view count differs")
    if scene.masks is None:
        raise RuntimeError("independent target replay requires retained alpha masks")
    device = torch.device(config["device"])
    independent_records: list[dict[str, Any]] = []
    alpha_records: list[dict[str, Any]] = []
    for local_index, global_index in enumerate(included):
        view = dataset.views[global_index]
        target, alpha, receipt = _render_compact_crop(
            view,
            device=device,
            renderer_name=str(config["structsplat_renderer"]),
            chunk=int(config["structsplat_chunk"]),
            apply_alpha=True,
        )
        if alpha is None:
            raise RuntimeError(f"independent replay has no alpha for {view.view_id}")
        scene_target = scene.images[local_index]
        scene_mask = scene.masks[local_index]
        if target.shape != scene_target.shape or target.dtype != scene_target.dtype:
            raise RuntimeError(f"{view.view_id} independent replay tensor metadata differs")
        scene_crop_sha256 = _tensor_sha256(scene_target)
        if scene_crop_sha256 != replayed[local_index].get("crop_sha256"):
            raise RuntimeError(f"{view.view_id} retained recovery target hash differs")
        independent_alpha = alpha.cpu()
        if not torch.equal(independent_alpha, scene_mask):
            raise RuntimeError(f"{view.view_id} independent replay alpha differs")
        alpha_sha256 = _tensor_sha256(scene_mask)
        alpha_records.append(
            {
                "view_id": view.view_id,
                "alpha_sha256": alpha_sha256,
                "foreground_count": int(scene_mask.sum()),
                "exact_match": True,
            }
        )
        independent_records.append({"global_index": global_index, **receipt})
        print(
            f"[targets-verify] {local_index + 1:02d}/{len(included):02d} "
            f"{view.view_id} exact deterministic replay",
            flush=True,
        )
        del target, alpha, independent_alpha
    return independent_records, {
        "all_alpha_masks_match": True,
        "view_count": len(alpha_records),
        "identity_sha256": _json_sha256(alpha_records),
        "views": alpha_records,
    }


def _train_config(config: dict[str, Any]) -> TrainConfig:
    use_absgrad = str(config["density_strategy"]) == "gsplat-default"
    density = DensityConfig(
        start_iter=int(config["densify_start"]),
        stop_iter=int(config["densify_stop"]),
        every=int(config["densify_every"]),
        grad_threshold=8e-4 if use_absgrad else 2e-4,
        absgrad=use_absgrad,
        prune_opacity=float(config["prune_opacity"]),
        prune_scale_frac=float(config["prune_scale_frac"]),
        max_gaussians=int(config["max_gaussians"]),
    )
    return TrainConfig(
        iterations=int(config["iterations"]),
        rasterizer="gsplat",
        device=str(config["device"]),
        densify=bool(config["densify"]),
        density_strategy=str(config["density_strategy"]),
        density=density,
        eval_every=int(config["eval_every"]),
        target_sh_degree=3,
        sh_degree_interval=1000,
        use_masks=True,
        random_background=False,
        packed=True,
        antialiased=True,
        validate_render_finite=True,
        seed=int(config["seed"]),
    )


def _save_ply_new(gaussians: Gaussians3D, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Gaussian artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(path)


def _train_and_finalize(
    out: Path,
    config: dict[str, Any],
    scene: SceneData,
    init: Gaussians3D,
    trainer_config: TrainConfig,
    *,
    recovery_receipt: Path | None = None,
    polish_receipt: Path | None = None,
    tail_receipt: Path | None = None,
    cooldown_receipt: Path | None = None,
    settle_receipt: Path | None = None,
    initialization_callback: Callable[[Gaussians3D], None] | None = None,
    trainer_entry_record: dict[str, Any] | None = None,
    selection_baseline_step: int = POLISH_PARENT_ITERATIONS,
    selection_final_step: int = POLISH_SCHEDULE_ITERATIONS,
) -> None:
    if (
        sum(
            receipt is not None
            for receipt in (
                recovery_receipt,
                polish_receipt,
                tail_receipt,
                cooldown_receipt,
                settle_receipt,
            )
        )
        > 1
    ):
        raise ValueError(
            "recovery, polish, tail, cooldown, and settle receipts are mutually exclusive"
        )
    checkpoints = out / "checkpoints"
    if recovery_receipt is None:
        checkpoints.mkdir()
    elif not checkpoints.is_dir():
        raise RuntimeError("recovery requires the existing checkpoints directory")

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        _save_ply_new(snapshot, checkpoints / f"gaussians_step_{step:06d}.ply")
        print(f"[checkpoint] step={step} gaussians={snapshot.n}", flush=True)

    action = (
        "resuming"
        if recovery_receipt is not None
        else (
            "polishing"
            if polish_receipt is not None
            else (
                "extending"
                if tail_receipt is not None
                else (
                    "cooling down"
                    if cooldown_receipt is not None
                    else ("settling" if settle_receipt is not None else "starting")
                )
            )
        )
    )
    print(f"[train] {action} native-resolution CUDA optimization", flush=True)
    final, history = Trainer(trainer_config).train(
        scene,
        init,
        checkpoint_callback=checkpoint,
        initialization_callback=initialization_callback,
    )
    recovery_record = None
    if recovery_receipt is not None:
        recovery_record = {
            "receipt": str(recovery_receipt.relative_to(out)),
            "receipt_sha256": _sha256_file(recovery_receipt),
        }
        history["resume_exact"] = False
        history["recovery"] = recovery_record
    polish_record = None
    if polish_receipt is not None:
        if final.n != init.n:
            raise RuntimeError("fixed-topology polish changed the Gaussian count")
        if trainer_entry_record is None or (
            trainer_entry_record.get("verified_before_optimizer_construction") is not True
        ):
            raise RuntimeError("polish did not verify trainer-entry opacity preservation")
        polish_record = {
            "receipt": str(polish_receipt.relative_to(out)),
            "receipt_sha256": _sha256_file(polish_receipt),
        }
        history["continuation_exact"] = False
        history["fixed_topology"] = True
        history["polish"] = polish_record
        history["trainer_entry_opacity"] = dict(trainer_entry_record)
    tail_record = None
    if tail_receipt is not None:
        if final.n != init.n:
            raise RuntimeError("fixed-topology tail changed the Gaussian count")
        if trainer_entry_record is None or (
            trainer_entry_record.get("verified_before_optimizer_construction") is not True
        ):
            raise RuntimeError("tail did not verify trainer-entry opacity preservation")
        tail_record = {
            "receipt": str(tail_receipt.relative_to(out)),
            "receipt_sha256": _sha256_file(tail_receipt),
        }
        history["continuation_exact"] = False
        history["fixed_topology"] = True
        history["tail"] = tail_record
        history["trainer_entry_opacity"] = dict(trainer_entry_record)
    cooldown_record = None
    if cooldown_receipt is not None:
        if final.n != init.n:
            raise RuntimeError("fixed-topology cooldown changed the Gaussian count")
        if trainer_entry_record is None or (
            trainer_entry_record.get("verified_before_optimizer_construction") is not True
        ):
            raise RuntimeError("cooldown did not verify trainer-entry opacity preservation")
        cooldown_record = {
            "receipt": str(cooldown_receipt.relative_to(out)),
            "receipt_sha256": _sha256_file(cooldown_receipt),
        }
        history["continuation_exact"] = False
        history["fixed_topology"] = True
        history["cooldown"] = cooldown_record
        history["trainer_entry_opacity"] = dict(trainer_entry_record)
    settle_record = None
    if settle_receipt is not None:
        if final.n != init.n:
            raise RuntimeError("fixed-topology settle changed the Gaussian count")
        if trainer_entry_record is None or (
            trainer_entry_record.get("verified_before_optimizer_construction") is not True
        ):
            raise RuntimeError("settle did not verify trainer-entry opacity preservation")
        settle_record = {
            "receipt": str(settle_receipt.relative_to(out)),
            "receipt_sha256": _sha256_file(settle_receipt),
        }
        history["continuation_exact"] = False
        history["fixed_topology"] = True
        history["settle"] = settle_record
        history["trainer_entry_opacity"] = dict(trainer_entry_record)
    model_selection_record = None
    if (
        polish_receipt is None
        and tail_receipt is None
        and cooldown_receipt is None
        and settle_receipt is None
    ):
        _save_ply_new(final, out / "gaussians_final.ply")
    else:
        selected_path, model_selection = _evaluate_polish_candidates(
            out,
            scene,
            trainer_config,
            expected_n=init.n,
            baseline_step=selection_baseline_step,
            final_step=selection_final_step,
        )
        selection_path = out / "model_selection.json"
        _write_json_new(selection_path, model_selection)
        _copy_file_new(selected_path, out / "gaussians_final.ply")
        if _sha256_file(out / "gaussians_final.ply") != model_selection["selected"]["sha256"]:
            raise RuntimeError("selected final PLY differs from model-selection receipt")
        final = Gaussians3D.load_ply(out / "gaussians_final.ply").to(
            torch.device(trainer_config.device)
        )
        model_selection_record = {
            "receipt": "model_selection.json",
            "receipt_sha256": _sha256_file(selection_path),
            "selected_global_step": model_selection["selected"]["global_step"],
            "selected_candidate_sha256": model_selection["selected"]["sha256"],
            "joint_convergence_status": model_selection["convergence"]["joint_status"],
        }
        history["model_selection"] = model_selection_record
    _write_json_new(out / "training_history.json", history)

    from rtgs.render.base import get_rasterizer

    renderer = get_rasterizer(
        "gsplat",
        device=final.means.device,
        packed=True,
        antialiased=True,
    )
    validation_local = [scene.view_names.index(EXPECTED_VIEW_NAMES[index]) for index in V_INDICES]
    compact_metrics = {
        "train": Trainer.evaluate_metrics(
            scene,
            final,
            renderer,
            indices=scene.training_views,
        ),
        "V": Trainer.evaluate_metrics(
            scene,
            final,
            renderer,
            indices=validation_local,
        ),
        "V_interpretation": config["validation_interpretation"],
    }
    if model_selection_record is not None:
        compact_metrics["model_selection"] = model_selection_record
    _write_json_new(out / "compact_metrics.json", compact_metrics)
    fit_receipt = {
        "schema": "rtgs.full_compact_reconstruction.fit_complete.v1",
        "completed_utc": _utc_now(),
        "final_ply": "gaussians_final.ply",
        "final_ply_sha256": _sha256_file(out / "gaussians_final.ply"),
        "n_final_gaussians": final.n,
    }
    if recovery_record is not None:
        fit_receipt["resume_exact"] = False
        fit_receipt["recovery"] = recovery_record
    continuation_record = (
        settle_record
        if settle_record is not None
        else (
            cooldown_record
            if cooldown_record is not None
            else (tail_record if tail_record is not None else polish_record)
        )
    )
    if continuation_record is not None:
        if settle_record is not None:
            receipt_key = "settle"
            fit_kind = "fixed_topology_settle"
        elif cooldown_record is not None:
            receipt_key = "cooldown"
            fit_kind = "fixed_topology_cooldown"
        elif tail_record is not None:
            receipt_key = "tail"
            fit_kind = "fixed_topology_tail"
        else:
            receipt_key = "polish"
            fit_kind = "fixed_topology_polish"
        fit_receipt["fit_kind"] = fit_kind
        fit_receipt["continuation_exact"] = False
        fit_receipt["fixed_topology"] = True
        fit_receipt["source_n_gaussians"] = init.n
        fit_receipt[receipt_key] = continuation_record
        fit_receipt["trainer_entry_opacity"] = dict(trainer_entry_record)
        fit_receipt["model_selection"] = model_selection_record
    _write_json_new(out / "fit_complete.json", fit_receipt)
    print(f"[fit] complete final_gaussians={final.n}", flush=True)


def _fit(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = CompactDataset.load(args.dataset.resolve(), device="cpu")
    config = _effective_config(args, dataset)
    _write_json_new(out / "config.json", config)
    _write_json_new(out / "provenance.json", _compact_provenance(dataset, config))
    print(
        f"[plan] views={len(config['fit_indices'])} "
        f"components={config['expected_component_center_candidates']} "
        f"tracks={config['max_tracks']} iterations={config['iterations']}",
        flush=True,
    )

    print("[placement] enumerating every selected component-center ray", flush=True)
    init, placement_receipt = _place_initial_gaussians(dataset, config)
    _save_ply_new(init, out / "gaussians_init.ply")
    _write_json_new(out / "placement.json", placement_receipt)
    print(
        f"[placement] candidates={placement_receipt['candidate_count']} "
        f"initial_3d={init.n} elapsed={placement_receipt['elapsed_seconds']:.1f}s",
        flush=True,
    )

    print("[targets] rendering native StructSplat crops", flush=True)
    with _deterministic_algorithms():
        scene, materialization = _materialize_training_scene(dataset, config)
    _write_json_new(
        out / "compact_targets.json",
        {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "deterministic_algorithms": True,
            "views": materialization,
        },
    )
    trainer_config = _train_config(config)
    _write_json_new(out / "training_config.json", dataclasses.asdict(trainer_config))
    _train_and_finalize(out, config, scene, init, trainer_config)


def _polish(args: argparse.Namespace) -> None:
    if args.parent_out is None:  # parse_args enforces this for CLI callers.
        raise ValueError("polish requires --parent-out")
    state = _polish_parent_preflight(args.parent_out, args.out)
    parent = state["parent"]
    out = state["out"]
    config = state["config"]
    provenance = state["provenance"]
    target_path = parent / "compact_targets.json"

    dataset = CompactDataset.load(Path(config["dataset"]).resolve(), device="cpu")
    if tuple(view.view_id for view in dataset.views) != tuple(config["view_names"]):
        raise RuntimeError("current compact dataset view order differs from the polish parent")
    manifest_path = dataset.path / "manifest.json"
    if _sha256_file(manifest_path) != provenance.get("manifest_sha256"):
        raise RuntimeError("current compact manifest differs from the polish parent")
    if dataset.calibration_sha256 != provenance.get("calibration_sha256"):
        raise RuntimeError("current compact calibration differs from the polish parent")
    current_provenance = _compact_provenance(dataset, config)
    provenance_replay = _verify_provenance_identity(provenance, current_provenance)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen polish parent config")

    init = Gaussians3D.load_ply(state["final_path"])
    if init.n != state["n_final"]:
        raise RuntimeError("parent final PLY count differs from its fit receipt")
    if init.sh_degree != 3:
        raise RuntimeError("polish requires a degree-3 parent final PLY")

    print("[targets] replaying and verifying every native StructSplat crop twice", flush=True)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        scene, replayed_targets = _materialize_training_scene(dataset, config)
        independently_replayed, alpha_replay = _independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed_targets,
        )
        target_replay = _verify_target_replay(
            target_path,
            replayed_targets,
            independently_replayed,
        )
    if torch.are_deterministic_algorithms_enabled() != previous_deterministic:
        raise RuntimeError("polish target replay did not restore deterministic algorithms")
    target_replay["prior_deterministic_algorithms_setting_restored"] = True
    target_replay["deterministic_algorithms_setting_after_replay"] = previous_deterministic
    target_replay["alpha_replay"] = alpha_replay
    if target_replay.get("reference_schema") != LEGACY_NONDETERMINISTIC_TARGET_SCHEMA:
        raise RuntimeError("polish replay did not consume the legacy-v1 parent targets")
    if target_replay.get("deterministic_recovery_replay_verified") is not True:
        raise RuntimeError("polish target replays were not independently deterministic")
    prior_target_replay = state["recovery"].get("target_replay")
    if not isinstance(prior_target_replay, dict):
        raise RuntimeError("parent recovery target replay is missing")
    if target_replay.get("deterministic_recovery_identity_sha256") != prior_target_replay.get(
        "deterministic_recovery_identity_sha256"
    ):
        raise RuntimeError("polish deterministic targets differ from the recovered parent replay")
    prior_alpha_replay = prior_target_replay.get("alpha_replay")
    if not isinstance(prior_alpha_replay, dict) or alpha_replay.get(
        "identity_sha256"
    ) != prior_alpha_replay.get("identity_sha256"):
        raise RuntimeError("polish alpha replay differs from the recovered parent replay")

    for name, record in state["artifacts"].items():
        if _sha256_file(parent / name) != record["sha256"]:
            raise RuntimeError(f"polish parent artifact changed during preflight: {name}")
    if _sha256_file(state["recovery_path"]) != state["recovery_sha256"]:
        raise RuntimeError("polish parent recovery receipt changed during preflight")

    trainer_config = _polish_train_config(state["recovery_trainer_config"])
    if _resolve_schedule_iterations(trainer_config) != POLISH_SCHEDULE_ITERATIONS:
        raise RuntimeError("internal polish schedule resolution differs from 40,000 steps")
    sh_interval = _resolve_sh_interval(trainer_config)
    first_active_degree = min(
        trainer_config.target_sh_degree,
        trainer_config.iteration_offset // sh_interval,
    )
    if first_active_degree != 3:
        raise RuntimeError("polish SH degree is not active from its first global step")
    opacity_entry_plan = _opacity_entry_plan(init, trainer_config.opacity_logit_epsilon)
    trainer_entry_record: dict[str, Any] = {}
    initialization_callback = _opacity_entry_verifier(
        init,
        trainer_config.opacity_logit_epsilon,
        trainer_entry_record,
    )

    out.mkdir(parents=True, exist_ok=False)
    for name in POLISH_COPIED_ARTIFACTS:
        _copy_file_new(parent / name, out / name)
    _write_json_new(
        out / "compact_targets.json",
        {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "deterministic_algorithms": True,
            "views": replayed_targets,
        },
    )
    _copy_file_new(state["final_path"], out / "gaussians_init.ply")
    _write_json_new(out / "training_config.json", dataclasses.asdict(trainer_config))

    receipt_path = out / "polish_start.json"
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.polish_start.v1",
        "created_utc": _utc_now(),
        "written_before_polish_training": True,
        "continuation_exact": False,
        "continuation_exact_reason": (
            "The parent PLY preserves Gaussian parameters but not Adam moments, per-parameter "
            "step counters, or the recovered RNG stream. The polish is a deliberate new "
            "fixed-topology optimizer segment, not an exact continuation."
        ),
        "parent_output": str(parent),
        "output": str(out),
        "parent_final": {
            "path": str(state["final_path"]),
            "sha256": state["final_sha256"],
            "bytes": state["final_path"].stat().st_size,
            "n_gaussians": init.n,
            "sh_degree": init.sh_degree,
        },
        "parent_recovery": {
            "path": str(state["recovery_path"]),
            "sha256": state["recovery_sha256"],
            "resume_exact": False,
            "resume_step": POLISH_PARENT_RESUME_STEP,
            "last_recovered_step": POLISH_PARENT_ITERATIONS,
        },
        "global_steps": {
            "parent_last": POLISH_PARENT_ITERATIONS,
            "first_polish": POLISH_PARENT_ITERATIONS + 1,
            "last_polish": POLISH_SCHEDULE_ITERATIONS,
            "segment_iterations": POLISH_ITERATIONS,
            "schedule_iterations": POLISH_SCHEDULE_ITERATIONS,
        },
        "fixed_topology": {
            "enabled": True,
            "densify": False,
            "source_n_gaussians": init.n,
        },
        "sh_schedule": {
            "target_degree": trainer_config.target_sh_degree,
            "degree_interval": sh_interval,
            "first_active_degree": first_active_degree,
            "active_from_first_polish_step": True,
        },
        "learning_rates": {
            "means_parent_base": state["recovery_trainer_config"].lr_means,
            "means_parent_terminal_factor": 0.01,
            "means_polish_base": trainer_config.lr_means,
            "means_polish_final_factor": trainer_config.means_lr_final_factor,
            "means_constant_during_polish": True,
            "other_parameter_factor": POLISH_OTHER_LR_FACTOR,
        },
        "rng": {
            "parent_seed": state["recovery_trainer_config"].seed,
            "polish_seed": trainer_config.seed,
            "new_deterministic_stream": True,
        },
        "parent_target_lineage": {
            "schema": LEGACY_NONDETERMINISTIC_TARGET_SCHEMA,
            "sha256": _sha256_file(parent / "compact_targets.json"),
            "original_tensor_equivalence_verified": target_replay[
                "original_tensor_equivalence_verified"
            ],
            "independent_deterministic_replay_verified": True,
        },
        "child_target_receipt": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "path": str((out / "compact_targets.json").resolve()),
            "sha256": _sha256_file(out / "compact_targets.json"),
            "promoted_from_independent_deterministic_replay": True,
        },
        "opacity_entry_plan": opacity_entry_plan,
        "provenance_replay": provenance_replay,
        "target_replay": target_replay,
        "parent_artifacts": state["artifacts"],
        "child_frozen_artifacts": {
            name: {
                "path": str((out / name).resolve()),
                "sha256": _sha256_file(out / name),
            }
            for name in (
                "config.json",
                "provenance.json",
                "compact_targets.json",
                "training_config.json",
                "gaussians_init.ply",
            )
        },
        "polish_training_config": dataclasses.asdict(trainer_config),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_revision": _git_revision(),
        },
    }
    _write_json_new(receipt_path, receipt)
    print(
        "[polish] immutable receipt=polish_start.json continuation_exact=false "
        "global_steps=30001..40000 fixed_topology=true",
        flush=True,
    )
    _train_and_finalize(
        out,
        config,
        scene,
        init,
        trainer_config,
        polish_receipt=receipt_path,
        initialization_callback=initialization_callback,
        trainer_entry_record=trainer_entry_record,
    )


def _tail(args: argparse.Namespace) -> None:
    if args.parent_out is None:  # parse_args enforces this for CLI callers.
        raise ValueError("tail requires --parent-out")
    state = _tail_parent_preflight(args.parent_out, args.out)
    parent = state["parent"]
    out = state["out"]
    config = state["config"]
    provenance = state["provenance"]

    dataset = CompactDataset.load(Path(config["dataset"]).resolve(), device="cpu")
    if tuple(view.view_id for view in dataset.views) != tuple(config["view_names"]):
        raise RuntimeError("current compact dataset view order differs from the tail parent")
    manifest_path = dataset.path / "manifest.json"
    if _sha256_file(manifest_path) != provenance.get("manifest_sha256"):
        raise RuntimeError("current compact manifest differs from the tail parent")
    if dataset.calibration_sha256 != provenance.get("calibration_sha256"):
        raise RuntimeError("current compact calibration differs from the tail parent")
    current_provenance = _compact_provenance(dataset, config)
    provenance_replay = _verify_provenance_identity(provenance, current_provenance)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen tail parent config")

    init = Gaussians3D.load_ply(state["final_path"])
    if init.n != state["n_final"] or init.sh_degree != 3:
        raise RuntimeError("tail parent final PLY has invalid fixed-topology geometry")

    print("[targets] replaying deterministic-v2 compact targets twice", flush=True)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        scene, replayed_targets = _materialize_training_scene(dataset, config)
        independently_replayed, alpha_replay = _independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed_targets,
        )
        target_replay = _verify_target_replay(
            parent / "compact_targets.json",
            replayed_targets,
            independently_replayed,
        )
    if torch.are_deterministic_algorithms_enabled() != previous_deterministic:
        raise RuntimeError("tail target replay did not restore deterministic algorithms")
    target_replay["prior_deterministic_algorithms_setting_restored"] = True
    target_replay["deterministic_algorithms_setting_after_replay"] = previous_deterministic
    target_replay["alpha_replay"] = alpha_replay
    if target_replay.get("reference_schema") != DETERMINISTIC_TARGET_SCHEMA:
        raise RuntimeError("tail replay did not consume deterministic-v2 parent targets")
    if target_replay.get("original_tensor_equivalence_verified") is not True:
        raise RuntimeError("tail deterministic target tensors differ from the parent")
    prior_target_replay = state["polish_start"].get("target_replay")
    if not isinstance(prior_target_replay, dict) or target_replay.get(
        "deterministic_recovery_identity_sha256"
    ) != prior_target_replay.get("deterministic_recovery_identity_sha256"):
        raise RuntimeError("tail deterministic targets differ from the parent polish replay")
    prior_alpha_replay = prior_target_replay.get("alpha_replay")
    if not isinstance(prior_alpha_replay, dict) or alpha_replay.get(
        "identity_sha256"
    ) != prior_alpha_replay.get("identity_sha256"):
        raise RuntimeError("tail alpha replay differs from the parent polish replay")

    for name, record in state["artifacts"].items():
        if _sha256_file(parent / name) != record["sha256"]:
            raise RuntimeError(f"tail parent artifact changed during preflight: {name}")

    trainer_config = _tail_train_config(state["trainer_config"])
    if _resolve_schedule_iterations(trainer_config) != TAIL_SCHEDULE_ITERATIONS:
        raise RuntimeError("internal tail schedule resolution differs from 50,000 steps")
    sh_interval = _resolve_sh_interval(trainer_config)
    first_active_degree = min(
        trainer_config.target_sh_degree,
        trainer_config.iteration_offset // sh_interval,
    )
    if first_active_degree != 3:
        raise RuntimeError("tail SH degree is not active from its first global step")
    opacity_entry_plan = _opacity_entry_plan(init, trainer_config.opacity_logit_epsilon)
    trainer_entry_record: dict[str, Any] = {}
    initialization_callback = _opacity_entry_verifier(
        init,
        trainer_config.opacity_logit_epsilon,
        trainer_entry_record,
    )

    out.mkdir(parents=True, exist_ok=False)
    for name in POLISH_COPIED_ARTIFACTS:
        _copy_file_new(parent / name, out / name)
    _copy_file_new(parent / "compact_targets.json", out / "compact_targets.json")
    _copy_file_new(state["final_path"], out / "gaussians_init.ply")
    _write_json_new(out / "training_config.json", dataclasses.asdict(trainer_config))

    receipt_path = out / "tail_start.json"
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.tail_start.v1",
        "created_utc": _utc_now(),
        "written_before_tail_training": True,
        "continuation_exact": False,
        "continuation_exact_reason": (
            "The selected parent PLY preserves Gaussian parameters but not Adam moments, "
            "per-parameter step counters, or the parent RNG stream. The tail is a new "
            "fixed-topology optimizer segment."
        ),
        "parent_output": str(parent),
        "output": str(out),
        "parent_selected_final": {
            "path": str(state["final_path"]),
            "sha256": state["final_sha256"],
            "bytes": state["final_path"].stat().st_size,
            "global_step": TAIL_PARENT_ITERATIONS,
            "n_gaussians": init.n,
            "sh_degree": init.sh_degree,
        },
        "parent_model_selection": {
            "path": str((parent / "model_selection.json").resolve()),
            "sha256": _sha256_file(parent / "model_selection.json"),
            "selected_global_step": TAIL_PARENT_ITERATIONS,
            "selected_candidate_sha256": state["final_sha256"],
            "joint_convergence_status": "still_improving",
        },
        "parent_fit_complete": {
            "path": str((parent / "fit_complete.json").resolve()),
            "sha256": _sha256_file(parent / "fit_complete.json"),
        },
        "global_steps": {
            "parent_last": TAIL_PARENT_ITERATIONS,
            "first_tail": TAIL_PARENT_ITERATIONS + 1,
            "last_tail": TAIL_SCHEDULE_ITERATIONS,
            "segment_iterations": TAIL_ITERATIONS,
            "schedule_iterations": TAIL_SCHEDULE_ITERATIONS,
        },
        "fixed_topology": {
            "enabled": True,
            "densify": False,
            "source_n_gaussians": init.n,
        },
        "sh_schedule": {
            "target_degree": trainer_config.target_sh_degree,
            "degree_interval": sh_interval,
            "first_active_degree": first_active_degree,
            "active_from_first_tail_step": True,
        },
        "learning_rates": {
            "preserved_from_parent_segment": True,
            "means_base": trainer_config.lr_means,
            "means_final_factor": trainer_config.means_lr_final_factor,
            "means_constant_during_tail": True,
            "quats": trainer_config.lr_quats,
            "scales": trainer_config.lr_scales,
            "opacity": trainer_config.lr_opacity,
            "sh": trainer_config.lr_sh,
            "sh_rest": trainer_config.lr_sh_rest,
        },
        "rng": {
            "parent_seed": state["trainer_config"].seed,
            "tail_seed": trainer_config.seed,
            "new_deterministic_stream": True,
        },
        "parent_target_lineage": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "sha256": _sha256_file(parent / "compact_targets.json"),
            "original_tensor_equivalence_verified": True,
            "independent_deterministic_replay_verified": True,
        },
        "child_target_receipt": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "path": str((out / "compact_targets.json").resolve()),
            "sha256": _sha256_file(out / "compact_targets.json"),
            "byte_copied_from_parent": True,
        },
        "opacity_entry_plan": opacity_entry_plan,
        "provenance_replay": provenance_replay,
        "target_replay": target_replay,
        "parent_artifacts": state["artifacts"],
        "child_frozen_artifacts": {
            name: {
                "path": str((out / name).resolve()),
                "sha256": _sha256_file(out / name),
            }
            for name in (
                "config.json",
                "provenance.json",
                "compact_targets.json",
                "training_config.json",
                "gaussians_init.ply",
            )
        },
        "tail_training_config": dataclasses.asdict(trainer_config),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_revision": _git_revision(),
        },
    }
    _write_json_new(receipt_path, receipt)
    print(
        "[tail] immutable receipt=tail_start.json continuation_exact=false "
        "global_steps=40001..50000 fixed_topology=true",
        flush=True,
    )
    _train_and_finalize(
        out,
        config,
        scene,
        init,
        trainer_config,
        tail_receipt=receipt_path,
        initialization_callback=initialization_callback,
        trainer_entry_record=trainer_entry_record,
        selection_baseline_step=TAIL_PARENT_ITERATIONS,
        selection_final_step=TAIL_SCHEDULE_ITERATIONS,
    )


def _cooldown(args: argparse.Namespace) -> None:
    if args.parent_out is None:  # parse_args enforces this for CLI callers.
        raise ValueError("cooldown requires --parent-out")
    state = _cooldown_parent_preflight(args.parent_out, args.out)
    parent = state["parent"]
    out = state["out"]
    config = state["config"]
    provenance = state["provenance"]

    dataset = CompactDataset.load(Path(config["dataset"]).resolve(), device="cpu")
    if tuple(view.view_id for view in dataset.views) != tuple(config["view_names"]):
        raise RuntimeError("current compact dataset view order differs from the cooldown parent")
    manifest_path = dataset.path / "manifest.json"
    if _sha256_file(manifest_path) != provenance.get("manifest_sha256"):
        raise RuntimeError("current compact manifest differs from the cooldown parent")
    if dataset.calibration_sha256 != provenance.get("calibration_sha256"):
        raise RuntimeError("current compact calibration differs from the cooldown parent")
    current_provenance = _compact_provenance(dataset, config)
    provenance_replay = _verify_provenance_identity(provenance, current_provenance)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen cooldown parent config")

    init = Gaussians3D.load_ply(state["final_path"])
    if init.n != state["n_final"] or init.sh_degree != 3:
        raise RuntimeError("cooldown parent final PLY has invalid fixed-topology geometry")

    print("[targets] replaying deterministic-v2 compact targets twice", flush=True)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        scene, replayed_targets = _materialize_training_scene(dataset, config)
        independently_replayed, alpha_replay = _independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed_targets,
        )
        target_replay = _verify_target_replay(
            parent / "compact_targets.json",
            replayed_targets,
            independently_replayed,
        )
    if torch.are_deterministic_algorithms_enabled() != previous_deterministic:
        raise RuntimeError("cooldown target replay did not restore deterministic algorithms")
    target_replay["prior_deterministic_algorithms_setting_restored"] = True
    target_replay["deterministic_algorithms_setting_after_replay"] = previous_deterministic
    target_replay["alpha_replay"] = alpha_replay
    if target_replay.get("reference_schema") != DETERMINISTIC_TARGET_SCHEMA:
        raise RuntimeError("cooldown replay did not consume deterministic-v2 parent targets")
    if target_replay.get("original_tensor_equivalence_verified") is not True:
        raise RuntimeError("cooldown deterministic target tensors differ from the parent")
    prior_target_replay = state["tail_start"].get("target_replay")
    if not isinstance(prior_target_replay, dict) or target_replay.get(
        "deterministic_recovery_identity_sha256"
    ) != prior_target_replay.get("deterministic_recovery_identity_sha256"):
        raise RuntimeError("cooldown deterministic targets differ from the parent replay")
    prior_alpha_replay = prior_target_replay.get("alpha_replay")
    if not isinstance(prior_alpha_replay, dict) or alpha_replay.get(
        "identity_sha256"
    ) != prior_alpha_replay.get("identity_sha256"):
        raise RuntimeError("cooldown alpha replay differs from the parent replay")

    for name, record in state["artifacts"].items():
        if _sha256_file(parent / name) != record["sha256"]:
            raise RuntimeError(f"cooldown parent artifact changed during preflight: {name}")

    trainer_config = _cooldown_train_config(state["trainer_config"])
    if _resolve_schedule_iterations(trainer_config) != COOLDOWN_SCHEDULE_ITERATIONS:
        raise RuntimeError("internal cooldown schedule resolution differs from 60,000 steps")
    sh_interval = _resolve_sh_interval(trainer_config)
    first_active_degree = min(
        trainer_config.target_sh_degree,
        trainer_config.iteration_offset // sh_interval,
    )
    if first_active_degree != 3:
        raise RuntimeError("cooldown SH degree is not active from its first global step")
    opacity_entry_plan = _opacity_entry_plan(init, trainer_config.opacity_logit_epsilon)
    trainer_entry_record: dict[str, Any] = {}
    initialization_callback = _opacity_entry_verifier(
        init,
        trainer_config.opacity_logit_epsilon,
        trainer_entry_record,
    )

    out.mkdir(parents=True, exist_ok=False)
    for name in POLISH_COPIED_ARTIFACTS:
        _copy_file_new(parent / name, out / name)
    _copy_file_new(parent / "compact_targets.json", out / "compact_targets.json")
    _copy_file_new(state["final_path"], out / "gaussians_init.ply")
    _write_json_new(out / "training_config.json", dataclasses.asdict(trainer_config))

    receipt_path = out / "cooldown_start.json"
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.cooldown_start.v1",
        "created_utc": _utc_now(),
        "written_before_cooldown_training": True,
        "continuation_exact": False,
        "continuation_exact_reason": (
            "The selected parent PLY preserves Gaussian parameters but not Adam moments, "
            "per-parameter step counters, or the parent RNG stream. The cooldown is a new "
            "lower-rate fixed-topology optimizer segment."
        ),
        "parent_output": str(parent),
        "output": str(out),
        "parent_selected_final": {
            "path": str(state["final_path"]),
            "sha256": state["final_sha256"],
            "bytes": state["final_path"].stat().st_size,
            "global_step": COOLDOWN_PARENT_ITERATIONS,
            "n_gaussians": init.n,
            "sh_degree": init.sh_degree,
        },
        "parent_model_selection": {
            "path": str((parent / "model_selection.json").resolve()),
            "sha256": _sha256_file(parent / "model_selection.json"),
            "selected_global_step": COOLDOWN_PARENT_ITERATIONS,
            "selected_candidate_sha256": state["final_sha256"],
            "joint_convergence_status": "still_improving",
        },
        "parent_fit_complete": {
            "path": str((parent / "fit_complete.json").resolve()),
            "sha256": _sha256_file(parent / "fit_complete.json"),
        },
        "global_steps": {
            "parent_last": COOLDOWN_PARENT_ITERATIONS,
            "first_cooldown": COOLDOWN_PARENT_ITERATIONS + 1,
            "last_cooldown": COOLDOWN_SCHEDULE_ITERATIONS,
            "segment_iterations": COOLDOWN_ITERATIONS,
            "schedule_iterations": COOLDOWN_SCHEDULE_ITERATIONS,
        },
        "fixed_topology": {
            "enabled": True,
            "densify": False,
            "source_n_gaussians": init.n,
        },
        "sh_schedule": {
            "target_degree": trainer_config.target_sh_degree,
            "degree_interval": sh_interval,
            "first_active_degree": first_active_degree,
            "active_from_first_cooldown_step": True,
        },
        "learning_rates": {
            "parent_to_cooldown_factor": COOLDOWN_LR_FACTOR,
            "constant_within_cooldown": True,
            "parent": {
                "means": state["trainer_config"].lr_means,
                "quats": state["trainer_config"].lr_quats,
                "scales": state["trainer_config"].lr_scales,
                "opacity": state["trainer_config"].lr_opacity,
                "sh": state["trainer_config"].lr_sh,
                "sh_rest": state["trainer_config"].lr_sh_rest,
            },
            "cooldown": {
                "means": trainer_config.lr_means,
                "quats": trainer_config.lr_quats,
                "scales": trainer_config.lr_scales,
                "opacity": trainer_config.lr_opacity,
                "sh": trainer_config.lr_sh,
                "sh_rest": trainer_config.lr_sh_rest,
            },
        },
        "rng": {
            "parent_seed": state["trainer_config"].seed,
            "cooldown_seed": trainer_config.seed,
            "new_deterministic_stream": True,
        },
        "parent_target_lineage": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "sha256": _sha256_file(parent / "compact_targets.json"),
            "original_tensor_equivalence_verified": True,
            "independent_deterministic_replay_verified": True,
        },
        "child_target_receipt": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "path": str((out / "compact_targets.json").resolve()),
            "sha256": _sha256_file(out / "compact_targets.json"),
            "byte_copied_from_parent": True,
        },
        "opacity_entry_plan": opacity_entry_plan,
        "provenance_replay": provenance_replay,
        "target_replay": target_replay,
        "parent_artifacts": state["artifacts"],
        "child_frozen_artifacts": {
            name: {
                "path": str((out / name).resolve()),
                "sha256": _sha256_file(out / name),
            }
            for name in (
                "config.json",
                "provenance.json",
                "compact_targets.json",
                "training_config.json",
                "gaussians_init.ply",
            )
        },
        "cooldown_training_config": dataclasses.asdict(trainer_config),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_revision": _git_revision(),
        },
    }
    _write_json_new(receipt_path, receipt)
    print(
        "[cooldown] immutable receipt=cooldown_start.json continuation_exact=false "
        "global_steps=50001..60000 fixed_topology=true lr_factor=0.25",
        flush=True,
    )
    _train_and_finalize(
        out,
        config,
        scene,
        init,
        trainer_config,
        cooldown_receipt=receipt_path,
        initialization_callback=initialization_callback,
        trainer_entry_record=trainer_entry_record,
        selection_baseline_step=COOLDOWN_PARENT_ITERATIONS,
        selection_final_step=COOLDOWN_SCHEDULE_ITERATIONS,
    )


def _settle(args: argparse.Namespace) -> None:
    if args.parent_out is None:  # parse_args enforces this for CLI callers.
        raise ValueError("settle requires --parent-out")
    state = _settle_parent_preflight(args.parent_out, args.out)
    parent = state["parent"]
    out = state["out"]
    config = state["config"]
    provenance = state["provenance"]

    dataset = CompactDataset.load(Path(config["dataset"]).resolve(), device="cpu")
    if tuple(view.view_id for view in dataset.views) != tuple(config["view_names"]):
        raise RuntimeError("current compact dataset view order differs from the settle parent")
    manifest_path = dataset.path / "manifest.json"
    if _sha256_file(manifest_path) != provenance.get("manifest_sha256"):
        raise RuntimeError("current compact manifest differs from the settle parent")
    if dataset.calibration_sha256 != provenance.get("calibration_sha256"):
        raise RuntimeError("current compact calibration differs from the settle parent")
    current_provenance = _compact_provenance(dataset, config)
    provenance_replay = _verify_provenance_identity(provenance, current_provenance)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen settle parent config")

    init = Gaussians3D.load_ply(state["final_path"])
    if init.n != state["n_final"] or init.sh_degree != 3:
        raise RuntimeError("settle parent final PLY has invalid fixed-topology geometry")

    print("[targets] replaying deterministic-v2 compact targets twice", flush=True)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        scene, replayed_targets = _materialize_training_scene(dataset, config)
        independently_replayed, alpha_replay = _independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed_targets,
        )
        target_replay = _verify_target_replay(
            parent / "compact_targets.json",
            replayed_targets,
            independently_replayed,
        )
    if torch.are_deterministic_algorithms_enabled() != previous_deterministic:
        raise RuntimeError("settle target replay did not restore deterministic algorithms")
    target_replay["prior_deterministic_algorithms_setting_restored"] = True
    target_replay["deterministic_algorithms_setting_after_replay"] = previous_deterministic
    target_replay["alpha_replay"] = alpha_replay
    if target_replay.get("reference_schema") != DETERMINISTIC_TARGET_SCHEMA:
        raise RuntimeError("settle replay did not consume deterministic-v2 parent targets")
    if target_replay.get("original_tensor_equivalence_verified") is not True:
        raise RuntimeError("settle deterministic target tensors differ from the parent")
    prior_target_replay = state["cooldown_start"].get("target_replay")
    if not isinstance(prior_target_replay, dict) or target_replay.get(
        "deterministic_recovery_identity_sha256"
    ) != prior_target_replay.get("deterministic_recovery_identity_sha256"):
        raise RuntimeError("settle deterministic targets differ from the parent replay")
    prior_alpha_replay = prior_target_replay.get("alpha_replay")
    if not isinstance(prior_alpha_replay, dict) or alpha_replay.get(
        "identity_sha256"
    ) != prior_alpha_replay.get("identity_sha256"):
        raise RuntimeError("settle alpha replay differs from the parent replay")

    for name, record in state["artifacts"].items():
        if _sha256_file(parent / name) != record["sha256"]:
            raise RuntimeError(f"settle parent artifact changed during preflight: {name}")

    trainer_config = _settle_train_config(state["trainer_config"])
    if _resolve_schedule_iterations(trainer_config) != SETTLE_SCHEDULE_ITERATIONS:
        raise RuntimeError("internal settle schedule resolution differs from 70,000 steps")
    sh_interval = _resolve_sh_interval(trainer_config)
    first_active_degree = min(
        trainer_config.target_sh_degree,
        trainer_config.iteration_offset // sh_interval,
    )
    if first_active_degree != 3:
        raise RuntimeError("settle SH degree is not active from its first global step")
    opacity_entry_plan = _opacity_entry_plan(init, trainer_config.opacity_logit_epsilon)
    trainer_entry_record: dict[str, Any] = {}
    initialization_callback = _opacity_entry_verifier(
        init,
        trainer_config.opacity_logit_epsilon,
        trainer_entry_record,
    )

    out.mkdir(parents=True, exist_ok=False)
    for name in POLISH_COPIED_ARTIFACTS:
        _copy_file_new(parent / name, out / name)
    _copy_file_new(parent / "compact_targets.json", out / "compact_targets.json")
    _copy_file_new(state["final_path"], out / "gaussians_init.ply")
    _write_json_new(out / "training_config.json", dataclasses.asdict(trainer_config))

    receipt_path = out / "settle_start.json"
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.settle_start.v1",
        "created_utc": _utc_now(),
        "written_before_settle_training": True,
        "continuation_exact": False,
        "continuation_exact_reason": (
            "The selected parent PLY preserves Gaussian parameters but not Adam moments, "
            "per-parameter step counters, or the parent RNG stream. Settle is a new "
            "lower-rate fixed-topology optimizer segment."
        ),
        "parent_output": str(parent),
        "output": str(out),
        "parent_selected_final": {
            "path": str(state["final_path"]),
            "sha256": state["final_sha256"],
            "bytes": state["final_path"].stat().st_size,
            "global_step": SETTLE_PARENT_ITERATIONS,
            "n_gaussians": init.n,
            "sh_degree": init.sh_degree,
        },
        "parent_model_selection": {
            "path": str((parent / "model_selection.json").resolve()),
            "sha256": _sha256_file(parent / "model_selection.json"),
            "selected_global_step": SETTLE_PARENT_ITERATIONS,
            "selected_candidate_sha256": state["final_sha256"],
            "joint_convergence_status": "still_improving",
        },
        "parent_fit_complete": {
            "path": str((parent / "fit_complete.json").resolve()),
            "sha256": _sha256_file(parent / "fit_complete.json"),
        },
        "global_steps": {
            "parent_last": SETTLE_PARENT_ITERATIONS,
            "first_settle": SETTLE_PARENT_ITERATIONS + 1,
            "last_settle": SETTLE_SCHEDULE_ITERATIONS,
            "segment_iterations": SETTLE_ITERATIONS,
            "schedule_iterations": SETTLE_SCHEDULE_ITERATIONS,
        },
        "fixed_topology": {
            "enabled": True,
            "densify": False,
            "source_n_gaussians": init.n,
        },
        "sh_schedule": {
            "target_degree": trainer_config.target_sh_degree,
            "degree_interval": sh_interval,
            "first_active_degree": first_active_degree,
            "active_from_first_settle_step": True,
        },
        "learning_rates": {
            "parent_to_settle_factor": SETTLE_LR_FACTOR,
            "constant_within_settle": True,
            "parent": {
                "means": state["trainer_config"].lr_means,
                "quats": state["trainer_config"].lr_quats,
                "scales": state["trainer_config"].lr_scales,
                "opacity": state["trainer_config"].lr_opacity,
                "sh": state["trainer_config"].lr_sh,
                "sh_rest": state["trainer_config"].lr_sh_rest,
            },
            "settle": {
                "means": trainer_config.lr_means,
                "quats": trainer_config.lr_quats,
                "scales": trainer_config.lr_scales,
                "opacity": trainer_config.lr_opacity,
                "sh": trainer_config.lr_sh,
                "sh_rest": trainer_config.lr_sh_rest,
            },
        },
        "rng": {
            "parent_seed": state["trainer_config"].seed,
            "settle_seed": trainer_config.seed,
            "new_deterministic_stream": True,
        },
        "parent_target_lineage": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "sha256": _sha256_file(parent / "compact_targets.json"),
            "original_tensor_equivalence_verified": True,
            "independent_deterministic_replay_verified": True,
        },
        "child_target_receipt": {
            "schema": DETERMINISTIC_TARGET_SCHEMA,
            "path": str((out / "compact_targets.json").resolve()),
            "sha256": _sha256_file(out / "compact_targets.json"),
            "byte_copied_from_parent": True,
        },
        "opacity_entry_plan": opacity_entry_plan,
        "provenance_replay": provenance_replay,
        "target_replay": target_replay,
        "parent_artifacts": state["artifacts"],
        "child_frozen_artifacts": {
            name: {
                "path": str((out / name).resolve()),
                "sha256": _sha256_file(out / name),
            }
            for name in (
                "config.json",
                "provenance.json",
                "compact_targets.json",
                "training_config.json",
                "gaussians_init.ply",
            )
        },
        "settle_training_config": dataclasses.asdict(trainer_config),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_revision": _git_revision(),
        },
    }
    _write_json_new(receipt_path, receipt)
    print(
        "[settle] immutable receipt=settle_start.json continuation_exact=false "
        "global_steps=60001..70000 fixed_topology=true lr_factor=0.25",
        flush=True,
    )
    _train_and_finalize(
        out,
        config,
        scene,
        init,
        trainer_config,
        settle_receipt=receipt_path,
        initialization_callback=initialization_callback,
        trainer_entry_record=trainer_entry_record,
        selection_baseline_step=SETTLE_PARENT_ITERATIONS,
        selection_final_step=SETTLE_SCHEDULE_ITERATIONS,
    )


def _recover(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if not out.is_dir():
        raise RuntimeError(f"recovery requires an existing run directory: {out}")
    existing_final = [name for name in FINAL_FIT_ARTIFACTS if (out / name).exists()]
    if existing_final:
        raise FileExistsError(
            f"refusing recovery because final fit artifacts already exist: {existing_final}"
        )
    required_names = (
        "config.json",
        "provenance.json",
        "training_config.json",
        "compact_targets.json",
    )
    missing = [name for name in required_names if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"recovery run is missing frozen artifacts: {missing}")
    if args.resume_checkpoint is None:  # parse_args enforces this for CLI callers.
        raise ValueError("recover requires --resume-checkpoint")
    checkpoint_path, resume_step = _resume_checkpoint(out, args.resume_checkpoint)

    config_path = out / "config.json"
    provenance_path = out / "provenance.json"
    training_config_path = out / "training_config.json"
    target_path = out / "compact_targets.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    original_trainer_config = _load_train_config(training_config_path)
    frozen_total = config.get("iterations")
    if isinstance(frozen_total, bool) or not isinstance(frozen_total, int):
        raise RuntimeError("frozen config iterations must be an integer")
    schedule_total = _resolve_schedule_iterations(original_trainer_config)
    if original_trainer_config.iteration_offset != 0:
        raise RuntimeError("recovery requires an original one-shot training config")
    if schedule_total != frozen_total or original_trainer_config.iterations != frozen_total:
        raise RuntimeError("frozen run and training-config iteration totals differ")
    if not 0 < resume_step < frozen_total:
        raise RuntimeError("resume checkpoint step must be strictly inside the frozen run")

    dataset = CompactDataset.load(Path(config["dataset"]).resolve(), device="cpu")
    if tuple(view.view_id for view in dataset.views) != tuple(config["view_names"]):
        raise RuntimeError("current compact dataset view order differs from the frozen fit")
    manifest_path = dataset.path / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != provenance.get("manifest_sha256"):
        raise RuntimeError("current compact manifest differs from the fit provenance")
    if dataset.calibration_sha256 != provenance.get("calibration_sha256"):
        raise RuntimeError("current compact calibration differs from the fit provenance")
    current_provenance = _compact_provenance(dataset, config)
    provenance_replay = _verify_provenance_identity(provenance, current_provenance)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen recovery training config")

    init = Gaussians3D.load_ply(checkpoint_path)
    if init.n > original_trainer_config.density.max_gaussians:
        raise RuntimeError("resume checkpoint exceeds the frozen density-control Gaussian budget")
    print(
        f"[recover] checkpoint_step={resume_step} gaussians={init.n} "
        f"remaining={frozen_total - resume_step}",
        flush=True,
    )
    print("[targets] replaying and verifying every native StructSplat crop", flush=True)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        scene, replayed_targets = _materialize_training_scene(dataset, config)
        independently_replayed, alpha_replay = _independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed_targets,
        )
        target_replay = _verify_target_replay(
            target_path,
            replayed_targets,
            independently_replayed,
        )
    if torch.are_deterministic_algorithms_enabled() != previous_deterministic:
        raise RuntimeError("target replay did not restore the torch deterministic setting")
    target_replay["prior_deterministic_algorithms_setting_restored"] = True
    target_replay["deterministic_algorithms_setting_after_replay"] = previous_deterministic
    target_replay["alpha_replay"] = alpha_replay
    print(
        "[targets] frozen_raw_hashes="
        f"{target_replay['frozen_raw_crop_hash_match_count']}/"
        f"{target_replay['frozen_raw_crop_hash_view_count']} "
        "original_tensor_equivalence_verified="
        f"{str(target_replay['original_tensor_equivalence_verified']).lower()}",
        flush=True,
    )

    trainer_config = dataclasses.replace(
        original_trainer_config,
        iterations=frozen_total - resume_step,
        iteration_offset=resume_step,
        schedule_iterations=frozen_total,
    )
    sh_interval = _resolve_sh_interval(trainer_config)
    means_gamma = 0.01 ** (1.0 / frozen_total)
    receipt_path = _next_recovery_receipt(out)
    receipt = {
        "schema": "rtgs.full_compact_reconstruction.recovery_attempt.v2",
        "created_utc": _utc_now(),
        "written_before_recovery_training": True,
        "resume_exact": False,
        "resume_exact_reason": (
            "PLY checkpoints contain Gaussian parameters only; optimizer, RNG-stream, and "
            "density-controller state cannot be reconstructed."
        ),
        "output": str(out),
        "resume_step": resume_step,
        "first_recovered_step": resume_step + 1,
        "last_recovered_step": frozen_total,
        "schedule_iterations": frozen_total,
        "remaining_iterations": frozen_total - resume_step,
        "checkpoint": {
            "absolute_path": str(checkpoint_path),
            "relative_path": str(checkpoint_path.relative_to(out)),
            "sha256": _sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "n_gaussians": init.n,
            "sh_degree": init.sh_degree,
        },
        "lost_state": {
            "optimizer": "Adam moments and per-parameter step counters reset",
            "rng": (
                "the configured seed restarts a new generator; the prior generator stream "
                "position is unrecoverable"
            ),
            "density_controller": (
                "screen-space gradient accumulators and gsplat strategy state reset"
            ),
        },
        "preserved_schedules": {
            "global_step_numbering": [resume_step + 1, frozen_total],
            "sh_degree_interval": sh_interval,
            "first_active_sh_degree": min(
                trainer_config.target_sh_degree,
                resume_step // sh_interval,
            ),
            "means_lr_gamma": means_gamma,
            "means_lr_offset_multiplier": means_gamma**resume_step,
            "density_start": trainer_config.density.start_iter,
            "density_stop": trainer_config.density.stop_iter,
            "density_every": trainer_config.density.every,
        },
        "provenance_replay": provenance_replay,
        "target_replay": target_replay,
        "frozen_artifacts": {
            name: {
                "path": str((out / name).resolve()),
                "sha256": _sha256_file(out / name),
            }
            for name in required_names
        },
        "recovery_training_config": dataclasses.asdict(trainer_config),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_revision": _git_revision(),
        },
    }
    _write_json_new(receipt_path, receipt)
    print(
        f"[recover] immutable receipt={receipt_path.relative_to(out)} resume_exact=false",
        flush=True,
    )
    _train_and_finalize(
        out,
        config,
        scene,
        init,
        trainer_config,
        recovery_receipt=receipt_path,
    )


def _assert_camera_match(expected: Camera, actual: Camera, view_name: str) -> None:
    scalars = ("fx", "fy", "cx", "cy")
    for name in scalars:
        if not math.isclose(
            float(getattr(expected, name)),
            float(getattr(actual, name)),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(f"{view_name} camera {name} mismatch")
    if (expected.width, expected.height) != (actual.width, actual.height):
        raise RuntimeError(f"{view_name} camera resolution mismatch")
    if not torch.allclose(expected.R, actual.R, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"{view_name} camera rotation mismatch")
    if not torch.allclose(expected.t, actual.t, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"{view_name} camera translation mismatch")


def _aggregate_records(records: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    values = [record[key] for record in records]
    metric_names = sorted(set.intersection(*(set(value) for value in values)))
    return {
        metric: sum(float(value[metric]) for value in values) / len(values)
        for metric in metric_names
    }


def _save_h_preview(
    path: Path,
    original: torch.Tensor,
    compact: torch.Tensor,
    final: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    if path.exists():
        raise FileExistsError(f"refusing to overwrite preview: {path}")
    panels = [original, compact, final, alpha[..., None].expand(-1, -1, 3)]
    labels = ("original", "compact playback", "final 3D", "final alpha")
    images = []
    for panel in panels:
        array = (panel.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        image = Image.fromarray(array)
        scale = min(1.0, 1000.0 / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        images.append(image)
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + 32
    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in zip(labels, images, strict=True):
        canvas.paste(image, (x, 32))
        draw.text((x + 6, 8), label, fill="white")
        x += image.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _evaluate(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    final_path = out / "gaussians_final.ply"
    fit_complete = out / "fit_complete.json"
    if not final_path.is_file() or not fit_complete.is_file():
        raise RuntimeError("evaluation requires a completed fit and gaussians_final.ply")
    if (out / "evaluation_complete.json").exists() or (out / "run_complete.json").exists():
        raise FileExistsError("refusing to overwrite a completed evaluation")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    dataset = CompactDataset.load(Path(config["dataset"]), device="cpu")
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    manifest_sha256 = _sha256_file(dataset.path / "manifest.json")
    if provenance["manifest_sha256"] != manifest_sha256:
        raise RuntimeError("compact dataset manifest differs from fit provenance")
    fit_receipt = json.loads(fit_complete.read_text(encoding="utf-8"))
    final_sha256 = _sha256_file(final_path)
    if fit_receipt.get("final_ply_sha256") != final_sha256:
        raise RuntimeError("final PLY differs from the completed fit receipt")
    if fit_receipt.get("fit_kind") in (
        "fixed_topology_polish",
        "fixed_topology_tail",
        "fixed_topology_cooldown",
        "fixed_topology_settle",
    ):
        selection_record = fit_receipt.get("model_selection")
        if not isinstance(selection_record, dict):
            raise RuntimeError("polished fit has no model-selection receipt binding")
        selection_relative = selection_record.get("receipt")
        if selection_relative != "model_selection.json":
            raise RuntimeError("polished fit model-selection receipt path is invalid")
        selection_path = out / selection_relative
        if not selection_path.is_file() or selection_record.get("receipt_sha256") != _sha256_file(
            selection_path
        ):
            raise RuntimeError("polished fit model-selection receipt hash differs")
        selection = _load_json_object(selection_path, label="model selection")
        if selection.get("schema") != MODEL_SELECTION_SCHEMA:
            raise RuntimeError("polished fit has an unsupported model-selection schema")
        selected = selection.get("selected")
        if not isinstance(selected, dict) or selected.get("sha256") != final_sha256:
            raise RuntimeError("polished final PLY differs from the selected checkpoint")
        if selection.get("written_before_gaussians_final") is not True:
            raise RuntimeError("model selection was not frozen before the polished final PLY")
    _write_json_new(
        out / "evaluation_request.json",
        {
            "schema": "rtgs.full_compact_reconstruction.evaluation_request.v1",
            "created_utc": _utc_now(),
            "written_before_source_rgb_access": True,
            "final_ply_sha256": final_sha256,
            "manifest_sha256": manifest_sha256,
            "original_frame": str(ORIGINAL_FRAME),
            "original_calibration": str(ORIGINAL_CALIBRATION),
        },
    )

    # Source RGB access begins only after the immutable request above and the final-Ply gate.
    if _sha256_file(ORIGINAL_CALIBRATION) != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("original calibration digest does not match compact provenance")
    for compact_view in dataset.views:
        rgb_record = compact_view.source["rgb"]
        mask_record = compact_view.source["mask"]
        rgb_path = ORIGINAL_FRAME / "rgb" / rgb_record["name"]
        mask_path = ORIGINAL_FRAME / "mask" / mask_record["name"]
        if _sha256_file(rgb_path) != rgb_record["sha256"]:
            raise RuntimeError(f"{compact_view.view_id} source RGB digest mismatch")
        if _sha256_file(mask_path) != mask_record["sha256"]:
            raise RuntimeError(f"{compact_view.view_id} source mask digest mismatch")
    from rtgs.data.calibrated import load_calibrated_scene
    from rtgs.render.base import get_rasterizer

    original_scene = load_calibrated_scene(
        ORIGINAL_FRAME,
        calibration_path=ORIGINAL_CALIBRATION,
        downscale=1,
        max_images=None,
        test_every=8,
        load_masks=True,
        undistort=True,
    )
    if tuple(original_scene.view_names or ()) != EXPECTED_VIEW_NAMES:
        raise RuntimeError("original scene view order differs from compact provenance")
    if original_scene.masks is None:
        raise RuntimeError("original scene is missing the provenance-matched masks")
    for compact_view, camera, name in zip(
        dataset.views,
        original_scene.cameras,
        EXPECTED_VIEW_NAMES,
        strict=True,
    ):
        _assert_camera_match(compact_view.camera, camera, name)

    device = torch.device(config["device"])
    final = Gaussians3D.load_ply(final_path).to(device)
    renderer = get_rasterizer(
        "gsplat",
        device=device,
        packed=True,
        antialiased=True,
    )
    records: list[dict[str, Any]] = []
    preview_dir = out / "H_previews"
    with torch.no_grad():
        for index, compact_view in enumerate(dataset.views):
            fit_x, fit_y, width, height = compact_view.observation.fit_window
            original = original_scene.images[index][
                fit_y : fit_y + height,
                fit_x : fit_x + width,
            ].to(device)
            original_mask = original_scene.masks[index][
                fit_y : fit_y + height,
                fit_x : fit_x + width,
            ].to(device)
            packed_mask = compact_view.alpha.crop_mask(device) if compact_view.alpha else None
            if packed_mask is None or not torch.equal(original_mask > 0.5, packed_mask):
                raise RuntimeError(f"{compact_view.view_id} original/packed alpha mismatch")
            compact_raw, _, _ = _render_compact_crop(
                compact_view,
                device=device,
                renderer_name=str(config["structsplat_renderer"]),
                chunk=int(config["structsplat_chunk"]),
                apply_alpha=False,
            )
            compact_raw = compact_raw.clamp(0, 1)
            crop_camera = _crop_camera(
                original_scene.cameras[index],
                compact_view.observation.fit_window,
            )
            final_output = renderer.render(final, crop_camera, sh_degree=3)
            compact_values = image_metrics(compact_raw, original, packed_mask)
            compact_masked_values = image_metrics(
                compact_raw * packed_mask[..., None],
                original,
                packed_mask,
            )
            final_values = image_metrics(final_output.color, original, packed_mask)
            foreground = packed_mask
            predicted = final_output.alpha > 0.5
            intersection = (foreground & predicted).sum()
            union = (foreground | predicted).sum().clamp_min(1)
            final_values.update(
                {
                    "alpha_iou": float(intersection / union),
                    "alpha_inside": float(final_output.alpha[foreground].mean()),
                    "alpha_outside": float(final_output.alpha[~foreground].mean()),
                }
            )
            role = "V" if index in V_INDICES else ("H" if index in H_INDICES else "T")
            record = {
                "index": index,
                "view_name": compact_view.view_id,
                "role": role,
                "was_fit": index in set(config["fit_indices"]),
                "compact_playback": compact_values,
                "compact_playback_alpha_applied": compact_masked_values,
                "final_3d": final_values,
            }
            records.append(record)
            if index in H_INDICES:
                _save_h_preview(
                    preview_dir / f"{index:02d}_{compact_view.view_id}.png",
                    original,
                    compact_raw,
                    final_output.color,
                    final_output.alpha,
                )
            del compact_raw, final_output, original, original_mask, packed_mask
    aggregates = {}
    for role in ("T", "V", "H"):
        role_records = [record for record in records if record["role"] == role]
        aggregates[role] = {
            "compact_playback": _aggregate_records(role_records, "compact_playback"),
            "compact_playback_alpha_applied": _aggregate_records(
                role_records,
                "compact_playback_alpha_applied",
            ),
            "final_3d": _aggregate_records(role_records, "final_3d"),
            "all_views_were_fit": all(record["was_fit"] for record in role_records),
        }
    metrics = {
        "schema": "rtgs.full_compact_reconstruction.original_evaluation.v1",
        "created_utc": _utc_now(),
        "source_rgb_evaluator_only": True,
        "fit_mode": config["fit_mode"],
        "no_heldout_claim_when_fit_mode_all": config["fit_mode"] == "all",
        "records": records,
        "aggregates": aggregates,
    }
    _write_json_new(out / "original_metrics.json", metrics)
    _write_json_new(
        out / "evaluation_complete.json",
        {
            "schema": "rtgs.full_compact_reconstruction.evaluation_complete.v1",
            "completed_utc": _utc_now(),
            "metrics": "original_metrics.json",
            "H_previews": "H_previews",
        },
    )
    _write_json_new(
        out / "run_complete.json",
        {
            "schema": "rtgs.full_compact_reconstruction.complete.v1",
            "completed_utc": _utc_now(),
            "final_ply_sha256": _sha256_file(final_path),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--phase",
        choices=(
            "fit",
            "recover",
            "polish",
            "tail",
            "cooldown",
            "settle",
            "evaluate",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--parent-out",
        type=Path,
        help=(
            "completed recovered 30k parent for polish, selected 40k child for tail, or "
            "selected 50k tail child for cooldown, or selected 60k cooldown child for settle"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="existing run/checkpoints/gaussians_step_XXXXXX.ply; valid only for recover",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("all", "protocol"),
        default="all",
        help="all fits 26/26 views; protocol fits T, validates V, and reserves H",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tracks", type=int, default=5_000)
    parser.add_argument("--depth-samples", type=int, default=32)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--robust-view-fraction", type=float, default=0.60)
    parser.add_argument("--min-placement-score", type=float, default=0.01)
    parser.add_argument("--init-opacity", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument(
        "--structsplat-renderer",
        choices=("cuda_tiled", "reference"),
        default="reference",
    )
    parser.add_argument("--structsplat-chunk", type=int, default=4096)
    parser.add_argument(
        "--density-strategy",
        choices=("classic", "gsplat-default", "gsplat-mcmc"),
        default="gsplat-default",
    )
    parser.add_argument("--no-densify", action="store_true")
    parser.add_argument("--densify-start", type=int, default=500)
    parser.add_argument("--densify-stop", type=int, default=15_000)
    parser.add_argument("--densify-every", type=int, default=100)
    parser.add_argument("--max-gaussians", type=int, default=100_000)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--prune-scale-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="force 32 tracks, two depth samples, one train iteration, and no density edits",
    )
    args = parser.parse_args(argv)
    if args.phase == "recover" and args.resume_checkpoint is None:
        parser.error("--phase recover requires --resume-checkpoint")
    if args.phase != "recover" and args.resume_checkpoint is not None:
        parser.error("--resume-checkpoint is valid only with --phase recover")
    if args.phase in {"polish", "tail", "cooldown", "settle"} and args.parent_out is None:
        parser.error(f"--phase {args.phase} requires --parent-out")
    if args.phase not in {"polish", "tail", "cooldown", "settle"} and (args.parent_out is not None):
        parser.error("--parent-out is valid only with --phase polish, tail, cooldown, or settle")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase != "recover" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested compact reconstruction")
    if args.phase in {"fit", "all"}:
        _fit(args)
    if args.phase == "recover":
        _recover(args)
    if args.phase == "polish":
        _polish(args)
    if args.phase == "tail":
        _tail(args)
    if args.phase == "cooldown":
        _cooldown(args)
    if args.phase == "settle":
        _settle(args)
    if args.phase in {"evaluate", "all"}:
        _evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
