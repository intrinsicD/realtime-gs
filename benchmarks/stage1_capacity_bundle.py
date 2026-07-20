#!/usr/bin/env python3
"""Promote one completed seven-view Stage-1 capacity sweep into an RGB-free bundle.

The bridge has two immutable CPU-only phases:

``plan``
    Strictly verifies the completed source sweep and acquires mask-derived object bounds from
    only the mask/camera bindings already frozen by that sweep.  It writes an integrity-bound
    bounds artifact and a bridge plan.  Source RGB is never resolved, opened, decoded, or copied.

``build``
    Re-verifies the source tree and every cell/teacher binding, loads the compact teachers
    strictly, and atomically writes a :class:`ReconstructionInputs` bundle.  It does not read
    masks or RGB in this phase.

The source sweep must contain exactly one arbitrary arm and exactly the frozen seven training
views.  Per-view optimized cardinalities remain variable ``m_opt,i^2D``; the bridge neither fits
teachers nor invents points or geometry beyond the explicitly bound mask-derived bounds hint.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _object_bounds, _resize_image, _undistort
from rtgs.data.reconstruction_inputs import ReconstructionInputs

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT
DEFAULT_SOURCE = REPOSITORY_ROOT / "runs/stage1_capacity_sweep_all_20260717"
DEFAULT_OUT = REPOSITORY_ROOT / "runs/stage1_capacity_bundle_20260717"

TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
HELDOUT_VIEW = "C1004"
SOURCE_PLAN_TYPE = "stage1_capacity_sweep_plan_v1"
SOURCE_CELL_TYPE = "stage1_capacity_sweep_cell_v1"
SOURCE_RESULT_TYPE = "stage1_capacity_sweep_result_v1"
PLAN_TYPE = "stage1_capacity_bundle_plan_v1"
RESULT_TYPE = "stage1_capacity_bundle_result_v1"
BOUNDS_RECEIPT_TYPE = "stage1_capacity_bundle_bounds_receipt_v1"
EXTERNAL_BOUNDS_TYPE = "stage1_capacity_bundle_external_bounds_v1"
FAILURE_TYPE = "stage1_capacity_bundle_failure_v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CAMERA_KEYS = frozenset({"fx", "fy", "cx", "cy", "width", "height", "R", "t"})
_TEACHER_MEMBERS = frozenset(
    {
        "metadata_utf8.npy",
        "means.npy",
        "log_scales.npy",
        "rotations.npy",
        "colors.npy",
        "amplitudes.npy",
        "mean_residuals.npy",
        "color_grads.npy",
        "filter_variance.npy",
    }
)
_REQUIRED_TEACHER_MEMBERS = frozenset(
    {
        "metadata_utf8.npy",
        "means.npy",
        "log_scales.npy",
        "rotations.npy",
        "colors.npy",
        "amplitudes.npy",
    }
)
_MAX_TEACHER_COMPRESSED_BYTES = 268_435_456
_MAX_TEACHER_MEMBER_BYTES = 268_435_456


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("refusing to serialize non-finite evidence")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        result = json.loads(
            path.read_bytes(),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict {label}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = canonical_bytes(_json_safe(value))
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_failure(
    out: Path,
    *,
    phase: str,
    error: BaseException,
    argv: Sequence[str] | None,
) -> None:
    failures = out / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact_type": FAILURE_TYPE,
        "status": "FAIL",
        "phase": phase,
        "argv": list(sys.argv if argv is None else argv),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    for index in range(1_000_000):
        path = failures / f"{index:06d}.json"
        try:
            _write_json_exclusive(path, record)
        except FileExistsError:
            continue
        return
    raise RuntimeError("failure receipt namespace exhausted")


def _ordinary_directory(path: Path, *, label: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be an ordinary directory")
    return path.resolve(strict=True)


def _strict_relative_file(root: Path, relative_text: str, *, label: str) -> Path:
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative_text
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{label} does not exist") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} must not contain symlinks")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its bound root") from error
    if not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise ValueError(f"{label} must name an ordinary file")
    return resolved


def _ordinary_file(path: Path, *, label: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be an ordinary file")
    return path.resolve(strict=True)


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"source sweep contains symlink {relative!r}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"source sweep contains non-regular entry {relative!r}")
        result[relative] = sha256_file(path)
    return result


def _command_output(*argv: str) -> str:
    return subprocess.check_output(argv, cwd=REPOSITORY_ROOT, text=True).strip()


def repository_source_binding() -> dict[str, Any]:
    paths = (
        Path("benchmarks/stage1_capacity_bundle.py"),
        Path("src/rtgs/core/camera.py"),
        Path("src/rtgs/core/observation2d.py"),
        Path("src/rtgs/data/calibrated.py"),
        Path("src/rtgs/data/reconstruction_inputs.py"),
    )
    hashes = {path.as_posix(): sha256_file(REPOSITORY_ROOT / path) for path in paths}
    return {
        "git_revision": _command_output("git", "rev-parse", "HEAD"),
        "files": hashes,
        "aggregate_sha256": canonical_hash(hashes),
    }


def environment_binding() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "device_policy": "cpu_only",
    }


def _validate_camera_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CAMERA_KEYS:
        raise ValueError(f"{label} camera keys are not exact")
    for name in ("fx", "fy", "cx", "cy"):
        scalar = value[name]
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
        ):
            raise ValueError(f"{label}.{name} must be finite")
    if float(value["fx"]) <= 0 or float(value["fy"]) <= 0:
        raise ValueError(f"{label} focal lengths must be positive")
    for name in ("width", "height"):
        scalar = value[name]
        if not isinstance(scalar, int) or isinstance(scalar, bool) or scalar <= 0:
            raise ValueError(f"{label}.{name} must be a positive integer")
    for name, count in (("R", 9), ("t", 3)):
        items = value[name]
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in items
            )
        ):
            raise ValueError(f"{label}.{name} must contain {count} finite values")
    return value


def _camera_from_record(value: dict[str, Any], *, label: str) -> Camera:
    record = _validate_camera_record(value, label=label)
    return Camera(
        fx=float(record["fx"]),
        fy=float(record["fy"]),
        cx=float(record["cx"]),
        cy=float(record["cy"]),
        width=int(record["width"]),
        height=int(record["height"]),
        R=torch.tensor(record["R"], dtype=torch.float32).reshape(3, 3),
        t=torch.tensor(record["t"], dtype=torch.float32),
    )


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


def observation_summary(field: GaussianObservationField) -> dict[str, Any]:
    tensors = {
        "means": field.means,
        "log_scales": field.log_scales,
        "rotations": field.rotations,
        "colors": field.colors,
        "amplitudes": field.amplitudes,
    }
    if field.mean_residuals is not None:
        tensors["mean_residuals"] = field.mean_residuals
    if field.color_grads is not None:
        tensors["color_grads"] = field.color_grads
    if field.filter_variance is not None:
        tensors["filter_variance"] = field.filter_variance
    return {
        "m_init_2d": field.n_init,
        "m_opt_2d": field.n,
        "canvas_size": [field.width, field.height],
        "fit_window": list(field.fit_window),
        "view_id": field.view_id,
        "provider": field.provider,
        "producer_version": field.producer_version,
        "producer_source_digest": field.producer_source_digest,
        "fit_config_digest": field.fit_config_digest,
        "blend_mode": field.blend_mode,
        "epsilon": field.epsilon,
        "sigma_cutoff": field.sigma_cutoff,
        "support_fade_alpha": field.support_fade_alpha,
        "aa_dilation": field.aa_dilation,
        "tensor_hashes": {name: tensor_hash(tensor) for name, tensor in tensors.items()},
    }


def _inspect_teacher(path: Path) -> None:
    if os.lstat(path).st_size > _MAX_TEACHER_COMPRESSED_BYTES:
        raise ValueError(f"teacher archive exceeds compressed byte cap: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid teacher NPZ: {path}") from error
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError(f"teacher archive has duplicate members: {path}")
    if not _REQUIRED_TEACHER_MEMBERS <= set(names) <= _TEACHER_MEMBERS:
        raise ValueError(f"teacher archive members are unsupported: {path}")
    for member in members:
        mode = member.external_attr >> 16
        if (
            member.is_dir()
            or "/" in member.filename
            or "\\" in member.filename
            or stat.S_ISLNK(mode)
            or member.flag_bits & 0x1
            or member.file_size > _MAX_TEACHER_MEMBER_BYTES
        ):
            raise ValueError(f"teacher archive contains unsafe member: {path}")


def _required_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(slots=True)
class VerifiedSweep:
    root: Path
    plan: dict[str, Any]
    result: dict[str, Any]
    plan_sha256: str
    result_sha256: str
    arm: dict[str, Any]
    records: list[dict[str, Any]]
    record_hashes: dict[str, str]
    observations: list[GaussianObservationField]
    cameras: list[Camera]
    tree_hashes: dict[str, str]

    @property
    def arm_name(self) -> str:
        return str(self.arm["name"])

    @property
    def camera_digest(self) -> str:
        return canonical_hash(
            [
                {"view_id": view_id, "camera": _camera_record(camera)}
                for view_id, camera in zip(TRAIN_VIEWS, self.cameras, strict=True)
            ]
        )

    @property
    def mask_binding_digest(self) -> str:
        bindings = []
        for view_id in TRAIN_VIEWS:
            record = self.plan["inputs"]["views"][view_id]
            bindings.append(
                {
                    "view_id": view_id,
                    "mask": record["mask"],
                    "undistorted_mask_tensor_sha256": record["undistorted_mask_tensor_sha256"],
                    "distortion": record["distortion"],
                    "native_resolution": record["native_resolution"],
                    "camera": record["camera"],
                }
            )
        return canonical_hash(bindings)


def _verify_result(
    root: Path,
    result: dict[str, Any],
    *,
    plan_sha256: str,
    arm_name: str,
) -> None:
    if result.get("artifact_type") != SOURCE_RESULT_TYPE or result.get("status") != "PASS":
        raise ValueError("source sweep result is not a completed PASS artifact")
    if result.get("plan_sha256") != plan_sha256:
        raise ValueError("source sweep result has a different plan hash")
    if result.get("selected_views") != list(TRAIN_VIEWS):
        raise ValueError("source sweep result views differ from the frozen training protocol")
    if result.get("heldout_view_excluded") != HELDOUT_VIEW:
        raise ValueError("source sweep result held-out policy changed")
    if result.get("completed_cell_count") != len(TRAIN_VIEWS):
        raise ValueError("source sweep does not contain exactly seven completed cells")
    if result.get("expected_cell_count") != len(TRAIN_VIEWS) or result.get("missing_cells") != []:
        raise ValueError("source sweep result is incomplete")
    rankings = _required_mapping(result.get("rankings"), label="source rankings")
    if set(rankings) != set(TRAIN_VIEWS):
        raise ValueError("source rankings have missing or extra views")
    for view_id in TRAIN_VIEWS:
        ranking = rankings[view_id]
        if (
            not isinstance(ranking, list)
            or len(ranking) != 1
            or not isinstance(ranking[0], dict)
            or ranking[0].get("arm") != arm_name
        ):
            raise ValueError(f"{view_id} does not have exactly the selected source arm")
    contact_sheets = _required_mapping(result.get("contact_sheets"), label="source contact sheets")
    if set(contact_sheets) != set(TRAIN_VIEWS):
        raise ValueError("source contact sheets have missing or extra views")
    for view_id in TRAIN_VIEWS:
        record = _required_mapping(contact_sheets[view_id], label=f"{view_id} contact sheet")
        path = _strict_relative_file(root, record.get("path"), label=f"{view_id} contact sheet")
        if sha256_file(path) != _require_sha(
            record.get("sha256"), label=f"{view_id} contact sheet hash"
        ):
            raise ValueError(f"{view_id} contact sheet digest mismatch")


def verify_source_sweep(source: str | Path) -> VerifiedSweep:
    """Strictly load a completed one-arm, seven-view Stage-1 sweep without raster access."""
    root = _ordinary_directory(Path(source).expanduser(), label="source sweep")
    tree_before = _tree_hashes(root)
    plan_path = _strict_relative_file(root, "plan.json", label="source plan")
    result_path = _strict_relative_file(root, "result.json", label="source result")
    plan_sha256 = sha256_file(plan_path)
    result_sha256 = sha256_file(result_path)
    plan = _load_json(plan_path, label="source plan")
    result = _load_json(result_path, label="source result")
    if plan.get("artifact_type") != SOURCE_PLAN_TYPE:
        raise ValueError("source sweep has the wrong plan artifact type")
    if plan.get("training_view_universe") != list(TRAIN_VIEWS):
        raise ValueError("source training-view universe changed")
    if plan.get("selected_views") != list(TRAIN_VIEWS):
        raise ValueError("source sweep must select all seven training views in frozen order")
    if plan.get("heldout_view_excluded") != HELDOUT_VIEW:
        raise ValueError("source sweep held-out policy changed")
    arms = plan.get("arms")
    if not isinstance(arms, list) or len(arms) != 1 or not isinstance(arms[0], dict):
        raise ValueError("source sweep must contain exactly one selected arm")
    arm = arms[0]
    arm_name = arm.get("name")
    if not isinstance(arm_name, str) or not arm_name or "/" in arm_name or "\\" in arm_name:
        raise ValueError("source arm has an unsafe name")
    inputs = _required_mapping(plan.get("inputs"), label="source inputs")
    view_inputs = _required_mapping(inputs.get("views"), label="source view inputs")
    if set(view_inputs) != set(TRAIN_VIEWS):
        raise ValueError("source plan inputs have missing or extra views")
    configs = _required_mapping(plan.get("effective_configs"), label="source effective configs")
    if set(configs) != set(TRAIN_VIEWS):
        raise ValueError("source effective configs have missing or extra views")
    repository = _required_mapping(plan.get("repository"), label="source repository binding")
    external = _required_mapping(
        plan.get("external_structsplat"), label="source StructSplat binding"
    )
    if not repository or not external:
        raise ValueError("source bindings must not be empty")
    _verify_result(root, result, plan_sha256=plan_sha256, arm_name=arm_name)

    expected_records = {f"records/{view_id}__{arm_name}.json" for view_id in TRAIN_VIEWS}
    expected_teachers = {f"teachers/{view_id}__{arm_name}.teacher.npz" for view_id in TRAIN_VIEWS}
    actual_records = {
        path.relative_to(root).as_posix() for path in (root / "records").rglob("*.json")
    }
    actual_teachers = {
        path.relative_to(root).as_posix() for path in (root / "teachers").rglob("*.npz")
    }
    if actual_records != expected_records:
        raise ValueError("source sweep records have missing or extra views/cells")
    if actual_teachers != expected_teachers:
        raise ValueError("source sweep teachers have missing or extra views/cells")

    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    observations: list[GaussianObservationField] = []
    cameras: list[Camera] = []
    for view_id in TRAIN_VIEWS:
        record_relative = f"records/{view_id}__{arm_name}.json"
        record_path = _strict_relative_file(root, record_relative, label=f"{view_id} cell")
        record = _load_json(record_path, label=f"{view_id} cell")
        if record.get("artifact_type") != SOURCE_CELL_TYPE or record.get("status") != "PASS":
            raise ValueError(f"{view_id} source cell is not PASS")
        if record.get("plan_sha256") != plan_sha256:
            raise ValueError(f"{view_id} source cell has a different plan")
        if record.get("view_id") != view_id or record.get("arm") != arm:
            raise ValueError(f"{view_id} source cell identity differs from the plan")
        if record.get("input") != view_inputs[view_id]:
            raise ValueError(f"{view_id} source cell input differs from the plan")
        if record.get("effective_config") != configs[view_id].get(arm_name):
            raise ValueError(f"{view_id} source cell config differs from the plan")
        if record.get("environment") != plan.get("environment_at_plan"):
            raise ValueError(f"{view_id} source cell environment differs from the plan")
        if record.get("loaded_cuda_extension") != plan.get("loaded_cuda_extension"):
            raise ValueError(f"{view_id} source cell extension differs from the plan")
        if record.get("external_structsplat") != external:
            raise ValueError(f"{view_id} source cell StructSplat binding differs from the plan")

        teacher_relative = f"teachers/{view_id}__{arm_name}.teacher.npz"
        if record.get("teacher_path") != teacher_relative:
            raise ValueError(f"{view_id} teacher path differs from the canonical cell path")
        teacher_path = _strict_relative_file(root, teacher_relative, label=f"{view_id} teacher")
        teacher_sha256 = _require_sha(record.get("teacher_sha256"), label=f"{view_id} teacher hash")
        if sha256_file(teacher_path) != teacher_sha256:
            raise ValueError(f"{view_id} teacher digest mismatch")
        _inspect_teacher(teacher_path)
        field = GaussianObservationField.load_npz(teacher_path, device="cpu", strict=True)
        if field.view_id != view_id:
            raise ValueError(f"{view_id} teacher view binding changed")
        summary = observation_summary(field)
        if record.get("teacher") != summary:
            raise ValueError(f"{view_id} teacher semantics/residual arrays changed")
        optimization = _required_mapping(
            record.get("optimization"), label=f"{view_id} optimization"
        )
        if optimization.get("m_init_2d") != field.n_init or optimization.get("m_opt_2d") != field.n:
            raise ValueError(f"{view_id} teacher cardinality differs from its cell")
        input_record = _required_mapping(view_inputs[view_id], label=f"{view_id} input")
        if input_record.get("view_id") != view_id:
            raise ValueError(f"{view_id} input identity changed")
        camera = _camera_from_record(input_record.get("camera"), label=f"{view_id} input")
        if field.width != camera.width or field.height != camera.height:
            raise ValueError(f"{view_id} teacher canvas differs from calibration")
        if list(field.fit_window) != input_record.get("fit_window"):
            raise ValueError(f"{view_id} teacher fit window differs from the Stage-1 input")

        panel = _required_mapping(record.get("panel"), label=f"{view_id} panel")
        panel_path = _strict_relative_file(root, panel.get("path"), label=f"{view_id} panel")
        if sha256_file(panel_path) != _require_sha(
            panel.get("sha256"), label=f"{view_id} panel hash"
        ):
            raise ValueError(f"{view_id} panel digest mismatch")
        records.append(record)
        record_hashes[record_relative] = sha256_file(record_path)
        observations.append(field)
        cameras.append(camera)

    tree_after = _tree_hashes(root)
    if tree_after != tree_before:
        raise RuntimeError("source sweep changed during strict verification")
    return VerifiedSweep(
        root=root,
        plan=plan,
        result=result,
        plan_sha256=plan_sha256,
        result_sha256=result_sha256,
        arm=arm,
        records=records,
        record_hashes=record_hashes,
        observations=observations,
        cameras=cameras,
        tree_hashes=tree_after,
    )


def _bound_data_file(relative_text: str, *, label: str) -> Path:
    return _strict_relative_file(
        _ordinary_directory(DATA_ROOT, label="data root"),
        relative_text,
        label=label,
    )


def _mask_fit_window(mask: torch.Tensor) -> list[int]:
    foreground = torch.nonzero(mask)
    if foreground.shape[0] < 1:
        raise ValueError("bound training mask is empty")
    height, width = mask.shape
    margin = max(2, round(max(height, width) * 0.05))
    y0 = max(0, int(foreground[:, 0].min()) - margin)
    y1 = min(height, int(foreground[:, 0].max()) + 1 + margin)
    x0 = max(0, int(foreground[:, 1].min()) - margin)
    x1 = min(width, int(foreground[:, 1].max()) + 1 + margin)
    return [x0, y0, x1 - x0, y1 - y0]


def acquire_bound_masks(sweep: VerifiedSweep) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Decode only plan-bound masks; an RGB alias fails before any decoder is called."""
    masks: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    for view_id, camera in zip(TRAIN_VIEWS, sweep.cameras, strict=True):
        record = _required_mapping(sweep.plan["inputs"]["views"][view_id], label=f"{view_id} input")
        mask_record = _required_mapping(record.get("mask"), label=f"{view_id} mask binding")
        rgb_record = _required_mapping(record.get("rgb"), label=f"{view_id} RGB binding")
        mask_relative = mask_record.get("path")
        if not isinstance(mask_relative, str):
            raise ValueError(f"{view_id} mask path must be a string")
        if mask_relative == rgb_record.get("path") or Path(mask_relative).suffix.lower() != ".png":
            raise ValueError(f"{view_id} mask binding aliases or resembles source RGB")
        mask_path = _bound_data_file(mask_relative, label=f"{view_id} bound mask")
        if sha256_file(mask_path) != _require_sha(
            mask_record.get("sha256"), label=f"{view_id} raw mask hash"
        ):
            raise ValueError(f"{view_id} raw mask changed after Stage-1 planning")
        distortion = record.get("distortion")
        if not isinstance(distortion, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in distortion
        ):
            raise ValueError(f"{view_id} distortion binding is invalid")
        native_resolution = record.get("native_resolution")
        if native_resolution != [camera.width, camera.height]:
            raise ValueError(f"{view_id} native resolution differs from its camera")
        mask = (
            _undistort(
                _resize_image(mask_path, camera.width, camera.height, mask=True),
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                [float(value) for value in distortion],
                mask=True,
            )
            > 0.5
        )
        expected_tensor_hash = _require_sha(
            record.get("undistorted_mask_tensor_sha256"),
            label=f"{view_id} undistorted mask hash",
        )
        if tensor_hash(mask) != expected_tensor_hash:
            raise ValueError(f"{view_id} undistorted mask differs from the Stage-1 plan")
        if int(mask.sum()) != record.get("foreground_pixels_full"):
            raise ValueError(f"{view_id} foreground count differs from the Stage-1 plan")
        if _mask_fit_window(mask) != record.get("fit_window"):
            raise ValueError(f"{view_id} mask-derived fit window differs from the Stage-1 plan")
        fit_x, fit_y, fit_width, fit_height = record["fit_window"]
        if int(mask[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width].sum()) != record.get(
            "foreground_pixels_crop"
        ):
            raise ValueError(f"{view_id} crop foreground count differs from the Stage-1 plan")
        masks.append(mask)
        receipts.append(
            {
                "view_id": view_id,
                "path": mask_relative,
                "raw_sha256": mask_record["sha256"],
                "undistorted_tensor_sha256": expected_tensor_hash,
                "foreground_pixels": int(mask.sum()),
                "role": "stage1_bounds_acquisition_only",
            }
        )
    return masks, receipts


def _load_external_bounds(
    path: Path,
    *,
    sweep: VerifiedSweep,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    receipt = _load_json(path, label="external bounds receipt")
    expected_keys = {
        "artifact_type",
        "source_plan_sha256",
        "training_views",
        "camera_digest",
        "mask_binding_digest",
        "method",
        "center",
        "extent",
    }
    if set(receipt) != expected_keys or receipt.get("artifact_type") != EXTERNAL_BOUNDS_TYPE:
        raise ValueError("external bounds receipt schema is not exact")
    expected = {
        "source_plan_sha256": sweep.plan_sha256,
        "training_views": list(TRAIN_VIEWS),
        "camera_digest": sweep.camera_digest,
        "mask_binding_digest": sweep.mask_binding_digest,
        "method": "rtgs.data.calibrated._object_bounds",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("external bounds receipt differs from the source sweep bindings")
    center = torch.tensor(receipt.get("center"), dtype=torch.float32)
    extent = receipt.get("extent")
    if (
        center.shape != (3,)
        or not bool(torch.isfinite(center).all())
        or isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not math.isfinite(float(extent))
        or float(extent) <= 0
    ):
        raise ValueError("external bounds center/extent is invalid")
    return (
        center,
        float(extent),
        {
            "mode": "explicit_bound_receipt",
            "receipt_path": str(path.resolve(strict=True)),
            "receipt_sha256": sha256_file(path),
        },
    )


def _write_bounds_npz(path: Path, center: torch.Tensor, extent: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                bounds_center=center.detach().cpu().numpy().astype(np.float32, copy=False),
                bounds_extent=np.asarray([extent], dtype=np.float64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_bounds_npz(path: Path) -> tuple[torch.Tensor, float]:
    _inspect_bounds_npz(path)
    with np.load(path, allow_pickle=False) as archive:
        center_array = np.asarray(archive["bounds_center"])
        extent_array = np.asarray(archive["bounds_extent"])
    center = torch.from_numpy(center_array.copy())
    extent = float(extent_array[0])
    if (
        center.dtype != torch.float32
        or center.shape != (3,)
        or extent_array.dtype != np.float64
        or extent_array.shape != (1,)
        or not bool(torch.isfinite(center).all())
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("bounds artifact arrays have invalid dtype, shape, or values")
    return center, extent


def _inspect_bounds_npz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("invalid bounds NPZ") from error
    names = [member.filename for member in members]
    if set(names) != {"bounds_center.npy", "bounds_extent.npy"} or len(names) != 2:
        raise ValueError("bounds NPZ members are not exact")
    if any(
        member.is_dir()
        or "/" in member.filename
        or "\\" in member.filename
        or member.file_size > 1_048_576
        for member in members
    ):
        raise ValueError("bounds NPZ contains an unsafe member")


def _bundle_name(arm_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", arm_name).strip(".-")
    normalized = normalized[:80] or "arm"
    value = f"stage1-capacity-{normalized}-{hashlib.sha256(arm_name.encode()).hexdigest()[:12]}"
    if _IDENTIFIER.fullmatch(value) is None:
        raise AssertionError("generated bundle name is not a strict identifier")
    return value


def prepare_plan(
    source: str | Path,
    out: str | Path,
    *,
    bounds_receipt: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze a verified source sweep and mask-bound bounds artifact."""
    source_path = Path(source).expanduser()
    out_path = Path(out).expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite bridge directory: {out_path}")
    source_root = _ordinary_directory(source_path, label="source sweep")
    try:
        out_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("bridge output must not be nested inside the source sweep")
    out_path.mkdir(parents=True)
    (out_path / "failures").mkdir()
    (out_path / "artifacts").mkdir()
    try:
        sweep = verify_source_sweep(source_root)
        source_tree_before_bounds = dict(sweep.tree_hashes)
        if bounds_receipt is None:
            masks, mask_receipts = acquire_bound_masks(sweep)
            center, extent = _object_bounds(sweep.cameras, masks)
            bounds_source = {
                "mode": "derived_from_plan_bound_training_masks",
                "method": "rtgs.data.calibrated._object_bounds",
                "masks": mask_receipts,
            }
        else:
            external_receipt = _ordinary_file(
                Path(bounds_receipt).expanduser(),
                label="external bounds receipt",
            )
            center, extent, bounds_source = _load_external_bounds(
                external_receipt,
                sweep=sweep,
            )
        if _tree_hashes(source_root) != source_tree_before_bounds:
            raise RuntimeError("source sweep changed during bounds acquisition")
        bounds_path = out_path / "artifacts" / "bounds.npz"
        _write_bounds_npz(bounds_path, center, extent)
        recovered_center, recovered_extent = _load_bounds_npz(bounds_path)
        if (
            not torch.equal(recovered_center, center.to(torch.float32))
            or recovered_extent != extent
        ):
            raise RuntimeError("bounds artifact roundtrip changed its values")
        bounds_record = {
            "artifact_type": BOUNDS_RECEIPT_TYPE,
            "source_plan_sha256": sweep.plan_sha256,
            "training_views": list(TRAIN_VIEWS),
            "heldout_view_excluded": HELDOUT_VIEW,
            "camera_digest": sweep.camera_digest,
            "mask_binding_digest": sweep.mask_binding_digest,
            "source": bounds_source,
            "bounds": {
                "path": "artifacts/bounds.npz",
                "sha256": sha256_file(bounds_path),
                "center": recovered_center.tolist(),
                "center_tensor_sha256": tensor_hash(recovered_center),
                "extent": recovered_extent,
            },
        }
        bounds_receipt_path = out_path / "artifacts" / "bounds_receipt.json"
        _write_json_exclusive(bounds_receipt_path, bounds_record)
        source_binding = {
            "root": str(source_root),
            "plan": {"path": "plan.json", "sha256": sweep.plan_sha256},
            "result": {"path": "result.json", "sha256": sweep.result_sha256},
            "tree_files": sweep.tree_hashes,
            "tree_aggregate_sha256": canonical_hash(sweep.tree_hashes),
            "record_sha256": sweep.record_hashes,
            "arm": sweep.arm,
            "training_views": list(TRAIN_VIEWS),
            "n_init_2d": [field.n_init for field in sweep.observations],
            "n_opt_2d": [field.n for field in sweep.observations],
            "teacher_sha256": {
                view_id: record["teacher_sha256"]
                for view_id, record in zip(TRAIN_VIEWS, sweep.records, strict=True)
            },
            "camera_digest": sweep.camera_digest,
            "mask_binding_digest": sweep.mask_binding_digest,
            "source_repository_binding_sha256": canonical_hash(sweep.plan["repository"]),
            "source_structsplat_binding_sha256": canonical_hash(sweep.plan["external_structsplat"]),
            "source_environment_binding_sha256": canonical_hash(sweep.plan["environment_at_plan"]),
            "source_cuda_extension_binding_sha256": canonical_hash(
                sweep.plan["loaded_cuda_extension"]
            ),
        }
        plan = {
            "artifact_type": PLAN_TYPE,
            "status": "FROZEN",
            "decision_bearing": False,
            "scope": (
                "CPU-only lossless Stage-1 teacher promotion with plan-bound mask-derived "
                "bounds; no teacher fitting, RGB, points, lift, refinement, or evaluation"
            ),
            "training_views": list(TRAIN_VIEWS),
            "heldout_view_excluded": HELDOUT_VIEW,
            "source_rgb_boundary": (
                "Source RGB is forbidden in every bridge phase. Planning may decode only the "
                "seven source-plan-bound masks when no explicit bound receipt is supplied; "
                "building decodes neither masks nor RGB."
            ),
            "source_sweep": source_binding,
            "bounds_receipt": {
                "path": "artifacts/bounds_receipt.json",
                "sha256": sha256_file(bounds_receipt_path),
                "bounds_path": "artifacts/bounds.npz",
                "bounds_sha256": sha256_file(bounds_path),
            },
            "bundle": {
                "path": "reconstruction_inputs",
                "name": _bundle_name(sweep.arm_name),
                "points": None,
                "point_visibility": None,
                "preserve_variable_m_opt_i_2d": True,
            },
            "repository": repository_source_binding(),
            "environment": environment_binding(),
            "argv_at_plan": list(sys.argv if argv is None else argv),
        }
        plan["semantic_digest"] = canonical_hash(plan)
        _write_json_exclusive(out_path / "plan.json", plan)
        return plan
    except BaseException as error:
        _append_failure(out_path, phase="plan", error=error, argv=argv)
        raise


def _load_bridge_plan(out: Path) -> tuple[dict[str, Any], str]:
    plan_path = _strict_relative_file(out, "plan.json", label="bridge plan")
    plan = _load_json(plan_path, label="bridge plan")
    digest_payload = dict(plan)
    semantic_digest = digest_payload.pop("semantic_digest", None)
    if semantic_digest != canonical_hash(digest_payload):
        raise ValueError("bridge plan semantic digest mismatch")
    if plan.get("artifact_type") != PLAN_TYPE or plan.get("status") != "FROZEN":
        raise ValueError("bridge plan is not a frozen supported artifact")
    if plan.get("training_views") != list(TRAIN_VIEWS):
        raise ValueError("bridge plan training views changed")
    if plan.get("heldout_view_excluded") != HELDOUT_VIEW:
        raise ValueError("bridge plan held-out policy changed")
    if plan.get("repository") != repository_source_binding():
        raise RuntimeError("bridge implementation sources drifted after planning")
    if plan.get("environment") != environment_binding():
        raise RuntimeError("bridge CPU environment drifted after planning")
    return plan, sha256_file(plan_path)


def _load_local_bounds(
    out: Path,
    plan: dict[str, Any],
    *,
    sweep: VerifiedSweep,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    binding = _required_mapping(plan.get("bounds_receipt"), label="bounds receipt binding")
    receipt_path = _strict_relative_file(out, binding.get("path"), label="bounds receipt")
    if sha256_file(receipt_path) != _require_sha(
        binding.get("sha256"), label="bounds receipt hash"
    ):
        raise ValueError("bounds receipt digest mismatch")
    receipt = _load_json(receipt_path, label="bounds receipt")
    if (
        receipt.get("artifact_type") != BOUNDS_RECEIPT_TYPE
        or receipt.get("source_plan_sha256") != sweep.plan_sha256
        or receipt.get("training_views") != list(TRAIN_VIEWS)
        or receipt.get("heldout_view_excluded") != HELDOUT_VIEW
        or receipt.get("camera_digest") != sweep.camera_digest
        or receipt.get("mask_binding_digest") != sweep.mask_binding_digest
    ):
        raise ValueError("bounds receipt differs from source bindings")
    bounds_record = _required_mapping(receipt.get("bounds"), label="bounds record")
    if bounds_record.get("path") != binding.get("bounds_path"):
        raise ValueError("bounds receipt path differs from the bridge plan")
    bounds_path = _strict_relative_file(out, binding.get("bounds_path"), label="bounds artifact")
    expected_sha = _require_sha(binding.get("bounds_sha256"), label="bounds artifact hash")
    if sha256_file(bounds_path) != expected_sha or bounds_record.get("sha256") != expected_sha:
        raise ValueError("bounds artifact digest mismatch")
    center, extent = _load_bounds_npz(bounds_path)
    if (
        bounds_record.get("center") != center.tolist()
        or bounds_record.get("center_tensor_sha256") != tensor_hash(center)
        or bounds_record.get("extent") != extent
    ):
        raise ValueError("bounds artifact semantics differ from its receipt")
    return center, extent, receipt


def _bundle_hashes(bundle: Path) -> dict[str, str]:
    return {
        path.relative_to(bundle).as_posix(): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    }


def _assert_bundle_semantics(
    loaded: ReconstructionInputs,
    *,
    sweep: VerifiedSweep,
    center: torch.Tensor,
    extent: float,
) -> None:
    if loaded.view_names != list(TRAIN_VIEWS):
        raise RuntimeError("strict bundle reload changed view order")
    if loaded.n_init_2d != [field.n_init for field in sweep.observations]:
        raise RuntimeError("strict bundle reload changed initialization counts")
    if loaded.n_opt_2d != [field.n for field in sweep.observations]:
        raise RuntimeError("strict bundle reload changed optimized per-view counts")
    if loaded.points is not None or loaded.point_visibility is not None:
        raise RuntimeError("bridge invented sparse geometry")
    if loaded.bounds_hint is None:
        raise RuntimeError("strict bundle reload lost mask-derived bounds")
    loaded_center, loaded_extent = loaded.bounds_hint
    if not torch.equal(loaded_center.cpu(), center.cpu()) or loaded_extent != extent:
        raise RuntimeError("strict bundle reload changed mask-derived bounds")
    if [observation_summary(field) for field in loaded.observations] != [
        observation_summary(field) for field in sweep.observations
    ]:
        raise RuntimeError("strict bundle reload changed teacher/residual semantics")
    if [_camera_record(camera) for camera in loaded.cameras] != [
        _camera_record(camera) for camera in sweep.cameras
    ]:
        raise RuntimeError("strict bundle reload changed calibrated cameras")


def build_bundle(
    out: str | Path,
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Re-verify the frozen source and atomically write the RGB-free bundle."""
    out_path = _ordinary_directory(Path(out).expanduser(), label="bridge directory")
    try:
        if (out_path / "result.json").exists() or (out_path / "reconstruction_inputs").exists():
            raise FileExistsError("refusing to overwrite bridge result or reconstruction bundle")
        plan, plan_sha256 = _load_bridge_plan(out_path)
        source_binding = _required_mapping(plan.get("source_sweep"), label="source sweep binding")
        source_root = _ordinary_directory(
            Path(source_binding.get("root")), label="bound source sweep"
        )
        if _tree_hashes(source_root) != source_binding.get("tree_files"):
            raise RuntimeError("bound source sweep tree changed after bridge planning")
        sweep = verify_source_sweep(source_root)
        checks = {
            "plan": sweep.plan_sha256,
            "result": sweep.result_sha256,
            "tree": canonical_hash(sweep.tree_hashes),
            "record": sweep.record_hashes,
            "arm": sweep.arm,
            "n_init": [field.n_init for field in sweep.observations],
            "n_opt": [field.n for field in sweep.observations],
            "camera": sweep.camera_digest,
            "mask": sweep.mask_binding_digest,
        }
        expected = {
            "plan": source_binding["plan"]["sha256"],
            "result": source_binding["result"]["sha256"],
            "tree": source_binding["tree_aggregate_sha256"],
            "record": source_binding["record_sha256"],
            "arm": source_binding["arm"],
            "n_init": source_binding["n_init_2d"],
            "n_opt": source_binding["n_opt_2d"],
            "camera": source_binding["camera_digest"],
            "mask": source_binding["mask_binding_digest"],
        }
        if checks != expected:
            raise RuntimeError("strict source sweep semantics differ from the bridge plan")
        source_binding_checks = {
            "source_repository_binding_sha256": canonical_hash(sweep.plan["repository"]),
            "source_structsplat_binding_sha256": canonical_hash(sweep.plan["external_structsplat"]),
            "source_environment_binding_sha256": canonical_hash(sweep.plan["environment_at_plan"]),
            "source_cuda_extension_binding_sha256": canonical_hash(
                sweep.plan["loaded_cuda_extension"]
            ),
        }
        if any(source_binding.get(key) != value for key, value in source_binding_checks.items()):
            raise RuntimeError("source plan bindings differ from the bridge plan")
        center, extent, bounds_receipt = _load_local_bounds(out_path, plan, sweep=sweep)
        bundle_config = _required_mapping(plan.get("bundle"), label="bundle config")
        candidate = Path(tempfile.mkdtemp(prefix=".bundle-candidate.", dir=out_path))
        shutil.rmtree(candidate)
        final_bundle = out_path / bundle_config["path"]
        try:
            reconstruction = ReconstructionInputs(
                observations=sweep.observations,
                cameras=sweep.cameras,
                view_names=list(TRAIN_VIEWS),
                points=None,
                point_visibility=None,
                bounds_hint=(center, extent),
                name=bundle_config["name"],
            )
            reconstruction.save(candidate)
            candidate_loaded = ReconstructionInputs.load(candidate, device="cpu", strict=True)
            _assert_bundle_semantics(
                candidate_loaded,
                sweep=sweep,
                center=center,
                extent=extent,
            )
            manifest_text = (candidate / "manifest.json").read_text(encoding="utf-8")
            forbidden = ("rgb", "mask", "image_path", "source_path")
            if any(token in manifest_text.lower() for token in forbidden):
                raise RuntimeError("bundle manifest contains a forbidden raster/source field")
            os.replace(candidate, final_bundle)
            candidate = None
        finally:
            if candidate is not None:
                shutil.rmtree(candidate, ignore_errors=True)
        loaded = ReconstructionInputs.load(final_bundle, device="cpu", strict=True)
        _assert_bundle_semantics(
            loaded,
            sweep=sweep,
            center=center,
            extent=extent,
        )
        manifest_path = final_bundle / "manifest.json"
        manifest = _load_json(manifest_path, label="reconstruction manifest")
        files = _bundle_hashes(final_bundle)
        if _tree_hashes(source_root) != sweep.tree_hashes:
            raise RuntimeError("source sweep changed while building the bundle")
        result = {
            "artifact_type": RESULT_TYPE,
            "status": "PASS",
            "decision_bearing": False,
            "scope": plan["scope"],
            "plan_sha256": plan_sha256,
            "training_views": list(TRAIN_VIEWS),
            "heldout_view_excluded": HELDOUT_VIEW,
            "selected_arm": sweep.arm,
            "n_views": loaded.n_views,
            "n_init_2d": loaded.n_init_2d,
            "n_opt_2d": loaded.n_opt_2d,
            "total_m_opt_2d": sum(loaded.n_opt_2d),
            "bounds": {
                "center": center.tolist(),
                "center_tensor_sha256": tensor_hash(center),
                "extent": extent,
                "receipt_sha256": plan["bounds_receipt"]["sha256"],
                "derivation_mode": bounds_receipt["source"]["mode"],
            },
            "source_sweep": {
                "root": str(source_root),
                "plan_sha256": sweep.plan_sha256,
                "result_sha256": sweep.result_sha256,
                "tree_aggregate_sha256": canonical_hash(sweep.tree_hashes),
                "record_sha256": sweep.record_hashes,
                "teacher_sha256": source_binding["teacher_sha256"],
                **source_binding_checks,
            },
            "bundle": {
                "path": bundle_config["path"],
                "manifest_sha256": sha256_file(manifest_path),
                "semantic_digest": manifest["semantic_digest"],
                "calibration_digest": manifest["calibration_digest"],
                "files": files,
                "aggregate_sha256": canonical_hash(files),
                "archive_stats": dataclasses.asdict(loaded.archive_stats),
                "contains_points": False,
                "contains_bounds_hint": True,
                "contains_dense_rgb_mask_or_source_path": False,
            },
            "repository": plan["repository"],
            "environment": plan["environment"],
            "argv_at_build": list(sys.argv if argv is None else argv),
            "viewer": {
                "status": "NOT_APPLICABLE_STAGE1_BUNDLE_ONLY",
                "reason": (
                    "rtgs view requires a lifted 3D PLY; this bridge stops at RGB-free inputs"
                ),
            },
        }
        _write_json_exclusive(out_path / "result.json", result)
        return result
    except BaseException as error:
        _append_failure(out_path, phase="build", error=error, argv=argv)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan = subparsers.add_parser("plan", help="verify source and freeze mask-derived bounds")
    plan.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    plan.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plan.add_argument(
        "--bounds-receipt",
        type=Path,
        help="optional explicit source-plan-bound bounds JSON; otherwise masks are acquired",
    )
    build = subparsers.add_parser("build", help="strictly assemble the planned RGB-free bundle")
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_argv = list(sys.argv if argv is None else argv)
    if args.action == "plan":
        result = prepare_plan(
            args.source,
            args.out,
            bounds_receipt=args.bounds_receipt,
            argv=effective_argv,
        )
    elif args.action == "build":
        result = build_bundle(args.out, argv=effective_argv)
    else:  # pragma: no cover
        raise AssertionError(args.action)
    print(json.dumps(_json_safe(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
