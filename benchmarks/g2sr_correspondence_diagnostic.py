"""Frozen G²SR-style Gaussian correspondence diagnostic on the calibrated fabric capture.

This harness deliberately separates four trust domains:

``plan``
    Writes the preregistered pair/crop/gate protocol without opening an image or importing
    torchvision.
``acquire``
    The only scientific stage allowed to decode source RGB. It runs a lazy, pluggable dense-flow
    backend and serializes flow, masks, calibration, the frozen compact reference teacher, and
    provenance. No source RGB tensor is written.
``analyze``
    Decodes and scientifically consumes only immutable post-flow artifacts and source masks. It
    never opens a JPEG, including for metadata or re-hashing; source-color provenance terminates at
    the immutable acquisition digest. It tracks all five Gaussian sigma points, applies geometric
    gates, triangulates centers, and writes a mechanism-only PLY.
``view-smoke``
    An explicitly reporting-only RGB exception. It launches the saved PLY in the full-resolution
    gsplat viewer using an exact training-view allowlist and records an end-to-end receipt.

The diagnostic is exploratory. C0014 is the frozen reference teacher, C0005 and C0026 are the
only flow targets, and held-out C1004 is rejected before any acquisition.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import bilinear_sample
from rtgs.lift.gaussian_correspondence import (
    track_gaussians_2d,
    triangulate_centers_dlt,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
COMPACT_VIEWS = SCENE / "gaussians2d"
TEACHER_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
DEFAULT_OUT = ROOT / "runs/g2sr_correspondence_diagnostic_20260717"

REFERENCE_VIEW = "C0014"
TARGET_VIEWS = ("C0005", "C0026")
HELDOUT_VIEW = "C1004"
BOUNDS_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
VIEWER_VIEWS = BOUNDS_VIEWS

# Full-canvas (x, y, width, height). Dimensions are exactly divisible by FLOW_DOWNSCALE.
FLOW_CROP = (1488, 1776, 3840, 1680)
FLOW_DOWNSCALE = 3
FLOW_WIDTH = FLOW_CROP[2] // FLOW_DOWNSCALE
FLOW_HEIGHT = FLOW_CROP[3] // FLOW_DOWNSCALE
FLOW_SHAPE = (FLOW_HEIGHT, FLOW_WIDTH)
NATIVE_WIDTH = 5328
NATIVE_HEIGHT = 4608
REFERENCE_COMPONENT_COUNT = 640

FLOW_BACKEND = "torchvision_raft_small_c_t_v2"
ACQUISITION_DEVICE = "cuda"
EXPLORATORY_QUESTION = (
    "Can five-sigma-point optical-flow correspondences from the frozen C0014 compact "
    "teacher yield geometrically valid two-view 3D center samples?"
)
EXPLORATORY_SCOPE = "exploratory mechanism diagnostic; not a reconstruction-quality claim"
EXPLORATORY_DECISION_POLICY = (
    "Report every preregistered gate distribution and count. This exploratory run "
    "has no pass/fail threshold and cannot establish mechanism success."
)
EXPLORATORY_CLAIM_SCOPE = (
    "descriptive mechanism-only correspondence/triangulation diagnostic; no "
    "reconstruction-quality, reproduction, or causal claim"
)
RGB_FREE_SOURCE_COLOR_POLICY = (
    "acquisition-bound digest replay only; analysis never opens, decodes, or re-hashes JPEG"
)
PRIMARY_FB_PX = 3.0
FB_SENSITIVITY_PX = (1.0, 3.0, 5.0)
BOUNDS_SCALE = 0.5
LOCAL_LIBSTDCXX_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
PLAN_ARTIFACT_TYPE = "g2sr_correspondence_diagnostic_plan_v4"
ACQUISITION_ARTIFACT_TYPE = "g2sr_flow_acquisition_v4"
RESULT_ARTIFACT_TYPE = "g2sr_correspondence_diagnostic_result_v4"
VIEWER_ARTIFACT_TYPE = "g2sr_viewer_smoke_v5"
NONFINITE_CODE_SUFFIX = "__nonfinite_code"
NONFINITE_CODE_SEMANTICS = {
    0: "finite",
    1: "nan",
    2: "positive_infinity",
    3: "negative_infinity",
}

# Bind the entire local Python implementation rather than attempting to maintain an incomplete
# hand-written transitive-import list.  This intentionally includes such easy-to-miss scientific
# dependencies as ``lift/base.py`` and ``core/sh.py`` as well as the viewer/render handoff.
REPLAY_CODE_PATHS = (
    Path("benchmarks/g2sr_correspondence_diagnostic.py"),
    *tuple(
        sorted(
            path.relative_to(ROOT)
            for path in (ROOT / "src" / "rtgs").rglob("*.py")
            if path.is_file()
        )
    ),
)


@dataclass(frozen=True)
class PairSpec:
    """One frozen reference-to-target flow and triangulation pair."""

    reference: str
    target: str

    @property
    def pair_id(self) -> str:
        return f"{self.reference}_to_{self.target}"


@dataclass(frozen=True)
class GateConfig:
    """Exploratory correspondence gates, all expressed in the 1280x560 flow frame unless noted."""

    fb_error_px: float = PRIMARY_FB_PX
    affine_source_design_max_condition: float = 1.0e4
    affine_recovered_max_condition: float = 1.0e4
    affine_max_rms_px: float = 1.5
    affine_min_determinant: float = 0.05
    affine_max_determinant: float = 20.0
    covariance_min_eigenvalue_px2: float = 1.0e-6
    target_mask_threshold: float = 0.5
    epipolar_max_error_native_px: float = 2.0
    dlt_max_condition: float = 1.0e8
    dlt_min_nullspace_gap: float = 2.0
    dlt_max_reprojection_native_px: float = 3.0
    min_ray_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and non-negative")
        if self.affine_min_determinant > self.affine_max_determinant:
            raise ValueError("affine determinant minimum cannot exceed maximum")


PAIRS = tuple(PairSpec(REFERENCE_VIEW, target) for target in TARGET_VIEWS)


class DenseFlowBackend(Protocol):
    """Pluggable acquisition-only dense-flow backend."""

    name: str

    def infer(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Infer one ``(H,W,2)`` FP32 displacement field from CHW float images."""

    def metadata(self) -> dict[str, Any]:
        """Return model/weights/runtime provenance after loading."""

    def close(self) -> None:
        """Release accelerator state."""


def _frozen_decision_rule() -> dict[str, Any]:
    return {
        "mode": "descriptive_only",
        "binary_mechanism_success": False,
        "policy": EXPLORATORY_DECISION_POLICY,
    }


def _validate_exploratory_contract(
    value: Mapping[str, Any],
    *,
    result: bool,
) -> None:
    if value.get("decision_rule") != _frozen_decision_rule():
        raise RuntimeError("exploratory decision policy changed")
    if result:
        if value.get("claim_scope") != EXPLORATORY_CLAIM_SCOPE:
            raise RuntimeError("exploratory result claim scope changed")
    elif value.get("question") != EXPLORATORY_QUESTION or value.get("scope") != EXPLORATORY_SCOPE:
        raise RuntimeError("exploratory question/scope changed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    """Return a sortable, timezone-explicit UTC timestamp."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_utc_timestamp(value: object, *, context: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{context} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo != dt.UTC:
        raise RuntimeError(f"{context} must be UTC")


def _require_sha256(value: object, *, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{context} must be a lowercase SHA-256 hex digest")


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _verify_file_binding(record: Mapping[str, Any], *, context: str) -> Path:
    _require_exact_keys(record, {"path", "sha256", "bytes", "mtime_ns"}, context)
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{context} file is absent: {path}")
    current = _file_binding(path)
    if current != dict(record):
        raise RuntimeError(f"{context} binding changed: expected {dict(record)}, got {current}")
    return path


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    head = run("rev-parse", "HEAD").decode().strip()
    status = run("status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "dirty_state_digest": digest.hexdigest(),
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a JSON object")
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{context} key set mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _find_heldout_occurrences(value: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    occurrences: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if HELDOUT_VIEW in key_text.upper():
                occurrences.append((*path, f"<key:{key_text}>"))
            occurrences.extend(_find_heldout_occurrences(item, (*path, key_text)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            occurrences.extend(_find_heldout_occurrences(item, (*path, str(index))))
    elif isinstance(value, str) and HELDOUT_VIEW in value.upper():
        occurrences.append(path)
    return occurrences


def _assert_heldout_only_in_declaration(value: Mapping[str, Any], *, context: str) -> None:
    occurrences = _find_heldout_occurrences(value)
    allowed = {("heldout", "view")}
    forbidden = [path for path in occurrences if path not in allowed]
    if forbidden or occurrences.count(("heldout", "view")) != 1:
        raise RuntimeError(
            f"{context} must mention {HELDOUT_VIEW} exactly once at heldout.view; "
            f"occurrences={occurrences}"
        )


def _teacher_binding() -> dict[str, Any]:
    manifest_path = TEACHER_BUNDLE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest.get("views")
    if not isinstance(views, list):
        raise RuntimeError("teacher bundle manifest has no views list")
    matches = [
        record
        for record in views
        if isinstance(record, Mapping) and record.get("view_id") == REFERENCE_VIEW
    ]
    if len(matches) != 1:
        raise RuntimeError("teacher bundle must contain exactly one C0014 record")
    record = matches[0]
    _require_exact_keys(
        record,
        {"camera", "n_init_2d", "n_opt_2d", "teacher", "teacher_sha256", "view_id"},
        "reference teacher record",
    )
    teacher_path = TEACHER_BUNDLE / str(record["teacher"])
    teacher = _file_binding(teacher_path)
    if teacher["sha256"] != record["teacher_sha256"]:
        raise RuntimeError("reference teacher digest differs from the bundle manifest")
    if int(record["n_opt_2d"]) != REFERENCE_COMPONENT_COUNT:
        raise RuntimeError("reference teacher component count changed")
    return {
        "bundle_path": display_path(TEACHER_BUNDLE),
        "manifest": _file_binding(manifest_path),
        "reference_view": REFERENCE_VIEW,
        "reference_teacher": teacher,
        "manifest_declared_teacher_path": str(record["teacher"]),
        "manifest_declared_teacher_sha256": str(record["teacher_sha256"]),
        "component_count": int(record["n_opt_2d"]),
    }


def _scientific_input_bindings() -> dict[str, Any]:
    records, calibration_path = _calibration_records()
    required_views = tuple(dict.fromkeys((REFERENCE_VIEW, *TARGET_VIEWS, *BOUNDS_VIEWS)))
    missing = [view for view in required_views if view not in records]
    if missing:
        raise RuntimeError(f"calibration lacks preregistered views: {missing}")
    # The checked-in dataset is now RGB-free. Each compact view contains the fitted color
    # observation and, for this masked scene, the exact packed alpha that replaces the old mask.
    colors = {
        view: _file_binding(COMPACT_VIEWS / f"{view}.rtgsv")
        for view in (REFERENCE_VIEW, *TARGET_VIEWS)
    }
    masks = {view: _file_binding(COMPACT_VIEWS / f"{view}.rtgsv") for view in required_views}
    return {
        "calibration": _file_binding(calibration_path),
        "source_colors": colors,
        "source_masks": masks,
    }


def _canonical_code_source(label: str) -> Path:
    expected_labels = {path.as_posix() for path in REPLAY_CODE_PATHS}
    if label not in expected_labels or Path(label).as_posix() != label:
        raise RuntimeError(f"code-source label is not canonical: {label!r}")
    source = (ROOT / label).resolve()
    try:
        source.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"code-source label escapes the repository: {label!r}") from exc
    if not source.is_file() or source != ROOT.joinpath(*Path(label).parts).resolve():
        raise RuntimeError(f"canonical code source is absent or misbound: {label!r}")
    return source


def _output_from_display(value: object) -> Path:
    output = Path(str(value))
    if not output.is_absolute():
        output = ROOT / output
    return output.resolve()


def _canonical_replay_source(output: Path, label: str) -> Path:
    _canonical_code_source(label)
    replay_root = (output.resolve() / "replay_sources").resolve()
    replay = replay_root.joinpath(*Path(label).parts)
    try:
        replay.resolve().relative_to(replay_root)
    except ValueError as exc:
        raise RuntimeError(f"replay-source label escapes its run directory: {label!r}") from exc
    return replay


def _assert_rtgs_import_origins() -> None:
    """Reject shadowed installed modules/functions before binding scientific code."""
    source_root = (ROOT / "src" / "rtgs").resolve()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "rtgs" and not module_name.startswith("rtgs."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"imported rtgs module has no repository origin: {module_name}")
        origin = Path(module_file).resolve()
        try:
            origin.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(
                f"imported rtgs module is shadowed outside this repository: "
                f"{module_name} -> {origin}"
            ) from exc
    required_origins = {
        "src/rtgs/core/camera.py": (Camera,),
        "src/rtgs/core/gaussians3d.py": (Gaussians3D,),
        "src/rtgs/data/reconstruction_inputs.py": (ReconstructionInputs,),
        "src/rtgs/lift/base.py": (bilinear_sample,),
        "src/rtgs/lift/gaussian_correspondence.py": (
            track_gaussians_2d,
            triangulate_centers_dlt,
        ),
    }
    for label, objects in required_origins.items():
        expected = _canonical_code_source(label)
        for value in objects:
            source_text = inspect.getsourcefile(value)
            if source_text is None or Path(source_text).resolve() != expected:
                raise RuntimeError(
                    f"imported scientific object {value!r} is not defined by canonical {label}"
                )


def _code_source_bindings(output: Path) -> dict[str, Any]:
    _assert_rtgs_import_origins()
    return {
        path.as_posix(): _file_binding(_canonical_code_source(path.as_posix()))
        | {"replay_path": display_path(_canonical_replay_source(output, path.as_posix()))}
        for path in REPLAY_CODE_PATHS
    }


def _current_source_hashes(
    plan: Mapping[str, Any],
    *,
    verify_source_color_bytes: bool = True,
) -> dict[str, str]:
    _assert_rtgs_import_origins()
    result: dict[str, str] = {}
    teacher = plan["teacher_bundle"]
    _verify_file_binding(teacher["manifest"], context="teacher manifest")
    _verify_file_binding(teacher["reference_teacher"], context="reference teacher")
    bindings = plan["scientific_inputs"]
    result["calibration"] = sha256_file(
        _verify_file_binding(bindings["calibration"], context="calibration")
    )
    for group in ("source_colors", "source_masks"):
        for view, record in bindings[group].items():
            if group == "source_colors" and not verify_source_color_bytes:
                # RGB-free verification consumes the acquisition-bound digest only.  It never
                # opens a JPEG, even for metadata or re-hashing.
                _require_sha256(record["sha256"], context=f"{group}/{view}.sha256")
                result[f"{group}/{view}"] = str(record["sha256"])
            else:
                result[f"{group}/{view}"] = sha256_file(
                    _verify_file_binding(record, context=f"{group}/{view}")
                )
    for label, record in plan["code_sources"].items():
        canonical_source = _canonical_code_source(label)
        if record["path"] != display_path(canonical_source):
            raise RuntimeError(f"code-source label {label} is misbound to {record['path']}")
        path = _verify_file_binding(
            {key: record[key] for key in ("path", "sha256", "bytes", "mtime_ns")},
            context=f"code source {label}",
        )
        if path != canonical_source:
            raise RuntimeError(f"code-source label {label} resolved away from its canonical file")
        result[f"code/{label}"] = sha256_file(path)
        output = _output_from_display(plan["output_path"])
        replay = _canonical_replay_source(output, label)
        if record["replay_path"] != display_path(replay):
            raise RuntimeError(f"replay-source label {label} is not under the bound run output")
        if not replay.is_file() or sha256_file(replay) != record["sha256"]:
            raise RuntimeError(f"replay source differs or is absent for {label}")
    result["teacher_manifest"] = teacher["manifest"]["sha256"]
    result["reference_teacher"] = teacher["reference_teacher"]["sha256"]
    return dict(sorted(result.items()))


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_pair_protocol(
    reference: str = REFERENCE_VIEW,
    targets: Sequence[str] = TARGET_VIEWS,
    *,
    heldout: str = HELDOUT_VIEW,
) -> tuple[PairSpec, ...]:
    normalized_reference = reference.upper()
    normalized_targets = tuple(target.upper() for target in targets)
    if heldout.upper() == normalized_reference or heldout.upper() in normalized_targets:
        raise ValueError(f"held-out view {heldout.upper()} is forbidden in correspondence pairs")
    if normalized_reference != REFERENCE_VIEW or normalized_targets != TARGET_VIEWS:
        raise ValueError(
            "pair protocol is frozen to "
            f"{REFERENCE_VIEW} -> {TARGET_VIEWS}, got "
            f"{normalized_reference} -> {normalized_targets}"
        )
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("target views must be unique")
    if normalized_reference in normalized_targets:
        raise ValueError("reference view cannot also be a target")
    return tuple(PairSpec(normalized_reference, target) for target in normalized_targets)


def native_to_flow_points(
    points_native: torch.Tensor,
    *,
    crop: tuple[int, int, int, int] = FLOW_CROP,
    downscale: int = FLOW_DOWNSCALE,
) -> torch.Tensor:
    """Map native half-pixel coordinates into the exact area-resized crop frame."""
    if points_native.shape[-1:] != (2,) or not points_native.dtype.is_floating_point:
        raise ValueError("points_native must be a floating tensor ending in two coordinates")
    origin = points_native.new_tensor(crop[:2])
    return (points_native - origin) / float(downscale)


def flow_to_native_points(
    points_flow: torch.Tensor,
    *,
    crop: tuple[int, int, int, int] = FLOW_CROP,
    downscale: int = FLOW_DOWNSCALE,
) -> torch.Tensor:
    """Invert :func:`native_to_flow_points` exactly."""
    if points_flow.shape[-1:] != (2,) or not points_flow.dtype.is_floating_point:
        raise ValueError("points_flow must be a floating tensor ending in two coordinates")
    origin = points_flow.new_tensor(crop[:2])
    return points_flow * float(downscale) + origin


def native_to_flow_covariances(
    covariances_native: torch.Tensor,
    *,
    downscale: int = FLOW_DOWNSCALE,
) -> torch.Tensor:
    if covariances_native.shape[-2:] != (2, 2):
        raise ValueError("covariances_native must end in shape (2,2)")
    return covariances_native / float(downscale**2)


def flow_to_native_covariances(
    covariances_flow: torch.Tensor,
    *,
    downscale: int = FLOW_DOWNSCALE,
) -> torch.Tensor:
    if covariances_flow.shape[-2:] != (2, 2):
        raise ValueError("covariances_flow must end in shape (2,2)")
    return covariances_flow * float(downscale**2)


def flow_camera(
    camera: Camera,
    *,
    crop: tuple[int, int, int, int] = FLOW_CROP,
    downscale: int = FLOW_DOWNSCALE,
) -> Camera:
    x, y, width, height = crop
    if width % downscale or height % downscale:
        raise ValueError("crop dimensions must be exactly divisible by downscale")
    return Camera(
        fx=camera.fx / downscale,
        fy=camera.fy / downscale,
        cx=(camera.cx - x) / downscale,
        cy=(camera.cy - y) / downscale,
        width=width // downscale,
        height=height // downscale,
        R=camera.R,
        t=camera.t,
    )


def camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": int(camera.width),
        "height": int(camera.height),
        "R": camera.R.detach().cpu().reshape(-1).tolist(),
        "t": camera.t.detach().cpu().tolist(),
    }


def camera_from_record(record: Mapping[str, Any]) -> Camera:
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


def observation_covariances(field: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Recover full native covariance matrices from the frozen StructSplat RS parameters."""
    variances = field.effective_variances().to(dtype=dtype)
    angle = field.rotations.to(dtype=dtype)
    cosine, sine = angle.cos(), angle.sin()
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    return rotation @ torch.diag_embed(variances) @ rotation.transpose(-1, -2)


def observation_native_means(field: Any, *, dtype: torch.dtype) -> torch.Tensor:
    """Prefer losslessly reconstructed producer means when the schema provides them."""
    native_api = getattr(field, "native_means", None)
    if callable(native_api):
        return native_api(dtype=dtype)
    return field.means.to(dtype=dtype)


def assert_crop_coordinate_convention() -> None:
    """Fail closed if the frozen half-pixel crop/area-resize equations ever drift."""
    x, y, _, _ = FLOW_CROP
    native_means = torch.tensor(
        [[x + 1.5, y + 1.5], [x + 4.5, y + 7.5]],
        dtype=torch.float64,
    )
    expected_flow = torch.tensor([[0.5, 0.5], [1.5, 2.5]], dtype=torch.float64)
    native_covariance = torch.tensor(
        [[[9.0, 4.5], [4.5, 36.0]]],
        dtype=torch.float64,
    )
    flow_means = native_to_flow_points(native_means)
    flow_covariance = native_to_flow_covariances(native_covariance)
    if not torch.equal(flow_means, expected_flow):
        raise RuntimeError("native-to-flow half-pixel center convention changed")
    if not torch.equal(flow_to_native_points(flow_means), native_means):
        raise RuntimeError("flow-to-native mean transform is not exactly invertible")
    if not torch.equal(flow_to_native_covariances(flow_covariance), native_covariance):
        raise RuntimeError("flow-to-native covariance transform is not exactly invertible")


def plan_payload(
    gates: GateConfig | None = None,
    *,
    output: Path = DEFAULT_OUT,
    created_utc: str | None = None,
) -> dict[str, Any]:
    assert_crop_coordinate_convention()
    validate_pair_protocol()
    gates = GateConfig() if gates is None else gates
    output = output.expanduser().resolve()
    created_utc = utc_now() if created_utc is None else created_utc
    flow_bytes = len(PAIRS) * 2 * FLOW_HEIGHT * FLOW_WIDTH * 2 * 4
    image_workspace_bytes = 2 * 3 * FLOW_HEIGHT * FLOW_WIDTH * 4
    code_sources = _code_source_bindings(output)
    commands = {
        "plan": (
            ".venv/bin/python benchmarks/g2sr_correspondence_diagnostic.py "
            f"plan --out {display_path(output)}"
        ),
        "acquire_local_weights_only": (
            f"LD_PRELOAD={LOCAL_LIBSTDCXX_PRELOAD} .venv/bin/python "
            "benchmarks/g2sr_correspondence_diagnostic.py acquire "
            f"--out {display_path(output)} --device cuda"
        ),
        "acquire_allow_official_weight_download": (
            f"LD_PRELOAD={LOCAL_LIBSTDCXX_PRELOAD} .venv/bin/python "
            "benchmarks/g2sr_correspondence_diagnostic.py acquire "
            f"--out {display_path(output)} --device cuda --allow-download"
        ),
        "analyze": (
            ".venv/bin/python benchmarks/g2sr_correspondence_diagnostic.py "
            f"analyze --out {display_path(output)}"
        ),
        "viewer_smoke": (
            f"LD_PRELOAD={LOCAL_LIBSTDCXX_PRELOAD} .venv/bin/python "
            "benchmarks/g2sr_correspondence_diagnostic.py view-smoke "
            f"--out {display_path(output)}"
        ),
    }
    payload = {
        "artifact_type": PLAN_ARTIFACT_TYPE,
        "status": "PREREGISTERED_NOT_RUN",
        "created_utc": created_utc,
        "output_path": display_path(output),
        "question": EXPLORATORY_QUESTION,
        "scope": EXPLORATORY_SCOPE,
        "decision_rule": _frozen_decision_rule(),
        "reference": REFERENCE_VIEW,
        "targets": list(TARGET_VIEWS),
        "heldout": {
            "view": HELDOUT_VIEW,
            "policy": (
                "declared held-out; excluded from every source, pair, bound, gate, and output"
            ),
        },
        "pairs": [dataclasses.asdict(pair) | {"pair_id": pair.pair_id} for pair in PAIRS],
        "teacher_bundle": _teacher_binding(),
        "scientific_inputs": _scientific_input_bindings(),
        "code_sources": code_sources,
        "git": _git_record(),
        "replay_package": {
            "path": display_path(output / "replay_sources"),
            "collection_sha256": canonical_hash(
                {label: record["sha256"] for label, record in code_sources.items()}
            ),
            "policy": "byte-exact copies of every listed executed/preprocess/core/viewer source",
        },
        "crop": {
            "native_xywh": list(FLOW_CROP),
            "expected_input_size": [NATIVE_WIDTH, NATIVE_HEIGHT],
            "crop_size": [FLOW_CROP[2], FLOW_CROP[3]],
            "downscale": FLOW_DOWNSCALE,
            "flow_size": [FLOW_WIDTH, FLOW_HEIGHT],
            "resize": "torch.nn.functional.interpolate(mode='area')",
            "mean_transform": "mu_flow=(mu_native-[crop_x,crop_y])/3",
            "covariance_transform": "Sigma_flow=Sigma_native/9",
        },
        "flow": {
            "backend": FLOW_BACKEND,
            "device": ACQUISITION_DEVICE,
            "weights": "torchvision Raft_Small_Weights.C_T_V2",
            "precision": "sequential FP32 forward/backward",
            "network_policy": "local cache only unless --allow-download is explicit",
            "all_reference_components": True,
            "all_five_sigma_points_required": True,
        },
        "gates": dataclasses.asdict(gates),
        "fb_sensitivity_px": list(FB_SENSITIVITY_PX),
        "bounds": {
            "views": list(BOUNDS_VIEWS),
            "source": "same seven undistorted training masks as corrected CompactCarve arm",
            "aabb": "center +/- extent*0.5",
            "bounds_scale": BOUNDS_SCALE,
        },
        "artifacts": {
            "post_flow_policy": (
                "decoded source color is transient acquisition input only; no source-color "
                "tensor or source-color-derived thumbnail is serialized. RGB-encoded flow "
                "visualizations contain only flow vectors."
            ),
            "ply": (
                "separate pair-colored per-pair PLYs plus a combined pair-sample PLY; duplicate "
                "component lineages across pairs remain duplicate samples"
            ),
            "viewer": (
                "native calibrated gsplat/CUDA snapshot is mandatory before the HTTP smoke; "
                f"the reporting scene decodes only training allowlist {list(VIEWER_VIEWS)}"
            ),
        },
        "estimated_memory": {
            "flow_arrays_bytes": flow_bytes,
            "two_flow_input_tensors_bytes": image_workspace_bytes,
            "conservative_peak_gpu_bytes": 5_905_580_032,
            "conservative_peak_gpu_gib": 5.5,
            "note": (
                "planning bound, not a measurement; acquisition records torch peak allocation "
                "and reservation per direction"
            ),
        },
        "commands": commands,
    }
    _validate_plan_schema(payload)
    return payload


def _validate_plan_schema(plan: Mapping[str, Any]) -> None:
    _require_exact_keys(
        plan,
        {
            "artifact_type",
            "status",
            "created_utc",
            "output_path",
            "question",
            "scope",
            "decision_rule",
            "reference",
            "targets",
            "heldout",
            "pairs",
            "teacher_bundle",
            "scientific_inputs",
            "code_sources",
            "git",
            "replay_package",
            "crop",
            "flow",
            "gates",
            "fb_sensitivity_px",
            "bounds",
            "artifacts",
            "estimated_memory",
            "commands",
        },
        "plan",
    )
    if plan["artifact_type"] != PLAN_ARTIFACT_TYPE or plan["status"] != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("plan type/status differs from the frozen schema")
    _validate_exploratory_contract(plan, result=False)
    _require_utc_timestamp(plan["created_utc"], context="plan.created_utc")
    if plan["reference"] != REFERENCE_VIEW or tuple(plan["targets"]) != TARGET_VIEWS:
        raise RuntimeError("plan view protocol changed")
    _require_exact_keys(plan["heldout"], {"view", "policy"}, "plan.heldout")
    if plan["heldout"]["view"] != HELDOUT_VIEW:
        raise RuntimeError("plan held-out declaration changed")
    _assert_heldout_only_in_declaration(plan, context="plan")
    expected_pairs = [
        {"reference": pair.reference, "target": pair.target, "pair_id": pair.pair_id}
        for pair in PAIRS
    ]
    if plan["pairs"] != expected_pairs:
        raise RuntimeError("plan pair records changed")
    _require_exact_keys(
        plan["decision_rule"],
        {"mode", "binary_mechanism_success", "policy"},
        "plan.decision_rule",
    )
    if plan["decision_rule"] != _frozen_decision_rule():
        raise RuntimeError("plan must remain explicitly descriptive-only")
    _require_exact_keys(
        plan["teacher_bundle"],
        {
            "bundle_path",
            "manifest",
            "reference_view",
            "reference_teacher",
            "manifest_declared_teacher_path",
            "manifest_declared_teacher_sha256",
            "component_count",
        },
        "plan.teacher_bundle",
    )
    if (
        plan["teacher_bundle"]["reference_view"] != REFERENCE_VIEW
        or plan["teacher_bundle"]["component_count"] != REFERENCE_COMPONENT_COUNT
        or plan["teacher_bundle"]["bundle_path"] != display_path(TEACHER_BUNDLE)
    ):
        raise RuntimeError("plan reference teacher changed")
    for name in ("manifest", "reference_teacher"):
        _require_exact_keys(
            plan["teacher_bundle"][name],
            {"path", "sha256", "bytes", "mtime_ns"},
            f"plan.teacher_bundle.{name}",
        )
        _require_sha256(
            plan["teacher_bundle"][name]["sha256"],
            context=f"plan.teacher_bundle.{name}.sha256",
        )
    if (
        plan["teacher_bundle"]["manifest_declared_teacher_sha256"]
        != plan["teacher_bundle"]["reference_teacher"]["sha256"]
    ):
        raise RuntimeError("plan teacher manifest/reference digest binding changed")
    declared_teacher = str(plan["teacher_bundle"]["manifest_declared_teacher_path"])
    declared_teacher_path = Path(declared_teacher)
    if (
        declared_teacher_path.is_absolute()
        or ".." in declared_teacher_path.parts
        or "\\" in declared_teacher
        or declared_teacher_path.as_posix() != declared_teacher
    ):
        raise RuntimeError("plan teacher archive path is not canonical inside the bundle")
    expected_teacher_path = (TEACHER_BUNDLE / declared_teacher_path).resolve()
    try:
        expected_teacher_path.relative_to(TEACHER_BUNDLE.resolve())
    except ValueError as exc:
        raise RuntimeError("plan teacher archive escapes the bundle") from exc
    if plan["teacher_bundle"]["manifest"]["path"] != display_path(
        TEACHER_BUNDLE / "manifest.json"
    ) or plan["teacher_bundle"]["reference_teacher"]["path"] != display_path(expected_teacher_path):
        raise RuntimeError("plan teacher bundle paths are not canonical")
    _require_exact_keys(
        plan["scientific_inputs"],
        {"calibration", "source_colors", "source_masks"},
        "plan.scientific_inputs",
    )
    _require_exact_keys(
        plan["scientific_inputs"]["calibration"],
        {"path", "sha256", "bytes", "mtime_ns"},
        "plan.scientific_inputs.calibration",
    )
    if set(plan["scientific_inputs"]["source_colors"]) != {REFERENCE_VIEW, *TARGET_VIEWS}:
        raise RuntimeError("plan source-color view set changed")
    if set(plan["scientific_inputs"]["source_masks"]) != {
        REFERENCE_VIEW,
        *TARGET_VIEWS,
        *BOUNDS_VIEWS,
    }:
        raise RuntimeError("plan source-mask view set changed")
    for group in ("source_colors", "source_masks"):
        for view, binding in plan["scientific_inputs"][group].items():
            _require_exact_keys(
                binding,
                {"path", "sha256", "bytes", "mtime_ns"},
                f"plan.scientific_inputs.{group}.{view}",
            )
            _require_sha256(
                binding["sha256"],
                context=f"plan.scientific_inputs.{group}.{view}.sha256",
            )
            expected_path = COMPACT_VIEWS / f"{view}.rtgsv"
            if binding["path"] != display_path(expected_path):
                raise RuntimeError(
                    f"plan.scientific_inputs.{group}.{view} is not bound to its compact view"
                )
    _require_sha256(
        plan["scientific_inputs"]["calibration"]["sha256"],
        context="plan.scientific_inputs.calibration.sha256",
    )
    if set(plan["code_sources"]) != {path.as_posix() for path in REPLAY_CODE_PATHS}:
        raise RuntimeError("plan code-source set changed")
    output = _output_from_display(plan["output_path"])
    for label, binding in plan["code_sources"].items():
        _require_exact_keys(
            binding,
            {"path", "sha256", "bytes", "mtime_ns", "replay_path"},
            f"plan.code_sources.{label}",
        )
        _require_sha256(binding["sha256"], context=f"plan.code_sources.{label}.sha256")
        if binding["path"] != display_path(_canonical_code_source(label)) or binding[
            "replay_path"
        ] != display_path(_canonical_replay_source(output, label)):
            raise RuntimeError(f"plan code-source label {label} is misbound")
    _require_exact_keys(
        plan["git"],
        {"head", "dirty", "status_sha256", "tracked_diff_sha256", "dirty_state_digest"},
        "plan.git",
    )
    for name in ("status_sha256", "tracked_diff_sha256", "dirty_state_digest"):
        _require_sha256(plan["git"][name], context=f"plan.git.{name}")
    _require_exact_keys(
        plan["replay_package"],
        {"path", "collection_sha256", "policy"},
        "plan.replay_package",
    )
    _require_sha256(
        plan["replay_package"]["collection_sha256"],
        context="plan.replay_package.collection_sha256",
    )
    if plan["replay_package"]["path"] != display_path(output / "replay_sources"):
        raise RuntimeError("plan replay package is not rooted under the bound run output")
    _require_exact_keys(
        plan["crop"],
        {
            "native_xywh",
            "expected_input_size",
            "crop_size",
            "downscale",
            "flow_size",
            "resize",
            "mean_transform",
            "covariance_transform",
        },
        "plan.crop",
    )
    if (
        plan["crop"]["native_xywh"] != list(FLOW_CROP)
        or plan["crop"]["expected_input_size"] != [NATIVE_WIDTH, NATIVE_HEIGHT]
        or plan["crop"]["crop_size"] != [FLOW_CROP[2], FLOW_CROP[3]]
        or plan["crop"]["downscale"] != FLOW_DOWNSCALE
        or plan["crop"]["flow_size"] != [FLOW_WIDTH, FLOW_HEIGHT]
    ):
        raise RuntimeError("plan crop protocol changed")
    if set(plan["gates"]) != {field.name for field in dataclasses.fields(GateConfig)}:
        raise RuntimeError("plan gate key set changed")
    GateConfig(**plan["gates"])
    if tuple(float(value) for value in plan["fb_sensitivity_px"]) != FB_SENSITIVITY_PX:
        raise RuntimeError("plan FB sensitivity protocol changed")
    _require_exact_keys(
        plan["flow"],
        {
            "backend",
            "device",
            "weights",
            "precision",
            "network_policy",
            "all_reference_components",
            "all_five_sigma_points_required",
        },
        "plan.flow",
    )
    if (
        plan["flow"]["backend"] != FLOW_BACKEND
        or plan["flow"]["device"] != ACQUISITION_DEVICE
        or plan["flow"]["all_reference_components"] is not True
        or plan["flow"]["all_five_sigma_points_required"] is not True
    ):
        raise RuntimeError("plan flow protocol changed")
    _require_exact_keys(
        plan["bounds"],
        {"views", "source", "aabb", "bounds_scale"},
        "plan.bounds",
    )
    if tuple(plan["bounds"]["views"]) != BOUNDS_VIEWS:
        raise RuntimeError("plan bounds view set changed")
    if float(plan["bounds"]["bounds_scale"]) != BOUNDS_SCALE:
        raise RuntimeError("plan bounds scale changed")
    _require_exact_keys(
        plan["artifacts"],
        {"post_flow_policy", "ply", "viewer"},
        "plan.artifacts",
    )
    _require_exact_keys(
        plan["estimated_memory"],
        {
            "flow_arrays_bytes",
            "two_flow_input_tensors_bytes",
            "conservative_peak_gpu_bytes",
            "conservative_peak_gpu_gib",
            "note",
        },
        "plan.estimated_memory",
    )
    expected_flow_bytes = len(PAIRS) * 2 * FLOW_HEIGHT * FLOW_WIDTH * 2 * 4
    expected_workspace_bytes = 2 * 3 * FLOW_HEIGHT * FLOW_WIDTH * 4
    if (
        plan["estimated_memory"]["flow_arrays_bytes"] != expected_flow_bytes
        or plan["estimated_memory"]["two_flow_input_tensors_bytes"] != expected_workspace_bytes
    ):
        raise RuntimeError("plan memory dimensions changed")
    _require_exact_keys(
        plan["commands"],
        {
            "plan",
            "acquire_local_weights_only",
            "acquire_allow_official_weight_download",
            "analyze",
            "viewer_smoke",
        },
        "plan.commands",
    )


def _load_and_verify_plan(
    output: Path,
    *,
    verify_sources: bool = True,
    verify_source_color_bytes: bool = True,
) -> dict[str, Any]:
    plan_path = output / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("run the plan phase first")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan_schema(plan)
    recorded_output = Path(str(plan["output_path"]))
    if not recorded_output.is_absolute():
        recorded_output = ROOT / recorded_output
    if recorded_output.resolve() != output.resolve():
        raise RuntimeError("plan output path does not bind this run directory")
    if verify_sources:
        source_hashes = _current_source_hashes(
            plan,
            verify_source_color_bytes=verify_source_color_bytes,
        )
        expected_collection = canonical_hash(
            {label: record["sha256"] for label, record in plan["code_sources"].items()}
        )
        if expected_collection != plan["replay_package"]["collection_sha256"]:
            raise RuntimeError("plan replay collection digest changed")
        if not source_hashes:
            raise RuntimeError("plan verified no scientific/code sources")
    return plan


def write_plan(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    replay = output / "replay_sources"
    if path.exists() or replay.exists():
        raise FileExistsError("refusing to overwrite immutable plan/replay package")
    payload = plan_payload(output=output)
    temporary = Path(tempfile.mkdtemp(prefix=".g2sr-replay-", dir=output))
    try:
        for label, record in payload["code_sources"].items():
            destination = temporary / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = _canonical_code_source(label)
            if record["path"] != display_path(source):
                raise RuntimeError(f"refusing a misbound code-source record for {label}")
            shutil.copyfile(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise RuntimeError(f"replay copy digest mismatch for {label}")
        os.replace(temporary, replay)
        temporary = None
        write_json_exclusive(path, payload)
    except BaseException:
        if path.exists():
            path.unlink()
        shutil.rmtree(replay, ignore_errors=True)
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return path


def _module_source_record(module_name: str, *, kind: str, qualname: str) -> dict[str, Any]:
    """Bind one imported runtime implementation to its defining module bytes."""
    module = importlib.import_module(module_name)
    source_text = getattr(module, "__file__", None)
    if source_text is None:
        raise RuntimeError(f"{kind} module {module_name!r} has no file-backed implementation")
    source = Path(source_text).resolve()
    if source.suffix in {".pyc", ".pyo"}:
        candidate = Path(importlib.util.source_from_cache(str(source)))
        if candidate.is_file():
            source = candidate.resolve()
    if not source.is_file():
        raise RuntimeError(f"{kind} implementation source is absent: {source}")
    return {
        "kind": kind,
        "module": module_name,
        "qualname": qualname,
        "path": str(source),
        "sha256": sha256_file(source),
    }


def _callable_source_records(
    callable_value: Any,
    *,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    """Record decorated entrypoint, unwrapped callable, and declared implementation module.

    ``inspect.getsourcefile`` alone is insufficient for torchvision entrypoints decorated with
    ``handle_legacy_interface``: it can identify the wrapper source while the numerical factory
    lives in the declared optical-flow module.  We therefore bind all three identities.
    """
    unwrapped = inspect.unwrap(callable_value)
    entrypoint_source_text = inspect.getsourcefile(callable_value)
    unwrapped_source_text = inspect.getsourcefile(unwrapped)
    records: dict[str, dict[str, Any]] = {}
    for label, value, source_text in (
        ("entrypoint", callable_value, entrypoint_source_text),
        ("unwrapped", unwrapped, unwrapped_source_text),
    ):
        if source_text is None:
            raise RuntimeError(f"{prefix} {label} callable has no inspectable source file")
        source = Path(source_text).resolve()
        records[f"{prefix}_{label}"] = {
            "kind": f"{prefix}_{label}",
            "module": str(getattr(value, "__module__", "")),
            "qualname": str(getattr(value, "__qualname__", getattr(value, "__name__", ""))),
            "path": str(source),
            "sha256": sha256_file(source),
        }
    declared_module = str(getattr(callable_value, "__module__", ""))
    records[f"{prefix}_module"] = _module_source_record(
        declared_module,
        kind=f"{prefix}_module",
        qualname=str(
            getattr(unwrapped, "__qualname__", getattr(callable_value, "__qualname__", ""))
        ),
    )
    return records


def _torchvision_runtime_source_records(
    raft_entrypoint: Any,
    transforms: Any,
) -> dict[str, dict[str, Any]]:
    records = _callable_source_records(raft_entrypoint, prefix="raft")
    transforms_type = type(transforms)
    records["transforms_module"] = _module_source_record(
        transforms_type.__module__,
        kind="transforms_module",
        qualname=transforms_type.__qualname__,
    )
    return records


class TorchvisionRaftSmallCV2:
    """Lazy torchvision RAFT-small C_T_V2 backend with local-only weights by default."""

    name = FLOW_BACKEND

    def __init__(self, device: str, *, allow_download: bool = False) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RAFT acquisition requested CUDA but CUDA is unavailable")
        self.allow_download = bool(allow_download)
        self._model: Any | None = None
        self._transforms: Any | None = None
        self._metadata: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        # Acquisition-only lazy import: importing this benchmark never imports torchvision.
        import torchvision
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights.C_T_V2
        filename = Path(weights.url).name
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        cache_existed = cache_path.is_file()
        if not cache_existed and not self.allow_download:
            raise FileNotFoundError(
                f"official RAFT-small C_T_V2 weights are not cached at {cache_path}; "
                "rerun with --allow-download to authorize torchvision's official URL"
            )
        model = raft_small(weights=weights, progress=False).eval().to(self.device)
        if not cache_path.is_file():
            raise RuntimeError("torchvision returned a weighted model without a cache artifact")
        model.float()
        transforms = weights.transforms()
        implementation_sources = _torchvision_runtime_source_records(raft_small, transforms)
        model_device = str(next(model.parameters()).device)
        if self.device.type != "cuda" or not model_device.startswith("cuda"):
            raise RuntimeError("frozen G²SR acquisition requires the RAFT model to execute on CUDA")
        self._model = model
        self._transforms = transforms
        self._metadata = {
            "backend": self.name,
            "torchvision_version": torchvision.__version__,
            "weights_enum": "Raft_Small_Weights.C_T_V2",
            "weights_url": weights.url,
            "weights_cache_path": str(cache_path),
            "weights_cache_preexisted": cache_existed,
            "download_authorized": self.allow_download,
            "weights_sha256": sha256_file(cache_path),
            "weights_recipe": weights.meta.get("recipe"),
            "implementation_sources": implementation_sources,
            "runtime": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "torchvision_module_path": str(Path(torchvision.__file__).resolve()),
                "torchvision_module_sha256": sha256_file(Path(torchvision.__file__).resolve()),
                "requested_device": str(self.device),
                "model_parameter_device": model_device,
                "cuda_runtime": torch.version.cuda,
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(self.device),
            "precision": "float32",
        }

    def infer(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        self._load()
        assert self._model is not None and self._transforms is not None
        if first.shape != second.shape or first.shape != (3, FLOW_HEIGHT, FLOW_WIDTH):
            raise ValueError(f"RAFT inputs must both have shape (3,{FLOW_HEIGHT},{FLOW_WIDTH})")
        first_batch, second_batch = self._transforms(
            first[None].to(self.device, dtype=torch.float32),
            second[None].to(self.device, dtype=torch.float32),
        )
        with torch.inference_mode():
            prediction = self._model(first_batch, second_batch)[-1]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        flow = prediction[0].permute(1, 2, 0).to(device="cpu", dtype=torch.float32)
        if flow.shape != (FLOW_HEIGHT, FLOW_WIDTH, 2) or not bool(torch.isfinite(flow).all()):
            raise RuntimeError("RAFT returned a malformed or non-finite flow field")
        del first_batch, second_batch, prediction
        return flow.contiguous()

    def metadata(self) -> dict[str, Any]:
        self._load()
        return dict(self._metadata)

    def close(self) -> None:
        self._model = None
        self._transforms = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def create_flow_backend(
    name: str,
    *,
    device: str,
    allow_download: bool,
) -> DenseFlowBackend:
    if device != ACQUISITION_DEVICE:
        raise ValueError(f"frozen acquisition device is {ACQUISITION_DEVICE!r}, got {device!r}")
    if name == FLOW_BACKEND:
        return TorchvisionRaftSmallCV2(device, allow_download=allow_download)
    raise ValueError(f"unknown flow backend {name!r}")


def _calibration_records() -> tuple[dict[str, dict[str, Any]], Path]:
    calibration_path = SCENE.parent / "calibration_dome.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    records = {str(record["camera_id"]).upper(): record for record in payload["cameras"]}
    return records, calibration_path


def _camera_for_native_view(view_id: str, record: Mapping[str, Any]) -> Camera:
    from PIL import Image as PILImage

    path = SCENE / "rgb" / f"{view_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    with PILImage.open(path) as image:
        width, height = image.size
    return _camera_for_dimensions(record, width=width, height=height)


def _camera_for_dimensions(
    record: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> Camera:
    intrinsics = record["intrinsics"]
    calibration_width, calibration_height = map(int, intrinsics["resolution"])
    sx, sy = width / calibration_width, height / calibration_height
    matrix = intrinsics["camera_matrix"]
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).view(4, 4)
    return Camera(
        fx=float(matrix[0]) * sx,
        fy=float(matrix[4]) * sy,
        cx=(float(matrix[2]) + 0.5) * sx,
        cy=(float(matrix[5]) + 0.5) * sy,
        width=width,
        height=height,
        R=view[:3, :3],
        t=view[:3, 3],
    )


def _load_undistorted_rgb(view_id: str, camera: Camera, record: Mapping[str, Any]) -> torch.Tensor:
    from rtgs.data.calibrated import _resize_image, _undistort

    path = SCENE / "rgb" / f"{view_id}.jpg"
    image = _resize_image(path, camera.width, camera.height)
    return _undistort(
        image,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        list(record["intrinsics"].get("distortion_coefficients", [])),
    )


def _load_undistorted_mask(
    view_id: str,
    camera: Camera,
    record: Mapping[str, Any],
) -> torch.Tensor:
    from rtgs.data.calibrated import _resize_image, _undistort

    path = SCENE / "mask" / f"mask_{view_id}.png"
    mask = _resize_image(path, camera.width, camera.height, mask=True)
    return _undistort(
        mask,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        list(record["intrinsics"].get("distortion_coefficients", [])),
        mask=True,
    ).float()


def crop_area_resize(tensor: torch.Tensor) -> torch.Tensor:
    """Crop native HWC/HW input and perform the exact 3x area resize."""
    x, y, width, height = FLOW_CROP
    if not isinstance(tensor, torch.Tensor) or not tensor.dtype.is_floating_point:
        raise TypeError("crop_area_resize expects a floating torch.Tensor")
    if tensor.ndim == 3:
        if tensor.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
            raise ValueError(
                "source-color tensor must have exact native HWC shape "
                f"({NATIVE_HEIGHT},{NATIVE_WIDTH},3), got {tuple(tensor.shape)}"
            )
        cropped = tensor[y : y + height, x : x + width].permute(2, 0, 1)[None]
        result = F.interpolate(cropped, size=FLOW_SHAPE, mode="area")[0]
    elif tensor.ndim == 2:
        if tensor.shape != (NATIVE_HEIGHT, NATIVE_WIDTH):
            raise ValueError(
                "mask tensor must have exact native HW shape "
                f"({NATIVE_HEIGHT},{NATIVE_WIDTH}), got {tuple(tensor.shape)}"
            )
        cropped = tensor[y : y + height, x : x + width][None, None]
        result = F.interpolate(cropped, size=FLOW_SHAPE, mode="area")[0, 0]
    else:
        raise ValueError("crop_area_resize expects an HWC or HW tensor")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("crop_area_resize input produced non-finite values")
    return result.contiguous()


def _camera_difference(left: Camera, right: Camera) -> float:
    scalars = max(
        abs(left.fx - right.fx),
        abs(left.fy - right.fy),
        abs(left.cx - right.cx),
        abs(left.cy - right.cy),
    )
    tensors = max(
        float((left.R - right.R).abs().max()),
        float((left.t - right.t).abs().max()),
    )
    return max(scalars, tensors)


def _derived_flow_visual(flow: torch.Tensor) -> np.ndarray:
    """Create an RGB visualization derived only from flow, never from a source frame."""
    magnitude = flow.norm(dim=-1)
    finite = magnitude[torch.isfinite(magnitude)]
    scale = float(torch.quantile(finite, 0.99).clamp_min(1.0)) if finite.numel() else 1.0
    x = (flow[..., 0] / scale).clamp(-1.0, 1.0)
    y = (flow[..., 1] / scale).clamp(-1.0, 1.0)
    strength = (magnitude / scale).clamp(0.0, 1.0)
    visual = torch.stack([0.5 + 0.5 * x, 0.5 + 0.5 * y, 1.0 - strength], dim=-1)
    return visual.clamp(0.0, 1.0).mul(255).round().byte().numpy()


def _save_flow_visual(path: Path, flow: torch.Tensor) -> None:
    from PIL import Image as PILImage

    PILImage.fromarray(_derived_flow_visual(flow), mode="RGB").save(path)


def _save_npz_exclusive(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_npz_no_source_rgb(path: Path) -> tuple[str, ...]:
    """Reject source-color payload names at the acquisition/analysis boundary.

    RGB-encoded flow visualizations are allowed as separate PNGs because their display values
    are derived only from displacement vectors. No decoded source-color tensor is allowed in an
    NPZ.
    """
    forbidden_tokens = ("rgb", "source_image", "target_image", "decoded_frame")
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive.files)
    bad = [name for name in names if any(token in name.lower() for token in forbidden_tokens)]
    if bad:
        raise RuntimeError(f"post-flow artifact contains forbidden source RGB keys: {bad}")
    return names


def _validate_np_array(
    array: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    finite: bool = True,
) -> None:
    expected_dtype = np.dtype(dtype)
    if array.shape != shape:
        raise RuntimeError(f"{name} shape must be {shape}, got {array.shape}")
    if array.dtype != expected_dtype:
        raise RuntimeError(f"{name} dtype must be {expected_dtype}, got {array.dtype}")
    if finite and not bool(np.isfinite(array).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def validate_flow_npz(path: Path) -> tuple[str, ...]:
    names = audit_npz_no_source_rgb(path)
    if set(names) != {"forward", "backward"} or len(names) != 2:
        raise RuntimeError(f"flow NPZ key set mismatch: {names}")
    with np.load(path, allow_pickle=False) as archive:
        for name in ("forward", "backward"):
            _validate_np_array(
                archive[name],
                name=f"{path.name}:{name}",
                shape=(FLOW_HEIGHT, FLOW_WIDTH, 2),
                dtype=np.float32,
            )
    return names


def _analysis_input_schema() -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    schema: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "reference_means_native": ((REFERENCE_COMPONENT_COUNT, 2), np.dtype(np.float64)),
        "reference_covariances_native": (
            (REFERENCE_COMPONENT_COUNT, 2, 2),
            np.dtype(np.float64),
        ),
        "reference_colors": ((REFERENCE_COMPONENT_COUNT, 3), np.dtype(np.float32)),
        "reference_amplitudes": ((REFERENCE_COMPONENT_COUNT,), np.dtype(np.float32)),
        "reference_foreground": ((REFERENCE_COMPONENT_COUNT,), np.dtype(np.bool_)),
    }
    for target in TARGET_VIEWS:
        schema[f"target_mask_flow_{target}"] = (FLOW_SHAPE, np.dtype(np.float32))
    return schema


def validate_analysis_input_npz(path: Path) -> tuple[str, ...]:
    names = audit_npz_no_source_rgb(path)
    schema = _analysis_input_schema()
    if set(names) != set(schema) or len(names) != len(schema):
        raise RuntimeError(f"analysis-input NPZ key set mismatch: {names}")
    if any(HELDOUT_VIEW in name.upper() for name in names):
        raise RuntimeError("held-out view leaked into analysis-input NPZ")
    with np.load(path, allow_pickle=False) as archive:
        for name, (shape, dtype) in schema.items():
            _validate_np_array(
                archive[name],
                name=f"{path.name}:{name}",
                shape=shape,
                dtype=dtype,
            )
    return names


def _environment_record(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_utc": utc_now(),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    return result


def _validate_acquisition_manifest_schema(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        {
            "artifact_type",
            "status",
            "started_utc",
            "completed_utc",
            "plan",
            "source_verification",
            "reference",
            "targets",
            "heldout",
            "teacher_bundle_manifest_sha256",
            "reference_teacher",
            "crop",
            "backend",
            "backend_replay",
            "environment",
            "pairs",
            "compact_analysis_input",
            "calibration",
            "bounds",
            "source_bindings",
            "post_flow_rgb_policy",
        },
        "acquisition manifest",
    )
    if manifest["artifact_type"] != ACQUISITION_ARTIFACT_TYPE or manifest["status"] != "COMPLETE":
        raise RuntimeError("acquisition manifest type/status changed")
    _require_utc_timestamp(manifest["started_utc"], context="acquisition.started_utc")
    _require_utc_timestamp(manifest["completed_utc"], context="acquisition.completed_utc")
    if manifest["reference"] != REFERENCE_VIEW or tuple(manifest["targets"]) != TARGET_VIEWS:
        raise RuntimeError("acquisition manifest view protocol changed")
    _require_exact_keys(manifest["heldout"], {"view", "policy"}, "acquisition.heldout")
    if manifest["heldout"]["view"] != HELDOUT_VIEW:
        raise RuntimeError("acquisition held-out declaration changed")
    _assert_heldout_only_in_declaration(manifest, context="acquisition manifest")
    _require_exact_keys(
        manifest["plan"],
        {"path", "file_sha256", "payload_sha256", "created_utc"},
        "acquisition.plan",
    )
    _require_utc_timestamp(
        manifest["plan"]["created_utc"],
        context="acquisition.plan.created_utc",
    )
    _require_sha256(
        manifest["plan"]["file_sha256"],
        context="acquisition.plan.file_sha256",
    )
    _require_sha256(
        manifest["plan"]["payload_sha256"],
        context="acquisition.plan.payload_sha256",
    )
    _require_exact_keys(
        manifest["source_verification"],
        {"verified_utc", "hashes", "git_at_acquisition"},
        "acquisition.source_verification",
    )
    _require_utc_timestamp(
        manifest["source_verification"]["verified_utc"],
        context="acquisition.source_verification.verified_utc",
    )
    _require_exact_keys(
        manifest["source_verification"]["git_at_acquisition"],
        {"head", "dirty", "status_sha256", "tracked_diff_sha256", "dirty_state_digest"},
        "acquisition.source_verification.git_at_acquisition",
    )
    _require_exact_keys(
        manifest["reference_teacher"],
        {
            "component_count",
            "foreground_component_count",
            "archive_path",
            "archive_sha256",
            "manifest_declared_sha256",
            "provider",
            "producer_version",
            "producer_source_digest",
            "fit_config_digest",
            "mean_residuals_present",
            "native_means_api",
        },
        "acquisition.reference_teacher",
    )
    if manifest["reference_teacher"]["component_count"] != REFERENCE_COMPONENT_COUNT:
        raise RuntimeError("acquisition component count changed")
    for name in ("teacher_bundle_manifest_sha256",):
        _require_sha256(manifest[name], context=f"acquisition.{name}")
    for name in (
        "archive_sha256",
        "manifest_declared_sha256",
        "producer_source_digest",
        "fit_config_digest",
    ):
        _require_sha256(
            manifest["reference_teacher"][name],
            context=f"acquisition.reference_teacher.{name}",
        )
    _require_exact_keys(
        manifest["backend"],
        {
            "backend",
            "torchvision_version",
            "weights_enum",
            "weights_url",
            "weights_cache_path",
            "weights_cache_preexisted",
            "download_authorized",
            "weights_sha256",
            "weights_recipe",
            "implementation_sources",
            "runtime",
            "parameter_count",
            "device",
            "precision",
        },
        "acquisition.backend",
    )
    expected_runtime_sources = {
        "raft_entrypoint",
        "raft_unwrapped",
        "raft_module",
        "transforms_module",
    }
    if set(manifest["backend"]["implementation_sources"]) != expected_runtime_sources:
        raise RuntimeError("acquisition backend implementation-source set changed")
    for label, record in manifest["backend"]["implementation_sources"].items():
        _require_exact_keys(
            record,
            {"kind", "module", "qualname", "path", "sha256"},
            f"acquisition.backend.implementation_sources.{label}",
        )
        if record["kind"] != label:
            raise RuntimeError(f"acquisition backend source kind changed for {label}")
        _require_sha256(
            record["sha256"],
            context=f"acquisition.backend.implementation_sources.{label}.sha256",
        )
    _require_exact_keys(
        manifest["backend"]["runtime"],
        {
            "python_version",
            "torch_version",
            "torchvision_version",
            "torchvision_module_path",
            "torchvision_module_sha256",
            "requested_device",
            "model_parameter_device",
            "cuda_runtime",
        },
        "acquisition.backend.runtime",
    )
    _require_sha256(
        manifest["backend"]["runtime"]["torchvision_module_sha256"],
        context="acquisition.backend.runtime.torchvision_module_sha256",
    )
    _require_sha256(
        manifest["backend"]["weights_sha256"],
        context="acquisition.backend.weights_sha256",
    )
    _require_exact_keys(
        manifest["backend_replay"],
        {"implementation_sources", "weights"},
        "acquisition.backend_replay",
    )
    if set(manifest["backend_replay"]["implementation_sources"]) != expected_runtime_sources:
        raise RuntimeError("acquisition backend replay source set changed")
    for name, record in manifest["backend_replay"]["implementation_sources"].items():
        _require_exact_keys(
            record,
            {"path", "sha256", "bytes"},
            f"acquisition.backend_replay.implementation_sources.{name}",
        )
        _require_sha256(
            record["sha256"],
            context=f"acquisition.backend_replay.implementation_sources.{name}.sha256",
        )
    _require_exact_keys(
        manifest["backend_replay"]["weights"],
        {"path", "sha256", "bytes"},
        "acquisition.backend_replay.weights",
    )
    _require_sha256(
        manifest["backend_replay"]["weights"]["sha256"],
        context="acquisition.backend_replay.weights.sha256",
    )
    expected_environment = {
        "captured_utc",
        "python",
        "python_executable",
        "platform",
        "torch",
        "numpy",
        "cuda_available",
        "device",
        "ld_preload",
    }
    if str(manifest["environment"]["device"]).startswith("cuda"):
        expected_environment.add("gpu")
    _require_exact_keys(
        manifest["environment"],
        expected_environment,
        "acquisition.environment",
    )
    _require_utc_timestamp(
        manifest["environment"]["captured_utc"],
        context="acquisition.environment.captured_utc",
    )
    if "gpu" in manifest["environment"]:
        _require_exact_keys(
            manifest["environment"]["gpu"],
            {"name", "total_memory_bytes", "capability"},
            "acquisition.environment.gpu",
        )
    if (
        manifest["backend"]["backend"] != FLOW_BACKEND
        or manifest["backend"]["precision"] != "float32"
        or manifest["backend"]["torchvision_version"]
        != manifest["backend"]["runtime"]["torchvision_version"]
        or manifest["backend"]["runtime"]["torch_version"] != manifest["environment"]["torch"]
        or manifest["environment"]["device"] != ACQUISITION_DEVICE
        or manifest["backend"]["device"] != ACQUISITION_DEVICE
        or manifest["backend"]["runtime"]["requested_device"] != ACQUISITION_DEVICE
        or not str(manifest["backend"]["runtime"]["model_parameter_device"]).startswith("cuda")
        or manifest["environment"]["cuda_available"] is not True
        or manifest["environment"]["ld_preload"] != str(LOCAL_LIBSTDCXX_PRELOAD)
    ):
        raise RuntimeError("acquisition did not execute on the frozen CUDA device")
    if set(manifest["pairs"]) != {pair.pair_id for pair in PAIRS}:
        raise RuntimeError("acquisition pair key set changed")
    for pair in PAIRS:
        record = manifest["pairs"][pair.pair_id]
        _require_exact_keys(
            record,
            {
                "forward_seconds",
                "backward_seconds",
                "forward_peak_vram",
                "backward_peak_vram",
                "flow_path",
                "flow_sha256",
                "flow_keys",
                "completed_utc",
                "flow_visuals",
            },
            f"acquisition.pairs.{pair.pair_id}",
        )
        if record["flow_keys"] != ["forward", "backward"]:
            raise RuntimeError(f"{pair.pair_id} flow key declaration changed")
        _require_utc_timestamp(
            record["completed_utc"],
            context=f"acquisition.pairs.{pair.pair_id}.completed_utc",
        )
        for name in ("forward_seconds", "backward_seconds"):
            value = record[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"{pair.pair_id} {name} is invalid")
        for name in ("forward_peak_vram", "backward_peak_vram"):
            peak = record[name]
            if peak is not None:
                _require_exact_keys(
                    peak,
                    {"allocated_bytes", "reserved_bytes"},
                    f"acquisition.pairs.{pair.pair_id}.{name}",
                )
        _require_exact_keys(
            record["flow_visuals"],
            {
                "forward_path",
                "forward_sha256",
                "backward_path",
                "backward_sha256",
                "policy",
            },
            f"acquisition.pairs.{pair.pair_id}.flow_visuals",
        )
        for name in ("flow_sha256",):
            _require_sha256(record[name], context=f"acquisition.pairs.{pair.pair_id}.{name}")
        for name in ("forward_sha256", "backward_sha256"):
            _require_sha256(
                record["flow_visuals"][name],
                context=f"acquisition.pairs.{pair.pair_id}.flow_visuals.{name}",
            )
    _require_exact_keys(
        manifest["compact_analysis_input"],
        {"path", "sha256", "keys", "contains_decoded_source_rgb"},
        "acquisition.compact_analysis_input",
    )
    if (
        set(manifest["compact_analysis_input"]["keys"]) != set(_analysis_input_schema())
        or manifest["compact_analysis_input"]["contains_decoded_source_rgb"] is not False
    ):
        raise RuntimeError("acquisition compact analysis-input declaration changed")
    _require_sha256(
        manifest["compact_analysis_input"]["sha256"],
        context="acquisition.compact_analysis_input.sha256",
    )
    _require_exact_keys(
        manifest["calibration"],
        {"path", "sha256", "source_path", "source_sha256"},
        "acquisition.calibration",
    )
    _require_exact_keys(manifest["bounds"], {"path", "sha256"}, "acquisition.bounds")
    _require_sha256(manifest["calibration"]["sha256"], context="acquisition.calibration.sha256")
    _require_sha256(
        manifest["calibration"]["source_sha256"],
        context="acquisition.calibration.source_sha256",
    )
    _require_sha256(manifest["bounds"]["sha256"], context="acquisition.bounds.sha256")
    if set(manifest["source_bindings"]) != {REFERENCE_VIEW, *TARGET_VIEWS}:
        raise RuntimeError("acquisition source-binding view set changed")
    for view, binding in manifest["source_bindings"].items():
        _require_exact_keys(
            binding,
            {
                "source_color_path",
                "source_color_sha256",
                "mask_path",
                "mask_sha256",
                "role",
            },
            f"acquisition.source_bindings.{view}",
        )
        if binding["role"] != "acquisition_only":
            raise RuntimeError(f"acquisition.source_bindings.{view}.role changed")
        for name in ("source_color_sha256", "mask_sha256"):
            _require_sha256(
                binding[name],
                context=f"acquisition.source_bindings.{view}.{name}",
            )
    if manifest["crop"]["native_xywh"] != list(FLOW_CROP):
        raise RuntimeError("acquisition crop differs from frozen protocol")


def _write_failure_attempt(
    output: Path,
    *,
    phase: str,
    started_utc: str,
    error: BaseException,
) -> Path:
    path = output / f"{phase}_failure_attempt_{time.time_ns()}.json"
    preserved = sorted(
        display_path(item) for item in output.glob(f"{phase}_failed_partial_*") if item.is_dir()
    )
    payload = {
        "artifact_type": f"g2sr_{phase}_failure_attempt_v1",
        "status": "FAILED",
        "phase": phase,
        "started_utc": started_utc,
        "failed_utc": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error))[-12000:],
        "preserved_partial_directories": preserved,
        "environment": _environment_record(torch.device("cpu")),
    }
    write_json_exclusive(path, payload)
    return path


def _acquire_once(
    output: Path,
    *,
    backend_name: str = FLOW_BACKEND,
    device: str = ACQUISITION_DEVICE,
    allow_download: bool = False,
) -> Path:
    """Run the sole RGB-decoding flow stage and atomically publish non-RGB artifacts."""
    validate_pair_protocol()
    assert_crop_coordinate_convention()
    plan_path = output / "plan.json"
    plan = _load_and_verify_plan(output)
    plan_sha256 = sha256_file(plan_path)
    if backend_name != plan["flow"]["backend"]:
        raise RuntimeError("requested flow backend differs from preregistered plan")
    if device != plan["flow"]["device"] or device != ACQUISITION_DEVICE:
        raise RuntimeError(
            f"requested acquisition device must equal frozen plan device {ACQUISITION_DEVICE!r}"
        )
    active_device = torch.device(device)
    if active_device.type != "cuda":
        raise RuntimeError("frozen G²SR acquisition is CUDA-only")
    if os.environ.get("LD_PRELOAD") != str(LOCAL_LIBSTDCXX_PRELOAD):
        raise RuntimeError(
            "CUDA acquisition requires the exact preregistered LD_PRELOAD; use the plan command"
        )
    planned_pairs = tuple(
        PairSpec(str(record["reference"]), str(record["target"])) for record in plan["pairs"]
    )
    current_source_hashes = _current_source_hashes(plan)
    started_utc = utc_now()
    final = output / "acquisition"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite immutable acquisition: {final}")
    inputs = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
    if tuple(inputs.view_names) != BOUNDS_VIEWS:
        raise RuntimeError("frozen teacher bundle view order changed")
    reference_index = inputs.view_names.index(REFERENCE_VIEW)
    reference_field = inputs.observations[reference_index]
    if reference_field.n != REFERENCE_COMPONENT_COUNT:
        raise RuntimeError("frozen C0014 teacher no longer contains exactly 640 components")
    bundle_manifest = json.loads((TEACHER_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    reference_teacher_record = next(
        record for record in bundle_manifest["views"] if record["view_id"] == REFERENCE_VIEW
    )
    reference_teacher_path = TEACHER_BUNDLE / reference_teacher_record["teacher"]
    reference_bundle_camera = inputs.cameras[reference_index]
    records, calibration_path = _calibration_records()
    if HELDOUT_VIEW in {REFERENCE_VIEW, *TARGET_VIEWS, *BOUNDS_VIEWS}:
        raise RuntimeError("held-out view leaked into acquisition protocol")

    temporary = Path(tempfile.mkdtemp(prefix=".g2sr-acquisition-", dir=output))
    backend: DenseFlowBackend | None = None
    try:
        flows_dir = temporary / "flows"
        visuals_dir = temporary / "flow_visuals"
        flows_dir.mkdir()
        visuals_dir.mkdir()
        backend = create_flow_backend(
            backend_name,
            device=device,
            allow_download=allow_download,
        )
        # Fail on absent/unapproved weights before decoding the first source color frame.
        backend_metadata = backend.metadata()
        backend_replay_dir = temporary / "backend_replay"
        backend_replay_dir.mkdir()
        backend_replay: dict[str, Any] = {"implementation_sources": {}}
        for label, source_record in backend_metadata["implementation_sources"].items():
            source = Path(source_record["path"])
            destination = backend_replay_dir / f"{label}{source.suffix}"
            shutil.copyfile(source, destination)
            backend_replay["implementation_sources"][label] = {
                "path": f"backend_replay/{destination.name}",
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        source = Path(backend_metadata["weights_cache_path"])
        destination = backend_replay_dir / "raft_small_c_t_v2.pth"
        shutil.copyfile(source, destination)
        backend_replay["weights"] = {
            "path": f"backend_replay/{destination.name}",
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
        for label, replay_record in backend_replay["implementation_sources"].items():
            if (
                replay_record["sha256"]
                != backend_metadata["implementation_sources"][label]["sha256"]
            ):
                raise RuntimeError(
                    f"backend replay copy differs from executed runtime source {label}"
                )
        if backend_replay["weights"]["sha256"] != backend_metadata["weights_sha256"]:
            raise RuntimeError("backend replay copy differs from executed weights")
        cameras: dict[str, Camera] = {
            view: _camera_for_native_view(view, records[view])
            for view in {REFERENCE_VIEW, *TARGET_VIEWS}
        }
        for view, camera in cameras.items():
            if (camera.width, camera.height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
                raise RuntimeError(
                    f"{view} native source dimensions must be "
                    f"{NATIVE_WIDTH}x{NATIVE_HEIGHT}, got {camera.width}x{camera.height}"
                )
        if _camera_difference(cameras[REFERENCE_VIEW], reference_bundle_camera) > 1.0e-4:
            raise RuntimeError("parsed C0014 calibration differs from frozen bundle")
        if "C0026" in inputs.view_names:
            bundle_camera = inputs.cameras[inputs.view_names.index("C0026")]
            if _camera_difference(cameras["C0026"], bundle_camera) > 1.0e-4:
                raise RuntimeError("parsed C0026 calibration differs from frozen bundle")

        # Correct bounds use exactly the same seven undistorted training masks/cameras as the
        # repaired CompactCarve branch. No held-out mask is opened.
        bound_masks = []
        bound_mask_bindings: dict[str, Any] = {}
        for view, camera in zip(inputs.view_names, inputs.cameras, strict=True):
            mask = _load_undistorted_mask(view, camera, records[view])
            if mask.shape != (NATIVE_HEIGHT, NATIVE_WIDTH) or not bool(torch.isfinite(mask).all()):
                raise RuntimeError(f"{view} undistorted bound mask has invalid native shape/values")
            mask_path = SCENE / "mask" / f"mask_{view}.png"
            bound_masks.append(mask)
            bound_mask_bindings[view] = {
                "path": display_path(mask_path),
                "source_sha256": sha256_file(mask_path),
                "undistorted_tensor_sha256": tensor_sha256(mask),
                "foreground_fraction": float((mask > 0.5).float().mean()),
            }
        from rtgs.data.calibrated import _object_bounds

        bounds_center, bounds_extent = _object_bounds(inputs.cameras, bound_masks)
        bounds_lower = bounds_center - float(bounds_extent) * BOUNDS_SCALE
        bounds_upper = bounds_center + float(bounds_extent) * BOUNDS_SCALE

        reference_mask = bound_masks[reference_index]
        # Keep exact crop-local float32 means through the large native offset. The flow sampler
        # promotes its FP32 field only for the 5N sparse queries; no source-color tensor enters
        # the serialized analysis payload.
        reference_means = observation_native_means(reference_field, dtype=torch.float64)
        reference_covariances = observation_covariances(
            reference_field,
            dtype=torch.float64,
        )
        reference_foreground = bilinear_sample(reference_mask, reference_means) > 0.5
        del bound_masks, reference_mask, mask

        reference_native = _load_undistorted_rgb(
            REFERENCE_VIEW,
            cameras[REFERENCE_VIEW],
            records[REFERENCE_VIEW],
        )
        reference_flow_frame = crop_area_resize(reference_native)
        del reference_native

        target_masks_flow: dict[str, torch.Tensor] = {}
        source_bindings: dict[str, Any] = {}
        pair_runtime: dict[str, Any] = {}
        for pair in planned_pairs:
            target_native = _load_undistorted_rgb(
                pair.target,
                cameras[pair.target],
                records[pair.target],
            )
            target_flow_frame = crop_area_resize(target_native)
            del target_native
            target_mask = _load_undistorted_mask(
                pair.target,
                cameras[pair.target],
                records[pair.target],
            )
            target_masks_flow[pair.target] = crop_area_resize(target_mask)
            del target_mask

            if active_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(active_device)
            start = time.perf_counter()
            forward = backend.infer(reference_flow_frame, target_flow_frame)
            forward_seconds = time.perf_counter() - start
            if active_device.type == "cuda":
                forward_peak = {
                    "allocated_bytes": torch.cuda.max_memory_allocated(active_device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(active_device),
                }
                torch.cuda.reset_peak_memory_stats(active_device)
            else:
                forward_peak = None
            start = time.perf_counter()
            backward = backend.infer(target_flow_frame, reference_flow_frame)
            backward_seconds = time.perf_counter() - start
            if active_device.type == "cuda":
                backward_peak = {
                    "allocated_bytes": torch.cuda.max_memory_allocated(active_device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(active_device),
                }
            else:
                backward_peak = None

            flow_path = flows_dir / f"{pair.pair_id}.npz"
            _save_npz_exclusive(
                flow_path,
                forward=forward.numpy(),
                backward=backward.numpy(),
            )
            flow_keys = validate_flow_npz(flow_path)
            _save_flow_visual(visuals_dir / f"{pair.pair_id}_forward.png", forward)
            _save_flow_visual(visuals_dir / f"{pair.pair_id}_backward.png", backward)
            forward_visual = visuals_dir / f"{pair.pair_id}_forward.png"
            backward_visual = visuals_dir / f"{pair.pair_id}_backward.png"
            pair_runtime[pair.pair_id] = {
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "forward_peak_vram": forward_peak,
                "backward_peak_vram": backward_peak,
                "flow_path": f"flows/{flow_path.name}",
                "flow_sha256": sha256_file(flow_path),
                "flow_keys": list(flow_keys),
                "completed_utc": utc_now(),
                "flow_visuals": {
                    "forward_path": f"flow_visuals/{forward_visual.name}",
                    "forward_sha256": sha256_file(forward_visual),
                    "backward_path": f"flow_visuals/{backward_visual.name}",
                    "backward_sha256": sha256_file(backward_visual),
                    "policy": "RGB display values derived only from flow vectors",
                },
            }
            del target_flow_frame, forward, backward

        compact_path = temporary / "analysis_input.npz"
        arrays: dict[str, np.ndarray] = {
            "reference_means_native": reference_means.numpy(),
            "reference_covariances_native": reference_covariances.numpy(),
            "reference_colors": reference_field.colors.float().numpy(),
            "reference_amplitudes": reference_field.amplitudes.float().numpy(),
            "reference_foreground": reference_foreground.numpy(),
        }
        for target, mask in target_masks_flow.items():
            arrays[f"target_mask_flow_{target}"] = mask.numpy()
        _save_npz_exclusive(compact_path, **arrays)
        compact_keys = validate_analysis_input_npz(compact_path)

        calibration_payload = {
            "native": {view: camera_record(camera) for view, camera in sorted(cameras.items())},
            "flow": {
                view: camera_record(flow_camera(camera)) for view, camera in sorted(cameras.items())
            },
        }
        write_json_exclusive(temporary / "calibration.json", calibration_payload)
        bounds_payload = {
            "source": "seven_training_masks",
            "views": list(BOUNDS_VIEWS),
            "center": bounds_center.tolist(),
            "extent": float(bounds_extent),
            "bounds_scale": BOUNDS_SCALE,
            "aabb_lower": bounds_lower.tolist(),
            "aabb_upper": bounds_upper.tolist(),
            "masks": bound_mask_bindings,
        }
        write_json_exclusive(temporary / "bounds.json", bounds_payload)

        for view in {REFERENCE_VIEW, *TARGET_VIEWS}:
            rgb_path = SCENE / "rgb" / f"{view}.jpg"
            mask_path = SCENE / "mask" / f"mask_{view}.png"
            source_bindings[view] = {
                "source_color_path": display_path(rgb_path),
                "source_color_sha256": sha256_file(rgb_path),
                "mask_path": display_path(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "role": "acquisition_only",
            }
        manifest = {
            "artifact_type": ACQUISITION_ARTIFACT_TYPE,
            "status": "COMPLETE",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "plan": {
                "path": display_path(plan_path),
                "file_sha256": plan_sha256,
                "payload_sha256": canonical_hash(plan),
                "created_utc": plan["created_utc"],
            },
            "source_verification": {
                "verified_utc": utc_now(),
                "hashes": current_source_hashes,
                "git_at_acquisition": _git_record(),
            },
            "reference": REFERENCE_VIEW,
            "targets": list(TARGET_VIEWS),
            "heldout": {
                "view": HELDOUT_VIEW,
                "policy": "declaration only; absent from all acquired inputs and outputs",
            },
            "teacher_bundle_manifest_sha256": sha256_file(TEACHER_BUNDLE / "manifest.json"),
            "reference_teacher": {
                "component_count": reference_field.n,
                "foreground_component_count": int(reference_foreground.sum()),
                "archive_path": display_path(reference_teacher_path),
                "archive_sha256": sha256_file(reference_teacher_path),
                "manifest_declared_sha256": reference_teacher_record["teacher_sha256"],
                "provider": reference_field.provider,
                "producer_version": reference_field.producer_version,
                "producer_source_digest": reference_field.producer_source_digest,
                "fit_config_digest": reference_field.fit_config_digest,
                "mean_residuals_present": reference_field.mean_residuals is not None,
                "native_means_api": (
                    "native_means"
                    if callable(getattr(reference_field, "native_means", None))
                    else "field.means compatibility fallback"
                ),
            },
            "crop": plan["crop"],
            "backend": backend_metadata,
            "backend_replay": backend_replay,
            "environment": _environment_record(active_device),
            "pairs": pair_runtime,
            "compact_analysis_input": {
                "path": compact_path.name,
                "sha256": sha256_file(compact_path),
                "keys": list(compact_keys),
                "contains_decoded_source_rgb": False,
            },
            "calibration": {
                "path": "calibration.json",
                "sha256": sha256_file(temporary / "calibration.json"),
                "source_path": display_path(calibration_path),
                "source_sha256": sha256_file(calibration_path),
            },
            "bounds": {
                "path": "bounds.json",
                "sha256": sha256_file(temporary / "bounds.json"),
            },
            "source_bindings": source_bindings,
            "post_flow_rgb_policy": (
                "decoded source-color tensors were used transiently only by flow acquisition "
                "and were not serialized; flow-visual PNG RGB values derive only from flow"
            ),
        }
        _validate_acquisition_manifest_schema(manifest)
        write_json_exclusive(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
        temporary = None
        return final
    except BaseException as exc:
        failure = output / f"acquisition_failed_partial_{time.time_ns()}"
        if temporary is not None and temporary.exists():
            write_json_exclusive(
                temporary / "failure_context.json",
                {
                    "artifact_type": "g2sr_acquisition_partial_failure_v1",
                    "failed_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "plan_file_sha256": plan_sha256,
                },
            )
            os.replace(temporary, failure)
            temporary = None
        raise
    finally:
        if backend is not None:
            backend.close()
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def acquire(
    output: Path,
    *,
    backend_name: str = FLOW_BACKEND,
    device: str = ACQUISITION_DEVICE,
    allow_download: bool = False,
) -> Path:
    """Run acquisition and preserve an immutable failure receipt on every exception."""
    started_utc = utc_now()
    try:
        return _acquire_once(
            output,
            backend_name=backend_name,
            device=device,
            allow_download=allow_download,
        )
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        _write_failure_attempt(
            output,
            phase="acquisition",
            started_utc=started_utc,
            error=exc,
        )
        raise


def calibrated_epipolar_residual(
    first: Camera,
    second: Camera,
    first_points: torch.Tensor,
    second_points: torch.Tensor,
) -> torch.Tensor:
    """Pixel-space Sampson residual from the calibrated camera pair."""
    if first_points.shape != second_points.shape or first_points.shape[-1:] != (2,):
        raise ValueError("epipolar point tensors must have identical (...,2) shapes")
    dtype = torch.promote_types(first_points.dtype, second_points.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    r1, t1 = first.R.to(dtype=dtype), first.t.to(dtype=dtype)
    r2, t2 = second.R.to(dtype=dtype), second.t.to(dtype=dtype)
    relative_rotation = r2 @ r1.T
    relative_translation = t2 - relative_rotation @ t1
    tx, ty, tz = relative_translation
    zero = tx.new_zeros(())
    skew = torch.stack(
        [
            torch.stack([zero, -tz, ty]),
            torch.stack([tz, zero, -tx]),
            torch.stack([-ty, tx, zero]),
        ]
    )
    essential = skew @ relative_rotation
    fundamental = (
        torch.linalg.inv(second.K.to(dtype)).T @ essential @ torch.linalg.inv(first.K.to(dtype))
    )
    ones = first_points.new_ones(first_points.shape[:-1] + (1,), dtype=dtype)
    x1 = torch.cat([first_points.to(dtype), ones], dim=-1)
    x2 = torch.cat([second_points.to(dtype), ones], dim=-1)
    fx1 = torch.einsum("ij,...j->...i", fundamental, x1)
    ftx2 = torch.einsum("ji,...j->...i", fundamental, x2)
    numerator = torch.einsum("...i,...i->...", x2, fx1).abs()
    denominator = (fx1[..., :2].square().sum(dim=-1) + ftx2[..., :2].square().sum(dim=-1)).sqrt()
    return numerator / denominator.clamp_min(torch.finfo(dtype).eps)


def ray_angle_degrees(
    first: Camera,
    second: Camera,
    first_points: torch.Tensor,
    second_points: torch.Tensor,
) -> torch.Tensor:
    _, first_rays = first.pixel_rays(first_points)
    _, second_rays = second.pixel_rays(second_points)
    first_unit = F.normalize(first_rays, dim=-1)
    second_unit = F.normalize(second_rays, dim=-1)
    cosine = (first_unit * second_unit).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _finite_quantiles(values: torch.Tensor) -> dict[str, float | None]:
    finite = values.detach().cpu()
    if finite.dtype in {torch.float16, torch.bfloat16}:
        finite = finite.float()
    finite = finite[torch.isfinite(finite)]
    if not finite.numel():
        return {str(q): None for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}
    return {str(q): float(torch.quantile(finite, q)) for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}


def _masked_distribution(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if values.shape != mask.shape:
        raise ValueError("distribution mask must match values")
    selected = values[mask].detach().cpu()
    finite = torch.isfinite(selected)
    nan = torch.isnan(selected)
    positive_infinity = torch.isposinf(selected)
    negative_infinity = torch.isneginf(selected)
    counts = {
        "total": int(selected.numel()),
        "finite": int(finite.sum()),
        "nan": int(nan.sum()),
        "positive_infinity": int(positive_infinity.sum()),
        "negative_infinity": int(negative_infinity.sum()),
    }
    if sum(counts[name] for name in counts if name != "total") != counts["total"]:
        raise RuntimeError("distribution value classes do not partition the selected values")
    return {
        "finite_quantiles": _finite_quantiles(selected),
        "value_class_counts": counts,
    }


def _encode_archive_float(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every nonfinite class without placing NaN/Inf in the immutable NPZ.

    The finite payload is convenient for strict archive readers, while the uint8 sidecar retains
    whether each replaced element was NaN, +Inf, or -Inf. In particular, +Inf is meaningful for
    an exact DLT nullspace gap and must not be silently turned into an ordinary zero.
    """
    if not value.dtype.is_floating_point:
        raise TypeError("nonfinite archive encoding requires a floating tensor")
    code = torch.zeros_like(value, dtype=torch.uint8)
    code = torch.where(torch.isnan(value), torch.ones_like(code), code)
    code = torch.where(torch.isposinf(value), torch.full_like(code, 2), code)
    code = torch.where(torch.isneginf(value), torch.full_like(code, 3), code)
    finite = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return finite, code


def restore_archived_nonfinite(value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """Invert :func:`_encode_archive_float` for gate/distribution replay."""
    if not value.dtype.is_floating_point or code.dtype != torch.uint8 or value.shape != code.shape:
        raise ValueError("archived value/code tensors must have equal shape and float/uint8 dtypes")
    if bool((code > 3).any()):
        raise ValueError("archived nonfinite code must be in [0,3]")
    result = value.clone()
    result[code == 1] = torch.nan
    result[code == 2] = torch.inf
    result[code == 3] = -torch.inf
    return result


def recovered_affine_condition(linear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return recovered-map singular values and fail-closed spectral condition."""
    if linear.shape[-2:] != (2, 2) or not linear.dtype.is_floating_point:
        raise ValueError("recovered affine maps must be floating tensors ending in (2,2)")
    singular_values = torch.linalg.svdvals(linear)
    smallest = singular_values[..., -1]
    condition = torch.where(
        smallest > 0,
        singular_values[..., 0] / smallest,
        torch.full_like(smallest, torch.inf),
    )
    return singular_values, condition


def analyze_pair_tensors(
    *,
    pair: PairSpec,
    reference_means_native: torch.Tensor,
    reference_covariances_native: torch.Tensor,
    reference_foreground: torch.Tensor,
    target_mask_flow: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    reference_camera: Camera,
    target_camera: Camera,
    bounds_lower: torch.Tensor,
    bounds_upper: torch.Tensor,
    gates: GateConfig,
    fb_sensitivity_px: Sequence[float] = FB_SENSITIVITY_PX,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pure tensor post-flow analysis used by the production harness and CPU tests."""
    if pair.reference == HELDOUT_VIEW or pair.target == HELDOUT_VIEW:
        raise ValueError(f"held-out view {HELDOUT_VIEW} cannot be analyzed")
    count = reference_means_native.shape[0]
    if reference_means_native.shape != (count, 2) or not bool(
        torch.isfinite(reference_means_native).all()
    ):
        raise ValueError("reference means must have finite shape (N,2)")
    if reference_covariances_native.shape != (count, 2, 2) or not bool(
        torch.isfinite(reference_covariances_native).all()
    ):
        raise ValueError("reference covariances must have finite shape (N,2,2)")
    if reference_foreground.shape != (count,) or reference_foreground.dtype != torch.bool:
        raise ValueError("reference_foreground must contain one bool per component")
    if target_mask_flow.shape != FLOW_SHAPE or not bool(torch.isfinite(target_mask_flow).all()):
        raise ValueError(f"target mask must have finite shape {FLOW_SHAPE}")
    for name, flow in (("forward", forward_flow), ("backward", backward_flow)):
        if flow.shape != (FLOW_HEIGHT, FLOW_WIDTH, 2) or not bool(torch.isfinite(flow).all()):
            raise ValueError(f"{name} flow must have finite shape ({FLOW_HEIGHT},{FLOW_WIDTH},2)")
    if bounds_lower.shape != (3,) or bounds_upper.shape != (3,):
        raise ValueError("bounds must each have shape (3,)")
    if not bool(torch.isfinite(bounds_lower).all() & torch.isfinite(bounds_upper).all()):
        raise ValueError("bounds must be finite")
    if not bool((bounds_lower < bounds_upper).all()):
        raise ValueError("bounds must have positive extent")
    sensitivity_thresholds = tuple(float(value) for value in fb_sensitivity_px)
    if sensitivity_thresholds != FB_SENSITIVITY_PX:
        raise ValueError("FB sensitivity thresholds differ from the frozen protocol")
    reference_means_flow = native_to_flow_points(reference_means_native)
    reference_covariances_flow = native_to_flow_covariances(reference_covariances_native)
    primary = track_gaussians_2d(
        reference_means_flow,
        reference_covariances_flow,
        forward_flow,
        backward_flow,
        max_roundtrip_error=gates.fb_error_px,
        require_all_sigma_points=True,
    )
    target_means_native = flow_to_native_points(primary.means)
    target_covariances_native = flow_to_native_covariances(primary.covariances)
    target_flow_camera = flow_camera(target_camera)

    all_sigma_samples = primary.forward_samples.valid.all(dim=-1)
    affine_rank = primary.affine.rank == 2
    affine_source_design_condition = torch.isfinite(primary.affine.design_condition) & (
        primary.affine.design_condition <= gates.affine_source_design_max_condition
    )
    (
        recovered_singular_values,
        affine_recovered_condition_value,
    ) = recovered_affine_condition(
        primary.affine.linear,
    )
    affine_recovered_condition = torch.isfinite(affine_recovered_condition_value) & (
        affine_recovered_condition_value <= gates.affine_recovered_max_condition
    )
    affine_rms = primary.affine.rms_residual <= gates.affine_max_rms_px
    affine_det = (primary.affine.determinant >= gates.affine_min_determinant) & (
        primary.affine.determinant <= gates.affine_max_determinant
    )
    covariance_eigenvalues = torch.linalg.eigvalsh(target_covariances_native)
    covariance_spd = torch.isfinite(covariance_eigenvalues).all(dim=-1) & (
        covariance_eigenvalues[:, 0] > gates.covariance_min_eigenvalue_px2
    )
    target_in_frame = target_flow_camera.in_image(primary.means)
    target_mask_score = bilinear_sample(target_mask_flow, primary.means).clamp(0.0, 1.0)
    target_mask = target_in_frame & (target_mask_score >= gates.target_mask_threshold)
    epipolar = calibrated_epipolar_residual(
        reference_camera,
        target_camera,
        reference_means_native,
        target_means_native,
    )
    epipolar_gate = torch.isfinite(epipolar) & (epipolar <= gates.epipolar_max_error_native_px)
    fb_gate = primary.consistency.valid
    pretri_without_fb = (
        primary.sigma_points.valid
        & all_sigma_samples
        & primary.affine.valid
        & affine_rank
        & affine_source_design_condition
        & affine_recovered_condition
        & affine_rms
        & affine_det
        & covariance_spd
        & target_mask
        & epipolar_gate
    )
    pretri = pretri_without_fb & fb_gate

    points = torch.stack([reference_means_native, target_means_native], dim=1)
    # Triangulate every component that passes the non-FB gates once. This keeps the 1/3/5 px
    # sensitivity analysis independent instead of silently censoring the 5 px arm with the
    # primary 3 px observation mask.
    observation_mask = pretri_without_fb[:, None].expand(-1, 2)
    triangulation = triangulate_centers_dlt(
        [reference_camera, target_camera],
        points,
        observation_mask=observation_mask,
        max_condition_number=gates.dlt_max_condition,
        max_reprojection_error=gates.dlt_max_reprojection_native_px,
    )
    angles = ray_angle_degrees(
        reference_camera,
        target_camera,
        reference_means_native,
        target_means_native,
    )
    angle_gate = torch.isfinite(angles) & (angles >= gates.min_ray_angle_deg)
    # An infinite gap is the ideal exact-DLT case (zero smallest singular value), not a failure.
    nullspace_gate = ~torch.isnan(triangulation.nullspace_gap) & (
        triangulation.nullspace_gap >= gates.dlt_min_nullspace_gap
    )
    in_bounds = (
        (triangulation.points_world >= bounds_lower.to(triangulation.points_world))
        & (triangulation.points_world <= bounds_upper.to(triangulation.points_world))
    ).all(dim=-1)
    geometry_without_fb = (
        pretri_without_fb & triangulation.valid & angle_gate & nullspace_gate & in_bounds
    )
    accepted_geometry = geometry_without_fb & fb_gate
    accepted_foreground = accepted_geometry & reference_foreground
    dlt_all_cheiral = (triangulation.cheiral | ~triangulation.observation_valid).all(dim=-1)

    arrays_raw = {
        "component_id": torch.arange(count, dtype=torch.int64),
        "reference_mean_native": reference_means_native,
        "reference_mean_flow": reference_means_flow,
        "target_mean_native": target_means_native,
        "target_mean_flow": primary.means,
        "target_covariance_native": target_covariances_native,
        "reference_foreground": reference_foreground,
        "all_sigma_samples_valid": all_sigma_samples,
        "fb_error_flow_px": primary.consistency.roundtrip_error,
        "affine_rank": primary.affine.rank,
        "affine_source_design_condition": primary.affine.design_condition,
        "affine_recovered_singular_values": recovered_singular_values,
        "affine_recovered_condition": affine_recovered_condition_value,
        "affine_rms_flow_px": primary.affine.rms_residual,
        "affine_determinant": primary.affine.determinant,
        "target_covariance_min_eigenvalue_native_px2": covariance_eigenvalues[:, 0],
        "target_mask_score": target_mask_score,
        "epipolar_residual_native_px": epipolar,
        "point_world": triangulation.points_world,
        "dlt_depths": triangulation.depths,
        "dlt_observation_valid": triangulation.observation_valid,
        "dlt_observation_count": triangulation.observation_count,
        "dlt_cheiral": triangulation.cheiral,
        "dlt_rank": triangulation.rank,
        "dlt_singular_values": triangulation.singular_values,
        "dlt_algebraic_residual": triangulation.algebraic_residual,
        "dlt_homogeneous_w": triangulation.homogeneous_w,
        "dlt_reprojection_native_px": triangulation.reprojection_error,
        "dlt_max_reprojection_native_px": triangulation.max_observed_reprojection_error,
        "dlt_condition": triangulation.condition_number,
        "dlt_nullspace_gap": triangulation.nullspace_gap,
        "dlt_valid": triangulation.valid,
        "dlt_all_cheiral": dlt_all_cheiral,
        "ray_angle_deg": angles,
        "in_bounds": in_bounds,
        "pretri_valid": pretri,
        "accepted_geometry": accepted_geometry,
        "accepted_foreground": accepted_foreground,
    }
    arrays: dict[str, torch.Tensor] = {}
    encoded_float_arrays: list[str] = []
    for name, value in arrays_raw.items():
        if value.dtype.is_floating_point:
            finite, nonfinite_code = _encode_archive_float(value)
            arrays[name] = finite
            arrays[f"{name}{NONFINITE_CODE_SUFFIX}"] = nonfinite_code
            encoded_float_arrays.append(name)
        else:
            arrays[name] = value
    sensitivity = {}
    for threshold in sensitivity_thresholds:
        threshold_gate = torch.isfinite(primary.consistency.roundtrip_error) & (
            primary.consistency.roundtrip_error <= threshold
        )
        sensitivity[str(threshold)] = {
            "pretri_count": int((pretri_without_fb & threshold_gate).sum()),
            "accepted_geometry_count": int((geometry_without_fb & threshold_gate).sum()),
            "accepted_foreground_count": int(
                (geometry_without_fb & threshold_gate & reference_foreground).sum()
            ),
        }
    gate_masks = {
        "sigma_points_valid": primary.sigma_points.valid,
        "all_sigma_samples_valid": all_sigma_samples,
        "fb_primary": fb_gate,
        "affine_valid": primary.affine.valid,
        "affine_rank": affine_rank,
        "affine_source_design_condition": affine_source_design_condition,
        "affine_recovered_condition": affine_recovered_condition,
        "affine_rms": affine_rms,
        "affine_determinant": affine_det,
        "target_covariance_spd": covariance_spd,
        "target_mask": target_mask,
        "epipolar": epipolar_gate,
        "pretri": pretri,
        "dlt_valid": triangulation.valid,
        "dlt_all_cheiral": dlt_all_cheiral,
        "ray_angle": angle_gate,
        "nullspace_gap": nullspace_gate,
        "in_bounds": in_bounds,
        "accepted_geometry": accepted_geometry,
        "accepted_foreground": accepted_foreground,
    }
    sequential_order = (
        "sigma_points_valid",
        "all_sigma_samples_valid",
        "affine_valid",
        "affine_rank",
        "affine_source_design_condition",
        "affine_recovered_condition",
        "affine_rms",
        "affine_determinant",
        "target_covariance_spd",
        "target_mask",
        "epipolar",
        "fb_primary",
        "dlt_valid",
        "dlt_all_cheiral",
        "ray_angle",
        "nullspace_gap",
        "in_bounds",
        "accepted_foreground",
    )
    cumulative = torch.ones(count, dtype=torch.bool, device=reference_means_native.device)
    sequential_counts: dict[str, int] = {}
    sequential_foreground_counts: dict[str, int] = {}
    for name in sequential_order:
        cumulative = cumulative & gate_masks[name]
        sequential_counts[name] = int(cumulative.sum())
        sequential_foreground_counts[name] = int((cumulative & reference_foreground).sum())
    stage_masks = {
        "pretri": pretri,
        "dlt_valid": pretri & triangulation.valid,
        "accepted_geometry": accepted_geometry,
        "accepted_foreground": accepted_foreground,
    }
    metric_values = {
        "fb_error_flow_px": primary.consistency.roundtrip_error,
        "affine_source_design_condition": primary.affine.design_condition,
        "affine_recovered_condition": affine_recovered_condition_value,
        "affine_rms_flow_px": primary.affine.rms_residual,
        "affine_determinant": primary.affine.determinant,
        "epipolar_residual_native_px": epipolar,
        "dlt_reprojection_native_px": triangulation.max_observed_reprojection_error,
        "dlt_condition": triangulation.condition_number,
        "dlt_nullspace_gap": triangulation.nullspace_gap,
        "ray_angle_deg": angles,
    }
    distributions: dict[str, Any] = {}
    observation_count_gate = triangulation.observation_count >= 2
    for stage, stage_mask in stage_masks.items():
        distributions[stage] = {
            metric: _masked_distribution(
                values,
                stage_mask & observation_count_gate if metric.startswith("dlt_") else stage_mask,
            )
            for metric, values in metric_values.items()
        }
    summary = {
        "pair_id": pair.pair_id,
        "component_count": count,
        "foreground_component_count": int(reference_foreground.sum()),
        "counts": {
            "marginal": {name: int(mask.sum()) for name, mask in gate_masks.items()},
            "marginal_foreground": {
                name: int((mask & reference_foreground).sum()) for name, mask in gate_masks.items()
            },
            "sequential_order": list(sequential_order),
            "sequential": sequential_counts,
            "sequential_foreground": sequential_foreground_counts,
        },
        "fb_sensitivity": sensitivity,
        "stage_distributions": distributions,
        "dlt_distribution_policy": {
            "minimum_observation_count": 2,
            "applies_to_prefix": "dlt_",
        },
        "archive_nonfinite_policy": {
            "encoding": "uint8_sidecar",
            "code_semantics": {
                str(code): meaning for code, meaning in NONFINITE_CODE_SEMANTICS.items()
            },
            "sidecar_suffix": NONFINITE_CODE_SUFFIX,
            "replacement_value": 0.0,
            "encoded_float_arrays": encoded_float_arrays,
            "replay": (
                "restore NaN/+Inf/-Inf from each sidecar before evaluating gates or "
                "finite-only quantiles"
            ),
        },
    }
    return arrays, summary


def _correspondence_base_schema(
    count: int = REFERENCE_COMPONENT_COUNT,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    float64 = np.dtype(np.float64)
    boolean = np.dtype(np.bool_)
    int64 = np.dtype(np.int64)
    return {
        "component_id": ((count,), int64),
        "reference_mean_native": ((count, 2), float64),
        "reference_mean_flow": ((count, 2), float64),
        "target_mean_native": ((count, 2), float64),
        "target_mean_flow": ((count, 2), float64),
        "target_covariance_native": ((count, 2, 2), float64),
        "reference_foreground": ((count,), boolean),
        "all_sigma_samples_valid": ((count,), boolean),
        "fb_error_flow_px": ((count,), float64),
        "affine_rank": ((count,), int64),
        "affine_source_design_condition": ((count,), float64),
        "affine_recovered_singular_values": ((count, 2), float64),
        "affine_recovered_condition": ((count,), float64),
        "affine_rms_flow_px": ((count,), float64),
        "affine_determinant": ((count,), float64),
        "target_covariance_min_eigenvalue_native_px2": ((count,), float64),
        "target_mask_score": ((count,), float64),
        "epipolar_residual_native_px": ((count,), float64),
        "point_world": ((count, 3), float64),
        "dlt_depths": ((count, 2), float64),
        "dlt_observation_valid": ((count, 2), boolean),
        "dlt_observation_count": ((count,), int64),
        "dlt_cheiral": ((count, 2), boolean),
        "dlt_rank": ((count,), int64),
        "dlt_singular_values": ((count, 4), float64),
        "dlt_algebraic_residual": ((count,), float64),
        "dlt_homogeneous_w": ((count,), float64),
        "dlt_reprojection_native_px": ((count, 2), float64),
        "dlt_max_reprojection_native_px": ((count,), float64),
        "dlt_condition": ((count,), float64),
        "dlt_nullspace_gap": ((count,), float64),
        "dlt_valid": ((count,), boolean),
        "dlt_all_cheiral": ((count,), boolean),
        "ray_angle_deg": ((count,), float64),
        "in_bounds": ((count,), boolean),
        "pretri_valid": ((count,), boolean),
        "accepted_geometry": ((count,), boolean),
        "accepted_foreground": ((count,), boolean),
    }


def _correspondence_schema(
    count: int = REFERENCE_COMPONENT_COUNT,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    schema = _correspondence_base_schema(count)
    uint8 = np.dtype(np.uint8)
    schema.update(
        {
            f"{name}{NONFINITE_CODE_SUFFIX}": (shape, uint8)
            for name, (shape, dtype) in tuple(schema.items())
            if np.issubdtype(dtype, np.floating)
        }
    )
    return schema


def _encoded_correspondence_float_names() -> list[str]:
    return [
        name
        for name, (_, dtype) in _correspondence_base_schema().items()
        if np.issubdtype(dtype, np.floating)
    ]


def validate_correspondence_npz(path: Path) -> tuple[str, ...]:
    names = audit_npz_no_source_rgb(path)
    schema = _correspondence_schema()
    if set(names) != set(schema) or len(names) != len(schema):
        raise RuntimeError(f"correspondence NPZ key set mismatch: {names}")
    with np.load(path, allow_pickle=False) as archive:
        for name, (shape, dtype) in schema.items():
            _validate_np_array(
                archive[name],
                name=f"{path.name}:{name}",
                shape=shape,
                dtype=dtype,
            )
        expected_ids = np.arange(REFERENCE_COMPONENT_COUNT, dtype=np.int64)
        if not np.array_equal(archive["component_id"], expected_ids):
            raise RuntimeError("correspondence component IDs are not the exact frozen lineage")
        for name in _encoded_correspondence_float_names():
            value = archive[name]
            code = archive[f"{name}{NONFINITE_CODE_SUFFIX}"]
            if not bool(np.isin(code, tuple(NONFINITE_CODE_SEMANTICS)).all()):
                raise RuntimeError(f"{name} nonfinite sidecar contains an unknown code")
            if not bool((value[code != 0] == 0.0).all()):
                raise RuntimeError(f"{name} nonfinite sidecar does not point to replacement zeros")
    return names


def reproduce_stage_distributions(path: Path) -> dict[str, Any]:
    """Recompute every reported stage distribution solely from one correspondence archive."""
    validate_correspondence_npz(path)
    metric_arrays = {
        "fb_error_flow_px": "fb_error_flow_px",
        "affine_source_design_condition": "affine_source_design_condition",
        "affine_recovered_condition": "affine_recovered_condition",
        "affine_rms_flow_px": "affine_rms_flow_px",
        "affine_determinant": "affine_determinant",
        "epipolar_residual_native_px": "epipolar_residual_native_px",
        "dlt_reprojection_native_px": "dlt_max_reprojection_native_px",
        "dlt_condition": "dlt_condition",
        "dlt_nullspace_gap": "dlt_nullspace_gap",
        "ray_angle_deg": "ray_angle_deg",
    }
    with np.load(path, allow_pickle=False) as archive:
        values = {
            metric: restore_archived_nonfinite(
                torch.from_numpy(np.asarray(archive[name]).copy()),
                torch.from_numpy(np.asarray(archive[f"{name}{NONFINITE_CODE_SUFFIX}"]).copy()),
            )
            for metric, name in metric_arrays.items()
        }
        pretri = torch.from_numpy(np.asarray(archive["pretri_valid"]).copy())
        stage_masks = {
            "pretri": pretri,
            "dlt_valid": pretri & torch.from_numpy(np.asarray(archive["dlt_valid"]).copy()),
            "accepted_geometry": torch.from_numpy(np.asarray(archive["accepted_geometry"]).copy()),
            "accepted_foreground": torch.from_numpy(
                np.asarray(archive["accepted_foreground"]).copy()
            ),
        }
        observation_count_gate = (
            torch.from_numpy(np.asarray(archive["dlt_observation_count"]).copy()) >= 2
        )
    return {
        stage: {
            metric: _masked_distribution(
                value,
                stage_mask & observation_count_gate if metric.startswith("dlt_") else stage_mask,
            )
            for metric, value in values.items()
        }
        for stage, stage_mask in stage_masks.items()
    }


def verify_correspondence_evidence(
    path: Path,
    summary: Mapping[str, Any],
    *,
    gates: GateConfig,
) -> None:
    """Bind reported DLT gates/distributions to replayed archive evidence."""
    reproduced = reproduce_stage_distributions(path)
    if reproduced != summary["stage_distributions"]:
        raise RuntimeError("reported stage distributions do not replay from correspondence NPZ")
    with np.load(path, allow_pickle=False) as archive:
        foreground = torch.from_numpy(np.asarray(archive["reference_foreground"]).copy())
        archived_dlt_valid = torch.from_numpy(np.asarray(archive["dlt_valid"]).copy())
        archived_dlt_all_cheiral = torch.from_numpy(np.asarray(archive["dlt_all_cheiral"]).copy())
        observation_valid = torch.from_numpy(np.asarray(archive["dlt_observation_valid"]).copy())
        observation_count = torch.from_numpy(np.asarray(archive["dlt_observation_count"]).copy())
        cheiral = torch.from_numpy(np.asarray(archive["dlt_cheiral"]).copy())
        rank = torch.from_numpy(np.asarray(archive["dlt_rank"]).copy())
        point_world = restore_archived_nonfinite(
            torch.from_numpy(np.asarray(archive["point_world"]).copy()),
            torch.from_numpy(np.asarray(archive[f"point_world{NONFINITE_CODE_SUFFIX}"]).copy()),
        )
        homogeneous_w = restore_archived_nonfinite(
            torch.from_numpy(np.asarray(archive["dlt_homogeneous_w"]).copy()),
            torch.from_numpy(
                np.asarray(archive[f"dlt_homogeneous_w{NONFINITE_CODE_SUFFIX}"]).copy()
            ),
        )
        condition = restore_archived_nonfinite(
            torch.from_numpy(np.asarray(archive["dlt_condition"]).copy()),
            torch.from_numpy(np.asarray(archive[f"dlt_condition{NONFINITE_CODE_SUFFIX}"]).copy()),
        )
        max_reprojection = restore_archived_nonfinite(
            torch.from_numpy(np.asarray(archive["dlt_max_reprojection_native_px"]).copy()),
            torch.from_numpy(
                np.asarray(archive[f"dlt_max_reprojection_native_px{NONFINITE_CODE_SUFFIX}"]).copy()
            ),
        )
        nullspace_gap = restore_archived_nonfinite(
            torch.from_numpy(np.asarray(archive["dlt_nullspace_gap"]).copy()),
            torch.from_numpy(
                np.asarray(archive[f"dlt_nullspace_gap{NONFINITE_CODE_SUFFIX}"]).copy()
            ),
        )
    dlt_all_cheiral = (cheiral | ~observation_valid).all(dim=-1)
    if not torch.equal(observation_valid.sum(dim=-1), observation_count):
        raise RuntimeError("archived DLT observation counts do not match validity masks")
    homogeneous_epsilon = 64.0 * torch.finfo(homogeneous_w.dtype).eps
    dlt_valid = (
        (observation_count >= 2)
        & (rank >= 3)
        & (homogeneous_w.abs() > homogeneous_epsilon)
        & torch.isfinite(point_world).all(dim=-1)
        & torch.isfinite(condition)
        & (condition <= gates.dlt_max_condition)
        & dlt_all_cheiral
        & (max_reprojection <= gates.dlt_max_reprojection_native_px)
    )
    if not torch.equal(dlt_valid, archived_dlt_valid):
        raise RuntimeError("archived DLT-valid mask does not replay from preserved diagnostics")
    if not torch.equal(dlt_all_cheiral, archived_dlt_all_cheiral):
        raise RuntimeError("archived DLT-cheirality mask does not replay from observations")
    masks = {
        "dlt_valid": dlt_valid,
        "dlt_all_cheiral": dlt_all_cheiral,
        # +Inf is deliberately accepted; only NaN and values below the threshold fail.
        "nullspace_gap": ~torch.isnan(nullspace_gap)
        & (nullspace_gap >= gates.dlt_min_nullspace_gap),
    }
    for name, mask in masks.items():
        if (
            int(mask.sum()) != summary["counts"]["marginal"][name]
            or int((mask & foreground).sum()) != summary["counts"]["marginal_foreground"][name]
        ):
            raise RuntimeError(f"reported {name} gate counts do not replay from correspondence NPZ")


def _validate_camera_payload(record: Mapping[str, Any], *, context: str) -> None:
    _require_exact_keys(record, {"fx", "fy", "cx", "cy", "width", "height", "R", "t"}, context)
    scalars = [record[key] for key in ("fx", "fy", "cx", "cy")]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in scalars
    ):
        raise RuntimeError(f"{context} contains invalid intrinsics")
    if record["width"] != NATIVE_WIDTH or record["height"] != NATIVE_HEIGHT:
        raise RuntimeError(f"{context} native dimensions changed")
    if len(record["R"]) != 9 or len(record["t"]) != 3:
        raise RuntimeError(f"{context} extrinsic shapes changed")
    if not all(math.isfinite(float(value)) for value in (*record["R"], *record["t"])):
        raise RuntimeError(f"{context} contains non-finite extrinsics")


def _verify_mask_and_bounds_replay(
    *,
    plan: Mapping[str, Any],
    acquisition: Path,
    manifest: Mapping[str, Any],
    bounds: Mapping[str, Any],
) -> None:
    """Replay undistortion, mask hashes, compact masks, and the exact object AABB."""
    from rtgs.data.calibrated import _object_bounds

    inputs = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
    if tuple(inputs.view_names) != BOUNDS_VIEWS:
        raise RuntimeError("bounds replay teacher/camera order changed")
    records, _ = _calibration_records()
    masks: list[torch.Tensor] = []
    by_view: dict[str, torch.Tensor] = {}
    for view, camera in zip(inputs.view_names, inputs.cameras, strict=True):
        planned = plan["scientific_inputs"]["source_masks"][view]
        mask_path = _verify_file_binding(planned, context=f"bounds replay mask {view}")
        recorded = bounds["masks"][view]
        if sha256_file(mask_path) != recorded["source_sha256"]:
            raise RuntimeError(f"bounds replay source-mask digest changed for {view}")
        mask = _load_undistorted_mask(view, camera, records[view])
        if mask.shape != (NATIVE_HEIGHT, NATIVE_WIDTH) or not bool(torch.isfinite(mask).all()):
            raise RuntimeError(f"bounds replay mask is malformed for {view}")
        if tensor_sha256(mask) != recorded["undistorted_tensor_sha256"]:
            raise RuntimeError(f"undistorted mask tensor digest does not replay for {view}")
        foreground_fraction = float((mask > 0.5).float().mean())
        if foreground_fraction != float(recorded["foreground_fraction"]):
            raise RuntimeError(f"undistorted mask foreground fraction changed for {view}")
        masks.append(mask)
        by_view[view] = mask

    for target in TARGET_VIEWS:
        if target in by_view:
            continue
        # Verification must never inspect JPEG metadata.  Native capture dimensions are frozen
        # by the preregistration and the camera comes solely from hash-bound calibration.
        camera = _camera_for_dimensions(
            records[target],
            width=NATIVE_WIDTH,
            height=NATIVE_HEIGHT,
        )
        by_view[target] = _load_undistorted_mask(target, camera, records[target])

    replay_center, replay_extent = _object_bounds(inputs.cameras, masks)
    replay_lower = replay_center - float(replay_extent) * BOUNDS_SCALE
    replay_upper = replay_center + float(replay_extent) * BOUNDS_SCALE
    recorded_center = torch.tensor(bounds["center"], dtype=replay_center.dtype)
    recorded_lower = torch.tensor(bounds["aabb_lower"], dtype=replay_lower.dtype)
    recorded_upper = torch.tensor(bounds["aabb_upper"], dtype=replay_upper.dtype)
    if (
        not torch.equal(recorded_center, replay_center.cpu())
        or float(bounds["extent"]) != float(replay_extent)
        or not torch.equal(recorded_lower, replay_lower.cpu())
        or not torch.equal(recorded_upper, replay_upper.cpu())
        or bounds["source"] != "seven_training_masks"
    ):
        raise RuntimeError("mask-derived center/extent/AABB does not replay exactly")

    compact_path = acquisition / manifest["compact_analysis_input"]["path"]
    with np.load(compact_path, allow_pickle=False) as archive:
        reference_index = inputs.view_names.index(REFERENCE_VIEW)
        reference_field = inputs.observations[reference_index]
        exact_teacher_arrays = {
            "reference_means_native": observation_native_means(
                reference_field,
                dtype=torch.float64,
            ).numpy(),
            "reference_covariances_native": observation_covariances(
                reference_field,
                dtype=torch.float64,
            ).numpy(),
            "reference_colors": reference_field.colors.to(dtype=torch.float32).numpy(),
            "reference_amplitudes": reference_field.amplitudes.to(dtype=torch.float32).numpy(),
        }
        for name, expected in exact_teacher_arrays.items():
            if not np.array_equal(np.asarray(archive[name]), expected):
                raise RuntimeError(
                    f"compact {name} does not replay exactly from the strict hash-bound "
                    f"{REFERENCE_VIEW} teacher"
                )
        for target in TARGET_VIEWS:
            expected = crop_area_resize(by_view[target]).numpy()
            observed = np.asarray(archive[f"target_mask_flow_{target}"])
            if not np.array_equal(observed, expected):
                raise RuntimeError(f"compact target mask does not replay for {target}")
        reference_means = observation_native_means(
            reference_field,
            dtype=torch.float64,
        )
        expected_foreground = (
            bilinear_sample(
                by_view[REFERENCE_VIEW],
                reference_means,
            )
            > 0.5
        )
        observed_foreground = torch.from_numpy(np.asarray(archive["reference_foreground"]).copy())
        if not torch.equal(observed_foreground, expected_foreground):
            raise RuntimeError("reference foreground lineage does not replay from undistorted mask")


def _verify_acquisition(
    output: Path,
    *,
    verify_source_color_bytes: bool = True,
) -> dict[str, Any]:
    acquisition = output / "acquisition"
    plan = _load_and_verify_plan(
        output,
        verify_source_color_bytes=verify_source_color_bytes,
    )
    plan_path = output / "plan.json"
    manifest_path = acquisition / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("completed acquisition manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_acquisition_manifest_schema(manifest)
    if manifest["plan"]["file_sha256"] != sha256_file(plan_path):
        raise RuntimeError("acquisition does not bind the current exact plan file")
    if manifest["plan"]["payload_sha256"] != canonical_hash(plan):
        raise RuntimeError("acquisition plan payload hash-chain mismatch")
    if manifest["plan"]["created_utc"] != plan["created_utc"]:
        raise RuntimeError("acquisition plan timestamp binding changed")
    if manifest["crop"] != plan["crop"]:
        raise RuntimeError("acquisition crop record differs from the exact plan")
    current_hashes = _current_source_hashes(
        plan,
        verify_source_color_bytes=verify_source_color_bytes,
    )
    if manifest["source_verification"]["hashes"] != current_hashes:
        raise RuntimeError("acquisition source verification differs from current exact sources")
    if (
        manifest["teacher_bundle_manifest_sha256"] != plan["teacher_bundle"]["manifest"]["sha256"]
        or manifest["reference_teacher"]["archive_sha256"]
        != plan["teacher_bundle"]["reference_teacher"]["sha256"]
    ):
        raise RuntimeError("acquisition teacher binding differs from the exact plan")
    for name, record in manifest["backend_replay"]["implementation_sources"].items():
        replay_path = acquisition / record["path"]
        if (
            not replay_path.is_file()
            or replay_path.stat().st_size != record["bytes"]
            or sha256_file(replay_path) != record["sha256"]
        ):
            raise RuntimeError(f"acquisition backend replay binding changed for {name}")
        if record["sha256"] != manifest["backend"]["implementation_sources"][name]["sha256"]:
            raise RuntimeError(f"acquisition runtime-source replay differs for {name}")
    weight_replay = manifest["backend_replay"]["weights"]
    weight_replay_path = acquisition / weight_replay["path"]
    if (
        not weight_replay_path.is_file()
        or weight_replay_path.stat().st_size != weight_replay["bytes"]
        or sha256_file(weight_replay_path) != weight_replay["sha256"]
        or weight_replay["sha256"] != manifest["backend"]["weights_sha256"]
    ):
        raise RuntimeError("acquisition backend replay differs from executed backend")
    compact = manifest["compact_analysis_input"]
    if compact["path"] != "analysis_input.npz":
        raise RuntimeError("analysis-input path changed")
    compact_path = acquisition / compact["path"]
    if sha256_file(compact_path) != compact["sha256"]:
        raise RuntimeError("compact post-flow input digest mismatch")
    compact_keys = validate_analysis_input_npz(compact_path)
    if list(compact_keys) != compact["keys"]:
        raise RuntimeError("analysis-input key declaration/order differs from archive")
    for pair in PAIRS:
        record = manifest["pairs"][pair.pair_id]
        if record["flow_path"] != f"flows/{pair.pair_id}.npz":
            raise RuntimeError(f"flow path changed for {pair.pair_id}")
        flow_path = acquisition / record["flow_path"]
        if sha256_file(flow_path) != record["flow_sha256"]:
            raise RuntimeError(f"flow digest mismatch for {pair.pair_id}")
        flow_keys = validate_flow_npz(flow_path)
        if list(flow_keys) != record["flow_keys"]:
            raise RuntimeError(f"flow key declaration changed for {pair.pair_id}")
        visuals = record["flow_visuals"]
        for direction in ("forward", "backward"):
            visual_path = acquisition / visuals[f"{direction}_path"]
            if sha256_file(visual_path) != visuals[f"{direction}_sha256"]:
                raise RuntimeError(f"{direction} flow visual digest mismatch for {pair.pair_id}")
    for key in ("calibration", "bounds"):
        record = manifest[key]
        if sha256_file(acquisition / record["path"]) != record["sha256"]:
            raise RuntimeError(f"{key} digest mismatch")
    calibration = json.loads((acquisition / manifest["calibration"]["path"]).read_text())
    if (
        manifest["calibration"]["source_sha256"]
        != plan["scientific_inputs"]["calibration"]["sha256"]
    ):
        raise RuntimeError("acquisition calibration source differs from plan")
    _require_exact_keys(calibration, {"native", "flow"}, "calibration payload")
    expected_views = {REFERENCE_VIEW, *TARGET_VIEWS}
    if set(calibration["native"]) != expected_views or set(calibration["flow"]) != expected_views:
        raise RuntimeError("calibration view key set changed")
    for view, record in calibration["native"].items():
        _validate_camera_payload(record, context=f"calibration.native.{view}")
    source_calibration = json.loads(
        _verify_file_binding(
            plan["scientific_inputs"]["calibration"],
            context="plan calibration",
        ).read_text(encoding="utf-8")
    )
    source_records = {
        str(record["camera_id"]).upper(): record for record in source_calibration["cameras"]
    }
    for view, record in calibration["native"].items():
        expected = _camera_for_dimensions(
            source_records[view],
            width=NATIVE_WIDTH,
            height=NATIVE_HEIGHT,
        )
        if _camera_difference(camera_from_record(record), expected) > 1.0e-4:
            raise RuntimeError(f"calibration.native.{view} differs from bound source calibration")
    for view, record in calibration["flow"].items():
        _require_exact_keys(
            record,
            {"fx", "fy", "cx", "cy", "width", "height", "R", "t"},
            f"calibration.flow.{view}",
        )
        if record["width"] != FLOW_WIDTH or record["height"] != FLOW_HEIGHT:
            raise RuntimeError(f"calibration.flow.{view} dimensions changed")
        numeric = [record[key] for key in ("fx", "fy", "cx", "cy", "width", "height")]
        if (
            len(record["R"]) != 9
            or len(record["t"]) != 3
            or not all(
                math.isfinite(float(value)) for value in [*numeric, *record["R"], *record["t"]]
            )
        ):
            raise RuntimeError(f"calibration.flow.{view} contains non-finite values")
    _assert_heldout_only_in_declaration(
        {"heldout": manifest["heldout"], "calibration": calibration},
        context="acquisition calibration",
    )
    bounds = json.loads((acquisition / manifest["bounds"]["path"]).read_text())
    _require_exact_keys(
        bounds,
        {
            "source",
            "views",
            "center",
            "extent",
            "bounds_scale",
            "aabb_lower",
            "aabb_upper",
            "masks",
        },
        "bounds payload",
    )
    if tuple(bounds["views"]) != BOUNDS_VIEWS or set(bounds["masks"]) != set(BOUNDS_VIEWS):
        raise RuntimeError("bounds view set changed")
    if (
        float(bounds["bounds_scale"]) != float(plan["bounds"]["bounds_scale"])
        or float(bounds["bounds_scale"]) != BOUNDS_SCALE
    ):
        raise RuntimeError("bounds scale differs from plan")
    bound_numbers = [
        *bounds["center"],
        bounds["extent"],
        bounds["bounds_scale"],
        *bounds["aabb_lower"],
        *bounds["aabb_upper"],
    ]
    if not all(math.isfinite(float(value)) for value in bound_numbers):
        raise RuntimeError("bounds payload contains non-finite values")
    if (
        len(bounds["center"]) != 3
        or len(bounds["aabb_lower"]) != 3
        or len(bounds["aabb_upper"]) != 3
    ):
        raise RuntimeError("bounds vector shapes changed")
    for view, record in bounds["masks"].items():
        _require_exact_keys(
            record,
            {
                "path",
                "source_sha256",
                "undistorted_tensor_sha256",
                "foreground_fraction",
            },
            f"bounds.masks.{view}",
        )
        planned_mask = plan["scientific_inputs"]["source_masks"][view]
        if (
            record["source_sha256"] != planned_mask["sha256"]
            or record["path"] != planned_mask["path"]
        ):
            raise RuntimeError(f"bounds mask source differs from plan for {view}")
        _require_sha256(
            record["undistorted_tensor_sha256"],
            context=f"bounds.masks.{view}.undistorted_tensor_sha256",
        )
    _verify_mask_and_bounds_replay(
        plan=plan,
        acquisition=acquisition,
        manifest=manifest,
        bounds=bounds,
    )
    for view, binding in manifest["source_bindings"].items():
        plan_color = plan["scientific_inputs"]["source_colors"][view]
        plan_mask = plan["scientific_inputs"]["source_masks"][view]
        if (
            binding["source_color_sha256"] != plan_color["sha256"]
            or binding["mask_sha256"] != plan_mask["sha256"]
        ):
            raise RuntimeError(f"source binding differs from plan for {view}")
    return manifest


def _mechanism_gaussians(
    points: torch.Tensor,
    colors: torch.Tensor,
    *,
    extent: float,
) -> Gaussians3D:
    count = points.shape[0]
    scale = max(float(extent) / 400.0, 1.0e-4)
    covariances = torch.eye(3, dtype=points.dtype)[None].expand(count, -1, -1) * (scale**2)
    return Gaussians3D.from_means_covs(
        points,
        covariances,
        colors.clamp(0.0, 1.0),
        torch.full((count,), 0.65, dtype=points.dtype),
        sh_degree=0,
    )


def _save_mechanism_ply(path: Path, gaussians: Gaussians3D) -> None:
    """Save a standard PLY, including the valid zero-vertex mechanism-failure case."""
    if gaussians.n:
        gaussians.save_ply(path)
        return
    properties = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 0"]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    with path.open("xb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))


def validate_viewer_allowlist(view_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(view).upper() for view in view_ids)
    if normalized != VIEWER_VIEWS:
        raise ValueError(f"viewer view allowlist is frozen to {VIEWER_VIEWS}, got {normalized}")
    if len(set(normalized)) != len(normalized) or HELDOUT_VIEW in normalized:
        raise ValueError("viewer allowlist must be unique and exclude the held-out view")
    return normalized


def viewer_command(
    output: Path,
    *,
    port: int = 8890,
    handshake_token: str | None = None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "benchmarks/g2sr_correspondence_diagnostic.py",
        "serve-viewer",
        "--out",
        display_path(output),
        "--view-ids",
        ",".join(VIEWER_VIEWS),
        "--port",
        str(port),
    ]
    if handshake_token is not None:
        command.extend(["--handshake-token", handshake_token])
    return command


def viewer_shell(command: Sequence[str]) -> str:
    return f"LD_PRELOAD={shlex.quote(str(LOCAL_LIBSTDCXX_PRELOAD))} " + shlex.join(
        [str(value) for value in command]
    )


def _viewer_source_bindings() -> dict[str, dict[str, Any]]:
    return {
        view: _file_binding(SCENE / "rgb" / f"{view}.jpg")
        for view in validate_viewer_allowlist(VIEWER_VIEWS)
    }


def _load_reporting_viewer_scene(output: Path, view_ids: Sequence[str]) -> Any:
    """Decode only the explicit reporting allowlist; never enumerate/load C1004."""
    from rtgs.data.scene import SceneData

    allowed = validate_viewer_allowlist(view_ids)
    _verify_analysis(output)
    records, _ = _calibration_records()
    images: list[torch.Tensor] = []
    cameras: list[Camera] = []
    for view in allowed:
        camera = _camera_for_native_view(view, records[view])
        if (camera.width, camera.height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
            raise RuntimeError(f"viewer source {view} is not native resolution")
        images.append(_load_undistorted_rgb(view, camera, records[view]))
        cameras.append(camera)
    manifest = json.loads((output / "acquisition" / "manifest.json").read_text(encoding="utf-8"))
    bounds = json.loads(
        (output / "acquisition" / manifest["bounds"]["path"]).read_text(encoding="utf-8")
    )
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=list(allowed),
        train_indices=list(range(len(allowed))),
        test_indices=[],
        bounds_hint=(
            torch.tensor(bounds["center"], dtype=torch.float32),
            float(bounds["extent"]),
        ),
        name=f"{SCENE.name}-g2sr-training-viewer",
    )
    scene.validate()
    return scene


def _viewer_handshake_marker(token: str, *, pid: int, port: int) -> str:
    return f"G2SR_VIEWER_HANDSHAKE token={token} pid={pid} port={port}"


def _assert_port_available(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("viewer port must lie in [1,65535]")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"viewer port {port} is already occupied; refusing a non-process-specific smoke"
            ) from exc


def _process_owns_listening_port(pid: int, port: int) -> bool:
    """Return whether ``pid`` owns the Linux listening socket for ``127.0.0.1:port``."""
    if pid <= 0 or port <= 0:
        return False
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    if not inodes:
        return False
    descriptors = Path(f"/proc/{pid}/fd")
    try:
        for descriptor in descriptors.iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return True
    except OSError:
        return False
    return False


def serve_viewer(
    output: Path,
    *,
    view_ids: Sequence[str],
    port: int,
    handshake_token: str | None = None,
) -> Path:
    """Launch the gsplat viewer with only explicitly allowlisted training images."""
    from rtgs.viewer import launch_viewer

    allowed = validate_viewer_allowlist(view_ids)
    _assert_port_available(port)
    result = _verify_analysis(output)
    ply_path = output / "analysis" / result["combined_pair_samples"]["path"]
    gaussians = Gaussians3D.load_ply(ply_path)
    scene = _load_reporting_viewer_scene(output, allowed)
    if handshake_token is not None:
        if len(handshake_token) != 64 or any(
            character not in "0123456789abcdef" for character in handshake_token
        ):
            raise ValueError("viewer handshake token must be 32 random bytes in lowercase hex")
        print(
            _viewer_handshake_marker(handshake_token, pid=os.getpid(), port=port),
            flush=True,
        )
    launch_viewer(
        {"final": gaussians},
        scene=scene,
        device="cuda",
        snapshot_rasterizer="gsplat",
        snapshot_packed=False,
        snapshot_antialiased=False,
        snapshot_dir=None,
        host="127.0.0.1",
        port=port,
        open_browser=False,
    )
    return ply_path


PAIR_DISPLAY_COLORS = {
    "C0014_to_C0005": torch.tensor([1.0, 0.32, 0.08]),
    "C0014_to_C0026": torch.tensor([0.08, 0.62, 1.0]),
}


def _validate_cross_pair_npz(path: Path, overlap_count: int) -> None:
    expected = {
        "component_id": ((overlap_count,), np.dtype(np.int64)),
        "point_C0014_to_C0005": ((overlap_count, 3), np.dtype(np.float64)),
        "point_C0014_to_C0026": ((overlap_count, 3), np.dtype(np.float64)),
        "distance_world": ((overlap_count,), np.dtype(np.float64)),
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected) or len(archive.files) != len(expected):
            raise RuntimeError("cross-pair disagreement NPZ key set mismatch")
        for name, (shape, dtype) in expected.items():
            _validate_np_array(
                archive[name],
                name=f"cross-pair:{name}",
                shape=shape,
                dtype=dtype,
            )


def cross_pair_disagreement(
    first_ids: torch.Tensor,
    first_points: torch.Tensor,
    second_ids: torch.Tensor,
    second_points: torch.Tensor,
) -> dict[str, Any]:
    """Match unique component lineages across two pair-sample sets."""
    for name, ids, points in (
        ("first", first_ids, first_points),
        ("second", second_ids, second_points),
    ):
        if ids.ndim != 1 or ids.dtype != torch.int64:
            raise ValueError(f"{name} IDs must be an int64 vector")
        if points.shape != (ids.numel(), 3) or not points.dtype.is_floating_point:
            raise ValueError(f"{name} points must have floating shape (N,3)")
        if not bool(torch.isfinite(points).all()):
            raise ValueError(f"{name} points must be finite")
        if len(set(ids.tolist())) != ids.numel():
            raise ValueError(f"{name} pair has duplicate component lineages")
    first_map = {int(component): index for index, component in enumerate(first_ids.tolist())}
    second_map = {int(component): index for index, component in enumerate(second_ids.tolist())}
    overlap_ids = sorted(set(first_map) & set(second_map))
    union_ids = sorted(set(first_map) | set(second_map))
    overlap_tensor = torch.tensor(overlap_ids, dtype=torch.int64)
    if overlap_ids:
        first_overlap = torch.stack(
            [first_points[first_map[component]] for component in overlap_ids]
        )
        second_overlap = torch.stack(
            [second_points[second_map[component]] for component in overlap_ids]
        )
    else:
        dtype = torch.promote_types(first_points.dtype, second_points.dtype)
        first_overlap = torch.empty((0, 3), dtype=dtype)
        second_overlap = torch.empty((0, 3), dtype=dtype)
    return {
        "component_id": overlap_tensor,
        "first_points": first_overlap,
        "second_points": second_overlap,
        "distance_world": (first_overlap - second_overlap).norm(dim=-1),
        "overlap_unique_component_count": len(overlap_ids),
        "union_unique_component_count": len(union_ids),
    }


def _verify_exact_mechanism_ply(
    path: Path,
    *,
    points: torch.Tensor,
    colors: torch.Tensor,
    extent: float,
    context: str,
) -> None:
    """Regenerate the claimed PLY byte-for-byte from replayed accepted evidence."""
    expected = _mechanism_gaussians(points, colors, extent=extent)
    with tempfile.TemporaryDirectory(prefix=".g2sr-ply-replay-", dir=path.parent) as temporary:
        replay = Path(temporary) / path.name
        _save_mechanism_ply(replay, expected)
        if sha256_file(replay) != sha256_file(path) or replay.read_bytes() != path.read_bytes():
            raise RuntimeError(f"{context} PLY does not replay from accepted correspondence rows")


def _load_replay_inputs(
    output: Path,
    manifest: Mapping[str, Any],
) -> tuple[
    dict[str, Camera],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    acquisition = output / "acquisition"
    calibration = json.loads(
        (acquisition / manifest["calibration"]["path"]).read_text(encoding="utf-8")
    )
    bounds = json.loads((acquisition / manifest["bounds"]["path"]).read_text(encoding="utf-8"))
    with np.load(
        acquisition / manifest["compact_analysis_input"]["path"],
        allow_pickle=False,
    ) as archive:
        reference_means = torch.from_numpy(np.asarray(archive["reference_means_native"]).copy())
        reference_covariances = torch.from_numpy(
            np.asarray(archive["reference_covariances_native"]).copy()
        )
        reference_foreground = torch.from_numpy(np.asarray(archive["reference_foreground"]).copy())
        target_masks = {
            target: torch.from_numpy(np.asarray(archive[f"target_mask_flow_{target}"]).copy())
            for target in TARGET_VIEWS
        }
    native_cameras = {
        view: camera_from_record(record) for view, record in calibration["native"].items()
    }
    return (
        native_cameras,
        target_masks,
        reference_means,
        reference_covariances,
        reference_foreground,
        torch.stack(
            [
                torch.tensor(bounds["aabb_lower"], dtype=torch.float64),
                torch.tensor(bounds["aabb_upper"], dtype=torch.float64),
            ]
        ),
    )


def _verify_full_analysis_replay(
    output: Path,
    *,
    analysis: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    """Replay every gate and every output claim from flow/mask/calibration evidence."""
    (
        native_cameras,
        target_masks,
        reference_means,
        reference_covariances,
        reference_foreground,
        bounds,
    ) = _load_replay_inputs(output, manifest)
    lower, upper = bounds
    gates = GateConfig(**plan["gates"])
    sensitivity = tuple(float(value) for value in plan["fb_sensitivity_px"])
    accepted_by_pair: dict[str, dict[str, torch.Tensor]] = {}
    combined_points: list[torch.Tensor] = []
    combined_colors: list[torch.Tensor] = []
    expected_lineage: list[dict[str, Any]] = []

    for pair in PAIRS:
        flow_record = manifest["pairs"][pair.pair_id]
        with np.load(
            output / "acquisition" / flow_record["flow_path"],
            allow_pickle=False,
        ) as flow:
            forward = torch.from_numpy(np.asarray(flow["forward"]).copy())
            backward = torch.from_numpy(np.asarray(flow["backward"]).copy())
        replay_arrays, replay_summary = analyze_pair_tensors(
            pair=pair,
            reference_means_native=reference_means,
            reference_covariances_native=reference_covariances,
            reference_foreground=reference_foreground,
            target_mask_flow=target_masks[pair.target],
            forward_flow=forward,
            backward_flow=backward,
            reference_camera=native_cameras[pair.reference],
            target_camera=native_cameras[pair.target],
            bounds_lower=lower,
            bounds_upper=upper,
            gates=gates,
            fb_sensitivity_px=sensitivity,
        )
        correspondence_record = result["pairs"][pair.pair_id]["artifacts"]["correspondence"]
        correspondence_path = analysis / correspondence_record["path"]
        validate_correspondence_npz(correspondence_path)
        with np.load(correspondence_path, allow_pickle=False) as archived:
            archived_accepted = torch.from_numpy(np.asarray(archived["accepted_geometry"]).copy())
            archived_dlt_valid = torch.from_numpy(np.asarray(archived["dlt_valid"]).copy())
            if bool((archived_accepted & ~archived_dlt_valid).any()):
                raise RuntimeError(
                    f"{pair.pair_id} claims accepted geometry without a DLT-valid solution"
                )
            for name, replay_tensor in replay_arrays.items():
                expected_array = replay_tensor.detach().cpu().numpy()
                if not np.array_equal(np.asarray(archived[name]), expected_array):
                    raise RuntimeError(
                        f"{pair.pair_id} correspondence field {name} does not replay "
                        "from acquisition evidence"
                    )
        reported_summary = dict(result["pairs"][pair.pair_id])
        artifacts = reported_summary.pop("artifacts")
        if reported_summary != replay_summary:
            raise RuntimeError(
                f"{pair.pair_id} gates, sequential counts, distributions, or FB sensitivity "
                "do not replay from acquisition evidence"
            )

        accepted = replay_arrays["accepted_foreground"]
        component_ids = replay_arrays["component_id"][accepted]
        points = replay_arrays["point_world"][accepted]
        pair_color = PAIR_DISPLAY_COLORS[pair.pair_id].to(
            dtype=points.dtype,
            device=points.device,
        )
        colors = pair_color[None].expand(points.shape[0], -1).clone()
        accepted_by_pair[pair.pair_id] = {
            "component_id": component_ids,
            "point_world": points,
        }
        combined_points.append(points)
        combined_colors.append(colors)
        offset = len(expected_lineage)
        expected_lineage.extend(
            {
                "pair_sample_index": offset + index,
                "pair_id": pair.pair_id,
                "component_id": int(component),
            }
            for index, component in enumerate(component_ids.tolist())
        )
        pair_ply = artifacts["pair_colored_ply"]
        if (
            pair_ply["sample_count"] != int(points.shape[0])
            or pair_ply["display_rgb"] != pair_color.tolist()
        ):
            raise RuntimeError(f"{pair.pair_id} pair PLY metadata does not replay")
        _verify_exact_mechanism_ply(
            analysis / pair_ply["path"],
            points=points,
            colors=colors,
            extent=float(
                json.loads(
                    (output / "acquisition" / manifest["bounds"]["path"]).read_text(
                        encoding="utf-8"
                    )
                )["extent"]
            ),
            context=pair.pair_id,
        )

    pair_ids = tuple(pair.pair_id for pair in PAIRS)
    cross_pair = cross_pair_disagreement(
        accepted_by_pair[pair_ids[0]]["component_id"],
        accepted_by_pair[pair_ids[0]]["point_world"],
        accepted_by_pair[pair_ids[1]]["component_id"],
        accepted_by_pair[pair_ids[1]]["point_world"],
    )
    overlap_count = int(cross_pair["overlap_unique_component_count"])
    union_count = int(cross_pair["union_unique_component_count"])
    disagreement_record = result["pair_overlap"]["artifact"]
    disagreement_path = analysis / disagreement_record["path"]
    _validate_cross_pair_npz(disagreement_path, overlap_count)
    with np.load(disagreement_path, allow_pickle=False) as archived:
        expected_cross = {
            "component_id": cross_pair["component_id"].numpy(),
            f"point_{pair_ids[0]}": cross_pair["first_points"].numpy(),
            f"point_{pair_ids[1]}": cross_pair["second_points"].numpy(),
            "distance_world": cross_pair["distance_world"].numpy(),
        }
        for name, expected in expected_cross.items():
            if not np.array_equal(np.asarray(archived[name]), expected):
                raise RuntimeError(f"cross-pair field {name} does not replay")
    expected_pair_counts = {
        pair_id: int(accepted_by_pair[pair_id]["component_id"].numel()) for pair_id in pair_ids
    }
    if (
        result["pair_overlap"]["accepted_component_counts"] != expected_pair_counts
        or result["pair_overlap"]["overlap_unique_component_count"] != overlap_count
        or result["pair_overlap"]["union_unique_component_count"] != union_count
        or result["pair_overlap"]["cross_pair_3d_disagreement_world"]
        != _finite_quantiles(cross_pair["distance_world"])
    ):
        raise RuntimeError("cross-pair counts/distributions do not replay")

    lineage = json.loads(
        (analysis / result["accepted_lineage"]["path"]).read_text(encoding="utf-8")
    )
    if (
        lineage["rows"] != expected_lineage
        or lineage["pair_sample_count"] != len(expected_lineage)
        or lineage["unique_component_count"] != union_count
    ):
        raise RuntimeError("accepted lineage does not replay from correspondence evidence")

    points = torch.cat(combined_points, dim=0)
    colors = torch.cat(combined_colors, dim=0)
    if result["accepted_counts"] != {
        "pair_sample_count": int(points.shape[0]),
        "unique_component_count": union_count,
    }:
        raise RuntimeError("combined accepted counts do not replay")
    bounds_payload = json.loads(
        (output / "acquisition" / manifest["bounds"]["path"]).read_text(encoding="utf-8")
    )
    _verify_exact_mechanism_ply(
        analysis / result["combined_pair_samples"]["path"],
        points=points,
        colors=colors,
        extent=float(bounds_payload["extent"]),
        context="combined pair-sample",
    )


def _validate_result_schema(result: Mapping[str, Any]) -> None:
    _require_exact_keys(
        result,
        {
            "artifact_type",
            "status",
            "completed_utc",
            "claim_scope",
            "decision_rule",
            "reference",
            "targets",
            "heldout",
            "plan",
            "acquisition",
            "source_verification",
            "gates",
            "fb_sensitivity_px",
            "pairs",
            "accepted_counts",
            "pair_overlap",
            "combined_pair_samples",
            "accepted_lineage",
            "rgb_boundary",
            "viewer",
        },
        "analysis result",
    )
    if (
        result["artifact_type"] != RESULT_ARTIFACT_TYPE
        or result["status"] != "EXPLORATORY_COMPLETE"
    ):
        raise RuntimeError("analysis result type/status changed")
    _validate_exploratory_contract(result, result=True)
    _require_utc_timestamp(result["completed_utc"], context="result.completed_utc")
    if result["reference"] != REFERENCE_VIEW or tuple(result["targets"]) != TARGET_VIEWS:
        raise RuntimeError("analysis result view protocol changed")
    _require_exact_keys(result["heldout"], {"view", "policy"}, "result.heldout")
    if result["heldout"]["view"] != HELDOUT_VIEW:
        raise RuntimeError("analysis result held-out declaration changed")
    _assert_heldout_only_in_declaration(result, context="analysis result")
    if set(result["pairs"]) != {pair.pair_id for pair in PAIRS}:
        raise RuntimeError("analysis result pair set changed")
    _require_exact_keys(
        result["decision_rule"],
        {"mode", "binary_mechanism_success", "policy"},
        "result.decision_rule",
    )
    if result["decision_rule"] != _frozen_decision_rule():
        raise RuntimeError("analysis result invented a binary mechanism verdict")
    _require_exact_keys(
        result["plan"],
        {"path", "file_sha256", "payload_sha256", "created_utc"},
        "result.plan",
    )
    _require_utc_timestamp(result["plan"]["created_utc"], context="result.plan.created_utc")
    for name in ("file_sha256", "payload_sha256"):
        _require_sha256(result["plan"][name], context=f"result.plan.{name}")
    _require_exact_keys(
        result["acquisition"],
        {"manifest_path", "manifest_sha256", "completed_utc", "backend"},
        "result.acquisition",
    )
    _require_utc_timestamp(
        result["acquisition"]["completed_utc"],
        context="result.acquisition.completed_utc",
    )
    _require_sha256(
        result["acquisition"]["manifest_sha256"],
        context="result.acquisition.manifest_sha256",
    )
    _require_exact_keys(
        result["source_verification"],
        {
            "verified_utc",
            "hashes",
            "source_color_policy",
            "git_at_analysis",
            "environment",
        },
        "result.source_verification",
    )
    if result["source_verification"]["source_color_policy"] != RGB_FREE_SOURCE_COLOR_POLICY:
        raise RuntimeError("analysis source-color verification boundary changed")
    _require_utc_timestamp(
        result["source_verification"]["verified_utc"],
        context="result.source_verification.verified_utc",
    )
    _require_exact_keys(
        result["source_verification"]["git_at_analysis"],
        {"head", "dirty", "status_sha256", "tracked_diff_sha256", "dirty_state_digest"},
        "result.source_verification.git_at_analysis",
    )
    if set(result["gates"]) != {field.name for field in dataclasses.fields(GateConfig)}:
        raise RuntimeError("analysis result gate schema changed")
    GateConfig(**result["gates"])
    if tuple(float(value) for value in result["fb_sensitivity_px"]) != FB_SENSITIVITY_PX:
        raise RuntimeError("analysis result FB sensitivity schema changed")
    expected_metric_names = {
        "fb_error_flow_px",
        "affine_source_design_condition",
        "affine_recovered_condition",
        "affine_rms_flow_px",
        "affine_determinant",
        "epipolar_residual_native_px",
        "dlt_reprojection_native_px",
        "dlt_condition",
        "dlt_nullspace_gap",
        "ray_angle_deg",
    }
    expected_stages = {
        "pretri",
        "dlt_valid",
        "accepted_geometry",
        "accepted_foreground",
    }
    expected_marginal_gates = {
        "sigma_points_valid",
        "all_sigma_samples_valid",
        "fb_primary",
        "affine_valid",
        "affine_rank",
        "affine_source_design_condition",
        "affine_recovered_condition",
        "affine_rms",
        "affine_determinant",
        "target_covariance_spd",
        "target_mask",
        "epipolar",
        "pretri",
        "dlt_valid",
        "dlt_all_cheiral",
        "ray_angle",
        "nullspace_gap",
        "in_bounds",
        "accepted_geometry",
        "accepted_foreground",
    }
    expected_sequential = [
        "sigma_points_valid",
        "all_sigma_samples_valid",
        "affine_valid",
        "affine_rank",
        "affine_source_design_condition",
        "affine_recovered_condition",
        "affine_rms",
        "affine_determinant",
        "target_covariance_spd",
        "target_mask",
        "epipolar",
        "fb_primary",
        "dlt_valid",
        "dlt_all_cheiral",
        "ray_angle",
        "nullspace_gap",
        "in_bounds",
        "accepted_foreground",
    ]
    for pair in PAIRS:
        summary = result["pairs"][pair.pair_id]
        _require_exact_keys(
            summary,
            {
                "pair_id",
                "component_count",
                "foreground_component_count",
                "counts",
                "fb_sensitivity",
                "stage_distributions",
                "dlt_distribution_policy",
                "archive_nonfinite_policy",
                "artifacts",
            },
            f"result.pairs.{pair.pair_id}",
        )
        if (
            summary["pair_id"] != pair.pair_id
            or summary["component_count"] != REFERENCE_COMPONENT_COUNT
        ):
            raise RuntimeError(f"result pair summary changed for {pair.pair_id}")
        _require_exact_keys(
            summary["counts"],
            {
                "marginal",
                "marginal_foreground",
                "sequential_order",
                "sequential",
                "sequential_foreground",
            },
            f"result.pairs.{pair.pair_id}.counts",
        )
        if (
            set(summary["counts"]["marginal"]) != expected_marginal_gates
            or set(summary["counts"]["marginal_foreground"]) != expected_marginal_gates
            or summary["counts"]["sequential_order"] != expected_sequential
            or set(summary["counts"]["sequential"]) != set(expected_sequential)
            or set(summary["counts"]["sequential_foreground"]) != set(expected_sequential)
        ):
            raise RuntimeError(f"result gate count schema changed for {pair.pair_id}")
        if set(summary["stage_distributions"]) != expected_stages:
            raise RuntimeError(f"result stage distribution set changed for {pair.pair_id}")
        for stage, metrics in summary["stage_distributions"].items():
            if set(metrics) != expected_metric_names:
                raise RuntimeError(
                    f"result distribution metric set changed for {pair.pair_id}/{stage}"
                )
            for metric, distribution in metrics.items():
                _require_exact_keys(
                    distribution,
                    {"finite_quantiles", "value_class_counts"},
                    f"result distribution {pair.pair_id}/{stage}/{metric}",
                )
                if set(distribution["finite_quantiles"]) != {
                    "0.0",
                    "0.1",
                    "0.5",
                    "0.9",
                    "0.99",
                    "1.0",
                }:
                    raise RuntimeError(
                        f"result quantile set changed for {pair.pair_id}/{stage}/{metric}"
                    )
                _require_exact_keys(
                    distribution["value_class_counts"],
                    {
                        "total",
                        "finite",
                        "nan",
                        "positive_infinity",
                        "negative_infinity",
                    },
                    f"result value classes {pair.pair_id}/{stage}/{metric}",
                )
                counts = distribution["value_class_counts"]
                if (
                    any(
                        not isinstance(value, int) or isinstance(value, bool) or value < 0
                        for value in counts.values()
                    )
                    or counts["total"]
                    != counts["finite"]
                    + counts["nan"]
                    + counts["positive_infinity"]
                    + counts["negative_infinity"]
                ):
                    raise RuntimeError(
                        f"result value-class counts changed for {pair.pair_id}/{stage}/{metric}"
                    )
        _require_exact_keys(
            summary["dlt_distribution_policy"],
            {"minimum_observation_count", "applies_to_prefix"},
            f"result.pairs.{pair.pair_id}.dlt_distribution_policy",
        )
        if summary["dlt_distribution_policy"] != {
            "minimum_observation_count": 2,
            "applies_to_prefix": "dlt_",
        }:
            raise RuntimeError("DLT result distribution policy changed")
        _require_exact_keys(
            summary["archive_nonfinite_policy"],
            {
                "encoding",
                "code_semantics",
                "sidecar_suffix",
                "replacement_value",
                "encoded_float_arrays",
                "replay",
            },
            f"result.pairs.{pair.pair_id}.archive_nonfinite_policy",
        )
        policy = summary["archive_nonfinite_policy"]
        if (
            policy["encoding"] != "uint8_sidecar"
            or policy["code_semantics"]
            != {str(code): meaning for code, meaning in NONFINITE_CODE_SEMANTICS.items()}
            or policy["sidecar_suffix"] != NONFINITE_CODE_SUFFIX
            or float(policy["replacement_value"]) != 0.0
            or policy["encoded_float_arrays"] != _encoded_correspondence_float_names()
        ):
            raise RuntimeError("correspondence nonfinite archive policy changed")
        if set(summary["fb_sensitivity"]) != {str(value) for value in FB_SENSITIVITY_PX}:
            raise RuntimeError(f"result FB sensitivity set changed for {pair.pair_id}")
        for threshold, record in summary["fb_sensitivity"].items():
            _require_exact_keys(
                record,
                {
                    "pretri_count",
                    "accepted_geometry_count",
                    "accepted_foreground_count",
                },
                f"result.pairs.{pair.pair_id}.fb_sensitivity.{threshold}",
            )
        _require_exact_keys(
            summary["artifacts"],
            {"correspondence", "pair_colored_ply"},
            f"result.pairs.{pair.pair_id}.artifacts",
        )
        _require_exact_keys(
            summary["artifacts"]["correspondence"],
            {"path", "sha256", "keys"},
            f"result.pairs.{pair.pair_id}.artifacts.correspondence",
        )
        _require_exact_keys(
            summary["artifacts"]["pair_colored_ply"],
            {"path", "sha256", "sample_count", "display_rgb"},
            f"result.pairs.{pair.pair_id}.artifacts.pair_colored_ply",
        )
        for artifact in ("correspondence", "pair_colored_ply"):
            _require_sha256(
                summary["artifacts"][artifact]["sha256"],
                context=f"result.pairs.{pair.pair_id}.artifacts.{artifact}.sha256",
            )
    _require_exact_keys(
        result["accepted_counts"],
        {"pair_sample_count", "unique_component_count"},
        "result.accepted_counts",
    )
    _require_exact_keys(
        result["pair_overlap"],
        {
            "pair_ids",
            "accepted_component_counts",
            "overlap_unique_component_count",
            "union_unique_component_count",
            "cross_pair_3d_disagreement_world",
            "artifact",
        },
        "result.pair_overlap",
    )
    if result["pair_overlap"]["pair_ids"] != [pair.pair_id for pair in PAIRS]:
        raise RuntimeError("analysis result overlap pair order changed")
    if set(result["pair_overlap"]["accepted_component_counts"]) != {pair.pair_id for pair in PAIRS}:
        raise RuntimeError("analysis result overlap count keys changed")
    _require_exact_keys(
        result["pair_overlap"]["artifact"],
        {"path", "sha256"},
        "result.pair_overlap.artifact",
    )
    _require_sha256(
        result["pair_overlap"]["artifact"]["sha256"],
        context="result.pair_overlap.artifact.sha256",
    )
    if set(result["pair_overlap"]["cross_pair_3d_disagreement_world"]) != {
        "0.0",
        "0.1",
        "0.5",
        "0.9",
        "0.99",
        "1.0",
    }:
        raise RuntimeError("analysis result disagreement quantiles changed")
    _require_exact_keys(
        result["combined_pair_samples"],
        {"path", "sha256", "render_semantics"},
        "result.combined_pair_samples",
    )
    _require_sha256(
        result["combined_pair_samples"]["sha256"],
        context="result.combined_pair_samples.sha256",
    )
    _require_exact_keys(
        result["accepted_lineage"],
        {"path", "sha256"},
        "result.accepted_lineage",
    )
    _require_sha256(
        result["accepted_lineage"]["sha256"],
        context="result.accepted_lineage.sha256",
    )
    _require_exact_keys(
        result["viewer"],
        {
            "command",
            "shell",
            "view_allowlist",
            "full_resolution",
            "rasterizer",
            "packed",
            "antialiased",
            "snapshot_requirement",
            "smoke_status",
        },
        "result.viewer",
    )
    if (
        result["viewer"]["rasterizer"] != "gsplat"
        or result["viewer"]["packed"] is not False
        or result["viewer"]["antialiased"] is not False
        or tuple(result["viewer"]["view_allowlist"]) != VIEWER_VIEWS
        or HELDOUT_VIEW in result["viewer"]["view_allowlist"]
        or not str(result["viewer"]["shell"]).startswith("LD_PRELOAD=")
    ):
        raise RuntimeError("analysis result viewer protocol changed")
    if (
        "never opened, decoded, or re-hashed JPEG" not in result["rgb_boundary"]
        or "never used as scientific inputs" not in result["rgb_boundary"]
    ):
        raise RuntimeError("analysis RGB-boundary declaration changed")


def analyze(output: Path) -> Path:
    """Run strictly RGB-free post-flow correspondence geometry and save a viewer-ready PLY."""
    started_utc = utc_now()
    final = output / "analysis"
    if final.exists():
        raise FileExistsError("refusing to overwrite immutable analysis directory")
    temporary: Path | None = None
    try:
        plan = _load_and_verify_plan(
            output,
            verify_source_color_bytes=False,
        )
        manifest = _verify_acquisition(
            output,
            verify_source_color_bytes=False,
        )
        acquisition = output / "acquisition"
        gates = GateConfig(**plan["gates"])
        sensitivity = tuple(float(value) for value in plan["fb_sensitivity_px"])
        planned_pairs = tuple(
            PairSpec(str(record["reference"]), str(record["target"])) for record in plan["pairs"]
        )
        temporary = Path(tempfile.mkdtemp(prefix=".g2sr-analysis-", dir=output))
        correspondence_dir = temporary / "correspondences"
        pair_ply_dir = temporary / "pair_plys"
        correspondence_dir.mkdir()
        pair_ply_dir.mkdir()

        calibration = json.loads(
            (acquisition / manifest["calibration"]["path"]).read_text(encoding="utf-8")
        )
        bounds = json.loads((acquisition / manifest["bounds"]["path"]).read_text(encoding="utf-8"))
        with np.load(acquisition / "analysis_input.npz", allow_pickle=False) as archive:
            reference_means = torch.from_numpy(np.asarray(archive["reference_means_native"]).copy())
            reference_covariances = torch.from_numpy(
                np.asarray(archive["reference_covariances_native"]).copy()
            )
            reference_foreground = torch.from_numpy(
                np.asarray(archive["reference_foreground"]).copy()
            )
            target_masks = {
                target: torch.from_numpy(np.asarray(archive[f"target_mask_flow_{target}"]).copy())
                for target in TARGET_VIEWS
            }

        native_cameras = {
            view: camera_from_record(record) for view, record in calibration["native"].items()
        }
        lower = torch.tensor(bounds["aabb_lower"], dtype=torch.float64)
        upper = torch.tensor(bounds["aabb_upper"], dtype=torch.float64)
        pair_summaries: dict[str, Any] = {}
        accepted_by_pair: dict[str, dict[str, torch.Tensor]] = {}
        accepted_points: list[torch.Tensor] = []
        accepted_pair_colors: list[torch.Tensor] = []
        accepted_lineage: list[dict[str, Any]] = []
        for pair in planned_pairs:
            flow_record = manifest["pairs"][pair.pair_id]
            with np.load(
                acquisition / flow_record["flow_path"],
                allow_pickle=False,
            ) as flow:
                forward = torch.from_numpy(np.asarray(flow["forward"]).copy())
                backward = torch.from_numpy(np.asarray(flow["backward"]).copy())
            arrays, summary = analyze_pair_tensors(
                pair=pair,
                reference_means_native=reference_means,
                reference_covariances_native=reference_covariances,
                reference_foreground=reference_foreground,
                target_mask_flow=target_masks[pair.target],
                forward_flow=forward,
                backward_flow=backward,
                reference_camera=native_cameras[pair.reference],
                target_camera=native_cameras[pair.target],
                bounds_lower=lower,
                bounds_upper=upper,
                gates=gates,
                fb_sensitivity_px=sensitivity,
            )
            correspondence_path = correspondence_dir / f"{pair.pair_id}.npz"
            _save_npz_exclusive(
                correspondence_path,
                **{name: tensor.detach().cpu().numpy() for name, tensor in arrays.items()},
            )
            correspondence_keys = validate_correspondence_npz(correspondence_path)
            accepted = arrays["accepted_foreground"]
            points = arrays["point_world"][accepted]
            component_ids = arrays["component_id"][accepted]
            pair_color = PAIR_DISPLAY_COLORS[pair.pair_id].to(
                dtype=points.dtype,
                device=points.device,
            )
            colors = pair_color[None].expand(points.shape[0], -1).clone()
            pair_gaussians = _mechanism_gaussians(
                points,
                colors,
                extent=float(bounds["extent"]),
            )
            pair_ply_path = pair_ply_dir / f"{pair.pair_id}.ply"
            _save_mechanism_ply(pair_ply_path, pair_gaussians)
            summary["artifacts"] = {
                "correspondence": {
                    "path": f"correspondences/{correspondence_path.name}",
                    "sha256": sha256_file(correspondence_path),
                    "keys": list(correspondence_keys),
                },
                "pair_colored_ply": {
                    "path": f"pair_plys/{pair_ply_path.name}",
                    "sha256": sha256_file(pair_ply_path),
                    "sample_count": int(points.shape[0]),
                    "display_rgb": pair_color.tolist(),
                },
            }
            pair_summaries[pair.pair_id] = summary
            accepted_by_pair[pair.pair_id] = {
                "component_id": component_ids,
                "point_world": points,
            }
            accepted_points.append(points)
            accepted_pair_colors.append(colors)
            lineage_offset = len(accepted_lineage)
            accepted_lineage.extend(
                {
                    "pair_sample_index": lineage_offset + local_index,
                    "pair_id": pair.pair_id,
                    "component_id": int(component),
                }
                for local_index, component in enumerate(component_ids.tolist())
            )

        pair_ids = tuple(pair.pair_id for pair in planned_pairs)
        first_ids = accepted_by_pair[pair_ids[0]]["component_id"]
        second_ids = accepted_by_pair[pair_ids[1]]["component_id"]
        cross_pair = cross_pair_disagreement(
            first_ids,
            accepted_by_pair[pair_ids[0]]["point_world"],
            second_ids,
            accepted_by_pair[pair_ids[1]]["point_world"],
        )
        overlap_tensor = cross_pair["component_id"]
        first_overlap = cross_pair["first_points"]
        second_overlap = cross_pair["second_points"]
        disagreement = cross_pair["distance_world"]
        overlap_count = cross_pair["overlap_unique_component_count"]
        union_count = cross_pair["union_unique_component_count"]
        disagreement_path = temporary / "cross_pair_disagreement.npz"
        _save_npz_exclusive(
            disagreement_path,
            component_id=overlap_tensor.numpy(),
            **{
                f"point_{pair_ids[0]}": first_overlap.numpy(),
                f"point_{pair_ids[1]}": second_overlap.numpy(),
            },
            distance_world=disagreement.numpy(),
        )
        _validate_cross_pair_npz(disagreement_path, overlap_count)

        combined_points = torch.cat(accepted_points, dim=0)
        combined_colors = torch.cat(accepted_pair_colors, dim=0)
        combined_gaussians = _mechanism_gaussians(
            combined_points,
            combined_colors,
            extent=float(bounds["extent"]),
        )
        combined_path = temporary / "accepted_pair_samples.ply"
        _save_mechanism_ply(combined_path, combined_gaussians)
        lineage_path = temporary / "accepted_lineage.json"
        write_json_exclusive(
            lineage_path,
            {
                "artifact_type": "g2sr_mechanism_pair_sample_lineage_v2",
                "semantics": (
                    "one row per accepted (pair, reference-component) sample; component IDs may "
                    "repeat across pairs"
                ),
                "pair_sample_count": len(accepted_lineage),
                "unique_component_count": union_count,
                "rows": accepted_lineage,
            },
        )
        command = viewer_command(output)
        plan_path = output / "plan.json"
        acquisition_manifest_path = acquisition / "manifest.json"
        result = {
            "artifact_type": RESULT_ARTIFACT_TYPE,
            "status": "EXPLORATORY_COMPLETE",
            "completed_utc": utc_now(),
            "claim_scope": EXPLORATORY_CLAIM_SCOPE,
            "decision_rule": plan["decision_rule"],
            "reference": REFERENCE_VIEW,
            "targets": list(TARGET_VIEWS),
            "heldout": {
                "view": HELDOUT_VIEW,
                "policy": "declaration only; absent from analysis inputs, gates, and outputs",
            },
            "plan": {
                "path": display_path(plan_path),
                "file_sha256": sha256_file(plan_path),
                "payload_sha256": canonical_hash(plan),
                "created_utc": plan["created_utc"],
            },
            "acquisition": {
                "manifest_path": display_path(acquisition_manifest_path),
                "manifest_sha256": sha256_file(acquisition_manifest_path),
                "completed_utc": manifest["completed_utc"],
                "backend": manifest["backend"],
            },
            "source_verification": {
                "verified_utc": utc_now(),
                "hashes": _current_source_hashes(
                    plan,
                    verify_source_color_bytes=False,
                ),
                "source_color_policy": RGB_FREE_SOURCE_COLOR_POLICY,
                "git_at_analysis": _git_record(),
                "environment": _environment_record(torch.device("cpu")),
            },
            "gates": plan["gates"],
            "fb_sensitivity_px": plan["fb_sensitivity_px"],
            "pairs": pair_summaries,
            "accepted_counts": {
                "pair_sample_count": combined_gaussians.n,
                "unique_component_count": union_count,
            },
            "pair_overlap": {
                "pair_ids": list(pair_ids),
                "accepted_component_counts": {
                    pair_id: int(accepted_by_pair[pair_id]["component_id"].numel())
                    for pair_id in pair_ids
                },
                "overlap_unique_component_count": overlap_count,
                "union_unique_component_count": union_count,
                "cross_pair_3d_disagreement_world": _finite_quantiles(disagreement),
                "artifact": {
                    "path": disagreement_path.name,
                    "sha256": sha256_file(disagreement_path),
                },
            },
            "combined_pair_samples": {
                "path": combined_path.name,
                "sha256": sha256_file(combined_path),
                "render_semantics": (
                    "small isotropic SH0 pair-colored markers; each accepted pair/component is "
                    "a separate sample, including duplicated component lineages across pairs"
                ),
            },
            "accepted_lineage": {
                "path": lineage_path.name,
                "sha256": sha256_file(lineage_path),
            },
            "rgb_boundary": (
                "analysis decoded and scientifically consumed only hash-verified NPZ/JSON "
                "post-flow artifacts and source masks; it trusted the immutable acquisition "
                "source-color digests and never opened, decoded, or re-hashed JPEG source colors, "
                "which were never used as scientific inputs. view-smoke separately decodes only "
                "its explicit training-view allowlist for reporting"
            ),
            "viewer": {
                "command": command,
                "shell": viewer_shell(command),
                "view_allowlist": list(VIEWER_VIEWS),
                "full_resolution": True,
                "rasterizer": "gsplat",
                "packed": False,
                "antialiased": False,
                "snapshot_requirement": (
                    "view-smoke must first save one native calibrated exact gsplat snapshot"
                ),
                "smoke_status": "run view-smoke phase",
            },
        }
        _validate_result_schema(result)
        write_json_exclusive(temporary / "result.json", result)
        # Verify the complete unpublished directory from the immutable flow/mask/calibration
        # evidence before the EXPLORATORY_COMPLETE artifact becomes visible at ``analysis/``.
        _verify_analysis_directory(output, temporary)
        os.replace(temporary, final)
        temporary = None
        return final / "result.json"
    except BaseException as exc:
        if temporary is not None and temporary.exists():
            failed = output / f"analysis_failed_partial_{time.time_ns()}"
            write_json_exclusive(
                temporary / "failure_context.json",
                {
                    "artifact_type": "g2sr_analysis_partial_failure_v1",
                    "failed_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            os.replace(temporary, failed)
            temporary = None
        output.mkdir(parents=True, exist_ok=True)
        _write_failure_attempt(
            output,
            phase="analysis",
            started_utc=started_utc,
            error=exc,
        )
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _verify_analysis_directory(output: Path, analysis: Path) -> dict[str, Any]:
    """Verify one published or not-yet-published analysis directory end to end."""
    plan = _load_and_verify_plan(
        output,
        verify_source_color_bytes=False,
    )
    acquisition = _verify_acquisition(
        output,
        verify_source_color_bytes=False,
    )
    result_path = analysis / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError("completed analysis result is absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate_result_schema(result)
    plan_path = output / "plan.json"
    acquisition_manifest_path = output / "acquisition" / "manifest.json"
    if (
        result["plan"]["file_sha256"] != sha256_file(plan_path)
        or result["plan"]["payload_sha256"] != canonical_hash(plan)
        or result["plan"]["created_utc"] != plan["created_utc"]
    ):
        raise RuntimeError("analysis result plan hash-chain mismatch")
    if (
        result["acquisition"]["manifest_sha256"] != sha256_file(acquisition_manifest_path)
        or result["acquisition"]["completed_utc"] != acquisition["completed_utc"]
        or result["acquisition"]["backend"] != acquisition["backend"]
    ):
        raise RuntimeError("analysis result acquisition hash-chain mismatch")
    if result["source_verification"]["hashes"] != _current_source_hashes(
        plan,
        verify_source_color_bytes=False,
    ):
        raise RuntimeError("analysis result source verification changed")
    if result["gates"] != plan["gates"] or result["fb_sensitivity_px"] != plan["fb_sensitivity_px"]:
        raise RuntimeError("analysis result gates/protocol differ from plan")
    expected_viewer_command = viewer_command(output)
    if (
        result["viewer"]["command"] != expected_viewer_command
        or result["viewer"]["shell"] != viewer_shell(expected_viewer_command)
        or tuple(result["viewer"]["view_allowlist"]) != VIEWER_VIEWS
    ):
        raise RuntimeError("analysis result viewer command/view binding changed")
    combined = analysis / result["combined_pair_samples"]["path"]
    if (
        result["combined_pair_samples"]["path"] != "accepted_pair_samples.ply"
        or sha256_file(combined) != result["combined_pair_samples"]["sha256"]
    ):
        raise RuntimeError("combined pair-sample PLY binding changed")
    lineage_path = analysis / result["accepted_lineage"]["path"]
    if (
        result["accepted_lineage"]["path"] != "accepted_lineage.json"
        or sha256_file(lineage_path) != result["accepted_lineage"]["sha256"]
    ):
        raise RuntimeError("accepted lineage binding changed")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    _require_exact_keys(
        lineage,
        {
            "artifact_type",
            "semantics",
            "pair_sample_count",
            "unique_component_count",
            "rows",
        },
        "accepted lineage",
    )
    if (
        lineage["artifact_type"] != "g2sr_mechanism_pair_sample_lineage_v2"
        or lineage["pair_sample_count"] != result["accepted_counts"]["pair_sample_count"]
        or lineage["unique_component_count"] != result["accepted_counts"]["unique_component_count"]
        or len(lineage["rows"]) != lineage["pair_sample_count"]
    ):
        raise RuntimeError("accepted lineage counts differ from result")
    observed_unique: set[int] = set()
    observed_pair_counts = {pair.pair_id: 0 for pair in PAIRS}
    for index, row in enumerate(lineage["rows"]):
        _require_exact_keys(
            row,
            {"pair_sample_index", "pair_id", "component_id"},
            f"accepted lineage row {index}",
        )
        pair_id = row["pair_id"]
        component_id = row["component_id"]
        if (
            row["pair_sample_index"] != index
            or pair_id not in observed_pair_counts
            or not isinstance(component_id, int)
            or not 0 <= component_id < REFERENCE_COMPONENT_COUNT
        ):
            raise RuntimeError(f"accepted lineage row {index} is malformed")
        observed_pair_counts[pair_id] += 1
        observed_unique.add(component_id)
    if (
        len(observed_unique) != lineage["unique_component_count"]
        or observed_pair_counts != result["pair_overlap"]["accepted_component_counts"]
    ):
        raise RuntimeError("accepted lineage membership/counts differ from result")
    _assert_heldout_only_in_declaration(
        {"heldout": result["heldout"], "lineage": lineage},
        context="accepted lineage",
    )
    disagreement_record = result["pair_overlap"]["artifact"]
    disagreement_path = analysis / disagreement_record["path"]
    if (
        disagreement_record["path"] != "cross_pair_disagreement.npz"
        or sha256_file(disagreement_path) != disagreement_record["sha256"]
    ):
        raise RuntimeError("cross-pair disagreement binding changed")
    _validate_cross_pair_npz(
        disagreement_path,
        result["pair_overlap"]["overlap_unique_component_count"],
    )
    for pair in PAIRS:
        artifacts = result["pairs"][pair.pair_id]["artifacts"]
        correspondence = artifacts["correspondence"]
        correspondence_path = analysis / correspondence["path"]
        if (
            correspondence["path"] != f"correspondences/{pair.pair_id}.npz"
            or sha256_file(correspondence_path) != correspondence["sha256"]
        ):
            raise RuntimeError(f"correspondence artifact binding changed for {pair.pair_id}")
        if list(validate_correspondence_npz(correspondence_path)) != correspondence["keys"]:
            raise RuntimeError(f"correspondence key declaration changed for {pair.pair_id}")
        verify_correspondence_evidence(
            correspondence_path,
            result["pairs"][pair.pair_id],
            gates=GateConfig(**result["gates"]),
        )
        pair_ply = artifacts["pair_colored_ply"]
        pair_ply_path = analysis / pair_ply["path"]
        if (
            pair_ply["path"] != f"pair_plys/{pair.pair_id}.ply"
            or sha256_file(pair_ply_path) != pair_ply["sha256"]
            or pair_ply["sample_count"]
            != result["pair_overlap"]["accepted_component_counts"][pair.pair_id]
        ):
            raise RuntimeError(f"pair PLY binding changed for {pair.pair_id}")
    _verify_full_analysis_replay(
        output,
        analysis=analysis,
        result=result,
        manifest=acquisition,
        plan=plan,
    )
    return result


def _verify_analysis(output: Path) -> dict[str, Any]:
    return _verify_analysis_directory(output, output / "analysis")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _render_required_native_snapshot(output: Path, directory: Path) -> dict[str, Any]:
    from PIL import Image as PILImage

    from rtgs.viewer import render_exact_snapshot

    result = _verify_analysis(output)
    manifest = json.loads((output / "acquisition" / "manifest.json").read_text(encoding="utf-8"))
    ply_path = output / "analysis" / result["combined_pair_samples"]["path"]
    if sha256_file(ply_path) != result["combined_pair_samples"]["sha256"]:
        raise RuntimeError("viewer input PLY differs from the immutable analysis result")
    gaussians = Gaussians3D.load_ply(ply_path)
    if gaussians.n == 0:
        raise RuntimeError("cannot satisfy mandatory gsplat snapshot with zero accepted samples")
    calibration_path = output / "acquisition" / manifest["calibration"]["path"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera_record_value = calibration["native"][REFERENCE_VIEW]
    _validate_camera_payload(camera_record_value, context="snapshot camera")
    camera = camera_from_record(camera_record_value)
    started_utc = utc_now()
    snapshot = render_exact_snapshot(
        gaussians,
        camera,
        device="cuda",
        rasterizer="gsplat",
        packed=False,
        antialiased=False,
    )
    if snapshot.color.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
        raise RuntimeError("exact gsplat snapshot has non-native dimensions")
    if not bool(torch.isfinite(snapshot.color).all()):
        raise RuntimeError("exact gsplat snapshot contains non-finite pixels")
    if "GsplatRasterizer" not in snapshot.backend or not snapshot.device.startswith("cuda"):
        raise RuntimeError(
            "mandatory snapshot was not rendered by the CUDA gsplat backend: "
            f"{snapshot.backend} on {snapshot.device}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / f"{REFERENCE_VIEW}_native_gsplat.png"
    image = snapshot.color.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
    PILImage.fromarray(image, mode="RGB").save(path)
    record = {
        "camera_view": REFERENCE_VIEW,
        "camera_record_sha256": canonical_hash(camera_record_value),
        "calibration_file_sha256": sha256_file(calibration_path),
        "source_ply_sha256": sha256_file(ply_path),
        "backend_class": snapshot.backend,
        "device": snapshot.device,
        "width": camera.width,
        "height": camera.height,
        "packed": False,
        "antialiased": False,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    _validate_snapshot_record(record, directory=directory)
    return record


def _validate_snapshot_record(record: Mapping[str, Any], *, directory: Path) -> None:
    _require_exact_keys(
        record,
        {
            "camera_view",
            "camera_record_sha256",
            "calibration_file_sha256",
            "source_ply_sha256",
            "backend_class",
            "device",
            "width",
            "height",
            "packed",
            "antialiased",
            "started_utc",
            "completed_utc",
            "path",
            "sha256",
            "bytes",
        },
        "viewer snapshot record",
    )
    if (
        record["camera_view"] != REFERENCE_VIEW
        or record["width"] != NATIVE_WIDTH
        or record["height"] != NATIVE_HEIGHT
        or record["packed"] is not False
        or record["antialiased"] is not False
    ):
        raise RuntimeError("viewer snapshot protocol metadata changed")
    if "GsplatRasterizer" not in record["backend_class"] or not str(record["device"]).startswith(
        "cuda"
    ):
        raise RuntimeError("viewer snapshot does not attest CUDA gsplat")
    _require_utc_timestamp(record["started_utc"], context="snapshot.started_utc")
    _require_utc_timestamp(record["completed_utc"], context="snapshot.completed_utc")
    for name in (
        "camera_record_sha256",
        "calibration_file_sha256",
        "source_ply_sha256",
        "sha256",
    ):
        _require_sha256(record[name], context=f"snapshot.{name}")
    path = directory / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError("viewer snapshot file/hash/size binding changed")


def _validate_viewer_receipt(
    receipt: Mapping[str, Any],
    *,
    output: Path,
    directory: Path,
) -> None:
    """Validate a receipt against current upstream files, sources, command, and render inputs."""
    _require_exact_keys(
        receipt,
        {
            "artifact_type",
            "status",
            "completed_utc",
            "plan_file_sha256",
            "acquisition_manifest_sha256",
            "analysis_result_sha256",
            "url",
            "http_status",
            "last_error",
            "command",
            "shell",
            "environment",
            "reporting_inputs",
            "full_resolution",
            "rasterizer",
            "packed",
            "antialiased",
            "mandatory_native_snapshot",
            "process_handshake",
            "process_output_tail",
        },
        "viewer receipt",
    )
    if receipt["artifact_type"] != VIEWER_ARTIFACT_TYPE or receipt["status"] not in {
        "PASS",
        "FAIL",
    }:
        raise RuntimeError("viewer receipt type/status changed")
    _require_utc_timestamp(receipt["completed_utc"], context="viewer.completed_utc")
    for name in (
        "plan_file_sha256",
        "acquisition_manifest_sha256",
        "analysis_result_sha256",
    ):
        _require_sha256(receipt[name], context=f"viewer.{name}")
    _require_exact_keys(receipt["environment"], {"LD_PRELOAD"}, "viewer.environment")
    if receipt["environment"]["LD_PRELOAD"] != str(LOCAL_LIBSTDCXX_PRELOAD):
        raise RuntimeError("viewer LD_PRELOAD record changed")
    _require_exact_keys(
        receipt["reporting_inputs"],
        {"view_allowlist", "source_colors", "calibration"},
        "viewer.reporting_inputs",
    )
    allowlist = validate_viewer_allowlist(receipt["reporting_inputs"]["view_allowlist"])
    if set(receipt["reporting_inputs"]["source_colors"]) != set(allowlist):
        raise RuntimeError("viewer reporting source-color view set differs from allowlist")
    for view, binding in receipt["reporting_inputs"]["source_colors"].items():
        _require_exact_keys(
            binding,
            {"path", "sha256", "bytes", "mtime_ns"},
            f"viewer.reporting_inputs.source_colors.{view}",
        )
        _require_sha256(
            binding["sha256"],
            context=f"viewer.reporting_inputs.source_colors.{view}.sha256",
        )
    _require_exact_keys(
        receipt["reporting_inputs"]["calibration"],
        {"path", "sha256", "bytes", "mtime_ns"},
        "viewer.reporting_inputs.calibration",
    )
    _require_sha256(
        receipt["reporting_inputs"]["calibration"]["sha256"],
        context="viewer.reporting_inputs.calibration.sha256",
    )
    if (
        receipt["full_resolution"] is not True
        or receipt["rasterizer"] != "gsplat"
        or receipt["packed"] is not False
        or receipt["antialiased"] is not False
        or not str(receipt["shell"]).startswith("LD_PRELOAD=")
    ):
        raise RuntimeError("viewer receipt render protocol changed")
    snapshot = receipt["mandatory_native_snapshot"]
    if snapshot is not None:
        _validate_snapshot_record(snapshot, directory=directory / "calibrated_snapshots")
    if receipt["status"] == "PASS" and (receipt["http_status"] != 200 or snapshot is None):
        raise RuntimeError("viewer PASS requires HTTP 200 and a verified native snapshot")
    handshake = receipt["process_handshake"]
    _require_exact_keys(
        handshake,
        {
            "token",
            "token_sha256",
            "pid",
            "port",
            "marker",
            "marker_observed",
            "listener_owner_verified",
        },
        "viewer.process_handshake",
    )
    token = handshake["token"]
    if (
        not isinstance(token, str)
        or len(token) != 64
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise RuntimeError("viewer handshake token is malformed")
    _require_sha256(handshake["token_sha256"], context="viewer.process_handshake.token_sha256")
    if handshake["token_sha256"] != hashlib.sha256(token.encode()).hexdigest():
        raise RuntimeError("viewer handshake token digest changed")
    if (
        not isinstance(handshake["pid"], int)
        or handshake["pid"] < -1
        or not isinstance(handshake["port"], int)
        or handshake["port"] <= 0
    ):
        raise RuntimeError("viewer process handshake identity is malformed")
    expected_marker = (
        _viewer_handshake_marker(
            token,
            pid=handshake["pid"],
            port=handshake["port"],
        )
        if handshake["pid"] > 0
        else ""
    )
    if handshake["marker"] != expected_marker:
        raise RuntimeError("viewer process handshake marker changed")
    if receipt["status"] == "PASS" and (
        handshake["marker_observed"] is not True
        or handshake["listener_owner_verified"] is not True
        or handshake["marker"] not in str(receipt["process_output_tail"])
    ):
        raise RuntimeError(
            "viewer PASS requires its spawned process marker and listening-socket ownership"
        )

    output = output.expanduser().resolve()
    result = _verify_analysis(output)
    plan_path = output / "plan.json"
    acquisition_manifest_path = output / "acquisition" / "manifest.json"
    analysis_result_path = output / "analysis" / "result.json"
    expected_upstream = {
        "plan_file_sha256": sha256_file(plan_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "analysis_result_sha256": sha256_file(analysis_result_path),
    }
    if any(receipt[name] != digest for name, digest in expected_upstream.items()):
        raise RuntimeError("viewer receipt upstream hash chain differs from current outputs")
    if receipt["reporting_inputs"]["source_colors"] != _viewer_source_bindings():
        raise RuntimeError("viewer reporting source-color bindings differ from current files")
    _, source_calibration_path = _calibration_records()
    if receipt["reporting_inputs"]["calibration"] != _file_binding(source_calibration_path):
        raise RuntimeError("viewer reporting calibration binding differs from current source")

    parsed_url = urllib.parse.urlparse(str(receipt["url"]))
    try:
        receipt_port = parsed_url.port
    except ValueError as exc:
        raise RuntimeError("viewer receipt URL has an invalid port") from exc
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname != "127.0.0.1"
        or receipt_port is None
        or receipt_port != handshake["port"]
        or parsed_url.path not in {"", "/"}
    ):
        raise RuntimeError("viewer receipt URL binding changed")
    expected_command = viewer_command(
        output,
        port=receipt_port,
        handshake_token=token,
    )
    if (
        receipt["command"] != expected_command
        or receipt["shell"] != viewer_shell(expected_command)
        or tuple(result["viewer"]["view_allowlist"]) != allowlist
    ):
        raise RuntimeError("viewer receipt command/view binding differs from current protocol")

    manifest = json.loads(acquisition_manifest_path.read_text(encoding="utf-8"))
    calibration_path = output / "acquisition" / manifest["calibration"]["path"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    ply_path = output / "analysis" / result["combined_pair_samples"]["path"]
    if snapshot is not None and (
        snapshot["camera_record_sha256"] != canonical_hash(calibration["native"][REFERENCE_VIEW])
        or snapshot["calibration_file_sha256"] != sha256_file(calibration_path)
        or snapshot["source_ply_sha256"] != sha256_file(ply_path)
        or snapshot["source_ply_sha256"] != result["combined_pair_samples"]["sha256"]
    ):
        raise RuntimeError("viewer snapshot camera/calibration/PLY binding differs from outputs")


def _verify_viewer(output: Path) -> dict[str, Any]:
    directory = output / "viewer"
    receipt_path = directory / "viewer_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("completed viewer receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _validate_viewer_receipt(receipt, output=output, directory=directory)
    return receipt


def smoke_viewer(output: Path, *, port: int | None = None, timeout: float = 180.0) -> Path:
    """Save a native exact gsplat snapshot, then smoke-test the reporting-only HTTP viewer."""
    final = output / "viewer"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite immutable viewer artifacts: {final}")
    if not (output / "analysis" / "accepted_pair_samples.ply").is_file():
        raise FileNotFoundError("run analysis before viewer smoke")
    temporary = Path(tempfile.mkdtemp(prefix=".g2sr-viewer-", dir=output))
    snapshots = temporary / "calibrated_snapshots"
    snapshot_record: dict[str, Any] | None = None
    process: subprocess.Popen[str] | None = None
    output_text = ""
    port = _free_port() if port is None else port
    token = secrets.token_hex(32)
    command = viewer_command(output, port=port, handshake_token=token)
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(LOCAL_LIBSTDCXX_PRELOAD)
    url = f"http://127.0.0.1:{port}"
    response_status = None
    error = None
    marker = ""
    marker_observed = False
    listener_owner_verified = False
    process_pid = -1
    process_log_path = temporary / "viewer_process.log"
    process_log: Any | None = None
    try:
        # Refuse an already-serving endpoint before doing any render or spawning a process.  The
        # later token/PID/socket-owner checks close the remaining identity gap after this preflight.
        _assert_port_available(port)
        # This mandatory programmatic render is completed and hashed before the HTTP process
        # starts, so a healthy web server alone can never satisfy the viewer handoff.
        snapshot_record = _render_required_native_snapshot(output, snapshots)
        process_log = process_log_path.open("w+", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=process_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process_pid = process.pid
        marker = _viewer_handshake_marker(token, pid=process.pid, port=port)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            process_log.flush()
            output_text = process_log_path.read_text(encoding="utf-8")
            marker_observed = marker in output_text
            if marker_observed:
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as response:
                        response_status = response.status
                        if response_status == 200:
                            listener_owner_verified = _process_owns_listening_port(
                                process.pid,
                                port,
                            )
                            if listener_owner_verified and process.poll() is None:
                                break
                except OSError as caught:
                    error = str(caught)
            time.sleep(0.25)
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process_log is not None:
            process_log.flush()
            process_log.close()
        if process_log_path.is_file():
            output_text = process_log_path.read_text(encoding="utf-8")
            marker_observed = marker in output_text
    receipt = {
        "artifact_type": VIEWER_ARTIFACT_TYPE,
        "status": (
            "PASS"
            if (
                response_status == 200
                and snapshot_record is not None
                and marker_observed
                and listener_owner_verified
            )
            else "FAIL"
        ),
        "completed_utc": utc_now(),
        "plan_file_sha256": sha256_file(output / "plan.json"),
        "acquisition_manifest_sha256": sha256_file(output / "acquisition" / "manifest.json"),
        "analysis_result_sha256": sha256_file(output / "analysis" / "result.json"),
        "url": url,
        "http_status": response_status,
        "last_error": error,
        "command": command,
        "shell": viewer_shell(command),
        "environment": {"LD_PRELOAD": str(LOCAL_LIBSTDCXX_PRELOAD)},
        "reporting_inputs": {
            "view_allowlist": list(VIEWER_VIEWS),
            "source_colors": _viewer_source_bindings(),
            "calibration": _file_binding(_calibration_records()[1]),
        },
        "full_resolution": True,
        "rasterizer": "gsplat",
        "packed": False,
        "antialiased": False,
        "mandatory_native_snapshot": snapshot_record,
        "process_handshake": {
            "token": token,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "pid": process_pid,
            "port": port,
            "marker": marker,
            "marker_observed": marker_observed,
            "listener_owner_verified": listener_owner_verified,
        },
        "process_output_tail": ((marker + "\n" if marker_observed else "") + output_text[-4000:]),
    }
    _validate_viewer_receipt(receipt, output=output, directory=temporary)
    receipt_path = temporary / "viewer_receipt.json"
    write_json_exclusive(receipt_path, receipt)
    os.replace(temporary, final)
    _verify_viewer(output)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"viewer smoke failed; see {final / 'viewer_receipt.json'}")
    return final / "viewer_receipt.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("plan", "analyze", "view-smoke"):
        child = subparsers.add_parser(phase)
        child.add_argument("--out", type=Path, default=DEFAULT_OUT)
    acquisition = subparsers.add_parser("acquire")
    acquisition.add_argument("--out", type=Path, default=DEFAULT_OUT)
    acquisition.add_argument("--backend", default=FLOW_BACKEND, choices=[FLOW_BACKEND])
    acquisition.add_argument(
        "--device",
        default=ACQUISITION_DEVICE,
        choices=[ACQUISITION_DEVICE],
        help="frozen acquisition device (CUDA is required)",
    )
    acquisition.add_argument(
        "--allow-download",
        action="store_true",
        help="allow torchvision to fetch official C_T_V2 weights when absent from the local cache",
    )
    viewer = subparsers.choices["view-smoke"]
    viewer.add_argument("--port", type=int)
    viewer.add_argument("--timeout", type=float, default=180.0)
    server = subparsers.add_parser("serve-viewer")
    server.add_argument("--out", type=Path, default=DEFAULT_OUT)
    server.add_argument(
        "--view-ids",
        type=lambda value: tuple(part.strip() for part in value.split(",") if part.strip()),
        required=True,
        help="comma-separated exact training-view allowlist",
    )
    server.add_argument("--port", type=int, required=True)
    server.add_argument(
        "--handshake-token",
        help="optional 32-byte lowercase-hex process identity token used by view-smoke",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.out.expanduser().resolve()
    if args.phase == "plan":
        path = write_plan(output)
    elif args.phase == "acquire":
        path = acquire(
            output,
            backend_name=args.backend,
            device=args.device,
            allow_download=args.allow_download,
        )
    elif args.phase == "analyze":
        path = analyze(output)
    elif args.phase == "view-smoke":
        path = smoke_viewer(output, port=args.port, timeout=args.timeout)
    else:
        path = serve_viewer(
            output,
            view_ids=args.view_ids,
            port=args.port,
            handshake_token=args.handshake_token,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
