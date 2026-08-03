"""Deterministic source adapters for the pre-outcome StructSplat BENCH-019 portfolio.

The adapters in this module bind source pixels, masks, cameras, and train/held-out roles before
field fitting.  They never fit a field, execute realtime-gs, or read a downstream outcome.
Development TUM RGB-D archives are inspected without filesystem extraction; confirmation payload
materialization is denied by default.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import io
import math
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage

from rtgs.bench019 import (
    ExportError,
    canonical_json,
    describe_artifact,
    load_json_object,
)
from rtgs.bench019_portfolio import source_digest, validate_capture_portfolio

ADAPTER_SCHEMA = "rtgs.structsplat_bench019.source_adapter.v1"
MATERIALIZATION_SCHEMA = "rtgs.structsplat_bench019.materialization.v1"
VIEW_COUNT = 26
HELDOUT_ORDINALS = (7, 15, 23)

TUM_ASSOCIATION_MAX_NS = 20_000_000
TUM_POSE_INTERPOLATION_MAX_NS = 20_000_000
TUM_MIN_TRANSLATION_M = 0.08
TUM_MIN_ROTATION_DEG = 8.0
TUM_WIDTH = 640
TUM_HEIGHT = 480
TUM_FX = 525.0
TUM_FY = 525.0
# TUM's ROS-default principal point is (319.5, 239.5) in integer-centered coordinates.
TUM_CX = 320.0
TUM_CY = 240.0
TUM_DEPTH_DIVISOR = 5000.0
TUM_MIN_DEPTH_M = 0.3
TUM_MAX_DEPTH_M = 5.0

_TOP_KEYS = frozenset(
    {
        "schema",
        "state",
        "capture_id",
        "role",
        "source_kind",
        "portfolio",
        "source_artifacts",
        "source_digest",
        "selection_policy",
        "mask_policy",
        "views",
        "semantic_digest",
    }
)
_SOURCE_KEYS = frozenset({"id", "artifact"})
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_VIEW_KEYS = frozenset(
    {
        "id",
        "ordinal",
        "split",
        "rgb_source",
        "mask_source",
        "camera",
        "preprocessing",
        "source_metadata",
    }
)
_CAMERA_KEYS = frozenset({"fx", "fy", "cx", "cy", "width", "height", "R", "t"})
_FILE_REFERENCE_KEYS = frozenset({"kind", "source_id"})
_TAR_REFERENCE_KEYS = frozenset({"kind", "source_id", "member", "sha256", "bytes"})
_DEPTH_MASK_KEYS = frozenset(
    {"kind", "depth_source", "depth_divisor", "min_depth_m", "max_depth_m"}
)
_PREPROCESSING_KEYS = frozenset({"rgb", "mask", "distortion_coefficients"})
_STAGE_METADATA_KEYS = frozenset({"kind", "camera_id"})
_TUM_METADATA_KEYS = frozenset(
    {
        "kind",
        "rgb_timestamp_ns",
        "rgb_timestamp_token",
        "depth_timestamp_ns",
        "depth_timestamp_token",
        "rgb_depth_delta_ns",
        "pose_center_m",
        "pose_quaternion_xyzw",
    }
)
_STAGE_SELECTION_KEYS = frozenset({"kind", "count", "heldout_ordinals"})
_TUM_SELECTION_KEYS = frozenset(
    {
        "kind",
        "association_max_ns_strict",
        "pose_interpolation_max_ns",
        "translation_m_inclusive",
        "rotation_deg_inclusive",
        "associated_triples",
        "pose_keyframes",
        "uniform_source_indices",
        "count",
        "heldout_ordinals",
        "rounding",
    }
)
_STAGE_MASK_POLICY_KEYS = frozenset({"kind", "threshold", "source"})
_TUM_MASK_POLICY_KEYS = frozenset({"kind", "depth_divisor", "min_depth_m", "max_depth_m", "source"})
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "adapter",
        "capture_id",
        "role",
        "outputs",
        "semantic_digest",
    }
)
_MATERIALIZED_OUTPUT_KEYS = frozenset({"id", "artifact"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class TimedPath:
    """One timestamped archive-relative RGB or depth path."""

    timestamp_ns: int
    timestamp_token: str
    path: str


@dataclass(frozen=True)
class TimedPose:
    """One TUM camera-to-world pose at an integer nanosecond timestamp."""

    timestamp_ns: int
    timestamp_token: str
    center: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class AssociatedFrame:
    """One deterministic RGB/depth pair with a pose at the depth timestamp."""

    rgb: TimedPath
    depth: TimedPath
    pose: TimedPose
    rgb_depth_delta_ns: int


def _exact(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ExportError(
            f"{label} keys are not exact "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ExportError(f"{label} must be a portable identifier")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExportError(f"{label} must be finite")
    return result


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExportError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExportError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _artifact(value: object, *, label: str, verify_file: bool) -> dict[str, Any]:
    record = _exact(value, _ARTIFACT_KEYS, label=label)
    path = record["path"]
    if not isinstance(path, str) or not path.strip() or not Path(path).is_absolute():
        raise ExportError(f"{label}.path must be a non-empty absolute path")
    _sha256(record["sha256"], label=f"{label}.sha256")
    _nonnegative_integer(record["bytes"], label=f"{label}.bytes")
    if verify_file and describe_artifact(path) != record:
        raise ExportError(f"{label} differs from its bound file")
    return record


def _source_inventory(
    value: object, *, verify_files: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise ExportError("adapter source_artifacts must be a non-empty list")
    sources: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        record = _exact(item, _SOURCE_KEYS, label=f"adapter source_artifacts[{index}]")
        source_id = _identifier(record["id"], label=f"adapter source_artifacts[{index}].id")
        if source_id in indexed:
            raise ExportError(f"adapter has duplicate source artifact {source_id}")
        descriptor = _artifact(
            record["artifact"],
            label=f"adapter source artifact {source_id}",
            verify_file=verify_files,
        )
        copied = {"id": source_id, "artifact": descriptor}
        sources.append(copied)
        indexed[source_id] = copied
    return sources, indexed


def _safe_relative_path(value: object, *, label: str = "archive member") -> str:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{label} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExportError(f"{label} is unsafe: {value!r}")
    return path.as_posix()


def _data_lines(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(stripped.split())
    return rows


def _timestamp_ns(token: str) -> int:
    try:
        seconds = Decimal(token)
    except Exception as error:
        raise ExportError(f"invalid TUM timestamp {token!r}") from error
    if not seconds.is_finite() or seconds < 0:
        raise ExportError(f"invalid TUM timestamp {token!r}")
    value = seconds * Decimal(1_000_000_000)
    integral = value.to_integral_value()
    if value != integral:
        raise ExportError(f"TUM timestamp has sub-nanosecond precision: {token!r}")
    try:
        return int(integral)
    except (OverflowError, ValueError) as error:
        raise ExportError(f"invalid TUM timestamp {token!r}") from error


def _parse_timed_paths(text: str, *, label: str) -> list[TimedPath]:
    result: list[TimedPath] = []
    timestamps: set[int] = set()
    paths: set[str] = set()
    for fields in _data_lines(text):
        if len(fields) != 2:
            raise ExportError(f"{label} rows must contain timestamp and path")
        timestamp = _timestamp_ns(fields[0])
        path = _safe_relative_path(fields[1], label=f"{label} payload path")
        if timestamp in timestamps or path in paths:
            raise ExportError(f"{label} contains duplicate timestamps or paths")
        timestamps.add(timestamp)
        paths.add(path)
        result.append(TimedPath(timestamp, fields[0], path))
    if not result:
        raise ExportError(f"{label} is empty")
    return sorted(result, key=lambda item: item.timestamp_ns)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ExportError("TUM ground-truth quaternion norm differs from one by more than 1e-3")
    return quaternion / norm


def _parse_timed_poses(text: str) -> list[TimedPose]:
    result: list[TimedPose] = []
    timestamps: set[int] = set()
    for fields in _data_lines(text):
        if len(fields) != 8:
            raise ExportError("TUM groundtruth rows must contain timestamp plus seven pose values")
        timestamp = _timestamp_ns(fields[0])
        if timestamp in timestamps:
            raise ExportError("TUM groundtruth contains a duplicate timestamp")
        timestamps.add(timestamp)
        try:
            numbers = np.asarray([float(Decimal(item)) for item in fields[1:]], dtype=np.float64)
        except Exception as error:
            raise ExportError("TUM groundtruth contains an invalid pose number") from error
        if not np.isfinite(numbers).all():
            raise ExportError("TUM groundtruth pose values must be finite")
        result.append(
            TimedPose(
                timestamp,
                fields[0],
                numbers[:3],
                _normalize_quaternion(numbers[3:]),
            )
        )
    if not result:
        raise ExportError("TUM groundtruth is empty")
    return sorted(result, key=lambda item: item.timestamp_ns)


def _associate_paths(
    first: list[TimedPath], second: list[TimedPath]
) -> list[tuple[TimedPath, TimedPath]]:
    candidates = [
        (
            abs(left.timestamp_ns - right.timestamp_ns),
            left.timestamp_ns,
            right.timestamp_ns,
            left,
            right,
        )
        for left in first
        for right in second
        if abs(left.timestamp_ns - right.timestamp_ns) < TUM_ASSOCIATION_MAX_NS
    ]
    candidates.sort(key=lambda item: item[:3])
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[TimedPath, TimedPath]] = []
    for _delta, first_ns, second_ns, left, right in candidates:
        if first_ns in used_first or second_ns in used_second:
            continue
        used_first.add(first_ns)
        used_second.add(second_ns)
        matches.append((left, right))
    return sorted(matches, key=lambda item: item[0].timestamp_ns)


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = math.acos(dot)
    result = (
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * q0
        + math.sin(fraction * theta) / math.sin(theta) * q1
    )
    return _normalize_quaternion(result)


def _interpolate_pose(poses: list[TimedPose], timestamp_ns: int) -> TimedPose | None:
    timestamps = [pose.timestamp_ns for pose in poses]
    position = bisect.bisect_left(timestamps, timestamp_ns)
    token = str(Decimal(timestamp_ns) / Decimal(1_000_000_000))
    if position < len(poses) and poses[position].timestamp_ns == timestamp_ns:
        pose = poses[position]
        return TimedPose(timestamp_ns, token, pose.center.copy(), pose.quaternion_xyzw.copy())
    if position == 0 or position == len(poses):
        return None
    lower = poses[position - 1]
    upper = poses[position]
    span = upper.timestamp_ns - lower.timestamp_ns
    if (
        not lower.timestamp_ns < timestamp_ns < upper.timestamp_ns
        or span > TUM_POSE_INTERPOLATION_MAX_NS
    ):
        return None
    fraction = (timestamp_ns - lower.timestamp_ns) / span
    return TimedPose(
        timestamp_ns,
        token,
        lower.center + fraction * (upper.center - lower.center),
        _slerp(lower.quaternion_xyzw, upper.quaternion_xyzw, fraction),
    )


def _associate_frames(
    rgb: list[TimedPath], depth: list[TimedPath], poses: list[TimedPose]
) -> list[AssociatedFrame]:
    frames: list[AssociatedFrame] = []
    for rgb_item, depth_item in _associate_paths(rgb, depth):
        pose = _interpolate_pose(poses, depth_item.timestamp_ns)
        if pose is not None:
            frames.append(
                AssociatedFrame(
                    rgb_item,
                    depth_item,
                    pose,
                    abs(rgb_item.timestamp_ns - depth_item.timestamp_ns),
                )
            )
    if not frames:
        raise ExportError("no TUM RGB/depth/pose triples survived association")
    return frames


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(_normalize_quaternion(left), _normalize_quaternion(right))))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def _half_up_uniform_indices(population: int, count: int) -> list[int]:
    if isinstance(population, bool) or not isinstance(population, int) or population < count:
        raise ExportError(f"cannot select {count} items from population {population}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ExportError("uniform selection count must be at least two")
    result: list[int] = []
    denominator = count - 1
    for index in range(count):
        numerator = index * (population - 1)
        quotient, remainder = divmod(numerator, denominator)
        result.append(quotient + int(2 * remainder >= denominator))
    if result[0] != 0 or result[-1] != population - 1:
        raise ExportError("half-up selection lost an endpoint")
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ExportError("half-up selection indices are not strictly increasing")
    return result


def _select_pose_keyframes(
    frames: list[AssociatedFrame],
) -> tuple[list[AssociatedFrame], list[int]]:
    keyframes = [frames[0]]
    for frame in frames[1:]:
        previous = keyframes[-1]
        translation = float(np.linalg.norm(frame.pose.center - previous.pose.center))
        rotation = _rotation_distance_deg(frame.pose.quaternion_xyzw, previous.pose.quaternion_xyzw)
        if translation >= TUM_MIN_TRANSLATION_M or rotation >= TUM_MIN_ROTATION_DEG:
            keyframes.append(frame)
    indices = _half_up_uniform_indices(len(keyframes), VIEW_COUNT)
    return [keyframes[index] for index in indices], indices


def _quaternion_to_rotation_c2w(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_from_pose(pose: TimedPose) -> dict[str, Any]:
    rotation = _quaternion_to_rotation_c2w(pose.quaternion_xyzw).T
    translation = -rotation @ pose.center
    if not np.allclose(rotation @ pose.center + translation, 0.0, atol=1e-10, rtol=0.0):
        raise ExportError("TUM camera center/extrinsic invariant failed")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise ExportError("TUM camera rotation determinant invariant failed")
    return {
        "fx": TUM_FX,
        "fy": TUM_FY,
        "cx": TUM_CX,
        "cy": TUM_CY,
        "width": TUM_WIDTH,
        "height": TUM_HEIGHT,
        "R": rotation.reshape(-1).tolist(),
        "t": translation.tolist(),
    }


class SafeTumArchive:
    """Read one TUM tgz through an audited, extraction-free member table."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._archive: tarfile.TarFile | None = None
        self._members: dict[str, tarfile.TarInfo] = {}
        self.prefix = ""

    def __enter__(self) -> SafeTumArchive:
        try:
            self._archive = tarfile.open(self.path, mode="r:gz")
        except (OSError, tarfile.TarError) as error:
            raise ExportError(f"cannot open TUM archive {self.path}") from error
        try:
            for member in self._archive.getmembers():
                name = _safe_relative_path(member.name)
                if name in self._members:
                    raise ExportError(f"TUM archive contains duplicate member {name}")
                if member.issym() or member.islnk():
                    raise ExportError(f"links are forbidden in TUM archive: {name}")
                if not member.isfile() and not member.isdir():
                    raise ExportError(f"special members are forbidden in TUM archive: {name}")
                self._members[name] = member
            roots = {
                name[: -len("/rgb.txt")]
                for name, member in self._members.items()
                if member.isfile() and name.endswith("/rgb.txt")
            }
            if len(roots) != 1:
                raise ExportError("TUM archive must contain exactly one rooted rgb.txt")
            self.prefix = next(iter(roots))
            for relative in ("rgb.txt", "depth.txt", "groundtruth.txt"):
                self.full_name(relative)
            return self
        except Exception:
            self._archive.close()
            self._archive = None
            self._members.clear()
            self.prefix = ""
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._archive is not None:
            self._archive.close()

    def full_name(self, relative: str) -> str:
        safe = _safe_relative_path(relative)
        full = f"{self.prefix}/{safe}"
        member = self._members.get(full)
        if member is None or not member.isfile():
            raise ExportError(f"TUM archive is missing regular member {safe}")
        return full

    def read(self, relative: str, *, max_bytes: int = 67_108_864) -> bytes:
        if self._archive is None:
            raise RuntimeError("TUM archive is not open")
        full = self.full_name(relative)
        member = self._members[full]
        if member.size > max_bytes:
            raise ExportError(f"TUM archive member exceeds its byte cap: {relative}")
        stream = self._archive.extractfile(member)
        if stream is None:
            raise ExportError(f"cannot read TUM archive member {relative}")
        payload = stream.read(max_bytes + 1)
        if len(payload) != member.size or len(payload) > max_bytes:
            raise ExportError(f"TUM archive member size differs while reading: {relative}")
        return payload

    def read_text(self, relative: str) -> str:
        if relative not in {"rgb.txt", "depth.txt", "groundtruth.txt"}:
            raise ExportError("only TUM metadata manifests may be read as text")
        try:
            return self.read(relative, max_bytes=16_777_216).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ExportError(f"TUM metadata {relative} is not UTF-8") from error

    def descriptor(self, relative: str) -> dict[str, Any]:
        payload = self.read(relative)
        return {
            "kind": "tar_member",
            "member": self.full_name(relative),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def read_bound(self, reference: Mapping[str, Any], *, label: str) -> bytes:
        """Read and verify one adapter-bound member from this open archive."""
        record = _exact(dict(reference), _TAR_REFERENCE_KEYS, label=label)
        if record["kind"] != "tar_member" or record["source_id"] != "official_archive":
            raise ExportError(f"{label} is not an official TUM member reference")
        member = _safe_relative_path(record["member"], label=f"{label}.member")
        prefix = f"{self.prefix}/"
        if not member.startswith(prefix):
            raise ExportError(f"{label} lies outside the bound TUM archive root")
        payload = self.read(member[len(prefix) :])
        _verify_member_payload(record, payload, label=label)
        return payload


def _load_capture(
    portfolio_path: str | Path, capture_id: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(portfolio_path).expanduser().resolve()
    portfolio = load_json_object(path, label="BENCH-019 capture portfolio")
    validate_capture_portfolio(portfolio, verify_files=False)
    capture = next((item for item in portfolio["captures"] if item["id"] == capture_id), None)
    if capture is None:
        raise ExportError(f"portfolio has no capture {capture_id}")
    return path, portfolio, capture


def _adapter_payload(
    *,
    portfolio_path: Path,
    capture: dict[str, Any],
    selection_policy: dict[str, Any],
    mask_policy: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    adapter = {
        "schema": ADAPTER_SCHEMA,
        "state": "ready_development",
        "capture_id": capture["id"],
        "role": capture["role"],
        "source_kind": capture["source_kind"],
        "portfolio": describe_artifact(portfolio_path),
        "source_artifacts": capture["source_artifacts"],
        "source_digest": capture["source_digest"],
        "selection_policy": selection_policy,
        "mask_policy": mask_policy,
        "views": views,
    }
    adapter["semantic_digest"] = _canonical_digest(adapter)
    # Each builder has already opened and verified every source it consumes.  Keep this call
    # structural; public source verification below performs one deterministic rebuild instead of
    # redundantly hashing every (potentially multi-hundred-MB) archive before that rebuild.
    validate_source_adapter(adapter, verify_sources=False, _replay=False)
    return adapter


def _camera_record(value: object, *, label: str) -> dict[str, Any]:
    record = _exact(value, _CAMERA_KEYS, label=label)
    for name in ("fx", "fy", "cx", "cy"):
        number = _finite(record[name], label=f"{label}.{name}")
        if name in {"fx", "fy"} and number <= 0.0:
            raise ExportError(f"{label}.{name} must be positive")
    width = _nonnegative_integer(record["width"], label=f"{label}.width")
    height = _nonnegative_integer(record["height"], label=f"{label}.height")
    if width <= 0 or height <= 0:
        raise ExportError(f"{label} dimensions must be positive")
    for name, count in (("R", 9), ("t", 3)):
        values = record[name]
        if not isinstance(values, list) or len(values) != count:
            raise ExportError(f"{label}.{name} must contain {count} values")
        for index, item in enumerate(values):
            _finite(item, label=f"{label}.{name}[{index}]")
    rotation = np.asarray(record["R"], dtype=np.float64).reshape(3, 3)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0.0):
        raise ExportError(f"{label}.R is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ExportError(f"{label}.R determinant differs from one")
    return record


def _calibration_camera(
    record: Mapping[str, Any], *, source_width: int, source_height: int, view_id: str
) -> tuple[dict[str, Any], list[float]]:
    try:
        intrinsics = record["intrinsics"]
        resolution = intrinsics["resolution"]
        calibration_width, calibration_height = int(resolution[0]), int(resolution[1])
        matrix = intrinsics["camera_matrix"]
        view = np.asarray(record["extrinsics"]["view_matrix"], dtype=np.float64).reshape(4, 4)
        distortion = [float(item) for item in intrinsics.get("distortion_coefficients", [])]
    except (KeyError, TypeError, ValueError) as error:
        raise ExportError(f"calibration record for {view_id} is invalid") from error
    if calibration_width <= 0 or calibration_height <= 0:
        raise ExportError(f"calibration record for {view_id} has invalid resolution")
    if len(matrix) != 9 or not np.isfinite(view).all() or not np.isfinite(distortion).all():
        raise ExportError(f"calibration record for {view_id} contains invalid numbers")
    if not np.allclose(view[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        raise ExportError(f"calibration record for {view_id} has invalid homogeneous row")
    sx = source_width / calibration_width
    sy = source_height / calibration_height
    camera = {
        "fx": float(matrix[0]) * sx,
        "fy": float(matrix[4]) * sy,
        "cx": (float(matrix[2]) + 0.5) * sx,
        "cy": (float(matrix[5]) + 0.5) * sy,
        "width": source_width,
        "height": source_height,
        "R": view[:3, :3].reshape(-1).tolist(),
        "t": view[:3, 3].tolist(),
    }
    _camera_record(camera, label=f"calibration camera {view_id}")
    return camera, distortion


def build_calibrated_adapter(
    portfolio_path: str | Path,
    *,
    capture_id: str = "janelle_stage_fabric",
) -> dict[str, Any]:
    """Bind one development calibrated capture with source-backed masks.

    The function intentionally rejects Karate because its portfolio record has no source mask
    policy.  No full-frame or field-derived fallback exists.
    """
    path, _portfolio, capture = _load_capture(portfolio_path, capture_id)
    if capture["source_kind"] != "calibrated_multiview":
        raise ExportError(f"capture {capture_id} is not a calibrated multiview source")
    if capture["mask_policy_state"] != "source_binary_masks_bound":
        raise ExportError(f"capture {capture_id} has no source-backed mask policy")
    if capture["role"] != "development":
        raise ExportError(f"capture {capture_id} is confirmation-sealed")
    views = capture["view_ids"]
    if len(views) != VIEW_COUNT:
        raise ExportError(f"capture {capture_id} must contain exactly {VIEW_COUNT} views")
    sources = {item["id"]: item["artifact"] for item in capture["source_artifacts"]}
    calibration_descriptor = sources.get("calibration")
    if (
        calibration_descriptor is None
        or describe_artifact(calibration_descriptor["path"]) != calibration_descriptor
    ):
        raise ExportError("calibrated source binding differs from its calibration file")
    calibration = load_json_object(calibration_descriptor["path"], label="calibration")
    records = calibration.get("cameras")
    if not isinstance(records, list):
        raise ExportError("calibration cameras must be a list")
    indexed = {
        str(record.get("camera_id", "")).upper(): record
        for record in records
        if isinstance(record, dict)
    }
    if len(indexed) != len(records):
        raise ExportError("calibration camera IDs must be unique non-empty strings")

    adapter_views: list[dict[str, Any]] = []
    for ordinal, view_id in enumerate(views):
        rgb_id = f"rgb_{view_id}"
        mask_id = f"mask_{view_id}"
        if rgb_id not in sources or mask_id not in sources:
            raise ExportError(f"capture {capture_id} is missing RGB/mask source {view_id}")
        for source_id in (rgb_id, mask_id):
            if describe_artifact(sources[source_id]["path"]) != sources[source_id]:
                raise ExportError(f"capture source {source_id} differs from its binding")
        with PILImage.open(sources[rgb_id]["path"]) as image:
            source_width, source_height = image.size
        with PILImage.open(sources[mask_id]["path"]) as mask_image:
            if mask_image.size != (source_width, source_height):
                raise ExportError(f"capture source {mask_id} dimensions differ from RGB")
        record = indexed.get(view_id.upper())
        if record is None:
            raise ExportError(f"calibration has no camera {view_id}")
        camera, distortion = _calibration_camera(
            record,
            source_width=source_width,
            source_height=source_height,
            view_id=view_id,
        )
        adapter_views.append(
            {
                "id": view_id,
                "ordinal": ordinal,
                "split": "heldout" if ordinal in HELDOUT_ORDINALS else "train",
                "rgb_source": {"kind": "portfolio_file", "source_id": rgb_id},
                "mask_source": {"kind": "portfolio_file", "source_id": mask_id},
                "camera": camera,
                "preprocessing": {
                    "rgb": "calibrated_bilinear_undistort",
                    "mask": "calibrated_nearest_undistort_threshold_gt_0.5",
                    "distortion_coefficients": distortion,
                },
                "source_metadata": {"kind": "calibrated_json", "camera_id": view_id},
            }
        )
    return _adapter_payload(
        portfolio_path=path,
        capture=capture,
        selection_policy={
            "kind": "ordered_portfolio_views",
            "count": VIEW_COUNT,
            "heldout_ordinals": list(HELDOUT_ORDINALS),
        },
        mask_policy={
            "kind": "source_binary_masks",
            "threshold": "greater_than_0.5_after_nearest_undistort",
            "source": "portfolio_bound_mask_files",
        },
        views=adapter_views,
    )


def build_tum_adapter(
    portfolio_path: str | Path,
    *,
    capture_id: str,
) -> dict[str, Any]:
    """Bind one development TUM RGB-D archive to the frozen 26-view recipe."""
    path, _portfolio, capture = _load_capture(portfolio_path, capture_id)
    if capture["role"] != "development":
        raise ExportError(f"capture {capture_id} is confirmation-sealed")
    if capture["source_kind"] != "tum_rgbd_archive":
        raise ExportError(f"capture {capture_id} is not a TUM RGB-D archive")
    if len(capture["source_artifacts"]) != 1:
        raise ExportError(f"capture {capture_id} must bind exactly one TUM archive")
    source = capture["source_artifacts"][0]
    if source["id"] != "official_archive":
        raise ExportError(f"capture {capture_id} has no official_archive binding")
    archive_descriptor = source["artifact"]
    if describe_artifact(archive_descriptor["path"]) != archive_descriptor:
        raise ExportError(f"capture {capture_id} archive differs from its binding")

    with SafeTumArchive(archive_descriptor["path"]) as archive:
        rgb = _parse_timed_paths(archive.read_text("rgb.txt"), label="TUM rgb.txt")
        depth = _parse_timed_paths(archive.read_text("depth.txt"), label="TUM depth.txt")
        poses = _parse_timed_poses(archive.read_text("groundtruth.txt"))
        associated = _associate_frames(rgb, depth, poses)
        selected, selected_indices = _select_pose_keyframes(associated)
        adapter_views: list[dict[str, Any]] = []
        for ordinal, frame in enumerate(selected):
            view_id = f"C{ordinal:04d}"
            rgb_source = archive.descriptor(frame.rgb.path)
            rgb_source["source_id"] = "official_archive"
            depth_source = archive.descriptor(frame.depth.path)
            depth_source["source_id"] = "official_archive"
            adapter_views.append(
                {
                    "id": view_id,
                    "ordinal": ordinal,
                    "split": "heldout" if ordinal in HELDOUT_ORDINALS else "train",
                    "rgb_source": rgb_source,
                    "mask_source": {
                        "kind": "registered_depth_validity",
                        "depth_source": depth_source,
                        "depth_divisor": TUM_DEPTH_DIVISOR,
                        "min_depth_m": TUM_MIN_DEPTH_M,
                        "max_depth_m": TUM_MAX_DEPTH_M,
                    },
                    "camera": _camera_from_pose(frame.pose),
                    "preprocessing": {
                        "rgb": "identity_registered_rgb_png",
                        "mask": "inclusive_registered_depth_range",
                        "distortion_coefficients": [0.0] * 5,
                    },
                    "source_metadata": {
                        "kind": "tum_rgbd_pose",
                        "rgb_timestamp_ns": frame.rgb.timestamp_ns,
                        "rgb_timestamp_token": frame.rgb.timestamp_token,
                        "depth_timestamp_ns": frame.depth.timestamp_ns,
                        "depth_timestamp_token": frame.depth.timestamp_token,
                        "rgb_depth_delta_ns": frame.rgb_depth_delta_ns,
                        "pose_center_m": frame.pose.center.tolist(),
                        "pose_quaternion_xyzw": frame.pose.quaternion_xyzw.tolist(),
                    },
                }
            )
    return _adapter_payload(
        portfolio_path=path,
        capture=capture,
        selection_policy={
            "kind": "tum_pose_keyframes_half_up_v1",
            "association_max_ns_strict": TUM_ASSOCIATION_MAX_NS,
            "pose_interpolation_max_ns": TUM_POSE_INTERPOLATION_MAX_NS,
            "translation_m_inclusive": TUM_MIN_TRANSLATION_M,
            "rotation_deg_inclusive": TUM_MIN_ROTATION_DEG,
            "associated_triples": len(associated),
            "pose_keyframes": max(selected_indices) + 1 if selected_indices else 0,
            "uniform_source_indices": selected_indices,
            "count": VIEW_COUNT,
            "heldout_ordinals": list(HELDOUT_ORDINALS),
            "rounding": "integer_half_up_endpoint_preserving",
        },
        mask_policy={
            "kind": "registered_depth_validity",
            "depth_divisor": TUM_DEPTH_DIVISOR,
            "min_depth_m": TUM_MIN_DEPTH_M,
            "max_depth_m": TUM_MAX_DEPTH_M,
            "source": "official_pre_registered_depth_png",
        },
        views=adapter_views,
    )


def _validate_reference(
    value: object,
    *,
    label: str,
    sources: Mapping[str, dict[str, Any]],
    expected_kind: Literal["portfolio_file", "tar_member"] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{label} must be an object")
    kind = value.get("kind")
    if kind == "portfolio_file":
        record = _exact(value, _FILE_REFERENCE_KEYS, label=label)
    elif kind == "tar_member":
        record = _exact(value, _TAR_REFERENCE_KEYS, label=label)
        _safe_relative_path(record["member"], label=f"{label}.member")
        _sha256(record["sha256"], label=f"{label}.sha256")
        _nonnegative_integer(record["bytes"], label=f"{label}.bytes")
    else:
        raise ExportError(f"{label}.kind is unsupported")
    if expected_kind is not None and kind != expected_kind:
        raise ExportError(f"{label}.kind must be {expected_kind}")
    source_id = _identifier(record["source_id"], label=f"{label}.source_id")
    if source_id not in sources:
        raise ExportError(f"{label} references unknown source artifact {source_id}")
    return record


def _validate_depth_mask(
    value: object, *, label: str, sources: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    record = _exact(value, _DEPTH_MASK_KEYS, label=label)
    if record["kind"] != "registered_depth_validity":
        raise ExportError(f"{label}.kind must be registered_depth_validity")
    _validate_reference(
        record["depth_source"],
        label=f"{label}.depth_source",
        sources=sources,
        expected_kind="tar_member",
    )
    divisor = _finite(record["depth_divisor"], label=f"{label}.depth_divisor")
    minimum = _finite(record["min_depth_m"], label=f"{label}.min_depth_m")
    maximum = _finite(record["max_depth_m"], label=f"{label}.max_depth_m")
    if divisor <= 0.0 or minimum < 0.0 or maximum <= minimum:
        raise ExportError(f"{label} depth range is invalid")
    return record


def validate_source_adapter(
    value: Mapping[str, Any],
    *,
    verify_sources: bool = False,
    _replay: bool = True,
) -> dict[str, int]:
    """Validate one development-ready adapter and optionally rehash all bound files."""
    adapter = _exact(dict(value), _TOP_KEYS, label="BENCH-019 source adapter")
    if adapter["schema"] != ADAPTER_SCHEMA or adapter["state"] != "ready_development":
        raise ExportError("adapter schema/state is unsupported")
    _identifier(adapter["capture_id"], label="adapter capture_id")
    if adapter["role"] != "development":
        raise ExportError("a ready adapter must remain development-only")
    if adapter["source_kind"] not in {"calibrated_multiview", "tum_rgbd_archive"}:
        raise ExportError("adapter source_kind is unsupported")
    verify_directly = verify_sources and not _replay
    _artifact(adapter["portfolio"], label="adapter portfolio", verify_file=verify_directly)
    sources, indexed_sources = _source_inventory(
        adapter["source_artifacts"], verify_files=verify_directly
    )
    if adapter["source_digest"] != source_digest(sources):
        raise ExportError("adapter source digest differs")
    payload = dict(adapter)
    recorded = _sha256(payload.pop("semantic_digest"), label="adapter semantic_digest")
    if recorded != _canonical_digest(payload):
        raise ExportError("adapter semantic digest differs")

    selection = adapter["selection_policy"]
    mask_policy = adapter["mask_policy"]
    source_kind = adapter["source_kind"]
    if source_kind == "calibrated_multiview":
        selected = _exact(selection, _STAGE_SELECTION_KEYS, label="calibrated selection policy")
        if selected["kind"] != "ordered_portfolio_views":
            raise ExportError("calibrated selection kind differs")
        mask = _exact(mask_policy, _STAGE_MASK_POLICY_KEYS, label="calibrated mask policy")
        if (
            mask["kind"] != "source_binary_masks"
            or mask["threshold"] != "greater_than_0.5_after_nearest_undistort"
            or mask["source"] != "portfolio_bound_mask_files"
        ):
            raise ExportError("calibrated mask policy differs")
        expected_reference_kind: Literal["portfolio_file", "tar_member"] = "portfolio_file"
    else:
        if set(indexed_sources) != {"official_archive"}:
            raise ExportError("TUM adapter must bind exactly the official_archive source")
        selected = _exact(selection, _TUM_SELECTION_KEYS, label="TUM selection policy")
        if (
            selected["kind"] != "tum_pose_keyframes_half_up_v1"
            or selected["association_max_ns_strict"] != TUM_ASSOCIATION_MAX_NS
            or selected["pose_interpolation_max_ns"] != TUM_POSE_INTERPOLATION_MAX_NS
            or selected["translation_m_inclusive"] != TUM_MIN_TRANSLATION_M
            or selected["rotation_deg_inclusive"] != TUM_MIN_ROTATION_DEG
            or selected["rounding"] != "integer_half_up_endpoint_preserving"
        ):
            raise ExportError("TUM selection policy differs")
        associated_triples = _nonnegative_integer(
            selected["associated_triples"], label="associated_triples"
        )
        pose_keyframes = _nonnegative_integer(selected["pose_keyframes"], label="pose_keyframes")
        if pose_keyframes < VIEW_COUNT or associated_triples < pose_keyframes:
            raise ExportError("TUM associated/keyframe counts are inconsistent")
        indices = selected["uniform_source_indices"]
        if (
            not isinstance(indices, list)
            or len(indices) != VIEW_COUNT
            or any(isinstance(item, bool) or not isinstance(item, int) for item in indices)
            or any(right <= left for left, right in zip(indices, indices[1:], strict=False))
            or indices[0] != 0
            or indices != _half_up_uniform_indices(pose_keyframes, VIEW_COUNT)
        ):
            raise ExportError("TUM uniform source indices are invalid")
        mask = _exact(mask_policy, _TUM_MASK_POLICY_KEYS, label="TUM mask policy")
        if (
            mask["kind"] != "registered_depth_validity"
            or mask["depth_divisor"] != TUM_DEPTH_DIVISOR
            or mask["min_depth_m"] != TUM_MIN_DEPTH_M
            or mask["max_depth_m"] != TUM_MAX_DEPTH_M
            or mask["source"] != "official_pre_registered_depth_png"
        ):
            raise ExportError("TUM mask policy differs")
        expected_reference_kind = "tar_member"
    if selected["count"] != VIEW_COUNT or selected["heldout_ordinals"] != list(HELDOUT_ORDINALS):
        raise ExportError("adapter count/heldout policy differs")

    views = adapter["views"]
    if not isinstance(views, list) or len(views) != VIEW_COUNT:
        raise ExportError(f"adapter must contain exactly {VIEW_COUNT} views")
    identifiers: set[str] = set()
    referenced_sources: set[str] = set()
    rgb_members: set[str] = set()
    depth_members: set[str] = set()
    previous_rgb_timestamp = -1
    previous_depth_timestamp = -1
    train = 0
    heldout = 0
    for ordinal, item in enumerate(views):
        view = _exact(item, _VIEW_KEYS, label=f"adapter views[{ordinal}]")
        view_id = _identifier(view["id"], label=f"adapter views[{ordinal}].id")
        if view_id in identifiers or view["ordinal"] != ordinal:
            raise ExportError("adapter view IDs/ordinals must be unique and ordered")
        identifiers.add(view_id)
        expected_split = "heldout" if ordinal in HELDOUT_ORDINALS else "train"
        if view["split"] != expected_split:
            raise ExportError(f"adapter view {view_id} split differs")
        if expected_split == "train":
            train += 1
        else:
            heldout += 1
        rgb_source = _validate_reference(
            view["rgb_source"],
            label=f"adapter view {view_id}.rgb_source",
            sources=indexed_sources,
            expected_kind=expected_reference_kind,
        )
        referenced_sources.add(rgb_source["source_id"])
        if source_kind == "calibrated_multiview":
            mask_source = _validate_reference(
                view["mask_source"],
                label=f"adapter view {view_id}.mask_source",
                sources=indexed_sources,
                expected_kind="portfolio_file",
            )
            referenced_sources.add(mask_source["source_id"])
            if rgb_source["source_id"] != f"rgb_{view_id}":
                raise ExportError(f"adapter view {view_id} RGB source identity differs")
            if mask_source["source_id"] != f"mask_{view_id}":
                raise ExportError(f"adapter view {view_id} mask source identity differs")
        else:
            depth_mask = _validate_depth_mask(
                view["mask_source"],
                label=f"adapter view {view_id}.mask_source",
                sources=indexed_sources,
            )
            if (
                depth_mask["depth_divisor"] != TUM_DEPTH_DIVISOR
                or depth_mask["min_depth_m"] != TUM_MIN_DEPTH_M
                or depth_mask["max_depth_m"] != TUM_MAX_DEPTH_M
            ):
                raise ExportError(f"adapter view {view_id} mask policy differs")
            referenced_sources.add(depth_mask["depth_source"]["source_id"])
            if view_id != f"C{ordinal:04d}":
                raise ExportError(f"adapter view {view_id} TUM identity differs")
            rgb_member = rgb_source["member"]
            depth_member = depth_mask["depth_source"]["member"]
            if rgb_member in rgb_members or depth_member in depth_members:
                raise ExportError("TUM adapter reuses an RGB or depth member")
            rgb_members.add(rgb_member)
            depth_members.add(depth_member)
        camera = _camera_record(view["camera"], label=f"adapter view {view_id}.camera")
        preprocessing = _exact(
            view["preprocessing"],
            _PREPROCESSING_KEYS,
            label=f"adapter view {view_id}.preprocessing",
        )
        distortion = preprocessing["distortion_coefficients"]
        if not isinstance(distortion, list) or len(distortion) > 8:
            raise ExportError(f"adapter view {view_id} distortion is invalid")
        for index, number in enumerate(distortion):
            _finite(number, label=f"adapter view {view_id} distortion[{index}]")
        metadata = view["source_metadata"]
        if source_kind == "calibrated_multiview":
            if (
                preprocessing["rgb"] != "calibrated_bilinear_undistort"
                or preprocessing["mask"] != "calibrated_nearest_undistort_threshold_gt_0.5"
            ):
                raise ExportError(f"adapter view {view_id} preprocessing differs")
            metadata = _exact(metadata, _STAGE_METADATA_KEYS, label=f"view {view_id} metadata")
            if metadata != {"kind": "calibrated_json", "camera_id": view_id}:
                raise ExportError(f"adapter view {view_id} calibration metadata differs")
        else:
            if (
                preprocessing["rgb"] != "identity_registered_rgb_png"
                or preprocessing["mask"] != "inclusive_registered_depth_range"
                or distortion != [0.0] * 5
            ):
                raise ExportError(f"adapter view {view_id} TUM preprocessing differs")
            metadata = _exact(metadata, _TUM_METADATA_KEYS, label=f"view {view_id} metadata")
            if metadata["kind"] != "tum_rgbd_pose":
                raise ExportError(f"adapter view {view_id} TUM metadata kind differs")
            rgb_timestamp = _nonnegative_integer(
                metadata["rgb_timestamp_ns"],
                label=f"adapter view {view_id} RGB timestamp",
            )
            depth_timestamp = _nonnegative_integer(
                metadata["depth_timestamp_ns"],
                label=f"adapter view {view_id} depth timestamp",
            )
            for name, timestamp in (
                ("rgb_timestamp_token", rgb_timestamp),
                ("depth_timestamp_token", depth_timestamp),
            ):
                token = metadata[name]
                if not isinstance(token, str) or _timestamp_ns(token) != timestamp:
                    raise ExportError(f"adapter view {view_id} {name} differs")
            if (
                rgb_timestamp <= previous_rgb_timestamp
                or depth_timestamp <= previous_depth_timestamp
            ):
                raise ExportError("TUM adapter timestamps are not strictly increasing")
            previous_rgb_timestamp = rgb_timestamp
            previous_depth_timestamp = depth_timestamp
            delta = _nonnegative_integer(
                metadata["rgb_depth_delta_ns"], label=f"adapter view {view_id} RGB/depth delta"
            )
            if delta != abs(rgb_timestamp - depth_timestamp):
                raise ExportError(f"adapter view {view_id} RGB/depth delta differs")
            if delta >= TUM_ASSOCIATION_MAX_NS:
                raise ExportError(f"adapter view {view_id} violates strict association threshold")
            for name, count in (("pose_center_m", 3), ("pose_quaternion_xyzw", 4)):
                values = metadata[name]
                if not isinstance(values, list) or len(values) != count:
                    raise ExportError(f"adapter view {view_id} {name} is invalid")
                for index, number in enumerate(values):
                    _finite(number, label=f"adapter view {view_id} {name}[{index}]")
            _normalize_quaternion(np.asarray(metadata["pose_quaternion_xyzw"], dtype=np.float64))
            expected_camera = _camera_from_pose(
                TimedPose(
                    depth_timestamp,
                    metadata["depth_timestamp_token"],
                    np.asarray(metadata["pose_center_m"], dtype=np.float64),
                    np.asarray(metadata["pose_quaternion_xyzw"], dtype=np.float64),
                )
            )
            if camera != expected_camera:
                raise ExportError(f"adapter view {view_id} camera differs from its TUM pose")
    expected_sources = referenced_sources | (
        {"calibration"} if source_kind == "calibrated_multiview" else set()
    )
    if set(indexed_sources) != expected_sources:
        raise ExportError("adapter source inventory contains missing or unreferenced artifacts")
    summary = {"views": len(views), "train_views": train, "heldout_views": heldout}
    if verify_sources and _replay:
        portfolio_path = adapter["portfolio"]["path"]
        if source_kind == "calibrated_multiview":
            expected = build_calibrated_adapter(
                portfolio_path,
                capture_id=adapter["capture_id"],
            )
        else:
            expected = build_tum_adapter(
                portfolio_path,
                capture_id=adapter["capture_id"],
            )
        if expected != adapter:
            raise ExportError("adapter differs from deterministic source replay")
    return summary


def _source_path(adapter: Mapping[str, Any], source_id: str) -> Path:
    source = next((item for item in adapter["source_artifacts"] if item["id"] == source_id), None)
    if source is None:
        raise ExportError(f"adapter has no source artifact {source_id}")
    return Path(source["artifact"]["path"])


def _verify_member_payload(reference: Mapping[str, Any], payload: bytes, *, label: str) -> None:
    if (
        len(payload) != reference["bytes"]
        or hashlib.sha256(payload).hexdigest() != reference["sha256"]
    ):
        raise ExportError(f"{label} differs from its adapter binding")


def _write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _write_json_new(path: Path, value: object) -> None:
    _write_bytes_new(path, canonical_json(value) + b"\n")


def write_source_adapter(value: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Validate and exclusively publish one canonical source-adapter manifest."""
    adapter = dict(value)
    summary = validate_source_adapter(adapter, verify_sources=True)
    output = Path(output_path).expanduser().resolve()
    _write_json_new(output, adapter)
    return {"artifact": describe_artifact(output), **summary}


def _mask_png(depth_payload: bytes, mask_source: Mapping[str, Any]) -> bytes:
    try:
        with PILImage.open(io.BytesIO(depth_payload)) as image:
            image.load()
            depth = np.asarray(image)
    except Exception as error:
        raise ExportError("cannot decode selected TUM depth PNG") from error
    if depth.shape != (TUM_HEIGHT, TUM_WIDTH) or not np.issubdtype(depth.dtype, np.integer):
        raise ExportError("TUM depth PNG must be a 640x480 integer image")
    distance = depth.astype(np.float64) / float(mask_source["depth_divisor"])
    mask = (
        np.isfinite(distance)
        & (distance >= float(mask_source["min_depth_m"]))
        & (distance <= float(mask_source["max_depth_m"]))
    )
    output = io.BytesIO()
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(output, format="PNG")
    return output.getvalue()


def derive_registered_depth_mask_png(
    depth_payload: bytes,
    mask_source: Mapping[str, Any],
) -> bytes:
    """Derive the canonical inclusive TUM validity-mask PNG bound by an adapter view."""
    return _mask_png(depth_payload, mask_source)


def _calibration_payload(adapter: Mapping[str, Any]) -> dict[str, Any]:
    cameras = []
    for view in adapter["views"]:
        camera = view["camera"]
        matrix = [
            camera["fx"],
            0.0,
            camera["cx"] - 0.5,
            0.0,
            camera["fy"],
            camera["cy"] - 0.5,
            0.0,
            0.0,
            1.0,
        ]
        extrinsic = [
            *camera["R"][0:3],
            camera["t"][0],
            *camera["R"][3:6],
            camera["t"][1],
            *camera["R"][6:9],
            camera["t"][2],
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        cameras.append(
            {
                "camera_id": view["id"],
                "extrinsics": {"view_matrix": extrinsic},
                "intrinsics": {
                    "camera_matrix": matrix,
                    "distortion_coefficients": view["preprocessing"]["distortion_coefficients"],
                    "resolution": [camera["width"], camera["height"]],
                },
            }
        )
    return {"cameras": cameras}


def materialized_calibration_payload(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Return the calibrated-loader JSON emitted for a validated TUM adapter."""
    validate_source_adapter(adapter, verify_sources=False)
    if adapter["source_kind"] != "tum_rgbd_archive":
        raise ExportError("materialized calibration is defined only for TUM adapters")
    return _calibration_payload(adapter)


def materialize_tum_adapter(
    adapter_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Materialize one ready TUM adapter into a new calibrated-capture directory.

    Confirmation is denied by the adapter schema.  A later prospectively reviewed task must make
    an explicit contract change before opening confirmation payloads.
    """
    adapter_file = Path(adapter_path).expanduser().resolve()
    adapter = load_json_object(adapter_file, label="BENCH-019 source adapter")
    validate_source_adapter(adapter, verify_sources=True)
    if adapter["source_kind"] != "tum_rgbd_archive":
        raise ExportError("materialization requires a TUM source adapter")
    if adapter["role"] != "development":
        raise ExportError("confirmation materialization is not authorized")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to materialize into existing path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to materialize into existing path: {output}") from error
    reserved_output = True
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        archive_path = _source_path(adapter, "official_archive")
        outputs: list[dict[str, Any]] = []
        with SafeTumArchive(archive_path) as archive:
            for view in adapter["views"]:
                rgb_reference = view["rgb_source"]
                rgb_payload = archive.read_bound(
                    rgb_reference,
                    label=f"view {view['id']} RGB",
                )
                rgb_path = temporary / "rgb" / f"{view['id']}.png"
                _write_bytes_new(rgb_path, rgb_payload)
                rgb_artifact = describe_artifact(rgb_path)
                rgb_artifact["path"] = str(output / "rgb" / f"{view['id']}.png")
                outputs.append({"id": f"rgb_{view['id']}", "artifact": rgb_artifact})

                mask_source = view["mask_source"]
                depth_reference = mask_source["depth_source"]
                depth_payload = archive.read_bound(
                    depth_reference,
                    label=f"view {view['id']} depth",
                )
                mask_path = temporary / "mask" / f"mask_{view['id']}.png"
                _write_bytes_new(mask_path, _mask_png(depth_payload, mask_source))
                mask_artifact = describe_artifact(mask_path)
                mask_artifact["path"] = str(output / "mask" / f"mask_{view['id']}.png")
                outputs.append({"id": f"mask_{view['id']}", "artifact": mask_artifact})

        calibration_path = temporary / "calibration_dome.json"
        _write_json_new(calibration_path, _calibration_payload(adapter))
        calibration_artifact = describe_artifact(calibration_path)
        calibration_artifact["path"] = str(output / "calibration_dome.json")
        outputs.insert(0, {"id": "calibration", "artifact": calibration_artifact})
        adapter_copy = temporary / "source_adapter.json"
        _write_bytes_new(adapter_copy, adapter_file.read_bytes())
        adapter_artifact = describe_artifact(adapter_copy)
        adapter_artifact["path"] = str(output / "source_adapter.json")
        outputs.insert(0, {"id": "source_adapter", "artifact": adapter_artifact})
        receipt = {
            "schema": MATERIALIZATION_SCHEMA,
            "adapter": adapter_artifact,
            "capture_id": adapter["capture_id"],
            "role": adapter["role"],
            "outputs": outputs,
        }
        receipt["semantic_digest"] = _canonical_digest(receipt)
        _write_json_new(temporary / "materialization_receipt.json", receipt)
        children = sorted(
            temporary.iterdir(),
            key=lambda item: (item.name == "materialization_receipt.json", item.name),
        )
        for child in children:
            os.rename(child, output / child.name)
        temporary.rmdir()
        temporary = None
        summary = validate_materialization(
            output / "materialization_receipt.json",
            verify_files=True,
        )
        reserved_output = False
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        if reserved_output and output.exists():
            shutil.rmtree(output)
    return summary


def validate_materialization(
    receipt_path: str | Path, *, verify_files: bool = False
) -> dict[str, Any]:
    """Validate a materialized TUM adapter receipt and every declared output."""
    path = Path(receipt_path).expanduser().resolve()
    receipt = _exact(
        load_json_object(path, label="BENCH-019 materialization receipt"),
        _MATERIALIZATION_KEYS,
        label="BENCH-019 materialization receipt",
    )
    if receipt["schema"] != MATERIALIZATION_SCHEMA:
        raise ExportError("materialization receipt schema is unsupported")
    _identifier(receipt["capture_id"], label="materialization capture_id")
    if receipt["role"] not in {"development", "confirmation"}:
        raise ExportError("materialization role is invalid")
    root = path.parent
    adapter_artifact = _artifact(
        receipt["adapter"], label="materialization adapter", verify_file=verify_files
    )
    if Path(adapter_artifact["path"]).resolve() != root / "source_adapter.json":
        raise ExportError("materialization adapter path is not canonical")
    adapter = load_json_object(adapter_artifact["path"], label="materialized source adapter")
    validate_source_adapter(adapter, verify_sources=verify_files)
    if (
        adapter["capture_id"] != receipt["capture_id"]
        or adapter["role"] != receipt["role"]
        or adapter["source_kind"] != "tum_rgbd_archive"
    ):
        raise ExportError("materialization receipt differs from its source adapter")
    payload = dict(receipt)
    recorded = _sha256(payload.pop("semantic_digest"), label="materialization semantic_digest")
    if recorded != _canonical_digest(payload):
        raise ExportError("materialization semantic digest differs")
    outputs = receipt["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ExportError("materialization outputs must be a non-empty list")
    ids: set[str] = set()
    expected_paths = {
        "source_adapter": root / "source_adapter.json",
        "calibration": root / "calibration_dome.json",
    }
    expected_paths.update(
        {f"rgb_C{index:04d}": root / "rgb" / f"C{index:04d}.png" for index in range(VIEW_COUNT)}
    )
    expected_paths.update(
        {
            f"mask_C{index:04d}": root / "mask" / f"mask_C{index:04d}.png"
            for index in range(VIEW_COUNT)
        }
    )
    for index, item in enumerate(outputs):
        output = _exact(item, _MATERIALIZED_OUTPUT_KEYS, label=f"materialization outputs[{index}]")
        output_id = _identifier(output["id"], label=f"materialization outputs[{index}].id")
        if output_id in ids:
            raise ExportError(f"duplicate materialized output {output_id}")
        ids.add(output_id)
        descriptor = _artifact(
            output["artifact"],
            label=f"materialized output {output_id}",
            verify_file=verify_files,
        )
        expected_path = expected_paths.get(output_id)
        if expected_path is None or Path(descriptor["path"]).resolve() != expected_path:
            raise ExportError(f"materialized output {output_id} path is not canonical")
    expected = {"source_adapter", "calibration"}
    expected.update(f"rgb_C{index:04d}" for index in range(VIEW_COUNT))
    expected.update(f"mask_C{index:04d}" for index in range(VIEW_COUNT))
    if ids != expected:
        raise ExportError("materialized output inventory differs from the 26-view contract")
    if verify_files:
        actual_files = {item.resolve() for item in root.rglob("*") if item.is_file()}
        declared_files = {path.resolve() for path in expected_paths.values()} | {path}
        if actual_files != declared_files:
            raise ExportError("materialized directory contains undeclared files")
    return {"capture_id": receipt["capture_id"], "role": receipt["role"], "outputs": len(ids)}


__all__ = [
    "ADAPTER_SCHEMA",
    "HELDOUT_ORDINALS",
    "MATERIALIZATION_SCHEMA",
    "SafeTumArchive",
    "TUM_ASSOCIATION_MAX_NS",
    "TUM_MAX_DEPTH_M",
    "TUM_MIN_DEPTH_M",
    "VIEW_COUNT",
    "build_calibrated_adapter",
    "build_tum_adapter",
    "derive_registered_depth_mask_png",
    "materialize_tum_adapter",
    "materialized_calibration_payload",
    "validate_materialization",
    "validate_source_adapter",
    "write_source_adapter",
]
