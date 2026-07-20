#!/usr/bin/env python3
"""Sealed Phase-A TUM RGB-D oriented-point validity audit.

The development phase calibrates the nine confirmatory thresholds on ``fr1/xyz``.
The confirmatory phase applies them once to ``fr1/desk``.  RGB and H-role PNGs are
never decoded; construction visibility uses T targets only.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import PIL
import torch
from PIL import Image as PILImage

from rtgs.core.camera import Camera
from rtgs.lift.surface import (
    OrientedPointPrediction,
    OrientedPointProvenance,
    canonicalize_oriented_point_prediction,
    estimate_registered_depth_normals,
)

PREREGISTRATION = Path("benchmarks/results/20260715_tum_rgbd_oriented_validity_PREREG.md")
ACQUISITION_MANIFEST = Path("benchmarks/results/20260715_tum_rgbd_ACQUISITION.json")
CONFIRMATORY_SEAL = Path(
    "benchmarks/results/20260715_tum_rgbd_oriented_validity_CONFIRMATORY_SEAL.json"
)
WIDTH = 640
HEIGHT = 480
FX = 525.0
FY = 525.0
CX = 320.0
CY = 240.0
MAX_ASSOCIATION_NS = 20_000_000
MIN_TRANSLATION_M = 0.08
MIN_ROTATION_DEG = 8.0
N_KEYFRAMES = 64
TARGET_COLUMNS = tuple(8 + 16 * index for index in range(40))
TARGET_ROWS = tuple(8 + 16 * index for index in range(30))
N_GRID_PER_VIEW = len(TARGET_COLUMNS) * len(TARGET_ROWS)

CONFIG: dict[str, Any] = {
    "association_max_ns_strict": MAX_ASSOCIATION_NS,
    "pose_interpolation_max_span_ns": MAX_ASSOCIATION_NS,
    "keyframes": {
        "translation_m_inclusive": MIN_TRANSLATION_M,
        "rotation_deg_inclusive": MIN_ROTATION_DEG,
        "count": N_KEYFRAMES,
        "half_up_subsample": True,
        "roles": {"H": "j%8==7", "V": "j%8==3", "T": "otherwise"},
    },
    "camera": {
        "width": WIDTH,
        "height": HEIGHT,
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
        "depth_divisor": 5000.0,
    },
    "target": {
        "columns": TARGET_COLUMNS,
        "rows": TARGET_ROWS,
        "pixel_offset": 2,
        "min_depth_m": 0.3,
        "max_depth_m": 5.0,
        "max_abs_depth_delta_m": 0.04,
        "max_relative_depth_delta": 0.02,
        "min_cross_norm": 1e-8,
        "min_abs_incidence": 0.20,
    },
    "visibility_depth_m": 0.020,
    "minimum_validation_views": 2,
    "free_space": {"absolute_m": 0.05, "relative": 0.03},
    "reduction": "cpu-float64-sorted-linear-quantile",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


CONFIG_ID = hashlib.sha256(_canonical_json(CONFIG)).hexdigest()
EXPERIMENT = "tum-rgbd-oriented-validity-phase-a"
METRIC_KEYS = frozenset({"A", "A_min", "S", "S_10", "R90", "D90", "C50", "C10", "F"})

SOURCES = {
    "development": {
        "sequence": "rgbd_dataset_freiburg1_xyz",
        "url": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"),
        "content_length": 448_204_271,
        "etag": '"1ab70def-4ae2a1dc2ae80"',
        "last_modified": "Fri, 30 Sep 2011 15:16:58 GMT",
        "sha256": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
    },
    "confirmatory": {
        "sequence": "rgbd_dataset_freiburg1_desk",
        "url": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"),
        "content_length": 344_011_403,
        "etag": '"1481328b-4ae2a151e2840"',
        "last_modified": "Fri, 30 Sep 2011 15:14:33 GMT",
        "sha256": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
    },
}


@dataclass(frozen=True)
class TimedPath:
    timestamp_ns: int
    timestamp_token: str
    path: str


@dataclass(frozen=True)
class TimedPose:
    timestamp_ns: int
    timestamp_token: str
    center: torch.Tensor
    quaternion_xyzw: torch.Tensor


@dataclass(frozen=True)
class AssociatedFrame:
    rgb: TimedPath
    depth: TimedPath
    pose: TimedPose
    rgb_depth_delta_ns: int

    @property
    def view_id(self) -> str:
        return f"rgb={self.rgb.timestamp_token}|depth={self.depth.timestamp_token}"


@dataclass(frozen=True)
class SelectedFrame:
    frame: AssociatedFrame
    selected_ordinal: int
    role: Literal["T", "V", "H"]
    role_ordinal: int


@dataclass(frozen=True)
class TargetBundle:
    points_world: torch.Tensor
    normals_world: torch.Tensor
    source_t_ordinals: torch.Tensor
    source_selected_ordinals: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    per_t_counts: tuple[int, ...]
    eligibility_mask: torch.Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains nonfinite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _finite_metric_mapping(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != METRIC_KEYS:
        raise ValueError(f"{label} has an unexpected metric set")
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} metric {key} must be numeric")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{label} metric {key} must be finite")
        result[key] = converted
    return result


def _tensor_hash(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _encoded_tensor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order="C")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "raw_sha256": _sha256_bytes(raw),
        "encoding": "base64(zlib(raw-c-order))",
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def _target_tensor_entries(targets: TargetBundle) -> list[tuple[str, torch.Tensor]]:
    return [
        ("points_world", targets.points_world),
        ("normals_world", targets.normals_world),
        ("source_t_ordinals", targets.source_t_ordinals),
        ("source_selected_ordinals", targets.source_selected_ordinals),
        ("rows", targets.rows),
        ("columns", targets.columns),
        ("eligibility_mask", targets.eligibility_mask),
    ]


def _target_composite_hash(
    tensor_sha256: str,
    construction_manifest_sha256: str,
    *,
    config_id: str = CONFIG_ID,
) -> str:
    """Bind target tensors to their calibrated construction cameras and backend config."""

    return _sha256_bytes(
        _canonical_json(
            {
                "config_id": config_id,
                "construction_manifest_sha256": construction_manifest_sha256,
                "target_tensor_sha256": tensor_sha256,
            }
        )
    )


def _timestamp_ns(token: str) -> int:
    value = Decimal(token) * Decimal(1_000_000_000)
    integral = value.to_integral_value()
    if value != integral:
        raise ValueError(f"timestamp has sub-nanosecond precision: {token!r}")
    return int(integral)


def _utc_from_posix_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    prefix = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{prefix}.{nanoseconds:09d}+00:00"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive-relative path: {value!r}")
    return path.as_posix()


def _data_lines(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(stripped.split())
    return rows


def _parse_timed_paths(text: str) -> list[TimedPath]:
    result: list[TimedPath] = []
    seen_timestamps: set[int] = set()
    seen_paths: set[str] = set()
    for fields in _data_lines(text):
        if len(fields) != 2:
            raise ValueError("RGB/depth manifest rows must contain timestamp and path")
        timestamp = _timestamp_ns(fields[0])
        if timestamp in seen_timestamps:
            raise ValueError("duplicate manifest timestamp")
        path = _safe_relative_path(fields[1])
        if path in seen_paths:
            raise ValueError("duplicate manifest payload path")
        seen_timestamps.add(timestamp)
        seen_paths.add(path)
        result.append(TimedPath(timestamp, fields[0], path))
    if not result:
        raise ValueError("empty RGB/depth manifest")
    return sorted(result, key=lambda item: item.timestamp_ns)


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion.to(dtype=torch.float64)
    norm = quaternion.norm()
    if not bool(torch.isfinite(norm)) or abs(float(norm) - 1.0) > 1e-3:
        raise ValueError("ground-truth quaternion norm differs from one by more than 1e-3")
    return quaternion / norm


def _parse_timed_poses(text: str) -> list[TimedPose]:
    result: list[TimedPose] = []
    seen: set[int] = set()
    for fields in _data_lines(text):
        if len(fields) != 8:
            raise ValueError("groundtruth rows must contain timestamp plus seven pose values")
        timestamp = _timestamp_ns(fields[0])
        if timestamp in seen:
            raise ValueError("duplicate ground-truth timestamp")
        seen.add(timestamp)
        values = torch.tensor([float(Decimal(value)) for value in fields[1:]], dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("ground-truth pose values must be finite")
        result.append(
            TimedPose(
                timestamp_ns=timestamp,
                timestamp_token=fields[0],
                center=values[:3],
                quaternion_xyzw=_normalize_quaternion(values[3:]),
            )
        )
    if not result:
        raise ValueError("empty ground-truth manifest")
    return sorted(result, key=lambda item: item.timestamp_ns)


def _associate(
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
        if abs(left.timestamp_ns - right.timestamp_ns) < MAX_ASSOCIATION_NS
    ]
    candidates.sort(key=lambda item: item[:3])
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[TimedPath, TimedPath]] = []
    for _, first_ns, second_ns, left, right in candidates:
        if first_ns in used_first or second_ns in used_second:
            continue
        used_first.add(first_ns)
        used_second.add(second_ns)
        matches.append((left, right))
    return sorted(matches, key=lambda item: item[0].timestamp_ns)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, fraction: float) -> torch.Tensor:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = float(torch.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return (q0 + fraction * (q1 - q0)) / (q0 + fraction * (q1 - q0)).norm()
    theta = math.acos(dot)
    result = (
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * q0
        + math.sin(fraction * theta) / math.sin(theta) * q1
    )
    return result / result.norm()


def _interpolate_pose(poses: list[TimedPose], timestamp_ns: int) -> TimedPose | None:
    timestamps = [pose.timestamp_ns for pose in poses]
    position = bisect.bisect_left(timestamps, timestamp_ns)
    if position < len(poses) and poses[position].timestamp_ns == timestamp_ns:
        pose = poses[position]
        return TimedPose(
            timestamp_ns,
            str(Decimal(timestamp_ns) / Decimal(1_000_000_000)),
            pose.center,
            pose.quaternion_xyzw,
        )
    if position == 0 or position == len(poses):
        return None
    lower = poses[position - 1]
    upper = poses[position]
    span = upper.timestamp_ns - lower.timestamp_ns
    if not lower.timestamp_ns < timestamp_ns < upper.timestamp_ns or span > MAX_ASSOCIATION_NS:
        return None
    fraction = (timestamp_ns - lower.timestamp_ns) / span
    center = lower.center + fraction * (upper.center - lower.center)
    quaternion = _slerp(lower.quaternion_xyzw, upper.quaternion_xyzw, fraction)
    token = str(Decimal(timestamp_ns) / Decimal(1_000_000_000))
    return TimedPose(timestamp_ns, token, center, quaternion)


class TumTar:
    """Safe, audited access to one TUM tgz without filesystem extraction."""

    def __init__(self, path: Path, *, allowed_depth_members: set[str] | None = None):
        self.path = path
        self.allowed_depth_members = allowed_depth_members
        self.attempted_depth_members: list[str] = []
        self.decoded_depth_members: list[str] = []
        self._tar: tarfile.TarFile | None = None
        self._members: dict[str, tarfile.TarInfo] = {}
        self.prefix = ""

    def __enter__(self) -> TumTar:
        self._tar = tarfile.open(self.path, mode="r:gz")
        for member in self._tar.getmembers():
            name = _safe_relative_path(member.name)
            if name in self._members:
                raise ValueError(f"duplicate tar member: {name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are forbidden in source archive: {name}")
            self._members[name] = member
        roots = {
            name[: -len("/rgb.txt")]
            for name, member in self._members.items()
            if member.isfile() and name.endswith("/rgb.txt")
        }
        if len(roots) != 1:
            raise ValueError("archive must contain exactly one rooted rgb.txt")
        self.prefix = next(iter(roots))
        for relative in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            full = f"{self.prefix}/{relative}"
            if full not in self._members or not self._members[full].isfile():
                raise ValueError(f"archive is missing {relative}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._tar is not None:
            self._tar.close()

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._members))

    def _read_member(self, full_name: str) -> bytes:
        if self._tar is None:
            raise RuntimeError("archive is not open")
        member = self._members.get(full_name)
        if member is None or not member.isfile():
            raise ValueError(f"missing regular archive member: {full_name}")
        stream = self._tar.extractfile(member)
        if stream is None:
            raise ValueError(f"could not read archive member: {full_name}")
        return stream.read()

    def read_text(self, relative: str) -> tuple[str, str]:
        if relative not in {"rgb.txt", "depth.txt", "groundtruth.txt"}:
            raise ValueError("only the three source text manifests may be opened as metadata")
        payload = self._read_member(f"{self.prefix}/{relative}")
        return payload.decode("utf-8"), _sha256_bytes(payload)

    def full_member_name(self, relative: str) -> str:
        safe = _safe_relative_path(relative)
        full = f"{self.prefix}/{safe}"
        if full not in self._members or not self._members[full].isfile():
            raise ValueError(f"manifest path is not a regular archive member: {safe}")
        return full

    def member_offset(self, relative: str) -> int:
        return self._members[self.full_member_name(relative)].offset_data

    def decode_depth(self, relative: str) -> torch.Tensor:
        full = self.full_member_name(relative)
        if self.allowed_depth_members is None or full not in self.allowed_depth_members:
            raise ValueError(f"depth payload is outside the frozen T/V allowlist: {full}")
        if full in self.attempted_depth_members:
            raise ValueError(f"depth payload opened more than once: {full}")
        self.attempted_depth_members.append(full)
        payload = self._read_member(full)
        with PILImage.open(io.BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image)
            mode = image.mode
        if array.shape != (HEIGHT, WIDTH) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"depth PNG must be 640x480 integer data, got {array.shape}/{array.dtype}"
            )
        if int(array.min()) < 0 or int(array.max()) > 65535:
            raise ValueError("depth PNG values must fit uint16")
        if mode not in {"I", "I;16", "I;16B", "I;16L"}:
            raise ValueError(f"unexpected depth PNG mode: {mode}")
        self.decoded_depth_members.append(full)
        return torch.from_numpy(array.astype(np.float64, copy=True)) / 5000.0


def _quaternion_to_rotation_c2w(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw).unbind()
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def _camera_from_pose(pose: TimedPose) -> Camera:
    rotation_c2w = _quaternion_to_rotation_c2w(pose.quaternion_xyzw)
    rotation_w2c = rotation_c2w.T
    translation = -rotation_w2c @ pose.center
    if not torch.allclose(
        rotation_w2c @ pose.center + translation,
        torch.zeros(3, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("camera center/extrinsic invariant failed")
    if not math.isclose(float(torch.linalg.det(rotation_w2c)), 1.0, abs_tol=1e-8):
        raise ValueError("camera rotation determinant invariant failed")
    camera = Camera(
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        width=WIDTH,
        height=HEIGHT,
        R=rotation_w2c,
        t=translation,
    )
    # Camera currently defaults its storage to float32 for training.  This sealed audit
    # restores the already validated float64 pose tensors so construction and metrics do
    # not inherit that optimization-oriented truncation.
    camera.R = rotation_w2c.detach().clone()
    camera.t = translation.detach().clone()
    if camera.R.dtype != torch.float64 or camera.t.dtype != torch.float64:
        raise AssertionError("audit camera extrinsics must remain float64")
    probe_uv = torch.tensor([[0.5, 0.5], [320.5, 240.5], [639.5, 479.5]], dtype=torch.float64)
    probe_depth = torch.tensor([0.5, 2.0, 4.5], dtype=torch.float64)
    points = camera.unproject(probe_uv, probe_depth)
    projected, depth = camera.project(points)
    if not torch.allclose(projected, probe_uv, atol=1e-10, rtol=0) or not torch.allclose(
        depth, probe_depth, atol=1e-10, rtol=0
    ):
        raise ValueError("camera project/unproject roundtrip invariant failed")
    return camera


def _rotation_distance_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    dot = abs(float(torch.dot(_normalize_quaternion(left), _normalize_quaternion(right))))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def _associate_frames(
    rgb: list[TimedPath], depth: list[TimedPath], poses: list[TimedPose]
) -> tuple[list[AssociatedFrame], dict[str, int]]:
    paired = _associate(rgb, depth)
    frames: list[AssociatedFrame] = []
    rejected_pose = 0
    for rgb_item, depth_item in paired:
        pose = _interpolate_pose(poses, depth_item.timestamp_ns)
        if pose is None:
            rejected_pose += 1
            continue
        frames.append(
            AssociatedFrame(
                rgb=rgb_item,
                depth=depth_item,
                pose=pose,
                rgb_depth_delta_ns=abs(rgb_item.timestamp_ns - depth_item.timestamp_ns),
            )
        )
    if not frames:
        raise ValueError("no RGB/depth/pose triples survived association")
    return frames, {
        "rgb_count": len(rgb),
        "depth_count": len(depth),
        "rgb_depth_pairs": len(paired),
        "pose_interpolation_rejections": rejected_pose,
        "associated_triples": len(frames),
    }


def _select_frames(frames: list[AssociatedFrame]) -> tuple[list[SelectedFrame], dict[str, Any]]:
    pose_keyframes = [frames[0]]
    for frame in frames[1:]:
        previous = pose_keyframes[-1]
        translation = float((frame.pose.center - previous.pose.center).norm())
        rotation = _rotation_distance_deg(frame.pose.quaternion_xyzw, previous.pose.quaternion_xyzw)
        if translation >= MIN_TRANSLATION_M or rotation >= MIN_ROTATION_DEG:
            pose_keyframes.append(frame)
    count = len(pose_keyframes)
    if count < N_KEYFRAMES:
        raise ValueError(f"fewer than 64 pose keyframes: {count}")
    keyframe_indices = []
    for index in range(64):
        numerator = index * (count - 1)
        quotient, remainder = divmod(numerator, 63)
        keyframe_indices.append(quotient + int(2 * remainder >= 63))
    if keyframe_indices[0] != 0 or keyframe_indices[-1] != count - 1:
        raise AssertionError("half-up keyframe selection lost an endpoint")
    if any(right <= left for left, right in zip(keyframe_indices, keyframe_indices[1:])):
        raise AssertionError("half-up keyframe indices are not strictly increasing")

    role_counts = {"T": 0, "V": 0, "H": 0}
    selected: list[SelectedFrame] = []
    for ordinal, keyframe_index in enumerate(keyframe_indices):
        role: Literal["T", "V", "H"]
        if ordinal % 8 == 7:
            role = "H"
        elif ordinal % 8 == 3:
            role = "V"
        else:
            role = "T"
        selected.append(
            SelectedFrame(
                frame=pose_keyframes[keyframe_index],
                selected_ordinal=ordinal,
                role=role,
                role_ordinal=role_counts[role],
            )
        )
        role_counts[role] += 1
    if role_counts != {"T": 48, "V": 8, "H": 8}:
        raise AssertionError(f"unexpected role counts: {role_counts}")
    return selected, {
        "pose_keyframe_count_before_uniform_selection": count,
        "uniform_source_indices": keyframe_indices,
        "role_counts": role_counts,
    }


class TumDepthBackend:
    def __init__(self, depths: dict[str, torch.Tensor]):
        self.depths = depths

    def predict(
        self, image_shape: tuple[int, int], camera: Camera, *, view_id: str
    ) -> OrientedPointPrediction:
        if view_id not in self.depths:
            raise KeyError(f"backend has no T/V depth for {view_id}")
        depth = self.depths[view_id]
        target_config = CONFIG["target"]
        normals, valid = estimate_registered_depth_normals(
            depth,
            camera,
            pixel_offset=target_config["pixel_offset"],
            min_depth=target_config["min_depth_m"],
            max_depth=target_config["max_depth_m"],
            max_abs_depth_delta=target_config["max_abs_depth_delta_m"],
            max_relative_depth_delta=target_config["max_relative_depth_delta"],
            min_cross_norm=target_config["min_cross_norm"],
            min_abs_incidence=target_config["min_abs_incidence"],
            orient_toward_camera=True,
        )
        return OrientedPointPrediction(
            geometry=depth,
            normals=normals,
            geometry_kind="camera_z_depth",
            normal_frame="camera",
            valid=valid,
            confidence=None,
            provenance=OrientedPointProvenance(
                view_id=view_id,
                backend_name="tum-registered-rgbd",
                backend_version="phase-a-v1",
                config_id=CONFIG_ID,
            ),
        )


def _canonical_map(backend: TumDepthBackend, selected: SelectedFrame, camera: Camera):
    prediction = backend.predict((HEIGHT, WIDTH), camera, view_id=selected.frame.view_id)
    return canonicalize_oriented_point_prediction(
        prediction,
        camera,
        (HEIGHT, WIDTH),
        expected_view_id=selected.frame.view_id,
        expected_config_id=CONFIG_ID,
        device="cpu",
        dtype=torch.float64,
    )


def _construct_targets(
    t_frames: list[SelectedFrame], cameras: dict[str, Camera], backend: TumDepthBackend
) -> TargetBundle:
    if len(t_frames) != 48 or [item.role_ordinal for item in t_frames] != list(range(48)):
        raise ValueError("target construction requires the frozen 48 ordered T views")
    grid_rows, grid_columns = torch.meshgrid(
        torch.tensor(TARGET_ROWS, dtype=torch.long),
        torch.tensor(TARGET_COLUMNS, dtype=torch.long),
        indexing="ij",
    )
    flat_rows = grid_rows.reshape(-1)
    flat_columns = grid_columns.reshape(-1)
    points: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    t_ordinals: list[torch.Tensor] = []
    selected_ordinals: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    columns: list[torch.Tensor] = []
    counts: list[int] = []
    eligibility_masks: list[torch.Tensor] = []
    for selected in t_frames:
        camera = cameras[selected.frame.view_id]
        canonical = _canonical_map(backend, selected, camera)
        valid = canonical.valid[flat_rows, flat_columns]
        count = int(valid.sum())
        counts.append(count)
        eligibility_masks.append(valid.detach().clone())
        points.append(canonical.points_world[flat_rows, flat_columns][valid])
        normals.append(canonical.normals_world[flat_rows, flat_columns][valid])
        t_ordinals.append(torch.full((count,), selected.role_ordinal, dtype=torch.long))
        selected_ordinals.append(torch.full((count,), selected.selected_ordinal, dtype=torch.long))
        rows.append(flat_rows[valid].clone())
        columns.append(flat_columns[valid].clone())
    if not points or sum(counts) == 0:
        raise ValueError("target constructor produced no eligible oriented points")
    if any(count == 0 for count in counts):
        raise ValueError("a T view has zero eligible targets, making A_min/S_10 undefined")
    bundle = TargetBundle(
        points_world=torch.cat(points).detach(),
        normals_world=torch.cat(normals).detach(),
        source_t_ordinals=torch.cat(t_ordinals),
        source_selected_ordinals=torch.cat(selected_ordinals),
        rows=torch.cat(rows),
        columns=torch.cat(columns),
        per_t_counts=tuple(counts),
        eligibility_mask=torch.stack(eligibility_masks),
    )
    if bundle.eligibility_mask.shape != (len(t_frames), N_GRID_PER_VIEW):
        raise AssertionError("target eligibility mask has the wrong shape")
    if bundle.eligibility_mask.dtype != torch.bool:
        raise AssertionError("target eligibility mask must be boolean")
    if bundle.eligibility_mask.sum().item() != bundle.points_world.shape[0]:
        raise AssertionError("target eligibility mask disagrees with compact target tensors")
    if tuple(int(row.sum()) for row in bundle.eligibility_mask) != bundle.per_t_counts:
        raise AssertionError("target eligibility rows disagree with per-view counts")
    if not bool(torch.isfinite(bundle.points_world).all()):
        raise ValueError("target points must be finite")
    if not bool(torch.isfinite(bundle.normals_world).all()):
        raise ValueError("target normals must be finite")
    if not torch.allclose(
        bundle.normals_world.norm(dim=-1),
        torch.ones_like(bundle.normals_world[:, 0]),
        atol=2e-12,
        rtol=0,
    ):
        raise ValueError("target normals must be unit length")
    return bundle


def _linear_quantile(values: torch.Tensor, q: float) -> float:
    values = values.detach().cpu().to(torch.float64).flatten()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("required metric population is empty or nonfinite")
    return float(torch.quantile(values, q, interpolation="linear"))


def _row_medians(values: torch.Tensor, minimum_count: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("row-median input must be a matrix")
    medians: list[float] = []
    supported = torch.zeros(values.shape[0], dtype=torch.bool)
    for row_index, row in enumerate(values):
        finite = row[torch.isfinite(row)]
        if finite.numel() >= minimum_count:
            supported[row_index] = True
            medians.append(_linear_quantile(finite, 0.5))
    return torch.tensor(medians, dtype=torch.float64), supported


def _audit_targets(
    targets: TargetBundle,
    v_frames: list[SelectedFrame],
    cameras: dict[str, Camera],
    depths: dict[str, torch.Tensor],
    backend: TumDepthBackend,
    *,
    t_frames: list[SelectedFrame] | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, torch.Tensor]]:
    count = targets.points_world.shape[0]
    n_validation = len(v_frames)
    depth_residuals = torch.full((count, n_validation), float("nan"), dtype=torch.float64)
    surface_residuals = torch.full_like(depth_residuals, float("nan"))
    normal_cosines = torch.full_like(depth_residuals, float("nan"))
    depth_pair_mask = torch.zeros(count, n_validation, dtype=torch.bool)
    oriented_pair_mask = torch.zeros_like(depth_pair_mask)
    free_space_mask = torch.zeros_like(depth_pair_mask)
    per_v: list[dict[str, Any]] = []
    metric_target_points = targets.points_world.detach().cpu().to(torch.float64)
    metric_target_normals = targets.normals_world.detach().cpu().to(torch.float64)

    for v_index, selected in enumerate(v_frames):
        view_id = selected.frame.view_id
        camera = cameras[view_id]
        depth_map = depths[view_id].detach().cpu().to(torch.float64)
        canonical = _canonical_map(backend, selected, camera)
        pixels, predicted_depth = camera.project(metric_target_points)
        in_frame = (
            torch.isfinite(pixels).all(dim=-1)
            & torch.isfinite(predicted_depth)
            & (predicted_depth > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < WIDTH)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < HEIGHT)
        )
        target_indices = torch.nonzero(in_frame, as_tuple=False).squeeze(-1)
        columns = torch.floor(pixels[target_indices, 0]).long()
        rows = torch.floor(pixels[target_indices, 1]).long()
        flat_pixels = rows * WIDTH + columns
        z_buffer = torch.full((HEIGHT * WIDTH,), float("inf"), dtype=predicted_depth.dtype)
        z_buffer.scatter_reduce_(
            0,
            flat_pixels,
            predicted_depth[target_indices],
            reduce="amin",
            include_self=True,
        )
        visible_local = (
            predicted_depth[target_indices] - z_buffer[flat_pixels] <= CONFIG["visibility_depth_m"]
        )
        visible_targets = target_indices[visible_local]
        visible_rows = rows[visible_local]
        visible_columns = columns[visible_local]
        observed_depth = depth_map[visible_rows, visible_columns]
        depth_valid = (
            torch.isfinite(observed_depth) & (observed_depth >= 0.3) & (observed_depth <= 5.0)
        )
        valid_targets = visible_targets[depth_valid]
        valid_rows = visible_rows[depth_valid]
        valid_columns = visible_columns[depth_valid]
        valid_observed_depth = observed_depth[depth_valid]
        valid_predicted_depth = predicted_depth[valid_targets]
        depth_pair_mask[valid_targets, v_index] = True
        depth_residuals[valid_targets, v_index] = (
            (valid_predicted_depth - valid_observed_depth).abs()
            / valid_observed_depth.clamp_min(0.3)
        ).to(torch.float64)
        free_threshold = torch.maximum(
            torch.full_like(valid_observed_depth, 0.05), 0.03 * valid_observed_depth
        )
        free = valid_predicted_depth < valid_observed_depth - free_threshold
        free_space_mask[valid_targets, v_index] = free

        oriented = canonical.valid[valid_rows, valid_columns]
        oriented_targets = valid_targets[oriented]
        oriented_rows = valid_rows[oriented]
        oriented_columns = valid_columns[oriented]
        oriented_depth = valid_observed_depth[oriented]
        oriented_pair_mask[oriented_targets, v_index] = True
        sample_pixels = torch.stack(
            [oriented_columns.to(torch.float64) + 0.5, oriented_rows.to(torch.float64) + 0.5],
            dim=-1,
        )
        observed_points = camera.unproject(sample_pixels, oriented_depth)
        observed_normals = canonical.normals_world[oriented_rows, oriented_columns].to(
            torch.float64
        )
        target_points = metric_target_points[oriented_targets]
        target_normals = metric_target_normals[oriented_targets]
        offsets = observed_points - target_points
        symmetric = 0.5 * (
            (target_normals * offsets).sum(dim=-1).abs()
            + (observed_normals * -offsets).sum(dim=-1).abs()
        )
        cosine = (target_normals * observed_normals).sum(dim=-1).abs().clamp(0, 1)
        surface_residuals[oriented_targets, v_index] = symmetric.to(torch.float64)
        normal_cosines[oriented_targets, v_index] = cosine.to(torch.float64)
        validation_diagnostics = {
            "selected_ordinal": selected.selected_ordinal,
            "view_id": view_id,
            "in_frame_targets": int(in_frame.sum()),
            "out_of_frame_targets": int((~in_frame).sum()),
            "construction_visible_pairs": int(visible_targets.numel()),
            "construction_invisible_targets": int(in_frame.sum() - visible_targets.numel()),
            "invalid_center_depth_pairs": int((~depth_valid).sum()),
            "depth_valid_pairs": int(depth_valid.sum()),
            "invalid_oriented_stencil_pairs": int(depth_valid.sum() - oriented.sum()),
            "oriented_valid_pairs": int(oriented.sum()),
            "free_space_pairs": int(free.sum()),
        }
        if (
            validation_diagnostics["out_of_frame_targets"]
            + validation_diagnostics["construction_invisible_targets"]
            + validation_diagnostics["invalid_center_depth_pairs"]
            + validation_diagnostics["depth_valid_pairs"]
            != count
        ):
            raise AssertionError("validation exclusion counts do not conserve targets")
        if (
            validation_diagnostics["invalid_oriented_stencil_pairs"]
            + validation_diagnostics["oriented_valid_pairs"]
            != validation_diagnostics["depth_valid_pairs"]
        ):
            raise AssertionError("validation oriented counts do not conserve depth-valid pairs")
        per_v.append(validation_diagnostics)

    if not torch.equal(torch.isfinite(depth_residuals), depth_pair_mask):
        raise ValueError("depth residual finiteness differs from the frozen pair population")
    if not torch.equal(torch.isfinite(surface_residuals), oriented_pair_mask) or not torch.equal(
        torch.isfinite(normal_cosines), oriented_pair_mask
    ):
        raise ValueError("oriented metric finiteness differs from the frozen pair population")
    depth_target_values, depth_supported_from_values = _row_medians(depth_residuals)
    surface_target_values, oriented_supported_from_surface = _row_medians(surface_residuals)
    cosine_target_values, oriented_supported_from_cosine = _row_medians(normal_cosines)
    depth_supported = depth_pair_mask.sum(dim=1) >= CONFIG["minimum_validation_views"]
    oriented_supported = oriented_pair_mask.sum(dim=1) >= CONFIG["minimum_validation_views"]
    if not torch.equal(depth_supported, depth_supported_from_values):
        raise AssertionError("depth support differs from the explicit pair-mask count")
    if not torch.equal(oriented_supported, oriented_supported_from_surface) or not torch.equal(
        oriented_supported, oriented_supported_from_cosine
    ):
        raise AssertionError("oriented support differs from the explicit pair-mask count")
    for v_index, validation_diagnostics in enumerate(per_v):
        validation_diagnostics["globally_depth_supported_targets_with_pair"] = int(
            (depth_supported & depth_pair_mask[:, v_index]).sum()
        )
        validation_diagnostics["globally_oriented_supported_targets_with_pair"] = int(
            (oriented_supported & oriented_pair_mask[:, v_index]).sum()
        )
    per_t_support: list[float] = []
    per_t: list[dict[str, Any]] = []
    if not targets.per_t_counts or any(count == 0 for count in targets.per_t_counts):
        raise ValueError("every T stratum must contain at least one eligible target")
    if t_frames is not None and len(t_frames) != len(targets.per_t_counts):
        raise ValueError("T frame list and target strata have different lengths")
    for t_ordinal, eligible_count in enumerate(targets.per_t_counts):
        source = targets.source_t_ordinals == t_ordinal
        if eligible_count == 0:
            per_t_support.append(0.0)
        else:
            per_t_support.append(float((oriented_supported & source).sum()) / eligible_count)
        selected = t_frames[t_ordinal] if t_frames is not None else None
        depth_supported_count = int((depth_supported & source).sum())
        oriented_supported_count = int((oriented_supported & source).sum())
        per_t.append(
            {
                "t_ordinal": t_ordinal,
                "selected_ordinal": selected.selected_ordinal if selected is not None else None,
                "view_id": selected.frame.view_id if selected is not None else None,
                "inspected_grid_points": N_GRID_PER_VIEW,
                "eligible_targets": eligible_count,
                "ineligible_targets": N_GRID_PER_VIEW - eligible_count,
                "depth_valid_pairs": int(depth_pair_mask[source].sum()),
                "oriented_valid_pairs": int(oriented_pair_mask[source].sum()),
                "depth_supported_targets": depth_supported_count,
                "depth_support_fraction": (
                    depth_supported_count / eligible_count if eligible_count else 0.0
                ),
                "oriented_supported_targets": oriented_supported_count,
                "oriented_support_fraction": (
                    oriented_supported_count / eligible_count if eligible_count else 0.0
                ),
            }
        )

    eligible_fraction = count / (48 * N_GRID_PER_VIEW)
    per_t_eligible = [value / N_GRID_PER_VIEW for value in targets.per_t_counts]
    depth_pair_count = int(depth_pair_mask.sum())
    if depth_pair_count == 0:
        raise ValueError("no depth-valid audit pairs")
    metrics = {
        "A": float(eligible_fraction),
        "A_min": float(min(per_t_eligible)),
        "S": float(oriented_supported.sum()) / count,
        "S_10": _linear_quantile(torch.tensor(per_t_support, dtype=torch.float64), 0.1),
        "R90": _linear_quantile(surface_target_values, 0.9),
        "D90": _linear_quantile(depth_target_values, 0.9),
        "C50": _linear_quantile(cosine_target_values.clamp(0, 1), 0.5),
        "C10": _linear_quantile(cosine_target_values.clamp(0, 1), 0.1),
        "F": float((free_space_mask & depth_pair_mask).sum()) / depth_pair_count,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("required audit metrics must be finite")
    diagnostics = {
        "eligible_targets": count,
        "ineligible_targets": 48 * N_GRID_PER_VIEW - count,
        "eligible_fraction_per_t_view": per_t_eligible,
        "oriented_support_fraction_per_t_view": per_t_support,
        "depth_valid_pairs": depth_pair_count,
        "oriented_valid_pairs": int(oriented_pair_mask.sum()),
        "depth_supported_targets": int(depth_supported.sum()),
        "oriented_supported_targets": int(oriented_supported.sum()),
        "free_space_pairs": int((free_space_mask & depth_pair_mask).sum()),
        "per_construction_view": per_t,
        "per_validation_view": per_v,
    }
    raw = {
        "depth_residuals": depth_residuals,
        "surface_residuals": surface_residuals,
        "normal_cosines": normal_cosines,
        "depth_pair_mask": depth_pair_mask,
        "oriented_pair_mask": oriented_pair_mask,
        "free_space_mask": free_space_mask,
        "depth_supported": depth_supported,
        "oriented_supported": oriented_supported,
    }
    return metrics, diagnostics, raw


def _thresholds(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "A": max(0.30, 0.70 * metrics["A"]),
        "A_min": max(0.10, 0.50 * metrics["A_min"]),
        "S": max(0.20, 0.60 * metrics["S"]),
        "S_10": max(0.05, 0.50 * metrics["S_10"]),
        "R90": min(0.050, max(0.020, 1.50 * metrics["R90"] + 0.005)),
        "D90": min(0.050, max(0.020, 1.50 * metrics["D90"] + 0.005)),
        "C50": max(0.65, metrics["C50"] - 0.15),
        "C10": max(0.10, metrics["C10"] - 0.15),
        "F": min(0.10, max(0.02, 1.50 * metrics["F"] + 0.01)),
    }


def _gate(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
    comparisons = {
        "A": metrics["A"] >= thresholds["A"],
        "A_min": metrics["A_min"] >= thresholds["A_min"],
        "S": metrics["S"] >= thresholds["S"],
        "S_10": metrics["S_10"] >= thresholds["S_10"],
        "R90": metrics["R90"] <= thresholds["R90"],
        "D90": metrics["D90"] <= thresholds["D90"],
        "C50": metrics["C50"] >= thresholds["C50"],
        "C10": metrics["C10"] >= thresholds["C10"],
        "F": metrics["F"] <= thresholds["F"],
    }
    comparisons["all"] = all(comparisons.values())
    return comparisons


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(command("git", "status", "--short")),
        "status_sha256": _sha256_bytes(command("git", "status", "--short").encode()),
    }


def _implementation_metadata(root: Path) -> dict[str, Any]:
    paths = [
        root / Path(__file__).resolve().relative_to(root),
        root / "src/rtgs/lift/surface.py",
        root / "src/rtgs/core/camera.py",
        root / PREREGISTRATION,
        root / ACQUISITION_MANIFEST,
        root / "pyproject.toml",
    ]
    hashes = {str(path.relative_to(root)): _sha256_file(path) for path in paths}
    return {
        "files": hashes,
        "aggregate_sha256": _sha256_bytes(_canonical_json(hashes)),
    }


def _selected_manifest(selected: list[SelectedFrame]) -> list[dict[str, Any]]:
    return [
        {
            "selected_ordinal": item.selected_ordinal,
            "role": item.role,
            "role_ordinal": item.role_ordinal,
            "view_id": item.frame.view_id,
            "rgb_timestamp_ns": item.frame.rgb.timestamp_ns,
            "rgb_timestamp_token": item.frame.rgb.timestamp_token,
            "rgb_path": item.frame.rgb.path,
            "depth_timestamp_ns": item.frame.depth.timestamp_ns,
            "depth_timestamp_token": item.frame.depth.timestamp_token,
            "depth_path": item.frame.depth.path,
            "rgb_depth_delta_ns": item.frame.rgb_depth_delta_ns,
            "pose_center_m": item.frame.pose.center.tolist(),
            "pose_quaternion_xyzw": item.frame.pose.quaternion_xyzw.tolist(),
        }
        for item in selected
    ]


def _association_manifest(frames: list[AssociatedFrame]) -> list[dict[str, Any]]:
    return [
        {
            "view_id": frame.view_id,
            "rgb_timestamp_ns": frame.rgb.timestamp_ns,
            "rgb_path": frame.rgb.path,
            "depth_timestamp_ns": frame.depth.timestamp_ns,
            "depth_path": frame.depth.path,
            "rgb_depth_delta_ns": frame.rgb_depth_delta_ns,
            "pose_center_m": frame.pose.center.tolist(),
            "pose_quaternion_xyzw": frame.pose.quaternion_xyzw.tolist(),
        }
        for frame in frames
    ]


def _bounds(points: torch.Tensor) -> dict[str, Any]:
    lower = points.amin(dim=0)
    upper = points.amax(dim=0)
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "diagonal": float((upper - lower).norm()),
        "gating": False,
    }


def _validate_source_archive_file(
    archive_path: Path, phase: Literal["development", "confirmatory"]
) -> tuple[str, os.stat_result]:
    source = SOURCES[phase]
    stat = archive_path.stat()
    if stat.st_size != source["content_length"]:
        raise ValueError("archive byte length differs from preregistered HTTP length")
    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 != source["sha256"]:
        raise ValueError("archive SHA-256 differs from the first frozen download")
    return archive_sha256, stat


def _validate_acquisition_record(
    archive_path: Path,
    archive_stat: os.stat_result,
    phase: Literal["development", "confirmatory"],
) -> tuple[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / ACQUISITION_MANIFEST
    payload = path.read_bytes()
    manifest = _json_object_from_bytes(payload, label="acquisition manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("acquisition manifest schema_version must be exactly 1")
    if not isinstance(manifest.get("sources"), dict) or set(manifest["sources"]) != set(SOURCES):
        raise ValueError("acquisition manifest must contain the exact two frozen sources")
    record = manifest["sources"][phase]
    if not isinstance(record, dict):
        raise ValueError("acquisition source record must be an object")
    expected = SOURCES[phase]
    for key in ("sequence", "url", "content_length", "etag", "last_modified", "sha256"):
        if record.get(key) != expected[key]:
            raise ValueError(f"acquisition source {key} differs from the frozen source")
    archive_path_value = record.get("archive_path")
    if (
        not isinstance(archive_path_value, str)
        or Path(archive_path_value).resolve() != archive_path.resolve()
    ):
        raise ValueError("acquisition source path differs from the audited archive")
    if record.get("filesystem_mtime_utc_completion_proxy") != _utc_from_posix_ns(
        archive_stat.st_mtime_ns
    ):
        raise ValueError("archive mtime differs from the frozen acquisition proxy")
    if "not server-observed retrieval timestamps" not in manifest.get("evidence_limit", ""):
        raise ValueError("acquisition manifest must state its timestamp evidence limitation")
    return _sha256_bytes(payload), record


def _load_source(
    archive_path: Path,
    phase: Literal["development", "confirmatory"],
) -> tuple[
    TumTar,
    list[AssociatedFrame],
    list[SelectedFrame],
    dict[str, str],
    dict[str, Any],
    dict[str, frozenset[str]],
]:
    source = SOURCES[phase]
    archive_sha256, archive_stat = _validate_source_archive_file(archive_path, phase)
    acquisition_sha256, acquisition_record = _validate_acquisition_record(
        archive_path, archive_stat, phase
    )
    archive = TumTar(archive_path)
    archive.__enter__()
    try:
        if archive.prefix != source["sequence"]:
            raise ValueError(
                f"archive root {archive.prefix!r} differs from expected {source['sequence']!r}"
            )
        rgb_text, rgb_hash = archive.read_text("rgb.txt")
        depth_text, depth_hash = archive.read_text("depth.txt")
        pose_text, pose_hash = archive.read_text("groundtruth.txt")
        rgb = _parse_timed_paths(rgb_text)
        depth = _parse_timed_paths(depth_text)
        poses = _parse_timed_poses(pose_text)
        rgb_members = frozenset(archive.full_member_name(item.path) for item in rgb)
        depth_members = frozenset(archive.full_member_name(item.path) for item in depth)
        if len(rgb_members) != len(rgb) or len(depth_members) != len(depth):
            raise AssertionError("manifest payload member identity is not unique")
        if rgb_members & depth_members:
            raise ValueError("RGB and depth manifests alias the same archive payload")
        frames, association_diagnostics = _associate_frames(rgb, depth, poses)
        selected, selection_diagnostics = _select_frames(frames)
        for item in selected:
            archive.full_member_name(item.frame.rgb.path)
            archive.full_member_name(item.frame.depth.path)
        manifest_hashes = {
            "rgb.txt": rgb_hash,
            "depth.txt": depth_hash,
            "groundtruth.txt": pose_hash,
            "association": _sha256_bytes(_canonical_json(_association_manifest(frames))),
            "split": _sha256_bytes(_canonical_json(_selected_manifest(selected))),
        }
        diagnostics = {
            "archive_sha256": archive_sha256,
            "local_file_mtime_utc": _utc_from_posix_ns(archive_stat.st_mtime_ns),
            "local_file_mtime_semantics": "download_completion_proxy_not_network_retrieval_time",
            "acquisition_manifest": {
                "path": str(ACQUISITION_MANIFEST),
                "sha256": acquisition_sha256,
                "record": acquisition_record,
            },
            "archive_member_count": len(archive.member_names),
            "manifest_payload_members": {
                "rgb_count": len(rgb_members),
                "depth_count": len(depth_members),
                "rgb_set_sha256": _sha256_bytes(_canonical_json(sorted(rgb_members))),
                "depth_set_sha256": _sha256_bytes(_canonical_json(sorted(depth_members))),
                "rgb_depth_disjoint": True,
                "all_regular_and_unique": True,
            },
            "association": association_diagnostics,
            "selection": selection_diagnostics,
        }
        return (
            archive,
            frames,
            selected,
            manifest_hashes,
            diagnostics,
            {"rgb": rgb_members, "depth": depth_members},
        )
    except BaseException:
        archive.__exit__(*sys.exc_info())
        raise


def _validate_payload_access(archive: TumTar, allowed: set[str]) -> None:
    attempted = archive.attempted_depth_members
    decoded = archive.decoded_depth_members
    if (
        set(attempted) != allowed
        or len(attempted) != len(allowed)
        or set(decoded) != allowed
        or len(decoded) != len(allowed)
        or attempted != decoded
    ):
        raise AssertionError(
            "attempted/decoded payload members differ from the exact T/V depth allowlist"
        )


def _run_audit(
    archive_path: Path,
    phase: Literal["development", "confirmatory"],
    *,
    before_payload_decode: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if phase == "confirmatory" and before_payload_decode is None:
        raise ValueError("confirmatory audit requires the atomic attempt-seal callback")
    if phase == "development" and before_payload_decode is not None:
        raise ValueError("development audit must not receive a confirmatory attempt callback")
    archive, frames, selected, manifest_hashes, source_diagnostics, manifest_members = _load_source(
        archive_path, phase
    )
    try:
        t_frames = [item for item in selected if item.role == "T"]
        v_frames = [item for item in selected if item.role == "V"]
        h_frames = [item for item in selected if item.role == "H"]
        decoded_frames = [item for item in selected if item.role in {"T", "V"}]
        allowed_ordered = [
            archive.full_member_name(item.frame.depth.path) for item in decoded_frames
        ]
        allowed = set(allowed_ordered)
        h_depths = {archive.full_member_name(item.frame.depth.path) for item in h_frames}
        all_rgb_members = set(manifest_members["rgb"])
        all_depth_members = set(manifest_members["depth"])
        if len(allowed_ordered) != 56 or len(allowed) != 56 or len(h_depths) != 8:
            raise AssertionError("the frozen T/V/H depth split lost unique member identity")
        if not allowed <= all_depth_members or not h_depths <= all_depth_members:
            raise AssertionError("selected depth allowlists are not semantic depth members")
        if allowed & h_depths or allowed & all_rgb_members or h_depths & all_rgb_members:
            raise AssertionError("T/V, H-depth, and all-RGB payload capabilities must be disjoint")
        if before_payload_decode is not None:
            before_payload_decode()
        archive.allowed_depth_members = allowed
        depths: dict[str, torch.Tensor] = {}
        cameras: dict[str, Camera] = {}
        # Decode in physical tar order. This changes no selected set or metric and avoids
        # repeatedly rewinding and inflating the gzip stream.
        decode_order = sorted(
            decoded_frames, key=lambda item: archive.member_offset(item.frame.depth.path)
        )
        for item in decode_order:
            view_id = item.frame.view_id
            if view_id in depths:
                raise ValueError("selected view IDs must be unique")
            depths[view_id] = archive.decode_depth(item.frame.depth.path)
            cameras[view_id] = _camera_from_pose(item.frame.pose)
        _validate_payload_access(archive, allowed)
        attempted = set(archive.attempted_depth_members)
        decoded = set(archive.decoded_depth_members)
        if attempted & all_rgb_members or decoded & all_rgb_members:
            raise AssertionError("an RGB payload was attempted or decoded")
        if attempted & h_depths or decoded & h_depths:
            raise AssertionError("an H-role depth payload was attempted or decoded")
        t_view_ids = {item.frame.view_id for item in t_frames}
        v_view_ids = {item.frame.view_id for item in v_frames}
        construction_backend = TumDepthBackend(
            {view_id: depth for view_id, depth in depths.items() if view_id in t_view_ids}
        )
        validation_backend = TumDepthBackend(
            {view_id: depth for view_id, depth in depths.items() if view_id in v_view_ids}
        )
        if set(construction_backend.depths) != t_view_ids:
            raise AssertionError("construction backend has anything other than exact T capability")
        if set(validation_backend.depths) != v_view_ids:
            raise AssertionError("validation backend has anything other than exact V capability")
        if set(construction_backend.depths) & set(validation_backend.depths):
            raise AssertionError("construction and validation backend capabilities overlap")
        targets = _construct_targets(t_frames, cameras, construction_backend)
        target_tensor_hash = _tensor_hash(_target_tensor_entries(targets))
        selected_manifest = _selected_manifest(selected)
        construction_manifest = [item for item in selected_manifest if item["role"] == "T"]
        construction_manifest_hash = _sha256_bytes(_canonical_json(construction_manifest))
        target_composite_hash = _target_composite_hash(
            target_tensor_hash, construction_manifest_hash
        )
        metrics, audit_diagnostics, raw = _audit_targets(
            targets,
            v_frames,
            cameras,
            validation_backend.depths,
            validation_backend,
            t_frames=t_frames,
        )
        if target_tensor_hash != _tensor_hash(_target_tensor_entries(targets)):
            raise AssertionError("target tensors mutated during audit")
        audit_hash = _tensor_hash(sorted(raw.items()))
        max_offset = max(frame.rgb_depth_delta_ns for frame in frames)
        return {
            "phase": phase,
            "source": {
                **SOURCES[phase],
                "archive_path": str(archive_path.resolve()),
                "manifest_hashes": manifest_hashes,
                **source_diagnostics,
            },
            "association_max_rgb_depth_delta_ns": max_offset,
            "selected_frames": selected_manifest,
            "frozen_payload_allowlist": {
                "selected_order_members": allowed_ordered,
                "selected_order_sha256": _sha256_bytes(_canonical_json(allowed_ordered)),
                "count": len(allowed_ordered),
            },
            "attempted_payload_members": archive.attempted_depth_members,
            "attempted_payload_count": len(attempted),
            "decoded_payload_members": archive.decoded_depth_members,
            "decoded_payload_count": len(decoded),
            "rgb_payload_count": 0,
            "h_payload_count": 0,
            "backend_capabilities": {
                "construction_view_ids": sorted(construction_backend.depths),
                "validation_view_ids": sorted(validation_backend.depths),
                "disjoint": True,
            },
            "config": CONFIG,
            "config_id": CONFIG_ID,
            "target": {
                "hash": target_composite_hash,
                "tensor_sha256": target_tensor_hash,
                "construction_manifest_sha256": construction_manifest_hash,
                "count": targets.points_world.shape[0],
                "per_t_counts": list(targets.per_t_counts),
                "bounds": _bounds(targets.points_world),
                "serialized": {
                    "points_world": _encoded_tensor(targets.points_world),
                    "normals_world": _encoded_tensor(targets.normals_world),
                    "source_t_ordinals": _encoded_tensor(targets.source_t_ordinals),
                    "source_selected_ordinals": _encoded_tensor(targets.source_selected_ordinals),
                    "rows": _encoded_tensor(targets.rows),
                    "columns": _encoded_tensor(targets.columns),
                    "eligibility_mask": _encoded_tensor(targets.eligibility_mask),
                },
            },
            "audit_tensor_hash": audit_hash,
            "audit_serialized": {name: _encoded_tensor(value) for name, value in raw.items()},
            "metrics": metrics,
            "diagnostics": audit_diagnostics,
        }
    finally:
        archive.__exit__(None, None, None)


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _sha256_bytes(payload)


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _write_bytes_exclusive(path, payload)


def _create_confirmatory_seal_once(path: Path, value: dict[str, Any]) -> str:
    """Atomically consume the sole confirmatory attempt before any desk PNG decode."""

    return _write_json_once(path, value)


def _validate_threshold_manifest(
    path: Path, root: Path, implementation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_bytes = path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest = _json_object_from_bytes(manifest_bytes, label="threshold manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("threshold manifest schema_version must be exactly 1")
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("threshold manifest experiment mismatch")
    if manifest.get("config_id") != CONFIG_ID:
        raise ValueError("confirmatory config differs from frozen development config")
    if manifest.get("development_archive_sha256") != SOURCES["development"]["sha256"]:
        raise ValueError("threshold manifest names the wrong development archive")
    if manifest.get("implementation_aggregate_sha256") != implementation["aggregate_sha256"]:
        raise ValueError("confirmatory implementation differs from development implementation")
    if manifest.get("preregistration_sha256") != implementation["files"][str(PREREGISTRATION)]:
        raise ValueError("confirmatory preregistration differs from development preregistration")
    development_artifact_value = manifest.get("development_artifact")
    if not isinstance(development_artifact_value, str) or not development_artifact_value:
        raise ValueError("threshold manifest development_artifact must be a path string")
    development_artifact = Path(development_artifact_value)
    if not development_artifact.is_absolute():
        development_artifact = root / development_artifact
    if not development_artifact.is_file():
        raise FileNotFoundError(development_artifact)
    development_bytes = development_artifact.read_bytes()
    if _sha256_bytes(development_bytes) != manifest.get("development_artifact_sha256"):
        raise ValueError("development artifact hash differs from frozen threshold manifest")
    development = _json_object_from_bytes(development_bytes, label="development artifact")
    if type(development.get("schema_version")) is not int or development["schema_version"] != 1:
        raise ValueError("development artifact schema_version must be exactly 1")
    if development.get("experiment") != EXPERIMENT:
        raise ValueError("development artifact experiment mismatch")
    if development.get("phase") != "development" or development.get("config_id") != CONFIG_ID:
        raise ValueError("threshold source is not the frozen development result")
    development_source = development.get("source", {})
    if (
        development_source.get("sha256") != SOURCES["development"]["sha256"]
        or development_source.get("archive_sha256") != SOURCES["development"]["sha256"]
    ):
        raise ValueError("development artifact source archive hash mismatch")
    if (
        development.get("implementation", {}).get("aggregate_sha256")
        != implementation["aggregate_sha256"]
    ):
        raise ValueError("development artifact implementation hash mismatch")
    if (
        development.get("implementation", {}).get("files", {}).get(str(PREREGISTRATION))
        != implementation["files"][str(PREREGISTRATION)]
    ):
        raise ValueError("development artifact preregistration hash mismatch")
    metrics = _finite_metric_mapping(development.get("metrics"), label="development artifact")
    manifest_metrics = _finite_metric_mapping(
        manifest.get("development_metrics"), label="threshold manifest development"
    )
    if manifest_metrics != metrics:
        raise ValueError("threshold manifest development metrics were edited")
    recomputed = _thresholds(metrics)
    recorded = _finite_metric_mapping(
        manifest.get("thresholds"), label="threshold manifest thresholds"
    )
    if any(recorded[key] != recomputed[key] for key in recomputed):
        raise ValueError("threshold values differ from the frozen mechanical formulas")
    derived = _finite_metric_mapping(
        development.get("derived_confirmatory_thresholds"),
        label="development artifact derived thresholds",
    )
    if derived != recomputed:
        raise ValueError("development artifact derived thresholds are inconsistent")
    return manifest, recomputed, manifest_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        help="required in development; append-only frozen threshold manifest",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="required in confirmatory; development threshold manifest",
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _preflight_artifact_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite append-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
        raise PermissionError(f"artifact parent is not writable: {path.parent}")


def main() -> None:
    args = _parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.phase == "development":
        if args.thresholds_output is None or args.thresholds is not None:
            raise ValueError("development requires --thresholds-output and forbids --thresholds")
        _preflight_artifact_path(args.thresholds_output)
    else:
        if args.thresholds is None or args.thresholds_output is not None:
            raise ValueError("confirmatory requires --thresholds and forbids --thresholds-output")
    _preflight_artifact_path(args.output)
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    implementation = _implementation_metadata(root)
    git_metadata = _git_metadata(root)
    threshold_manifest = None
    threshold_manifest_sha256 = None
    thresholds = None
    confirmatory_seal_sha256 = None
    before_payload_decode: Callable[[], None] | None = None
    started = datetime.now(timezone.utc)
    if args.phase == "confirmatory":
        # Validate the entire development seal before any desk PNG can reach the decoder.
        threshold_manifest, thresholds, threshold_manifest_sha256 = _validate_threshold_manifest(
            args.thresholds, root, implementation
        )

        def consume_confirmatory_attempt() -> None:
            nonlocal confirmatory_seal_sha256
            if confirmatory_seal_sha256 is not None:
                raise AssertionError("confirmatory attempt callback ran more than once")
            confirmatory_seal_sha256 = _create_confirmatory_seal_once(
                root / CONFIRMATORY_SEAL,
                {
                    "schema_version": 1,
                    "experiment": EXPERIMENT,
                    "status": "started",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "confirmatory_archive_path": str(args.archive.resolve()),
                    "confirmatory_archive_sha256": SOURCES["confirmatory"]["sha256"],
                    "threshold_manifest_path": str(args.thresholds.resolve()),
                    "threshold_manifest_sha256": threshold_manifest_sha256,
                    "implementation_aggregate_sha256": implementation["aggregate_sha256"],
                    "config_id": CONFIG_ID,
                    "preregistration_sha256": implementation["files"][str(PREREGISTRATION)],
                    "git_status_sha256": git_metadata["status_sha256"],
                    "requested_output": str(args.output.resolve()),
                },
            )

        before_payload_decode = consume_confirmatory_attempt
    result = _run_audit(
        args.archive,
        args.phase,
        before_payload_decode=before_payload_decode,
    )
    result.update(
        {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "threads": args.threads,
            "implementation": implementation,
            "git": git_metadata,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
    )

    if args.phase == "development":
        thresholds = _thresholds(result["metrics"])
        result["derived_confirmatory_thresholds"] = thresholds
        result_sha256 = _write_json_once(args.output, result)
        threshold_manifest = {
            "schema_version": 1,
            "experiment": result["experiment"],
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "development_artifact": str(args.output.resolve()),
            "development_artifact_sha256": result_sha256,
            "development_archive_sha256": SOURCES["development"]["sha256"],
            "development_metrics": result["metrics"],
            "thresholds": thresholds,
            "config_id": CONFIG_ID,
            "implementation_aggregate_sha256": implementation["aggregate_sha256"],
            "preregistration_sha256": implementation["files"][str(PREREGISTRATION)],
        }
        _write_json_once(args.thresholds_output, threshold_manifest)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "thresholds": str(args.thresholds_output),
                    "metrics": result["metrics"],
                },
                indent=2,
            )
        )
        return

    if (
        threshold_manifest is None
        or threshold_manifest_sha256 is None
        or thresholds is None
        or confirmatory_seal_sha256 is None
    ):
        raise AssertionError("confirmatory threshold seal was not validated before decoding")
    comparisons = _gate(result["metrics"], thresholds)
    result["frozen_threshold_manifest"] = {
        "path": str(args.thresholds),
        "sha256": threshold_manifest_sha256,
    }
    result["thresholds"] = thresholds
    result["gate"] = comparisons
    result["decision"] = "pass" if comparisons["all"] else "stop"
    result["confirmatory_attempt_seal"] = {
        "path": str(CONFIRMATORY_SEAL),
        "sha256": confirmatory_seal_sha256,
    }
    _write_json_once(args.output, result)
    print(
        json.dumps(
            {"output": str(args.output), "metrics": result["metrics"], "gate": comparisons},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
