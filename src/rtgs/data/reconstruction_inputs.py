"""RGB-free calibrated inputs consumed after per-view observation fitting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField

if TYPE_CHECKING:
    from rtgs.data.scene import SceneData

_SCHEMA = "rtgs.reconstruction_inputs.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MANIFEST_KEYS = frozenset(
    {"schema", "name", "views", "geometry", "calibration_digest", "semantic_digest"}
)
_VIEW_KEYS = frozenset({"view_id", "teacher", "teacher_sha256", "n_init_2d", "n_opt_2d", "camera"})
_CAMERA_KEYS = frozenset({"fx", "fy", "cx", "cy", "width", "height", "R", "t"})
_GEOMETRY_KEYS = frozenset({"path", "sha256"})
_TEACHER_ZIP_MEMBERS = frozenset(
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
_GEOMETRY_ZIP_MEMBERS = frozenset(
    {
        "points.npy",
        "visibility_offsets.npy",
        "visibility_indices.npy",
        "bounds_center.npy",
        "bounds_extent.npy",
    }
)


@dataclass(frozen=True, slots=True)
class BundleLoadLimits:
    """Fail-closed limits for opt-in strict reconstruction-bundle loading."""

    max_manifest_bytes: int = 8_388_608
    max_teacher_archives: int = 64
    max_archive_compressed_bytes: int = 268_435_456
    max_total_compressed_bytes: int = 2_147_483_648
    max_zip_members: int = 64
    max_member_uncompressed_bytes: int = 268_435_456
    max_archive_uncompressed_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class NpzArchiveStats:
    """Central-directory statistics collected without decompressing an NPZ archive."""

    relative_path: str
    compressed_bytes: int
    member_count: int
    uncompressed_bytes: int
    max_member_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class BundleArchiveStats:
    """Strict bundle preflight evidence retained across device transfers."""

    manifest_bytes: int
    teacher_archives: tuple[NpzArchiveStats, ...]
    geometry_archive: NpzArchiveStats | None

    @property
    def archives(self) -> tuple[NpzArchiveStats, ...]:
        if self.geometry_archive is None:
            return self.teacher_archives
        return (*self.teacher_archives, self.geometry_archive)

    @property
    def total_compressed_bytes(self) -> int:
        return sum(item.compressed_bytes for item in self.archives)

    @property
    def total_uncompressed_bytes(self) -> int:
        return sum(item.uncompressed_bytes for item in self.archives)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(payload: bytes, *, label: str) -> dict:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys are not exact (missing={missing}, extra={extra})")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_IDENTIFIER.pattern!r}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_bundle_root(directory: Path) -> Path:
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError as error:
        raise ValueError("strict reconstruction bundle does not exist") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("strict reconstruction bundle root must be an ordinary directory")
    return directory.resolve(strict=True)


def _strict_regular_file(root: Path, relative_text: str, *, label: str) -> Path:
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative_text
    ):
        raise ValueError(f"{label} path must be a canonical relative POSIX path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{label} path does not exist") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} path must not contain symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the reconstruction bundle") from error
    if not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise ValueError(f"{label} path must name an ordinary file")
    return resolved


def _inspect_npz(
    path: Path,
    relative_path: str,
    *,
    limits: BundleLoadLimits,
    allowed_members: frozenset[str],
) -> NpzArchiveStats:
    compressed_bytes = os.lstat(path).st_size
    if compressed_bytes > limits.max_archive_compressed_bytes:
        raise ValueError(
            f"archive compressed byte cap exceeded for {relative_path!r}: "
            f"{compressed_bytes} > {limits.max_archive_compressed_bytes}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid NPZ archive {relative_path!r}") from error
    if len(members) > limits.max_zip_members:
        raise ValueError(
            f"ZIP member cap exceeded for {relative_path!r}: "
            f"{len(members)} > {limits.max_zip_members}"
        )
    names = [item.filename for item in members]
    if len(set(names)) != len(names):
        raise ValueError(f"NPZ archive {relative_path!r} contains duplicate members")
    if not set(names) <= allowed_members:
        raise ValueError(f"NPZ archive {relative_path!r} contains unsupported members")
    total_uncompressed = 0
    max_member = 0
    for item in members:
        mode = item.external_attr >> 16
        if (
            item.is_dir()
            or "/" in item.filename
            or "\\" in item.filename
            or stat.S_ISLNK(mode)
            or item.flag_bits & 0x1
        ):
            raise ValueError(f"NPZ archive {relative_path!r} contains an unsafe member")
        if item.file_size > limits.max_member_uncompressed_bytes:
            raise ValueError(
                f"ZIP member uncompressed byte cap exceeded for {relative_path!r}: "
                f"{item.file_size} > {limits.max_member_uncompressed_bytes}"
            )
        total_uncompressed += item.file_size
        max_member = max(max_member, item.file_size)
        if total_uncompressed > limits.max_archive_uncompressed_bytes:
            raise ValueError(
                f"archive uncompressed byte cap exceeded for {relative_path!r}: "
                f"{total_uncompressed} > {limits.max_archive_uncompressed_bytes}"
            )
    return NpzArchiveStats(
        relative_path=relative_path,
        compressed_bytes=compressed_bytes,
        member_count=len(members),
        uncompressed_bytes=total_uncompressed,
        max_member_uncompressed_bytes=max_member,
    )


def _validate_strict_camera_record(value: object, *, label: str) -> dict:
    record = _require_exact_keys(value, _CAMERA_KEYS, label=label)
    for name in ("fx", "fy", "cx", "cy"):
        scalar = record[name]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise ValueError(f"{label}.{name} must be a finite number")
        if not np.isfinite(float(scalar)):
            raise ValueError(f"{label}.{name} must be a finite number")
    if float(record["fx"]) <= 0 or float(record["fy"]) <= 0:
        raise ValueError(f"{label} focal lengths must be positive")
    for name in ("width", "height"):
        value = record[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label}.{name} must be a positive integer")
    for name, length in (("R", 9), ("t", 3)):
        values = record[name]
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not np.isfinite(float(item))
                for item in values
            )
        ):
            raise ValueError(f"{label}.{name} must contain {length} finite numbers")
    return record


def _camera_record(camera: Camera) -> dict:
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


def _camera_from_record(record: dict) -> Camera:
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


@dataclass(slots=True)
class ReconstructionInputs:
    """Calibrated compact observations and optional sparse geometry, never source RGB."""

    observations: list[GaussianObservationField]
    cameras: list[Camera]
    view_names: list[str]
    points: torch.Tensor | None = None
    point_visibility: list[torch.Tensor] | None = None
    bounds_hint: tuple[torch.Tensor, float] | None = None
    name: str = "scene"
    archive_stats: BundleArchiveStats | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_views(self) -> int:
        return len(self.observations)

    @property
    def n_opt_2d(self) -> list[int]:
        """Per-view optimized 2D cardinalities."""
        return [field.n for field in self.observations]

    @property
    def n_init_2d(self) -> list[int | None]:
        """Per-view initialization cardinalities when recorded by the provider."""
        return [field.n_init for field in self.observations]

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.archive_stats is not None and not isinstance(
            self.archive_stats, BundleArchiveStats
        ):
            raise TypeError("archive_stats must be BundleArchiveStats when supplied")
        if not self.observations:
            raise ValueError("reconstruction inputs require at least one observation")
        if len(self.cameras) != self.n_views or len(self.view_names) != self.n_views:
            raise ValueError("observations, cameras, and view_names must have equal length")
        if len(set(self.view_names)) != len(self.view_names):
            raise ValueError("view_names must be unique")
        for name, observation, camera in zip(
            self.view_names, self.observations, self.cameras, strict=True
        ):
            if observation.view_id is not None and observation.view_id != name:
                raise ValueError("observation view_id does not match the ordered view name")
            if observation.width != camera.width or observation.height != camera.height:
                raise ValueError("observation canvas does not match its calibrated camera")
        if self.points is not None:
            if self.points.ndim != 2 or self.points.shape[1] != 3:
                raise ValueError("points must have shape (M,3)")
            if not bool(torch.isfinite(self.points).all()):
                raise ValueError("points must be finite")
        if self.point_visibility is not None:
            if self.points is None or len(self.point_visibility) != self.n_views:
                raise ValueError("point_visibility requires points and one entry per view")
            for indices in self.point_visibility:
                if indices.ndim != 1:
                    raise ValueError("point_visibility entries must be one-dimensional")
                if indices.dtype not in {torch.int32, torch.int64}:
                    raise TypeError("point_visibility entries must use an integer dtype")
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= self.points.shape[0]
                ):
                    raise ValueError("point_visibility index is out of range")
        if self.bounds_hint is not None:
            center, extent = self.bounds_hint
            if center.shape != (3,) or not bool(torch.isfinite(center).all()) or extent <= 0:
                raise ValueError("bounds_hint must contain a finite center and positive extent")

    @classmethod
    def from_scene(
        cls,
        scene: SceneData,
        observations: list[GaussianObservationField],
        indices: list[int] | None = None,
    ) -> ReconstructionInputs:
        """Cross the Stage-1 boundary without retaining any image or mask tensors."""
        selected = list(scene.training_views if indices is None else indices)
        if len(observations) == scene.n_views:
            selected_observations = [observations[index] for index in selected]
        elif len(observations) == len(selected):
            selected_observations = list(observations)
        else:
            raise ValueError("observation count must match all scene views or selected views")
        names = (
            [scene.view_names[index] for index in selected]
            if scene.view_names is not None
            else [f"view-{index:04d}" for index in selected]
        )
        return cls(
            observations=selected_observations,
            cameras=[scene.cameras[index] for index in selected],
            view_names=names,
            points=scene.points,
            point_visibility=(
                None
                if scene.point_visibility is None
                else [scene.point_visibility[index] for index in selected]
            ),
            bounds_hint=scene.bounds_hint,
            name=scene.name,
        )

    def to(self, device: torch.device | str) -> ReconstructionInputs:
        hint = None
        if self.bounds_hint is not None:
            hint = (self.bounds_hint[0].to(device), self.bounds_hint[1])
        return ReconstructionInputs(
            observations=[field.to(device) for field in self.observations],
            cameras=[camera.to(device) for camera in self.cameras],
            view_names=list(self.view_names),
            points=None if self.points is None else self.points.to(device),
            point_visibility=(
                None
                if self.point_visibility is None
                else [indices.to(device) for indices in self.point_visibility]
            ),
            bounds_hint=hint,
            name=self.name,
            archive_stats=self.archive_stats,
        )

    def save(self, directory: str | Path) -> None:
        """Atomically save a bundle with no declared RGB, mask, or source-path field."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite reconstruction inputs: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            teachers_dir = temporary / "teachers"
            teachers_dir.mkdir()
            view_records = []
            for index, (name, observation, camera) in enumerate(
                zip(self.view_names, self.observations, self.cameras, strict=True)
            ):
                relative = Path("teachers") / f"{index:04d}.teacher.npz"
                teacher_path = temporary / relative
                observation.save_npz(teacher_path)
                view_records.append(
                    {
                        "view_id": name,
                        "teacher": relative.as_posix(),
                        "teacher_sha256": _file_sha256(teacher_path),
                        "n_init_2d": observation.n_init,
                        "n_opt_2d": observation.n,
                        "camera": _camera_record(camera),
                    }
                )

            geometry_record = self._save_geometry(temporary)
            calibration_payload = [
                {"view_id": record["view_id"], "camera": record["camera"]}
                for record in view_records
            ]
            manifest = {
                "schema": _SCHEMA,
                "name": self.name,
                "views": view_records,
                "geometry": geometry_record,
                "calibration_digest": hashlib.sha256(
                    _canonical_json(calibration_payload)
                ).hexdigest(),
            }
            manifest["semantic_digest"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("wb") as stream:
                stream.write(_canonical_json(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory)
            temporary = None
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: torch.device | str = "cpu",
        *,
        strict: bool = False,
        limits: BundleLoadLimits | None = None,
    ) -> ReconstructionInputs:
        """Load a bundle without image decoders, optionally using fail-closed preflight.

        The default path retains the original schema-v1 compatibility behavior. ``strict=True``
        additionally rejects unknown keys and identifiers, symlinks/resolved escapes, and ZIPs
        exceeding ``limits``. Every referenced archive is inspected before the first ``np.load``.
        """
        directory = Path(directory)
        archive_stats = None
        if not strict and limits is not None:
            raise ValueError("custom bundle limits require strict=True")
        if strict:
            active_limits = BundleLoadLimits() if limits is None else limits
            if not isinstance(active_limits, BundleLoadLimits):
                raise TypeError("limits must be BundleLoadLimits")
            directory = _strict_bundle_root(directory)
            manifest_path = _strict_regular_file(
                directory, "manifest.json", label="reconstruction manifest"
            )
            manifest_bytes = os.lstat(manifest_path).st_size
            if manifest_bytes > active_limits.max_manifest_bytes:
                raise ValueError(
                    "manifest byte cap exceeded: "
                    f"{manifest_bytes} > {active_limits.max_manifest_bytes}"
                )
            manifest = _strict_json_object(
                manifest_path.read_bytes(), label="reconstruction manifest"
            )
            _require_exact_keys(manifest, _MANIFEST_KEYS, label="reconstruction manifest")
        else:
            manifest = json.loads((directory / "manifest.json").read_bytes())
        if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
            raise ValueError("unsupported reconstruction inputs manifest")
        digest_payload = dict(manifest)
        stored_digest = digest_payload.pop("semantic_digest", None)
        if stored_digest != hashlib.sha256(_canonical_json(digest_payload)).hexdigest():
            raise ValueError("reconstruction inputs semantic digest mismatch")
        if strict:
            _require_sha256(stored_digest, label="manifest semantic_digest")
            _require_sha256(manifest.get("calibration_digest"), label="manifest calibration_digest")
            _require_identifier(manifest.get("name"), label="manifest name")
            records = manifest.get("views")
            if not isinstance(records, list) or not records:
                raise ValueError("strict reconstruction manifest requires a non-empty views list")
            if len(records) > active_limits.max_teacher_archives:
                raise ValueError(
                    "teacher archive count cap exceeded: "
                    f"{len(records)} > {active_limits.max_teacher_archives}"
                )

            teacher_paths: list[Path] = []
            view_ids: list[str] = []
            for index, record in enumerate(records):
                record = _require_exact_keys(record, _VIEW_KEYS, label=f"view record {index}")
                view_ids.append(
                    _require_identifier(record["view_id"], label=f"view record {index} view_id")
                )
                expected_teacher = f"teachers/{index:04d}.teacher.npz"
                if record["teacher"] != expected_teacher:
                    raise ValueError(
                        f"view record {index} teacher path must be {expected_teacher!r}"
                    )
                _require_sha256(
                    record["teacher_sha256"], label=f"view record {index} teacher_sha256"
                )
                n_init = record["n_init_2d"]
                if n_init is not None and (
                    not isinstance(n_init, int) or isinstance(n_init, bool) or n_init <= 0
                ):
                    raise ValueError(f"view record {index} n_init_2d is invalid")
                n_opt = record["n_opt_2d"]
                if not isinstance(n_opt, int) or isinstance(n_opt, bool) or n_opt <= 0:
                    raise ValueError(f"view record {index} n_opt_2d is invalid")
                _validate_strict_camera_record(
                    record["camera"], label=f"view record {index} camera"
                )
                teacher_paths.append(
                    _strict_regular_file(
                        directory, expected_teacher, label=f"teacher archive {index}"
                    )
                )
            if len(set(view_ids)) != len(view_ids):
                raise ValueError("strict reconstruction view identifiers must be unique")

            geometry_record = manifest["geometry"]
            geometry_path = None
            if geometry_record is not None:
                geometry_record = _require_exact_keys(
                    geometry_record, _GEOMETRY_KEYS, label="geometry record"
                )
                if geometry_record["path"] != "geometry.npz":
                    raise ValueError("strict geometry path must be exactly 'geometry.npz'")
                _require_sha256(geometry_record["sha256"], label="geometry sha256")
                geometry_path = _strict_regular_file(
                    directory, "geometry.npz", label="geometry archive"
                )

            for index, (record, teacher_path) in enumerate(
                zip(records, teacher_paths, strict=True)
            ):
                if _file_sha256(teacher_path) != record["teacher_sha256"]:
                    raise ValueError(f"teacher archive digest mismatch for view {index}")
            if geometry_path is not None and (
                _file_sha256(geometry_path) != geometry_record["sha256"]
            ):
                raise ValueError("geometry archive digest mismatch")

            teacher_stats = tuple(
                _inspect_npz(
                    path,
                    str(record["teacher"]),
                    limits=active_limits,
                    allowed_members=_TEACHER_ZIP_MEMBERS,
                )
                for record, path in zip(records, teacher_paths, strict=True)
            )
            geometry_stats = (
                None
                if geometry_path is None
                else _inspect_npz(
                    geometry_path,
                    "geometry.npz",
                    limits=active_limits,
                    allowed_members=_GEOMETRY_ZIP_MEMBERS,
                )
            )
            archive_stats = BundleArchiveStats(
                manifest_bytes=manifest_bytes,
                teacher_archives=teacher_stats,
                geometry_archive=geometry_stats,
            )
            if archive_stats.total_compressed_bytes > active_limits.max_total_compressed_bytes:
                raise ValueError(
                    "bundle aggregate compressed byte cap exceeded: "
                    f"{archive_stats.total_compressed_bytes} > "
                    f"{active_limits.max_total_compressed_bytes}"
                )
        observations = []
        cameras = []
        names = []
        calibration_payload = [
            {"view_id": record["view_id"], "camera": record["camera"]}
            for record in manifest["views"]
        ]
        if (
            manifest.get("calibration_digest")
            != hashlib.sha256(_canonical_json(calibration_payload)).hexdigest()
        ):
            raise ValueError("calibration digest mismatch")
        for record in manifest["views"]:
            relative = Path(record["teacher"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("teacher path must remain inside the bundle")
            teacher_path = directory / relative
            if _file_sha256(teacher_path) != record["teacher_sha256"]:
                raise ValueError("teacher archive digest mismatch")
            observation = GaussianObservationField.load_npz(
                teacher_path, device=device, strict=strict
            )
            if observation.view_id is not None and observation.view_id != record["view_id"]:
                raise ValueError("teacher view_id does not match manifest")
            if observation.n != int(record["n_opt_2d"]):
                raise ValueError("teacher cardinality does not match manifest")
            if observation.n_init != record["n_init_2d"]:
                raise ValueError("teacher initialization cardinality does not match manifest")
            observations.append(observation)
            cameras.append(_camera_from_record(record["camera"]).to(device))
            names.append(str(record["view_id"]))
        points, visibility, bounds = cls._load_geometry(
            directory, manifest.get("geometry"), device, strict=strict
        )
        return cls(
            observations=observations,
            cameras=cameras,
            view_names=names,
            points=points,
            point_visibility=visibility,
            bounds_hint=bounds,
            name=str(manifest["name"]),
            archive_stats=archive_stats,
        )

    def _save_geometry(self, directory: Path) -> dict | None:
        if self.points is None and self.bounds_hint is None:
            return None
        arrays: dict[str, np.ndarray] = {}
        if self.points is not None:
            arrays["points"] = self.points.detach().cpu().numpy()
        if self.point_visibility is not None:
            lengths = [indices.numel() for indices in self.point_visibility]
            arrays["visibility_offsets"] = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
            )
            arrays["visibility_indices"] = np.concatenate(
                [
                    indices.detach().cpu().numpy().astype(np.int64)
                    for indices in self.point_visibility
                ]
            )
        if self.bounds_hint is not None:
            arrays["bounds_center"] = self.bounds_hint[0].detach().cpu().numpy()
            arrays["bounds_extent"] = np.asarray([self.bounds_hint[1]], dtype=np.float64)
        path = directory / "geometry.npz"
        np.savez_compressed(path, **arrays)
        return {"path": path.name, "sha256": _file_sha256(path)}

    @staticmethod
    def _load_geometry(
        directory: Path,
        record: dict | None,
        device: torch.device | str,
        *,
        strict: bool = False,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None, tuple[torch.Tensor, float] | None]:
        if record is None:
            return None, None, None
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("geometry path must remain inside the bundle")
        path = directory / relative
        if _file_sha256(path) != record["sha256"]:
            raise ValueError("geometry archive digest mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if strict:
                actual = set(archive.files)
                allowed = {name.removesuffix(".npy") for name in _GEOMETRY_ZIP_MEMBERS}
                if not actual <= allowed:
                    raise ValueError("strict geometry archive has unsupported array names")
                if ("visibility_offsets" in actual) != ("visibility_indices" in actual):
                    raise ValueError("strict geometry visibility arrays must occur together")
                if ("bounds_center" in actual) != ("bounds_extent" in actual):
                    raise ValueError("strict geometry bounds arrays must occur together")
            points = (
                torch.from_numpy(np.asarray(archive["points"]).copy()).to(device)
                if "points" in archive.files
                else None
            )
            visibility = None
            if "visibility_offsets" in archive.files:
                offsets = np.asarray(archive["visibility_offsets"], dtype=np.int64)
                indices = np.asarray(archive["visibility_indices"], dtype=np.int64)
                if strict and (
                    offsets.ndim != 1
                    or offsets.size == 0
                    or offsets[0] != 0
                    or offsets[-1] != indices.size
                    or bool((offsets[1:] < offsets[:-1]).any())
                ):
                    raise ValueError("strict geometry visibility offsets are invalid")
                visibility = [
                    torch.from_numpy(indices[offsets[i] : offsets[i + 1]].copy()).to(device)
                    for i in range(offsets.size - 1)
                ]
            bounds = None
            if "bounds_center" in archive.files:
                bounds = (
                    torch.from_numpy(np.asarray(archive["bounds_center"]).copy()).to(device),
                    float(np.asarray(archive["bounds_extent"])[0]),
                )
        return points, visibility, bounds
