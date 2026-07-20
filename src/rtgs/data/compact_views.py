"""Strict, RGB-free per-view containers for fitted 2D Gaussian observations.

Each ``.rtgsv`` file is a small ZIP containing the exact
:class:`~rtgs.core.observation2d.GaussianObservationField` NPZ, an integrity-bound camera
record, and optionally a lossless crop-local alpha bitmap.  The complete container is capped,
not merely the inner teacher archive.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs

COMPACT_VIEW_BYTE_CAP = 168_000

_VIEW_SCHEMA = "rtgs.compact_view.v1"
_DATASET_SCHEMA = "rtgs.compact_dataset.v1"
_METADATA_MEMBER = "metadata.json"
_TEACHER_MEMBER = "teacher.npz"
_ALPHA_MEMBER = "alpha.packbits"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VIEW_METADATA_KEYS = frozenset(
    {
        "schema",
        "view_id",
        "byte_cap",
        "camera",
        "calibration_sha256",
        "teacher",
        "alpha",
        "source",
        "preprocessing",
        "semantic_digest",
    }
)
_CAMERA_KEYS = frozenset({"fx", "fy", "cx", "cy", "width", "height", "R", "t"})
_TEACHER_KEYS = frozenset({"member", "bytes", "sha256", "n_gaussians", "n_init"})
_ALPHA_KEYS = frozenset(
    {
        "member",
        "encoding",
        "bitorder",
        "shape",
        "origin",
        "valid_bits",
        "foreground_count",
        "bytes",
        "sha256",
    }
)
_SOURCE_KEYS = frozenset({"rgb", "mask"})
_SOURCE_FILE_KEYS = frozenset({"name", "sha256"})
_PREPROCESSING_KEYS = frozenset({"rgb", "alpha"})
_DATASET_KEYS = frozenset(
    {"schema", "name", "calibration_sha256", "bounds_hint", "views", "semantic_digest"}
)
_DATASET_VIEW_KEYS = frozenset({"view_id", "path", "sha256", "bytes", "n_gaussians", "has_alpha"})
_BOUNDS_KEYS = frozenset({"center", "extent"})
_TEACHER_NPZ_MEMBERS = frozenset(
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


class CompactViewTooLarge(ValueError):
    """Raised when a complete compact-view container exceeds its byte cap."""

    def __init__(self, actual_bytes: int, byte_cap: int) -> None:
        super().__init__(f"compact view is {actual_bytes} bytes, above the {byte_cap}-byte cap")
        self.actual_bytes = actual_bytes
        self.byte_cap = byte_cap


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of an ordinary file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(payload: bytes, *, label: str) -> dict:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys are not exact "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is not a valid identifier")
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _camera_record(camera: Camera) -> dict:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": int(camera.width),
        "height": int(camera.height),
        "R": camera.R.detach().cpu().reshape(-1).tolist(),
        "t": camera.t.detach().cpu().reshape(-1).tolist(),
    }


def _validate_camera_record(value: object, *, label: str = "camera") -> dict:
    record = _exact_keys(value, _CAMERA_KEYS, label=label)
    for key in ("fx", "fy", "cx", "cy"):
        scalar = record[key]
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not np.isfinite(float(scalar))
        ):
            raise ValueError(f"{label}.{key} must be finite")
    if float(record["fx"]) <= 0 or float(record["fy"]) <= 0:
        raise ValueError(f"{label} focal lengths must be positive")
    for key in ("width", "height"):
        _positive_int(record[key], label=f"{label}.{key}")
    for key, count in (("R", 9), ("t", 3)):
        entries = record[key]
        if (
            not isinstance(entries, list)
            or len(entries) != count
            or any(
                isinstance(entry, bool)
                or not isinstance(entry, (int, float))
                or not np.isfinite(float(entry))
                for entry in entries
            )
        ):
            raise ValueError(f"{label}.{key} must contain {count} finite numbers")
    return record


def _camera_from_record(record: dict, device: torch.device | str) -> Camera:
    return Camera(
        fx=float(record["fx"]),
        fy=float(record["fy"]),
        cx=float(record["cx"]),
        cy=float(record["cy"]),
        width=int(record["width"]),
        height=int(record["height"]),
        R=torch.tensor(record["R"], dtype=torch.float32, device=device).reshape(3, 3),
        t=torch.tensor(record["t"], dtype=torch.float32, device=device),
    )


def _zip_info(name: str, *, compress_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compress_type
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _source_record(name: str, digest: str) -> dict:
    if Path(name).name != name or not name:
        raise ValueError("source names must be non-empty basenames")
    _digest(digest, label="source digest")
    return {"name": name, "sha256": digest}


def _inspect_teacher_npz(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as error:
        raise ValueError("compact view contains an invalid teacher NPZ") from error
    names = [member.filename for member in members]
    if (
        len(names) != len(set(names))
        or not set(names) <= _TEACHER_NPZ_MEMBERS
        or "metadata_utf8.npy" not in names
    ):
        raise ValueError("compact view teacher NPZ has unsupported members")
    total = 0
    for member in members:
        mode = member.external_attr >> 16
        if (
            member.is_dir()
            or "/" in member.filename
            or "\\" in member.filename
            or stat.S_ISLNK(mode)
            or member.flag_bits & 0x1
        ):
            raise ValueError("compact view teacher NPZ contains an unsafe member")
        if member.file_size > 8_388_608:
            raise ValueError("compact view teacher NPZ member exceeds its safety cap")
        total += member.file_size
        if total > 16_777_216:
            raise ValueError("compact view teacher NPZ exceeds its uncompressed safety cap")


@dataclass(frozen=True, slots=True)
class PackedAlpha:
    """Lossless packed foreground alpha for one observation fit window."""

    payload: bytes
    shape: tuple[int, int]
    origin: tuple[int, int]
    foreground_count: int

    @property
    def valid_bits(self) -> int:
        return self.shape[0] * self.shape[1]

    def crop_mask(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Decode the exact crop-local boolean alpha."""
        packed = np.frombuffer(self.payload, dtype=np.uint8)
        unpacked = np.unpackbits(packed, bitorder="little", count=self.valid_bits)
        return torch.from_numpy(unpacked.reshape(self.shape).astype(np.bool_, copy=True)).to(device)

    def full_mask(
        self,
        canvas_size: tuple[int, int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Decode alpha into a zero-filled ``(height, width)`` canvas."""
        height, width = canvas_size
        x, y = self.origin
        crop_height, crop_width = self.shape
        if x < 0 or y < 0 or x + crop_width > width or y + crop_height > height:
            raise ValueError("alpha crop lies outside the requested canvas")
        result = torch.zeros(height, width, dtype=torch.bool, device=device)
        result[y : y + crop_height, x : x + crop_width] = self.crop_mask(device)
        return result


@dataclass(slots=True)
class CompactView:
    """A strict compact view loaded without Pillow, StructSplat, or CUDA."""

    observation: GaussianObservationField
    camera: Camera
    alpha: PackedAlpha | None
    calibration_sha256: str
    source: dict
    path: Path
    bytes: int
    sha256: str

    @property
    def view_id(self) -> str:
        assert self.observation.view_id is not None
        return self.observation.view_id

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        *,
        byte_cap: int = COMPACT_VIEW_BYTE_CAP,
    ) -> CompactView:
        """Strictly load one integrity-bound compact view."""
        path = Path(path)
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError as error:
            raise ValueError("compact view does not exist") from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError("compact view must be an ordinary file")
        file_bytes = os.lstat(path).st_size
        if file_bytes > byte_cap:
            raise CompactViewTooLarge(file_bytes, byte_cap)
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if len(names) != len(set(names)):
                    raise ValueError("compact view contains duplicate ZIP members")
                allowed_sets = [
                    {_METADATA_MEMBER, _TEACHER_MEMBER},
                    {_METADATA_MEMBER, _TEACHER_MEMBER, _ALPHA_MEMBER},
                ]
                if set(names) not in allowed_sets:
                    raise ValueError("compact view ZIP member set is not exact")
                by_name = {member.filename: member for member in members}
                for member in members:
                    mode = member.external_attr >> 16
                    if (
                        member.is_dir()
                        or member.filename.startswith("/")
                        or "/" in member.filename
                        or "\\" in member.filename
                        or ".." in member.filename
                        or stat.S_ISLNK(mode)
                        or member.flag_bits & 0x1
                    ):
                        raise ValueError("compact view contains an unsafe ZIP member")
                if by_name[_METADATA_MEMBER].file_size > 16_384:
                    raise ValueError("compact view metadata exceeds its safety cap")
                if by_name[_TEACHER_MEMBER].file_size > byte_cap:
                    raise ValueError("compact view teacher exceeds its safety cap")
                if _ALPHA_MEMBER in by_name and by_name[_ALPHA_MEMBER].file_size > 67_108_864:
                    raise ValueError("compact view alpha exceeds its uncompressed safety cap")
                metadata_payload = archive.read(_METADATA_MEMBER)
                teacher_payload = archive.read(_TEACHER_MEMBER)
                alpha_payload = archive.read(_ALPHA_MEMBER) if _ALPHA_MEMBER in by_name else None
        except zipfile.BadZipFile as error:
            raise ValueError("compact view is not a valid ZIP archive") from error

        metadata = _strict_json(metadata_payload, label="compact view metadata")
        _exact_keys(metadata, _VIEW_METADATA_KEYS, label="compact view metadata")
        if metadata["schema"] != _VIEW_SCHEMA:
            raise ValueError("unsupported compact view schema")
        digest_payload = dict(metadata)
        stored_semantic_digest = digest_payload.pop("semantic_digest")
        _digest(stored_semantic_digest, label="compact view semantic_digest")
        if stored_semantic_digest != _sha256(_canonical_json(digest_payload)):
            raise ValueError("compact view semantic digest mismatch")
        view_id = _identifier(metadata["view_id"], label="compact view view_id")
        declared_cap = _positive_int(metadata["byte_cap"], label="compact view byte_cap")
        if declared_cap != byte_cap:
            raise ValueError("compact view byte cap does not match the loader contract")
        calibration_sha256 = _digest(
            metadata["calibration_sha256"],
            label="compact view calibration_sha256",
        )
        camera_record = _validate_camera_record(metadata["camera"])

        teacher = _exact_keys(metadata["teacher"], _TEACHER_KEYS, label="teacher record")
        if teacher["member"] != _TEACHER_MEMBER:
            raise ValueError("teacher member name is not canonical")
        if _positive_int(teacher["bytes"], label="teacher bytes") != len(teacher_payload):
            raise ValueError("teacher byte count mismatch")
        if _digest(teacher["sha256"], label="teacher sha256") != _sha256(teacher_payload):
            raise ValueError("teacher digest mismatch")
        _positive_int(teacher["n_gaussians"], label="teacher n_gaussians")
        if teacher["n_init"] is not None:
            _positive_int(teacher["n_init"], label="teacher n_init")
        _inspect_teacher_npz(teacher_payload)

        source = _exact_keys(metadata["source"], _SOURCE_KEYS, label="source record")
        _validate_source_file(source["rgb"], label="source rgb")
        if source["mask"] is not None:
            _validate_source_file(source["mask"], label="source mask")
        preprocessing = _exact_keys(
            metadata["preprocessing"],
            _PREPROCESSING_KEYS,
            label="preprocessing record",
        )
        if preprocessing["rgb"] != "calibrated_bilinear_undistort":
            raise ValueError("unsupported compact-view RGB preprocessing")

        alpha = _load_alpha(metadata["alpha"], alpha_payload)
        if (alpha is None) != (preprocessing["alpha"] is None):
            raise ValueError("alpha preprocessing declaration mismatch")
        if alpha is not None and (
            preprocessing["alpha"] != "calibrated_nearest_undistort_threshold_gt_0.5"
        ):
            raise ValueError("unsupported compact-view alpha preprocessing")

        with tempfile.TemporaryDirectory(prefix="rtgs-compact-view-") as directory:
            teacher_path = Path(directory) / _TEACHER_MEMBER
            teacher_path.write_bytes(teacher_payload)
            observation = GaussianObservationField.load_npz(
                teacher_path,
                device=device,
                strict=True,
            )
        camera = _camera_from_record(camera_record, device)
        if observation.view_id != view_id:
            raise ValueError("teacher view_id does not match compact-view metadata")
        if observation.n != int(teacher["n_gaussians"]):
            raise ValueError("teacher Gaussian count does not match compact-view metadata")
        if observation.n_init != teacher["n_init"]:
            raise ValueError("teacher initial count does not match compact-view metadata")
        if observation.width != camera.width or observation.height != camera.height:
            raise ValueError("teacher canvas does not match compact-view camera")
        if alpha is not None:
            fit_x, fit_y, fit_width, fit_height = observation.fit_window
            if alpha.origin != (fit_x, fit_y) or alpha.shape != (fit_height, fit_width):
                raise ValueError("alpha crop does not match teacher fit_window")
        return cls(
            observation=observation,
            camera=camera,
            alpha=alpha,
            calibration_sha256=calibration_sha256,
            source=source,
            path=path.resolve(),
            bytes=file_bytes,
            sha256=file_sha256(path),
        )


def _validate_source_file(value: object, *, label: str) -> dict:
    record = _exact_keys(value, _SOURCE_FILE_KEYS, label=label)
    name = record["name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"{label}.name must be a basename")
    _digest(record["sha256"], label=f"{label}.sha256")
    return record


def _load_alpha(value: object, payload: bytes | None) -> PackedAlpha | None:
    if value is None:
        if payload is not None:
            raise ValueError("compact view has undeclared alpha payload")
        return None
    record = _exact_keys(value, _ALPHA_KEYS, label="alpha record")
    if payload is None:
        raise ValueError("compact view is missing its declared alpha payload")
    if (
        record["member"] != _ALPHA_MEMBER
        or record["encoding"] != "numpy.packbits.v1"
        or record["bitorder"] != "little"
    ):
        raise ValueError("unsupported compact-view alpha encoding")
    shape = record["shape"]
    origin = record["origin"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape)
    ):
        raise ValueError("alpha shape must contain two positive integers")
    if (
        not isinstance(origin, list)
        or len(origin) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in origin)
    ):
        raise ValueError("alpha origin must contain two non-negative integers")
    valid_bits = _positive_int(record["valid_bits"], label="alpha valid_bits")
    if valid_bits != shape[0] * shape[1]:
        raise ValueError("alpha valid_bits does not match its shape")
    expected_bytes = (valid_bits + 7) // 8
    if _positive_int(record["bytes"], label="alpha bytes") != expected_bytes:
        raise ValueError("alpha byte count does not match its bit count")
    if len(payload) != expected_bytes:
        raise ValueError("alpha payload length mismatch")
    if _digest(record["sha256"], label="alpha sha256") != _sha256(payload):
        raise ValueError("alpha digest mismatch")
    foreground_count = record["foreground_count"]
    if (
        not isinstance(foreground_count, int)
        or isinstance(foreground_count, bool)
        or not 0 < foreground_count <= valid_bits
    ):
        raise ValueError("alpha foreground_count is invalid")
    remainder = valid_bits % 8
    if remainder and payload[-1] >> remainder:
        raise ValueError("alpha payload has non-zero padding bits")
    alpha = PackedAlpha(
        payload=payload,
        shape=(shape[0], shape[1]),
        origin=(origin[0], origin[1]),
        foreground_count=foreground_count,
    )
    if int(alpha.crop_mask().sum()) != foreground_count:
        raise ValueError("alpha foreground_count does not match its payload")
    return alpha


def save_compact_view(
    path: str | Path,
    observation: GaussianObservationField,
    camera: Camera,
    *,
    calibration_sha256: str,
    source_rgb_name: str,
    source_rgb_sha256: str,
    alpha_crop: torch.Tensor | np.ndarray | None = None,
    source_mask_name: str | None = None,
    source_mask_sha256: str | None = None,
    byte_cap: int = COMPACT_VIEW_BYTE_CAP,
    overwrite: bool = False,
) -> int:
    """Atomically serialize one complete view and enforce the total byte cap."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite compact view: {path}")
    if observation.view_id is None:
        raise ValueError("compact-view observations require a view_id")
    if observation.width != camera.width or observation.height != camera.height:
        raise ValueError("observation canvas does not match camera")
    _digest(calibration_sha256, label="calibration_sha256")
    _positive_int(byte_cap, label="byte_cap")
    rgb_record = _source_record(source_rgb_name, source_rgb_sha256)
    if (source_mask_name is None) != (source_mask_sha256 is None):
        raise ValueError("source mask name and digest must occur together")
    mask_record = (
        None
        if source_mask_name is None
        else _source_record(source_mask_name, source_mask_sha256 or "")
    )

    alpha_payload = None
    alpha_record = None
    if alpha_crop is not None:
        alpha_array = np.asarray(
            alpha_crop.detach().cpu().numpy() if torch.is_tensor(alpha_crop) else alpha_crop
        )
        if alpha_array.ndim != 2:
            raise ValueError("alpha_crop must be two-dimensional")
        alpha_array = np.ascontiguousarray(alpha_array.astype(np.bool_, copy=False))
        fit_x, fit_y, fit_width, fit_height = observation.fit_window
        if alpha_array.shape != (fit_height, fit_width):
            raise ValueError("alpha_crop shape must match observation fit_window")
        foreground_count = int(alpha_array.sum())
        if foreground_count <= 0:
            raise ValueError("alpha_crop must contain foreground")
        alpha_payload = np.packbits(
            alpha_array.reshape(-1),
            bitorder="little",
        ).tobytes()
        alpha_record = {
            "member": _ALPHA_MEMBER,
            "encoding": "numpy.packbits.v1",
            "bitorder": "little",
            "shape": [fit_height, fit_width],
            "origin": [fit_x, fit_y],
            "valid_bits": fit_height * fit_width,
            "foreground_count": foreground_count,
            "bytes": len(alpha_payload),
            "sha256": _sha256(alpha_payload),
        }
    if (alpha_payload is None) != (mask_record is None):
        raise ValueError("lossless alpha must be present exactly when a source mask is declared")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    teacher_temporary: Path | None = None
    try:
        descriptor, teacher_name = tempfile.mkstemp(
            prefix=f".{path.name}.teacher.",
            suffix=".npz",
            dir=path.parent,
        )
        os.close(descriptor)
        teacher_temporary = Path(teacher_name)
        teacher_temporary.unlink()
        observation.save_npz(teacher_temporary)
        teacher_payload = teacher_temporary.read_bytes()
        metadata = {
            "schema": _VIEW_SCHEMA,
            "view_id": observation.view_id,
            "byte_cap": byte_cap,
            "camera": _camera_record(camera),
            "calibration_sha256": calibration_sha256,
            "teacher": {
                "member": _TEACHER_MEMBER,
                "bytes": len(teacher_payload),
                "sha256": _sha256(teacher_payload),
                "n_gaussians": observation.n,
                "n_init": observation.n_init,
            },
            "alpha": alpha_record,
            "source": {"rgb": rgb_record, "mask": mask_record},
            "preprocessing": {
                "rgb": "calibrated_bilinear_undistort",
                "alpha": (
                    None
                    if alpha_payload is None
                    else "calibrated_nearest_undistort_threshold_gt_0.5"
                ),
            },
        }
        metadata["semantic_digest"] = _sha256(_canonical_json(metadata))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(
                _zip_info(_METADATA_MEMBER, compress_type=zipfile.ZIP_DEFLATED),
                _canonical_json(metadata),
                compresslevel=9,
            )
            archive.writestr(
                _zip_info(_TEACHER_MEMBER, compress_type=zipfile.ZIP_STORED),
                teacher_payload,
            )
            if alpha_payload is not None:
                archive.writestr(
                    _zip_info(_ALPHA_MEMBER, compress_type=zipfile.ZIP_DEFLATED),
                    alpha_payload,
                    compresslevel=9,
                )
        actual_bytes = temporary.stat().st_size
        if actual_bytes > byte_cap:
            raise CompactViewTooLarge(actual_bytes, byte_cap)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite compact view: {path}")
        os.replace(temporary, path)
        temporary = None
        try:
            CompactView.load(path, byte_cap=byte_cap)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return actual_bytes
    finally:
        if teacher_temporary is not None:
            teacher_temporary.unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class CompactDataset:
    """An ordered frame of compact views ready for the existing reconstruction pipeline."""

    views: list[CompactView]
    name: str
    calibration_sha256: str
    bounds_hint: tuple[torch.Tensor, float] | None
    path: Path

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def alphas(self) -> list[PackedAlpha | None]:
        return [view.alpha for view in self.views]

    def to_reconstruction_inputs(self) -> ReconstructionInputs:
        """Drop only optional alpha while preserving ordered teachers and cameras."""
        return ReconstructionInputs(
            observations=[view.observation for view in self.views],
            cameras=[view.camera for view in self.views],
            view_names=[view.view_id for view in self.views],
            bounds_hint=self.bounds_hint,
            name=self.name,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: torch.device | str = "cpu",
        *,
        byte_cap: int = COMPACT_VIEW_BYTE_CAP,
    ) -> CompactDataset:
        """Strictly load and verify every view listed by a frame manifest."""
        directory = Path(directory)
        if directory.name != "gaussians2d" and (directory / "gaussians2d").is_dir():
            directory = directory / "gaussians2d"
        try:
            mode = os.lstat(directory).st_mode
        except FileNotFoundError as error:
            raise ValueError("compact dataset does not exist") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("compact dataset root must be an ordinary directory")
        manifest_path = directory / "manifest.json"
        try:
            manifest_mode = os.lstat(manifest_path).st_mode
        except FileNotFoundError as error:
            raise ValueError("compact dataset manifest does not exist") from error
        if stat.S_ISLNK(manifest_mode) or not stat.S_ISREG(manifest_mode):
            raise ValueError("compact dataset manifest must be an ordinary file")
        payload = manifest_path.read_bytes()
        if len(payload) > 1_048_576:
            raise ValueError("compact dataset manifest exceeds its safety cap")
        manifest = _strict_json(payload, label="compact dataset manifest")
        _exact_keys(manifest, _DATASET_KEYS, label="compact dataset manifest")
        if manifest["schema"] != _DATASET_SCHEMA:
            raise ValueError("unsupported compact dataset schema")
        digest_payload = dict(manifest)
        semantic_digest = digest_payload.pop("semantic_digest")
        _digest(semantic_digest, label="compact dataset semantic_digest")
        if semantic_digest != _sha256(_canonical_json(digest_payload)):
            raise ValueError("compact dataset semantic digest mismatch")
        name = _identifier(manifest["name"], label="compact dataset name")
        calibration_sha256 = _digest(
            manifest["calibration_sha256"],
            label="compact dataset calibration_sha256",
        )
        bounds_hint = _load_bounds(manifest["bounds_hint"], device)
        records = manifest["views"]
        if not isinstance(records, list) or not records:
            raise ValueError("compact dataset views must be a non-empty list")
        if len(records) > 256:
            raise ValueError("compact dataset view count exceeds its safety cap")
        views: list[CompactView] = []
        seen: set[str] = set()
        for index, value in enumerate(records):
            record = _exact_keys(
                value,
                _DATASET_VIEW_KEYS,
                label=f"compact dataset view {index}",
            )
            view_id = _identifier(record["view_id"], label=f"compact dataset view {index} id")
            if view_id in seen:
                raise ValueError("compact dataset has duplicate view identifiers")
            seen.add(view_id)
            expected_path = f"{view_id}.rtgsv"
            if record["path"] != expected_path:
                raise ValueError("compact dataset view path is not canonical")
            bundle_path = directory / expected_path
            if (
                _positive_int(record["bytes"], label=f"compact dataset view {index} bytes")
                != os.lstat(bundle_path).st_size
            ):
                raise ValueError("compact dataset view byte count mismatch")
            if _digest(
                record["sha256"],
                label=f"compact dataset view {index} sha256",
            ) != file_sha256(bundle_path):
                raise ValueError("compact dataset view digest mismatch")
            n_gaussians = _positive_int(
                record["n_gaussians"],
                label=f"compact dataset view {index} n_gaussians",
            )
            if not isinstance(record["has_alpha"], bool):
                raise ValueError(f"compact dataset view {index} has_alpha must be boolean")
            view = CompactView.load(bundle_path, device=device, byte_cap=byte_cap)
            if view.view_id != view_id:
                raise ValueError("compact dataset view identifier mismatch")
            if view.calibration_sha256 != calibration_sha256:
                raise ValueError("compact dataset calibration binding mismatch")
            if view.observation.n != n_gaussians:
                raise ValueError("compact dataset Gaussian count mismatch")
            if (view.alpha is not None) != record["has_alpha"]:
                raise ValueError("compact dataset alpha declaration mismatch")
            views.append(view)
        actual_names = {
            path.name for path in directory.iterdir() if path.is_file() and path.suffix == ".rtgsv"
        }
        expected_names = {f"{view.view_id}.rtgsv" for view in views}
        if actual_names != expected_names:
            raise ValueError("compact dataset contains unlisted view bundles")
        result = cls(
            views=views,
            name=name,
            calibration_sha256=calibration_sha256,
            bounds_hint=bounds_hint,
            path=directory.resolve(),
        )
        result.to_reconstruction_inputs().validate()
        return result


def _load_bounds(
    value: object,
    device: torch.device | str,
) -> tuple[torch.Tensor, float] | None:
    if value is None:
        return None
    record = _exact_keys(value, _BOUNDS_KEYS, label="compact dataset bounds")
    center = record["center"]
    extent = record["extent"]
    if (
        not isinstance(center, list)
        or len(center) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(float(item))
            for item in center
        )
    ):
        raise ValueError("compact dataset bounds center must contain three finite numbers")
    if (
        isinstance(extent, bool)
        or not isinstance(extent, (int, float))
        or not np.isfinite(float(extent))
        or float(extent) <= 0
    ):
        raise ValueError("compact dataset bounds extent must be finite and positive")
    return torch.tensor(center, dtype=torch.float32, device=device), float(extent)


def write_compact_dataset_manifest(
    directory: str | Path,
    *,
    name: str,
    calibration_sha256: str,
    view_paths: list[Path],
    bounds_hint: tuple[torch.Tensor, float] | None,
    overwrite: bool = False,
) -> Path:
    """Write an ordered, integrity-bound manifest after all view files verify."""
    directory = Path(directory)
    _identifier(name, label="compact dataset name")
    _digest(calibration_sha256, label="compact dataset calibration_sha256")
    if not view_paths:
        raise ValueError("compact dataset manifest requires at least one view")
    views = [CompactView.load(path) for path in view_paths]
    if len({view.view_id for view in views}) != len(views):
        raise ValueError("compact dataset view identifiers must be unique")
    if any(view.calibration_sha256 != calibration_sha256 for view in views):
        raise ValueError("compact dataset views do not share the calibration binding")
    bounds_record = None
    if bounds_hint is not None:
        center, extent = bounds_hint
        if center.shape != (3,) or not bool(torch.isfinite(center).all()) or extent <= 0:
            raise ValueError("bounds_hint is invalid")
        bounds_record = {
            "center": center.detach().cpu().tolist(),
            "extent": float(extent),
        }
    manifest = {
        "schema": _DATASET_SCHEMA,
        "name": name,
        "calibration_sha256": calibration_sha256,
        "bounds_hint": bounds_record,
        "views": [
            {
                "view_id": view.view_id,
                "path": f"{view.view_id}.rtgsv",
                "sha256": view.sha256,
                "bytes": view.bytes,
                "n_gaussians": view.observation.n,
                "has_alpha": view.alpha is not None,
            }
            for view in views
        ],
    }
    manifest["semantic_digest"] = _sha256(_canonical_json(manifest))
    path = directory / "manifest.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite compact dataset manifest: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.",
        suffix=".tmp",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    CompactDataset.load(directory)
    return path
