#!/usr/bin/env python3
"""Fresh-root Iteration 2 test of residual topology repair for exact Gaussian fibers."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import resource
import stat
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmarks import inverse_projection_fiber_iter2_transaction as i2tx

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import quat_to_rotmat
from rtgs.data.synthetic import make_gt_gaussians
from rtgs.lift.inverse_projection_fiber import (
    InverseProjectionFiber,
    pairwise_center_cost,
    pairwise_conic_cost,
    pairwise_gaussian_geometry_cost,
    spd_affine_invariant_squared,
)
from rtgs.lift.topology import (
    cyclic_shift_grouped_scores,
    exact_linear_assignment,
    radius_connected_components,
    select_component_representatives,
)
from rtgs.render.projection import project_covariances_ewa

ROOT = Path(
    os.environ.get("RTGS_ITER2_WORKSPACE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
NAMESPACE = "rtgs.inverse-projection-fiber.iter2.v1"
SCENE_ROOTS = (27_688_011, 27_688_012, 27_688_013)
DEPTH_ROOTS = (27_688_111, 27_688_112, 27_688_113)
ORDER_ROOTS = (27_688_211, 27_688_212, 27_688_213)
DEVELOPMENT_ROOTS = (
    (982_011, 982_111, 982_211),
    (982_012, 982_112, 982_212),
    (982_013, 982_113, 982_213),
)
RELABEL_SENTINEL_ROOTS = (981_901, 981_902, 981_903)
OFFICIAL_OUT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter2_RESULT.json"
OFFICIAL_ARTIFACTS = ROOT / "runs/inverse_projection_fiber_iter2_official_20260717"
PREREG = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter2_PREREG.md"
PREREG_SHA256 = "95adcf0f9d03761ca57bb36444a051f5c581e21e06eb399a355437bda9f6d28e"
PREREG_REVIEW = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter2_PREREG_REVIEW.md"
IMPLEMENTATION_REVIEW = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter2_IMPLEMENTATION_REVIEW.json"
)
ITER1_RESULT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json"
ITER1_RESULT_SHA256 = "2601a45d19d1d8a636d3c0db5ef8b14adf5f4137baaf718c86e1f80a84cecf9e"
BOUND_ITER1_ARTIFACTS: dict[Path, str] = {
    Path(
        "benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json"
    ): ITER1_RESULT_SHA256,
    Path(
        "benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.md"
    ): "a108c099ee77dbf42857c3fd2e7b37d06e9d472e0c432c8a010fde7fc48880f3",
    Path(
        "benchmarks/results/20260717_inverse_projection_fiber_iter1e_AUDIT.json"
    ): "c45cdc9a67a61c34796b07388308dd4e678d268ebbcc4062419ccab7c379515a",
    Path(
        "benchmarks/results/20260717_inverse_projection_fiber_iter1e_AUDIT.md"
    ): "3ccaf78782521bafbce57464d678183cae51468efd6c4c6f64400944e7c147a4",
    Path(
        "benchmarks/results/20260717_inverse_projection_fiber_iter1e_EXECUTED_SOURCES.tar"
    ): "cc23e3ab9e95307453e97193d71f84040a832b16b08fb4e9d231f661ecb1f5a5",
}

N_GAUSSIANS = 8
N_FIT_VIEWS = 4
N_HELDOUT_VIEWS = 2
N_HYPOTHESES = N_GAUSSIANS * N_FIT_VIEWS
DEPTH_LOWER = 1.2
DEPTH_UPPER = 3.6
DILATION = 0.3
CONIC_WEIGHT = 0.25
LEARNING_RATE = 0.025
TOPOLOGY_STEP = 400
TOTAL_UPDATES = 600
RECOVERY_UPDATES = 200
CHECKPOINT_INTERVAL = 20
RESIDUAL_THRESHOLD = 0.1
CLUSTER_RADIUS = 0.01


@dataclass(frozen=True)
class CandidateInputs:
    """The complete label-free fitting API available to candidate topology code."""

    cameras: tuple[Camera, ...]
    target_means2d: tuple[torch.Tensor, ...]
    target_covariances2d: tuple[torch.Tensor, ...]
    source_view_indices: torch.Tensor
    source_local_rows: torch.Tensor
    source_means2d: torch.Tensor
    source_covariances2d: torch.Tensor
    initial_depths: torch.Tensor


@dataclass(frozen=True)
class EvaluatorData:
    gt_means: torch.Tensor
    gt_covariances: torch.Tensor
    fit_labels: tuple[torch.Tensor, ...]
    source_labels: torch.Tensor


@dataclass(frozen=True)
class HeldoutRecipe:
    """Non-observation state sufficient to materialize held-out views after release."""

    order_generator_state_before_heldout: torch.Tensor


@dataclass(frozen=True)
class HeldoutData:
    """Held-out observations that must not exist before the durable release receipt."""

    cameras: tuple[Camera, ...]
    means2d: tuple[torch.Tensor, ...]
    covariances2d: tuple[torch.Tensor, ...]
    labels: tuple[torch.Tensor, ...]
    receipt: dict[str, Any]


@dataclass
class TopologyDecision:
    arm: str
    scores: torch.Tensor
    retained: torch.Tensor
    components: tuple[tuple[int, ...], ...]
    representatives: torch.Tensor
    assignments: tuple[torch.Tensor, ...]
    child: InverseProjectionFiber | None
    rejected: bool
    rejection_reason: str | None


@dataclass
class RootState:
    roots: tuple[int, int, int]
    root_directory: Path
    candidate: CandidateInputs
    evaluator: EvaluatorData
    hardmin_400: InverseProjectionFiber
    hardmin_600: InverseProjectionFiber
    proposed: TopologyDecision
    shuffled: TopologyDecision
    oracle: InverseProjectionFiber
    oracle_assignments: tuple[torch.Tensor, ...]
    summaries: dict[str, Any]
    fit_receipt: dict[str, Any]


@dataclass(frozen=True)
class RootInputs:
    """One once-constructed root bundle retained from planning through execution."""

    roots: tuple[int, int, int]
    candidate: CandidateInputs
    evaluator: EvaluatorData
    heldout_recipe: HeldoutRecipe
    receipt: dict[str, Any]


@dataclass(frozen=True)
class MemberSpec:
    """Independent exact schema for one evidence-array member."""

    dtype: str
    shape: tuple[int, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(b"\0")
    digest.update(_canonical_bytes(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    data = _canonical_bytes(payload)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    metadata = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _array_semantic_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(_canonical_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return np.ascontiguousarray(value.detach().cpu().numpy())


def _write_npz_exclusive(
    path: Path,
    arrays: dict[str, torch.Tensor | np.ndarray],
) -> dict[str, Any]:
    numpy_arrays = {name: _as_numpy(value) for name, value in arrays.items()}
    with path.open("xb") as stream:
        np.savez(stream, **numpy_arrays)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    metadata = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "semantic_sha256": _array_semantic_hash(numpy_arrays),
        "members": {
            name: {"dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in sorted(numpy_arrays.items())
        },
    }


def _capture_regular_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Capture one stable regular-file identity and its bytes through the same descriptor."""
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"evidence path is not a regular file: {path}")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise RuntimeError(
                f"evidence path size {before.st_size} exceeds bound {max_bytes}: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        offset = 0
        while remaining:
            block = os.pread(descriptor, min(1 << 20, remaining), offset)
            if not block:
                raise RuntimeError(f"short read while capturing evidence: {path}")
            chunks.append(block)
            offset += len(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        path_state = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise RuntimeError(f"evidence metadata changed during capture: {path}")
        if (path_state.st_dev, path_state.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"evidence path identity changed during capture: {path}")
        return payload, {
            "path": str(path),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": after.st_mode,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def _validate_json_descriptor(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
    expected_payload: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    capture: dict[str, Any] | None = None
    try:
        payload, capture = _capture_regular_bytes(expected_path, max_bytes=64 * 1024 * 1024)
        expected_bytes = _canonical_bytes(expected_payload)
        if payload != expected_bytes:
            errors.append("canonical_payload_mismatch")
        try:
            parsed = json.loads(payload)
        except Exception as error:
            errors.append(f"json_parse:{type(error).__name__}:{error}")
        else:
            if parsed != expected_payload:
                errors.append("parsed_payload_mismatch")
        for key in ("path", "bytes", "sha256", "device", "inode"):
            expected_value = str(expected_path) if key == "path" else capture[key]
            if descriptor.get(key) != expected_value:
                errors.append(f"descriptor_{key}_mismatch")
    except Exception as error:
        errors.append(f"capture:{type(error).__name__}:{error}")
    return {"capture": capture, "errors": errors, "pass": not errors}


_NPZ_CAPTURE_MAX_BYTES = 512 * 1024 * 1024
_NPY_HEADER_MAX_BYTES = 10_000


def _member_spec_nbytes(spec: MemberSpec) -> int:
    dtype = np.dtype(spec.dtype)
    if dtype.str != spec.dtype:
        raise ValueError(f"non-canonical schema dtype {spec.dtype!r}")
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError(f"unsafe schema dtype {spec.dtype!r}")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in spec.shape):
        raise ValueError(f"invalid schema shape {spec.shape!r}")
    return math.prod(spec.shape) * dtype.itemsize


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream,
            max_header_size=_NPY_HEADER_MAX_BYTES,
        )
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
            stream,
            max_header_size=_NPY_HEADER_MAX_BYTES,
        )
    else:
        raise ValueError(f"unsupported npy version {version!r}")
    if not isinstance(shape, tuple) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
    ):
        raise ValueError(f"invalid npy shape {shape!r}")
    return shape, bool(fortran_order), np.dtype(dtype), int(stream.tell())


def _inspect_npz_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    name: str,
    expected: MemberSpec,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    expected_nbytes = _member_spec_nbytes(expected)
    if info.flag_bits & 0x1:
        return None, [f"encrypted_zip_member:{name}"]
    if info.is_dir():
        return None, [f"directory_zip_member:{name}"]
    maximum_file_size = expected_nbytes + _NPY_HEADER_MAX_BYTES + 16
    if info.file_size < 0 or info.file_size > maximum_file_size:
        return None, [f"member_size_bound_exceeded:{name}"]
    try:
        with archive.open(info, mode="r") as stream:
            shape, fortran_order, dtype, header_bytes = _read_npy_header(stream)
    except Exception as error:
        return None, [f"npy_header:{name}:{type(error).__name__}:{error}"]
    observed = {"dtype": dtype.str, "shape": list(shape)}
    if dtype.hasobject:
        errors.append(f"object_dtype:{name}")
    if dtype.fields is not None:
        errors.append(f"structured_dtype:{name}")
    if dtype.subdtype is not None:
        errors.append(f"subdtype:{name}")
    if fortran_order:
        errors.append(f"fortran_order:{name}")
    if dtype.str != expected.dtype:
        errors.append(f"dtype_mismatch:{name}")
    if shape != expected.shape:
        errors.append(f"shape_mismatch:{name}")
    if info.file_size != header_bytes + expected_nbytes:
        errors.append(f"byte_count_mismatch:{name}")
    return observed, errors


def _choose_npz_member_spec(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    name: str,
    allowed: tuple[MemberSpec, ...],
) -> tuple[MemberSpec | None, list[str]]:
    """Choose among a closed set of schemas without trusting the NPY header."""
    if not allowed:
        return None, [f"empty_allowed_schema:{name}"]
    if info.flag_bits & 0x1:
        return None, [f"encrypted_zip_member:{name}"]
    maximum_nbytes = max(_member_spec_nbytes(spec) for spec in allowed)
    if info.file_size < 0 or info.file_size > maximum_nbytes + _NPY_HEADER_MAX_BYTES + 16:
        return None, [f"member_size_bound_exceeded:{name}"]
    try:
        with archive.open(info, mode="r") as stream:
            shape, fortran_order, dtype, header_bytes = _read_npy_header(stream)
    except Exception as error:
        return None, [f"npy_header:{name}:{type(error).__name__}:{error}"]
    errors: list[str] = []
    if dtype.hasobject:
        errors.append(f"object_dtype:{name}")
    if dtype.fields is not None:
        errors.append(f"structured_dtype:{name}")
    if dtype.subdtype is not None:
        errors.append(f"subdtype:{name}")
    if fortran_order:
        errors.append(f"fortran_order:{name}")
    matches = [spec for spec in allowed if spec.dtype == dtype.str and spec.shape == shape]
    if len(matches) != 1:
        errors.append(f"schema_alternative_mismatch:{name}")
        return None, errors
    chosen = matches[0]
    if info.file_size != header_bytes + _member_spec_nbytes(chosen):
        errors.append(f"byte_count_mismatch:{name}")
    return chosen, errors


def _load_npz_payload_arrays(
    payload: bytes,
    expected_specs: dict[str, MemberSpec],
    *,
    exact_members: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], list[str]]:
    """Validate all headers and byte counts before allocating any evidence array."""
    errors: list[str] = []
    observed_members: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    try:
        expected_total = sum(_member_spec_nbytes(spec) for spec in expected_specs.values())
    except Exception as error:
        return {}, {}, [f"invalid_expected_schema:{type(error).__name__}:{error}"]
    if expected_total > _NPZ_CAPTURE_MAX_BYTES:
        return {}, {}, ["expected_uncompressed_size_bound_exceeded"]
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            zip_names = [info.filename for info in infos]
            if len(zip_names) != len(set(zip_names)):
                errors.append("duplicate_zip_member")
            expected_zip_names = {f"{name}.npy" for name in expected_specs}
            observed_zip_names = set(zip_names)
            if exact_members and observed_zip_names != expected_zip_names:
                errors.append("zip_member_set_mismatch")
            if not exact_members and not expected_zip_names.issubset(observed_zip_names):
                errors.append("zip_member_subset_missing")
            if any(not name.endswith(".npy") for name in zip_names):
                errors.append("non_npy_zip_member")
            info_by_name = {info.filename: info for info in infos}
            for name, expected in sorted(expected_specs.items()):
                info = info_by_name.get(f"{name}.npy")
                if info is None:
                    continue
                observed, member_errors = _inspect_npz_member(
                    archive,
                    info,
                    name=name,
                    expected=expected,
                )
                if observed is not None:
                    observed_members[name] = observed
                errors.extend(member_errors)
            if errors:
                return observed_members, {}, errors

            # All members now have an exact independent type/shape/byte-count schema.
            # Reading to EOF verifies each ZIP CRC before an array is retained.
            for name, expected in sorted(expected_specs.items()):
                info = info_by_name[f"{name}.npy"]
                with archive.open(info, mode="r") as stream:
                    shape, fortran_order, dtype, _header_bytes = _read_npy_header(stream)
                    if (
                        shape != expected.shape
                        or fortran_order
                        or dtype.str != expected.dtype
                        or dtype.hasobject
                        or dtype.fields is not None
                        or dtype.subdtype is not None
                    ):
                        raise RuntimeError(f"member header changed between passes: {name}")
                    nbytes = _member_spec_nbytes(expected)
                    raw = stream.read(nbytes)
                    trailing = stream.read(1)
                    if len(raw) != nbytes or trailing:
                        raise RuntimeError(f"member data byte count changed: {name}")
                arrays[name] = (
                    np.frombuffer(raw, dtype=np.dtype(expected.dtype))
                    .reshape(expected.shape)
                    .copy(order="C")
                )
    except Exception as error:
        errors.append(f"npz_parse:{type(error).__name__}:{error}")
        arrays = {}
    return observed_members, arrays, errors


def _load_validated_npz_arrays(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
    expected_specs: dict[str, MemberSpec],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Recapture, independently validate, and then load an exact typed NPZ schema."""
    errors: list[str] = []
    capture: dict[str, Any] | None = None
    observed_members: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    semantic_sha256: str | None = None
    try:
        payload, capture = _capture_regular_bytes(
            expected_path,
            max_bytes=_NPZ_CAPTURE_MAX_BYTES,
        )
        observed_members, arrays, parse_errors = _load_npz_payload_arrays(
            payload,
            expected_specs,
        )
        errors.extend(parse_errors)
        if arrays:
            semantic_sha256 = _array_semantic_hash(arrays)
        if descriptor.get("members") != observed_members:
            errors.append("descriptor_member_metadata_mismatch")
        if descriptor.get("semantic_sha256") != semantic_sha256:
            errors.append("descriptor_semantic_sha256_mismatch")
        for key in ("path", "bytes", "sha256", "device", "inode"):
            expected_value = str(expected_path) if key == "path" else capture[key]
            if descriptor.get(key) != expected_value:
                errors.append(f"descriptor_{key}_mismatch")
    except Exception as error:
        errors.append(f"capture_or_parse:{type(error).__name__}:{error}")
        arrays = {}
    report = {
        "capture": capture,
        "semantic_sha256": semantic_sha256,
        "members": observed_members,
        "errors": errors,
        "pass": not errors,
    }
    return report, arrays


def _validate_npz_descriptor(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
    expected_specs: dict[str, MemberSpec],
) -> dict[str, Any]:
    report, _arrays = _load_validated_npz_arrays(
        descriptor,
        expected_path=expected_path,
        expected_specs=expected_specs,
    )
    return report


def _save_ply(path: Path, model: InverseProjectionFiber) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    model.as_gaussians().save_ply(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _source_manifest(paths: tuple[Path, ...]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        try:
            hashes[relative.as_posix()] = _sha256_file(path)
        except Exception as error:  # pragma: no cover - fail-closed evidence path
            errors[relative.as_posix()] = f"{type(error).__name__}: {error}"
    return {"hashes": hashes, "errors": errors}


def _bound_iter1_result() -> dict[str, Any]:
    for relative, expected in BOUND_ITER1_ARTIFACTS.items():
        observed = _sha256_file(ROOT / relative)
        if observed != expected:
            raise RuntimeError(
                f"bound Iteration 1 artifact hash mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )
    prior = json.loads(ITER1_RESULT.read_bytes())
    if prior.get("status") != "FAIL":
        raise RuntimeError("bound Iteration 1 result no longer records the committed FAIL")
    return prior


def _declared_sources() -> tuple[Path, ...]:
    prior = _bound_iter1_result()
    paths: set[Path] = set(BOUND_ITER1_ARTIFACTS)
    for raw_path in prior["source_observation_end"]["hashes"]:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Iteration 1 source path: {raw_path!r}")
        paths.add(relative)
    paths.update(
        {
            Path("benchmarks/inverse_projection_fiber_iter2.py"),
            Path("benchmarks/inverse_projection_fiber_iter2_launcher.py"),
            Path("benchmarks/inverse_projection_fiber_iter2_transaction.py"),
            Path("benchmarks/inverse_projection_fiber_sealed_archive.py"),
            Path("src/rtgs/data/synthetic.py"),
            Path("src/rtgs/lift/topology.py"),
            Path("tests/test_inverse_projection_topology.py"),
            Path("tests/test_inverse_projection_fiber_iter2.py"),
            Path("tests/test_inverse_projection_fiber_iter2_launcher.py"),
            Path("tests/test_inverse_projection_fiber_iter2_transaction.py"),
            Path("tests/test_inverse_projection_fiber_sealed_archive.py"),
            PREREG.relative_to(ROOT),
            PREREG_REVIEW.relative_to(ROOT),
            Path(
                "benchmarks/results/"
                "20260717_inverse_projection_fiber_iter2_PREREG_REVIEW_INITIAL_FAIL.md"
            ),
        }
    )
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _archive_sources(
    path: Path,
    sources: tuple[Path, ...],
    *,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    with path.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for relative in sources:
                archive.add(source_root / relative, arcname=relative.as_posix(), recursive=False)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "members": len(sources),
    }


def _load_official_attempt() -> tuple[dict[str, Any], dict[str, Any]]:
    raw_public = os.environ.get("RTGS_ITER2_OFFICIAL_ATTEMPT_PUBLIC_JSON")
    if not raw_public:
        raise RuntimeError("official execution requires the pre-import launcher context")
    artifact_fd = _official_fd("RTGS_ITER2_ARTIFACTS_FD", require_directory=True)
    try:
        expected_public = json.loads(raw_public)
    except Exception as error:
        raise RuntimeError("official attempt public descriptor is not valid JSON") from error
    attempt, capture = i2tx.capture_receipt(
        OFFICIAL_ARTIFACTS,
        "ATTEMPT.json",
        "attempt",
        expected_public=expected_public,
        directory_fd=artifact_fd,
    )
    if (
        attempt.get("namespace") != NAMESPACE
        or attempt.get("receipt_domain") != "official"
        or attempt.get("result_path") != str(OFFICIAL_OUT)
        or attempt.get("artifacts_path") != str(OFFICIAL_ARTIFACTS)
        or attempt.get("root_consumption_status") != i2tx.RESERVED_UNCONSUMED
        or attempt.get("roots") != list(i2tx.OFFICIAL_ROOTS)
        or attempt.get("official_phase") != "ATTEMPT_RESERVED"
    ):
        raise RuntimeError("official attempt metadata is outside the frozen execution domain")
    return attempt, capture


def _official_fd(name: str, *, require_directory: bool) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        raise RuntimeError(f"official execution requires inherited descriptor {name}")
    descriptor = int(raw)
    metadata = os.fstat(descriptor)
    valid = stat.S_ISDIR(metadata.st_mode) if require_directory else stat.S_ISREG(metadata.st_mode)
    if not valid:
        kind = "directory" if require_directory else "regular file"
        raise RuntimeError(f"{name} does not identify a {kind}")
    return descriptor


def _loaded_project_modules_receipt(
    snapshot_root: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    raw_archive_fd = os.environ.get("RTGS_ITER2_ARCHIVE_FD")
    if raw_archive_fd is not None:
        return _loaded_sealed_modules_receipt(source_hashes)
    loaded: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path or not (
            module_name == "__main__"
            or module_name == "benchmarks"
            or module_name.startswith("benchmarks.")
            or module_name == "rtgs"
            or module_name.startswith("rtgs.")
        ):
            continue
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(snapshot_root).as_posix()
        except ValueError:
            errors.append(f"project module outside reviewed snapshot: {module_name}={path}")
            continue
        observed = _sha256_file(path)
        expected = source_hashes.get(relative)
        if expected is None:
            errors.append(f"loaded project module omitted from source map: {relative}")
        elif expected != observed:
            errors.append(f"loaded project module hash mismatch: {relative}")
        loaded[module_name] = {"path": relative, "sha256": observed}
    return {
        "snapshot_root": str(snapshot_root),
        "loaded": loaded,
        "errors": errors,
        "pass": not errors,
    }


def _loaded_sealed_modules_receipt(member_hashes: dict[str, str]) -> dict[str, Any]:
    archive_fd = _official_fd("RTGS_ITER2_ARCHIVE_FD", require_directory=False)
    expected_archive_sha256 = os.environ.get("RTGS_ITER2_ARCHIVE_SHA256")
    if (
        expected_archive_sha256 is None
        or len(expected_archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_archive_sha256)
    ):
        raise RuntimeError("official archive SHA-256 is absent or malformed")
    metadata = os.fstat(archive_fd)
    if metadata.st_size < 0 or metadata.st_size > 512 * 1024 * 1024:
        raise RuntimeError("sealed source archive size exceeds the worker bound")
    required_seals = 0x01 | 0x02 | 0x04 | 0x08
    observed_seals = fcntl.fcntl(archive_fd, 1034)
    if observed_seals != required_seals:
        raise RuntimeError("sealed source archive does not have the complete seal mask")
    blocks: list[bytes] = []
    offset = 0
    while offset < metadata.st_size:
        block = os.pread(archive_fd, min(1 << 20, metadata.st_size - offset), offset)
        if not block:
            raise RuntimeError("sealed source archive ended before its captured size")
        blocks.append(block)
        offset += len(block)
    if os.pread(archive_fd, 1, metadata.st_size):
        raise RuntimeError("sealed source archive grew beyond its captured size")
    archive_sha256 = _sha256_bytes(b"".join(blocks))
    if archive_sha256 != expected_archive_sha256:
        raise RuntimeError("sealed source archive hash differs from launcher intent")
    archive_path = f"/proc/self/fd/{archive_fd}"
    proc_metadata = os.stat(archive_path)
    if (proc_metadata.st_dev, proc_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise RuntimeError("archive /proc path differs from inherited descriptor")

    loaded: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    def verify_one(label: str, spec: Any, raw_path: Any) -> None:
        if spec is None or spec.loader is None or type(raw_path) is not str:
            errors.append(f"project module has no sealed loader: {label}")
            return
        prefix = archive_path + "/"
        if not raw_path.startswith(prefix):
            errors.append(f"project module outside sealed archive: {label}={raw_path}")
            return
        member = raw_path[len(prefix) :]
        expected = member_hashes.get(member)
        if expected is None:
            errors.append(f"sealed module omitted from archive manifest: {label}={member}")
            return
        loader_archive = getattr(spec.loader, "archive", None)
        if loader_archive != archive_path:
            errors.append(f"project module loader archive differs: {label}={loader_archive}")
            return
        get_data = getattr(spec.loader, "get_data", None)
        if not callable(get_data):
            errors.append(f"project module loader cannot recapture bytes: {label}")
            return
        payload = get_data(raw_path)
        observed = _sha256_bytes(payload)
        if observed != expected:
            errors.append(f"sealed module hash mismatch: {label}={member}")
            return
        loaded[label] = {"path": member, "sha256": observed}

    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name == "benchmarks"
            or module_name.startswith("benchmarks.")
            or module_name == "rtgs"
            or module_name.startswith("rtgs.")
        ):
            continue
        verify_one(
            module_name,
            getattr(module, "__spec__", None),
            getattr(module, "__file__", None),
        )
    verify_one(
        "benchmarks.inverse_projection_fiber_iter2[worker-entrypoint]",
        globals().get("__spec__"),
        globals().get("__file__"),
    )
    workspace_paths = [
        value
        for value in sys.path
        if value and (value == str(ROOT) or value.startswith(str(ROOT) + os.sep))
    ]
    if workspace_paths:
        errors.append(f"workspace paths remain importable: {workspace_paths}")
    return {
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_size": metadata.st_size,
        "archive_device": metadata.st_dev,
        "archive_inode": metadata.st_ino,
        "archive_seals": observed_seals,
        "loaded": loaded,
        "errors": errors,
        "pass": not errors,
    }


def _rename_exchange(directory_fd: int, first_name: str, second_name: str) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_fd,
            os.fsencode(first_name),
            directory_fd,
            os.fsencode(second_name),
            2,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), first_name, second_name)


def _capture_matches_descriptor(
    capture: dict[str, Any],
    descriptor: dict[str, Any],
    *,
    include_path: bool = True,
) -> bool:
    keys = ["bytes", "sha256", "device", "inode"]
    if include_path:
        keys.append("path")
    return all(capture.get(key) == descriptor.get(key) for key in keys)


def _publish_reserved_result(
    out: Path,
    payload: dict[str, Any],
    *,
    attempt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transaction_id = attempt["transaction_id"]
    recovery = out.parent / f".{out.name}.recovery.{transaction_id}.prepared"
    prepared = _write_json_exclusive(recovery, payload)
    reservation_bytes, reservation_capture = _capture_regular_bytes(
        out,
        max_bytes=64 * 1024 * 1024,
    )
    if reservation_bytes != _canonical_bytes(attempt["reservation_payload"]):
        raise RuntimeError("public result reservation payload changed before exchange")
    if not _capture_matches_descriptor(
        reservation_capture,
        attempt["reservation_descriptor"],
    ):
        raise RuntimeError("public result reservation identity changed before exchange")
    directory_fd = os.open(out.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _rename_exchange(directory_fd, recovery.name, out.name)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _public_bytes, public_capture = _capture_regular_bytes(out, max_bytes=512 * 1024 * 1024)
    _displaced_bytes, displaced_capture = _capture_regular_bytes(
        recovery,
        max_bytes=64 * 1024 * 1024,
    )
    accepted = (
        public_capture["sha256"] == prepared["sha256"]
        and public_capture["device"] == prepared["device"]
        and public_capture["inode"] == prepared["inode"]
        and _capture_matches_descriptor(
            displaced_capture,
            attempt["reservation_descriptor"],
            include_path=False,
        )
    )
    if not accepted:
        rollback_fd = os.open(out.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            _rename_exchange(rollback_fd, recovery.name, out.name)
            os.fsync(rollback_fd)
        finally:
            os.close(rollback_fd)
        raise RuntimeError("owned result exchange was not accepted and was rolled back")
    result_descriptor = {
        key: public_capture[key] for key in ("path", "bytes", "sha256", "device", "inode")
    }
    return result_descriptor, {
        "operation": "renameat2_owned_exchange",
        "accepted": True,
        "transaction_id": transaction_id,
        "prepared": prepared,
        "public": public_capture,
        "displaced_reservation": displaced_capture,
        "reservation_recovery_path": str(recovery),
    }


def _camera_hash(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R_sha256": _tensor_sha256(camera.R),
        "t_sha256": _tensor_sha256(camera.t),
    }


def _ring_camera(view: int) -> Camera:
    """Construct exactly one member of the frozen six-camera synthetic ring."""
    if view < 0 or view >= N_FIT_VIEWS + N_HELDOUT_VIEWS:
        raise ValueError(f"ring-camera view is outside the frozen domain: {view}")
    angle = 2 * math.pi * view / (N_FIT_VIEWS + N_HELDOUT_VIEWS)
    eye = torch.tensor([2.4 * math.cos(angle), 0.6 * math.sin(2 * angle), 2.4 * math.sin(angle)])
    return Camera.look_at(
        eye,
        torch.zeros(3),
        fov_x_deg=45.0,
        width=64,
        height=64,
    )


def _build_root_inputs(
    scene_root: int,
    depth_root: int,
    order_root: int,
) -> tuple[CandidateInputs, EvaluatorData, HeldoutRecipe, dict[str, Any]]:
    cameras = tuple(_ring_camera(view) for view in range(N_FIT_VIEWS))
    scene_generator = torch.Generator(device="cpu").manual_seed(scene_root)
    scene_before = scene_generator.get_state().clone()
    gt = make_gt_gaussians(
        n=N_GAUSSIANS,
        seed=scene_root,
        generator=scene_generator,
    )
    scene_after = scene_generator.get_state().clone()
    gt_means = gt.means.to(dtype=torch.float64, device="cpu")
    rotations = quat_to_rotmat(gt.quats.to(dtype=torch.float64, device="cpu"))
    scales = gt.log_scales.to(dtype=torch.float64, device="cpu").exp()
    rotated = rotations * scales[:, None, :]
    gt_covariances = rotated @ rotated.transpose(-1, -2)

    unpermuted_means: list[torch.Tensor] = []
    unpermuted_covariances: list[torch.Tensor] = []
    for camera in cameras:
        projection = project_covariances_ewa(
            gt_means,
            gt_covariances,
            camera,
            dilation=DILATION,
        )
        if not bool(torch.isfinite(projection.means2d).all()):
            raise RuntimeError("non-finite projected means")
        if not bool((torch.linalg.eigvalsh(projection.covariances2d) > 0).all()):
            raise RuntimeError("projected covariance is not SPD")
        if not bool((projection.depth > 0).all()):
            raise RuntimeError("non-positive projection depth")
        unpermuted_means.append(projection.means2d)
        unpermuted_covariances.append(projection.covariances2d)

    order_generator = torch.Generator(device="cpu").manual_seed(order_root)
    permutations: list[torch.Tensor] = []
    order_records: list[dict[str, Any]] = []
    for view in range(N_FIT_VIEWS):
        before = order_generator.get_state().clone()
        permutation = torch.randperm(N_GAUSSIANS, generator=order_generator)
        after = order_generator.get_state().clone()
        permutations.append(permutation)
        order_records.append(
            {
                "view": view,
                "before_sha256": _tensor_sha256(before),
                "after_sha256": _tensor_sha256(after),
                "permutation": permutation.tolist(),
                "permutation_sha256": _tensor_sha256(permutation),
            }
        )
    order_state_after_fit = order_generator.get_state().clone()
    permuted_means = tuple(
        unpermuted_means[view][permutations[view]] for view in range(N_FIT_VIEWS)
    )
    permuted_covariances = tuple(
        unpermuted_covariances[view][permutations[view]] for view in range(N_FIT_VIEWS)
    )

    source_views = torch.arange(N_FIT_VIEWS, dtype=torch.long).repeat_interleave(N_GAUSSIANS)
    source_rows = torch.arange(N_GAUSSIANS, dtype=torch.long).repeat(N_FIT_VIEWS)
    source_means = torch.cat(permuted_means[:N_FIT_VIEWS], dim=0)
    source_covariances = torch.cat(permuted_covariances[:N_FIT_VIEWS], dim=0)
    depth_generator = torch.Generator(device="cpu").manual_seed(depth_root)
    depth_before = depth_generator.get_state().clone()
    initial_depths = DEPTH_LOWER + (DEPTH_UPPER - DEPTH_LOWER) * torch.rand(
        N_HYPOTHESES,
        generator=depth_generator,
        dtype=torch.float64,
    )
    depth_after = depth_generator.get_state().clone()

    candidate = CandidateInputs(
        cameras=cameras[:N_FIT_VIEWS],
        target_means2d=permuted_means[:N_FIT_VIEWS],
        target_covariances2d=permuted_covariances[:N_FIT_VIEWS],
        source_view_indices=source_views,
        source_local_rows=source_rows,
        source_means2d=source_means,
        source_covariances2d=source_covariances,
        initial_depths=initial_depths,
    )
    evaluator = EvaluatorData(
        gt_means=gt_means,
        gt_covariances=gt_covariances,
        fit_labels=tuple(permutations[:N_FIT_VIEWS]),
        source_labels=torch.cat(permutations[:N_FIT_VIEWS]),
    )
    heldout_recipe = HeldoutRecipe(
        order_generator_state_before_heldout=order_state_after_fit,
    )
    candidate_hashes = {
        "cameras": [_camera_hash(camera) for camera in candidate.cameras],
        "target_means2d": [_tensor_sha256(value) for value in candidate.target_means2d],
        "target_covariances2d": [_tensor_sha256(value) for value in candidate.target_covariances2d],
        "source_view_indices": _tensor_sha256(source_views),
        "source_local_rows": _tensor_sha256(source_rows),
        "source_means2d": _tensor_sha256(source_means),
        "source_covariances2d": _tensor_sha256(source_covariances),
        "initial_depths": _tensor_sha256(initial_depths),
    }
    evaluator_hashes = {
        "gt_means": _tensor_sha256(gt_means),
        "gt_covariances": _tensor_sha256(gt_covariances),
        "fit_labels": [_tensor_sha256(value) for value in evaluator.fit_labels],
        "source_labels": _tensor_sha256(evaluator.source_labels),
    }
    receipt = {
        "scene_root": scene_root,
        "depth_root": depth_root,
        "observation_order_root": order_root,
        "candidate_api_fields": sorted(CandidateInputs.__dataclass_fields__),
        "candidate_hashes": candidate_hashes,
        "evaluator_hashes": evaluator_hashes,
        "heldout_recipe": {
            "materialized": False,
            "order_generator_state_before_heldout_sha256": _tensor_sha256(order_state_after_fit),
        },
        "scene_generator": {
            "before_sha256": _tensor_sha256(scene_before),
            "after_sha256": _tensor_sha256(scene_after),
        },
        "order_generator": order_records,
        "depth_generator": {
            "before_sha256": _tensor_sha256(depth_before),
            "after_sha256": _tensor_sha256(depth_after),
        },
    }
    return candidate, evaluator, heldout_recipe, receipt


def _materialize_heldout(
    evaluator: EvaluatorData,
    recipe: HeldoutRecipe,
) -> HeldoutData:
    """Release held-out cameras, projections, and labels from non-observation recipe state."""
    generator = torch.Generator(device="cpu")
    generator.set_state(recipe.order_generator_state_before_heldout.clone())
    cameras: list[Camera] = []
    means2d: list[torch.Tensor] = []
    covariances2d: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for view in range(N_FIT_VIEWS, N_FIT_VIEWS + N_HELDOUT_VIEWS):
        camera = _ring_camera(view)
        projection = project_covariances_ewa(
            evaluator.gt_means,
            evaluator.gt_covariances,
            camera,
            dilation=DILATION,
        )
        _require_valid_projection(projection, context=f"held-out materialization view {view}")
        before = generator.get_state().clone()
        permutation = torch.randperm(N_GAUSSIANS, generator=generator)
        after = generator.get_state().clone()
        camera_means = projection.means2d[permutation]
        camera_covariances = projection.covariances2d[permutation]
        cameras.append(camera)
        means2d.append(camera_means)
        covariances2d.append(camera_covariances)
        labels.append(permutation)
        records.append(
            {
                "view": view,
                "camera": _camera_hash(camera),
                "before_sha256": _tensor_sha256(before),
                "after_sha256": _tensor_sha256(after),
                "permutation": permutation.tolist(),
                "permutation_sha256": _tensor_sha256(permutation),
                "means2d_sha256": _tensor_sha256(camera_means),
                "covariances2d_sha256": _tensor_sha256(camera_covariances),
            }
        )
    return HeldoutData(
        cameras=tuple(cameras),
        means2d=tuple(means2d),
        covariances2d=tuple(covariances2d),
        labels=tuple(labels),
        receipt={
            "materialized": True,
            "order_generator_state_before_heldout_sha256": _tensor_sha256(
                recipe.order_generator_state_before_heldout
            ),
            "views": records,
            "order_generator_state_after_release_sha256": _tensor_sha256(generator.get_state()),
        },
    )


def _construct_root_inputs_once(
    roots: list[tuple[int, int, int]],
    *,
    on_constructed: Callable[[int, RootInputs], None] | None = None,
) -> list[RootInputs]:
    """Construct every requested root exactly once and retain the resulting objects."""
    bundles: list[RootInputs] = []
    seen: set[tuple[int, int, int]] = set()
    for index, root_values in enumerate(roots):
        if root_values in seen:
            raise RuntimeError(f"duplicate root bundle requested: {root_values}")
        seen.add(root_values)
        candidate, evaluator, heldout_recipe, receipt = _build_root_inputs(*root_values)
        bundle = RootInputs(
            roots=root_values,
            candidate=candidate,
            evaluator=evaluator,
            heldout_recipe=heldout_recipe,
            receipt=receipt,
        )
        bundles.append(bundle)
        if on_constructed is not None:
            on_constructed(index, bundle)
    return bundles


def _new_fiber(candidate: CandidateInputs) -> InverseProjectionFiber:
    return InverseProjectionFiber(
        cameras=candidate.cameras,
        source_view_indices=candidate.source_view_indices,
        source_component_indices=candidate.source_local_rows,
        source_means2d=candidate.source_means2d,
        source_covariances2d=candidate.source_covariances2d,
        initial_depths=candidate.initial_depths,
        depth_lower=DEPTH_LOWER,
        depth_upper=DEPTH_UPPER,
        dilation=DILATION,
    )


def _require_valid_projection(projection: Any, *, context: str) -> None:
    tensors = {
        "means2d": projection.means2d,
        "covariances2d": projection.covariances2d,
        "depth": projection.depth,
    }
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{context}: non-finite projected {name}")
    if not bool((projection.depth > 0).all()):
        raise RuntimeError(f"{context}: non-positive projected loss-view depth")
    eigenvalues = torch.linalg.eigvalsh(projection.covariances2d)
    if not bool(torch.isfinite(eigenvalues).all() and (eigenvalues > 0).all()):
        raise RuntimeError(f"{context}: projected covariance is not finite SPD")


def _fit_costs(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    costs: list[torch.Tensor] = []
    projected_depths: list[torch.Tensor] = []
    for view, camera in enumerate(candidate.cameras):
        projection = model.project(camera)
        _require_valid_projection(projection, context=f"fit view {view}")
        target_mean = candidate.target_means2d[view]
        target_covariance = candidate.target_covariances2d[view]
        if not bool(torch.isfinite(target_mean).all() and torch.isfinite(target_covariance).all()):
            raise RuntimeError(f"fit view {view}: non-finite target geometry")
        if not bool((torch.linalg.eigvalsh(target_covariance) > 0).all()):
            raise RuntimeError(f"fit view {view}: target covariance is not SPD")
        cost = pairwise_gaussian_geometry_cost(
            projection.means2d,
            projection.covariances2d,
            target_mean,
            target_covariance,
            include_conic=True,
            conic_weight=CONIC_WEIGHT,
        )
        if not bool(torch.isfinite(cost).all()):
            raise RuntimeError(f"fit view {view}: non-finite full geometry cost matrix")
        costs.append(cost)
        projected_depths.append(projection.depth)
    return tuple(costs), torch.stack(projected_depths)


def _hardmin_loss(model: InverseProjectionFiber, candidate: CandidateInputs) -> torch.Tensor:
    costs, _projected_depths = _fit_costs(model, candidate)
    selected: list[torch.Tensor] = []
    for view in range(N_FIT_VIEWS):
        rows = (model.source_view_indices != view).nonzero(as_tuple=True)[0]
        selected.append(costs[view][rows].min(dim=1).values)
    return torch.cat(selected).mean()


def _fixed_loss(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    assignments: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    costs, _projected_depths = _fit_costs(model, candidate)
    rows = torch.arange(model.n, dtype=torch.long)
    selected: list[torch.Tensor] = []
    for view in range(N_FIT_VIEWS):
        mask = model.source_view_indices != view
        selected.append(costs[view][rows[mask], assignments[view][mask]])
    return torch.cat(selected).mean()


def _source_residuals(model: InverseProjectionFiber) -> tuple[float, float]:
    means, covariances, _depths = model.source_projection()
    center = torch.linalg.vector_norm(means - model.source_means2d, dim=-1)
    covariance = torch.linalg.matrix_norm(
        covariances - model.source_covariances2d,
        ord="fro",
        dim=(-2, -1),
    ) / torch.linalg.matrix_norm(model.source_covariances2d, ord="fro", dim=(-2, -1))
    return float(center.detach().max()), float(covariance.detach().max())


def _checkpoint(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    *,
    step: int,
    loss: torch.Tensor,
) -> dict[str, Any]:
    means, covariances = model.means_covariances()
    gradients = [parameter.grad for parameter in model.parameters()]
    gradient_norm = math.sqrt(
        sum(
            float(torch.linalg.vector_norm(value.detach())) ** 2
            for value in gradients
            if value is not None
        )
    )
    eigenvalues = torch.linalg.eigvalsh(covariances)
    source_center, source_covariance = _source_residuals(model)
    depths = model.depths()
    with torch.no_grad():
        _costs, loss_view_depths = _fit_costs(model, candidate)
    passed = bool(
        torch.isfinite(loss)
        and torch.isfinite(means).all()
        and torch.isfinite(covariances).all()
        and (eigenvalues > 0).all()
        and (depths >= DEPTH_LOWER).all()
        and (depths <= DEPTH_UPPER).all()
        and all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
        and source_center <= 1e-6
        and source_covariance <= 1e-5
    )
    return {
        "step": step,
        "loss": float(loss.detach()),
        "gradient_l2": gradient_norm,
        "minimum_covariance_eigenvalue": float(eigenvalues.detach().min()),
        "depth_min": float(depths.detach().min()),
        "depth_max": float(depths.detach().max()),
        "loss_view_depth_min": float(loss_view_depths.detach().min()),
        "loss_view_depth_max": float(loss_view_depths.detach().max()),
        "source_center_max_px": source_center,
        "source_covariance_relative_max": source_covariance,
        "pass": passed,
    }


def _train_hardmin(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
) -> tuple[InverseProjectionFiber, InverseProjectionFiber, list[dict[str, Any]], float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    checkpoints: list[dict[str, Any]] = []
    topology_snapshot: InverseProjectionFiber | None = None
    started = time.perf_counter()
    for step in range(TOTAL_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _hardmin_loss(model, candidate)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"hardmin loss is non-finite at step {step}")
        loss.backward()
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint = _checkpoint(model, candidate, step=step, loss=loss)
            if not checkpoint["pass"]:
                raise RuntimeError(f"hardmin checkpoint failed at step {step}")
            checkpoints.append(checkpoint)
        if step == TOPOLOGY_STEP:
            topology_snapshot = model.subset(torch.arange(model.n, dtype=torch.long))
        if step == TOTAL_UPDATES:
            break
        optimizer.step()
    if topology_snapshot is None:
        raise RuntimeError("missing update-400 topology snapshot")
    return topology_snapshot, model, checkpoints, time.perf_counter() - started


def _train_fixed(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    assignments: tuple[torch.Tensor, ...],
    *,
    updates: int,
) -> tuple[InverseProjectionFiber, list[dict[str, Any]], float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _fixed_loss(model, candidate, assignments)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"fixed loss is non-finite at step {step}")
        loss.backward()
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint = _checkpoint(model, candidate, step=step, loss=loss)
            if not checkpoint["pass"]:
                raise RuntimeError(f"fixed checkpoint failed at step {step}")
            checkpoints.append(checkpoint)
        if step == updates:
            break
        optimizer.step()
    return model, checkpoints, time.perf_counter() - started


def _fit_evaluation(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    evaluator: EvaluatorData,
    *,
    assignments: tuple[torch.Tensor, ...] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with torch.no_grad():
        costs, projected_depths = _fit_costs(model, candidate)
        source_labels = torch.cat(
            [
                evaluator.fit_labels[int(view)][int(row)][None]
                for view, row in zip(
                    model.source_view_indices.tolist(),
                    model.source_component_indices.tolist(),
                    strict=True,
                )
            ]
        )
        full_assignments = torch.empty((N_FIT_VIEWS, model.n), dtype=torch.long)
        if assignments is None:
            for view in range(N_FIT_VIEWS):
                full_assignments[view] = costs[view].min(dim=1).indices
                own = model.source_view_indices == view
                full_assignments[view, own] = model.source_component_indices[own]
        else:
            full_assignments = torch.stack(assignments)

        correct_matrix = torch.empty((N_FIT_VIEWS, model.n), dtype=torch.bool)
        for view in range(N_FIT_VIEWS):
            labels = evaluator.fit_labels[view][full_assignments[view]]
            correct_matrix[view] = labels == source_labels
        non_source = torch.arange(N_FIT_VIEWS)[:, None] != model.source_view_indices[None, :]
        if assignments is None:
            fitting_accuracy = float(correct_matrix[non_source].double().mean())
        else:
            fitting_accuracy = float(correct_matrix.double().mean())
        complete = correct_matrix.all(dim=0)
        residuals = torch.empty(model.n, dtype=torch.float64)
        for row in range(model.n):
            source_view = int(model.source_view_indices[row])
            residuals[row] = torch.stack(
                [costs[view][row].min() for view in range(N_FIT_VIEWS) if view != source_view]
            ).mean()
        return (
            {
                "fitting_assignment_accuracy": fitting_accuracy,
                "exact_track_fraction": float(complete.double().mean()),
                "complete_correct_count": int(complete.sum()),
                "track_denominator": model.n,
                "assignment_denominator": int(non_source.sum())
                if assignments is None
                else N_FIT_VIEWS * model.n,
            },
            {
                "fit_costs": torch.stack(costs),
                "fit_projected_depths": projected_depths,
                "fit_assignments": full_assignments,
                "fit_correct": correct_matrix,
                "complete_correct": complete,
                "residuals": residuals,
                "source_labels": source_labels,
            },
        )


def _component_ids(components: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    result = torch.full((N_HYPOTHESES,), -1, dtype=torch.long)
    for component_id, members in enumerate(components):
        result[list(members)] = component_id
    return result


def _representative_ids(
    components: tuple[tuple[int, ...], ...],
    representatives: torch.Tensor,
) -> torch.Tensor:
    result = torch.full((N_HYPOTHESES,), -1, dtype=torch.long)
    if len(components) != int(representatives.numel()):
        raise ValueError("every component must have one representative")
    for component, representative in zip(
        components,
        representatives.tolist(),
        strict=True,
    ):
        result[list(component)] = int(representative)
    return result


def _decision_codes(decision: TopologyDecision) -> torch.Tensor:
    # 0 = pruned by strict residual gate, 1 = retained duplicate, 2 = representative.
    result = torch.zeros((N_HYPOTHESES,), dtype=torch.int8)
    result[decision.retained] = 1
    result[decision.representatives] = 2
    return result


def _topology_decision(
    arm: str,
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    scores: torch.Tensor,
) -> TopologyDecision:
    retained = scores < RESIDUAL_THRESHOLD
    retained_indices = retained.nonzero(as_tuple=True)[0]
    means, _covariances = model.means_covariances()
    components = radius_connected_components(
        means.detach(),
        retained_indices,
        radius=CLUSTER_RADIUS,
    )
    representatives = select_component_representatives(
        components,
        scores,
        model.source_view_indices,
        model.source_component_indices,
    )
    if len(components) != N_GAUSSIANS:
        return TopologyDecision(
            arm=arm,
            scores=scores.detach().clone(),
            retained=retained,
            components=components,
            representatives=representatives,
            assignments=(),
            child=None,
            rejected=True,
            rejection_reason=f"component_count={len(components)} expected={N_GAUSSIANS}",
        )
    child = model.subset(representatives)
    assignments: list[torch.Tensor] = []
    costs, _projected_depths = _fit_costs(child, candidate)
    for view in range(N_FIT_VIEWS):
        fixed = {
            row: int(child.source_component_indices[row])
            for row in range(child.n)
            if int(child.source_view_indices[row]) == view
        }
        if len(set(fixed.values())) != len(fixed):
            return TopologyDecision(
                arm=arm,
                scores=scores.detach().clone(),
                retained=retained,
                components=components,
                representatives=representatives,
                assignments=(),
                child=None,
                rejected=True,
                rejection_reason=f"source-row conflict in view {view}",
            )
        assignment = exact_linear_assignment(costs[view], fixed=fixed)
        if sorted(assignment.tolist()) != list(range(N_GAUSSIANS)):
            raise RuntimeError("topology assignment is not bijective")
        if any(int(assignment[row]) != column for row, column in fixed.items()):
            raise RuntimeError("topology assignment violated a source fix")
        assignments.append(assignment)
    return TopologyDecision(
        arm=arm,
        scores=scores.detach().clone(),
        retained=retained,
        components=components,
        representatives=representatives,
        assignments=tuple(assignments),
        child=child,
        rejected=False,
        rejection_reason=None,
    )


def _linear_quantile(values: torch.Tensor, probability: float) -> float:
    return float(
        torch.quantile(
            values.detach().to(torch.float64),
            probability,
            interpolation="linear",
        )
    )


def _selection_metrics(
    decision: TopologyDecision,
    complete_correct: torch.Tensor,
    source_labels: torch.Tensor,
    source_views: torch.Tensor,
    source_rows: torch.Tensor,
    means: torch.Tensor,
) -> dict[str, Any]:
    selected = decision.retained
    true_positive = int((selected & complete_correct).sum())
    positives = int(complete_correct.sum())
    selected_count = int(selected.sum())
    precision = true_positive / selected_count if selected_count else 0.0
    recall = true_positive / positives if positives else 0.0
    selected_labels = source_labels[selected]
    covered_labels = sorted(set(int(value) for value in selected_labels.tolist()))
    coverage = len(covered_labels) / N_GAUSSIANS
    pure_components = 0
    label_components: dict[int, set[int]] = {label: set() for label in covered_labels}
    diameters: list[float] = []
    for component_id, component in enumerate(decision.components):
        labels = {int(source_labels[index]) for index in component}
        pure_components += int(len(labels) == 1)
        for label in labels:
            if label in label_components:
                label_components[label].add(component_id)
        members = torch.tensor(component, dtype=torch.long)
        if members.numel() <= 1:
            diameters.append(0.0)
        else:
            diameters.append(float(torch.cdist(means[members], means[members]).max()))
    purity = pure_components / len(decision.components) if decision.components else 0.0
    completeness = (
        sum(len(component_ids) == 1 for component_ids in label_components.values())
        / len(covered_labels)
        if covered_labels
        else 0.0
    )
    diameter_p90 = (
        _linear_quantile(torch.tensor(diameters, dtype=torch.float64), 0.9) if diameters else 0.0
    )
    return {
        "survivor_count": selected_count,
        "positive_count": positives,
        "true_positive_count": true_positive,
        "survivor_precision": precision,
        "survivor_recall": recall,
        "hidden_mode_coverage": coverage,
        "covered_hidden_modes": covered_labels,
        "component_count": len(decision.components),
        "component_sizes": [len(component) for component in decision.components],
        "component_purity": purity,
        "component_completeness": completeness,
        "component_diameter_p90": diameter_p90,
        "representative_count": int(decision.representatives.numel()),
        "representative_indices": decision.representatives.tolist(),
        "representative_source_views": source_views[decision.representatives].tolist(),
        "representative_source_rows": source_rows[decision.representatives].tolist(),
        "decision_codebook": {
            "0": "pruned_by_strict_residual_gate",
            "1": "retained_duplicate_member",
            "2": "selected_component_representative",
        },
        "topology_rejected": decision.rejected,
        "rejection_reason": decision.rejection_reason,
    }


def _oracle_assignments(
    model: InverseProjectionFiber,
    evaluator: EvaluatorData,
) -> tuple[torch.Tensor, ...]:
    source_labels = evaluator.fit_labels[0][model.source_component_indices]
    assignments: list[torch.Tensor] = []
    for view in range(N_FIT_VIEWS):
        inverse = torch.empty(N_GAUSSIANS, dtype=torch.long)
        inverse[evaluator.fit_labels[view]] = torch.arange(N_GAUSSIANS)
        assignments.append(inverse[source_labels])
    return tuple(assignments)


def _model_arrays(prefix: str, model: InverseProjectionFiber) -> dict[str, torch.Tensor]:
    means, covariances = model.means_covariances()
    source_means, source_covariances, source_depths = model.source_projection()
    return {
        f"{prefix}_means": means.detach().clone(),
        f"{prefix}_covariances": covariances.detach().clone(),
        f"{prefix}_depth_logits": model.depth_logits.detach().clone(),
        f"{prefix}_cross": model.cross.detach().clone(),
        f"{prefix}_log_ray_scale": model.log_ray_scale.detach().clone(),
        f"{prefix}_source_view_indices": model.source_view_indices.detach().clone(),
        f"{prefix}_source_local_rows": model.source_component_indices.detach().clone(),
        f"{prefix}_source_projected_means": source_means.detach().clone(),
        f"{prefix}_source_projected_covariances": source_covariances.detach().clone(),
        f"{prefix}_source_depths": source_depths.detach().clone(),
    }


def _fit_input_arrays(
    candidate: CandidateInputs,
    evaluator: EvaluatorData,
) -> dict[str, torch.Tensor]:
    intrinsics = torch.tensor(
        [[camera.fx, camera.fy, camera.cx, camera.cy] for camera in candidate.cameras],
        dtype=torch.float64,
    )
    image_sizes = torch.tensor(
        [[camera.width, camera.height] for camera in candidate.cameras],
        dtype=torch.int64,
    )
    return {
        "input_fit_camera_R": torch.stack(
            [camera.R.to(torch.float64) for camera in candidate.cameras]
        ),
        "input_fit_camera_t": torch.stack(
            [camera.t.to(torch.float64) for camera in candidate.cameras]
        ),
        "input_fit_camera_intrinsics": intrinsics,
        "input_fit_camera_image_sizes": image_sizes,
        "input_fit_target_means2d": torch.stack(candidate.target_means2d),
        "input_fit_target_covariances2d": torch.stack(candidate.target_covariances2d),
        "input_source_view_indices": candidate.source_view_indices,
        "input_source_local_rows": candidate.source_local_rows,
        "input_source_means2d": candidate.source_means2d,
        "input_source_covariances2d": candidate.source_covariances2d,
        "input_initial_depths": candidate.initial_depths,
        "evaluator_gt_means": evaluator.gt_means,
        "evaluator_gt_covariances": evaluator.gt_covariances,
        "evaluator_fit_labels": torch.stack(evaluator.fit_labels),
        "evaluator_source_labels": evaluator.source_labels,
    }


def _commit_root_input(
    bundle: RootInputs,
    *,
    index: int,
    root_directory: Path,
    artifacts_directory: Path,
    development: bool,
    transaction_id: str | None,
    artifact_fd: int | None,
) -> dict[str, Any]:
    evidence = _write_npz_exclusive(
        root_directory / "input_evidence.npz",
        _fit_input_arrays(bundle.candidate, bundle.evaluator),
    )
    payload = {
        "transaction_id": transaction_id,
        "root_index": index,
        "root_bundle": list(bundle.roots),
        "input": bundle.receipt,
        "input_evidence": evidence,
        "input_evidence_relative_path": f"scene_{bundle.roots[0]}/input_evidence.npz",
    }
    target_name = f"INPUT_ROOT_{index}.json"
    if development:
        receipt = {
            "schema": "inverse_projection_fiber_iter2_development_input_v1",
            "namespace": f"{NAMESPACE}.development",
            "root_consumption_status": "DEVELOPMENT_ONLY",
            **payload,
        }
        publication: dict[str, Any] = {
            "public": _write_json_exclusive(artifacts_directory / target_name, receipt)
        }
    else:
        if transaction_id is None or artifact_fd is None:
            raise RuntimeError("official input commit requires transaction and artifact authority")
        receipt = i2tx.official_domain().make_receipt(
            "input",
            payload,
            root_consumption_status=i2tx.PARTIALLY_CONSUMED,
            roots=bundle.roots,
            official_phase="INPUT_MATERIALIZED",
        )
        publication = i2tx.publish_receipt(
            OFFICIAL_ARTIFACTS,
            target_name,
            "input",
            receipt,
            nonce=f"{index + 1:032x}",
            directory_fd=artifact_fd,
        )
    return {
        "target_name": target_name,
        "receipt": receipt,
        "publication": publication,
        "evidence": evidence,
    }


def _heldout_input_arrays(heldout: HeldoutData) -> dict[str, torch.Tensor]:
    intrinsics = torch.tensor(
        [[camera.fx, camera.fy, camera.cx, camera.cy] for camera in heldout.cameras],
        dtype=torch.float64,
    )
    image_sizes = torch.tensor(
        [[camera.width, camera.height] for camera in heldout.cameras],
        dtype=torch.int64,
    )
    return {
        "heldout_camera_R": torch.stack([camera.R.to(torch.float64) for camera in heldout.cameras]),
        "heldout_camera_t": torch.stack([camera.t.to(torch.float64) for camera in heldout.cameras]),
        "heldout_camera_intrinsics": intrinsics,
        "heldout_camera_image_sizes": image_sizes,
        "heldout_target_means2d": torch.stack(heldout.means2d),
        "heldout_target_covariances2d": torch.stack(heldout.covariances2d),
        "heldout_labels": torch.stack(heldout.labels),
    }


def _semantic_tensor_hash(arrays: dict[str, torch.Tensor]) -> str:
    return _array_semantic_hash({name: _as_numpy(value) for name, value in arrays.items()})


def _learned_state_semantic_sha256(state: RootState) -> str:
    arrays: dict[str, torch.Tensor] = {}
    for prefix, model in (
        ("hardmin_400", state.hardmin_400),
        ("hardmin_600", state.hardmin_600),
        ("oracle", state.oracle),
    ):
        arrays.update(_model_arrays(prefix, model))
    arrays.update(_decision_arrays("proposed", state.proposed))
    arrays.update(_decision_arrays("shuffled", state.shuffled))
    if state.proposed.child is not None:
        arrays.update(_model_arrays("proposed_final", state.proposed.child))
    if state.shuffled.child is not None:
        arrays.update(_model_arrays("shuffled_final", state.shuffled.child))
    return _semantic_tensor_hash(arrays)


def _decision_arrays(prefix: str, decision: TopologyDecision) -> dict[str, torch.Tensor]:
    assignments = (
        torch.stack(decision.assignments)
        if decision.assignments
        else torch.empty((0, 0), dtype=torch.long)
    )
    return {
        f"{prefix}_scores": decision.scores,
        f"{prefix}_retained": decision.retained,
        f"{prefix}_component_ids": _component_ids(decision.components),
        f"{prefix}_representative_for_hypothesis": _representative_ids(
            decision.components,
            decision.representatives,
        ),
        f"{prefix}_decision_codes": _decision_codes(decision),
        f"{prefix}_representatives": decision.representatives,
        f"{prefix}_assignments": assignments,
    }


def _fit_arrays(prefix: str, arrays: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}_{name}": value for name, value in arrays.items()}


_CHECKPOINT_VALUE_FIELDS = (
    "loss",
    "gradient_l2",
    "minimum_covariance_eigenvalue",
    "depth_min",
    "depth_max",
    "loss_view_depth_min",
    "loss_view_depth_max",
    "source_center_max_px",
    "source_covariance_relative_max",
)


def _checkpoint_arrays(
    prefix: str,
    checkpoints: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_checkpoint_steps": torch.tensor(
            [int(checkpoint["step"]) for checkpoint in checkpoints],
            dtype=torch.int64,
        ),
        f"{prefix}_checkpoint_values": torch.tensor(
            [
                [float(checkpoint[field]) for field in _CHECKPOINT_VALUE_FIELDS]
                for checkpoint in checkpoints
            ],
            dtype=torch.float64,
        ),
    }


def _sentinel_topology_snapshot(
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
) -> InverseProjectionFiber:
    """Fit exactly to the frozen topology point without constructing an official root."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    for step in range(TOPOLOGY_STEP + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _hardmin_loss(model, candidate)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"relabel sentinel loss is non-finite at step {step}")
        loss.backward()
        if step == TOPOLOGY_STEP:
            return model
        optimizer.step()
    raise AssertionError("unreachable relabel sentinel training exit")


def _copy_geometry_parameters(
    source: InverseProjectionFiber,
    destination: InverseProjectionFiber,
) -> None:
    source_parameters = dict(source.named_parameters())
    destination_parameters = dict(destination.named_parameters())
    if source_parameters.keys() != destination_parameters.keys():
        raise RuntimeError("fiber parameter sets differ during physical relabel")
    with torch.no_grad():
        for name in source_parameters:
            destination_parameters[name].copy_(source_parameters[name])


def _physical_relabel_candidate(
    candidate: CandidateInputs,
    permutations: tuple[torch.Tensor, ...],
) -> tuple[CandidateInputs, tuple[torch.Tensor, ...]]:
    inverses: list[torch.Tensor] = []
    relabeled_rows = candidate.source_local_rows.clone()
    for view, permutation in enumerate(permutations):
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), dtype=torch.long)
        inverses.append(inverse)
        source_mask = candidate.source_view_indices == view
        relabeled_rows[source_mask] = inverse[candidate.source_local_rows[source_mask]]
    return (
        CandidateInputs(
            cameras=candidate.cameras,
            target_means2d=tuple(
                candidate.target_means2d[view][permutations[view]] for view in range(N_FIT_VIEWS)
            ),
            target_covariances2d=tuple(
                candidate.target_covariances2d[view][permutations[view]]
                for view in range(N_FIT_VIEWS)
            ),
            source_view_indices=candidate.source_view_indices.clone(),
            source_local_rows=relabeled_rows,
            source_means2d=candidate.source_means2d.clone(),
            source_covariances2d=candidate.source_covariances2d.clone(),
            initial_depths=candidate.initial_depths.clone(),
        ),
        tuple(inverses),
    )


def _residual_scores(
    costs: tuple[torch.Tensor, ...],
    source_views: torch.Tensor,
) -> torch.Tensor:
    scores = torch.empty(source_views.numel(), dtype=torch.float64)
    for row, source_view in enumerate(source_views.tolist()):
        scores[row] = torch.stack(
            [costs[view][row].min() for view in range(N_FIT_VIEWS) if view != source_view]
        ).mean()
    return scores


def _relabeling_sentinel(
    candidate: CandidateInputs,
    topology_model: InverseProjectionFiber,
    construction_receipt: dict[str, Any],
) -> dict[str, Any]:
    candidate_fields = sorted(CandidateInputs.__dataclass_fields__)
    forbidden_fields = [
        field for field in candidate_fields if "label" in field or field.startswith("gt")
    ]
    permutations = tuple(
        torch.roll(torch.arange(N_GAUSSIANS, dtype=torch.long), shifts=view + 1)
        for view in range(N_FIT_VIEWS)
    )
    relabeled_candidate, inverses = _physical_relabel_candidate(candidate, permutations)
    relabeled_model = _new_fiber(relabeled_candidate)
    _copy_geometry_parameters(topology_model, relabeled_model)

    original_costs, original_depths = _fit_costs(topology_model, candidate)
    relabeled_costs, relabeled_depths = _fit_costs(relabeled_model, relabeled_candidate)
    physical_costs = tuple(relabeled_costs[view][:, inverses[view]] for view in range(N_FIT_VIEWS))
    original_scores = _residual_scores(
        original_costs,
        topology_model.source_view_indices,
    ).detach()
    relabeled_scores = _residual_scores(
        relabeled_costs,
        relabeled_model.source_view_indices,
    ).detach()
    original_decision = _topology_decision(
        "relabel_sentinel_original",
        topology_model,
        candidate,
        original_scores,
    )
    relabeled_decision = _topology_decision(
        "relabel_sentinel_relabeled",
        relabeled_model,
        relabeled_candidate,
        relabeled_scores,
    )
    tied_components = [
        component
        for component in original_decision.components
        if len(component) > 1
        and torch.unique(original_scores[list(component)]).numel() != len(component)
    ]
    physical_assignments = tuple(
        permutations[view][assignment]
        for view, assignment in enumerate(relabeled_decision.assignments)
    )
    original_means, original_covariances = topology_model.means_covariances()
    relabeled_means, relabeled_covariances = relabeled_model.means_covariances()
    source_rows_undone = torch.empty_like(relabeled_model.source_component_indices)
    for view, permutation in enumerate(permutations):
        mask = relabeled_model.source_view_indices == view
        source_rows_undone[mask] = permutation[relabeled_model.source_component_indices[mask]]
    balanced_reached = bool(
        not original_decision.rejected
        and not relabeled_decision.rejected
        and len(original_decision.assignments) == N_FIT_VIEWS
        and len(relabeled_decision.assignments) == N_FIT_VIEWS
    )
    checks = {
        "candidate_label_denial": not forbidden_fields,
        "disjoint_development_root": tuple(
            construction_receipt[key]
            for key in (
                "scene_root",
                "depth_root",
                "observation_order_root",
            )
        )
        == RELABEL_SENTINEL_ROOTS,
        "physical_target_means_undo": all(
            torch.equal(
                relabeled_candidate.target_means2d[view][inverses[view]],
                candidate.target_means2d[view],
            )
            for view in range(N_FIT_VIEWS)
        ),
        "physical_target_covariances_undo": all(
            torch.equal(
                relabeled_candidate.target_covariances2d[view][inverses[view]],
                candidate.target_covariances2d[view],
            )
            for view in range(N_FIT_VIEWS)
        ),
        "source_rows_undo": torch.equal(
            source_rows_undone,
            topology_model.source_component_indices,
        ),
        "fitted_geometry_unchanged": torch.equal(original_means, relabeled_means)
        and torch.equal(original_covariances, relabeled_covariances),
        "physical_costs_unchanged": all(
            torch.equal(original, physical)
            for original, physical in zip(original_costs, physical_costs, strict=True)
        ),
        "loss_view_depths_unchanged": torch.equal(original_depths, relabeled_depths),
        "residuals_unchanged": torch.equal(original_scores, relabeled_scores),
        "selection_unchanged": torch.equal(
            original_decision.retained,
            relabeled_decision.retained,
        ),
        "cluster_membership_unchanged": (
            original_decision.components == relabeled_decision.components
        ),
        "cluster_geometry_unchanged": torch.equal(original_means, relabeled_means),
        "no_exact_within_component_score_ties": not tied_components,
        "representatives_unchanged": torch.equal(
            original_decision.representatives,
            relabeled_decision.representatives,
        ),
        "balanced_rematching_reached": balanced_reached,
        "balanced_physical_assignments_unchanged": balanced_reached
        and all(
            torch.equal(original, physical)
            for original, physical in zip(
                original_decision.assignments,
                physical_assignments,
                strict=True,
            )
        ),
    }
    return {
        "sentinel_roots": list(RELABEL_SENTINEL_ROOTS),
        "construction_receipt": construction_receipt,
        "candidate_api_fields": candidate_fields,
        "forbidden_candidate_fields": forbidden_fields,
        "row_permutations_new_to_physical": [item.tolist() for item in permutations],
        "row_inverses_physical_to_new": [item.tolist() for item in inverses],
        "original_scores_sha256": _tensor_sha256(original_scores),
        "relabeled_scores_sha256": _tensor_sha256(relabeled_scores),
        "original_retained": original_decision.retained.tolist(),
        "relabeled_retained": relabeled_decision.retained.tolist(),
        "original_components": [list(item) for item in original_decision.components],
        "relabeled_components": [list(item) for item in relabeled_decision.components],
        "original_representatives": original_decision.representatives.tolist(),
        "relabeled_representatives": relabeled_decision.representatives.tolist(),
        "original_assignments": [item.tolist() for item in original_decision.assignments],
        "relabeled_local_assignments": [item.tolist() for item in relabeled_decision.assignments],
        "relabeled_physical_assignments": [item.tolist() for item in physical_assignments],
        "exact_tie_components": [list(item) for item in tied_components],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_relabeling_sentinel() -> dict[str, Any]:
    candidate, _evaluator, _heldout_recipe, receipt = _build_root_inputs(*RELABEL_SENTINEL_ROOTS)
    snapshot = _sentinel_topology_snapshot(_new_fiber(candidate), candidate)
    return _relabeling_sentinel(candidate, snapshot, receipt)


def _geometry_error_arrays(
    model: InverseProjectionFiber,
    source_labels: torch.Tensor,
    evaluator: EvaluatorData,
) -> dict[str, torch.Tensor]:
    means, covariances = model.means_covariances()
    target_means = evaluator.gt_means[source_labels]
    target_covariances = evaluator.gt_covariances[source_labels]
    center = torch.linalg.vector_norm(means - target_means, dim=-1)
    covariance = spd_affine_invariant_squared(covariances, target_covariances).sqrt()
    if not bool(torch.isfinite(center).all() and torch.isfinite(covariance).all()):
        raise RuntimeError("non-finite per-primitive GT geometry errors")
    return {
        "gt_center_errors": center.detach().clone(),
        "gt_covariance_errors": covariance.detach().clone(),
    }


def _geometry_metrics(
    model: InverseProjectionFiber,
    source_labels: torch.Tensor,
    evaluator: EvaluatorData,
) -> dict[str, Any]:
    errors = _geometry_error_arrays(model, source_labels, evaluator)
    center = errors["gt_center_errors"]
    covariance = errors["gt_covariance_errors"]
    source_center, source_covariance = _source_residuals(model)
    return {
        "gt_center_median": _linear_quantile(center, 0.5),
        "gt_center_p90": _linear_quantile(center, 0.9),
        "gt_covariance_median": _linear_quantile(covariance, 0.5),
        "gt_covariance_p90": _linear_quantile(covariance, 0.9),
        "source_center_max_px": source_center,
        "source_covariance_relative_max": source_covariance,
        "primitive_count": model.n,
    }


def _rejected_geometry_metrics(primitive_count: int) -> dict[str, Any]:
    return {
        "gt_center_median": 0.0,
        "gt_center_p90": 0.0,
        "gt_covariance_median": 0.0,
        "gt_covariance_p90": 0.0,
        "source_center_max_px": 0.0,
        "source_covariance_relative_max": 0.0,
        "primitive_count": primitive_count,
    }


def _committed_geometry(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "gt_center_median",
        "gt_center_p90",
        "gt_covariance_median",
        "gt_covariance_p90",
        "source_center_max_px",
        "source_covariance_relative_max",
        "primitive_count",
    )
    return {key: summary[key] for key in keys}


def _heldout_evaluation(
    model: InverseProjectionFiber,
    source_labels: torch.Tensor,
    heldout: HeldoutData,
    *,
    overcomplete_groups: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    selected_center_costs: list[torch.Tensor] = []
    selected_conic_costs: list[torch.Tensor] = []
    selected_combined_costs: list[torch.Tensor] = []
    correct: list[torch.Tensor] = []
    assignment_arrays: list[torch.Tensor] = []
    center_cost_arrays: list[torch.Tensor] = []
    conic_cost_arrays: list[torch.Tensor] = []
    combined_cost_arrays: list[torch.Tensor] = []
    projected_depth_arrays: list[torch.Tensor] = []
    with torch.no_grad():
        means, covariances = model.means_covariances()
        for heldout_index, camera in enumerate(heldout.cameras):
            projection = project_covariances_ewa(means, covariances, camera, dilation=DILATION)
            _require_valid_projection(
                projection,
                context=f"held-out view {heldout_index}",
            )
            target_mean = heldout.means2d[heldout_index]
            target_covariance = heldout.covariances2d[heldout_index]
            if not bool(
                torch.isfinite(target_mean).all()
                and torch.isfinite(target_covariance).all()
                and (torch.linalg.eigvalsh(target_covariance) > 0).all()
            ):
                raise RuntimeError(f"held-out view {heldout_index}: invalid target geometry")
            center_cost = pairwise_center_cost(
                projection.means2d,
                projection.covariances2d,
                target_mean,
                target_covariance,
            )
            conic_cost = pairwise_conic_cost(
                projection.covariances2d,
                target_covariance,
            )
            combined_cost = center_cost + CONIC_WEIGHT * conic_cost
            if not bool(
                torch.isfinite(center_cost).all()
                and torch.isfinite(conic_cost).all()
                and torch.isfinite(combined_cost).all()
            ):
                raise RuntimeError(f"held-out view {heldout_index}: non-finite cost matrix")
            center_cost_arrays.append(center_cost)
            conic_cost_arrays.append(conic_cost)
            combined_cost_arrays.append(combined_cost)
            projected_depth_arrays.append(projection.depth)
            if overcomplete_groups:
                view_assignment = torch.empty(model.n, dtype=torch.long)
                view_rows: list[torch.Tensor] = []
                for source_view in range(N_FIT_VIEWS):
                    rows = (model.source_view_indices == source_view).nonzero(as_tuple=True)[0]
                    assignment = exact_linear_assignment(combined_cost[rows])
                    view_assignment[rows] = assignment
                    view_rows.append(rows)
                selected_rows = torch.cat(view_rows)
            else:
                view_assignment = exact_linear_assignment(combined_cost)
                selected_rows = torch.arange(model.n, dtype=torch.long)
            assignment_arrays.append(view_assignment)
            selected_center_costs.append(
                center_cost[selected_rows, view_assignment[selected_rows]].mean()
            )
            selected_conic_costs.append(
                conic_cost[selected_rows, view_assignment[selected_rows]].mean()
            )
            selected_combined_costs.append(
                combined_cost[selected_rows, view_assignment[selected_rows]].mean()
            )
            labels = heldout.labels[heldout_index][view_assignment]
            correct.append(labels == source_labels)
    correctness = torch.stack(correct)
    return (
        {
            "heldout_association_accuracy": float(correctness.double().mean()),
            "heldout_center_cost": float(torch.stack(selected_center_costs).mean()),
            "heldout_conic_cost": float(torch.stack(selected_conic_costs).mean()),
            "heldout_geometry_cost": float(torch.stack(selected_combined_costs).mean()),
            "heldout_assignment_denominator": int(correctness.numel()),
        },
        {
            "heldout_center_costs": torch.stack(center_cost_arrays),
            "heldout_conic_costs": torch.stack(conic_cost_arrays),
            "heldout_combined_costs": torch.stack(combined_cost_arrays),
            "heldout_projected_depths": torch.stack(projected_depth_arrays),
            "heldout_assignments": torch.stack(assignment_arrays),
            "heldout_correct": correctness,
        },
    )


def _run_root(
    roots: tuple[int, int, int],
    root_directory: Path,
    *,
    candidate: CandidateInputs,
    evaluator: EvaluatorData,
) -> RootState:
    scene_root, depth_root, order_root = roots
    common = _new_fiber(candidate)
    common_initial_arrays = _model_arrays("common_initial", common)
    common_initial_hash = _semantic_tensor_hash(common_initial_arrays)
    initial_ply = _save_ply(root_directory / "gaussians_init.ply", common)
    oracle = common.subset(torch.arange(N_GAUSSIANS, dtype=torch.long))
    oracle_initial_arrays = _model_arrays("oracle_initial", oracle)
    common_prefix = common.subset(torch.arange(N_GAUSSIANS, dtype=torch.long))
    oracle_initial_pairing = all(
        torch.equal(left, right)
        for left, right in zip(
            common_prefix.state_dict().values(),
            oracle.state_dict().values(),
            strict=True,
        )
    )
    hardmin_400, hardmin_600, hardmin_checkpoints, hardmin_seconds = _train_hardmin(
        common,
        candidate,
    )
    fit_400, arrays_400 = _fit_evaluation(
        hardmin_400,
        candidate,
        evaluator,
        assignments=None,
    )
    fit_600, arrays_600 = _fit_evaluation(
        hardmin_600,
        candidate,
        evaluator,
        assignments=None,
    )

    proposed = _topology_decision(
        "residual_topology_8",
        hardmin_400,
        candidate,
        arrays_400["residuals"],
    )
    shift = 1 + order_root % 7
    shuffled_scores = cyclic_shift_grouped_scores(
        arrays_400["residuals"],
        hardmin_400.source_view_indices,
        hardmin_400.source_component_indices,
        shift=shift,
        group_size=N_GAUSSIANS,
    )
    shuffled = _topology_decision(
        "shuffled_residual_topology",
        hardmin_400,
        candidate,
        shuffled_scores,
    )
    means_400, _covariances_400 = hardmin_400.means_covariances()
    proposed_selection = _selection_metrics(
        proposed,
        arrays_400["complete_correct"],
        arrays_400["source_labels"],
        hardmin_400.source_view_indices,
        hardmin_400.source_component_indices,
        means_400.detach(),
    )
    shuffled_selection = _selection_metrics(
        shuffled,
        arrays_400["complete_correct"],
        arrays_400["source_labels"],
        hardmin_400.source_view_indices,
        hardmin_400.source_component_indices,
        means_400.detach(),
    )

    proposed_checkpoints: list[dict[str, Any]] = []
    proposed_seconds = 0.0
    proposed_fit: dict[str, Any] | None = None
    proposed_arrays: dict[str, torch.Tensor] = {}
    if not proposed.rejected and proposed.child is not None:
        proposed.child, proposed_checkpoints, proposed_seconds = _train_fixed(
            proposed.child,
            candidate,
            proposed.assignments,
            updates=RECOVERY_UPDATES,
        )
        proposed_fit, proposed_arrays = _fit_evaluation(
            proposed.child,
            candidate,
            evaluator,
            assignments=proposed.assignments,
        )

    shuffled_checkpoints: list[dict[str, Any]] = []
    shuffled_seconds = 0.0
    shuffled_fit: dict[str, Any] | None = None
    shuffled_arrays: dict[str, torch.Tensor] = {}
    if not shuffled.rejected and shuffled.child is not None:
        shuffled.child, shuffled_checkpoints, shuffled_seconds = _train_fixed(
            shuffled.child,
            candidate,
            shuffled.assignments,
            updates=RECOVERY_UPDATES,
        )
        shuffled_fit, shuffled_arrays = _fit_evaluation(
            shuffled.child,
            candidate,
            evaluator,
            assignments=shuffled.assignments,
        )

    oracle_assignments = _oracle_assignments(oracle, evaluator)
    oracle, oracle_checkpoints, oracle_seconds = _train_fixed(
        oracle,
        candidate,
        oracle_assignments,
        updates=TOTAL_UPDATES,
    )
    oracle_fit, oracle_arrays = _fit_evaluation(
        oracle,
        candidate,
        evaluator,
        assignments=oracle_assignments,
    )

    hardmin_400_geometry = _geometry_metrics(
        hardmin_400,
        arrays_400["source_labels"],
        evaluator,
    )
    hardmin_600_geometry = _geometry_metrics(
        hardmin_600,
        arrays_600["source_labels"],
        evaluator,
    )
    proposed_geometry = (
        _geometry_metrics(
            proposed.child,
            evaluator.source_labels[proposed.representatives],
            evaluator,
        )
        if proposed.child is not None and not proposed.rejected
        else _rejected_geometry_metrics(int(proposed.representatives.numel()))
    )
    shuffled_geometry = (
        _geometry_metrics(
            shuffled.child,
            evaluator.source_labels[shuffled.representatives],
            evaluator,
        )
        if shuffled.child is not None and not shuffled.rejected
        else _rejected_geometry_metrics(int(shuffled.representatives.numel()))
    )
    oracle_geometry = _geometry_metrics(
        oracle,
        evaluator.fit_labels[0][oracle.source_component_indices],
        evaluator,
    )

    fit_arrays: dict[str, torch.Tensor | np.ndarray] = {}
    fit_arrays.update(_fit_input_arrays(candidate, evaluator))
    fit_arrays.update(common_initial_arrays)
    fit_arrays.update(oracle_initial_arrays)
    fit_arrays.update(_model_arrays("hardmin_400", hardmin_400))
    fit_arrays.update(_fit_arrays("hardmin_400", arrays_400))
    fit_arrays.update(
        _fit_arrays(
            "hardmin_400",
            _geometry_error_arrays(
                hardmin_400,
                arrays_400["source_labels"],
                evaluator,
            ),
        )
    )
    fit_arrays.update(_model_arrays("hardmin_600", hardmin_600))
    fit_arrays.update(_fit_arrays("hardmin_600", arrays_600))
    fit_arrays.update(
        _fit_arrays(
            "hardmin_600",
            _geometry_error_arrays(
                hardmin_600,
                arrays_600["source_labels"],
                evaluator,
            ),
        )
    )
    fit_arrays.update(_decision_arrays("proposed", proposed))
    fit_arrays.update(_decision_arrays("shuffled", shuffled))
    fit_arrays.update(_model_arrays("oracle", oracle))
    fit_arrays.update(_fit_arrays("oracle", oracle_arrays))
    fit_arrays.update(
        _fit_arrays(
            "oracle",
            _geometry_error_arrays(
                oracle,
                evaluator.fit_labels[0][oracle.source_component_indices],
                evaluator,
            ),
        )
    )
    if proposed.child is not None:
        fit_arrays.update(_model_arrays("proposed_final", proposed.child))
        fit_arrays.update(_fit_arrays("proposed_final", proposed_arrays))
        fit_arrays.update(
            _fit_arrays(
                "proposed_final",
                _geometry_error_arrays(
                    proposed.child,
                    evaluator.source_labels[proposed.representatives],
                    evaluator,
                ),
            )
        )
    if shuffled.child is not None:
        fit_arrays.update(_model_arrays("shuffled_final", shuffled.child))
        fit_arrays.update(_fit_arrays("shuffled_final", shuffled_arrays))
        fit_arrays.update(
            _fit_arrays(
                "shuffled_final",
                _geometry_error_arrays(
                    shuffled.child,
                    evaluator.source_labels[shuffled.representatives],
                    evaluator,
                ),
            )
        )
    fit_arrays.update(_checkpoint_arrays("hardmin", hardmin_checkpoints))
    fit_arrays.update(_checkpoint_arrays("oracle", oracle_checkpoints))
    if proposed.child is not None:
        fit_arrays.update(_checkpoint_arrays("proposed", proposed_checkpoints))
    if shuffled.child is not None:
        fit_arrays.update(_checkpoint_arrays("shuffled", shuffled_checkpoints))
    fit_evidence = _write_npz_exclusive(root_directory / "fit_evidence.npz", fit_arrays)
    topology_snapshot_hash = _semantic_tensor_hash(_model_arrays("topology_snapshot", hardmin_400))

    artifacts = {
        "initial_ply": initial_ply,
        "hardmin_600_ply": _save_ply(root_directory / "hardmin_600.ply", hardmin_600),
        "oracle_ply": _save_ply(root_directory / "oracle_8.ply", oracle),
        "fit_evidence": fit_evidence,
    }
    if proposed.child is not None:
        artifacts["proposed_ply"] = _save_ply(
            root_directory / "residual_topology_8.ply",
            proposed.child,
        )
    if shuffled.child is not None:
        artifacts["shuffled_ply"] = _save_ply(
            root_directory / "shuffled_residual_topology.ply",
            shuffled.child,
        )

    summaries = {
        "roots": list(roots),
        "initialization_pairing": {
            "common_initial_semantic_sha256": common_initial_hash,
            "oracle_initial_is_exact_common_first_eight": oracle_initial_pairing,
            "topology_snapshot_semantic_sha256": topology_snapshot_hash,
            "proposed_topology_input_semantic_sha256": topology_snapshot_hash,
            "shuffled_topology_input_semantic_sha256": topology_snapshot_hash,
            "pass": oracle_initial_pairing,
        },
        "hardmin_400": {
            **fit_400,
            **hardmin_400_geometry,
            "checkpoints_through_400": [
                item for item in hardmin_checkpoints if item["step"] <= TOPOLOGY_STEP
            ],
        },
        "hardmin_600": {
            **fit_600,
            **hardmin_600_geometry,
            "checkpoints": hardmin_checkpoints,
            "wall_time_seconds": hardmin_seconds,
        },
        "proposed": {
            "selection": proposed_selection,
            "fit": proposed_fit,
            "geometry": proposed_geometry,
            "checkpoints": proposed_checkpoints,
            "wall_time_seconds": proposed_seconds,
        },
        "shuffled": {
            "shift": shift,
            "selection": shuffled_selection,
            "fit": shuffled_fit,
            "geometry": shuffled_geometry,
            "checkpoints": shuffled_checkpoints,
            "wall_time_seconds": shuffled_seconds,
        },
        "oracle": {
            "fit": oracle_fit,
            "geometry": oracle_geometry,
            "checkpoints": oracle_checkpoints,
            "wall_time_seconds": oracle_seconds,
        },
        "artifacts": artifacts,
    }
    fit_receipt = _write_json_exclusive(root_directory / "FIT_RECEIPT.json", summaries)
    return RootState(
        roots=roots,
        root_directory=root_directory,
        candidate=candidate,
        evaluator=evaluator,
        hardmin_400=hardmin_400,
        hardmin_600=hardmin_600,
        proposed=proposed,
        shuffled=shuffled,
        oracle=oracle,
        oracle_assignments=oracle_assignments,
        summaries=summaries,
        fit_receipt=fit_receipt,
    )


def _evaluate_root_heldout(
    state: RootState,
    root_directory: Path,
    heldout: HeldoutData,
) -> dict[str, Any]:
    evaluator = state.evaluator
    candidate = state.candidate
    hardmin_fit, _hardmin_fit_arrays = _fit_evaluation(
        state.hardmin_600,
        candidate,
        evaluator,
        assignments=None,
    )
    hardmin_source_labels = evaluator.source_labels
    hardmin_geometry = _committed_geometry(state.summaries["hardmin_600"])
    hardmin_heldout, hardmin_heldout_arrays = _heldout_evaluation(
        state.hardmin_600,
        hardmin_source_labels,
        heldout,
        overcomplete_groups=True,
    )

    heldout_arrays: dict[str, torch.Tensor | np.ndarray] = {}
    heldout_arrays.update(_fit_arrays("hardmin", hardmin_heldout_arrays))

    def evaluate_topology(
        prefix: str,
        decision: TopologyDecision,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if decision.rejected or decision.child is None:
            return (
                {
                    "topology_rejected": True,
                    "fitting_assignment_accuracy": 0.0,
                    "exact_track_fraction": 0.0,
                    "complete_correct_count": 0,
                    "track_denominator": 0,
                    "assignment_denominator": 0,
                    "heldout_association_accuracy": 0.0,
                    "heldout_center_cost": 0.0,
                    "heldout_conic_cost": 0.0,
                    "heldout_geometry_cost": 0.0,
                    "heldout_assignment_denominator": 0,
                    **state.summaries[prefix]["geometry"],
                },
                {},
            )
        source_labels = evaluator.source_labels[decision.representatives]
        fit, _fit_arrays_local = _fit_evaluation(
            decision.child,
            candidate,
            evaluator,
            assignments=decision.assignments,
        )
        geometry = state.summaries[prefix]["geometry"]
        heldout_metrics, arrays = _heldout_evaluation(
            decision.child,
            source_labels,
            heldout,
            overcomplete_groups=False,
        )
        return {"topology_rejected": False, **fit, **geometry, **heldout_metrics}, arrays

    proposed_metrics, proposed_arrays = evaluate_topology("proposed", state.proposed)
    shuffled_metrics, shuffled_arrays = evaluate_topology("shuffled", state.shuffled)
    heldout_arrays.update(_fit_arrays("proposed", proposed_arrays))
    heldout_arrays.update(_fit_arrays("shuffled", shuffled_arrays))

    oracle_source_labels = evaluator.fit_labels[0][state.oracle.source_component_indices]
    oracle_fit, _oracle_fit_arrays = _fit_evaluation(
        state.oracle,
        candidate,
        evaluator,
        assignments=state.oracle_assignments,
    )
    oracle_geometry = state.summaries["oracle"]["geometry"]
    oracle_heldout, oracle_heldout_arrays = _heldout_evaluation(
        state.oracle,
        oracle_source_labels,
        heldout,
        overcomplete_groups=False,
    )
    heldout_arrays.update(_fit_arrays("oracle", oracle_heldout_arrays))
    heldout_arrays.update(_heldout_input_arrays(heldout))
    evidence = _write_npz_exclusive(root_directory / "heldout_evidence.npz", heldout_arrays)
    summary = {
        "roots": list(state.roots),
        "heldout_materialization": heldout.receipt,
        "hardmin_600": {**hardmin_fit, **hardmin_geometry, **hardmin_heldout},
        "proposed": proposed_metrics,
        "shuffled": shuffled_metrics,
        "oracle": {**oracle_fit, **oracle_geometry, **oracle_heldout},
        "heldout_evidence": evidence,
    }
    descriptor = _write_json_exclusive(root_directory / "HELDOUT_RECEIPT.json", summary)
    return {**summary, "receipt": descriptor}


_F64_DTYPE = np.dtype(np.float64).str
_I64_DTYPE = np.dtype(np.int64).str
_I8_DTYPE = np.dtype(np.int8).str
_BOOL_DTYPE = np.dtype(np.bool_).str


def _member_spec(dtype: str, *shape: int) -> MemberSpec:
    return MemberSpec(dtype=dtype, shape=tuple(shape))


def _fit_input_member_specs() -> dict[str, MemberSpec]:
    return {
        "input_fit_camera_R": _member_spec(_F64_DTYPE, N_FIT_VIEWS, 3, 3),
        "input_fit_camera_t": _member_spec(_F64_DTYPE, N_FIT_VIEWS, 3),
        "input_fit_camera_intrinsics": _member_spec(_F64_DTYPE, N_FIT_VIEWS, 4),
        "input_fit_camera_image_sizes": _member_spec(_I64_DTYPE, N_FIT_VIEWS, 2),
        "input_fit_target_means2d": _member_spec(_F64_DTYPE, N_FIT_VIEWS, N_GAUSSIANS, 2),
        "input_fit_target_covariances2d": _member_spec(_F64_DTYPE, N_FIT_VIEWS, N_GAUSSIANS, 2, 2),
        "input_source_view_indices": _member_spec(_I64_DTYPE, N_HYPOTHESES),
        "input_source_local_rows": _member_spec(_I64_DTYPE, N_HYPOTHESES),
        "input_source_means2d": _member_spec(_F64_DTYPE, N_HYPOTHESES, 2),
        "input_source_covariances2d": _member_spec(_F64_DTYPE, N_HYPOTHESES, 2, 2),
        "input_initial_depths": _member_spec(_F64_DTYPE, N_HYPOTHESES),
        "evaluator_gt_means": _member_spec(_F64_DTYPE, N_GAUSSIANS, 3),
        "evaluator_gt_covariances": _member_spec(_F64_DTYPE, N_GAUSSIANS, 3, 3),
        "evaluator_fit_labels": _member_spec(_I64_DTYPE, N_FIT_VIEWS, N_GAUSSIANS),
        "evaluator_source_labels": _member_spec(_I64_DTYPE, N_HYPOTHESES),
    }


def _model_member_specs(prefix: str, count: int) -> dict[str, MemberSpec]:
    return {
        f"{prefix}_means": _member_spec(_F64_DTYPE, count, 3),
        f"{prefix}_covariances": _member_spec(_F64_DTYPE, count, 3, 3),
        f"{prefix}_depth_logits": _member_spec(_F64_DTYPE, count),
        f"{prefix}_cross": _member_spec(_F64_DTYPE, count, 2),
        f"{prefix}_log_ray_scale": _member_spec(_F64_DTYPE, count),
        f"{prefix}_source_view_indices": _member_spec(_I64_DTYPE, count),
        f"{prefix}_source_local_rows": _member_spec(_I64_DTYPE, count),
        f"{prefix}_source_projected_means": _member_spec(_F64_DTYPE, count, 2),
        f"{prefix}_source_projected_covariances": _member_spec(_F64_DTYPE, count, 2, 2),
        f"{prefix}_source_depths": _member_spec(_F64_DTYPE, count),
    }


def _fit_member_specs(prefix: str, count: int) -> dict[str, MemberSpec]:
    return {
        f"{prefix}_fit_costs": _member_spec(_F64_DTYPE, N_FIT_VIEWS, count, N_GAUSSIANS),
        f"{prefix}_fit_projected_depths": _member_spec(_F64_DTYPE, N_FIT_VIEWS, count),
        f"{prefix}_fit_assignments": _member_spec(_I64_DTYPE, N_FIT_VIEWS, count),
        f"{prefix}_fit_correct": _member_spec(_BOOL_DTYPE, N_FIT_VIEWS, count),
        f"{prefix}_complete_correct": _member_spec(_BOOL_DTYPE, count),
        f"{prefix}_residuals": _member_spec(_F64_DTYPE, count),
        f"{prefix}_source_labels": _member_spec(_I64_DTYPE, count),
    }


def _geometry_member_specs(prefix: str, count: int) -> dict[str, MemberSpec]:
    return {
        f"{prefix}_gt_center_errors": _member_spec(_F64_DTYPE, count),
        f"{prefix}_gt_covariance_errors": _member_spec(_F64_DTYPE, count),
    }


def _checkpoint_member_specs(prefix: str, updates: int) -> dict[str, MemberSpec]:
    checkpoint_count = updates // CHECKPOINT_INTERVAL + 1
    return {
        f"{prefix}_checkpoint_steps": _member_spec(_I64_DTYPE, checkpoint_count),
        f"{prefix}_checkpoint_values": _member_spec(
            _F64_DTYPE,
            checkpoint_count,
            len(_CHECKPOINT_VALUE_FIELDS),
        ),
    }


def _decision_fixed_member_specs(prefix: str) -> dict[str, MemberSpec]:
    return {
        f"{prefix}_scores": _member_spec(_F64_DTYPE, N_HYPOTHESES),
        f"{prefix}_retained": _member_spec(_BOOL_DTYPE, N_HYPOTHESES),
        f"{prefix}_component_ids": _member_spec(_I64_DTYPE, N_HYPOTHESES),
        f"{prefix}_representative_for_hypothesis": _member_spec(_I64_DTYPE, N_HYPOTHESES),
        f"{prefix}_decision_codes": _member_spec(_I8_DTYPE, N_HYPOTHESES),
    }


def _heldout_input_member_specs() -> dict[str, MemberSpec]:
    return {
        "heldout_camera_R": _member_spec(_F64_DTYPE, N_HELDOUT_VIEWS, 3, 3),
        "heldout_camera_t": _member_spec(_F64_DTYPE, N_HELDOUT_VIEWS, 3),
        "heldout_camera_intrinsics": _member_spec(_F64_DTYPE, N_HELDOUT_VIEWS, 4),
        "heldout_camera_image_sizes": _member_spec(_I64_DTYPE, N_HELDOUT_VIEWS, 2),
        "heldout_target_means2d": _member_spec(_F64_DTYPE, N_HELDOUT_VIEWS, N_GAUSSIANS, 2),
        "heldout_target_covariances2d": _member_spec(
            _F64_DTYPE, N_HELDOUT_VIEWS, N_GAUSSIANS, 2, 2
        ),
        "heldout_labels": _member_spec(_I64_DTYPE, N_HELDOUT_VIEWS, N_GAUSSIANS),
    }


def _heldout_evaluation_member_specs(prefix: str, count: int) -> dict[str, MemberSpec]:
    cost_shape = (N_HELDOUT_VIEWS, count, N_GAUSSIANS)
    return {
        f"{prefix}_heldout_center_costs": MemberSpec(_F64_DTYPE, cost_shape),
        f"{prefix}_heldout_conic_costs": MemberSpec(_F64_DTYPE, cost_shape),
        f"{prefix}_heldout_combined_costs": MemberSpec(_F64_DTYPE, cost_shape),
        f"{prefix}_heldout_projected_depths": _member_spec(_F64_DTYPE, N_HELDOUT_VIEWS, count),
        f"{prefix}_heldout_assignments": _member_spec(_I64_DTYPE, N_HELDOUT_VIEWS, count),
        f"{prefix}_heldout_correct": _member_spec(_BOOL_DTYPE, N_HELDOUT_VIEWS, count),
    }


def _decision_integrity(decision: TopologyDecision) -> dict[str, bool]:
    rejection_complete = (
        decision.rejected
        and decision.child is None
        and not decision.assignments
        and bool(decision.rejection_reason)
    )
    if decision.rejected:
        return {
            "complete_rejection_receipt": rejection_complete,
            "conditional_bijections_valid": True,
            "conditional_source_fixes_valid": True,
        }
    if decision.child is None or len(decision.assignments) != N_FIT_VIEWS:
        return {
            "complete_rejection_receipt": False,
            "conditional_bijections_valid": False,
            "conditional_source_fixes_valid": False,
        }
    bijections_valid = all(
        sorted(assignment.tolist()) == list(range(N_GAUSSIANS))
        for assignment in decision.assignments
    )
    source_fixes_valid = all(
        int(decision.assignments[view][row]) == int(decision.child.source_component_indices[row])
        for view in range(N_FIT_VIEWS)
        for row in range(decision.child.n)
        if int(decision.child.source_view_indices[row]) == view
    )
    return {
        "complete_rejection_receipt": True,
        "conditional_bijections_valid": bijections_valid,
        "conditional_source_fixes_valid": source_fixes_valid,
    }


def _base_fit_member_specs() -> dict[str, MemberSpec]:
    specs = _fit_input_member_specs()
    for prefix, count in (
        ("common_initial", N_HYPOTHESES),
        ("oracle_initial", N_GAUSSIANS),
        ("hardmin_400", N_HYPOTHESES),
        ("hardmin_600", N_HYPOTHESES),
        ("oracle", N_GAUSSIANS),
    ):
        specs.update(_model_member_specs(prefix, count))
    for prefix, count in (
        ("hardmin_400", N_HYPOTHESES),
        ("hardmin_600", N_HYPOTHESES),
        ("oracle", N_GAUSSIANS),
    ):
        specs.update(_fit_member_specs(prefix, count))
        specs.update(_geometry_member_specs(prefix, count))
    specs.update(_checkpoint_member_specs("hardmin", TOTAL_UPDATES))
    specs.update(_checkpoint_member_specs("oracle", TOTAL_UPDATES))
    return specs


def _decision_bootstrap_specs(payload: bytes) -> tuple[dict[str, MemberSpec], list[str]]:
    specs: dict[str, MemberSpec] = {}
    errors: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                return {}, ["duplicate_zip_member"]
            info_by_name = {info.filename: info for info in infos}
            for prefix in ("proposed", "shuffled"):
                alternatives: dict[str, tuple[MemberSpec, ...]] = {
                    **{
                        name: (spec,) for name, spec in _decision_fixed_member_specs(prefix).items()
                    },
                    f"{prefix}_representatives": tuple(
                        _member_spec(_I64_DTYPE, count) for count in range(N_HYPOTHESES + 1)
                    ),
                    f"{prefix}_assignments": (
                        _member_spec(_I64_DTYPE, 0, 0),
                        _member_spec(_I64_DTYPE, N_FIT_VIEWS, N_GAUSSIANS),
                    ),
                }
                for name, allowed in alternatives.items():
                    info = info_by_name.get(f"{name}.npy")
                    if info is None:
                        errors.append(f"decision_member_missing:{name}")
                        continue
                    chosen, member_errors = _choose_npz_member_spec(
                        archive,
                        info,
                        name=name,
                        allowed=allowed,
                    )
                    errors.extend(member_errors)
                    if chosen is not None:
                        specs[name] = chosen
    except Exception as error:
        errors.append(f"decision_bootstrap:{type(error).__name__}:{error}")
    return specs, errors


def _decision_specs_from_reopened(
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, MemberSpec], bool, list[str]]:
    errors: list[str] = []
    component_ids = arrays[f"{prefix}_component_ids"]
    retained = arrays[f"{prefix}_retained"]
    representative_ids = arrays[f"{prefix}_representative_for_hypothesis"]
    decision_codes = arrays[f"{prefix}_decision_codes"]
    representatives = arrays[f"{prefix}_representatives"]
    assignments = arrays[f"{prefix}_assignments"]
    scores = arrays[f"{prefix}_scores"]

    if not np.isfinite(scores).all():
        errors.append(f"nonfinite_decision_scores:{prefix}")
    if np.any(component_ids < -1):
        errors.append(f"invalid_negative_component_id:{prefix}")
    observed_component_ids = sorted(
        {int(value) for value in component_ids[component_ids >= 0].tolist()}
    )
    if observed_component_ids != list(range(len(observed_component_ids))):
        errors.append(f"noncontiguous_component_ids:{prefix}")
    component_count = len(observed_component_ids)
    if not np.array_equal(retained, component_ids >= 0):
        errors.append(f"retained_component_mismatch:{prefix}")
    if representatives.shape != (component_count,):
        errors.append(f"representative_count_mismatch:{prefix}")

    representatives_valid = (
        representatives.shape == (component_count,)
        and np.all((representatives >= 0) & (representatives < N_HYPOTHESES))
        and len(set(int(value) for value in representatives.tolist())) == component_count
    )
    if not representatives_valid:
        errors.append(f"invalid_representatives:{prefix}")
    else:
        for component_id, representative in enumerate(representatives.tolist()):
            if int(component_ids[int(representative)]) != component_id:
                errors.append(f"representative_not_in_component:{prefix}:{component_id}")
        expected_representative_ids = np.full(N_HYPOTHESES, -1, dtype=np.int64)
        for row, component_id in enumerate(component_ids.tolist()):
            if component_id >= 0 and component_id < component_count:
                expected_representative_ids[row] = representatives[component_id]
        if not np.array_equal(representative_ids, expected_representative_ids):
            errors.append(f"representative_map_mismatch:{prefix}")
        expected_codes = np.zeros(N_HYPOTHESES, dtype=np.int8)
        expected_codes[retained] = 1
        expected_codes[representatives] = 2
        if not np.array_equal(decision_codes, expected_codes):
            errors.append(f"decision_code_mismatch:{prefix}")

    accepted = assignments.shape == (N_FIT_VIEWS, N_GAUSSIANS)
    if accepted:
        if component_count != N_GAUSSIANS:
            errors.append(f"accepted_component_count_mismatch:{prefix}")
        expected_columns = list(range(N_GAUSSIANS))
        if any(sorted(row.tolist()) != expected_columns for row in assignments):
            errors.append(f"accepted_assignment_not_bijective:{prefix}")
    elif assignments.shape != (0, 0):
        errors.append(f"invalid_assignment_shape:{prefix}")

    specs = _decision_fixed_member_specs(prefix)
    specs[f"{prefix}_representatives"] = _member_spec(_I64_DTYPE, component_count)
    specs[f"{prefix}_assignments"] = (
        _member_spec(_I64_DTYPE, N_FIT_VIEWS, N_GAUSSIANS)
        if accepted
        else _member_spec(_I64_DTYPE, 0, 0)
    )
    return specs, accepted, errors


def _load_fit_evidence_arrays(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, bool]]:
    """Load fit evidence with conditional members derived from reopened decisions."""
    acceptance: dict[str, bool] = {}
    bootstrap_errors: list[str] = []
    capture: dict[str, Any] | None = None
    try:
        payload, capture = _capture_regular_bytes(
            expected_path,
            max_bytes=_NPZ_CAPTURE_MAX_BYTES,
        )
        bootstrap_specs, bootstrap_errors = _decision_bootstrap_specs(payload)
        if not bootstrap_errors:
            _members, decision_arrays, load_errors = _load_npz_payload_arrays(
                payload,
                bootstrap_specs,
                exact_members=False,
            )
            bootstrap_errors.extend(load_errors)
    except Exception as error:
        bootstrap_errors.append(f"decision_capture:{type(error).__name__}:{error}")
        decision_arrays = {}

    if bootstrap_errors:
        report = {
            "capture": capture,
            "semantic_sha256": None,
            "members": {},
            "required_members": [],
            "conditional_acceptance": acceptance,
            "errors": bootstrap_errors,
            "pass": False,
        }
        return report, {}, acceptance

    expected_specs = _base_fit_member_specs()
    decision_errors: list[str] = []
    for prefix in ("proposed", "shuffled"):
        specs, accepted, prefix_errors = _decision_specs_from_reopened(
            prefix,
            decision_arrays,
        )
        expected_specs.update(specs)
        acceptance[prefix] = accepted
        decision_errors.extend(prefix_errors)
        if accepted:
            expected_specs.update(_model_member_specs(f"{prefix}_final", N_GAUSSIANS))
            expected_specs.update(_fit_member_specs(f"{prefix}_final", N_GAUSSIANS))
            expected_specs.update(_geometry_member_specs(f"{prefix}_final", N_GAUSSIANS))
            expected_specs.update(_checkpoint_member_specs(prefix, RECOVERY_UPDATES))

    report, arrays = _load_validated_npz_arrays(
        descriptor,
        expected_path=expected_path,
        expected_specs=expected_specs,
    )
    report["required_members"] = sorted(expected_specs)
    report["conditional_acceptance"] = acceptance
    report["errors"] = decision_errors + report["errors"]
    report["pass"] = not report["errors"]
    return report, arrays, acceptance


def _load_input_evidence_arrays(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reopen the independently committed, pre-fit input evidence."""
    expected_specs = _fit_input_member_specs()
    report, arrays = _load_validated_npz_arrays(
        descriptor,
        expected_path=expected_path,
        expected_specs=expected_specs,
    )
    report["required_members"] = sorted(expected_specs)
    return report, arrays


def _heldout_member_specs(acceptance: dict[str, bool]) -> dict[str, MemberSpec]:
    specs = _heldout_input_member_specs()
    specs.update(_heldout_evaluation_member_specs("hardmin", N_HYPOTHESES))
    specs.update(_heldout_evaluation_member_specs("oracle", N_GAUSSIANS))
    for prefix in ("proposed", "shuffled"):
        if acceptance.get(prefix, False):
            specs.update(_heldout_evaluation_member_specs(prefix, N_GAUSSIANS))
    return specs


def _load_heldout_evidence_arrays(
    descriptor: dict[str, Any],
    *,
    expected_path: Path,
    fit_acceptance: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_specs = _heldout_member_specs(fit_acceptance)
    report, arrays = _load_validated_npz_arrays(
        descriptor,
        expected_path=expected_path,
        expected_specs=expected_specs,
    )
    report["required_members"] = sorted(expected_specs)
    report["conditional_acceptance"] = dict(fit_acceptance)
    return report, arrays


def _checkpoint_steps(checkpoints: list[dict[str, Any]]) -> list[int]:
    return [int(item["step"]) for item in checkpoints]


def _checkpoint_evidence_valid(
    arrays: dict[str, np.ndarray],
    prefix: str,
    expected_steps: list[int],
) -> bool:
    steps = arrays.get(f"{prefix}_checkpoint_steps")
    values = arrays.get(f"{prefix}_checkpoint_values")
    if steps is None or values is None:
        return False
    if steps.tolist() != expected_steps or values.shape != (
        len(expected_steps),
        len(_CHECKPOINT_VALUE_FIELDS),
    ):
        return False
    columns = {name: index for index, name in enumerate(_CHECKPOINT_VALUE_FIELDS)}
    return bool(
        np.isfinite(values).all()
        and np.all(values[:, columns["gradient_l2"]] >= 0)
        and np.all(values[:, columns["minimum_covariance_eigenvalue"]] > 0)
        and np.all(values[:, columns["depth_min"]] >= DEPTH_LOWER)
        and np.all(values[:, columns["depth_max"]] <= DEPTH_UPPER)
        and np.all(values[:, columns["loss_view_depth_min"]] > 0)
        and np.all(values[:, columns["loss_view_depth_max"]] > 0)
        and np.all(values[:, columns["source_center_max_px"]] <= 1e-6)
        and np.all(values[:, columns["source_covariance_relative_max"]] <= 1e-5)
    )


def _fit_commit_integrity(state: RootState) -> dict[str, Any]:
    fit_descriptor = state.summaries["artifacts"]["fit_evidence"]
    fit_npz, fit_arrays, acceptance = _load_fit_evidence_arrays(
        fit_descriptor,
        expected_path=state.root_directory / "fit_evidence.npz",
    )
    fit_receipt = _validate_json_descriptor(
        state.fit_receipt,
        expected_path=state.root_directory / "FIT_RECEIPT.json",
        expected_payload=state.summaries,
    )
    expected_full = list(range(0, TOTAL_UPDATES + 1, CHECKPOINT_INTERVAL))
    expected_topology = list(range(0, TOPOLOGY_STEP + 1, CHECKPOINT_INTERVAL))
    expected_recovery = list(range(0, RECOVERY_UPDATES + 1, CHECKPOINT_INTERVAL))
    expected_proposed = expected_recovery if acceptance.get("proposed") else []
    expected_shuffled = expected_recovery if acceptance.get("shuffled") else []
    checks = {
        "fit_npz_recaptured_and_valid": fit_npz["pass"],
        "fit_receipt_recaptured_and_valid": fit_receipt["pass"],
        "hardmin_checkpoint_steps_exact": _checkpoint_steps(
            state.summaries["hardmin_600"]["checkpoints"]
        )
        == expected_full,
        "hardmin_topology_checkpoint_steps_exact": _checkpoint_steps(
            state.summaries["hardmin_400"]["checkpoints_through_400"]
        )
        == expected_topology,
        "oracle_checkpoint_steps_exact": _checkpoint_steps(state.summaries["oracle"]["checkpoints"])
        == expected_full,
        "proposed_checkpoint_steps_exact_conditional": _checkpoint_steps(
            state.summaries["proposed"]["checkpoints"]
        )
        == expected_proposed,
        "shuffled_checkpoint_steps_exact_conditional": _checkpoint_steps(
            state.summaries["shuffled"]["checkpoints"]
        )
        == expected_shuffled,
        "hardmin_raw_checkpoint_evidence_valid": _checkpoint_evidence_valid(
            fit_arrays,
            "hardmin",
            expected_full,
        ),
        "hardmin_raw_topology_steps_exact": (
            fit_arrays.get("hardmin_checkpoint_steps", np.array([], dtype=np.int64))[
                : len(expected_topology)
            ].tolist()
            == expected_topology
        ),
        "oracle_raw_checkpoint_evidence_valid": _checkpoint_evidence_valid(
            fit_arrays,
            "oracle",
            expected_full,
        ),
        "proposed_raw_checkpoint_evidence_valid_conditional": (
            _checkpoint_evidence_valid(fit_arrays, "proposed", expected_recovery)
            if acceptance.get("proposed")
            else "proposed_checkpoint_steps" not in fit_arrays
        ),
        "shuffled_raw_checkpoint_evidence_valid_conditional": (
            _checkpoint_evidence_valid(fit_arrays, "shuffled", expected_recovery)
            if acceptance.get("shuffled")
            else "shuffled_checkpoint_steps" not in fit_arrays
        ),
    }
    return {
        "required_members": fit_npz["required_members"],
        "conditional_acceptance": acceptance,
        "npz": fit_npz,
        "receipt": fit_receipt,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _heldout_commit_integrity(
    state: RootState,
    heldout_report: dict[str, Any],
) -> dict[str, Any]:
    fit_npz, _fit_arrays_reopened, acceptance = _load_fit_evidence_arrays(
        state.summaries["artifacts"]["fit_evidence"],
        expected_path=state.root_directory / "fit_evidence.npz",
    )
    npz, _heldout_arrays_reopened = _load_heldout_evidence_arrays(
        heldout_report["heldout_evidence"],
        expected_path=state.root_directory / "heldout_evidence.npz",
        fit_acceptance=acceptance,
    )
    expected_receipt = {key: value for key, value in heldout_report.items() if key != "receipt"}
    receipt = _validate_json_descriptor(
        heldout_report["receipt"],
        expected_path=state.root_directory / "HELDOUT_RECEIPT.json",
        expected_payload=expected_receipt,
    )
    checks = {
        "conditioning_fit_npz_recaptured_and_valid": fit_npz["pass"],
        "heldout_npz_recaptured_and_valid": npz["pass"],
        "heldout_receipt_recaptured_and_valid": receipt["pass"],
    }
    return {
        "required_members": npz["required_members"],
        "conditional_acceptance": acceptance,
        "npz": npz,
        "receipt": receipt,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _root_evidence_integrity(
    state: RootState,
    heldout_report: dict[str, Any],
) -> dict[str, Any]:
    fit_commit = _fit_commit_integrity(state)
    heldout_commit = _heldout_commit_integrity(state, heldout_report)

    proposed_integrity = _decision_integrity(state.proposed)
    shuffled_integrity = _decision_integrity(state.shuffled)
    checks = {
        "fit_commit_recaptured_and_valid": fit_commit["pass"],
        "heldout_commit_recaptured_and_valid": heldout_commit["pass"],
        "initialization_pairing": state.summaries["initialization_pairing"]["pass"],
        "proposed_decision_integrity": all(proposed_integrity.values()),
        "shuffled_decision_integrity": all(shuffled_integrity.values()),
    }
    return {
        "checks": checks,
        "fit_commit": fit_commit,
        "heldout_commit": heldout_commit,
        "proposed_decision": proposed_integrity,
        "shuffled_decision": shuffled_integrity,
        "pass": all(checks.values()),
    }


def _evidence_tensor(arrays: dict[str, np.ndarray], name: str) -> torch.Tensor:
    """Copy one already schema-validated evidence array into a CPU tensor."""
    return torch.from_numpy(np.array(arrays[name], copy=True, order="C"))


def _tensor_matches_evidence(
    expected: torch.Tensor,
    arrays: dict[str, np.ndarray],
    name: str,
) -> bool:
    if name not in arrays:
        return False
    observed = _evidence_tensor(arrays, name)
    expected_cpu = expected.detach().cpu()
    if observed.dtype != expected_cpu.dtype or observed.shape != expected_cpu.shape:
        return False
    if expected_cpu.is_floating_point() or expected_cpu.is_complex():
        return bool(
            torch.isfinite(observed).all()
            and torch.allclose(observed, expected_cpu, rtol=1e-12, atol=1e-12)
        )
    return torch.equal(observed, expected_cpu)


def _array_bundle_checks(
    expected: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    return {
        f"array_matches:{name}": _tensor_matches_evidence(value, arrays, name)
        for name, value in sorted(expected.items())
    }


def _metric_mismatches(
    expected: Any,
    observed: Any,
    *,
    path: str,
) -> list[str]:
    """Compare a recomputed metric subtree against its committed JSON counterpart."""
    if isinstance(observed, dict):
        if not isinstance(expected, dict):
            return [path]
        mismatches: list[str] = []
        for key, value in observed.items():
            if key not in expected:
                mismatches.append(f"{path}.{key}:missing")
            else:
                mismatches.extend(_metric_mismatches(expected[key], value, path=f"{path}.{key}"))
        return mismatches
    if isinstance(observed, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or len(expected) != len(observed):
            return [path]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            mismatches.extend(_metric_mismatches(left, right, path=f"{path}[{index}]"))
        return mismatches
    if isinstance(observed, bool) or isinstance(expected, bool):
        return [] if type(expected) is type(observed) and expected == observed else [path]
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(observed, float) or isinstance(expected, float):
            return (
                []
                if math.isfinite(float(observed))
                and math.isfinite(float(expected))
                and math.isclose(float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12)
                else [path]
            )
        return [] if expected == observed else [path]
    return [] if expected == observed else [path]


def _cameras_from_evidence(
    arrays: dict[str, np.ndarray],
    prefix: str,
    count: int,
) -> tuple[Camera, ...]:
    rotations = _evidence_tensor(arrays, f"{prefix}_camera_R")
    translations = _evidence_tensor(arrays, f"{prefix}_camera_t")
    intrinsics = _evidence_tensor(arrays, f"{prefix}_camera_intrinsics")
    image_sizes = _evidence_tensor(arrays, f"{prefix}_camera_image_sizes")
    cameras: list[Camera] = []
    for index in range(count):
        fx, fy, cx, cy = intrinsics[index].tolist()
        width, height = image_sizes[index].tolist()
        cameras.append(
            Camera(
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                width=int(width),
                height=int(height),
                R=rotations[index],
                t=translations[index],
            )
        )
    return tuple(cameras)


def _candidate_evaluator_from_evidence(
    arrays: dict[str, np.ndarray],
) -> tuple[CandidateInputs, EvaluatorData, dict[str, bool]]:
    cameras = _cameras_from_evidence(arrays, "input_fit", N_FIT_VIEWS)
    target_means = _evidence_tensor(arrays, "input_fit_target_means2d")
    target_covariances = _evidence_tensor(arrays, "input_fit_target_covariances2d")
    source_views = _evidence_tensor(arrays, "input_source_view_indices")
    source_rows = _evidence_tensor(arrays, "input_source_local_rows")
    source_means = _evidence_tensor(arrays, "input_source_means2d")
    source_covariances = _evidence_tensor(arrays, "input_source_covariances2d")
    fit_labels_tensor = _evidence_tensor(arrays, "evaluator_fit_labels")
    derived_source_labels = fit_labels_tensor[source_views, source_rows]
    candidate = CandidateInputs(
        cameras=cameras,
        target_means2d=tuple(target_means),
        target_covariances2d=tuple(target_covariances),
        source_view_indices=source_views,
        source_local_rows=source_rows,
        source_means2d=source_means,
        source_covariances2d=source_covariances,
        initial_depths=_evidence_tensor(arrays, "input_initial_depths"),
    )
    evaluator = EvaluatorData(
        gt_means=_evidence_tensor(arrays, "evaluator_gt_means"),
        gt_covariances=_evidence_tensor(arrays, "evaluator_gt_covariances"),
        fit_labels=tuple(fit_labels_tensor),
        source_labels=derived_source_labels,
    )
    checks = {
        "input_source_means_equal_indexed_targets": torch.equal(
            source_means, target_means[source_views, source_rows]
        ),
        "input_source_covariances_equal_indexed_targets": torch.equal(
            source_covariances, target_covariances[source_views, source_rows]
        ),
        "stored_source_labels_equal_derived_labels": _tensor_matches_evidence(
            derived_source_labels, arrays, "evaluator_source_labels"
        ),
        "all_fit_inputs_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                target_means,
                target_covariances,
                source_means,
                source_covariances,
                candidate.initial_depths,
                evaluator.gt_means,
                evaluator.gt_covariances,
            )
        ),
        "fit_target_covariances_spd": bool((torch.linalg.eigvalsh(target_covariances) > 0).all()),
        "source_covariances_spd": bool((torch.linalg.eigvalsh(source_covariances) > 0).all()),
        "gt_covariances_spd": bool((torch.linalg.eigvalsh(evaluator.gt_covariances) > 0).all()),
    }
    return candidate, evaluator, checks


def _committed_input_binding(
    committed_input: dict[str, Any],
    *,
    roots: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, bool]]:
    receipt = committed_input.get("receipt")
    direct_descriptor = committed_input.get("evidence")
    receipt_descriptor = receipt.get("input_evidence") if type(receipt) is dict else None
    input_payload = receipt.get("input") if type(receipt) is dict else None
    expected_relative = f"scene_{roots[0]}/input_evidence.npz"
    checks = {
        "input_commit_receipt_is_object": type(receipt) is dict,
        "input_commit_direct_evidence_is_object": type(direct_descriptor) is dict,
        "input_commit_receipt_evidence_is_object": type(receipt_descriptor) is dict,
        "input_commit_evidence_descriptor_exact": (
            type(direct_descriptor) is dict
            and type(receipt_descriptor) is dict
            and direct_descriptor == receipt_descriptor
        ),
        "input_commit_root_bundle_exact": (
            type(receipt) is dict and receipt.get("root_bundle") == list(roots)
        ),
        "input_commit_relative_path_exact": (
            type(receipt) is dict
            and receipt.get("input_evidence_relative_path") == expected_relative
        ),
        "input_commit_seed_roots_exact": (
            type(input_payload) is dict
            and input_payload.get("scene_root") == roots[0]
            and input_payload.get("depth_root") == roots[1]
            and input_payload.get("observation_order_root") == roots[2]
        ),
    }
    return receipt_descriptor if type(receipt_descriptor) is dict else {}, checks


def _fit_input_binding_checks(
    input_arrays: dict[str, np.ndarray],
    fit_arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name in sorted(_fit_input_member_specs()):
        committed = input_arrays.get(name)
        duplicated = fit_arrays.get(name)
        checks[f"fit_input_exactly_bound:{name}"] = bool(
            isinstance(committed, np.ndarray)
            and isinstance(duplicated, np.ndarray)
            and committed.dtype.str == duplicated.dtype.str
            and committed.shape == duplicated.shape
            and np.array_equal(committed, duplicated)
        )
    return checks


def _canonical_common_initial_checks(
    candidate: CandidateInputs,
    fit_arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    expected = _model_arrays("common_initial", _new_fiber(candidate))
    defining_members = (
        "common_initial_depth_logits",
        "common_initial_cross",
        "common_initial_log_ray_scale",
        "common_initial_source_view_indices",
        "common_initial_source_local_rows",
    )
    return {
        f"canonical_input_initialization:{name}": bool(
            name in fit_arrays
            and fit_arrays[name].dtype.str == _as_numpy(expected[name]).dtype.str
            and fit_arrays[name].shape == tuple(expected[name].shape)
            and np.array_equal(fit_arrays[name], _as_numpy(expected[name]))
        )
        for name in defining_members
    }


def _model_from_evidence(
    prefix: str,
    arrays: dict[str, np.ndarray],
    candidate: CandidateInputs,
) -> tuple[InverseProjectionFiber, dict[str, bool]]:
    source_views = _evidence_tensor(arrays, f"{prefix}_source_view_indices")
    source_rows = _evidence_tensor(arrays, f"{prefix}_source_local_rows")
    target_means = torch.stack(candidate.target_means2d)
    target_covariances = torch.stack(candidate.target_covariances2d)
    anchors = target_means[source_views, source_rows]
    anchor_covariances = target_covariances[source_views, source_rows]
    initial_depths = anchors.new_full((source_views.numel(),), (DEPTH_LOWER + DEPTH_UPPER) / 2)
    model = InverseProjectionFiber(
        cameras=candidate.cameras,
        source_view_indices=source_views,
        source_component_indices=source_rows,
        source_means2d=anchors,
        source_covariances2d=anchor_covariances,
        initial_depths=initial_depths,
        depth_lower=DEPTH_LOWER,
        depth_upper=DEPTH_UPPER,
        dilation=DILATION,
    )
    with torch.no_grad():
        model.depth_logits.copy_(_evidence_tensor(arrays, f"{prefix}_depth_logits"))
        model.cross.copy_(_evidence_tensor(arrays, f"{prefix}_cross"))
        model.log_ray_scale.copy_(_evidence_tensor(arrays, f"{prefix}_log_ray_scale"))
    expected = _model_arrays(prefix, model)
    checks = _array_bundle_checks(expected, arrays)
    means, covariances = model.means_covariances()
    source_means, source_covariances, source_depths = model.source_projection()
    checks.update(
        {
            f"{prefix}:all_model_values_finite": all(
                bool(torch.isfinite(value).all())
                for value in (means, covariances, source_means, source_covariances)
            ),
            f"{prefix}:covariances_spd": bool((torch.linalg.eigvalsh(covariances) > 0).all()),
            f"{prefix}:source_depths_in_bounds": bool(
                (source_depths >= DEPTH_LOWER).all() and (source_depths <= DEPTH_UPPER).all()
            ),
        }
    )
    return model, checks


def _checkpoint_json_matches(
    arrays: dict[str, np.ndarray],
    prefix: str,
    checkpoints: list[dict[str, Any]],
) -> bool:
    steps = arrays.get(f"{prefix}_checkpoint_steps")
    values = arrays.get(f"{prefix}_checkpoint_values")
    if steps is None or values is None or len(checkpoints) != len(steps):
        return False
    if steps.tolist() != [int(item["step"]) for item in checkpoints]:
        return False
    expected_values = np.asarray(
        [[float(item[field]) for field in _CHECKPOINT_VALUE_FIELDS] for item in checkpoints],
        dtype=np.float64,
    )
    return bool(np.array_equal(values, expected_values))


def _fit_and_geometry_from_evidence(
    prefix: str,
    model: InverseProjectionFiber,
    candidate: CandidateInputs,
    evaluator: EvaluatorData,
    arrays: dict[str, np.ndarray],
    *,
    assignments: tuple[torch.Tensor, ...] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], dict[str, bool]]:
    fit, fit_arrays = _fit_evaluation(model, candidate, evaluator, assignments=assignments)
    source_labels = fit_arrays["source_labels"]
    geometry = _geometry_metrics(model, source_labels, evaluator)
    expected_arrays = _fit_arrays(prefix, fit_arrays)
    expected_arrays.update(
        _fit_arrays(prefix, _geometry_error_arrays(model, source_labels, evaluator))
    )
    checks = _array_bundle_checks(expected_arrays, arrays)
    return fit, geometry, fit_arrays, checks


def _heldout_data_from_evidence(
    arrays: dict[str, np.ndarray],
) -> tuple[HeldoutData, dict[str, bool]]:
    cameras = _cameras_from_evidence(arrays, "heldout", N_HELDOUT_VIEWS)
    means = _evidence_tensor(arrays, "heldout_target_means2d")
    covariances = _evidence_tensor(arrays, "heldout_target_covariances2d")
    labels = _evidence_tensor(arrays, "heldout_labels")
    checks = {
        "heldout_inputs_finite": bool(
            torch.isfinite(means).all() and torch.isfinite(covariances).all()
        ),
        "heldout_target_covariances_spd": bool((torch.linalg.eigvalsh(covariances) > 0).all()),
    }
    return (
        HeldoutData(
            cameras=cameras,
            means2d=tuple(means),
            covariances2d=tuple(covariances),
            labels=tuple(labels),
            receipt={},
        ),
        checks,
    )


def _heldout_metrics_from_evidence(
    evidence_prefix: str,
    model: InverseProjectionFiber,
    source_labels: torch.Tensor,
    heldout: HeldoutData,
    arrays: dict[str, np.ndarray],
    *,
    overcomplete_groups: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    metrics, recomputed_arrays = _heldout_evaluation(
        model,
        source_labels,
        heldout,
        overcomplete_groups=overcomplete_groups,
    )
    checks = _array_bundle_checks(_fit_arrays(evidence_prefix, recomputed_arrays), arrays)
    return metrics, checks


def _checkpoint_prefix_matches(
    arrays: dict[str, np.ndarray],
    prefix: str,
    checkpoints: list[dict[str, Any]],
) -> bool:
    steps = arrays.get(f"{prefix}_checkpoint_steps")
    values = arrays.get(f"{prefix}_checkpoint_values")
    if steps is None or values is None or len(checkpoints) > len(steps):
        return False
    count = len(checkpoints)
    expected_values = np.asarray(
        [[float(item[field]) for field in _CHECKPOINT_VALUE_FIELDS] for item in checkpoints],
        dtype=np.float64,
    )
    return bool(
        steps[:count].tolist() == [int(item["step"]) for item in checkpoints]
        and np.array_equal(values[:count], expected_values)
    )


def _rederive_root_evidence(
    state: RootState,
    committed_heldout: dict[str, Any],
    committed_input: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct every gate input from independently reopened exact tensors."""
    input_descriptor, input_commit_checks = _committed_input_binding(
        committed_input,
        roots=state.roots,
    )
    input_report, input_arrays = _load_input_evidence_arrays(
        input_descriptor,
        expected_path=state.root_directory / "input_evidence.npz",
    )
    fit_report, fit_arrays, acceptance = _load_fit_evidence_arrays(
        state.summaries["artifacts"]["fit_evidence"],
        expected_path=state.root_directory / "fit_evidence.npz",
    )
    heldout_report, heldout_arrays = _load_heldout_evidence_arrays(
        committed_heldout["heldout_evidence"],
        expected_path=state.root_directory / "heldout_evidence.npz",
        fit_acceptance=acceptance,
    )
    base_checks = {
        **input_commit_checks,
        "input_typed_evidence_valid": bool(input_report["pass"]),
        "fit_typed_evidence_valid": bool(fit_report["pass"]),
        "heldout_typed_evidence_valid": bool(heldout_report["pass"]),
    }
    if not all(base_checks.values()):
        return {
            "roots": list(state.roots),
            "checks": base_checks,
            "errors": [
                *input_report["errors"],
                *fit_report["errors"],
                *heldout_report["errors"],
            ],
            "pass": False,
        }

    try:
        checks = dict(base_checks)
        checks["all_reopened_float_arrays_finite"] = all(
            not np.issubdtype(value.dtype, np.floating) or bool(np.isfinite(value).all())
            for value in (*input_arrays.values(), *fit_arrays.values(), *heldout_arrays.values())
        )
        checks.update(_fit_input_binding_checks(input_arrays, fit_arrays))
        candidate, evaluator, input_checks = _candidate_evaluator_from_evidence(input_arrays)
        checks.update(input_checks)
        checks.update(_canonical_common_initial_checks(candidate, fit_arrays))

        models: dict[str, InverseProjectionFiber] = {}
        for prefix in (
            "common_initial",
            "oracle_initial",
            "hardmin_400",
            "hardmin_600",
            "oracle",
        ):
            model, model_checks = _model_from_evidence(prefix, fit_arrays, candidate)
            models[prefix] = model
            checks.update(model_checks)

        common_first_eight = models["common_initial"].subset(
            torch.arange(N_GAUSSIANS, dtype=torch.long)
        )
        checks["initialization_pairing_rederived"] = all(
            torch.equal(left, right)
            for left, right in zip(
                common_first_eight.state_dict().values(),
                models["oracle_initial"].state_dict().values(),
                strict=True,
            )
        )

        fit_400, geometry_400, arrays_400, fit_400_checks = _fit_and_geometry_from_evidence(
            "hardmin_400",
            models["hardmin_400"],
            candidate,
            evaluator,
            fit_arrays,
            assignments=None,
        )
        checks.update(fit_400_checks)
        fit_600, geometry_600, arrays_600, fit_600_checks = _fit_and_geometry_from_evidence(
            "hardmin_600",
            models["hardmin_600"],
            candidate,
            evaluator,
            fit_arrays,
            assignments=None,
        )
        checks.update(fit_600_checks)

        oracle_assignments = _oracle_assignments(models["oracle"], evaluator)
        oracle_fit, oracle_geometry, oracle_fit_arrays, oracle_checks = (
            _fit_and_geometry_from_evidence(
                "oracle",
                models["oracle"],
                candidate,
                evaluator,
                fit_arrays,
                assignments=oracle_assignments,
            )
        )
        checks.update(oracle_checks)

        proposed = _topology_decision(
            "residual_topology_8",
            models["hardmin_400"],
            candidate,
            arrays_400["residuals"],
        )
        shift = 1 + state.roots[2] % 7
        shuffled_scores = cyclic_shift_grouped_scores(
            arrays_400["residuals"],
            models["hardmin_400"].source_view_indices,
            models["hardmin_400"].source_component_indices,
            shift=shift,
            group_size=N_GAUSSIANS,
        )
        shuffled = _topology_decision(
            "shuffled_residual_topology",
            models["hardmin_400"],
            candidate,
            shuffled_scores,
        )
        decisions = {"proposed": proposed, "shuffled": shuffled}
        for prefix, decision in decisions.items():
            checks.update(_array_bundle_checks(_decision_arrays(prefix, decision), fit_arrays))
            checks[f"{prefix}:acceptance_rederived"] = bool(
                acceptance[prefix] == (not decision.rejected)
            )

        means_400, _covariances_400 = models["hardmin_400"].means_covariances()
        selections = {
            prefix: _selection_metrics(
                decision,
                arrays_400["complete_correct"],
                arrays_400["source_labels"],
                models["hardmin_400"].source_view_indices,
                models["hardmin_400"].source_component_indices,
                means_400.detach(),
            )
            for prefix, decision in decisions.items()
        }

        final_fits: dict[str, dict[str, Any] | None] = {}
        final_geometries: dict[str, dict[str, Any]] = {}
        final_fit_arrays: dict[str, dict[str, torch.Tensor] | None] = {}
        for prefix, decision in decisions.items():
            if decision.rejected:
                final_fits[prefix] = None
                final_fit_arrays[prefix] = None
                final_geometries[prefix] = _rejected_geometry_metrics(
                    int(decision.representatives.numel())
                )
                continue
            if decision.child is None:
                raise RuntimeError(f"accepted {prefix} decision has no child")
            final_model, final_model_checks = _model_from_evidence(
                f"{prefix}_final", fit_arrays, candidate
            )
            models[f"{prefix}_final"] = final_model
            checks.update(final_model_checks)
            checks[f"{prefix}:final_sources_equal_representatives"] = bool(
                torch.equal(
                    final_model.source_view_indices,
                    decision.child.source_view_indices,
                )
                and torch.equal(
                    final_model.source_component_indices,
                    decision.child.source_component_indices,
                )
            )
            fit, geometry, recomputed_arrays, final_checks = _fit_and_geometry_from_evidence(
                f"{prefix}_final",
                final_model,
                candidate,
                evaluator,
                fit_arrays,
                assignments=decision.assignments,
            )
            checks.update(final_checks)
            final_fits[prefix] = fit
            final_geometries[prefix] = geometry
            final_fit_arrays[prefix] = recomputed_arrays

        released, heldout_input_checks = _heldout_data_from_evidence(heldout_arrays)
        checks.update(heldout_input_checks)
        hardmin_heldout, hardmin_heldout_checks = _heldout_metrics_from_evidence(
            "hardmin",
            models["hardmin_600"],
            arrays_600["source_labels"],
            released,
            heldout_arrays,
            overcomplete_groups=True,
        )
        checks.update(hardmin_heldout_checks)
        oracle_heldout, oracle_heldout_checks = _heldout_metrics_from_evidence(
            "oracle",
            models["oracle"],
            oracle_fit_arrays["source_labels"],
            released,
            heldout_arrays,
            overcomplete_groups=False,
        )
        checks.update(oracle_heldout_checks)

        final_heldout: dict[str, dict[str, Any]] = {}
        for prefix, decision in decisions.items():
            if decision.rejected:
                final_heldout[prefix] = {
                    "heldout_association_accuracy": 0.0,
                    "heldout_center_cost": 0.0,
                    "heldout_conic_cost": 0.0,
                    "heldout_geometry_cost": 0.0,
                    "heldout_assignment_denominator": 0,
                }
                continue
            local_fit_arrays = final_fit_arrays[prefix]
            if local_fit_arrays is None:
                raise RuntimeError(f"accepted {prefix} arm has no recomputed fit arrays")
            metrics, local_checks = _heldout_metrics_from_evidence(
                prefix,
                models[f"{prefix}_final"],
                local_fit_arrays["source_labels"],
                released,
                heldout_arrays,
                overcomplete_groups=False,
            )
            checks.update(local_checks)
            final_heldout[prefix] = metrics

        arm_reports: dict[str, dict[str, Any]] = {
            "hardmin_600": {**fit_600, **geometry_600, **hardmin_heldout},
            "oracle": {**oracle_fit, **oracle_geometry, **oracle_heldout},
        }
        for prefix, decision in decisions.items():
            if decision.rejected:
                arm_reports[prefix] = {
                    "topology_rejected": True,
                    "fitting_assignment_accuracy": 0.0,
                    "exact_track_fraction": 0.0,
                    "complete_correct_count": 0,
                    "track_denominator": 0,
                    "assignment_denominator": 0,
                    **final_heldout[prefix],
                    **final_geometries[prefix],
                }
            else:
                fit = final_fits[prefix]
                if fit is None:
                    raise RuntimeError(f"accepted {prefix} arm has no fit metrics")
                arm_reports[prefix] = {
                    "topology_rejected": False,
                    **fit,
                    **final_geometries[prefix],
                    **final_heldout[prefix],
                }

        checkpoint_checks = {
            "hardmin_raw_checkpoints_valid": _checkpoint_evidence_valid(
                fit_arrays,
                "hardmin",
                list(range(0, TOTAL_UPDATES + 1, CHECKPOINT_INTERVAL)),
            ),
            "oracle_raw_checkpoints_valid": _checkpoint_evidence_valid(
                fit_arrays,
                "oracle",
                list(range(0, TOTAL_UPDATES + 1, CHECKPOINT_INTERVAL)),
            ),
            "hardmin_checkpoint_json_matches_raw": _checkpoint_json_matches(
                fit_arrays,
                "hardmin",
                state.summaries["hardmin_600"]["checkpoints"],
            ),
            "hardmin_400_checkpoint_json_matches_raw_prefix": _checkpoint_prefix_matches(
                fit_arrays,
                "hardmin",
                state.summaries["hardmin_400"]["checkpoints_through_400"],
            ),
            "oracle_checkpoint_json_matches_raw": _checkpoint_json_matches(
                fit_arrays,
                "oracle",
                state.summaries["oracle"]["checkpoints"],
            ),
        }
        for prefix, decision in decisions.items():
            if decision.rejected:
                checkpoint_checks[f"{prefix}_rejected_has_no_checkpoint_arrays"] = (
                    f"{prefix}_checkpoint_steps" not in fit_arrays
                    and f"{prefix}_checkpoint_values" not in fit_arrays
                    and not state.summaries[prefix]["checkpoints"]
                )
            else:
                checkpoint_checks[f"{prefix}_raw_checkpoints_valid"] = _checkpoint_evidence_valid(
                    fit_arrays,
                    prefix,
                    list(range(0, RECOVERY_UPDATES + 1, CHECKPOINT_INTERVAL)),
                )
                checkpoint_checks[f"{prefix}_checkpoint_json_matches_raw"] = (
                    _checkpoint_json_matches(
                        fit_arrays,
                        prefix,
                        state.summaries[prefix]["checkpoints"],
                    )
                )
        checks.update(checkpoint_checks)

        summary_mismatches: list[str] = []
        summary_mismatches.extend(
            _metric_mismatches(
                state.summaries["hardmin_400"],
                {**fit_400, **geometry_400},
                path="fit.hardmin_400",
            )
        )
        summary_mismatches.extend(
            _metric_mismatches(
                state.summaries["hardmin_600"],
                {**fit_600, **geometry_600},
                path="fit.hardmin_600",
            )
        )
        summary_mismatches.extend(
            _metric_mismatches(
                state.summaries["oracle"]["fit"],
                oracle_fit,
                path="fit.oracle",
            )
        )
        summary_mismatches.extend(
            _metric_mismatches(
                state.summaries["oracle"]["geometry"],
                oracle_geometry,
                path="geometry.oracle",
            )
        )
        for prefix in ("proposed", "shuffled"):
            summary_mismatches.extend(
                _metric_mismatches(
                    state.summaries[prefix]["selection"],
                    selections[prefix],
                    path=f"selection.{prefix}",
                )
            )
            if final_fits[prefix] is None:
                if state.summaries[prefix]["fit"] is not None:
                    summary_mismatches.append(f"fit.{prefix}:expected_null")
            else:
                summary_mismatches.extend(
                    _metric_mismatches(
                        state.summaries[prefix]["fit"],
                        final_fits[prefix],
                        path=f"fit.{prefix}",
                    )
                )
            summary_mismatches.extend(
                _metric_mismatches(
                    state.summaries[prefix]["geometry"],
                    final_geometries[prefix],
                    path=f"geometry.{prefix}",
                )
            )
        for prefix, recomputed in arm_reports.items():
            summary_mismatches.extend(
                _metric_mismatches(
                    committed_heldout[prefix],
                    recomputed,
                    path=f"heldout.{prefix}",
                )
            )
        checks["all_committed_scalar_summaries_match_rederivation"] = not summary_mismatches
        return {
            "roots": list(state.roots),
            "checks": checks,
            "summary_mismatches": summary_mismatches,
            "acceptance": acceptance,
            "fit": {
                "hardmin_400": {**fit_400, **geometry_400},
                "hardmin_600": {**fit_600, **geometry_600},
                "oracle": {**oracle_fit, **oracle_geometry},
            },
            "selection": selections,
            "arms": arm_reports,
            "pass": all(checks.values()),
        }
    except Exception as error:
        return {
            "roots": list(state.roots),
            "checks": base_checks,
            "errors": [f"{type(error).__name__}: {error}"],
            "pass": False,
        }


def _scientific_gates_legacy(
    states: list[RootState],
    heldout: list[dict[str, Any]],
    sentinel: dict[str, Any],
    protocol_integrity: dict[str, Any],
) -> dict[str, Any]:
    validity_checks: list[dict[str, Any]] = []
    for state, report in zip(states, heldout, strict=True):
        evidence = _root_evidence_integrity(state, report)
        all_checkpoints = (
            state.summaries["hardmin_600"]["checkpoints"]
            + state.summaries["oracle"]["checkpoints"]
            + state.summaries["proposed"]["checkpoints"]
            + state.summaries["shuffled"]["checkpoints"]
        )
        hardmin = report["hardmin_600"]
        oracle = report["oracle"]
        proposed = report["proposed"]
        shuffled = report["shuffled"]
        checks = {
            "all_reached_checkpoints_pass": all(item["pass"] for item in all_checkpoints),
            "oracle_train_accuracy_1": oracle["fitting_assignment_accuracy"] == 1.0,
            "oracle_heldout_accuracy_1": oracle["heldout_association_accuracy"] == 1.0,
            "oracle_exact_track_1": oracle["exact_track_fraction"] == 1.0,
            "oracle_center_p90_le_0.01": oracle["gt_center_p90"] <= 0.01,
            "oracle_covariance_median_le_0.01": oracle["gt_covariance_median"] <= 0.01,
            "hardmin_400_source_invariant": state.summaries["hardmin_400"]["source_center_max_px"]
            <= 1e-6
            and state.summaries["hardmin_400"]["source_covariance_relative_max"] <= 1e-5,
            "hardmin_600_source_invariant": hardmin["source_center_max_px"] <= 1e-6
            and hardmin["source_covariance_relative_max"] <= 1e-5,
            "oracle_source_invariant": oracle["source_center_max_px"] <= 1e-6
            and oracle["source_covariance_relative_max"] <= 1e-5,
            "proposed_conditional_source_invariant": proposed["topology_rejected"]
            or (
                proposed["source_center_max_px"] <= 1e-6
                and proposed["source_covariance_relative_max"] <= 1e-5
            ),
            "shuffled_conditional_source_invariant": shuffled["topology_rejected"]
            or (
                shuffled["source_center_max_px"] <= 1e-6
                and shuffled["source_covariance_relative_max"] <= 1e-5
            ),
            "exact_evidence_complete": evidence["pass"],
        }
        validity_checks.append(
            {
                "roots": list(state.roots),
                "checks": checks,
                "evidence": evidence,
                "pass": all(checks.values()),
            }
        )
    gate1_pass = (
        sentinel["pass"]
        and protocol_integrity["pass"]
        and all(item["pass"] for item in validity_checks)
    )

    gate2_roots: list[dict[str, Any]] = []
    for state in states:
        metrics = state.summaries["proposed"]["selection"]
        checks = {
            "precision_ge_0.95": metrics["survivor_precision"] >= 0.95,
            "recall_ge_0.90": metrics["survivor_recall"] >= 0.90,
            "coverage_1": metrics["hidden_mode_coverage"] == 1.0,
            "component_count_8": metrics["component_count"] == N_GAUSSIANS,
            "representative_count_8": metrics["representative_count"] == N_GAUSSIANS,
            "purity_1": metrics["component_purity"] == 1.0,
            "diameter_p90_le_0.01": metrics["component_diameter_p90"] <= 0.01,
        }
        gate2_roots.append(
            {"roots": list(state.roots), "checks": checks, "pass": all(checks.values())}
        )
    gate2_pass = all(item["pass"] for item in gate2_roots)

    gate3_roots: list[dict[str, Any]] = []
    for state, report in zip(states, heldout, strict=True):
        proposed = report["proposed"]
        hardmin = report["hardmin_600"]
        oracle = report["oracle"]
        absolute_checks = {
            "not_rejected": not proposed["topology_rejected"],
            "primitive_count_8": proposed["primitive_count"] == N_GAUSSIANS,
            "fit_accuracy_1": proposed["fitting_assignment_accuracy"] == 1.0,
            "exact_track_1": proposed["exact_track_fraction"] == 1.0,
            "heldout_accuracy_ge_0.95": proposed["heldout_association_accuracy"] >= 0.95,
            "center_p90_le_0.05": proposed["gt_center_p90"] <= 0.05,
            "covariance_median_le_0.10": proposed["gt_covariance_median"] <= 0.10,
            "center_oracle_noninferior": proposed["gt_center_p90"]
            <= max(0.05, 1.25 * oracle["gt_center_p90"]),
            "heldout_cost_oracle_noninferior": proposed["heldout_geometry_cost"]
            <= 1.10 * oracle["heldout_geometry_cost"] + 0.01,
        }
        paired_relative_checks = {
            "center_beats_hardmin": proposed["gt_center_p90"] < hardmin["gt_center_p90"],
            "tracks_beat_hardmin": proposed["exact_track_fraction"]
            > hardmin["exact_track_fraction"],
        }
        gate3_roots.append(
            {
                "roots": list(state.roots),
                "absolute_checks": absolute_checks,
                "paired_relative_checks": paired_relative_checks,
                "absolute_pass": all(absolute_checks.values()),
                "relative_pass": all(paired_relative_checks.values()),
            }
        )
    mean_hardmin_track = sum(item["hardmin_600"]["exact_track_fraction"] for item in heldout) / 3
    mean_proposed_track = sum(item["proposed"]["exact_track_fraction"] for item in heldout) / 3
    mean_hardmin_center = sum(item["hardmin_600"]["gt_center_p90"] for item in heldout) / 3
    mean_proposed_center = sum(item["proposed"]["gt_center_p90"] for item in heldout) / 3
    relative_center_reduction = (mean_hardmin_center - mean_proposed_center) / max(
        mean_hardmin_center,
        1e-12,
    )
    hardmin_absolute_passes = sum(
        item["hardmin_600"]["heldout_association_accuracy"] >= 0.95
        and item["hardmin_600"]["exact_track_fraction"] == 1.0
        and item["hardmin_600"]["gt_center_p90"] <= 0.05
        and item["hardmin_600"]["gt_covariance_median"] <= 0.10
        for item in heldout
    )
    relative_checks = {
        "track_mean_improvement_ge_0.20": mean_proposed_track - mean_hardmin_track >= 0.20,
        "center_mean_reduction_ge_0.50": relative_center_reduction >= 0.50,
        "paired_root_wins": all(item["relative_pass"] for item in gate3_roots),
    }
    gate3_absolute_pass = all(item["absolute_pass"] for item in gate3_roots)
    relative_attribution_inconclusive = hardmin_absolute_passes >= 2
    if not gate3_absolute_pass:
        gate3_status = "FAIL"
    elif relative_attribution_inconclusive:
        gate3_status = "INCONCLUSIVE"
    elif all(relative_checks.values()):
        gate3_status = "PASS"
    else:
        gate3_status = "FAIL"

    mean_proposed_coverage = (
        sum(state.summaries["proposed"]["selection"]["hidden_mode_coverage"] for state in states)
        / 3
    )
    mean_shuffled_coverage = (
        sum(state.summaries["shuffled"]["selection"]["hidden_mode_coverage"] for state in states)
        / 3
    )
    mean_shuffled_tracks = sum(item["shuffled"]["exact_track_fraction"] for item in heldout) / 3
    shuffled_failures = 0
    for state, report in zip(states, heldout, strict=True):
        selection = state.summaries["shuffled"]["selection"]
        final = report["shuffled"]
        failed = (
            selection["hidden_mode_coverage"] < 1.0
            or selection["component_count"] != N_GAUSSIANS
            or selection["component_purity"] < 1.0
            or final["topology_rejected"]
            or final["heldout_association_accuracy"] < 0.95
            or final["exact_track_fraction"] < 1.0
            or final["gt_center_p90"] > 0.05
            or final["gt_covariance_median"] > 0.10
        )
        shuffled_failures += int(failed)
    gate4_checks = {
        "coverage_improvement_ge_0.25": mean_proposed_coverage - mean_shuffled_coverage >= 0.25,
        "track_improvement_ge_0.25": mean_proposed_track - mean_shuffled_tracks >= 0.25,
        "shuffled_fails_at_least_two_roots": shuffled_failures >= 2,
    }
    gate4_pass = all(gate4_checks.values())
    return {
        "gate_1_validity": {
            "status": "PASS" if gate1_pass else "INVALID",
            "sentinel": sentinel,
            "protocol_integrity": protocol_integrity,
            "roots": validity_checks,
        },
        "gate_2_selection_contraction": {
            "status": "PASS" if gate2_pass else "FAIL",
            "roots": gate2_roots,
        },
        "gate_3_correspondence_geometry": {
            "status": gate3_status,
            "roots": gate3_roots,
            "relative_checks": relative_checks,
            "relative_attribution_inconclusive": relative_attribution_inconclusive,
            "mean_hardmin_track": mean_hardmin_track,
            "mean_proposed_track": mean_proposed_track,
            "mean_hardmin_center_p90": mean_hardmin_center,
            "mean_proposed_center_p90": mean_proposed_center,
            "relative_center_reduction": relative_center_reduction,
            "hardmin_absolute_pass_roots": hardmin_absolute_passes,
        },
        "gate_4_negative_control": {
            "status": "PASS" if gate4_pass else "FAIL",
            "checks": gate4_checks,
            "mean_proposed_coverage": mean_proposed_coverage,
            "mean_shuffled_coverage": mean_shuffled_coverage,
            "mean_proposed_track": mean_proposed_track,
            "mean_shuffled_track": mean_shuffled_tracks,
            "shuffled_failure_roots": shuffled_failures,
        },
        "overall_status": "INVALID"
        if not gate1_pass
        else "PASS"
        if gate2_pass and gate3_status == "PASS" and gate4_pass
        else "INCONCLUSIVE"
        if gate2_pass and gate3_status == "INCONCLUSIVE" and gate4_pass
        else "FAIL",
    }


def _scientific_gates(
    states: list[RootState],
    heldout: list[dict[str, Any]],
    input_commits: list[dict[str, Any]],
    sentinel: dict[str, Any],
    protocol_integrity: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen gates only from independently reopened exact evidence."""
    rederived = [
        _rederive_root_evidence(state, report, committed_input)
        for state, report, committed_input in zip(
            states,
            heldout,
            input_commits,
            strict=True,
        )
    ]
    validity_checks: list[dict[str, Any]] = []
    for state, report, root in zip(states, heldout, rederived, strict=True):
        committed_integrity = _root_evidence_integrity(state, report)
        if not root["pass"]:
            checks = {
                "reopened_gate_evidence_rederived": False,
                "exact_evidence_complete": committed_integrity["pass"],
            }
        else:
            hardmin_400 = root["fit"]["hardmin_400"]
            hardmin = root["arms"]["hardmin_600"]
            oracle = root["arms"]["oracle"]
            proposed = root["arms"]["proposed"]
            shuffled = root["arms"]["shuffled"]
            checks = {
                "reopened_gate_evidence_rederived": True,
                "all_reached_checkpoints_rederived_valid": all(
                    value for name, value in root["checks"].items() if "checkpoint" in name
                ),
                "oracle_train_accuracy_1": oracle["fitting_assignment_accuracy"] == 1.0,
                "oracle_heldout_accuracy_1": oracle["heldout_association_accuracy"] == 1.0,
                "oracle_exact_track_1": oracle["exact_track_fraction"] == 1.0,
                "oracle_center_p90_le_0.01": oracle["gt_center_p90"] <= 0.01,
                "oracle_covariance_median_le_0.01": oracle["gt_covariance_median"] <= 0.01,
                "hardmin_fit_denominator_96": hardmin["assignment_denominator"] == 96,
                "hardmin_heldout_denominator_64": hardmin["heldout_assignment_denominator"] == 64,
                "oracle_fit_denominator_32": oracle["assignment_denominator"] == 32,
                "oracle_heldout_denominator_16": oracle["heldout_assignment_denominator"] == 16,
                "hardmin_400_source_invariant": hardmin_400["source_center_max_px"] <= 1e-6
                and hardmin_400["source_covariance_relative_max"] <= 1e-5,
                "hardmin_600_source_invariant": hardmin["source_center_max_px"] <= 1e-6
                and hardmin["source_covariance_relative_max"] <= 1e-5,
                "oracle_source_invariant": oracle["source_center_max_px"] <= 1e-6
                and oracle["source_covariance_relative_max"] <= 1e-5,
                "proposed_conditional_source_invariant": proposed["topology_rejected"]
                or (
                    proposed["source_center_max_px"] <= 1e-6
                    and proposed["source_covariance_relative_max"] <= 1e-5
                ),
                "shuffled_conditional_source_invariant": shuffled["topology_rejected"]
                or (
                    shuffled["source_center_max_px"] <= 1e-6
                    and shuffled["source_covariance_relative_max"] <= 1e-5
                ),
                "proposed_conditional_denominators": proposed["topology_rejected"]
                or (
                    proposed["assignment_denominator"] == 32
                    and proposed["heldout_assignment_denominator"] == 16
                ),
                "shuffled_conditional_denominators": shuffled["topology_rejected"]
                or (
                    shuffled["assignment_denominator"] == 32
                    and shuffled["heldout_assignment_denominator"] == 16
                ),
                "exact_evidence_complete": committed_integrity["pass"],
            }
        validity_checks.append(
            {
                "roots": list(state.roots),
                "checks": checks,
                "committed_integrity": committed_integrity,
                "rederived": root,
                "pass": all(checks.values()) and root["pass"],
            }
        )
    gate1_pass = bool(
        sentinel["pass"]
        and protocol_integrity["pass"]
        and all(item["pass"] for item in validity_checks)
    )

    if not all(root["pass"] for root in rederived):
        return {
            "gate_1_validity": {
                "status": "INVALID",
                "sentinel": sentinel,
                "protocol_integrity": protocol_integrity,
                "roots": validity_checks,
            },
            "gate_2_selection_contraction": {
                "status": "INVALID",
                "roots": [],
            },
            "gate_3_correspondence_geometry": {
                "status": "INVALID",
                "roots": [],
            },
            "gate_4_negative_control": {
                "status": "INVALID",
                "checks": {},
            },
            "overall_status": "INVALID",
        }

    gate2_roots: list[dict[str, Any]] = []
    for root in rederived:
        metrics = root["selection"]["proposed"]
        checks = {
            "precision_ge_0.95": metrics["survivor_precision"] >= 0.95,
            "recall_ge_0.90": metrics["survivor_recall"] >= 0.90,
            "coverage_1": metrics["hidden_mode_coverage"] == 1.0,
            "component_count_8": metrics["component_count"] == N_GAUSSIANS,
            "representative_count_8": metrics["representative_count"] == N_GAUSSIANS,
            "purity_1": metrics["component_purity"] == 1.0,
            "diameter_p90_le_0.01": metrics["component_diameter_p90"] <= 0.01,
        }
        gate2_roots.append({"roots": root["roots"], "checks": checks, "pass": all(checks.values())})
    gate2_pass = all(item["pass"] for item in gate2_roots)

    gate3_roots: list[dict[str, Any]] = []
    hardmin_comparable_roots: list[dict[str, Any]] = []
    for root in rederived:
        proposed = root["arms"]["proposed"]
        hardmin = root["arms"]["hardmin_600"]
        oracle = root["arms"]["oracle"]
        absolute_checks = {
            "not_rejected": not proposed["topology_rejected"],
            "primitive_count_8": proposed["primitive_count"] == N_GAUSSIANS,
            "fit_assignment_denominator_32": proposed["assignment_denominator"] == 32,
            "heldout_assignment_denominator_16": proposed["heldout_assignment_denominator"] == 16,
            "fit_accuracy_1": proposed["fitting_assignment_accuracy"] == 1.0,
            "exact_track_1": proposed["exact_track_fraction"] == 1.0,
            "heldout_accuracy_ge_0.95": proposed["heldout_association_accuracy"] >= 0.95,
            "center_p90_le_0.05": proposed["gt_center_p90"] <= 0.05,
            "covariance_median_le_0.10": proposed["gt_covariance_median"] <= 0.10,
            "center_oracle_noninferior": proposed["gt_center_p90"]
            <= max(0.05, 1.25 * oracle["gt_center_p90"]),
            "heldout_cost_oracle_noninferior": proposed["heldout_geometry_cost"]
            <= 1.10 * oracle["heldout_geometry_cost"] + 0.01,
        }
        paired_relative_checks = {
            "center_beats_hardmin": proposed["gt_center_p90"] < hardmin["gt_center_p90"],
            "tracks_beat_hardmin": proposed["exact_track_fraction"]
            > hardmin["exact_track_fraction"],
        }
        hardmin_comparable_checks = {
            "fit_accuracy_1": hardmin["fitting_assignment_accuracy"] == 1.0,
            "exact_track_1": hardmin["exact_track_fraction"] == 1.0,
            "heldout_accuracy_ge_0.95": hardmin["heldout_association_accuracy"] >= 0.95,
            "center_p90_le_0.05": hardmin["gt_center_p90"] <= 0.05,
            "covariance_median_le_0.10": hardmin["gt_covariance_median"] <= 0.10,
            "center_oracle_noninferior": hardmin["gt_center_p90"]
            <= max(0.05, 1.25 * oracle["gt_center_p90"]),
            "heldout_cost_oracle_noninferior": hardmin["heldout_geometry_cost"]
            <= 1.10 * oracle["heldout_geometry_cost"] + 0.01,
        }
        gate3_roots.append(
            {
                "roots": root["roots"],
                "absolute_checks": absolute_checks,
                "paired_relative_checks": paired_relative_checks,
                "absolute_pass": all(absolute_checks.values()),
                "relative_pass": all(paired_relative_checks.values()),
            }
        )
        hardmin_comparable_roots.append(
            {
                "roots": root["roots"],
                "checks": hardmin_comparable_checks,
                "pass": all(hardmin_comparable_checks.values()),
            }
        )

    root_count = len(rederived)
    mean_hardmin_track = (
        sum(root["arms"]["hardmin_600"]["exact_track_fraction"] for root in rederived) / root_count
    )
    mean_proposed_track = (
        sum(root["arms"]["proposed"]["exact_track_fraction"] for root in rederived) / root_count
    )
    mean_hardmin_center = (
        sum(root["arms"]["hardmin_600"]["gt_center_p90"] for root in rederived) / root_count
    )
    mean_proposed_center = (
        sum(root["arms"]["proposed"]["gt_center_p90"] for root in rederived) / root_count
    )
    relative_center_reduction = (mean_hardmin_center - mean_proposed_center) / max(
        mean_hardmin_center, 1e-12
    )
    hardmin_absolute_passes = sum(item["pass"] for item in hardmin_comparable_roots)
    relative_checks = {
        "track_mean_improvement_ge_0.20": mean_proposed_track - mean_hardmin_track >= 0.20,
        "center_mean_reduction_ge_0.50": relative_center_reduction >= 0.50,
        "paired_root_wins": all(item["relative_pass"] for item in gate3_roots),
    }
    gate3_absolute_pass = all(item["absolute_pass"] for item in gate3_roots)
    relative_attribution_inconclusive = hardmin_absolute_passes >= 2
    if not gate3_absolute_pass:
        gate3_status = "FAIL"
    elif relative_attribution_inconclusive:
        gate3_status = "INCONCLUSIVE"
    elif all(relative_checks.values()):
        gate3_status = "PASS"
    else:
        gate3_status = "FAIL"

    mean_proposed_coverage = (
        sum(root["selection"]["proposed"]["hidden_mode_coverage"] for root in rederived)
        / root_count
    )
    mean_shuffled_coverage = (
        sum(root["selection"]["shuffled"]["hidden_mode_coverage"] for root in rederived)
        / root_count
    )
    mean_shuffled_tracks = (
        sum(root["arms"]["shuffled"]["exact_track_fraction"] for root in rederived) / root_count
    )
    shuffled_root_checks: list[dict[str, Any]] = []
    for root in rederived:
        selection = root["selection"]["shuffled"]
        final = root["arms"]["shuffled"]
        oracle = root["arms"]["oracle"]
        success_checks = {
            "coverage_1": selection["hidden_mode_coverage"] == 1.0,
            "component_count_8": selection["component_count"] == N_GAUSSIANS,
            "component_purity_1": selection["component_purity"] == 1.0,
            "not_rejected": not final["topology_rejected"],
            "primitive_count_8": final["primitive_count"] == N_GAUSSIANS,
            "fit_assignment_denominator_32": final["assignment_denominator"] == 32,
            "fit_accuracy_1": final["fitting_assignment_accuracy"] == 1.0,
            "exact_track_1": final["exact_track_fraction"] == 1.0,
            "heldout_assignment_denominator_16": final["heldout_assignment_denominator"] == 16,
            "heldout_accuracy_ge_0.95": final["heldout_association_accuracy"] >= 0.95,
            "center_p90_le_0.05": final["gt_center_p90"] <= 0.05,
            "covariance_median_le_0.10": final["gt_covariance_median"] <= 0.10,
            "center_oracle_noninferior": final["gt_center_p90"]
            <= max(0.05, 1.25 * oracle["gt_center_p90"]),
            "heldout_cost_oracle_noninferior": final["heldout_geometry_cost"]
            <= 1.10 * oracle["heldout_geometry_cost"] + 0.01,
        }
        shuffled_root_checks.append(
            {
                "roots": root["roots"],
                "success_checks": success_checks,
                "failed": not all(success_checks.values()),
            }
        )
    shuffled_failures = sum(item["failed"] for item in shuffled_root_checks)
    gate4_checks = {
        "coverage_improvement_ge_0.25": mean_proposed_coverage - mean_shuffled_coverage >= 0.25,
        "track_improvement_ge_0.25": mean_proposed_track - mean_shuffled_tracks >= 0.25,
        "shuffled_fails_at_least_two_roots": shuffled_failures >= 2,
    }
    gate4_pass = all(gate4_checks.values())
    return {
        "gate_1_validity": {
            "status": "PASS" if gate1_pass else "INVALID",
            "sentinel": sentinel,
            "protocol_integrity": protocol_integrity,
            "roots": validity_checks,
        },
        "gate_2_selection_contraction": {
            "status": "PASS" if gate2_pass else "FAIL",
            "roots": gate2_roots,
        },
        "gate_3_correspondence_geometry": {
            "status": gate3_status,
            "roots": gate3_roots,
            "hardmin_comparable_absolute_roots": hardmin_comparable_roots,
            "hardmin_comparable_rule": (
                "exclude only topology/cardinality predicates unavailable to the designed "
                "32-hypothesis control; require all comparable absolute quality and oracle-"
                "noninferiority predicates"
            ),
            "relative_checks": relative_checks,
            "relative_attribution_inconclusive": relative_attribution_inconclusive,
            "mean_hardmin_track": mean_hardmin_track,
            "mean_proposed_track": mean_proposed_track,
            "mean_hardmin_center_p90": mean_hardmin_center,
            "mean_proposed_center_p90": mean_proposed_center,
            "relative_center_reduction": relative_center_reduction,
            "hardmin_absolute_pass_roots": hardmin_absolute_passes,
        },
        "gate_4_negative_control": {
            "status": "PASS" if gate4_pass else "FAIL",
            "checks": gate4_checks,
            "roots": shuffled_root_checks,
            "mean_proposed_coverage": mean_proposed_coverage,
            "mean_shuffled_coverage": mean_shuffled_coverage,
            "mean_proposed_track": mean_proposed_track,
            "mean_shuffled_track": mean_shuffled_tracks,
            "shuffled_failure_roots": shuffled_failures,
        },
        "overall_status": "INVALID"
        if not gate1_pass
        else "PASS"
        if gate2_pass and gate3_status == "PASS" and gate4_pass
        else "INCONCLUSIVE"
        if gate2_pass and gate3_status == "INCONCLUSIVE" and gate4_pass
        else "FAIL",
    }


def _runtime_receipt() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _official_freshness_receipt() -> dict[str, Any]:
    root_tokens = [str(value) for value in (*SCENE_ROOTS, *DEPTH_ROOTS, *ORDER_ROOTS)]
    run_path_matches: list[str] = []
    runs_root = ROOT / "runs"
    for path in runs_root.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if any(token in relative for token in root_tokens):
            run_path_matches.append(relative)

    allowed_results = {
        PREREG.resolve(),
        PREREG_REVIEW.resolve(),
        IMPLEMENTATION_REVIEW.resolve(),
        (
            ROOT / "benchmarks/results/"
            "20260717_inverse_projection_fiber_iter2_PREREG_REVIEW_INITIAL_FAIL.md"
        ).resolve(),
    }
    result_matches: list[str] = []
    for path in (ROOT / "benchmarks/results").glob("*"):
        if path.resolve() in allowed_results or not path.is_file():
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            continue
        try:
            text = path.read_text(errors="strict")
        except UnicodeDecodeError:
            continue
        if NAMESPACE in text or any(token in text for token in root_tokens):
            result_matches.append(path.relative_to(ROOT).as_posix())
    checks = {
        "official_result_absent": not OFFICIAL_OUT.exists(),
        "official_artifacts_absent": not OFFICIAL_ARTIFACTS.exists(),
        "no_official_root_named_run_paths": not run_path_matches,
        "no_unexpected_prior_result_mentions": not result_matches,
    }
    return {
        "root_tokens": root_tokens,
        "run_path_matches": sorted(run_path_matches),
        "unexpected_result_mentions": sorted(result_matches),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _capture_identity_equal(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> bool:
    if first is None or second is None:
        return False
    return all(first.get(key) == second.get(key) for key in ("bytes", "sha256", "device", "inode"))


def _commit_validation_stable(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    return bool(
        first.get("pass")
        and second.get("pass")
        and _capture_identity_equal(
            first.get("npz", {}).get("capture"),
            second.get("npz", {}).get("capture"),
        )
        and _capture_identity_equal(
            first.get("receipt", {}).get("capture"),
            second.get("receipt", {}).get("capture"),
        )
    )


def run(
    out: Path,
    artifacts_dir: Path,
    *,
    development: bool = False,
    official_attempt: dict[str, Any] | None = None,
    official_attempt_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    requested_artifacts_dir = artifacts_dir.expanduser().resolve()
    if development:
        artifacts_dir = requested_artifacts_dir
        artifact_fd: int | None = None
        transaction_id: str | None = None
        if not out.parent.is_dir() or not artifacts_dir.parent.is_dir():
            raise FileNotFoundError("result and artifact parents must already exist")
        if out.exists() or artifacts_dir.exists():
            raise FileExistsError("development result and artifact paths must both be absent")
        artifacts_dir.mkdir(exist_ok=False)
        _fsync_directory(artifacts_dir.parent)
        freshness = None
        review: dict[str, Any] | None = None
        review_sha256 = None
        snapshot_root = ROOT
        attempt_validation: dict[str, Any] = {"mode": "development", "pass": True}
    else:
        if out != OFFICIAL_OUT.resolve() or requested_artifacts_dir != OFFICIAL_ARTIFACTS.resolve():
            raise ValueError("official execution requires the frozen output and artifact paths")
        if official_attempt is None or official_attempt_capture is None:
            raise RuntimeError("official execution is forbidden without a launcher attempt")
        artifact_fd = _official_fd("RTGS_ITER2_ARTIFACTS_FD", require_directory=True)
        _official_fd("RTGS_ITER2_OUTPUT_PARENT_FD", require_directory=True)
        _official_fd("RTGS_ITER2_WORKSPACE_FD", require_directory=True)
        artifacts_dir = Path(f"/proc/self/fd/{artifact_fd}")
        if not artifacts_dir.is_dir():
            raise RuntimeError("held official artifact directory is unavailable")
        transaction_id = official_attempt.get("transaction_id")
        if (
            type(transaction_id) is not str
            or len(transaction_id) != 32
            or any(character not in "0123456789abcdef" for character in transaction_id)
        ):
            raise RuntimeError("official attempt transaction ID is malformed")
        attempt_validation = {
            "mode": "held_directory_domain_capture",
            "public": official_attempt_capture,
            "pass": True,
        }
        freshness = official_attempt["freshness"]
        if not freshness.get("pass"):
            raise RuntimeError("launcher did not establish pre-reservation freshness")
        review = official_attempt["implementation_review_payload"]
        review_sha256 = official_attempt["implementation_review_sha256"]
        archive_fd = _official_fd("RTGS_ITER2_ARCHIVE_FD", require_directory=False)
        snapshot_root = Path(f"/proc/self/fd/{archive_fd}")
    if development:
        if _sha256_file(PREREG) != PREREG_SHA256:
            raise RuntimeError("preregistration hash mismatch")
        review_text = PREREG_REVIEW.read_text()
        if "Verdict: **PASS**" not in review_text or PREREG_SHA256 not in review_text:
            raise RuntimeError("preregistration does not have a matching PASS review")
        sources = _declared_sources()
        source_start = _source_manifest(sources)
        if source_start["errors"]:
            raise RuntimeError(f"source closure errors: {source_start['errors']}")
        source_hashes_for_modules = source_start["hashes"]
    else:
        assert official_attempt is not None and review is not None
        if official_attempt.get("preregistration_sha256") != PREREG_SHA256:
            raise RuntimeError("launcher-bound preregistration hash differs")
        source_start = {
            "hashes": official_attempt["archive_member_hashes"],
            "errors": {},
            "authority": "fully_sealed_memfd",
        }
        source_hashes_for_modules = source_start["hashes"]
    loaded_modules_start = _loaded_project_modules_receipt(
        snapshot_root,
        source_hashes_for_modules,
    )
    if not loaded_modules_start["pass"]:
        raise RuntimeError(f"loaded project modules are not source-bound: {loaded_modules_start}")
    archive = (
        _archive_sources(
            artifacts_dir / "EXECUTED_SOURCES.tar",
            sources,
            source_root=snapshot_root,
        )
        if development
        else official_attempt["executed_sources"]
    )

    sentinel = _run_relabeling_sentinel()
    if not sentinel["pass"]:
        raise RuntimeError(f"candidate-boundary/relabeling sentinel failed: {sentinel}")
    roots = (
        list(zip(SCENE_ROOTS, DEPTH_ROOTS, ORDER_ROOTS, strict=True))
        if not development
        else list(DEVELOPMENT_ROOTS)
    )
    root_directories: list[Path] = []
    root_directory_fds: list[int] = []
    input_commits: list[dict[str, Any]] = []

    def commit_constructed(index: int, bundle: RootInputs) -> None:
        directory_name = f"scene_{bundle.roots[0]}"
        if development:
            root_directory = artifacts_dir / directory_name
            root_directory.mkdir(exist_ok=False)
            _fsync_directory(artifacts_dir)
        else:
            assert artifact_fd is not None
            os.mkdir(directory_name, 0o700, dir_fd=artifact_fd)
            os.fsync(artifact_fd)
            flags = (
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            root_fd = os.open(directory_name, flags, dir_fd=artifact_fd)
            os.fchmod(root_fd, 0o700)
            root_directory_fds.append(root_fd)
            root_directory = Path(f"/proc/self/fd/{root_fd}")
        root_directories.append(root_directory)
        input_commits.append(
            _commit_root_input(
                bundle,
                index=index,
                root_directory=root_directory,
                artifacts_directory=artifacts_dir,
                development=development,
                transaction_id=transaction_id,
                artifact_fd=artifact_fd,
            )
        )

    root_inputs = _construct_root_inputs_once(roots, on_constructed=commit_constructed)
    input_plan = [bundle.receipt for bundle in root_inputs]
    plan = {
        "schema": "inverse_projection_fiber_iter2_plan_v1",
        "namespace": NAMESPACE if not development else f"{NAMESPACE}.development",
        "development": development,
        "preregistration": str(PREREG),
        "preregistration_sha256": PREREG_SHA256,
        "preregistration_review": str(PREREG_REVIEW),
        "preregistration_review_sha256": (
            _sha256_file(PREREG_REVIEW)
            if development
            else official_attempt["preregistration_review_sha256"]
        ),
        "implementation_review": str(IMPLEMENTATION_REVIEW) if not development else None,
        "implementation_review_sha256": review_sha256,
        "official_attempt": official_attempt_capture if not development else None,
        "official_attempt_validation": attempt_validation,
        "config": {
            "n_gaussians": N_GAUSSIANS,
            "n_fit_views": N_FIT_VIEWS,
            "n_heldout_views": N_HELDOUT_VIEWS,
            "n_hypotheses": N_HYPOTHESES,
            "depth_bounds": [DEPTH_LOWER, DEPTH_UPPER],
            "dilation": DILATION,
            "conic_weight": CONIC_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "topology_step": TOPOLOGY_STEP,
            "total_updates": TOTAL_UPDATES,
            "recovery_updates": RECOVERY_UPDATES,
            "residual_threshold_strict": RESIDUAL_THRESHOLD,
            "cluster_radius_strict": CLUSTER_RADIUS,
        },
        "runtime": _runtime_receipt(),
        "official_freshness": freshness,
        "source_observation_start": source_start,
        "source_archive": archive,
        "loaded_project_modules_start": loaded_modules_start,
        "candidate_boundary_and_relabeling_sentinel": sentinel,
        "input_plan": input_plan,
        "input_commits": input_commits,
        "root_construction_counts": [1 for _bundle in root_inputs],
    }
    plan_descriptor = _write_json_exclusive(artifacts_dir / "PLAN.json", plan)
    plan_validation_initial = _validate_json_descriptor(
        plan_descriptor,
        expected_path=artifacts_dir / "PLAN.json",
        expected_payload=plan,
    )
    if not plan_validation_initial["pass"]:
        raise RuntimeError(f"plan commit failed immediate recapture: {plan_validation_initial}")

    states: list[RootState] = []
    fit_commits_initial: list[dict[str, Any]] = []
    for bundle, root_directory in zip(root_inputs, root_directories, strict=True):
        state = _run_root(
            bundle.roots,
            root_directory,
            candidate=bundle.candidate,
            evaluator=bundle.evaluator,
        )
        states.append(state)
        validation = _fit_commit_integrity(state)
        fit_commits_initial.append(validation)
        if not validation["pass"]:
            raise RuntimeError(f"FIT commit failed immediate recapture: {validation}")
    learned_state_before_heldout = [_learned_state_semantic_sha256(state) for state in states]
    preheldout = {
        "schema": "inverse_projection_fiber_iter2_preheldout_commit_v1",
        "namespace": plan["namespace"],
        "heldout_released": False,
        "roots": [list(state.roots) for state in states],
        "fit_receipts": [state.fit_receipt for state in states],
        "fit_receipt_count": len(states),
        "fit_commit_validations": fit_commits_initial,
        "learned_state_semantic_sha256": learned_state_before_heldout,
        "plan": plan_descriptor,
        "plan_validation": plan_validation_initial,
    }
    preheldout_descriptor = _write_json_exclusive(
        artifacts_dir / "PREHELDOUT.json",
        preheldout,
    )
    preheldout_validation_initial = _validate_json_descriptor(
        preheldout_descriptor,
        expected_path=artifacts_dir / "PREHELDOUT.json",
        expected_payload=preheldout,
    )
    fit_commits_pre_release = [_fit_commit_integrity(state) for state in states]
    plan_validation_pre_release = _validate_json_descriptor(
        plan_descriptor,
        expected_path=artifacts_dir / "PLAN.json",
        expected_payload=plan,
    )
    preheldout_validation_pre_release = _validate_json_descriptor(
        preheldout_descriptor,
        expected_path=artifacts_dir / "PREHELDOUT.json",
        expected_payload=preheldout,
    )
    pre_release_checks = {
        "preheldout_immediate_recapture": preheldout_validation_initial["pass"],
        "plan_stable_before_release": plan_validation_pre_release["pass"]
        and _capture_identity_equal(
            plan_validation_initial["capture"],
            plan_validation_pre_release["capture"],
        ),
        "preheldout_stable_before_release": preheldout_validation_pre_release["pass"]
        and _capture_identity_equal(
            preheldout_validation_initial["capture"],
            preheldout_validation_pre_release["capture"],
        ),
        "fit_commits_stable_before_release": all(
            _commit_validation_stable(first, second)
            for first, second in zip(
                fit_commits_initial,
                fit_commits_pre_release,
                strict=True,
            )
        ),
    }
    if not all(pre_release_checks.values()):
        raise RuntimeError(f"pre-held-out evidence barrier failed: {pre_release_checks}")
    release_payload = {
        "schema": "inverse_projection_fiber_iter2_heldout_release_v1",
        "namespace": plan["namespace"],
        "release_count": 1,
        "preheldout": preheldout_descriptor,
        "pre_release_checks": pre_release_checks,
        "fit_commit_validations": fit_commits_pre_release,
    }
    release_descriptor = _write_json_exclusive(
        artifacts_dir / "HELDOUT_RELEASE.json",
        release_payload,
    )
    release_validation_initial = _validate_json_descriptor(
        release_descriptor,
        expected_path=artifacts_dir / "HELDOUT_RELEASE.json",
        expected_payload=release_payload,
    )
    if not release_validation_initial["pass"]:
        raise RuntimeError("held-out release receipt failed immediate recapture")

    heldout_data = [
        _materialize_heldout(state.evaluator, bundle.heldout_recipe)
        for state, bundle in zip(states, root_inputs, strict=True)
    ]
    heldout_reports = [
        _evaluate_root_heldout(state, state.root_directory, released)
        for state, released in zip(states, heldout_data, strict=True)
    ]
    learned_state_after_heldout = [_learned_state_semantic_sha256(state) for state in states]
    heldout_commits_initial = [
        _heldout_commit_integrity(state, report)
        for state, report in zip(states, heldout_reports, strict=True)
    ]
    if development:
        source_end = _source_manifest(sources)
        source_stable = source_end == source_start and not source_end["errors"]
    else:
        source_end = {
            "hashes": dict(source_hashes_for_modules),
            "errors": {},
            "authority": "fully_sealed_memfd",
        }
        source_stable = source_end == source_start
    loaded_modules_end = _loaded_project_modules_receipt(
        snapshot_root,
        source_hashes_for_modules,
    )
    if development:
        review_end: dict[str, Any] | None = None
        review_stable = True
    else:
        review_end = official_attempt["implementation_review_capture"]
        review_stable = bool(
            official_attempt["implementation_review_sha256"] == review_sha256
            and official_attempt["implementation_review_payload"] == review
        )
    fit_commits_final = [_fit_commit_integrity(state) for state in states]
    heldout_commits_final = [
        _heldout_commit_integrity(state, report)
        for state, report in zip(states, heldout_reports, strict=True)
    ]
    plan_validation_final = _validate_json_descriptor(
        plan_descriptor,
        expected_path=artifacts_dir / "PLAN.json",
        expected_payload=plan,
    )
    preheldout_validation_final = _validate_json_descriptor(
        preheldout_descriptor,
        expected_path=artifacts_dir / "PREHELDOUT.json",
        expected_payload=preheldout,
    )
    release_validation_final = _validate_json_descriptor(
        release_descriptor,
        expected_path=artifacts_dir / "HELDOUT_RELEASE.json",
        expected_payload=release_payload,
    )
    protocol_checks = {
        "pre_release_barrier_passed": all(pre_release_checks.values()),
        "plan_stable_through_final_gate": plan_validation_final["pass"]
        and _capture_identity_equal(
            plan_validation_pre_release["capture"],
            plan_validation_final["capture"],
        ),
        "preheldout_stable_through_final_gate": preheldout_validation_final["pass"]
        and _capture_identity_equal(
            preheldout_validation_pre_release["capture"],
            preheldout_validation_final["capture"],
        ),
        "release_receipt_stable": release_validation_final["pass"]
        and _capture_identity_equal(
            release_validation_initial["capture"],
            release_validation_final["capture"],
        ),
        "fit_commits_stable_through_final_gate": all(
            _commit_validation_stable(first, second)
            for first, second in zip(
                fit_commits_pre_release,
                fit_commits_final,
                strict=True,
            )
        ),
        "heldout_commits_stable_through_final_gate": all(
            _commit_validation_stable(first, second)
            for first, second in zip(
                heldout_commits_initial,
                heldout_commits_final,
                strict=True,
            )
        ),
        "source_closure_stable": source_stable,
        "implementation_review_stable": review_stable,
        "loaded_project_modules_start_bound": loaded_modules_start["pass"],
        "loaded_project_modules_end_bound": loaded_modules_end["pass"],
        "heldout_release_count_exactly_one": release_payload["release_count"] == 1,
        "learned_state_unchanged_after_heldout": (
            learned_state_after_heldout == learned_state_before_heldout
        ),
    }
    protocol_integrity = {
        "checks": protocol_checks,
        "pre_release_checks": pre_release_checks,
        "fit_commits_initial": fit_commits_initial,
        "fit_commits_pre_release": fit_commits_pre_release,
        "fit_commits_final": fit_commits_final,
        "heldout_commits_initial": heldout_commits_initial,
        "heldout_commits_final": heldout_commits_final,
        "plan_validation_final": plan_validation_final,
        "preheldout_validation_final": preheldout_validation_final,
        "release_validation_final": release_validation_final,
        "loaded_modules_start": loaded_modules_start,
        "loaded_modules_end": loaded_modules_end,
        "implementation_review_end": review_end,
        "learned_state_before_heldout": learned_state_before_heldout,
        "learned_state_after_heldout": learned_state_after_heldout,
        "pass": all(protocol_checks.values()),
    }
    gates = _scientific_gates(
        states,
        heldout_reports,
        input_commits,
        sentinel,
        protocol_integrity,
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result = {
        "schema": "inverse_projection_fiber_iter2_result_v1",
        "namespace": plan["namespace"],
        "status": gates["overall_status"],
        "development": development,
        "plan": plan_descriptor,
        "preheldout_commit": preheldout_descriptor,
        "heldout_release": release_descriptor,
        "heldout_release_count": 1,
        "source_observation_start": source_start,
        "source_observation_end": source_end,
        "source_stable": source_stable,
        "roots": [list(state.roots) for state in states],
        "root_summaries": [state.summaries for state in states],
        "input_commits": input_commits,
        "heldout_reports": heldout_reports,
        "protocol_integrity": protocol_integrity,
        "scientific_gates": gates,
        "process_peak_rss_bytes": peak_rss,
        "claim_scope": (
            "noiseless balanced synthetic prune/contraction/source-fixed-rematch/refit bundle only"
        ),
        "transaction": {
            "transaction_id": official_attempt["transaction_id"]
            if official_attempt is not None
            else None,
            "result_was_reserved_before_roots": not development,
            "terminal_path": str(
                (artifacts_dir if development else OFFICIAL_ARTIFACTS) / "TERMINAL.json"
            ),
            "lifecycle_path": str(
                (artifacts_dir if development else OFFICIAL_ARTIFACTS) / "LIFECYCLE.json"
            ),
        },
    }
    if not development:
        assert official_attempt is not None and artifact_fd is not None
        transaction_id = official_attempt["transaction_id"]
        scientific_payload = {
            key: value
            for key, value in result.items()
            if key not in {"schema", "namespace", "roots"}
        }
        scientific_payload["root_bundles"] = result["roots"]
        scientific_payload["transaction_id"] = transaction_id
        result = i2tx.official_domain().make_receipt(
            "result",
            scientific_payload,
            root_consumption_status=i2tx.CONSUMED,
            roots=i2tx.OFFICIAL_ROOTS,
            official_phase="FINAL",
        )
        result_nonce = hashlib.sha256(f"{transaction_id}:prepared-result".encode()).hexdigest()[:32]
        prepared_result = i2tx.prepare_receipt(
            OFFICIAL_OUT.parent,
            OFFICIAL_OUT.name,
            "result",
            result,
            nonce=result_nonce,
            directory_fd=_official_fd(
                "RTGS_ITER2_OUTPUT_PARENT_FD",
                require_directory=True,
            ),
        )
        handoff = i2tx.official_domain().make_receipt(
            "worker_handoff",
            {
                "transaction_id": transaction_id,
                "scientific_status": gates["overall_status"],
                "prepared_result": prepared_result,
                "input_receipts": [commit["publication"]["public"] for commit in input_commits],
                "input_receipt_names": [commit["target_name"] for commit in input_commits],
                "plan": plan_descriptor,
                "preheldout": preheldout_descriptor,
                "heldout_release": release_descriptor,
                "protocol_integrity_pass": protocol_integrity["pass"],
                "source_observation_end": source_end,
                "learned_state_before_heldout": learned_state_before_heldout,
                "learned_state_after_heldout": learned_state_after_heldout,
            },
            root_consumption_status=i2tx.PARTIALLY_CONSUMED,
            roots=i2tx.OFFICIAL_ROOTS,
            official_phase="WORKER_HANDOFF",
        )
        handoff_nonce = hashlib.sha256(f"{transaction_id}:worker-handoff".encode()).hexdigest()[:32]
        i2tx.publish_receipt(
            OFFICIAL_ARTIFACTS,
            "WORKER_HANDOFF.json",
            "worker_handoff",
            handoff,
            nonce=handoff_nonce,
            directory_fd=artifact_fd,
        )
        return result

    result_descriptor = _write_json_exclusive(out, result)
    terminal = {
        "schema": "inverse_projection_fiber_iter2_terminal_v1",
        "namespace": plan["namespace"],
        "transaction_id": official_attempt["transaction_id"]
        if official_attempt is not None
        else None,
        "transaction_status": "READY_TO_COMMIT",
        "scientific_status": gates["overall_status"],
        "roots_consumed_once": [list(state.roots) for state in states],
        "result": result_descriptor,
        "result_exchange": None,
        "source_observation_end": source_end,
        "protocol_integrity_pass": protocol_integrity["pass"],
    }
    terminal_descriptor = _write_json_exclusive(artifacts_dir / "TERMINAL.json", terminal)
    lifecycle = {
        "schema": "inverse_projection_fiber_iter2_lifecycle_v1",
        "namespace": plan["namespace"],
        "transaction_id": official_attempt["transaction_id"]
        if official_attempt is not None
        else None,
        "commit_state": "DEVELOPMENT_COMMITTED",
        "scientific_status": gates["overall_status"],
        "attempt": None,
        "plan": plan_descriptor,
        "preheldout": preheldout_descriptor,
        "heldout_release": release_descriptor,
        "result": result_descriptor,
        "terminal": terminal_descriptor,
        "source_observation_end": source_end,
    }
    _write_json_exclusive(artifacts_dir / "LIFECYCLE.json", lifecycle)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OFFICIAL_OUT)
    parser.add_argument("--artifacts-dir", type=Path, default=OFFICIAL_ARTIFACTS)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.development:
        official_attempt = None
        official_attempt_capture = None
    else:
        official_attempt, official_attempt_capture = _load_official_attempt()
    result = run(
        args.out,
        args.artifacts_dir,
        development=args.development,
        official_attempt=official_attempt,
        official_attempt_capture=official_attempt_capture,
    )
    print(
        json.dumps(
            {
                "namespace": result["namespace"],
                "status": result["status"],
                "out": str(args.out),
                "artifacts_dir": str(args.artifacts_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
